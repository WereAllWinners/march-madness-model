"""
Team name crosswalk — maps bracket page names to canonical SR stats names.
No Kaggle required. Both data sources are Sports-Reference.

SR stats files use the full school name + "\xa0NCAA" suffix for tournament teams.
Bracket pages use short/nickname versions (UConn, UNC, etc.).

Crosswalk output columns:
    bracket_name   str   (as it appears on SR bracket pages)
    canonical_name str   (clean SR stats name, \xa0NCAA stripped)
    team_id        int32 (sequential integer, consistent across seasons)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# Manual overrides: bracket page name → stats page canonical name
BRACKET_TO_STATS_OVERRIDES: dict[str, str] = {
    "UConn":            "Connecticut",
    "UNC":              "North Carolina",
    "Saint Mary's":     "Saint Mary's (CA)",
    "St. Peter's":      "Saint Peter's",
    "BYU":              "Brigham Young",
    "LSU":              "LSU",
    "VCU":              "Virginia Commonwealth",
    "USC":              "Southern California",
    "UAB":              "Alabama Birmingham",
    "UTEP":             "Texas El Paso",
    "UTSA":             "UT San Antonio",
    "UTRGV":            "UT Rio Grande Valley",
    "UIC":              "Illinois Chicago",
    "UMKC":             "Missouri Kansas City",
    "UMBC":             "Maryland Baltimore County",
    "FIU":              "Florida International",
    "FAU":              "Florida Atlantic",
    "SFA":              "Stephen F. Austin",
    "NJIT":             "New Jersey Tech",
    "App State":        "Appalachian State",
    "Loyola-Chicago":   "Loyola Chicago",
    "LIU":              "Long Island University",
    "Little Rock":      "Arkansas Little Rock",
    "North Carolina A&T": "NC A&T",
    "McNeese State":    "McNeese State",
    "Texas A&M":        "Texas A&M",
    # Abbreviation → full name (fuzzy match fails due to short acronyms)
    "ETSU":             "East Tennessee State",
    "LSU":              "Louisiana State",
    "Ole Miss":         "Mississippi",
    "SMU":              "Southern Methodist",
    "St. Joseph's":     "Saint Joseph's",
    "UCSB":             "UC Santa Barbara",
    "UMass":            "Massachusetts",
    "UNLV":             "Nevada-Las Vegas",
}


def clean_stats_name(raw_name: str) -> str:
    """Strip \xa0NCAA and other suffixes from SR stats page team names."""
    if not isinstance(raw_name, str):
        return raw_name
    # Remove non-breaking space and everything after (e.g., "\xa0NCAA", "\xa0NAIA")
    name = raw_name.split("\xa0")[0].strip()
    return name


def build_crosswalk(
    stats_dfs: list[pd.DataFrame] | None = None,
    bracket_dfs: list[pd.DataFrame] | None = None,
    similarity_threshold: float = 85.0,
) -> pd.DataFrame:
    """
    Build crosswalk mapping bracket team names → canonical stats names → team_id.

    Args:
        stats_dfs:    list of adv_stats DataFrames (one per year), must have 'torvik_name'
        bracket_dfs:  list of tournament result DataFrames, must have 'W_team'/'L_team'
        similarity_threshold: rapidfuzz WRatio cutoff for fuzzy matching fallback

    Returns crosswalk DataFrame saved to PROCESSED_DIR/team_id_crosswalk.parquet
    """
    from rapidfuzz import fuzz, process as rfprocess

    # Collect all unique canonical stats names across all seasons
    if stats_dfs:
        all_stats_names = set()
        for df in stats_dfs:
            if "torvik_name" in df.columns:
                for n in df["torvik_name"].dropna().tolist():
                    all_stats_names.add(clean_stats_name(n))
    else:
        # Load from disk
        all_stats_names = _load_all_stats_names()

    canonical_list = sorted(all_stats_names)

    # Collect all unique bracket names
    if bracket_dfs:
        bracket_names = set()
        for df in bracket_dfs:
            if not df.empty and "W_team" in df.columns:
                bracket_names.update(df["W_team"].dropna().tolist())
                bracket_names.update(df["L_team"].dropna().tolist())
    else:
        bracket_names = set()

    # Build bracket_name → canonical_name mapping
    rows = []
    for b_name in sorted(bracket_names):
        canonical = _resolve_bracket_name(
            b_name, canonical_list, rfprocess, fuzz, similarity_threshold
        )
        rows.append({"bracket_name": b_name, "canonical_name": canonical})

    # Also add all stats names as self-mappings (for direct stats lookups)
    bracket_set = {r["bracket_name"] for r in rows}
    for c_name in canonical_list:
        if c_name not in bracket_set:
            rows.append({"bracket_name": c_name, "canonical_name": c_name})

    cw = pd.DataFrame(rows).drop_duplicates(subset=["bracket_name"])

    # Assign sequential team_id based on sorted canonical names
    unique_canonical = sorted(cw["canonical_name"].dropna().unique())
    id_map = {name: i + 1 for i, name in enumerate(unique_canonical)}
    cw["team_id"] = cw["canonical_name"].map(id_map).astype("Int32")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "team_id_crosswalk.parquet"
    cw.to_parquet(out_path, index=False)

    n_matched = cw["canonical_name"].notna().sum()
    n_unmatched = cw["canonical_name"].isna().sum()
    logger.info(
        f"Crosswalk: {n_matched} teams matched, {n_unmatched} unmatched → {out_path}"
    )
    if n_unmatched > 0:
        unmatched = cw[cw["canonical_name"].isna()]["bracket_name"].tolist()
        logger.warning(f"  Unmatched bracket names: {unmatched}")

    return cw


def load_crosswalk() -> pd.DataFrame:
    """Loads the team crosswalk from disk."""
    path = PROCESSED_DIR / "team_id_crosswalk.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Crosswalk not found. Run scripts/build_features.py --steps crosswalk"
        )
    return pd.read_parquet(path)


def bracket_name_to_canonical(name: str, crosswalk_df: pd.DataFrame) -> str | None:
    """Look up canonical stats name for a bracket team name."""
    rows = crosswalk_df[crosswalk_df["bracket_name"] == name]
    if rows.empty:
        return None
    return rows.iloc[0]["canonical_name"]


def canonical_to_team_id(canonical_name: str, crosswalk_df: pd.DataFrame) -> int | None:
    """Look up integer team_id for a canonical name."""
    rows = crosswalk_df[crosswalk_df["canonical_name"] == canonical_name]
    if rows.empty:
        return None
    tid = rows.iloc[0]["team_id"]
    return int(tid) if pd.notna(tid) else None


# ─── internal helpers ──────────────────────────────────────────────────────────

def _resolve_bracket_name(
    b_name: str,
    canonical_list: list[str],
    rfprocess,
    fuzz,
    threshold: float,
) -> str | None:
    # 1. Exact match
    if b_name in canonical_list:
        return b_name
    # 2. Manual override
    if b_name in BRACKET_TO_STATS_OVERRIDES:
        override = BRACKET_TO_STATS_OVERRIDES[b_name]
        if override in canonical_list:
            return override
        # Fuzzy search for the override target
        result = rfprocess.extractOne(override, canonical_list, scorer=fuzz.WRatio, score_cutoff=threshold)
        if result:
            return result[0]
    # 3. Fuzzy match
    result = rfprocess.extractOne(b_name, canonical_list, scorer=fuzz.WRatio, score_cutoff=threshold)
    if result:
        return result[0]
    logger.warning(f"  No canonical match for bracket name: '{b_name}'")
    return None


def _load_all_stats_names() -> set[str]:
    """Load canonical team names from all adv_stats parquet files on disk."""
    sr_dir = RAW_DIR / "sports_ref"
    names = set()
    for path in sorted(sr_dir.glob("adv_stats_*.parquet")):
        df = pd.read_parquet(path, columns=["torvik_name"])
        for n in df["torvik_name"].dropna():
            names.add(clean_stats_name(n))
    return names
