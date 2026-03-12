"""
Fetches team-season efficiency data.

PRIMARY SOURCE: Sports-Reference CBB Advanced Stats
  URL: sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html
  Provides: Pace (ORtg/DRtg), eFG%, TOV%, ORB%, FTr, BLK%, STL%, SRS, SOS
  Note: ORtg/DRtg are NOT opponent-quality adjusted (unlike Torvik/KenPom).
        SRS IS opponent-quality adjusted (net point differential per game).

OPTIONAL: Bart Torvik (barttorvik.com)
  Torvik has bot protection and requires manual data export.
  If you have a Torvik CSV export, place it in:
    data/raw/torvik/trank_{year}.csv
  and it will be used preferentially (better opponent-adjusted metrics).

HOW TO GET TORVIK DATA MANUALLY:
  1. Open barttorvik.com in a browser
  2. Navigate to T-Rank for the desired year
  3. Use the CSV/export button (or copy the table)
  4. Save as data/raw/torvik/trank_{year}.csv
"""

import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    TORVIK_DIR, SPORTS_REF_DIR, TORVIK_BASE_URL,
    TORVIK_DELAY_SECS, SPORTS_REF_DELAY_SECS, ALL_SEASONS,
)

logger = logging.getLogger(__name__)

# Sports-Reference Advanced Stats URL template
SR_ADV_URL = "https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html"

# Column mapping: Sports-Reference advanced column → our canonical name
SR_ADV_COL_MAP = {
    "School": "torvik_name",   # team name (we call it torvik_name for compatibility)
    "Conf.": "conf",
    "G": "G",
    "W": "W",
    "L": "L",
    "W-L%": "W_pct",
    "SRS": "SRS",
    "SOS": "SOS",
    "Pace": "AdjT",            # tempo proxy (possessions per 40 min)
    "ORtg": "AdjO",            # offensive rating proxy (NOT opponent-adjusted)
    "FTr": "FTR",
    "3PAr": "three_par",
    "TS%": "TS_pct",
    "TRB%": "TRB_pct",
    "AST%": "Stl_pct",         # closest proxy available
    "STL%": "Stl_pct",
    "BLK%": "Blk_pct",
    "eFG%": "eFG_pct",
    "TOV%": "TO_pct",
    "ORB%": "OR_pct",
    "FT/FGA": "FTR",
}


def fetch_trank_season(year: int, delay: float = TORVIK_DELAY_SECS) -> list[dict]:
    """
    Tries to fetch from Torvik (with bot protection warning), then falls back
    to Sports-Reference advanced stats.

    Saves result to TORVIK_DIR/trank_{year}.json.
    Returns list of raw dicts (one per team).
    """
    # 1. Check for manually-placed Torvik CSV first
    torvik_csv = TORVIK_DIR / f"trank_{year}.csv"
    if torvik_csv.exists():
        logger.info(f"Using manually-placed Torvik CSV: {torvik_csv}")
        return _load_torvik_csv(torvik_csv, year)

    # 2. Try direct Torvik API (often blocked by bot protection)
    result = _try_torvik_api(year, delay)
    if result:
        out_path = TORVIK_DIR / f"trank_{year}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        return result

    # 3. Fall back to Sports-Reference advanced stats
    logger.info(f"Torvik blocked — using Sports-Reference advanced stats for {year}")
    return _fetch_from_sports_ref(year, delay)


def load_trank_season(year: int) -> pd.DataFrame:
    """
    Loads team-season stats for a given year.
    Priority: Torvik CSV > Torvik JSON > Sports-Reference parquet.
    """
    # Check Torvik CSV
    torvik_csv = TORVIK_DIR / f"trank_{year}.csv"
    if torvik_csv.exists():
        df = pd.read_csv(torvik_csv)
        df["year"] = year
        df = _normalize_column_names(df)
        return _cast_dtypes(df, year)

    # Check Torvik JSON
    torvik_json = TORVIK_DIR / f"trank_{year}.json"
    if torvik_json.exists():
        with open(torvik_json) as f:
            raw = json.load(f)
        records = [_parse_trank_record(r, year) for r in raw]
        df = pd.DataFrame(records)
        df["year"] = year
        return _cast_dtypes(df, year)

    # Check Sports-Reference parquet (fallback)
    sr_path = SPORTS_REF_DIR / f"adv_stats_{year}.parquet"
    if sr_path.exists():
        df = pd.read_parquet(sr_path)
        return _cast_dtypes(df, year)

    raise FileNotFoundError(
        f"No stats data for {year}. Run: python scripts/fetch_data.py --years {year}"
    )


def fetch_all_seasons(
    years: list[int] = None,
    force_refresh: bool = False,
) -> None:
    """Fetches all seasons. Skips cached years unless force_refresh=True."""
    if years is None:
        years = ALL_SEASONS

    for year in years:
        # Check if any cached version exists
        paths = [
            TORVIK_DIR / f"trank_{year}.csv",
            TORVIK_DIR / f"trank_{year}.json",
            SPORTS_REF_DIR / f"adv_stats_{year}.parquet",
        ]
        if any(p.exists() for p in paths) and not force_refresh:
            logger.info(f"  Skipping {year} (cached)")
            continue
        try:
            fetch_trank_season(year)
        except Exception as e:
            logger.warning(f"  Failed to fetch {year}: {e}")


def load_all_seasons(years: list[int] = None) -> pd.DataFrame:
    """Loads all cached seasons into a single DataFrame."""
    if years is None:
        years = ALL_SEASONS

    dfs = []
    for year in years:
        try:
            dfs.append(load_trank_season(year))
        except FileNotFoundError:
            logger.warning(f"  No data for {year} — run fetch_data.py first")
        except Exception as e:
            logger.warning(f"  Could not load {year}: {e}")

    if not dfs:
        raise RuntimeError(
            "No team data loaded. Run: python scripts/fetch_data.py"
        )

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"Loaded {len(combined)} team-season records "
        f"({combined['year'].min()}–{combined['year'].max()})"
    )
    return combined


# ── Private helpers ────────────────────────────────────────────────────────────

def _try_torvik_api(year: int, delay: float) -> list[dict] | None:
    """Attempts Torvik API. Returns data or None if blocked."""
    url = f"{TORVIK_BASE_URL}?year={year}&json=1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://barttorvik.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        # Check if it's the bot-verification page
        if "Verifying Browser" in resp.text or "<html>" in resp.text[:50]:
            logger.debug(f"Torvik bot protection triggered for {year}")
            return None
        data = resp.json()
        if isinstance(data, dict):
            data = data.get("data", list(data.values())[0] if data else [])
        time.sleep(delay)
        logger.info(f"  Torvik API success for {year}: {len(data)} teams")
        return data
    except Exception as e:
        logger.debug(f"Torvik API failed for {year}: {e}")
        return None


def _fetch_from_sports_ref(year: int, delay: float = SPORTS_REF_DELAY_SECS) -> list[dict]:
    """
    Fetches Sports-Reference CBB Advanced Stats as Torvik fallback.
    Saves to SPORTS_REF_DIR/adv_stats_{year}.parquet.
    """
    url = SR_ADV_URL.format(year=year)
    logger.info(f"  Fetching SR Advanced Stats: {url}")

    # Use header=[0,1] to get multi-row header; second level has actual stat names
    tables = pd.read_html(url, header=[0, 1])
    if not tables:
        raise ValueError(f"No tables found at {url}")

    df = tables[0]
    df = _clean_sr_advanced(df, year)

    # Save to disk
    SPORTS_REF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPORTS_REF_DIR / f"adv_stats_{year}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"  Saved SR advanced stats: {len(df)} teams → {out_path}")

    time.sleep(delay)
    return df.to_dict(orient="records")


def _clean_sr_advanced(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Cleans the Sports-Reference advanced stats table (multi-row header format)."""
    # The table uses a 2-level MultiIndex header from pd.read_html(..., header=[0,1])
    # Level 0: group names ("Overall", "School Advanced", etc.)
    # Level 1: stat names ("G", "W", "SRS", "Pace", "ORtg", "eFG%", etc.)
    if isinstance(df.columns, pd.MultiIndex):
        # Use the second level (actual stat names), deduplicate as needed
        new_cols = []
        seen = {}
        for tup in df.columns:
            name = tup[1]  # second level = actual stat name
            # Skip unnamed/empty
            if "Unnamed" in str(name) or str(name).strip() == "":
                name = f"_drop_{len(new_cols)}"
            # Handle duplicates (e.g., two "W" columns)
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            new_cols.append(name)
        df.columns = new_cols

    # Drop _drop_* columns
    df = df[[c for c in df.columns if not c.startswith("_drop_")]].copy()

    # Drop repeated header rows (rows where 'School' == 'School')
    if "School" in df.columns:
        df = df[df["School"] != "School"].copy()

    # Rename School → torvik_name
    if "School" in df.columns:
        df = df.rename(columns={"School": "torvik_name"})
    elif "Rk" in df.columns:
        # Second column is team name
        team_col = df.columns[1]
        df = df.rename(columns={team_col: "torvik_name"})

    # Strip footnote markers from team names
    if "torvik_name" in df.columns:
        df["torvik_name"] = df["torvik_name"].astype(str).str.replace(r"\*$", "", regex=True).str.strip()

    # Drop rows with null/nan team names
    df = df[df["torvik_name"].notna() & (df["torvik_name"] != "nan")].copy()

    # ── Rename second-level SR column names to canonical names ───────────────
    # At this point columns come from level-1 of MultiIndex (actual stat names):
    # Rk, School, G, W, L, W-L%, SRS, SOS, ..., Pace, ORtg, FTr, 3PAr, TS%,
    # TRB%, AST%, STL%, BLK%, eFG%, TOV%, ORB%, FT/FGA
    rename = {
        "Pace":   "AdjT",
        "ORtg":   "AdjO",
        "eFG%":   "eFG_pct",
        "TOV%":   "TO_pct",
        "ORB%":   "OR_pct",
        "FTr":    "FTR",       # primary FTR source; FT/FGA dropped below
        "BLK%":   "Blk_pct",
        "STL%":   "Stl_pct",
        "AST%":   "ast_pct",
        "SRS":    "SRS",
        "SOS":    "SOS",
        "W-L%":   "W_pct",
        "G":      "G",
        "W":      "W",
        "L":      "L",
        "3PAr":   "three_par",
        "TS%":    "TS_pct",
        "TRB%":   "TRB_pct",
    }
    # Drop FT/FGA before renaming to avoid duplicate FTR
    df = df.drop(columns=["FT/FGA"], errors="ignore")
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Drop duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Conference name
    conf_col = None
    for c in ["Conf.", "conf", "Conference"]:
        if c in df.columns:
            conf_col = c
            break
    if conf_col:
        df["conf"] = df[conf_col].astype(str)
        if conf_col != "conf":
            df = df.drop(columns=[conf_col], errors="ignore")
    else:
        df["conf"] = "Unknown"

    # ── Derive AdjD: Opp_pts / G / Pace * 100 ≈ Defensive Rating ─────────────
    # Best approximation: (total opp points / games / pace) * 100 = DRtg
    # Fallback if Tm./Opp. not available: use SRS-based formula clipped to 85.
    if "Opp." in df.columns and "G" in df.columns and "AdjT" in df.columns:
        opp_pts = pd.to_numeric(df["Opp."], errors="coerce")
        games   = pd.to_numeric(df["G"], errors="coerce")
        pace    = pd.to_numeric(df["AdjT"], errors="coerce")
        opp_per_game = opp_pts / games.replace(0, np.nan)
        adj_d_raw = opp_per_game * (100.0 / pace.replace(0, np.nan))
        df["AdjD"] = adj_d_raw.clip(lower=85.0, upper=130.0).astype("float32")
    elif "AdjO" in df.columns and "SRS" in df.columns:
        adj_o      = pd.to_numeric(df["AdjO"], errors="coerce")
        srs        = pd.to_numeric(df["SRS"], errors="coerce")
        srs_per100 = srs * (100.0 / 70.0)
        df["AdjD"] = (adj_o - srs_per100).clip(lower=85.0, upper=130.0).astype("float32")
    else:
        df["AdjD"] = np.nan

    if "AdjD" in df.columns and "SRS" in df.columns:
        srs = pd.to_numeric(df["SRS"], errors="coerce")
        df["barthag"] = (srs / 30.0 + 0.5).clip(0, 1).astype("float32")
    else:
        df["barthag"] = np.nan

    # Opponent eFG% proxy (higher SRS teams allow lower eFG%)
    if "eFG_pct" in df.columns:
        efg = pd.to_numeric(df["eFG_pct"], errors="coerce")
        srs_series = pd.to_numeric(df["SRS"], errors="coerce") if "SRS" in df.columns else pd.Series(0.0, index=df.index)
        df["eFG_pct_d"] = (efg - srs_series * 0.002).clip(0.3, 0.7).astype("float32")

    # Derive W_pct if not present
    if "W_pct" not in df.columns and "W" in df.columns and "G" in df.columns:
        W = pd.to_numeric(df["W"], errors="coerce")
        G = pd.to_numeric(df["G"], errors="coerce")
        df["W_pct"] = (W / G.replace(0, np.nan)).astype("float32")
    elif "W_pct" in df.columns:
        df["W_pct"] = pd.to_numeric(df["W_pct"], errors="coerce").astype("float32")

    # Derive DR%: 100 - OR%
    if "OR_pct" in df.columns:
        df["DR_pct"] = (100 - pd.to_numeric(df["OR_pct"], errors="coerce")).astype("float32")

    # Forced TO% proxy
    if "TO_pct" in df.columns:
        df["TO_forced_pct"] = pd.to_numeric(df["TO_pct"], errors="coerce").astype("float32")

    # Rank by net rating
    if "AdjO" in df.columns and "AdjD" in df.columns:
        net = pd.to_numeric(df["AdjO"], errors="coerce") - pd.to_numeric(df["AdjD"], errors="coerce")
        df["rank"] = net.rank(ascending=False, method="min").astype("Int32")

    df["year"] = year

    # Drop unnamed/garbage columns
    df = df[[c for c in df.columns if "Unnamed" not in str(c) and not c.startswith("_drop_")]].copy()

    # Cast all numeric columns
    for col in df.columns:
        if col not in ("torvik_name", "conf", "year"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def _load_torvik_csv(path: Path, year: int) -> list[dict]:
    """Loads manually placed Torvik CSV file."""
    df = pd.read_csv(path)
    df["year"] = year
    df = _normalize_column_names(df)
    df = _cast_dtypes(df, year)
    return df.to_dict(orient="records")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names to canonical schema."""
    rename = {}
    for k, v in SR_ADV_COL_MAP.items():
        if k in df.columns:
            rename[k] = v
    # Also try common Torvik CSV headers
    torvik_map = {
        "team": "torvik_name", "adjoe": "AdjO", "adjde": "AdjD",
        "efg_o": "eFG_pct", "efg_d": "eFG_pct_d",
        "to_o": "TO_pct", "to_d": "TO_forced_pct",
        "or_o": "OR_pct", "or_d": "DR_pct",
        "ftr_o": "FTR", "blk": "Blk_pct", "stl": "Stl_pct",
        "adj_t": "AdjT", "wins": "W", "games": "G",
        "barthag": "barthag", "sos": "SOS", "conf": "conf", "seed": "seed",
    }
    for k, v in torvik_map.items():
        if k in df.columns:
            rename[k] = v
    return df.rename(columns=rename)


def _cast_dtypes(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Casts to correct dtypes and ensures 'year' column is set."""
    str_cols = {"torvik_name", "conf"}
    for col in df.columns:
        if col in str_cols:
            df[col] = df[col].astype(str)
        elif col != "year":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["year"] = int(year)
    return df


def _parse_trank_record(raw, year: int) -> dict:
    """Parses a single raw Torvik JSON record."""
    if isinstance(raw, list):
        cols = [
            "rank", "torvik_name", "conf", "G", "W",
            "AdjO", "AdjD", "barthag",
            "eFG_pct", "eFG_pct_d",
            "TO_pct", "TO_forced_pct",
            "OR_pct", "DR_pct",
            "FTR", "FTR_d",
            "twop_pct", "twop_pct_d",
            "threep_pct", "threep_pct_d",
            "AdjT", "Blk_pct", "Stl_pct",
            "SOS", "_year", "seed",
        ]
        rec = {cols[i]: raw[i] for i in range(min(len(cols), len(raw)))}
        rec.pop("_year", None)
    else:
        rec = {}
        torvik_map = {
            "team": "torvik_name", "conf": "conf", "adjoe": "AdjO",
            "adjde": "AdjD", "barthag": "barthag", "efg_o": "eFG_pct",
            "efg_d": "eFG_pct_d", "to_o": "TO_pct", "to_d": "TO_forced_pct",
            "or_o": "OR_pct", "or_d": "DR_pct", "ftr_o": "FTR",
            "blk": "Blk_pct", "stl": "Stl_pct", "adj_t": "AdjT",
            "wins": "W", "games": "G", "year": "_year", "seed": "seed",
            "sos": "SOS", "rank": "rank",
        }
        for src, dst in torvik_map.items():
            if src in raw:
                rec[dst] = raw[src]
        for k, v in raw.items():
            if k not in torvik_map and k not in rec:
                rec[k] = v

    rec["year"] = year
    # Derive W_pct
    if "W" in rec and "G" in rec:
        try:
            rec["W_pct"] = float(rec["W"]) / max(float(rec["G"]), 1)
        except (ValueError, TypeError):
            pass
    return rec
