"""
Feature engineering: computes all differential and interaction features
from game_level.parquet + style_clusters.parquet.

Key design principles:
  - AdjO_diff = AdjO_A - AdjD_B  (cross-matchup: offense vs opponent's defense)
  - All diff features flip sign when team_A/B are swapped — symmetric by design
  - Seed features are 0 for regular season games (treated as neutral games)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    FEATURES_DIR, FEATURE_COLS, TARGET_COL, TEAM_STAT_COLS,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, LIVE_SEASON,
)

logger = logging.getLogger(__name__)


def build_feature_matrix(
    game_level_df: pd.DataFrame = None,
    clusters_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Main entry point. Builds and saves feature_matrix.parquet.

    Args:
        game_level_df: from game_builder.py (with rolling stats). Loaded if None.
        clusters_df:   from clustering.py. Loaded if None.

    Returns feature_matrix DataFrame saved to FEATURES_DIR/feature_matrix.parquet.
    """
    if game_level_df is None:
        from src.processing.game_builder import load_game_level
        game_level_df = load_game_level()

    if clusters_df is None:
        try:
            from src.features.clustering import load_cluster_assignments
            clusters_df = load_cluster_assignments()
        except FileNotFoundError:
            logger.warning("Style clusters not found — cluster features will be 0")
            clusters_df = pd.DataFrame()

    df = game_level_df.copy()

    # Merge style clusters
    if not clusters_df.empty:
        df = _merge_clusters(df, clusters_df)
    else:
        df["style_cluster_A"] = 0
        df["style_cluster_B"] = 0

    # Compute all differential features
    df = _compute_differentials(df)
    df = _compute_tempo_interaction(df)
    df = _compute_rolling_diffs(df)
    df = _compute_seed_features(df)
    df = _compute_cluster_interaction(df)

    # Add split tags
    df = assign_split_tags(df)

    # Keep metadata + features + target
    meta_cols = [
        "game_id", "season", "day_num", "is_tournament",
        "tournament_round", "team_A_id", "team_B_id", "split",
    ]

    # Only include FEATURE_COLS that were successfully computed
    avail_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Could not compute features (filling with 0): {missing}")
        for col in missing:
            df[col] = 0.0

    keep_cols = [c for c in meta_cols if c in df.columns] + FEATURE_COLS + [TARGET_COL]
    feature_df = df[keep_cols].copy()

    # Fill remaining NaNs in feature columns
    for col in FEATURE_COLS:
        feature_df[col] = feature_df[col].fillna(0.0).astype("float32")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "feature_matrix.parquet"
    feature_df.to_parquet(out_path, index=False)

    by_split = feature_df["split"].value_counts().to_dict()
    logger.info(f"Feature matrix saved ({len(feature_df)} games): {by_split} → {out_path}")
    return feature_df


def _compute_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized computation of cross-matchup differential features."""

    def diff(col_a: str, col_b: str, new_col: str) -> None:
        if col_a in df.columns and col_b in df.columns:
            df[new_col] = (df[col_a] - df[col_b]).astype("float32")
        else:
            df[new_col] = np.nan

    # --- Efficiency cross-matchup differentials ---
    # AdjO_A vs AdjD_B: how much better is A's offense than B's defense quality
    diff("AdjO_A", "AdjD_B", "AdjO_diff")
    # AdjD_A vs AdjO_B: how much better is A's defense than B's offense quality
    diff("AdjD_A", "AdjO_B", "AdjD_diff")

    # Net rating differential
    if "AdjO_A" in df.columns and "AdjD_A" in df.columns:
        net_A = df["AdjO_A"] - df["AdjD_A"]
        net_B = df.get("AdjO_B", 0) - df.get("AdjD_B", 0)
        df["AdjNetRtg_diff"] = (net_A - net_B).astype("float32")
    else:
        df["AdjNetRtg_diff"] = np.nan

    # Barthag (power rating)
    diff("barthag_A", "barthag_B", "barthag_diff")

    # Tempo differential
    diff("AdjT_A", "AdjT_B", "AdjT_diff")

    # --- Four Factors ---
    # eFG%: A offense vs B defense quality
    diff("eFG_pct_A", "eFG_pct_d_B", "eFG_diff")
    # eFG defense: A defense vs B offense
    diff("eFG_pct_d_A", "eFG_pct_B", "eFG_d_diff")
    # Turnover rate (offense): lower is better for offense
    diff("TO_pct_A", "TO_pct_B", "TO_off_diff")
    # Forced turnovers (defense): higher is better for defense
    diff("TO_forced_pct_A", "TO_forced_pct_B", "TO_forced_diff")
    # Offensive rebounding vs opponent's defensive rebounding
    diff("OR_pct_A", "DR_pct_B", "OR_diff")
    # Free throw rate
    diff("FTR_A", "FTR_B", "FTR_diff")

    # --- Supplemental ---
    diff("Blk_pct_A", "Blk_pct_B", "Blk_diff")
    diff("Stl_pct_A", "Stl_pct_B", "Stl_diff")
    diff("W_pct_A", "W_pct_B", "W_pct_diff")
    diff("SOS_A", "SOS_B", "SOS_diff")

    return df


def _compute_tempo_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """AdjT_A * AdjT_B — high = both teams run; asymmetry = style clash."""
    if "AdjT_A" in df.columns and "AdjT_B" in df.columns:
        df["AdjT_interaction"] = (df["AdjT_A"] * df["AdjT_B"]).astype("float32")
    else:
        df["AdjT_interaction"] = np.nan
    return df


def _compute_rolling_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Differential rolling form features (last 5 games)."""
    for prefix in ["roll5_AdjO", "roll5_AdjD"]:
        col_a = f"{prefix}_A"
        col_b = f"{prefix}_B"
        diff_col = f"{prefix}_diff"
        if col_a in df.columns and col_b in df.columns:
            df[diff_col] = (df[col_a] - df[col_b]).astype("float32")
        else:
            df[diff_col] = np.nan

    # Rolling net rating diff
    for sfx in ["A", "B"]:
        if f"roll5_net_{sfx}" not in df.columns:
            o = df.get(f"roll5_AdjO_{sfx}")
            d = df.get(f"roll5_AdjD_{sfx}")
            if o is not None and d is not None:
                df[f"roll5_net_{sfx}"] = (o - d).astype("float32")

    if "roll5_net_A" in df.columns and "roll5_net_B" in df.columns:
        df["roll5_net_diff"] = (df["roll5_net_A"] - df["roll5_net_B"]).astype("float32")
    else:
        df["roll5_net_diff"] = np.nan

    return df


def _compute_seed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    seed_diff = seed_A - seed_B (negative = A is better seed)
    seed_upset_flag = 1 if seed_A > seed_B (A is underdog)
    For non-tournament games: both are 0.
    """
    if "seed_A" in df.columns and "seed_B" in df.columns:
        df["seed_diff"] = (
            df["seed_A"].fillna(8.5) - df["seed_B"].fillna(8.5)
        ).astype("float32")
        df["seed_upset_flag"] = (df["seed_A"] > df["seed_B"]).fillna(False).astype("int8")
        # Zero out for non-tournament games
        df.loc[~df["is_tournament"], "seed_diff"] = 0.0
        df.loc[~df["is_tournament"], "seed_upset_flag"] = 0
    else:
        df["seed_diff"] = 0.0
        df["seed_upset_flag"] = 0

    return df


def _compute_cluster_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """style_cluster_interaction = cluster_A * 5 + cluster_B (25 combos)."""
    df["style_cluster_interaction"] = (
        (df["style_cluster_A"] * 5 + df["style_cluster_B"]).astype("int8")
    )
    return df


def _merge_clusters(
    game_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merges cluster assignments for team_A and team_B."""
    # clusters_df has: kaggle_id (or torvik_name), year, style_cluster

    join_col = "kaggle_id" if "kaggle_id" in clusters_df.columns else "torvik_name"
    rename_season = {"year": "season"} if "year" in clusters_df.columns else {}

    cdf = clusters_df[[join_col, "year", "style_cluster"]].copy() if "year" in clusters_df.columns else clusters_df

    if "year" in cdf.columns:
        cdf = cdf.rename(columns={"year": "season"})

    if join_col == "kaggle_id":
        merge_A = cdf.rename(columns={"kaggle_id": "team_A_id", "style_cluster": "style_cluster_A"})
        merge_B = cdf.rename(columns={"kaggle_id": "team_B_id", "style_cluster": "style_cluster_B"})

        if "season" in merge_A.columns:
            game_df = game_df.merge(merge_A, on=["team_A_id", "season"], how="left")
            game_df = game_df.merge(merge_B, on=["team_B_id", "season"], how="left")
        else:
            game_df = game_df.merge(merge_A, on="team_A_id", how="left")
            game_df = game_df.merge(merge_B, on="team_B_id", how="left")

    game_df["style_cluster_A"] = game_df.get("style_cluster_A", pd.Series(0, index=game_df.index)).fillna(0).astype("int8")
    game_df["style_cluster_B"] = game_df.get("style_cluster_B", pd.Series(0, index=game_df.index)).fillna(0).astype("int8")

    return game_df


def assign_split_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'split' column based on season year."""
    conditions = [
        df["season"].isin(TRAIN_SEASONS),
        df["season"].isin(VAL_SEASONS),
        df["season"].isin(TEST_SEASONS),
        df["season"] == LIVE_SEASON,
    ]
    choices = ["train", "val", "test", "live"]
    df["split"] = np.select(conditions, choices, default="other")
    return df


def load_feature_matrix(split: str = None) -> pd.DataFrame:
    """
    Loads feature_matrix.parquet.
    Optionally filters to a specific split: "train" | "val" | "test" | "live"
    """
    path = FEATURES_DIR / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Feature matrix not found. Run scripts/build_features.py --steps features"
        )
    df = pd.read_parquet(path)
    if split:
        df = df[df["split"] == split].copy()
    return df


def get_X_y(
    feature_df: pd.DataFrame,
    splits: list[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extracts feature matrix X and target y from feature_df.

    Args:
        feature_df: from load_feature_matrix()
        splits:     optional filter (e.g., ["train"])

    Returns (X, y, feature_names)
    """
    if splits:
        feature_df = feature_df[feature_df["split"].isin(splits)]

    avail_features = [c for c in FEATURE_COLS if c in feature_df.columns]
    X = feature_df[avail_features].values.astype("float32")
    y = feature_df[TARGET_COL].values.astype("int8")
    return X, y, avail_features
