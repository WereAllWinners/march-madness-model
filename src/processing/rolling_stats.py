"""
Computes rolling N-game efficiency metrics for each team.
Captures "form" — recent performance vs. season average.

For tournament games, rolling stats are derived from the most recent
regular-season games played (the last 5 before DayNum=134).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR

logger = logging.getLogger(__name__)

ROLLING_STAT_COLS = ["AdjO", "AdjD"]
DEFAULT_WINDOW = 5


def compute_rolling_stats(
    game_level_df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    stat_cols: list[str] = ROLLING_STAT_COLS,
) -> pd.DataFrame:
    """
    Computes rolling window mean for stat_cols for each team.
    Adds columns: roll5_AdjO_A, roll5_AdjD_A, roll5_net_A (and _B equivalents).

    Strategy:
    1. Melt game_level_df to long format (one row per team-game)
    2. Sort by (team_id, season, day_num)
    3. Apply rolling(window, min_periods=3).mean() per team-season group
    4. Pivot back to wide and join onto game_level_df

    Returns game_level_df with added rolling columns.
    """
    df = game_level_df.copy()

    # Build long-format: one row per team per game
    rows_A = df[["game_id", "season", "day_num", "team_A_id"]].copy()
    rows_A["team_id"] = rows_A["team_A_id"]
    rows_A["role"] = "A"
    for col in stat_cols:
        col_a = f"{col}_A"
        if col_a in df.columns:
            rows_A[col] = df[col_a].values

    rows_B = df[["game_id", "season", "day_num", "team_B_id"]].copy()
    rows_B["team_id"] = rows_B["team_B_id"]
    rows_B["role"] = "B"
    for col in stat_cols:
        col_b = f"{col}_B"
        if col_b in df.columns:
            rows_B[col] = df[col_b].values

    long = pd.concat([
        rows_A[["game_id", "season", "day_num", "team_id", "role"] + stat_cols],
        rows_B[["game_id", "season", "day_num", "team_id", "role"] + stat_cols],
    ], ignore_index=True)

    # Sort for rolling
    long = long.sort_values(["team_id", "season", "day_num"]).reset_index(drop=True)

    # Compute rolling per team-season group (shift(1) so we use PAST games only)
    prefix = f"roll{window}_"
    for col in stat_cols:
        long[f"{prefix}{col}"] = (
            long.groupby(["team_id", "season"])[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
            .astype("float32")
        )

    # Compute rolling net rating
    if "AdjO" in stat_cols and "AdjD" in stat_cols:
        long[f"{prefix}net"] = (
            long[f"{prefix}AdjO"] - long[f"{prefix}AdjD"]
        ).astype("float32")

    # Pivot back: separate A and B rows, then merge
    roll_cols = [f"{prefix}{c}" for c in stat_cols]
    if "AdjO" in stat_cols and "AdjD" in stat_cols:
        roll_cols.append(f"{prefix}net")

    long_A = long[long["role"] == "A"][["game_id"] + roll_cols].copy()
    long_B = long[long["role"] == "B"][["game_id"] + roll_cols].copy()

    long_A = long_A.rename(columns={c: f"{c}_A" for c in roll_cols})
    long_B = long_B.rename(columns={c: f"{c}_B" for c in roll_cols})

    # Merge back (game_id is unique per team-game row)
    # Need to handle that each game appears twice in long (once per team)
    # Take first occurrence per game_id for each role
    long_A = long_A.drop_duplicates("game_id")
    long_B = long_B.drop_duplicates("game_id")

    df = df.merge(long_A, on="game_id", how="left")
    df = df.merge(long_B, on="game_id", how="left")

    added_cols = [f"{c}_A" for c in roll_cols] + [f"{c}_B" for c in roll_cols]
    logger.info(f"Added rolling stats: {added_cols}")
    return df


def add_rolling_stats_to_game_level(
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Loads game_level.parquet, adds rolling stats, saves back.
    """
    from src.processing.game_builder import load_game_level

    logger.info("Loading game_level.parquet for rolling stats computation...")
    df = load_game_level()

    if "day_num" not in df.columns:
        logger.info(
            "  'day_num' not present (tournament-only dataset) — skipping rolling stats. "
            "Remove 'rolling' from --steps to suppress this message."
        )
        return df

    df = compute_rolling_stats(df, window=window)

    out_path = PROCESSED_DIR / "game_level.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved game_level.parquet with rolling stats → {out_path}")
    return df
