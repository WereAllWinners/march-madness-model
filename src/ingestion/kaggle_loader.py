"""
Loads manually-downloaded Kaggle 'March Machine Learning Mania' CSVs.
Does NOT download files — user must place them in KAGGLE_DIR manually.

Download from:
  https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data

Required files: MTeams.csv, MNCAATourneySeeds.csv,
                MNCAATourneyCompactResults.csv, MRegularSeasonCompactResults.csv
Optional (for richer stats): MNCAATourneyDetailedResults.csv,
                              MRegularSeasonDetailedResults.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import KAGGLE_DIR, KAGGLE_FILES, KAGGLE_REQUIRED, KAGGLE_DOWNLOAD_URL

logger = logging.getLogger(__name__)


def validate_kaggle_files() -> dict[str, bool]:
    """
    Checks KAGGLE_DIR for each expected file.
    Returns dict: file_key → True/False (exists).
    Raises FileNotFoundError with clear instructions if required files are missing.
    """
    status = {}
    missing_required = []

    for key, fname in KAGGLE_FILES.items():
        path = KAGGLE_DIR / fname
        exists = path.exists()
        status[key] = exists
        if not exists:
            if key in KAGGLE_REQUIRED:
                missing_required.append(fname)
            else:
                logger.warning(f"  Optional Kaggle file missing: {fname}")

    if missing_required:
        missing_str = "\n  ".join(missing_required)
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"MISSING REQUIRED KAGGLE FILES:\n  {missing_str}\n\n"
            f"Download from:\n  {KAGGLE_DOWNLOAD_URL}\n\n"
            f"Place all CSV files in:\n  {KAGGLE_DIR}\n"
            f"{'='*60}"
        )

    return status


def load_teams() -> pd.DataFrame:
    """
    Loads MTeams.csv.
    Returns: DataFrame with TeamID (int32), TeamName (str)
    """
    path = KAGGLE_DIR / KAGGLE_FILES["teams"]
    df = pd.read_csv(path)
    df = df.rename(columns={"TeamID": "kaggle_id", "TeamName": "kaggle_name"})
    df["kaggle_id"] = df["kaggle_id"].astype("int32")
    df["kaggle_name"] = df["kaggle_name"].astype(str)
    return df[["kaggle_id", "kaggle_name"]]


def load_seeds(year: int | None = None) -> pd.DataFrame:
    """
    Loads MNCAATourneySeeds.csv.
    Parses seed string (e.g., 'W01a') into: region, seed_num, play_in.
    If year provided, filters to that season.

    Returns: Season, region, seed_num, play_in, Seed (raw), kaggle_id
    """
    path = KAGGLE_DIR / KAGGLE_FILES["seeds"]
    df = pd.read_csv(path)
    df = df.rename(columns={"Season": "season", "TeamID": "kaggle_id"})
    df["kaggle_id"] = df["kaggle_id"].astype("int32")
    df["season"] = df["season"].astype("int16")

    # Parse seed string: "W01" → region=W, seed_num=1, play_in=False
    #                    "W16a" → region=W, seed_num=16, play_in=True
    df["region"] = df["Seed"].str[0]
    df["seed_num"] = df["Seed"].str[1:3].astype(int).astype("int8")
    df["play_in"] = df["Seed"].str.len() > 3

    # Check for manually provided 2026 seeds override
    manual_path = Path(__file__).parent.parent.parent / "data" / "raw" / "manual" / "seeds_2026.csv"
    if manual_path.exists():
        manual_df = pd.read_csv(manual_path)
        manual_df = manual_df.rename(columns={"Season": "season", "TeamID": "kaggle_id"})
        manual_df["kaggle_id"] = manual_df["kaggle_id"].astype("int32")
        manual_df["season"] = manual_df["season"].astype("int16")
        if "Seed" in manual_df.columns:
            manual_df["region"] = manual_df["Seed"].str[0]
            manual_df["seed_num"] = manual_df["Seed"].str[1:3].astype(int).astype("int8")
            manual_df["play_in"] = manual_df["Seed"].str.len() > 3
        # Remove any existing 2026 rows and add manual override
        df = df[df["season"] != 2026]
        df = pd.concat([df, manual_df], ignore_index=True)

    if year is not None:
        df = df[df["season"] == year].copy()

    return df.reset_index(drop=True)


def load_tourney_results(
    years: list[int] | None = None,
    detailed: bool = True,
) -> pd.DataFrame:
    """
    Loads tournament game results.
    detailed=True uses MNCAATourneyDetailedResults.csv (more columns).
    Falls back to compact if detailed not available.

    Returns DataFrame with winner/loser team IDs, scores, box stats.
    """
    key = "tourney_detailed" if detailed else "tourney_compact"
    path = KAGGLE_DIR / KAGGLE_FILES[key]

    if not path.exists():
        logger.warning(f"Detailed tourney results not found, falling back to compact.")
        path = KAGGLE_DIR / KAGGLE_FILES["tourney_compact"]
        detailed = False

    df = pd.read_csv(path)
    df = df.rename(columns={"Season": "season"})
    df["season"] = df["season"].astype("int16")

    if years:
        df = df[df["season"].isin(years)].copy()

    return df.reset_index(drop=True)


def load_regular_season(
    years: list[int] | None = None,
    detailed: bool = False,
) -> pd.DataFrame:
    """
    Loads regular season game results.
    detailed=True uses MRegularSeasonDetailedResults.csv.

    Returns same schema as load_tourney_results().
    """
    key = "reg_detailed" if detailed else "reg_compact"
    path = KAGGLE_DIR / KAGGLE_FILES[key]

    if not path.exists() and detailed:
        logger.warning("Detailed regular season not found, falling back to compact.")
        path = KAGGLE_DIR / KAGGLE_FILES["reg_compact"]
        detailed = False

    df = pd.read_csv(path)
    df = df.rename(columns={"Season": "season"})
    df["season"] = df["season"].astype("int16")

    if years:
        df = df[df["season"].isin(years)].copy()

    return df.reset_index(drop=True)
