"""
Scrapes school-season stats from sports-reference.com/cbb/.
Uses pandas.read_html() with polite rate limiting.
Primarily used for supplemental data; Torvik is the main efficiency source.
"""

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    SPORTS_REF_DIR, SPORTS_REF_STATS_URL,
    SPORTS_REF_DELAY_SECS, ALL_SEASONS,
)

logger = logging.getLogger(__name__)


def fetch_season_stats(year: int, delay: float = SPORTS_REF_DELAY_SECS) -> pd.DataFrame:
    """
    Fetches school-level stats from Sports-Reference for a given season year.
    year = season end year (e.g., 2025 = 2024-25 season).
    Saves to SPORTS_REF_DIR/season_stats_{year}.parquet.
    """
    url = SPORTS_REF_STATS_URL.format(year=year)
    logger.info(f"Fetching Sports-Reference stats for {year}: {url}")

    try:
        tables = pd.read_html(url, header=0)
        if not tables:
            raise ValueError("No tables found on page")
        df = tables[0]
    except Exception as e:
        logger.warning(f"  Could not fetch Sports-Ref {year}: {e}")
        return pd.DataFrame()

    df = _clean_sports_ref_table(df, year)
    out_path = SPORTS_REF_DIR / f"season_stats_{year}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"  Saved {len(df)} teams → {out_path}")

    time.sleep(delay)
    return df


def load_season_stats(year: int) -> pd.DataFrame:
    """Loads cached parquet for a given year."""
    path = SPORTS_REF_DIR / f"season_stats_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Sports-Ref data missing for {year}. Run fetch_season_stats({year}).")
    return pd.read_parquet(path)


def fetch_all_seasons(
    years: list[int] = None,
    force_refresh: bool = False,
) -> None:
    """Fetches all seasons with rate limiting, skipping cached files."""
    if years is None:
        years = ALL_SEASONS

    for year in years:
        path = SPORTS_REF_DIR / f"season_stats_{year}.parquet"
        if path.exists() and not force_refresh:
            logger.info(f"  Skipping Sports-Ref {year} (cached)")
            continue
        fetch_season_stats(year)


def load_all_seasons(years: list[int] = None) -> pd.DataFrame:
    """Loads all cached Sports-Reference season stats."""
    if years is None:
        years = ALL_SEASONS

    dfs = []
    for year in years:
        path = SPORTS_REF_DIR / f"season_stats_{year}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        logger.warning("No Sports-Reference data found.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _clean_sports_ref_table(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Clean raw Sports-Reference table:
    - Drop mid-table header repeats (rows where 'School' == 'School')
    - Flatten MultiIndex columns if present
    - Cast numerics to float32
    - Add year column
    """
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col if c != "").strip()
            for col in df.columns.values
        ]

    # Remove repeated header rows
    df = df[df.iloc[:, 0] != df.columns[0]].copy()

    # Rename first column to sr_name if it looks like school name
    first_col = df.columns[0]
    if first_col.lower() in ("school", "team", ""):
        df = df.rename(columns={first_col: "sr_name"})
    else:
        df.insert(0, "sr_name", df[first_col])

    # Strip footnote markers from school names
    df["sr_name"] = df["sr_name"].astype(str).str.replace(r"\s*\*", "", regex=True).str.strip()

    # Drop rows with NaN school names
    df = df[df["sr_name"].notna() & (df["sr_name"] != "nan")].copy()

    # Cast all non-string columns to float32
    for col in df.columns:
        if col != "sr_name":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df["year"] = year

    # Standardize common column names
    rename_map = {
        "W": "W_sr", "L": "L_sr", "W-L%": "W_pct_sr",
        "SRS": "SRS", "SOS": "SOS_sr",
        "Tm.": "pts_per_g", "Opp.": "opp_pts_per_g",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df.reset_index(drop=True)
