"""
Scrapes regular season + conference tournament game schedules for NCAA tournament teams.
Adds ~25,000 additional training examples to supplement the ~800 tournament-only dataset.

Approach:
  1. Extract school slugs from cached bracket parquet files (re-parse bracket HTML)
  2. For each (slug, year), fetch the team's schedule page
  3. Keep Type==REG and Type==CTOURN games; parse W/L, scores, opponent
  4. Save per-team-year to data/raw/sports_ref/schedules/{year}/{slug}.parquet
  5. build_all_reg_season_games() aggregates into a deduplicated game-level dataset

URL pattern: https://www.sports-reference.com/cbb/schools/{slug}/men/{year}-schedule.html
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

_SR_BRACKET_URL = "https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
_SR_SCHED_URL   = "https://www.sports-reference.com/cbb/schools/{slug}/men/{year}-schedule.html"
_RAW_DIR        = Path(__file__).parent.parent.parent / "data" / "raw" / "sports_ref"
_SCHED_DIR      = _RAW_DIR / "schedules"
_DELAY          = 4.0


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_slug_map(year: int, force_refresh: bool = False) -> dict[str, str]:
    """
    Returns {bracket_team_name: slug} for all tournament teams in a given year.
    Fetches and parses the bracket page; caches to slug_map_{year}.parquet.
    """
    cache = _RAW_DIR / f"slug_map_{year}.parquet"
    if cache.exists() and not force_refresh:
        df = pd.read_parquet(cache)
        return dict(zip(df["bracket_name"], df["slug"]))

    url = _SR_BRACKET_URL.format(year=year)
    log.info(f"  Fetching slug map for {year}: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"  Failed to fetch bracket for {year}: {e}")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    slugs = {}
    # Search all bracket divs for school links
    for bracket in soup.find_all("div", id="bracket"):
        for link in bracket.find_all("a", href=lambda h: h and "/cbb/schools/" in h and "/men/" in h):
            href = link["href"]  # /cbb/schools/connecticut/men/2024.html
            parts = href.split("/")
            if len(parts) >= 5:
                slug = parts[3]
                name = link.text.strip()
                if name and slug and name not in slugs:
                    slugs[name] = slug

    df = pd.DataFrame(list(slugs.items()), columns=["bracket_name", "slug"])
    df.to_parquet(cache, index=False)
    log.info(f"  Slug map: {len(slugs)} teams → {cache.name}")
    return slugs


def fetch_team_schedule(slug: str, year: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches the regular season schedule for one team-year.
    Returns DataFrame with columns:
        slug, year, date, type, home_away, opponent_raw, team_score, opp_score, result
    Returns empty DataFrame on error.
    """
    out_dir = _SCHED_DIR / str(year)
    out_path = out_dir / f"{slug}.parquet"
    if out_path.exists() and not force_refresh:
        return pd.read_parquet(out_path)

    url = _SR_SCHED_URL.format(slug=slug, year=year)
    try:
        tables = pd.read_html(url, attrs={"id": "schedule"})
        if not tables:
            return pd.DataFrame()
        df = tables[0]
    except Exception as e:
        log.debug(f"  Schedule fetch failed {slug}/{year}: {e}")
        return pd.DataFrame()

    parsed = _parse_schedule(df, slug, year)
    if parsed.empty:
        return pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(out_path, index=False)
    return parsed


def fetch_all_schedules(
    years: list[int],
    force_refresh: bool = False,
    game_types: tuple[str, ...] = ("REG", "CTOURN"),
) -> None:
    """
    Fetches schedules for all tournament teams across all years.
    Saves one parquet per (year, team). Respects 4s rate limit.

    Args:
        years:        list of seasons to fetch
        force_refresh: re-fetch even if cached
        game_types:   which game types to keep ('REG', 'CTOURN', 'NCAA')
    """
    total_fetched = 0
    for year in years:
        slug_map = fetch_slug_map(year, force_refresh=force_refresh)
        if not slug_map:
            log.warning(f"  No slugs for {year}, skipping")
            continue

        out_dir = _SCHED_DIR / str(year)
        slugs_to_fetch = []
        for name, slug in slug_map.items():
            out_path = out_dir / f"{slug}.parquet"
            if not out_path.exists() or force_refresh:
                slugs_to_fetch.append((name, slug))

        if not slugs_to_fetch:
            log.info(f"  {year}: all {len(slug_map)} schedules already cached")
            continue

        log.info(f"  {year}: fetching {len(slugs_to_fetch)}/{len(slug_map)} schedules...")
        time.sleep(_DELAY)  # pause after slug map fetch

        for i, (name, slug) in enumerate(slugs_to_fetch):
            fetch_team_schedule(slug, year, force_refresh=force_refresh)
            total_fetched += 1
            if i < len(slugs_to_fetch) - 1:
                time.sleep(_DELAY)

        log.info(f"  {year}: done ({len(slug_map)} teams)")

    log.info(f"Schedule fetch complete. Total fetched: {total_fetched}")


def build_reg_season_games(
    years: list[int],
    game_types: tuple[str, ...] = ("REG", "CTOURN"),
) -> pd.DataFrame:
    """
    Aggregates all cached schedule parquets into a deduplicated game-level DataFrame.

    Returns DataFrame with columns:
        year, date, slug_home, slug_away, team_name, opponent_raw,
        team_score, opp_score, result, home_away, game_type
    Deduplication: each game appears once (canonical: lower slug alphabetically = team_A).
    """
    frames = []
    for year in years:
        out_dir = _SCHED_DIR / str(year)
        if not out_dir.exists():
            continue
        for parquet_path in out_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_games = pd.concat(frames, ignore_index=True)

    # Filter game types
    if game_types:
        all_games = all_games[all_games["game_type"].isin(game_types)].copy()

    # Deduplicate: each game appears in both teams' schedules
    # Use (year, date, sorted team scores) as dedup key
    all_games["_score_pair"] = all_games.apply(
        lambda r: f"{min(r['team_score'], r['opp_score'])}_{max(r['team_score'], r['opp_score'])}",
        axis=1,
    )
    all_games["_dedup_key"] = (
        all_games["year"].astype(str) + "_"
        + all_games["date"].astype(str) + "_"
        + all_games["_score_pair"]
    )
    # Keep one row per unique game (the winning team's row)
    all_games = all_games[all_games["result"] == "W"].drop_duplicates(subset=["_dedup_key"])
    all_games = all_games.drop(columns=["_score_pair", "_dedup_key"])

    log.info(
        f"Regular season games: {len(all_games)} unique games "
        f"across {all_games['year'].nunique()} seasons"
    )
    return all_games.reset_index(drop=True)


# ── Internal helpers ────────────────────────────────────────────────────────────

def _parse_schedule(df: pd.DataFrame, slug: str, year: int) -> pd.DataFrame:
    """Parse a raw schedule DataFrame from pd.read_html into our schema."""
    # The schedule table has columns like: G, Date, Time, Type, Unnamed:4, Opponent, Conf, SRS, Unnamed:8, Tm, Opp, OT, W, L, Streak, Arena
    # Unnamed:4 = home/away indicator (@=away, N=neutral, NaN=home)
    # Unnamed:8 = result W or L

    # Find the right columns by position if needed
    cols = df.columns.tolist()

    # Identify key columns
    type_col   = _find_col(cols, ["Type"])
    opp_col    = _find_col(cols, ["Opponent"])
    result_col = _find_col_by_position(cols, 8)   # Unnamed: 8 or similar
    tm_col     = _find_col(cols, ["Tm"])
    opp_sc_col = _find_col(cols, ["Opp"])
    ha_col     = _find_col_by_position(cols, 4)   # Unnamed: 4 = home/away

    if not all([type_col, opp_col, tm_col, opp_sc_col]):
        log.debug(f"  {slug}/{year}: missing expected columns in schedule table")
        return pd.DataFrame()

    # Filter to game rows (drop header rows that repeat)
    df = df.dropna(subset=[tm_col]).copy()
    df = df[pd.to_numeric(df[tm_col], errors="coerce").notna()].copy()

    # Filter to relevant game types
    if type_col:
        df = df[df[type_col].isin(["REG", "CTOURN", "NCAA"])].copy()

    if df.empty:
        return pd.DataFrame()

    n = len(df)
    out = pd.DataFrame({
        "slug":         [slug] * n,
        "year":         [year] * n,
        "date":         df["Date"].values if "Date" in df.columns else [""] * n,
        "game_type":    df[type_col].values if type_col else ["REG"] * n,
        "home_away":    df[ha_col].fillna("H").values if ha_col else ["H"] * n,
        "opponent_raw": df[opp_col].apply(_clean_opponent).values,
        "team_score":   pd.to_numeric(df[tm_col], errors="coerce").values,
        "opp_score":    pd.to_numeric(df[opp_sc_col], errors="coerce").values,
        "result":       df[result_col].astype(str).str.strip().str[0].values if result_col else [None] * n,
    })

    # Drop rows with missing scores
    out = out.dropna(subset=["team_score", "opp_score"]).copy()
    out["team_score"] = out["team_score"].astype("int16")
    out["opp_score"]  = out["opp_score"].astype("int16")

    return out.reset_index(drop=True)


def _clean_opponent(raw: str) -> str:
    """Strip seed annotations like '\xa0(15)' and ranking annotations."""
    if not isinstance(raw, str):
        return str(raw)
    # Remove \xa0(N) type seed/ranking annotations
    cleaned = re.sub(r"\xa0\(\d+\)", "", raw)
    cleaned = re.sub(r"\s*\(\d+\)\s*$", "", cleaned)
    return cleaned.strip()


def _find_col(cols: list, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _find_col_by_position(cols: list, pos: int) -> str | None:
    """Find the column at a given index position."""
    if pos < len(cols):
        return cols[pos]
    return None
