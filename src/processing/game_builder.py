"""
Constructs the game-level dataset.
Sources:
  - NCAA tournament results (scraped from SR bracket pages)          ~800 games
  - Regular season + conf tournament games for all tournament teams  ~25,000 games (optional)

Each row = one game with season stats for both teams.
team_A/team_B roles are randomly assigned (50/50) to prevent positional bias.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_DIR, TEAM_STAT_COLS,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, LIVE_SEASON, ALL_SEASONS,
)

logger = logging.getLogger(__name__)


def build_game_level_dataset(
    seasons: list[int] | None = None,
    include_regular_season: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build and save game_level.parquet from SR tournament + regular season results.

    Args:
        seasons:                years to include (default: ALL_SEASONS minus LIVE_SEASON)
        include_regular_season: if True, add regular season games for tournament teams
        random_seed:            seed for random team_A/team_B role assignment

    Returns game_level DataFrame (also saved to PROCESSED_DIR/game_level.parquet).
    """
    from src.ingestion.tourney_scraper import fetch_all_tournament_results
    from src.ingestion.torvik import load_all_seasons
    from src.processing.crosswalk import load_crosswalk, clean_stats_name

    if seasons is None:
        # Exclude live season (no results yet) and 2020 (no tournament)
        seasons = [s for s in ALL_SEASONS if s != LIVE_SEASON and s != 2020]

    logger.info(f"Building game-level dataset for seasons: {seasons}")

    # ── 1. Load crosswalk ───────────────────────────────────────────────────
    logger.info("Loading crosswalk...")
    crosswalk = load_crosswalk()
    bracket_to_canonical = dict(
        zip(crosswalk["bracket_name"], crosswalk["canonical_name"])
    )

    # ── 2. Load SR season stats ─────────────────────────────────────────────
    logger.info("Loading SR season stats...")
    stats_df = load_all_seasons(seasons)
    # Clean the team name (strip \xa0NCAA etc.)
    stats_df["canonical_name"] = stats_df["torvik_name"].apply(clean_stats_name)

    # Keep only the stat columns we need
    stat_cols = [c for c in TEAM_STAT_COLS if c in stats_df.columns]
    stats_sub = stats_df[["canonical_name", "year"] + stat_cols].copy()

    # ── 3. Load tournament results ──────────────────────────────────────────
    logger.info("Fetching tournament results from cache...")
    tourney_df = fetch_all_tournament_results(seasons)
    if tourney_df.empty:
        raise RuntimeError(
            "No tournament results found. Run scripts/fetch_data.py first."
        )
    logger.info(f"  Loaded {len(tourney_df)} tournament games across {tourney_df['year'].nunique()} seasons")

    # ── 4. Map bracket names → canonical names ──────────────────────────────
    tourney_df["W_canonical"] = tourney_df["W_team"].map(bracket_to_canonical)
    tourney_df["L_canonical"] = tourney_df["L_team"].map(bracket_to_canonical)

    # Log any unresolved names
    w_unresolved = tourney_df[tourney_df["W_canonical"].isna()]["W_team"].unique()
    l_unresolved = tourney_df[tourney_df["L_canonical"].isna()]["L_team"].unique()
    all_unresolved = set(w_unresolved) | set(l_unresolved)
    if all_unresolved:
        logger.warning(f"  {len(all_unresolved)} bracket names not in crosswalk: {sorted(all_unresolved)[:10]}")

    # Drop games where we couldn't resolve team names
    tourney_df = tourney_df.dropna(subset=["W_canonical", "L_canonical"]).copy()
    logger.info(f"  {len(tourney_df)} games after name resolution")

    # ── 5. Random team_A/team_B assignment ─────────────────────────────────
    rng = np.random.default_rng(random_seed)
    flip_mask = rng.random(len(tourney_df)) < 0.5

    tourney_df["team_A_name"] = np.where(flip_mask, tourney_df["L_canonical"], tourney_df["W_canonical"])
    tourney_df["team_B_name"] = np.where(flip_mask, tourney_df["W_canonical"], tourney_df["L_canonical"])
    tourney_df["team_A_win"]  = (~flip_mask).astype("int8")
    tourney_df["seed_A"]      = np.where(flip_mask, tourney_df["L_seed"], tourney_df["W_seed"]).astype("float32")
    tourney_df["seed_B"]      = np.where(flip_mask, tourney_df["W_seed"], tourney_df["L_seed"]).astype("float32")
    tourney_df["score_A"]     = np.where(flip_mask, tourney_df["L_score"], tourney_df["W_score"])
    tourney_df["score_B"]     = np.where(flip_mask, tourney_df["W_score"], tourney_df["L_score"])

    # Margin from team_A perspective (positive = team_A won)
    tourney_df["score_A"] = pd.to_numeric(tourney_df["score_A"], errors="coerce")
    tourney_df["score_B"] = pd.to_numeric(tourney_df["score_B"], errors="coerce")
    tourney_df["margin"] = (tourney_df["score_A"] - tourney_df["score_B"]).astype("float32")

    tourney_df["game_id"] = (
        tourney_df["year"].astype(str) + "_"
        + tourney_df["round"] + "_"
        + tourney_df["team_A_name"].str.replace(" ", "_")
        + "_vs_"
        + tourney_df["team_B_name"].str.replace(" ", "_")
    )
    tourney_df["season"] = tourney_df["year"].astype("int16")
    tourney_df["is_tournament"] = True
    tourney_df = tourney_df.rename(columns={"round": "tournament_round"})

    # ── 6. Merge season stats for each team ────────────────────────────────
    logger.info("Merging season stats for team_A...")
    games = tourney_df.merge(
        stats_sub.rename(columns={"canonical_name": "team_A_name", "year": "season"}),
        on=["team_A_name", "season"],
        how="left",
    )
    # Rename merged stat cols with _A suffix
    for col in stat_cols:
        if col in games.columns:
            games = games.rename(columns={col: f"{col}_A"})

    logger.info("Merging season stats for team_B...")
    games = games.merge(
        stats_sub.rename(columns={"canonical_name": "team_B_name", "year": "season"}),
        on=["team_B_name", "season"],
        how="left",
        suffixes=("", "_B_tmp"),
    )
    for col in stat_cols:
        # After merge, team_B stats come in without suffix (since _A was already renamed)
        if col in games.columns:
            games = games.rename(columns={col: f"{col}_B"})

    # Log merge quality
    n_total = len(games)
    n_missing_A = games[[f"{c}_A" for c in stat_cols if f"{c}_A" in games.columns]].isna().all(axis=1).sum()
    n_missing_B = games[[f"{c}_B" for c in stat_cols if f"{c}_B" in games.columns]].isna().all(axis=1).sum()
    logger.info(f"  Stats merge: {n_total} games, {n_missing_A} missing team_A stats, {n_missing_B} missing team_B stats")

    # ── 7a. Add rest days before tournament ─────────────────────────────────
    slug_to_canonical = _build_slug_to_canonical(seasons, bracket_to_canonical)
    rest_map = _compute_rest_days(seasons, slug_to_canonical)
    games["rest_days_A"] = games.apply(
        lambda r: rest_map.get((r["team_A_name"], int(r["season"])), float("nan")), axis=1
    ).astype("float32")
    games["rest_days_B"] = games.apply(
        lambda r: rest_map.get((r["team_B_name"], int(r["season"])), float("nan")), axis=1
    ).astype("float32")
    n_with_rest = games["rest_days_A"].notna().sum()
    logger.info(f"  Rest days computed for {n_with_rest}/{len(games)} tournament games")

    # ── 7. Optionally add regular season games ──────────────────────────────
    if include_regular_season:
        reg_games = _build_reg_season_rows(seasons, stats_sub, stat_cols, bracket_to_canonical, rng)
        if not reg_games.empty:
            logger.info(f"  Adding {len(reg_games)} regular season games to dataset")
            games = pd.concat([games, reg_games], ignore_index=True)

    # ── 8. Assign crosswalk team_ids ────────────────────────────────────────
    name_to_id = dict(zip(crosswalk["canonical_name"], crosswalk["team_id"]))
    games["team_A_id"] = games["team_A_name"].map(name_to_id).astype("Int32")
    games["team_B_id"] = games["team_B_name"].map(name_to_id).astype("Int32")

    # ── 9. Select final columns ─────────────────────────────────────────────
    meta_cols = [
        "game_id", "season", "is_tournament", "tournament_round",
        "team_A_id", "team_B_id", "team_A_name", "team_B_name",
        "team_A_win", "score_A", "score_B", "margin", "seed_A", "seed_B",
        "rest_days_A", "rest_days_B",
    ]
    stat_cols_A = [f"{c}_A" for c in TEAM_STAT_COLS if f"{c}_A" in games.columns]
    stat_cols_B = [f"{c}_B" for c in TEAM_STAT_COLS if f"{c}_B" in games.columns]
    keep_cols = [c for c in meta_cols + stat_cols_A + stat_cols_B if c in games.columns]
    games = games[keep_cols].copy()

    # ── 10. Save ─────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "game_level.parquet"
    games.to_parquet(out_path, index=False)

    n_tourn = games["is_tournament"].sum()
    n_reg   = (~games["is_tournament"]).sum()
    logger.info(
        f"game_level.parquet: {len(games)} games "
        f"({n_tourn} tournament, {n_reg} regular season) "
        f"across {games['season'].nunique()} seasons → {out_path}"
    )
    return games


def _build_reg_season_rows(
    seasons: list[int],
    stats_sub: pd.DataFrame,
    stat_cols: list[str],
    bracket_to_canonical: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Builds regular season game rows from cached schedule parquets.
    Each game: winning team = team_A (since we kept W rows for dedup);
    then randomly flip to prevent positional bias.
    """
    from src.ingestion.schedule_scraper import build_reg_season_games

    # slug-to-canonical mapping: slugs fetched from bracket pages map to bracket names,
    # which we then map to canonical. Build slug→canonical via slug_map parquets.
    slug_to_canonical = _build_slug_to_canonical(seasons, bracket_to_canonical)

    raw = build_reg_season_games(seasons, game_types=("REG", "CTOURN"))
    if raw.empty:
        logger.info("  No regular season schedule data found — skipping")
        return pd.DataFrame()

    # Map slug → canonical name for the team whose schedule we read
    raw["team_canonical"] = raw["slug"].map(slug_to_canonical)
    # Map opponent_raw → canonical name via the same crosswalk logic
    # opponent_raw names come from SR schedule pages and should match stats canonical names
    all_canonical = set(stats_sub["canonical_name"].dropna())
    raw["opp_canonical"] = raw["opponent_raw"].apply(
        lambda n: n if n in all_canonical else _fuzzy_resolve(n, all_canonical)
    )

    # Drop rows where we can't resolve either team
    raw = raw.dropna(subset=["team_canonical", "opp_canonical"]).copy()
    raw = raw[raw["team_canonical"] != raw["opp_canonical"]].copy()

    if raw.empty:
        return pd.DataFrame()

    # Random A/B assignment (raw["result"] == "W" means the schedule team won)
    flip_mask = rng.random(len(raw)) < 0.5
    team_won = (raw["result"] == "W").values

    rows = pd.DataFrame()
    rows["team_A_name"] = np.where(flip_mask, raw["opp_canonical"].values, raw["team_canonical"].values)
    rows["team_B_name"] = np.where(flip_mask, raw["team_canonical"].values, raw["opp_canonical"].values)
    # team_canonical won if result==W; after flip, team_A_win depends on flip
    rows["team_A_win"]  = np.where(flip_mask, (~team_won).astype("int8"), team_won.astype("int8"))
    rows["score_A"]     = np.where(flip_mask, raw["opp_score"].values, raw["team_score"].values).astype("float32")
    rows["score_B"]     = np.where(flip_mask, raw["team_score"].values, raw["opp_score"].values).astype("float32")
    rows["margin"]      = (rows["score_A"] - rows["score_B"]).astype("float32")
    rows["season"]      = raw["year"].astype("int16").values
    rows["is_tournament"] = False
    rows["tournament_round"] = raw["game_type"].values
    rows["seed_A"]      = float("nan")
    rows["seed_B"]      = float("nan")
    rows["game_id"]     = (
        raw["year"].astype(str) + "_reg_"
        + raw["date"].astype(str).str.replace(",", "").str.replace(" ", "_") + "_"
        + rows["team_A_name"].str.replace(" ", "_") + "_vs_"
        + rows["team_B_name"].str.replace(" ", "_")
    )
    rows["rest_days_A"] = float("nan")
    rows["rest_days_B"] = float("nan")

    # Merge season stats for team_A
    rows = rows.merge(
        stats_sub.rename(columns={"canonical_name": "team_A_name", "year": "season"}),
        on=["team_A_name", "season"],
        how="left",
    )
    for col in stat_cols:
        if col in rows.columns:
            rows = rows.rename(columns={col: f"{col}_A"})

    # Merge season stats for team_B
    rows = rows.merge(
        stats_sub.rename(columns={"canonical_name": "team_B_name", "year": "season"}),
        on=["team_B_name", "season"],
        how="left",
        suffixes=("", "_B_tmp"),
    )
    for col in stat_cols:
        if col in rows.columns:
            rows = rows.rename(columns={col: f"{col}_B"})

    # Drop rows missing stats for either team (non-D1 opponents, etc.)
    stat_cols_A = [f"{c}_A" for c in stat_cols if f"{c}_A" in rows.columns]
    stat_cols_B = [f"{c}_B" for c in stat_cols if f"{c}_B" in rows.columns]
    if stat_cols_A:
        rows = rows.dropna(subset=[stat_cols_A[0], stat_cols_B[0] if stat_cols_B else stat_cols_A[0]])

    n_before = len(raw)
    logger.info(f"  Regular season: {len(rows)}/{n_before} games kept after stats merge")
    return rows.reset_index(drop=True)


def _build_slug_to_canonical(seasons: list[int], bracket_to_canonical: dict) -> dict:
    """Build slug → canonical_name mapping from cached slug_map parquets."""
    from pathlib import Path
    sr_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "sports_ref"
    slug_to_canonical = {}
    for year in seasons:
        cache = sr_dir / f"slug_map_{year}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            for _, row in df.iterrows():
                bracket_name = row["bracket_name"]
                slug         = row["slug"]
                canonical    = bracket_to_canonical.get(bracket_name, bracket_name)
                slug_to_canonical[slug] = canonical
    return slug_to_canonical


def _compute_rest_days(seasons: list[int], slug_to_canonical: dict) -> dict:
    """
    Computes days of rest before tournament for each team-year.
    rest_days = days between last non-NCAA game and first NCAA game in schedule.
    Returns dict: (canonical_name, year) -> rest_days (int).
    """
    sr_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "sports_ref" / "schedules"
    rest_map = {}

    for year in seasons:
        sched_dir = sr_dir / str(year)
        if not sched_dir.exists():
            continue
        for parquet_path in sched_dir.glob("*.parquet"):
            slug = parquet_path.stem
            canonical = slug_to_canonical.get(slug)
            if canonical is None:
                continue
            try:
                df = pd.read_parquet(parquet_path)
                if df.empty or "date" not in df.columns or "game_type" not in df.columns:
                    continue
                df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date_parsed"])
                pre  = df[df["game_type"].isin(["REG", "CTOURN"])]
                ncaa = df[df["game_type"] == "NCAA"]
                if pre.empty or ncaa.empty:
                    continue
                rest = (ncaa["date_parsed"].min() - pre["date_parsed"].max()).days
                if 0 <= rest <= 30:
                    rest_map[(canonical, year)] = rest
            except Exception:
                continue

    return rest_map


_fuzzy_cache: dict[str, str | None] = {}

def _fuzzy_resolve(name: str, canonical_set: set) -> str | None:
    """Fuzzy-match a team name to the canonical stats name set."""
    if name in _fuzzy_cache:
        return _fuzzy_cache[name]
    try:
        from rapidfuzz import fuzz, process as rfp
        result = rfp.extractOne(name, list(canonical_set), scorer=fuzz.WRatio, score_cutoff=85)
        resolved = result[0] if result else None
    except Exception:
        resolved = None
    _fuzzy_cache[name] = resolved
    return resolved


def load_game_level() -> pd.DataFrame:
    """Loads the processed game-level dataset."""
    path = PROCESSED_DIR / "game_level.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "game_level.parquet not found. Run scripts/build_features.py --steps games"
        )
    return pd.read_parquet(path)
