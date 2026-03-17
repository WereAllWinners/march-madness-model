"""
2026 NCAA tournament bracket structure.
Handles: 68 teams, 4 regions, First Four, seedings, matchup scheduling.

Pre-Selection Sunday (March 15):
  Derives projected bracket from Torvik AdjNetRtg top-68.

Post-Selection Sunday:
  Loads actual seeds from data/raw/manual/seeds_2026.csv
  Required columns:
    Seed      — region code + 2-digit seed, e.g. "W01", "X12", "Y11a" (a/b for play-in)
                Region codes: W=West, X=Midwest, Y=South, Z=East
    team_name — canonical team name matching the crosswalk (e.g. "Duke", "UConn")
  Optional:
    conf      — conference abbreviation (e.g. "ACC", "Big Ten")
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    MANUAL_DIR, KAGGLE_DIR, KAGGLE_FILES,
    REGIONS, ROUND_ORDER, LIVE_SEASON,
)

logger = logging.getLogger(__name__)


@dataclass
class Team:
    kaggle_id:    int
    name:         str
    seed:         int       # 1-16
    region:       str       # "East" | "West" | "South" | "Midwest"
    play_in:      bool = False
    torvik_rank:  int = 0
    adj_net_rtg:  float = 0.0
    conf:         str = ""


@dataclass
class BracketGame:
    game_id:    str
    round_name: str           # "FF" | "R64" | "R32" | "S16" | "E8" | "F4" | "NCG"
    region:     str
    team_A:     Optional[Team] = None
    team_B:     Optional[Team] = None
    winner:     Optional[Team] = None


# Standard seed matchups in the R64 (by region)
# Each tuple: (high_seed, low_seed) — high seed is the favorite
R64_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

# First Four play-in games (seeds 11 and 16 for at-large vs lowest automatic qualifiers)
FIRST_FOUR_SEEDS = [16, 16, 11, 11]  # 2 games for 16s, 2 for 11s


def build_bracket_2026(
    seeds_df: pd.DataFrame | None = None,
    torvik_df: pd.DataFrame | None = None,
    use_projected: bool = True,
) -> tuple[list[BracketGame], list[Team]]:
    """
    Constructs the 68-team 2026 bracket.

    Args:
        seeds_df:      actual seeds DataFrame (from kaggle_loader.load_seeds(2026))
                       If provided, takes precedence over projected.
        torvik_df:     current Torvik rankings (for projected bracket)
        use_projected: if True and seeds_df is None, build projected bracket

    Returns:
        (bracket_games, all_teams)
        bracket_games: ordered list of BracketGame objects (First Four first)
        all_teams:     list of all 68 Team objects
    """
    # Try to load actual 2026 seeds
    if seeds_df is None:
        seeds_df = _try_load_actual_seeds()

    if seeds_df is not None and not seeds_df.empty:
        logger.info("Building bracket from actual 2026 seeds.")
        all_teams = _teams_from_seeds(seeds_df, torvik_df)
    elif use_projected and torvik_df is not None:
        logger.info("Building projected bracket from Torvik rankings.")
        all_teams = _teams_from_torvik(torvik_df)
    else:
        logger.warning("No seed or Torvik data available. Returning empty bracket.")
        return [], []

    bracket_games = _build_bracket_structure(all_teams)
    return bracket_games, all_teams


def get_round_games(
    bracket: list[BracketGame],
    round_name: str,
) -> list[BracketGame]:
    """Returns all games for a specific round."""
    return [g for g in bracket if g.round_name == round_name]


def advance_winner(
    bracket: list[BracketGame],
    completed_game: BracketGame,
    winner: Team,
) -> list[BracketGame]:
    """
    Records the winner of a completed game and advances them to their
    next-round game in the bracket.

    Returns the updated bracket list.
    """
    # Find next-round game where this winner should appear
    next_game = _find_next_game(bracket, completed_game)
    if next_game is None:
        return bracket  # Final game (NCG)

    # Place winner as team_A or team_B in next game
    if next_game.team_A is None:
        next_game.team_A = winner
    else:
        next_game.team_B = winner

    completed_game.winner = winner
    return bracket


def teams_to_dataframe(teams: list[Team]) -> pd.DataFrame:
    """Converts list of Team objects to DataFrame for easy display."""
    return pd.DataFrame([
        {
            "kaggle_id": t.kaggle_id,
            "name": t.name,
            "seed": t.seed,
            "region": t.region,
            "play_in": t.play_in,
            "torvik_rank": t.torvik_rank,
            "adj_net_rtg": t.adj_net_rtg,
            "conf": t.conf,
        }
        for t in teams
    ]).sort_values(["region", "seed"]).reset_index(drop=True)


# ── Private helpers ────────────────────────────────────────────────────────────

def _try_load_actual_seeds() -> pd.DataFrame | None:
    """Tries to load 2026 seeds from manual override or Kaggle CSV."""
    # Check manual override first
    manual_path = MANUAL_DIR / "seeds_2026.csv"
    if manual_path.exists():
        logger.info(f"Loading actual seeds from {manual_path}")
        df = pd.read_csv(manual_path)
        return df

    # Try Kaggle seeds file for 2026
    kaggle_path = KAGGLE_DIR / KAGGLE_FILES.get("seeds", "MNCAATourneySeeds.csv")
    if kaggle_path.exists():
        df = pd.read_csv(kaggle_path)
        df = df.rename(columns={"Season": "season", "TeamID": "kaggle_id"})
        df_2026 = df[df["season"] == LIVE_SEASON]
        if not df_2026.empty:
            logger.info("Loaded 2026 seeds from Kaggle file.")
            return df_2026

    logger.info("No actual 2026 seeds found — will use projected bracket.")
    return None


def _teams_from_seeds(
    seeds_df: pd.DataFrame,
    torvik_df: pd.DataFrame | None,
) -> list[Team]:
    """
    Builds Team objects from seeds_2026.csv.

    Expected CSV columns:
      Seed      — e.g. "W01", "X12a" (region + 2-digit seed + optional play-in letter)
      team_name — canonical name matching crosswalk (e.g. "Duke", "UConn")
      conf      — optional conference abbreviation
    """
    from src.processing.crosswalk import load_crosswalk, clean_stats_name
    from src.features.engineer import load_feature_matrix

    # Build name → team_id lookup from crosswalk
    try:
        cw = load_crosswalk()
        name_to_id = dict(zip(cw["canonical_name"], cw["team_id"]))
    except Exception as e:
        logger.warning(f"Could not load crosswalk: {e}")
        name_to_id = {}

    # Build team_id → 2026 stats lookup from feature matrix
    stats_lookup: dict[int, dict] = {}
    try:
        fm = load_feature_matrix()
        season_stats = fm[fm["season"] == LIVE_SEASON].copy() if "season" in fm.columns else pd.DataFrame()
        for col in ("team_A_id", "team_B_id"):
            if col in season_stats.columns:
                for _, row in season_stats.drop_duplicates(col).iterrows():
                    tid = row.get(col)
                    if pd.notna(tid):
                        stats_lookup.setdefault(int(tid), row.to_dict())
    except Exception as e:
        logger.warning(f"Could not load feature matrix for stats: {e}")

    region_map = {"W": "West", "X": "Midwest", "Y": "South", "Z": "East"}
    teams = []
    for i, row in seeds_df.iterrows():
        seed_str = str(row.get("Seed", "")).strip()
        if len(seed_str) < 3:
            continue

        region_code = seed_str[0].upper()
        seed_num = int(seed_str[1:3])
        play_in = len(seed_str) > 3
        region = region_map.get(region_code, region_code)

        # Resolve team name → team_id via crosswalk
        raw_name = str(row.get("team_name", "")).strip()
        canonical = clean_stats_name(raw_name)
        team_id = name_to_id.get(canonical) or name_to_id.get(raw_name)

        if team_id is None:
            # Fuzzy fallback: find closest canonical name
            from rapidfuzz import process as rfp
            match = rfp.extractOne(canonical, list(name_to_id.keys()), score_cutoff=80)
            if match:
                team_id = name_to_id[match[0]]
                logger.debug(f"Fuzzy matched '{raw_name}' → '{match[0]}' (score {match[1]:.0f})")
            else:
                logger.warning(f"Could not match team '{raw_name}' — assigning placeholder ID {i}")
                team_id = -(i + 1)

        team_id = int(team_id)
        stats = stats_lookup.get(team_id, {})

        adj_o = float(stats.get("AdjO_A", stats.get("AdjO", 100)) or 100)
        adj_d = float(stats.get("AdjD_A", stats.get("AdjD", 100)) or 100)
        net_rtg = adj_o - adj_d

        teams.append(Team(
            kaggle_id=team_id,
            name=canonical or raw_name,
            seed=seed_num,
            region=region,
            play_in=play_in,
            torvik_rank=0,
            adj_net_rtg=round(net_rtg, 2),
            conf=str(row.get("conf", "")),
        ))

    return teams


def _teams_from_torvik(torvik_df: pd.DataFrame) -> list[Team]:
    """
    Builds a projected 68-team bracket from Torvik rankings.
    Uses top-68 teams by AdjNetRtg, assigning seeds 1-16 per region.
    NOTE: This is a rough projection — real seeding involves committee judgments.
    """
    from src.processing.crosswalk import load_crosswalk, clean_stats_name

    torvik_df = torvik_df.copy()

    # Map torvik_name → team_id via crosswalk (using canonical_name)
    try:
        cw = load_crosswalk()
        name_to_id = dict(zip(cw["canonical_name"], cw["team_id"]))
        torvik_df["canonical_name"] = torvik_df["torvik_name"].apply(clean_stats_name)
        torvik_df["team_id"] = torvik_df["canonical_name"].map(name_to_id).astype("Int32")
    except Exception as e:
        logger.warning(f"Could not load crosswalk for projected bracket: {e}")
        torvik_df["canonical_name"] = torvik_df["torvik_name"]
        torvik_df["team_id"] = pd.array(range(len(torvik_df)), dtype="Int32")

    # Use SRS (opponent-adjusted schedule strength metric) for ranking.
    # SRS is already opponent-adjusted; raw AdjO-AdjD is NOT, so small-conf teams
    # would otherwise rank too high.
    if "SRS" in torvik_df.columns:
        torvik_df["net_rtg"] = pd.to_numeric(torvik_df["SRS"], errors="coerce").fillna(0)
    elif "barthag" in torvik_df.columns:
        torvik_df["net_rtg"] = pd.to_numeric(torvik_df["barthag"], errors="coerce").fillna(0) * 30
    elif "AdjO" in torvik_df.columns and "AdjD" in torvik_df.columns:
        torvik_df["net_rtg"] = torvik_df["AdjO"] - torvik_df["AdjD"]
    else:
        torvik_df["net_rtg"] = 0

    # Also store adj_net_rtg as AdjO-AdjD for display purposes
    if "AdjO" in torvik_df.columns and "AdjD" in torvik_df.columns:
        torvik_df["adj_net_rtg_display"] = torvik_df["AdjO"] - torvik_df["AdjD"]
    else:
        torvik_df["adj_net_rtg_display"] = torvik_df["net_rtg"]

    top68 = torvik_df.nlargest(68, "net_rtg").reset_index(drop=True)

    teams = []
    # Assign to 4 regions, 17 teams per region (1 play-in per 16 seed)
    # Simple serpentine seeding: 1,2,3...16 per region
    for i, (_, row) in enumerate(top68.iterrows()):
        region_idx = i % 4
        seed = (i // 4) + 1
        play_in = seed > 16

        region = REGIONS[region_idx]
        team_id = int(row["team_id"]) if pd.notna(row.get("team_id")) else i
        name = str(row.get("canonical_name", row.get("torvik_name", f"Team{i}")))
        net_rtg = float(row.get("adj_net_rtg_display", row.get("net_rtg", 0)))
        rank = int(row.get("rank", i + 1)) if pd.notna(row.get("rank")) else i + 1

        if seed <= 16:
            teams.append(Team(
                kaggle_id=team_id,   # using team_id as the unique identifier
                name=name,
                seed=seed,
                region=region,
                play_in=play_in,
                torvik_rank=rank,
                adj_net_rtg=round(net_rtg, 2),
                conf=str(row.get("conf", "")),
            ))

    return teams


def _build_bracket_structure(teams: list[Team]) -> list[BracketGame]:
    """
    Creates all 67 BracketGame objects representing the full tournament.
    Returns them ordered: First Four → R64 → R32 → S16 → E8 → F4 → NCG.
    """
    # Group teams by region and seed
    region_teams: dict[str, dict[int, list[Team]]] = {r: {} for r in REGIONS}
    play_in_teams: dict[str, list[Team]] = {}

    for team in teams:
        region = team.region
        seed = team.seed
        if region not in region_teams:
            region_teams[region] = {}
        if seed not in region_teams[region]:
            region_teams[region][seed] = []
        region_teams[region][seed].append(team)

    games = []
    game_counter = [0]

    def new_game_id(round_name: str, region: str) -> str:
        game_counter[0] += 1
        return f"2026_{round_name}_{region}_{game_counter[0]:03d}"

    # First Four (play-in games for seeds 11 and 16)
    ff_games = []
    for region, seed_dict in region_teams.items():
        for seed, team_list in seed_dict.items():
            if len(team_list) == 2:  # Two teams competing for one R64 spot
                g = BracketGame(
                    game_id=new_game_id("FF", region),
                    round_name="FF",
                    region=region,
                    team_A=team_list[0],
                    team_B=team_list[1],
                )
                ff_games.append(g)
    games.extend(ff_games)

    # R64: one bracket slot per seed matchup per region
    r64_games = []
    for region in REGIONS:
        seed_dict = region_teams.get(region, {})
        for hi_seed, lo_seed in R64_MATCHUPS:
            hi_teams = seed_dict.get(hi_seed, [])
            lo_teams = seed_dict.get(lo_seed, [])

            # Handle play-in: if two teams for a seed, R64 slot is TBD
            team_hi = hi_teams[0] if hi_teams else None
            team_lo = lo_teams[0] if lo_teams else None

            # If a First Four game feeds this slot, leave the slot TBD
            if len(lo_teams) == 2:
                team_lo = None  # Will be filled by FF winner
            if len(hi_teams) == 2:
                team_hi = None

            g = BracketGame(
                game_id=new_game_id("R64", region),
                round_name="R64",
                region=region,
                team_A=team_hi,
                team_B=team_lo,
            )
            r64_games.append(g)
    games.extend(r64_games)

    # R32, S16, E8: within-region rounds
    for round_name, n_games_per_region in [("R32", 4), ("S16", 2), ("E8", 1)]:
        for region in REGIONS:
            for i in range(n_games_per_region):
                games.append(BracketGame(
                    game_id=new_game_id(round_name, region),
                    round_name=round_name,
                    region=region,
                ))

    # Final Four
    games.append(BracketGame(game_id=new_game_id("F4", "East-Midwest"), round_name="F4", region="East-Midwest"))
    games.append(BracketGame(game_id=new_game_id("F4", "South-West"), round_name="F4", region="South-West"))

    # National Championship
    games.append(BracketGame(game_id=new_game_id("NCG", "National"), round_name="NCG", region="National"))

    return games


def _find_next_game(
    bracket: list[BracketGame],
    completed: BracketGame,
) -> BracketGame | None:
    """Finds the next-round game that this winner should advance to."""
    round_order = ROUND_ORDER
    curr_idx = round_order.index(completed.round_name) if completed.round_name in round_order else -1
    if curr_idx < 0 or curr_idx >= len(round_order) - 1:
        return None

    next_round = round_order[curr_idx + 1]
    next_games = [g for g in bracket if g.round_name == next_round]

    if not next_games:
        return None

    # Regional rounds: find the next available game in same region
    if next_round not in ("F4", "NCG"):
        region_games = [g for g in next_games if g.region == completed.region]
        for g in region_games:
            if g.team_A is None or g.team_B is None:
                return g

    # Final Four: match regions
    if next_round == "F4":
        region_pairings = {
            "East": "East-Midwest", "Midwest": "East-Midwest",
            "South": "South-West", "West": "South-West",
        }
        target_region = region_pairings.get(completed.region)
        for g in next_games:
            if g.region == target_region and (g.team_A is None or g.team_B is None):
                return g

    # NCG: first available
    for g in next_games:
        if g.team_A is None or g.team_B is None:
            return g

    return None
