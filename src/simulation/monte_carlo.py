"""
Monte Carlo tournament simulation engine.
Runs N_SIMULATIONS iterations of the full 68-team bracket.
Each game is resolved via Bernoulli(p) where p = model win probability.

Output: DataFrame with advancement probabilities per team per round,
        plus expected bracket score and upset probability matrix.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    N_SIMULATIONS, RANDOM_SEED, ROUND_ORDER, ROUND_POINTS,
    FEATURES_DIR, LIVE_SEASON, FEATURE_COLS, SPORTS_REF_DIR, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def run_simulation(
    all_teams: list,         # list[Team] from bracket.py
    bracket_template: list,  # list[BracketGame] from build_bracket_2026()
    feature_matrix_df: pd.DataFrame,
    artifacts: dict,
    n_sims: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
    progress_callback=None,  # optional callable(sim_idx, n_sims)
    injury_adjustments: dict = None,  # team_name → {"AdjO_delta": float, "AdjD_delta": float}
) -> pd.DataFrame:
    """
    Main simulation entry point.

    Args:
        all_teams:          68 Team objects
        bracket_template:   ordered BracketGame list (not mutated)
        feature_matrix_df:  feature_matrix.parquet loaded
        artifacts:          loaded model artifacts
        n_sims:             number of Monte Carlo iterations
        seed:               random seed
        progress_callback:  optional callback for progress updates

    Returns DataFrame (one row per team):
        kaggle_id, name, seed, region,
        p_R64, p_R32, p_S16, p_E8, p_F4, p_NCG, p_Champion,
        expected_wins, expected_bracket_pts
    """
    from src.simulation.bracket import Team, BracketGame, advance_winner

    rng = np.random.default_rng(seed)

    # Track advancement counts: team_id → {round_name: count}
    advancement_counts: dict[int, dict[str, int]] = {
        t.kaggle_id: {r: 0 for r in ROUND_ORDER}
        for t in all_teams
    }

    # Pre-compute win probabilities for all matchup pairs (avoids 630k predict_proba calls)
    live_features = _build_live_feature_lookup(feature_matrix_df, all_teams, injury_adjustments)
    prob_cache = _precompute_prob_cache(all_teams, live_features, artifacts)

    logger.info(f"Starting {n_sims:,} tournament simulations...")

    for sim_idx in range(n_sims):
        if progress_callback:
            progress_callback(sim_idx, n_sims)
        elif sim_idx % 2000 == 0 and sim_idx > 0:
            logger.debug(f"  Simulation {sim_idx:,}/{n_sims:,}")

        # Deep copy bracket for this simulation
        bracket = _copy_bracket(bracket_template)

        # Simulate each round in order
        for round_name in ROUND_ORDER:
            round_games = [g for g in bracket if g.round_name == round_name]
            for game in round_games:
                if game.team_A is None or game.team_B is None:
                    continue

                # Fast dict lookup instead of predict_proba each iteration
                p_A = _cached_prob(game.team_A, game.team_B, prob_cache, live_features, artifacts)
                # Bernoulli trial
                team_A_wins = rng.random() < p_A
                winner = game.team_A if team_A_wins else game.team_B

                # Record advancement
                advancement_counts[winner.kaggle_id][round_name] += 1
                game.winner = winner

                # Advance winner to next round
                _advance_in_bracket(bracket, game, winner)

    # Aggregate results
    results = _aggregate_results(all_teams, advancement_counts, n_sims)

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "simulation_results.parquet"
    results.to_parquet(out_path, index=False)
    logger.info(f"Simulation complete. Results saved → {out_path}")

    return results


def predict_matchup_prob(
    team_A,    # Team
    team_B,    # Team
    live_features: dict,
    artifacts: dict,
) -> float:
    """
    Returns P(team_A wins) for a single matchup.
    Uses ensemble model with current-season stats.

    Args:
        team_A, team_B:  Team objects
        live_features:   dict mapping (team_id_A, team_id_B) → feature vector
        artifacts:       loaded model artifacts

    Returns float in [0, 1].
    """
    key = (team_A.kaggle_id, team_B.kaggle_id)
    reverse_key = (team_B.kaggle_id, team_A.kaggle_id)

    seed_diff = float(team_A.seed - team_B.seed)
    seed_upset = 1.0 if team_A.seed > team_B.seed else 0.0

    seed_diff_idx = FEATURE_COLS.index("seed_diff") if "seed_diff" in FEATURE_COLS else -1
    seed_flag_idx = FEATURE_COLS.index("seed_upset_flag") if "seed_upset_flag" in FEATURE_COLS else -1

    if key in live_features:
        x = live_features[key].copy().reshape(1, -1)
        if seed_diff_idx >= 0:
            x[0, seed_diff_idx] = seed_diff
            x[0, seed_flag_idx] = seed_upset
        prob = artifacts["ensemble"].predict_proba(x)[0, 1]
        return float(prob)
    elif reverse_key in live_features:
        x = live_features[reverse_key].copy().reshape(1, -1)
        if seed_diff_idx >= 0:
            x[0, seed_diff_idx] = -seed_diff
            x[0, seed_flag_idx] = 1.0 - seed_upset
        prob = artifacts["ensemble"].predict_proba(x)[0, 1]
        return float(1.0 - prob)  # flip because roles are reversed
    else:
        # Fallback: use seed-based prior
        return _seed_prior(team_A.seed, team_B.seed)


def load_simulation_results() -> pd.DataFrame:
    """Loads saved simulation results from disk."""
    path = FEATURES_DIR / "simulation_results.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Simulation results not found. Run scripts/run_simulation.py first."
        )
    return pd.read_parquet(path)


def build_upset_probability_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes P(lower seed beats higher seed) across all seed matchup combinations.
    Returns a 16x16 DataFrame where cell [i,j] = P(seed i beats seed j) in tournament.
    """
    # Compute historical rates from simulation results
    seeds = list(range(1, 17))
    matrix = pd.DataFrame(index=seeds, columns=seeds, dtype=float)

    # Historical seed win rates (approximate from 1985-2025 data)
    HISTORICAL_UPSET_RATES = {
        (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
        (5, 12): 0.65, (6, 11): 0.62, (7, 10): 0.61, (8, 9): 0.51,
    }

    for s1 in seeds:
        for s2 in seeds:
            if s1 < s2:
                rate = HISTORICAL_UPSET_RATES.get((s1, s2), 0.5 + (s2 - s1) * 0.02)
                matrix.loc[s1, s2] = round(rate, 3)
                matrix.loc[s2, s1] = round(1 - rate, 3)
            elif s1 == s2:
                matrix.loc[s1, s2] = 0.5

    return matrix


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_live_feature_lookup(
    feature_matrix_df: pd.DataFrame,
    teams: list,
    injury_adjustments: dict = None,
) -> dict:
    """
    Pre-builds a dict: (team_A_id, team_B_id) → feature vector (np.ndarray)
    for all tournament teams × teams combinations using 2026 season stats.

    If a pair has not been played yet (no game in feature matrix),
    constructs the feature vector from season-level stats in feature_matrix_df.
    """
    import itertools

    live_df = feature_matrix_df[
        feature_matrix_df["season"] == LIVE_SEASON
    ].copy()

    team_ids = [t.kaggle_id for t in teams]

    # Build from existing game rows first
    lookup = {}
    if not live_df.empty:
        for _, row in live_df.iterrows():
            a_id = int(row.get("team_A_id", 0))
            b_id = int(row.get("team_B_id", 0))
            feat_vec = np.array([
                row.get(c, 0.0) for c in FEATURE_COLS
            ], dtype="float32")
            lookup[(a_id, b_id)] = feat_vec

    # For missing pairs: construct synthetic feature vector from team stats
    # using a separate lookup of team-level stats
    team_stats = _get_team_stats_2026(feature_matrix_df, team_ids)

    # Apply injury adjustments (modify stats before building feature vectors)
    if injury_adjustments:
        name_to_id = {t.name: t.kaggle_id for t in teams}
        for team_name, adj in injury_adjustments.items():
            tid = name_to_id.get(team_name)
            if tid and tid in team_stats:
                stats = dict(team_stats[tid])
                if "AdjO_delta" in adj:
                    stats["AdjO"] = float(stats.get("AdjO") or 0) + adj["AdjO_delta"]
                if "AdjD_delta" in adj:
                    stats["AdjD"] = float(stats.get("AdjD") or 0) + adj["AdjD_delta"]
                team_stats[tid] = stats

    for a_id, b_id in itertools.combinations(team_ids, 2):
        if (a_id, b_id) not in lookup and (b_id, a_id) not in lookup:
            feat_vec = _construct_feature_vector(a_id, b_id, team_stats)
            if feat_vec is not None:
                lookup[(a_id, b_id)] = feat_vec

    return lookup


def _get_team_stats_2026(
    feature_matrix_df: pd.DataFrame,
    team_ids: list[int],
) -> dict[int, dict]:
    """
    Extracts current season team-level stats for tournament teams.
    Uses Sports-Reference advanced stats (always available) merged with
    style clusters. Falls back to empty dict on any error.
    Returns dict: team_id → stats dict.
    """
    try:
        sr_path = SPORTS_REF_DIR / f"adv_stats_{LIVE_SEASON}.parquet"
        cw_path = PROCESSED_DIR / "team_id_crosswalk.parquet"
        sc_path = FEATURES_DIR / "style_clusters.parquet"

        sr_df = pd.read_parquet(sr_path)
        cw = pd.read_parquet(cw_path)

        # SR uses "torvik_name" column for the team name (historical naming)
        sr_df = sr_df.merge(
            cw[["canonical_name", "team_id"]].rename(columns={"canonical_name": "torvik_name"}),
            on="torvik_name", how="left"
        )
        sr_df["team_id"] = sr_df["team_id"].astype("Int32")

        # Merge style clusters
        if sc_path.exists():
            sc_df = pd.read_parquet(sc_path)
            sc_2026 = sc_df[sc_df["year"] == LIVE_SEASON][["torvik_name", "style_cluster"]]
            sr_df = sr_df.merge(sc_2026, on="torvik_name", how="left")
            sr_df["style_cluster"] = sr_df["style_cluster"].fillna(0.0)

        stats = {}
        for _, row in sr_df.iterrows():
            tid = row.get("team_id")
            if pd.notna(tid):
                stats[int(tid)] = row.to_dict()

        logger.info(f"Loaded SR stats for {len(stats)} teams (season {LIVE_SEASON})")
        return stats
    except Exception as e:
        logger.warning(f"Could not load {LIVE_SEASON} SR stats: {e}")
        return {}


def _construct_feature_vector(
    a_id: int,
    b_id: int,
    team_stats: dict,
) -> np.ndarray | None:
    """
    Constructs a feature vector from season stats for a hypothetical matchup.
    Returns None if stats not available.
    """
    stats_A = team_stats.get(a_id)
    stats_B = team_stats.get(b_id)

    if stats_A is None or stats_B is None:
        return None

    def get(stats, key, default=0.0):
        v = stats.get(key, default)
        return float(v) if pd.notna(v) else default

    feat = {
        "AdjO_diff":              get(stats_A, "AdjO") - get(stats_B, "AdjD"),
        "AdjD_diff":              get(stats_A, "AdjD") - get(stats_B, "AdjO"),
        "AdjNetRtg_diff":         (get(stats_A, "AdjO") - get(stats_A, "AdjD")) -
                                  (get(stats_B, "AdjO") - get(stats_B, "AdjD")),
        "barthag_diff":           get(stats_A, "barthag") - get(stats_B, "barthag"),
        "AdjT_diff":              get(stats_A, "AdjT") - get(stats_B, "AdjT"),
        "AdjT_interaction":       get(stats_A, "AdjT") * get(stats_B, "AdjT"),
        "eFG_diff":               get(stats_A, "eFG_pct") - get(stats_B, "eFG_pct_d"),
        "eFG_d_diff":             get(stats_A, "eFG_pct_d") - get(stats_B, "eFG_pct"),
        "TO_off_diff":            get(stats_A, "TO_pct") - get(stats_B, "TO_pct"),
        "TO_forced_diff":         get(stats_A, "TO_forced_pct") - get(stats_B, "TO_forced_pct"),
        "OR_diff":                get(stats_A, "OR_pct") - get(stats_B, "DR_pct"),
        "FTR_diff":               get(stats_A, "FTR") - get(stats_B, "FTR"),
        "Blk_diff":               get(stats_A, "Blk_pct") - get(stats_B, "Blk_pct"),
        "Stl_diff":               get(stats_A, "Stl_pct") - get(stats_B, "Stl_pct"),
        "W_pct_diff":             get(stats_A, "W_pct") - get(stats_B, "W_pct"),
        "SOS_diff":               get(stats_A, "SOS") - get(stats_B, "SOS"),
        "style_cluster_A":        get(stats_A, "style_cluster", 0.0),
        "style_cluster_B":        get(stats_B, "style_cluster", 0.0),
        "style_cluster_interaction": get(stats_A, "style_cluster", 0.0) * 5 + get(stats_B, "style_cluster", 0.0),
        "rest_days_diff":         0.0,  # neutral assumption pre-tournament
        "seed_diff":              0.0,  # filled by caller using actual seeds
        "seed_upset_flag":        0.0,
    }

    return np.array([feat.get(c, 0.0) for c in FEATURE_COLS], dtype="float32")


def _precompute_prob_cache(
    teams: list,
    live_features: dict,
    artifacts: dict,
) -> dict:
    """
    Pre-computes win probabilities for all unique team pairs.
    Returns dict: (id_A, id_B) → float probability in [0,1].
    Much faster than calling predict_proba inside the 10k sim loop.
    """
    import itertools
    import warnings

    team_ids = [(t.kaggle_id, t.seed) for t in teams]
    pairs = list(itertools.combinations(team_ids, 2))

    if not pairs:
        return {}

    # Build batched feature matrix for all pairs
    rows = []
    keys = []
    fallback = []

    for (id_A, seed_A), (id_B, seed_B) in pairs:
        key = (id_A, id_B)
        rev = (id_B, id_A)
        seed_diff = float(seed_A - seed_B)
        seed_flag = 1.0 if seed_A > seed_B else 0.0

        sd_idx = FEATURE_COLS.index("seed_diff") if "seed_diff" in FEATURE_COLS else -1
        sf_idx = FEATURE_COLS.index("seed_upset_flag") if "seed_upset_flag" in FEATURE_COLS else -1

        if key in live_features:
            x = live_features[key].copy()
            if sd_idx >= 0:
                x[sd_idx] = seed_diff
                x[sf_idx] = seed_flag
            rows.append(x)
            keys.append((id_A, id_B, False))
        elif rev in live_features:
            x = live_features[rev].copy()
            if sd_idx >= 0:
                x[sd_idx] = -seed_diff
                x[sf_idx] = 1.0 - seed_flag
            rows.append(x)
            keys.append((id_A, id_B, True))
        else:
            fallback.append((id_A, id_B, seed_A, seed_B))

    prob_cache = {}

    # Batch predict
    if rows:
        X_batch = np.stack(rows, axis=0)  # shape: (N, n_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = artifacts["ensemble"].predict_proba(X_batch)[:, 1]

        for (id_A, id_B, reversed_), p in zip(keys, probs):
            if reversed_:
                prob_cache[(id_A, id_B)] = float(1.0 - p)
            else:
                prob_cache[(id_A, id_B)] = float(p)

    # Fallback: seed prior
    for id_A, id_B, seed_A, seed_B in fallback:
        prob_cache[(id_A, id_B)] = _seed_prior(seed_A, seed_B)

    return prob_cache


def _cached_prob(team_A, team_B, prob_cache: dict, live_features: dict, artifacts: dict) -> float:
    """Returns cached win probability for team_A, falling back to live inference."""
    key = (team_A.kaggle_id, team_B.kaggle_id)
    if key in prob_cache:
        return prob_cache[key]
    rev = (team_B.kaggle_id, team_A.kaggle_id)
    if rev in prob_cache:
        return 1.0 - prob_cache[rev]
    # Not in cache: seed prior
    return _seed_prior(team_A.seed, team_B.seed)


def _seed_prior(seed_A: int, seed_B: int) -> float:
    """
    Historical seed-based win probability (fallback when no stats available).
    Based on 1985-2025 tournament outcomes.
    """
    SEED_WIN_RATES = {
        1: 0.95, 2: 0.88, 3: 0.82, 4: 0.78, 5: 0.68,
        6: 0.65, 7: 0.62, 8: 0.52, 9: 0.48, 10: 0.52,
        11: 0.45, 12: 0.41, 13: 0.30, 14: 0.22, 15: 0.12, 16: 0.05,
    }
    r_A = SEED_WIN_RATES.get(seed_A, 0.5)
    r_B = SEED_WIN_RATES.get(seed_B, 0.5)
    total = r_A + r_B
    return r_A / total if total > 0 else 0.5


def _copy_bracket(template: list) -> list:
    """Deep-copies bracket for a single simulation run."""
    from src.simulation.bracket import BracketGame, Team
    new_bracket = []
    for g in template:
        new_g = BracketGame(
            game_id=g.game_id,
            round_name=g.round_name,
            region=g.region,
            team_A=g.team_A,   # Teams are immutable — shallow copy OK
            team_B=g.team_B,
            winner=None,
        )
        new_bracket.append(new_g)
    return new_bracket


def _advance_in_bracket(bracket: list, completed_game, winner) -> None:
    """Advances winner to the next round in this simulation's bracket."""
    from config import ROUND_ORDER

    round_order = ROUND_ORDER
    curr_idx = round_order.index(completed_game.round_name) if completed_game.round_name in round_order else -1
    if curr_idx < 0 or curr_idx >= len(round_order) - 1:
        return

    next_round = round_order[curr_idx + 1]
    next_games = [g for g in bracket if g.round_name == next_round]

    if not next_games:
        return

    # Regional advancement
    if next_round not in ("F4", "NCG"):
        region_games = [g for g in next_games if g.region == completed_game.region]
        for g in region_games:
            if g.team_A is None:
                g.team_A = winner
                return
            elif g.team_B is None:
                g.team_B = winner
                return

    # Final Four
    if next_round == "F4":
        region_pairings = {
            "East": "East-Midwest", "Midwest": "East-Midwest",
            "South": "South-West", "West": "South-West",
        }
        target = region_pairings.get(completed_game.region)
        for g in next_games:
            if g.region == target:
                if g.team_A is None:
                    g.team_A = winner
                    return
                elif g.team_B is None:
                    g.team_B = winner
                    return

    # NCG
    for g in next_games:
        if g.team_A is None:
            g.team_A = winner
            return
        elif g.team_B is None:
            g.team_B = winner
            return


def _aggregate_results(
    teams: list,
    advancement_counts: dict,
    n_sims: int,
) -> pd.DataFrame:
    """Converts simulation counts to probability DataFrame."""
    rows = []
    for team in teams:
        counts = advancement_counts.get(team.kaggle_id, {})
        p = {r: counts.get(r, 0) / n_sims for r in ROUND_ORDER if r != "FF"}

        expected_wins = sum(p.values())
        expected_pts = sum(
            p.get(r, 0) * ROUND_POINTS.get(r, 0)
            for r in ROUND_POINTS
        )

        rows.append({
            "kaggle_id": team.kaggle_id,
            "name": team.name,
            "seed": team.seed,
            "region": team.region,
            "torvik_rank": team.torvik_rank,
            "adj_net_rtg": team.adj_net_rtg,
            "p_R64": round(p.get("R64", 0), 4),
            "p_R32": round(p.get("R32", 0), 4),
            "p_S16": round(p.get("S16", 0), 4),
            "p_E8": round(p.get("E8", 0), 4),
            "p_F4": round(p.get("F4", 0), 4),
            "p_NCG": round(p.get("NCG", 0), 4),
            "p_Champion": round(p.get("NCG", 0), 4),  # alias
            "expected_wins": round(expected_wins, 3),
            "expected_bracket_pts": round(expected_pts, 2),
        })

    df = pd.DataFrame(rows).sort_values("p_Champion", ascending=False).reset_index(drop=True)
    return df
