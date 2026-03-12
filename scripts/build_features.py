"""
Feature building CLI script.

Usage:
    python scripts/build_features.py                         # run all steps
    python scripts/build_features.py --steps crosswalk       # just crosswalk
    python scripts/build_features.py --steps games,rolling   # specific steps
    python scripts/build_features.py --steps features        # features only
    python scripts/build_features.py --seasons 2026          # live season update

Available steps: crosswalk, games, rolling, clusters, features
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_STEPS = ["crosswalk", "games", "rolling", "clusters", "features"]


def main():
    parser = argparse.ArgumentParser(description="Build feature engineering pipeline")
    parser.add_argument(
        "--steps",
        type=str,
        default=",".join(ALL_STEPS),
        help=f"Comma-separated steps: {','.join(ALL_STEPS)}"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Filter to specific season(s) for 'features' step (e.g. '2026' or '2023-2026')"
    )
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",")]
    logger.info(f"Steps to run: {steps}")

    # ── Step 1: Crosswalk ─────────────────────────────────────────────────────
    if "crosswalk" in steps:
        logger.info("\n── Building Team ID Crosswalk ──────────────────────────")
        try:
            from src.ingestion.torvik import load_all_seasons
            from src.ingestion.tourney_scraper import fetch_all_tournament_results
            from src.processing.crosswalk import build_crosswalk
            from config import ALL_SEASONS, LIVE_SEASON

            training_seasons = [s for s in ALL_SEASONS if s != LIVE_SEASON and s != 2020]
            stats_dfs = [load_all_seasons(training_seasons)]
            bracket_df = fetch_all_tournament_results(training_seasons)
            build_crosswalk(stats_dfs=stats_dfs, bracket_dfs=[bracket_df])
            logger.info("✅ Crosswalk built")
        except Exception as e:
            logger.error(f"Crosswalk failed: {e}")
            raise

    # ── Step 2: Game-level dataset ────────────────────────────────────────────
    if "games" in steps:
        logger.info("\n── Building Game-Level Dataset ─────────────────────────")
        try:
            from config import ALL_SEASONS
            from src.processing.game_builder import build_game_level_dataset

            seasons = ALL_SEASONS
            if args.seasons:
                from scripts.fetch_data import parse_years
                seasons = parse_years(args.seasons)

            build_game_level_dataset(seasons=seasons)
            logger.info("✅ Game-level dataset built")
        except Exception as e:
            logger.error(f"Game-level build failed: {e}")
            raise

    # ── Step 3: Rolling stats ─────────────────────────────────────────────────
    if "rolling" in steps:
        logger.info("\n── Computing Rolling Stats ──────────────────────────────")
        try:
            from src.processing.rolling_stats import add_rolling_stats_to_game_level
            add_rolling_stats_to_game_level()
            logger.info("✅ Rolling stats computed")
        except Exception as e:
            logger.error(f"Rolling stats failed: {e}")
            raise

    # ── Step 4: Style clusters ────────────────────────────────────────────────
    if "clusters" in steps:
        logger.info("\n── Fitting Style Clusters ───────────────────────────────")
        try:
            from src.ingestion.torvik import load_all_seasons
            from src.features.clustering import fit_style_clusters, assign_clusters

            torvik_df = load_all_seasons()
            kmeans, scaler = fit_style_clusters(torvik_df)
            assign_clusters(torvik_df, kmeans, scaler)
            logger.info("✅ Style clusters computed")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

    # ── Step 5: Feature matrix ────────────────────────────────────────────────
    if "features" in steps:
        logger.info("\n── Building Feature Matrix ──────────────────────────────")
        try:
            from src.features.engineer import build_feature_matrix
            build_feature_matrix()
            logger.info("✅ Feature matrix built")
        except Exception as e:
            logger.error(f"Feature matrix failed: {e}")
            raise

    logger.info("\n✅ All steps complete!")
    logger.info("Next step: python scripts/train_models.py")


if __name__ == "__main__":
    main()
