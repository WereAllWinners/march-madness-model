"""
Tournament simulation CLI script.

Usage:
    python scripts/run_simulation.py                          # run 10,000 simulations
    python scripts/run_simulation.py --n-sims 25000           # custom count
    python scripts/run_simulation.py --actual-seeds           # use actual seeds (post-Selection Sunday)
    python scripts/run_simulation.py --print-results          # show top teams in console
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


def main():
    parser = argparse.ArgumentParser(description="Run March Madness tournament simulation")
    parser.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000)"
    )
    parser.add_argument(
        "--actual-seeds",
        action="store_true",
        help="Use actual 2026 seeds (post-Selection Sunday). Requires data/raw/manual/seeds_2026.csv"
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print top-20 teams to console after simulation"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Tournament year (default: 2026)"
    )
    args = parser.parse_args()

    logger.info("─" * 60)
    logger.info(f"March Madness {args.year} Tournament Simulation")
    logger.info(f"Simulations: {args.n_sims:,}")
    logger.info("─" * 60)

    # ── Load artifacts ─────────────────────────────────────────────────────────
    from src.models.train import load_artifacts
    from src.features.engineer import load_feature_matrix
    from src.ingestion.torvik import load_trank_season
    from src.simulation.bracket import build_bracket_2026
    from src.simulation.monte_carlo import run_simulation

    logger.info("Loading model artifacts...")
    artifacts = load_artifacts()

    logger.info("Loading feature matrix...")
    fm = load_feature_matrix()

    logger.info(f"Loading {args.year} Torvik data...")
    try:
        torvik = load_trank_season(args.year)
    except FileNotFoundError:
        logger.warning(
            f"Torvik data for {args.year} not found. "
            "Run: python scripts/fetch_data.py --years {args.year} --force-refresh"
        )
        torvik = None

    # ── Build bracket ──────────────────────────────────────────────────────────
    logger.info("Building bracket...")
    if args.actual_seeds:
        # Check for manual seeds file
        seeds_path = Path("data/raw/manual/seeds_2026.csv")
        if seeds_path.exists():
            import pandas as pd
            seeds_df = pd.read_csv(seeds_path)
            logger.info(f"Using actual seeds from {seeds_path}")
        else:
            logger.warning(
                f"--actual-seeds specified but {seeds_path} not found. "
                "Place seeds_2026.csv in data/raw/manual/ after Selection Sunday."
            )
            seeds_df = None
        bracket_games, all_teams = build_bracket_2026(
            seeds_df=seeds_df, torvik_df=torvik
        )
    else:
        bracket_games, all_teams = build_bracket_2026(
            torvik_df=torvik, use_projected=True
        )

    if not all_teams:
        logger.error(
            "Could not build bracket. Make sure Torvik data is available:\n"
            f"  python scripts/fetch_data.py --years {args.year}"
        )
        sys.exit(1)

    logger.info(f"Bracket: {len(all_teams)} teams loaded")

    # ── Run simulation ─────────────────────────────────────────────────────────
    logger.info(f"\nRunning {args.n_sims:,} simulations...")

    results = run_simulation(
        all_teams=all_teams,
        bracket_template=bracket_games,
        feature_matrix_df=fm,
        artifacts=artifacts,
        n_sims=args.n_sims,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    if args.print_results or True:  # Always print brief summary
        logger.info("\n" + "="*60)
        logger.info("CHAMPIONSHIP PROBABILITY — TOP 20")
        logger.info("="*60)
        logger.info(f"{'Rank':<5} {'Team':<25} {'Seed':<5} {'Region':<10} {'P(Champ)':>10}")
        logger.info("-"*60)
        for i, (_, row) in enumerate(results.head(20).iterrows()):
            logger.info(
                f"{i+1:<5} {row['name']:<25} {int(row['seed']):<5} "
                f"{row['region']:<10} {row['p_Champion']:>10.1%}"
            )
        logger.info("="*60)

    logger.info(f"\n✅ Simulation complete! Results saved to data/features/simulation_results.parquet")
    logger.info("Launch dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
