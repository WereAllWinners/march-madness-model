"""
Data fetching CLI script.

Usage:
    python scripts/fetch_data.py                        # fetch all seasons (2010-2026)
    python scripts/fetch_data.py --years 2024-2026      # fetch specific range
    python scripts/fetch_data.py --years 2026           # fetch single year
    python scripts/fetch_data.py --validate-only        # just check Kaggle files
    python scripts/fetch_data.py --years 2026 --force-refresh  # re-fetch cached
    python scripts/fetch_data.py --skip-sportsref       # Torvik only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from config import ALL_SEASONS, FIRST_SEASON, CURRENT_SEASON

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_years(years_str: str) -> list[int]:
    """Parse year spec: '2024-2026' → [2024, 2025, 2026] or '2026' → [2026]"""
    if "-" in years_str:
        parts = years_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(years_str)]


def main():
    parser = argparse.ArgumentParser(description="Fetch NCAA basketball data")
    parser.add_argument(
        "--years",
        type=str,
        default=f"{FIRST_SEASON}-{CURRENT_SEASON}",
        help="Year(s) to fetch: '2026' or '2010-2026'"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check Kaggle files, don't fetch"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch even if cached"
    )
    parser.add_argument(
        "--skip-sportsref",
        action="store_true",
        help="Skip Sports-Reference fetching"
    )
    parser.add_argument(
        "--torvik-only",
        action="store_true",
        help="Only fetch Torvik data"
    )
    parser.add_argument(
        "--schedules",
        action="store_true",
        help="Also fetch regular season schedules for all tournament teams (~55 min, cached after first run)"
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    logger.info(f"Target years: {years[0]}–{years[-1]} ({len(years)} seasons)")

    if args.validate_only:
        # Show what we have on disk
        from pathlib import Path
        sr_dir = Path("data/raw/sports_ref")
        adv_files = sorted(sr_dir.glob("adv_stats_*.parquet"))
        tourney_files = sorted(sr_dir.glob("tourney_*.parquet"))
        logger.info(f"  Advanced stats files: {len(adv_files)}")
        logger.info(f"  Tournament result files: {len(tourney_files)}")
        logger.info("Validation complete.")
        return

    # ── Step 1: Fetch SR advanced stats (AdjO/AdjD/etc.) ─────────────────────
    logger.info(f"\n── SR Advanced Stats ({len(years)} seasons) ─────────────────")
    from src.ingestion.torvik import fetch_all_seasons as fetch_torvik
    fetch_torvik(years=years, force_refresh=args.force_refresh)

    # ── Step 2: Fetch SR tournament bracket results ───────────────────────────
    if not args.skip_sportsref and not args.torvik_only:
        logger.info(f"\n── SR Tournament Results ({len(years)} seasons) ─────────")
        logger.info("  (4 second delay between requests — please be patient)")
        from src.ingestion.tourney_scraper import fetch_tournament_results
        import time
        tourney_years = [y for y in years if y != 2020 and y != 2026]
        for i, year in enumerate(tourney_years):
            fetch_tournament_results(year, force_refresh=args.force_refresh)
            if i < len(tourney_years) - 1:
                time.sleep(4)

    # ── Step 3: Fetch regular season schedules (optional, ~55 min) ───────────
    if args.schedules and not args.torvik_only:
        sched_years = [y for y in years if y not in [2020, 2026]]
        n_teams_est = len(sched_years) * 64
        logger.info(f"\n── Regular Season Schedules ─────────────────────────────")
        logger.info(f"  ~{n_teams_est} team schedules × 4s = ~{n_teams_est*4//60} minutes")
        logger.info("  (each team-year cached; safe to interrupt and resume)")
        from src.ingestion.schedule_scraper import fetch_all_schedules
        fetch_all_schedules(sched_years, force_refresh=args.force_refresh)

    logger.info("\n✅ Data fetch complete!")
    logger.info("Next step: python scripts/build_features.py")


if __name__ == "__main__":
    main()
