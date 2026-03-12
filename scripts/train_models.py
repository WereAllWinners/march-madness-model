"""
Model training CLI script.

Usage:
    python scripts/train_models.py                     # train on 2010-2022
    python scripts/train_models.py --final             # retrain on all data (pre-tournament)
    python scripts/train_models.py --evaluate          # evaluate after training
    python scripts/train_models.py --shap              # compute global SHAP after training
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
    parser = argparse.ArgumentParser(description="Train March Madness models")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Retrain on TRAIN+VAL+TEST (2010-2025) for final tournament predictions"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Compute and cache global SHAP values after training"
    )
    parser.add_argument(
        "--tournament-only",
        action="store_true",
        help=(
            "Train only on NCAA tournament games (755 train / 126 val / 63 test). "
            "Produces tournament-calibrated win probabilities — use this for simulation."
        )
    )
    args = parser.parse_args()

    logger.info("─" * 60)
    logger.info("March Madness Model Training")
    logger.info("─" * 60)

    if args.final:
        logger.info("Mode: FINAL (retrain on all historical data for tournament)")
    elif args.tournament_only:
        logger.info("Mode: TOURNAMENT-ONLY (train on tournament games → better sim calibration)")
    else:
        logger.info("Mode: DEVELOPMENT (train on 2010-2022, evaluate on 2023-2025)")

    # ── Train models ──────────────────────────────────────────────────────────
    from src.models.train import train_all_models
    artifacts = train_all_models(
        retrain_on_all=args.final,
        tournament_only=args.tournament_only,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.evaluate or not args.final:
        logger.info("\n── Model Evaluation ────────────────────────────────────")
        from src.features.engineer import load_feature_matrix
        from src.models.evaluate import evaluate_all, print_evaluation_report

        fm = load_feature_matrix()
        metrics_df = evaluate_all(fm, artifacts)
        print_evaluation_report(metrics_df)

        # Save metrics
        from config import FEATURES_DIR
        metrics_df.to_parquet(FEATURES_DIR / "model_metrics.parquet", index=False)
        logger.info(f"Metrics saved → data/features/model_metrics.parquet")

    # ── SHAP values ───────────────────────────────────────────────────────────
    if args.shap:
        logger.info("\n── Computing Global SHAP Values ────────────────────────")
        from src.features.engineer import load_feature_matrix
        from src.models.shap_explainer import compute_and_cache_global_shap

        fm = load_feature_matrix()
        try:
            summaries = compute_and_cache_global_shap(artifacts, fm)
            logger.info(f"SHAP computed for: {list(summaries.keys())}")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

    logger.info("\n✅ Training complete!")
    if not args.final:
        logger.info(
            "Tip: Run `python scripts/train_models.py --final` before the tournament "
            "to retrain on all 2010-2025 data."
        )
    logger.info("Next step: python scripts/run_simulation.py")


if __name__ == "__main__":
    main()
