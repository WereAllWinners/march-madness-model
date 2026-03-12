"""
Model evaluation: AUC, log loss, Brier score, accuracy, calibration.
Produces tidy DataFrame of metrics per model × split.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURE_COLS

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    split_name: str,
    scaler=None,
) -> dict:
    """
    Computes evaluation metrics for a single model on a single split.

    Returns dict:
        model_name, split, auc, log_loss, brier, accuracy,
        cal_fraction_pos (list), cal_mean_pred (list)
    """
    # Apply scaler if needed (LogReg)
    if model_name == "logreg" and scaler is not None:
        X_in = scaler.transform(X)
    else:
        X_in = X

    try:
        probs = model.predict_proba(X_in)[:, 1]
    except Exception as e:
        logger.warning(f"predict_proba failed for {model_name}: {e}")
        return {}

    preds = (probs >= 0.5).astype(int)

    # Calibration curve (10 bins)
    try:
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="uniform")
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])

    return {
        "model_name":       model_name,
        "split":            split_name,
        "auc":              round(float(roc_auc_score(y, probs)), 4),
        "log_loss":         round(float(log_loss(y, probs)), 4),
        "brier":            round(float(brier_score_loss(y, probs)), 4),
        "accuracy":         round(float(accuracy_score(y, preds)), 4),
        "cal_fraction_pos": frac_pos.tolist(),
        "cal_mean_pred":    mean_pred.tolist(),
        "n_games":          len(y),
    }


def evaluate_all(
    feature_matrix_df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    """
    Evaluates all models on val and test splits.

    Returns tidy DataFrame with one row per (model, split).
    """
    from src.features.engineer import get_X_y

    model_names = ["logreg", "xgb", "lgbm", "ensemble"]
    scaler = artifacts.get("scaler")
    rows = []

    for split in ["val", "test"]:
        X, y, _ = get_X_y(feature_matrix_df, splits=[split])
        if len(y) == 0:
            logger.warning(f"No {split} data found")
            continue

        for name in model_names:
            model = artifacts.get(name)
            if model is None:
                continue

            # Ensemble uses its own scaler internally
            X_in = X
            s = scaler if name != "ensemble" else None

            row = evaluate_model(model, X_in, y, name, split, scaler=s)
            if row:
                rows.append(row)
                logger.info(
                    f"  {name:10s} [{split:5s}] "
                    f"AUC={row['auc']:.3f}  "
                    f"Brier={row['brier']:.3f}  "
                    f"Acc={row['accuracy']:.3f}"
                )

    return pd.DataFrame(rows)


def print_evaluation_report(metrics_df: pd.DataFrame) -> None:
    """Pretty-prints a summary of the evaluation metrics."""
    print("\n" + "="*70)
    print("MODEL EVALUATION REPORT")
    print("="*70)

    for split in ["val", "test"]:
        split_df = metrics_df[metrics_df["split"] == split]
        if split_df.empty:
            continue
        print(f"\n  {split.upper()} SET:")
        print(f"  {'Model':<15} {'AUC':>7} {'LogLoss':>10} {'Brier':>8} {'Accuracy':>10} {'N':>7}")
        print(f"  {'-'*60}")
        for _, row in split_df.iterrows():
            print(
                f"  {row['model_name']:<15} "
                f"{row['auc']:>7.4f} "
                f"{row['log_loss']:>10.4f} "
                f"{row['brier']:>8.4f} "
                f"{row['accuracy']:>10.4f} "
                f"{int(row['n_games']):>7,}"
            )

    print("="*70 + "\n")
