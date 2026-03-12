"""
SHAP explanations for all model types.
  - XGB / LGBM:  shap.TreeExplainer  (fast, exact)
  - LogReg:      shap.LinearExplainer
  - Ensemble:    uses XGB sub-model's SHAP values (most informative)

Pre-computes global SHAP values on val+test sets and caches to FEATURES_DIR.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURES_DIR, FEATURE_COLS

logger = logging.getLogger(__name__)


def get_explainer(model, model_name: str, X_background: np.ndarray):
    """
    Returns the appropriate SHAP Explainer for a given model type.

    Args:
        model:        fitted sklearn-compatible model
        model_name:   "logreg" | "xgb" | "lgbm" | "ensemble"
        X_background: background dataset for kernel/linear explainer (not needed for tree)

    Returns shap.Explainer object.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap required. Run: pip install shap")

    if model_name in ("xgb", "lgbm"):
        # Unwrap CalibratedClassifierCV if needed
        base = model
        if hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            base = model.estimator
        return shap.TreeExplainer(base)

    elif model_name == "logreg":
        base = model
        if hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            base = model.estimator
        return shap.LinearExplainer(base, X_background)

    elif model_name == "ensemble":
        # Use the XGB sub-model for SHAP in the ensemble
        for sub_model, sub_name in zip(model.base_models, model.base_names):
            if sub_name == "xgb":
                return get_explainer(sub_model, "xgb", X_background)
        # Fallback: use first available tree model
        for sub_model, sub_name in zip(model.base_models, model.base_names):
            if sub_name in ("xgb", "lgbm"):
                return get_explainer(sub_model, sub_name, X_background)

    raise ValueError(f"Unknown model_name: {model_name}")


def compute_shap_values(
    explainer,
    X: np.ndarray,
    feature_names: list[str],
) -> "shap.Explanation":
    """Computes SHAP values for X. Returns shap.Explanation object."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap required. Run: pip install shap")

    shap_vals = explainer(X)
    return shap_vals


def shap_for_matchup(
    X_row: np.ndarray,
    explainer,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Computes SHAP values for a single game (matchup).

    Args:
        X_row:         1D feature array (one game)
        explainer:     fitted SHAP explainer
        feature_names: list of feature names (aligns with X_row)

    Returns DataFrame:
        feature      str
        shap_value   float32  (positive = increases P(team_A wins))
        feature_value float32
    Sorted by abs(shap_value) descending.
    """
    X_2d = X_row.reshape(1, -1)
    sv = explainer(X_2d)

    # Handle different SHAP output formats
    if hasattr(sv, "values"):
        vals = sv.values[0]
        if vals.ndim > 1:
            vals = vals[:, 1]  # class 1 (team_A wins)
    else:
        vals = np.array(sv[0])

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": vals.astype("float32"),
        "feature_value": X_row.astype("float32"),
    })
    df = df.reindex(df["shap_value"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)


def global_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Computes mean absolute SHAP per feature.

    Args:
        shap_values:   2D array [n_samples, n_features]
        feature_names: list of feature names

    Returns DataFrame:
        feature          str
        mean_abs_shap    float32
    Sorted descending.
    """
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # class 1

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs.astype("float32"),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def compute_and_cache_global_shap(
    artifacts: dict,
    feature_matrix_df: pd.DataFrame,
    splits: list[str] = None,
    max_samples: int = 5000,
) -> dict:
    """
    Pre-computes global SHAP values for specified splits and saves to disk.
    Returns dict: model_name → global SHAP summary DataFrame.

    Args:
        artifacts:          loaded model artifacts
        feature_matrix_df:  full feature matrix
        splits:             which splits to use (default ["val", "test"])
        max_samples:        subsample to avoid OOM for large datasets
    """
    from src.features.engineer import get_X_y

    if splits is None:
        splits = ["val", "test"]

    X, _, feat_names = get_X_y(feature_matrix_df, splits=splits)
    scaler = artifacts.get("scaler")

    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]

    summaries = {}
    for model_name in ["xgb", "lgbm"]:
        model = artifacts.get(model_name)
        if model is None:
            continue
        logger.info(f"Computing global SHAP for {model_name} ({len(X)} samples)...")
        try:
            expl = get_explainer(model, model_name, X[:100])
            sv_obj = compute_shap_values(expl, X, feat_names)

            if hasattr(sv_obj, "values"):
                sv = sv_obj.values
            else:
                sv = np.array(sv_obj)

            if sv.ndim == 3:
                sv = sv[:, :, 1]

            summary = global_shap_summary(sv, feat_names)
            summaries[model_name] = summary

            # Cache to disk
            out_path = FEATURES_DIR / f"shap_global_{model_name}.parquet"
            summary.to_parquet(out_path, index=False)
            logger.info(f"  Saved global SHAP → {out_path}")

        except Exception as e:
            logger.warning(f"  SHAP failed for {model_name}: {e}")

    return summaries


def load_global_shap(model_name: str = "xgb") -> pd.DataFrame:
    """Loads cached global SHAP summary for a given model."""
    path = FEATURES_DIR / f"shap_global_{model_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Global SHAP not found for {model_name}. Run scripts/train_models.py first."
        )
    return pd.read_parquet(path)
