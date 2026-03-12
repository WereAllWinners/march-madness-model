"""
Trains all models and the stacking ensemble.
Pipeline:
  1. Load feature_matrix.parquet
  2. Fit StandardScaler on TRAIN only
  3. Train LogReg, XGBoost, LightGBM on TRAIN
  4. Calibrate each with isotonic regression on VAL
  5. Build stacking ensemble (LogReg meta-learner on OOF probs)
  6. Save all artifacts to ARTIFACTS_DIR

For final tournament predictions: retrain on TRAIN+VAL+TEST (2010-2025).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ARTIFACTS_DIR, FEATURE_COLS, TARGET_COL,
    LOGREG_PARAMS, XGB_PARAMS, LGBM_PARAMS,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
)

logger = logging.getLogger(__name__)


def train_all_models(
    feature_matrix_path: Path = None,
    retrain_on_all: bool = False,
    tournament_only: bool = False,
) -> dict:
    """
    Full training pipeline.

    Args:
        feature_matrix_path: path to feature_matrix.parquet (default auto-detect)
        retrain_on_all:       if True, retrain final models on TRAIN+VAL+TEST
                              (use this before tournament simulation)
        tournament_only:      if True, train only on NCAA tournament games.
                              Produces better-calibrated tournament win probabilities
                              since regular season games overwhelm the tournament signal.

    Returns dict: {"logreg", "xgb", "lgbm", "ensemble", "scaler"}
    """
    from src.features.engineer import load_feature_matrix, get_X_y

    logger.info("Loading feature matrix...")
    fm = load_feature_matrix()

    if tournament_only and "is_tournament" in fm.columns:
        logger.info("Filtering to tournament games only (is_tournament==1)...")
        fm = fm[fm["is_tournament"] == 1].copy()
        logger.info(f"  Tournament games: {len(fm)}")

    X_train, y_train, feat_names = get_X_y(fm, splits=["train"])
    X_val, y_val, _             = get_X_y(fm, splits=["val"])
    X_test, y_test, _           = get_X_y(fm, splits=["test"])

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(f"Features: {len(feat_names)}")

    # Fit scaler on TRAIN only
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Train base models
    logreg = _train_logreg(X_train_s, y_train)
    xgb    = _train_xgb(X_train, y_train, X_val, y_val)  # tree models: use unscaled
    lgbm   = _train_lgbm(X_train, y_train, X_val, y_val)

    # Calibrate on VAL set
    logger.info("Calibrating models on validation set...")
    logreg_cal = _calibrate(logreg, X_val_s,  y_val, use_scaled=True,  scaler=None)
    xgb_cal    = _calibrate(xgb,    X_val,    y_val, use_scaled=False, scaler=None)
    lgbm_cal   = _calibrate(lgbm,   X_val,    y_val, use_scaled=False, scaler=None)

    # Stacking ensemble
    logger.info("Building stacking ensemble...")
    ensemble = _build_ensemble(
        [logreg_cal, xgb_cal, lgbm_cal],
        ["logreg", "xgb", "lgbm"],
        X_train_s, X_train, y_train,
        scaler,
    )

    # Optionally retrain on all data for final predictions
    if retrain_on_all:
        logger.info("Retraining on TRAIN+VAL+TEST for final tournament predictions...")
        X_all, y_all, _ = get_X_y(fm, splits=["train", "val", "test"])
        X_all_s = scaler.transform(X_all)
        logreg_cal  = _train_logreg(X_all_s, y_all)
        xgb_cal     = _train_xgb(X_all, y_all)
        lgbm_cal    = _train_lgbm(X_all, y_all)
        ensemble    = _build_ensemble(
            [logreg_cal, xgb_cal, lgbm_cal],
            ["logreg", "xgb", "lgbm"],
            X_all_s, X_all, y_all,
            scaler,
        )

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "logreg": logreg_cal,
        "xgb": xgb_cal,
        "lgbm": lgbm_cal,
        "ensemble": ensemble,
        "scaler": scaler,
        "feature_names": feat_names,
    }
    for name, obj in artifacts.items():
        path = ARTIFACTS_DIR / f"{name}.pkl"
        joblib.dump(obj, path)
        logger.info(f"  Saved {name}.pkl")

    logger.info("All models trained and saved.")
    return artifacts


def load_artifacts() -> dict:
    """Loads all saved model artifacts from ARTIFACTS_DIR."""
    artifacts = {}
    for name in ["logreg", "xgb", "lgbm", "ensemble", "scaler", "feature_names"]:
        path = ARTIFACTS_DIR / f"{name}.pkl"
        if path.exists():
            artifacts[name] = joblib.load(path)
        else:
            logger.warning(f"Artifact not found: {path}")
    if not artifacts:
        raise FileNotFoundError(
            f"No model artifacts found in {ARTIFACTS_DIR}. Run scripts/train_models.py first."
        )
    return artifacts


def predict_proba(
    X: np.ndarray,
    model_name: str = "ensemble",
    artifacts: dict = None,
) -> np.ndarray:
    """
    Returns win probability for team_A for each row in X.
    Applies scaler transform if model requires scaled features.

    Args:
        X:           raw feature array (not scaled)
        model_name:  "logreg" | "xgb" | "lgbm" | "ensemble"
        artifacts:   loaded artifacts dict (loads from disk if None)

    Returns 1D array of probabilities in [0, 1].
    """
    if artifacts is None:
        artifacts = load_artifacts()

    model = artifacts[model_name]
    scaler = artifacts.get("scaler")

    if model_name == "logreg" and scaler:
        X_in = scaler.transform(X)
    else:
        X_in = X

    return model.predict_proba(X_in)[:, 1]


# ── Private helpers ────────────────────────────────────────────────────────────

def _train_logreg(X: np.ndarray, y: np.ndarray):
    logger.info(f"Training LogisticRegression...")
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X, y)
    logger.info("  LogReg done")
    return model


def _train_xgb(X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost required. Run: pip install xgboost")

    logger.info("Training XGBClassifier...")
    params = dict(XGB_PARAMS)
    model = XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X, y)

    logger.info("  XGB done")
    return model


def _train_lgbm(X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("lightgbm required. Run: pip install lightgbm")

    logger.info("Training LGBMClassifier...")
    model = LGBMClassifier(**LGBM_PARAMS)

    if X_val is not None and y_val is not None:
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )
    else:
        model.fit(X, y)

    logger.info("  LGBM done")
    return model


def _calibrate(model, X_val: np.ndarray, y_val: np.ndarray, use_scaled: bool, scaler):
    """Calibrates model using isotonic regression on the validation set."""
    cal = CalibratedClassifierCV(
        estimator=model,
        method="isotonic",
        cv="prefit",  # model is already fitted — calibrate on new data
    )
    cal.fit(X_val, y_val)
    return cal


def _build_ensemble(
    base_models: list,
    base_names: list[str],
    X_train_scaled: np.ndarray,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    n_folds: int = 5,
) -> object:
    """
    Builds stacking ensemble via OOF predictions.
    Meta-learner: LogisticRegression on [logreg_oof, xgb_oof, lgbm_oof].
    """
    n = len(y_train)
    oof_probs = np.zeros((n, len(base_models)), dtype="float32")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (idx_tr, idx_val) in enumerate(skf.split(X_train_raw, y_train)):
        logger.info(f"  Ensemble fold {fold_idx + 1}/{n_folds}...")

        for i, (model, name) in enumerate(zip(base_models, base_names)):
            if name == "logreg":
                X_f_tr = X_train_scaled[idx_tr]
                X_f_val = X_train_scaled[idx_val]
            else:
                X_f_tr = X_train_raw[idx_tr]
                X_f_val = X_train_raw[idx_val]

            y_f_tr = y_train[idx_tr]

            # Clone and refit for OOF
            from sklearn.base import clone
            m_clone = clone(model.estimator if hasattr(model, "estimator") else model)
            m_clone.fit(X_f_tr, y_f_tr)
            oof_probs[idx_val, i] = m_clone.predict_proba(X_f_val)[:, 1]

    # Fit meta-learner on OOF
    meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta.fit(oof_probs, y_train)

    ensemble = StackingEnsemble(base_models, base_names, meta, scaler)
    return ensemble


class StackingEnsemble:
    """Stacking ensemble: LogReg meta-learner on top of base model OOF probs."""

    def __init__(self, base_models, base_names, meta, scaler):
        self.base_models = base_models
        self.base_names  = base_names
        self.meta        = meta
        self.scaler      = scaler

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_raw) if self.scaler else X_raw
        probs = np.column_stack([
            m.predict_proba(X_scaled if n == "logreg" else X_raw)[:, 1]
            for m, n in zip(self.base_models, self.base_names)
        ])
        p1 = self.meta.predict_proba(probs)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_raw)[:, 1] >= 0.5).astype(int)
