"""
K-Means style clustering pipeline.
Identifies 5 playing-style archetypes based on efficiency and style metrics.

Approximate cluster meanings (will vary by run):
  0 — Slow, grind-it-out defensive teams
  1 — Fast-paced, high-scoring offensive teams
  2 — Efficient mid-majors
  3 — Transition-heavy, live-ball teams
  4 — Methodical, free-throw-heavy teams

Clusters are fit on TRAIN seasons only, then assigned to all seasons.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CLUSTER_FEATURES, N_STYLE_CLUSTERS, RANDOM_SEED,
    TRAIN_SEASONS, FEATURES_DIR, ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)


def fit_style_clusters(
    torvik_df: pd.DataFrame,
    n_clusters: int = N_STYLE_CLUSTERS,
    random_state: int = RANDOM_SEED,
) -> tuple[KMeans, StandardScaler]:
    """
    Fits K-Means on CLUSTER_FEATURES from training seasons only.
    Saves kmeans.pkl and cluster_scaler.pkl to ARTIFACTS_DIR.

    Args:
        torvik_df:    DataFrame with all seasons, must contain CLUSTER_FEATURES + 'year'
        n_clusters:   number of style clusters
        random_state: for reproducibility

    Returns (fitted_kmeans, fitted_scaler)
    """
    # Filter to training seasons only to prevent data leakage
    train_df = torvik_df[torvik_df["year"].isin(TRAIN_SEASONS)].copy()

    avail_cols = [c for c in CLUSTER_FEATURES if c in train_df.columns]
    if len(avail_cols) < 3:
        raise ValueError(
            f"Too few cluster features available: {avail_cols}. "
            f"Need at least 3 of {CLUSTER_FEATURES}."
        )

    X = train_df[avail_cols].dropna()
    logger.info(f"Fitting K-Means (k={n_clusters}) on {len(X)} team-seasons, features: {avail_cols}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=500,
    )
    kmeans.fit(X_scaled)

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, ARTIFACTS_DIR / "kmeans.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "cluster_scaler.pkl")
    joblib.dump(avail_cols, ARTIFACTS_DIR / "cluster_feature_cols.pkl")

    inertia = kmeans.inertia_
    logger.info(f"K-Means fitted. Inertia: {inertia:.1f}. Saved to {ARTIFACTS_DIR}")
    return kmeans, scaler


def assign_clusters(
    torvik_df: pd.DataFrame,
    kmeans: KMeans = None,
    scaler: StandardScaler = None,
) -> pd.DataFrame:
    """
    Assigns cluster labels to all team-season rows.
    Loads saved artifacts if kmeans/scaler not provided.

    Returns torvik_df with added 'style_cluster' column (int8).
    Also saves to FEATURES_DIR/style_clusters.parquet.
    """
    if kmeans is None or scaler is None:
        kmeans, scaler, avail_cols = _load_cluster_artifacts()
    else:
        avail_cols_path = ARTIFACTS_DIR / "cluster_feature_cols.pkl"
        avail_cols = joblib.load(avail_cols_path) if avail_cols_path.exists() else CLUSTER_FEATURES

    avail_cols = [c for c in avail_cols if c in torvik_df.columns]
    df = torvik_df.copy()

    # Fill NaN with column medians for prediction (don't drop rows)
    X = df[avail_cols].copy()
    for col in avail_cols:
        X[col] = X[col].fillna(X[col].median())

    X_scaled = scaler.transform(X)
    df["style_cluster"] = kmeans.predict(X_scaled).astype("int8")

    # Save cluster assignments
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cluster_cols = ["torvik_name", "year", "style_cluster"]
    if "kaggle_id" in df.columns:
        cluster_cols = ["kaggle_id"] + cluster_cols
    save_cols = [c for c in cluster_cols if c in df.columns]

    out = df[save_cols].copy()
    out.to_parquet(FEATURES_DIR / "style_clusters.parquet", index=False)
    logger.info(f"Cluster assignments saved → {FEATURES_DIR}/style_clusters.parquet")

    # Log cluster distribution
    dist = df["style_cluster"].value_counts().sort_index()
    logger.info(f"Cluster distribution:\n{dist.to_string()}")

    return df


def load_cluster_assignments() -> pd.DataFrame:
    """Loads saved cluster assignments."""
    path = FEATURES_DIR / "style_clusters.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Style clusters not found. Run scripts/build_features.py --steps clusters"
        )
    return pd.read_parquet(path)


def _load_cluster_artifacts():
    """Loads saved kmeans + scaler + feature cols from ARTIFACTS_DIR."""
    kmeans_path = ARTIFACTS_DIR / "kmeans.pkl"
    scaler_path = ARTIFACTS_DIR / "cluster_scaler.pkl"
    cols_path = ARTIFACTS_DIR / "cluster_feature_cols.pkl"

    if not kmeans_path.exists():
        raise FileNotFoundError(
            "K-Means artifacts not found. Run scripts/build_features.py --steps clusters"
        )

    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    avail_cols = joblib.load(cols_path) if cols_path.exists() else CLUSTER_FEATURES
    return kmeans, scaler, avail_cols
