"""
Home page: model performance metrics + current top-25 team leaderboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import TOP_N_TEAMS, LIVE_SEASON


def render_home():
    st.title("🏀 March Madness Model 2026")
    st.markdown(
        "A machine learning ensemble (LogReg + XGBoost + LightGBM) trained on "
        "~16 seasons of college basketball data. Use the sidebar to navigate."
    )

    artifacts = st.session_state.get("artifacts")
    fm = st.session_state.get("feature_matrix")
    torvik = st.session_state.get("torvik_current")

    # ── Model Performance Cards ──────────────────────────────────────────────
    st.subheader("Model Performance")

    if fm is not None and artifacts is not None:
        from src.models.evaluate import evaluate_all
        with st.spinner("Computing metrics..."):
            try:
                metrics_df = evaluate_all(fm, artifacts)
                if not metrics_df.empty:
                    _render_metrics_cards(metrics_df)
                    _render_calibration_plots(metrics_df, fm, artifacts)
            except Exception as e:
                st.warning(f"Could not compute metrics: {e}")
    else:
        st.info("Train models and build features to see performance metrics.")
        _render_placeholder_cards()

    # ── Top 25 Leaderboard ───────────────────────────────────────────────────
    st.subheader(f"Current Top {TOP_N_TEAMS} Teams (2025-26 Season)")

    if torvik is not None and not torvik.empty:
        _render_leaderboard(torvik)
    else:
        st.info(
            "Run `python scripts/fetch_data.py --years 2026` "
            "to load current season data."
        )

    # ── Data Status ──────────────────────────────────────────────────────────
    st.subheader("Data & Pipeline Status")
    _render_data_status(fm)


def _render_metrics_cards(metrics_df: pd.DataFrame):
    """Displays AUC / Brier / Accuracy metric cards per model."""
    models = metrics_df["model_name"].unique()
    cols = st.columns(len(models))

    for col, model in zip(cols, models):
        with col:
            st.markdown(f"**{model.upper()}**")
            val_row = metrics_df[(metrics_df["model_name"] == model) & (metrics_df["split"] == "val")]
            test_row = metrics_df[(metrics_df["model_name"] == model) & (metrics_df["split"] == "test")]

            if not val_row.empty:
                r = val_row.iloc[0]
                st.metric("AUC (Val)", f"{r['auc']:.3f}")
                st.metric("Brier (Val)", f"{r['brier']:.3f}")
                st.metric("Accuracy (Val)", f"{r['accuracy']:.1%}")
            if not test_row.empty:
                r = test_row.iloc[0]
                st.metric("AUC (Test)", f"{r['auc']:.3f}")


def _render_placeholder_cards():
    """Shows placeholder cards when no data is available."""
    cols = st.columns(4)
    for col, name in zip(cols, ["LogReg", "XGB", "LightGBM", "Ensemble"]):
        with col:
            st.markdown(f"**{name}**")
            st.metric("AUC", "—")
            st.metric("Brier", "—")
            st.metric("Accuracy", "—")


def _render_calibration_plots(metrics_df, fm, artifacts):
    """Renders calibration reliability diagrams."""
    from src.features.engineer import get_X_y
    from sklearn.calibration import calibration_curve

    st.subheader("Calibration Curves (Val Set)")
    st.markdown(
        "A perfectly calibrated model's curve lies on the diagonal. "
        "S-shaped = overconfident; concave = underconfident."
    )

    try:
        X_val, y_val, _ = get_X_y(fm, splits=["val"])
        scaler = artifacts.get("scaler")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dash", color="gray"),
            name="Perfect calibration"
        ))

        colors = {"logreg": "#1f77b4", "xgb": "#ff7f0e", "lgbm": "#2ca02c", "ensemble": "#d62728"}
        for name in ["logreg", "xgb", "lgbm", "ensemble"]:
            model = artifacts.get(name)
            if model is None:
                continue
            try:
                X_in = scaler.transform(X_val) if name == "logreg" and scaler else X_val
                probs = model.predict_proba(X_in)[:, 1]
                frac, mean_pred = calibration_curve(y_val, probs, n_bins=10, strategy="uniform")
                fig.add_trace(go.Scatter(
                    x=mean_pred, y=frac,
                    mode="lines+markers", name=name.upper(),
                    line=dict(color=colors.get(name)),
                ))
            except Exception:
                pass

        fig.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render calibration plots: {e}")


def _render_leaderboard(torvik_df: pd.DataFrame):
    """Renders top-N teams table with key stats."""
    display_cols_map = {
        "torvik_name": "Team",
        "conf": "Conf",
        "rank": "Rank",
        "AdjO": "AdjO",
        "AdjD": "AdjD",
        "AdjT": "Tempo",
        "eFG_pct": "eFG%",
        "TO_pct": "TO%",
        "OR_pct": "OR%",
        "W_pct": "Win%",
        "SOS": "SOS",
    }

    avail = {k: v for k, v in display_cols_map.items() if k in torvik_df.columns}
    df = torvik_df[list(avail.keys())].rename(columns=avail)

    # Add net rating
    if "AdjO" in avail and "AdjD" in avail:
        df["NetRtg"] = (torvik_df["AdjO"] - torvik_df["AdjD"]).round(1)

    # Sort by rank or NetRtg
    sort_col = "Rank" if "Rank" in df.columns else "NetRtg" if "NetRtg" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=True).head(TOP_N_TEAMS)

    # Format
    for col in ["AdjO", "AdjD", "Tempo", "NetRtg"]:
        if col in df.columns:
            df[col] = df[col].round(1)
    for col in ["eFG%", "TO%", "OR%", "Win%"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1).astype(str) + "%"

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Conference breakdown
    if "Conf" in df.columns:
        st.subheader("Conference Representation (Top 25)")
        conf_counts = df["Conf"].value_counts().reset_index()
        conf_counts.columns = ["Conference", "Count"]
        fig = px.bar(conf_counts, x="Conference", y="Count", color="Conference")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)


def _render_data_status(fm):
    """Shows pipeline status indicators."""
    from config import ARTIFACTS_DIR, FEATURES_DIR, TORVIK_DIR, KAGGLE_DIR

    checks = {
        "Torvik data fetched": any(TORVIK_DIR.glob("trank_*.json")),
        "Kaggle files present": (KAGGLE_DIR / "MTeams.csv").exists(),
        "Feature matrix built": (FEATURES_DIR / "feature_matrix.parquet").exists(),
        "Models trained": (ARTIFACTS_DIR / "ensemble.pkl").exists(),
        "Simulation run": (FEATURES_DIR / "simulation_results.parquet").exists(),
    }

    cols = st.columns(len(checks))
    for col, (label, status) in zip(cols, checks.items()):
        with col:
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} **{label}**")


render_home()
