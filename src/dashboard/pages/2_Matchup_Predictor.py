"""
Matchup Predictor: select two teams → win probability + SHAP breakdown.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import FEATURE_COLS, LIVE_SEASON


def render_matchup_predictor():
    st.title("🎯 Matchup Predictor")
    st.markdown(
        "Select two teams to get a win probability prediction from each model "
        "and a SHAP breakdown showing which stats are driving the result."
    )

    artifacts = st.session_state.get("artifacts")
    fm = st.session_state.get("feature_matrix")
    torvik = st.session_state.get("torvik_current")

    if artifacts is None:
        st.error("Models not loaded. Run `python scripts/train_models.py` first.")
        return
    if torvik is None or torvik.empty:
        st.warning("No current season data. Run `python scripts/fetch_data.py --years 2026`.")
        return

    # Build team list
    team_names = sorted(torvik["torvik_name"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        team_A_name = st.selectbox("Team A", team_names, index=0, key="pred_team_a")
    with col2:
        team_B_name = st.selectbox(
            "Team B",
            team_names,
            index=min(1, len(team_names) - 1),
            key="pred_team_b"
        )

    predict_btn = st.button("🔮 Predict Winner", type="primary", use_container_width=True)

    if predict_btn or (team_A_name and team_B_name):
        _run_matchup_prediction(team_A_name, team_B_name, torvik, artifacts, fm)


def _run_matchup_prediction(team_A_name, team_B_name, torvik, artifacts, fm):
    if team_A_name == team_B_name:
        st.warning("Select two different teams.")
        return

    stats_A = torvik[torvik["torvik_name"] == team_A_name]
    stats_B = torvik[torvik["torvik_name"] == team_B_name]

    if stats_A.empty or stats_B.empty:
        st.error("Could not find stats for one or both teams.")
        return

    stats_A = stats_A.iloc[0]
    stats_B = stats_B.iloc[0]

    # Build feature vector
    feat_vec = _build_feature_vector(stats_A, stats_B)
    X = np.array([feat_vec], dtype="float32")

    # Get probabilities from all models
    scaler = artifacts.get("scaler")
    model_probs = {}
    for model_name in ["logreg", "xgb", "lgbm", "ensemble"]:
        model = artifacts.get(model_name)
        if model is None:
            continue
        try:
            X_in = scaler.transform(X) if model_name == "logreg" and scaler else X
            p = float(model.predict_proba(X_in)[0, 1])
            model_probs[model_name] = p
        except Exception as e:
            st.warning(f"{model_name} prediction failed: {e}")

    if not model_probs:
        st.error("No model predictions available.")
        return

    p_ensemble = model_probs.get("ensemble", list(model_probs.values())[0])

    # ── Main probability gauge ───────────────────────────────────────────────
    st.subheader("Win Probability")
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown(f"### {team_A_name}")
        pct_A = p_ensemble * 100
        color_A = "#2ca02c" if pct_A > 50 else "#d62728"
        st.markdown(
            f"<h1 style='color:{color_A};text-align:center'>{pct_A:.1f}%</h1>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("<h2 style='text-align:center;padding-top:30px'>vs</h2>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"### {team_B_name}")
        pct_B = (1 - p_ensemble) * 100
        color_B = "#2ca02c" if pct_B > 50 else "#d62728"
        st.markdown(
            f"<h1 style='color:{color_B};text-align:center'>{pct_B:.1f}%</h1>",
            unsafe_allow_html=True
        )

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_ensemble * 100,
        title={"text": f"{team_A_name} Win Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 60], "color": "#ffffcc"},
                {"range": [60, 100], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
        number={"suffix": "%", "valueformat": ".1f"},
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Model Agreement Panel ─────────────────────────────────────────────────
    st.subheader("Model Agreement")
    model_cols = st.columns(len(model_probs))
    for col, (name, prob) in zip(model_cols, model_probs.items()):
        with col:
            winner = team_A_name if prob > 0.5 else team_B_name
            st.metric(
                label=name.upper(),
                value=f"{prob * 100:.1f}%",
                delta=f"→ {winner}",
                delta_color="off",
            )

    # ── Key Stats Comparison ─────────────────────────────────────────────────
    st.subheader("Key Stats Comparison")
    _render_stats_comparison(stats_A, stats_B, team_A_name, team_B_name)

    # ── SHAP Breakdown ───────────────────────────────────────────────────────
    st.subheader(f"What's Driving the Prediction? (SHAP — XGBoost)")
    _render_shap_waterfall(X[0], artifacts, feat_vec, team_A_name)


def _build_feature_vector(stats_A: pd.Series, stats_B: pd.Series) -> list[float]:
    """Constructs a feature vector from two team stat rows."""
    def g(stats, key, default=0.0):
        v = stats.get(key, default)
        return float(v) if pd.notna(v) else default

    feat = {
        "AdjO_diff":             g(stats_A, "AdjO") - g(stats_B, "AdjD"),
        "AdjD_diff":             g(stats_A, "AdjD") - g(stats_B, "AdjO"),
        "AdjNetRtg_diff":        (g(stats_A, "AdjO") - g(stats_A, "AdjD")) - (g(stats_B, "AdjO") - g(stats_B, "AdjD")),
        "barthag_diff":          g(stats_A, "barthag") - g(stats_B, "barthag"),
        "AdjT_diff":             g(stats_A, "AdjT") - g(stats_B, "AdjT"),
        "AdjT_interaction":      g(stats_A, "AdjT") * g(stats_B, "AdjT"),
        "eFG_diff":              g(stats_A, "eFG_pct") - g(stats_B, "eFG_pct_d"),
        "eFG_d_diff":            g(stats_A, "eFG_pct_d") - g(stats_B, "eFG_pct"),
        "TO_off_diff":           g(stats_A, "TO_pct") - g(stats_B, "TO_pct"),
        "TO_forced_diff":        g(stats_A, "TO_forced_pct") - g(stats_B, "TO_forced_pct"),
        "OR_diff":               g(stats_A, "OR_pct") - g(stats_B, "DR_pct"),
        "FTR_diff":              g(stats_A, "FTR") - g(stats_B, "FTR"),
        "Blk_diff":              g(stats_A, "Blk_pct") - g(stats_B, "Blk_pct"),
        "Stl_diff":              g(stats_A, "Stl_pct") - g(stats_B, "Stl_pct"),
        "W_pct_diff":            g(stats_A, "W_pct") - g(stats_B, "W_pct"),
        "SOS_diff":              g(stats_A, "SOS") - g(stats_B, "SOS"),
        "roll5_AdjO_diff":       0.0,
        "roll5_AdjD_diff":       0.0,
        "roll5_net_diff":        0.0,
        "style_cluster_A":       0.0,
        "style_cluster_B":       0.0,
        "style_cluster_interaction": 0.0,
        "seed_diff":             0.0,
        "seed_upset_flag":       0.0,
    }
    return [feat.get(c, 0.0) for c in FEATURE_COLS]


def _render_stats_comparison(stats_A, stats_B, name_A, name_B):
    """Side-by-side stats table with color coding."""
    compare_stats = {
        "AdjO (Off. Efficiency)":    ("AdjO", True),
        "AdjD (Def. Efficiency)":    ("AdjD", False),
        "Tempo (poss/40 min)":       ("AdjT", True),
        "eFG% (Offense)":            ("eFG_pct", True),
        "eFG% Allowed (Defense)":    ("eFG_pct_d", False),
        "Turnover Rate":             ("TO_pct", False),
        "Forced TO Rate":            ("TO_forced_pct", True),
        "Off. Rebound %":            ("OR_pct", True),
        "Def. Rebound %":            ("DR_pct", True),
        "Free Throw Rate":           ("FTR", True),
        "Block %":                   ("Blk_pct", True),
        "Steal %":                   ("Stl_pct", True),
        "Win %":                     ("W_pct", True),
        "Strength of Schedule":      ("SOS", True),
    }

    rows = []
    for label, (col, higher_is_better) in compare_stats.items():
        val_A = float(stats_A.get(col, np.nan)) if pd.notna(stats_A.get(col)) else np.nan
        val_B = float(stats_B.get(col, np.nan)) if pd.notna(stats_B.get(col)) else np.nan

        if not np.isnan(val_A) and not np.isnan(val_B):
            a_better = (val_A > val_B) == higher_is_better
            rows.append({
                "Stat": label,
                name_A: f"{val_A:.2f}",
                name_B: f"{val_B:.2f}",
                "Advantage": name_A if a_better else name_B,
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_shap_waterfall(
    x_row: np.ndarray,
    artifacts: dict,
    feat_vals: list,
    team_A_name: str,
):
    """Renders SHAP waterfall chart for a single matchup."""
    try:
        from src.models.shap_explainer import get_explainer, shap_for_matchup

        xgb_model = artifacts.get("xgb")
        if xgb_model is None:
            st.info("XGBoost model not available for SHAP.")
            return

        # Build background (use zeros as reference)
        X_bg = np.zeros((10, len(FEATURE_COLS)), dtype="float32")
        explainer = get_explainer(xgb_model, "xgb", X_bg)
        shap_df = shap_for_matchup(x_row, explainer, FEATURE_COLS)

        # Show top-15 features
        top = shap_df.head(15)

        colors = ["#2ca02c" if v > 0 else "#d62728" for v in top["shap_value"]]
        fig = go.Figure(go.Bar(
            x=top["shap_value"],
            y=top["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in top["shap_value"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"SHAP Values (positive = favors {team_A_name})",
            xaxis_title="SHAP Value",
            height=450,
            yaxis={"autorange": "reversed"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Green bars = stat advantage for Team A. "
            "Red bars = stat advantage for Team B."
        )

    except Exception as e:
        st.warning(f"SHAP visualization not available: {e}")
        st.caption("Install SHAP: `pip install shap`")


render_matchup_predictor()
