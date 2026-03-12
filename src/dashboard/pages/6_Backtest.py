"""
Backtest page: tests the model against historical brackets with known outcomes.
Shows per-game accuracy, stat correlations with winners, and bracket score analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import VAL_SEASONS, TEST_SEASONS


def render_backtest():
    st.title("📈 Backtest: Model vs Historical Brackets")
    st.markdown(
        "Validate the model against historical tournament brackets where outcomes are known. "
        "See which statistics most strongly correlate with tournament winners."
    )

    artifacts = st.session_state.get("artifacts")
    fm = st.session_state.get("feature_matrix")

    if artifacts is None or fm is None:
        st.error("Models or feature matrix not loaded. Run the full pipeline first.")
        return

    tab1, tab2, tab3 = st.tabs([
        "🏀 Tournament Backtest",
        "📊 Stat Correlations",
        "📅 Multi-Year Summary",
    ])

    with tab1:
        _render_tournament_backtest(artifacts, fm)

    with tab2:
        _render_stat_correlations(fm)

    with tab3:
        _render_multi_year_summary(artifacts, fm)


def _render_tournament_backtest(artifacts, fm):
    st.subheader("Single-Year Tournament Backtest")

    available_years = sorted(
        fm[fm["is_tournament"] == True]["season"].unique().tolist(),
        reverse=True
    )
    if not available_years:
        st.warning("No tournament games in feature matrix.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        year = st.selectbox(
            "Select Tournament Year",
            available_years,
            format_func=lambda y: f"{y} Tournament ({y-1}-{str(y)[2:]} season)"
        )
    with col2:
        model_name = st.selectbox("Model", ["ensemble", "xgb", "lgbm", "logreg"])

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner(f"Running backtest for {year}..."):
            try:
                from src.models.backtest import backtest_tournament
                result = backtest_tournament(year, artifacts=artifacts, model_name=model_name)
                _display_backtest_result(result)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)
    else:
        st.info("Select a year and model, then click 'Run Backtest'.")


def _display_backtest_result(result: dict):
    """Displays backtest results in multiple sections."""
    metrics = result["metrics"]
    games_df = result["games_df"]
    by_round = result["by_round"]
    by_seed = result["by_seed_matchup"]

    year = result["year"]
    st.success(f"✅ {year} Tournament Backtest Complete")

    # ── Summary Metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("AUC", f"{metrics['auc']:.3f}")
    col3.metric("Brier Score", f"{metrics['brier']:.3f}")
    col4.metric("Bracket Score", f"{result['bracket_score']:.0f} pts")

    # ── By Round ──────────────────────────────────────────────────────────
    if not by_round.empty:
        st.subheader("Accuracy by Tournament Round")
        fig = px.bar(
            by_round,
            x="round",
            y="accuracy",
            color="accuracy",
            color_continuous_scale="RdYlGn",
            text=[f"{a:.1%}" for a in by_round["accuracy"]],
            title="Prediction Accuracy per Round",
            labels={"accuracy": "Accuracy", "round": "Round"},
        )
        fig.update_layout(coloraxis_showscale=False, height=300, yaxis_tickformat=".0%")
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="50% baseline")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_round, use_container_width=True, hide_index=True)

    # ── By Seed Matchup ───────────────────────────────────────────────────
    if not by_seed.empty:
        st.subheader("Performance by Seed Matchup")
        fig = px.scatter(
            by_seed,
            x="upset_rate",
            y="accuracy",
            text="matchup",
            size="n_games",
            color="avg_model_upset_prob",
            color_continuous_scale="RdYlGn",
            title="Model Accuracy vs Historical Upset Rate by Seed Matchup",
            labels={
                "upset_rate": "Historical Upset Rate",
                "accuracy": "Model Accuracy",
                "avg_model_upset_prob": "Model Avg Upset Prob",
            },
        )
        fig.add_hline(y=0.5, line_dash="dash", color="gray")
        fig.add_vline(x=0.5, line_dash="dash", color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # ── Game-level Table ──────────────────────────────────────────────────
    with st.expander("📋 All Games Detail"):
        display_cols = [
            "tournament_round", "team_A_id", "team_B_id",
            "seed_A", "seed_B", "pred_prob_A", "team_A_win", "correct", "is_upset"
        ]
        avail_cols = [c for c in display_cols if c in games_df.columns]
        st.dataframe(
            games_df[avail_cols].sort_values("tournament_round"),
            use_container_width=True, hide_index=True
        )

    # ── Upset Analysis ─────────────────────────────────────────────────────
    upset_df = result.get("upset_analysis", pd.DataFrame())
    if not upset_df.empty:
        st.subheader("Upset Games: Model Confidence")
        st.markdown("Games where the underdog won. How confident was the model in the wrong team?")
        upset_display = upset_df.copy()
        upset_display["model_predicted_upset"] = (upset_display["pred_prob_A"] > 0.5).astype(int)
        st.dataframe(upset_display, use_container_width=True, hide_index=True)


def _render_stat_correlations(fm: pd.DataFrame):
    st.subheader("Stat Correlation with Tournament Winners")
    st.markdown(
        "Pearson correlation between each engineered feature and game outcome. "
        "**Positive** = stat favors Team A winning. Filter by context."
    )

    col1, col2 = st.columns(2)
    with col1:
        context = st.radio(
            "Game context",
            ["Tournament only", "Regular season only", "All games"],
        )
    with col2:
        year_filter = st.selectbox(
            "Season",
            ["All seasons"] + sorted(fm["season"].unique().tolist(), reverse=True),
        )

    if st.button("📊 Compute Correlations", type="primary"):
        with st.spinner("Computing correlations..."):
            try:
                from src.models.backtest import stat_correlation_with_wins
                tourney_only = context == "Tournament only"
                all_games = context == "All games"
                year = None if year_filter == "All seasons" else int(year_filter)

                # For all games, pass tourney_only=False but don't filter
                subset = fm.copy()
                if year:
                    subset = subset[subset["season"] == year]
                if context == "Tournament only":
                    subset = subset[subset["is_tournament"] == True]
                elif context == "Regular season only":
                    subset = subset[subset["is_tournament"] == False]

                if subset.empty:
                    st.warning("No data for selected filters.")
                    return

                from src.features.engineer import load_feature_matrix
                from scipy import stats as scipy_stats
                from config import FEATURE_COLS

                rows = []
                for col in FEATURE_COLS:
                    if col not in subset.columns:
                        continue
                    valid = subset[[col, "team_A_win"]].dropna()
                    if len(valid) < 30:
                        continue
                    r, p = scipy_stats.pearsonr(
                        valid[col].astype(float),
                        valid["team_A_win"].astype(float)
                    )
                    rows.append({
                        "Feature": col,
                        "Correlation": round(float(r), 4),
                        "|Correlation|": round(abs(float(r)), 4),
                        "p-value": round(float(p), 6),
                        "Significant": "✅" if p < 0.05 else "❌",
                    })

                corr_df = pd.DataFrame(rows).sort_values("|Correlation|", ascending=False)

                # Bar chart
                top20 = corr_df.head(20)
                colors = ["#2ca02c" if r > 0 else "#d62728" for r in top20["Correlation"]]
                fig = go.Figure(go.Bar(
                    x=top20["Correlation"],
                    y=top20["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{r:.3f}" for r in top20["Correlation"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Top 20 Feature Correlations with Win ({context})",
                    xaxis_title="Pearson r",
                    yaxis={"autorange": "reversed"},
                    height=500,
                )
                fig.add_vline(x=0, line_dash="solid", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Correlation analysis failed: {e}")
                st.exception(e)


def _render_multi_year_summary(artifacts, fm):
    st.subheader("Multi-Year Tournament Performance")
    st.markdown(
        "Model accuracy across multiple tournament years. "
        "Helps assess consistency and identify seasons where the model struggled."
    )

    available_years = sorted(
        fm[fm["is_tournament"] == True]["season"].unique().tolist(),
        reverse=True
    )

    selected_years = st.multiselect(
        "Select years to backtest",
        available_years,
        default=available_years[:min(5, len(available_years))],
    )

    if not selected_years:
        st.info("Select at least one year.")
        return

    if st.button("📈 Run Multi-Year Backtest", type="primary"):
        with st.spinner("Running backtests..."):
            try:
                from src.models.backtest import backtest_tournament

                rows = []
                for year in sorted(selected_years):
                    try:
                        result = backtest_tournament(year, artifacts=artifacts, model_name="ensemble")
                        m = result["metrics"]
                        rows.append({
                            "Year": year,
                            "Accuracy": m["accuracy"],
                            "AUC": m["auc"],
                            "Brier": m["brier"],
                            "Bracket Score": result["bracket_score"],
                            "N Games": m["n_games"],
                        })
                    except Exception as e:
                        st.warning(f"  Year {year} failed: {e}")

                if rows:
                    summary_df = pd.DataFrame(rows).sort_values("Year")

                    # Multi-metric line chart
                    fig = go.Figure()
                    for metric, color in [("Accuracy", "#1f77b4"), ("AUC", "#ff7f0e")]:
                        fig.add_trace(go.Scatter(
                            x=summary_df["Year"],
                            y=summary_df[metric],
                            mode="lines+markers",
                            name=metric,
                            line_color=color,
                        ))
                    fig.add_hline(y=0.5, line_dash="dash", annotation_text="50% baseline")
                    fig.update_layout(
                        title="Tournament Accuracy and AUC Over Years",
                        height=350,
                        yaxis_tickformat=".2f",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Format for display
                    display = summary_df.copy()
                    display["Accuracy"] = display["Accuracy"].apply(lambda x: f"{x:.1%}")
                    display["AUC"] = display["AUC"].apply(lambda x: f"{x:.3f}")
                    display["Brier"] = display["Brier"].apply(lambda x: f"{x:.3f}")
                    st.dataframe(display, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Multi-year backtest failed: {e}")
                st.exception(e)


render_backtest()
