"""
Bracket Simulator: run Monte Carlo tournament simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import N_SIMULATIONS, LIVE_SEASON, FEATURES_DIR


def render_bracket_simulator():
    st.title("🏆 Bracket Simulator")
    st.markdown(
        "Run 10,000 Monte Carlo simulations of the 2026 NCAA tournament. "
        "Each game is resolved using the ensemble model's win probability."
    )

    artifacts = st.session_state.get("artifacts")
    fm = st.session_state.get("feature_matrix")
    torvik = st.session_state.get("torvik_current")

    if artifacts is None:
        st.error("Models not trained. Run `python scripts/train_models.py` first.")
        return

    # Check for existing simulation results
    sim_results = None
    results_path = FEATURES_DIR / "simulation_results.parquet"
    if results_path.exists():
        sim_results = pd.read_parquet(results_path)

    # ── Run Simulation Button ─────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        n_sims = st.select_slider(
            "Number of simulations",
            options=[1000, 5000, 10000, 25000],
            value=N_SIMULATIONS,
        )
    with col2:
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        sim_results = _run_simulation(torvik, fm, artifacts, n_sims)

    if sim_results is None or sim_results.empty:
        st.info(
            "No simulation results yet. Click 'Run Simulation' above, "
            "or run `python scripts/run_simulation.py` from the command line."
        )
        return

    # ── Results Display ───────────────────────────────────────────────────────
    st.subheader("Championship Probabilities")
    _render_championship_probs(sim_results)

    st.subheader("Advancement Probabilities by Round")
    _render_advancement_table(sim_results)

    st.subheader("Upset Probability Heatmap (by seed matchup)")
    _render_upset_heatmap()

    st.subheader("Expected Bracket Score (ESPN scoring)")
    _render_bracket_score(sim_results)


def _run_simulation(torvik, fm, artifacts, n_sims):
    """Runs Monte Carlo simulation with progress bar."""
    from src.simulation.bracket import build_bracket_2026
    from src.simulation.monte_carlo import run_simulation

    if torvik is None or torvik.empty:
        st.error("No Torvik data available for bracket construction.")
        return None

    if fm is None:
        st.error("Feature matrix not loaded.")
        return None

    progress_bar = st.progress(0.0, text="Building bracket...")
    status_text = st.empty()

    try:
        # Build bracket
        bracket_games, all_teams = build_bracket_2026(torvik_df=torvik)
        if not all_teams:
            st.error("Could not build bracket. Check that Torvik data is available.")
            return None

        st.info(f"Bracket: {len(all_teams)} teams loaded")
        progress_bar.progress(0.05, text="Bracket built. Starting simulations...")

        last_update = [0]
        def update_progress(sim_idx, total):
            pct = sim_idx / total
            if sim_idx - last_update[0] >= total // 20:
                last_update[0] = sim_idx
                progress_bar.progress(0.05 + pct * 0.90, text=f"Simulation {sim_idx:,}/{total:,}...")

        results = run_simulation(
            all_teams=all_teams,
            bracket_template=bracket_games,
            feature_matrix_df=fm,
            artifacts=artifacts,
            n_sims=n_sims,
            progress_callback=update_progress,
        )
        progress_bar.progress(1.0, text="Complete!")
        status_text.success(f"✅ {n_sims:,} simulations complete!")
        return results

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.exception(e)
        return None


def _render_championship_probs(sim_results: pd.DataFrame):
    """Bar chart of championship probabilities for top teams."""
    top = sim_results.head(20).copy()
    top["label"] = top["name"] + " (" + top["seed"].astype(str) + " seed)"

    fig = px.bar(
        top,
        x="p_Champion",
        y="label",
        orientation="h",
        color="p_Champion",
        color_continuous_scale="Blues",
        text=[f"{p:.1%}" for p in top["p_Champion"]],
    )
    fig.update_layout(
        xaxis_title="P(Champion)",
        yaxis_title="",
        yaxis={"autorange": "reversed"},
        coloraxis_showscale=False,
        height=500,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def _render_advancement_table(sim_results: pd.DataFrame):
    """Shows full advancement probability table."""
    rounds = ["p_R64", "p_R32", "p_S16", "p_E8", "p_F4", "p_NCG", "p_Champion"]
    avail_rounds = [r for r in rounds if r in sim_results.columns]

    display = sim_results[["name", "seed", "region"] + avail_rounds + ["expected_wins"]].copy()
    display = display.rename(columns={
        "name": "Team", "seed": "Seed", "region": "Region",
        "p_R64": "R64", "p_R32": "R32", "p_S16": "S16",
        "p_E8": "E8", "p_F4": "F4", "p_NCG": "NCG",
        "p_Champion": "Champion",
        "expected_wins": "Exp. Wins",
    })

    # Format as percentages
    pct_cols = ["R64", "R32", "S16", "E8", "F4", "NCG", "Champion"]
    for col in [c for c in pct_cols if c in display.columns]:
        display[col] = display[col].apply(lambda x: f"{x:.1%}")

    # Region filter
    regions = ["All"] + sorted(sim_results["region"].dropna().unique().tolist())
    selected_region = st.selectbox("Filter by region", regions, key="sim_region_filter")
    if selected_region != "All":
        display = display[display["Region"] == selected_region]

    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_upset_heatmap():
    """Historical upset probability heatmap (seed 1 vs seed 16, etc.)."""
    from src.simulation.monte_carlo import build_upset_probability_matrix

    matrix = build_upset_probability_matrix(pd.DataFrame())

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values.astype(float),
        x=[f"Seed {s}" for s in matrix.columns],
        y=[f"Seed {s}" for s in matrix.index],
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in matrix.values.astype(float)],
        texttemplate="%{text}",
        hovertemplate="Seed %{y} vs Seed %{x}: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Historical Win Rate by Seed Matchup (Row seed wins)",
        height=500,
        xaxis_title="Opponent Seed",
        yaxis_title="Team Seed",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Based on 1985–2025 NCAA tournament historical data.")


def _render_bracket_score(sim_results: pd.DataFrame):
    """Shows expected bracket score and top picks."""
    if "expected_bracket_pts" not in sim_results.columns:
        return

    total_exp = sim_results["expected_bracket_pts"].sum()
    perfect_score = 1 + 2 + 4 + 8 + 16 + 32  # per game × n games
    n_games = 63  # R64 through NCG

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expected Total Bracket Score", f"{total_exp:.1f} pts")
    with col2:
        # Best value picks: high sim advancement but low seed (potential upsets)
        potential_upsets = sim_results[sim_results["seed"] >= 10].sort_values(
            "p_S16", ascending=False
        ).head(5)
        st.markdown("**Best Value Upsets (Top S16 candidates, seed 10+)**")
        if not potential_upsets.empty:
            for _, row in potential_upsets.iterrows():
                st.markdown(
                    f"- **{row['name']}** (#{row['seed']} seed): "
                    f"S16={row.get('p_S16', 0):.1%}, "
                    f"Champion={row.get('p_Champion', 0):.1%}"
                )


render_bracket_simulator()
