"""
Team Profiles: per-team radar chart, efficiency trends, style cluster, schedule.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import LIVE_SEASON


def render_team_profiles():
    st.title("👥 Team Profiles")
    st.markdown("Deep dive into individual team stats and style for the 2026 season.")

    torvik = st.session_state.get("torvik_current")
    if torvik is None or torvik.empty:
        st.error("No current season Torvik data. Run `python scripts/fetch_data.py --years 2026`.")
        return

    team_names = sorted(torvik["torvik_name"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.selectbox("Select Team", team_names, key="profile_team")
    with col2:
        compare_name = st.selectbox(
            "Compare Against (optional)", ["None"] + team_names,
            key="compare_team"
        )

    if not team_name:
        return

    team_row = torvik[torvik["torvik_name"] == team_name].iloc[0]
    compare_row = torvik[torvik["torvik_name"] == compare_name].iloc[0] if compare_name != "None" else None

    # ── Team Header ──────────────────────────────────────────────────────────
    rank = int(team_row.get("rank", 0)) if pd.notna(team_row.get("rank")) else "N/A"
    net_rtg = float(team_row.get("AdjO", 0) - team_row.get("AdjD", 0)) if pd.notna(team_row.get("AdjO")) else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Torvik Rank", f"#{rank}")
    with col2:
        st.metric("Net Rating", f"{net_rtg:.1f}")
    with col3:
        st.metric("Adj. Offense", f"{float(team_row.get('AdjO', 0)):.1f}")
    with col4:
        st.metric("Adj. Defense", f"{float(team_row.get('AdjD', 100)):.1f}")

    # ── Radar Chart ──────────────────────────────────────────────────────────
    st.subheader("Style Radar Chart")
    _render_radar(torvik, team_row, compare_row, team_name, compare_name)

    # ── Key Stats Table ──────────────────────────────────────────────────────
    st.subheader("Full Stats")
    _render_stats_table(team_row, compare_row, team_name, compare_name, torvik)

    # ── Style Cluster ─────────────────────────────────────────────────────────
    _render_cluster_info(team_name)


def _render_radar(torvik, team_row, compare_row, team_name, compare_name):
    """Percentile-normalized radar chart."""
    radar_stats = {
        "AdjO": "Offense",
        "AdjD_inv": "Defense",   # inverted: lower AdjD = better
        "AdjT": "Tempo",
        "eFG_pct": "Shooting",
        "TO_forced_pct": "Force TOs",
        "OR_pct": "Off. Rebounds",
    }

    def get_percentile(val, col, invert=False):
        series = pd.to_numeric(torvik[col], errors="coerce").dropna()
        if series.empty or pd.isna(val):
            return 50.0
        pct = float((series < val).mean() * 100)
        return 100 - pct if invert else pct

    categories = list(radar_stats.values())
    team_vals = []
    compare_vals = []

    for col, label in radar_stats.items():
        is_inv = col == "AdjD_inv"
        actual_col = "AdjD" if is_inv else col
        val = float(team_row.get(actual_col, np.nan)) if pd.notna(team_row.get(actual_col)) else np.nan
        team_vals.append(get_percentile(val, actual_col, invert=is_inv))
        if compare_row is not None:
            val_b = float(compare_row.get(actual_col, np.nan)) if pd.notna(compare_row.get(actual_col)) else np.nan
            compare_vals.append(get_percentile(val_b, actual_col, invert=is_inv))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=team_vals + [team_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=team_name,
        line_color="#1f77b4",
    ))
    if compare_row is not None:
        fig.add_trace(go.Scatterpolar(
            r=compare_vals + [compare_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=compare_name,
            line_color="#ff7f0e",
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickfont_size=9)),
        showlegend=True,
        height=400,
        title="Percentile Rankings vs All D1 Teams",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher = better percentile (Defense is inverted: higher = fewer points allowed)")


def _render_stats_table(team_row, compare_row, team_name, compare_name, torvik):
    """Side-by-side stats table with percentile ranks."""
    stat_labels = {
        "AdjO": "Adj. Offensive Efficiency",
        "AdjD": "Adj. Defensive Efficiency",
        "AdjT": "Adjusted Tempo (poss/40)",
        "eFG_pct": "eFG% (Offense)",
        "eFG_pct_d": "eFG% Allowed (Defense)",
        "TO_pct": "Turnover Rate (Offense)",
        "TO_forced_pct": "Forced Turnover Rate",
        "OR_pct": "Offensive Rebound %",
        "DR_pct": "Defensive Rebound %",
        "FTR": "Free Throw Rate",
        "Blk_pct": "Block %",
        "Stl_pct": "Steal %",
        "W_pct": "Win %",
        "SOS": "Strength of Schedule",
        "barthag": "Power Rating (Barthag)",
    }

    rows = []
    for col, label in stat_labels.items():
        if col not in team_row.index and col not in torvik.columns:
            continue
        val_A = float(team_row.get(col, np.nan)) if pd.notna(team_row.get(col)) else np.nan
        if np.isnan(val_A):
            continue

        series = pd.to_numeric(torvik[col], errors="coerce").dropna()
        pct = float((series < val_A).mean() * 100) if len(series) > 0 else 50.0

        row = {
            "Stat": label,
            team_name: f"{val_A:.3f}",
            f"{team_name} %ile": f"{pct:.0f}%",
        }
        if compare_row is not None:
            val_B = float(compare_row.get(col, np.nan)) if pd.notna(compare_row.get(col)) else np.nan
            if not np.isnan(val_B):
                pct_B = float((series < val_B).mean() * 100) if len(series) > 0 else 50.0
                row[compare_name] = f"{val_B:.3f}"
                row[f"{compare_name} %ile"] = f"{pct_B:.0f}%"

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_cluster_info(team_name: str):
    """Shows style cluster assignment if available."""
    try:
        from src.features.clustering import load_cluster_assignments
        from src.ingestion.torvik import load_all_seasons
        from src.processing.crosswalk import load_crosswalk

        clusters = load_cluster_assignments()
        if "torvik_name" not in clusters.columns:
            return

        team_cluster = clusters[
            (clusters["torvik_name"] == team_name) &
            (clusters["year"] == LIVE_SEASON)
        ]

        if not team_cluster.empty:
            cluster_id = int(team_cluster.iloc[0]["style_cluster"])
            cluster_names = {
                0: "Slow Grind / Elite Defense",
                1: "Fast-Paced High Scoring",
                2: "Efficient Mid-Major",
                3: "Transition Heavy",
                4: "Methodical Free-Throw Heavy",
            }
            st.subheader("Playing Style")
            st.info(f"**Style Cluster {cluster_id}**: {cluster_names.get(cluster_id, 'Unknown')}")

    except Exception:
        pass


render_team_profiles()
