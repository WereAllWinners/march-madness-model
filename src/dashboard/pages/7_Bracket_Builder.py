"""
Bracket Builder: interactive round-by-round bracket picker with model win probabilities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import pandas as pd

from config import MANUAL_DIR

# Abbreviated round names (match BracketGame.round_name) → display labels
ROUND_LABELS = {
    "FF":  "First Four",
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8":  "Elite 8",
    "F4":  "Final Four",
    "NCG": "Championship",
}
ROUND_ORDER = ["FF", "R64", "R32", "S16", "E8", "F4", "NCG"]
REGION_ORDER = ["East", "West", "Midwest", "South"]


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading bracket & win probabilities...")
def _load_base(_artifacts, _fm):
    """Load the original bracket template and pre-build live features. Cached."""
    from src.simulation.bracket import build_bracket_2026
    from src.simulation.monte_carlo import _build_live_feature_lookup

    seeds_path = MANUAL_DIR / "seeds_2026.csv"
    seeds_df = pd.read_csv(seeds_path) if seeds_path.exists() else None
    bracket_games, all_teams = build_bracket_2026(seeds_df=seeds_df)
    live_features = _build_live_feature_lookup(_fm, all_teams)
    team_map = {t.kaggle_id: t for t in all_teams}
    return bracket_games, team_map, live_features


def _fresh_bracket(bracket_template):
    """Return a fresh mutable copy of the bracket template."""
    from src.simulation.monte_carlo import _copy_bracket
    return _copy_bracket(bracket_template)


def _init_state(bracket_template):
    """Initialize session state from a fresh bracket copy."""
    if "bb_bracket" not in st.session_state:
        st.session_state.bb_bracket = _fresh_bracket(bracket_template)
    if "bb_picks" not in st.session_state:
        # game_id → team_id of the picked winner
        st.session_state.bb_picks = {}


# ── Win probability ───────────────────────────────────────────────────────────

def _win_prob(team_a, team_b, live_features, artifacts):
    from src.simulation.monte_carlo import predict_matchup_prob
    return predict_matchup_prob(team_a, team_b, live_features, artifacts)


# ── Pick handler ──────────────────────────────────────────────────────────────

def _make_pick(game_id, winner_team, bracket, bracket_template, live_features, artifacts):
    """Record a pick, advance winner in the working bracket, rerun."""
    from src.simulation.monte_carlo import _advance_in_bracket

    game = next((g for g in bracket if g.game_id == game_id), None)
    if game is None:
        return

    game.winner = winner_team
    st.session_state.bb_picks[game_id] = winner_team.kaggle_id
    _advance_in_bracket(bracket, game, winner_team)
    st.rerun()


def _undo_pick(game_id, bracket_template):
    """Remove all picks from game_id onward and rebuild the bracket from scratch."""
    # Find the round of the undone game to know which picks to drop
    original_games = {g.game_id: g for g in bracket_template}
    target_game = original_games.get(game_id)
    if target_game is None:
        return

    round_idx = ROUND_ORDER.index(target_game.round_name) if target_game.round_name in ROUND_ORDER else 0

    # Drop picks for this round and all later rounds
    picks_to_remove = []
    new_bracket = _fresh_bracket(bracket_template)

    # Replay all picks up to (but not including) this round
    from src.simulation.monte_carlo import _advance_in_bracket
    kept_picks = {gid: wid for gid, wid in st.session_state.bb_picks.items()
                  if gid != game_id}

    # Rebuild bracket by replaying kept picks in round order
    team_map = st.session_state.get("bb_team_map", {})
    for round_name in ROUND_ORDER:
        for g in new_bracket:
            if g.round_name != round_name:
                continue
            if g.game_id in kept_picks:
                winner = team_map.get(kept_picks[g.game_id])
                if winner:
                    g.winner = winner
                    _advance_in_bracket(new_bracket, g, winner)
                else:
                    # winner not in team_map — drop this pick too
                    kept_picks.pop(g.game_id, None)

    st.session_state.bb_bracket = new_bracket
    st.session_state.bb_picks = kept_picks
    st.rerun()


# ── Game card ─────────────────────────────────────────────────────────────────

def _render_game(game, picks, team_map, live_features, artifacts, bracket, bracket_template):
    if game.team_A is None or game.team_B is None:
        return  # not yet available

    team_a = game.team_A
    team_b = game.team_B
    game_id = game.game_id
    picked_id = picks.get(game_id)

    prob_a = _win_prob(team_a, team_b, live_features, artifacts)
    prob_b = 1.0 - prob_a
    fav_id = team_a.kaggle_id if prob_a >= prob_b else team_b.kaggle_id

    def _btn_label(t, prob):
        return f"({t.seed}) {t.name}  {prob:.0%}"

    col_a, col_mid, col_b = st.columns([5, 1, 5])

    with col_a:
        if picked_id == team_a.kaggle_id:
            is_upset = team_a.kaggle_id != fav_id
            icon = "🔴" if is_upset else "🟢"
            st.success(f"{icon} **({team_a.seed}) {team_a.name}** — {prob_a:.0%} ✓")
            if st.button("↩ Undo", key=f"undo_a_{game_id}", use_container_width=True):
                _undo_pick(game_id, bracket_template)
        elif picked_id is not None:
            st.markdown(f"~~({team_a.seed}) {team_a.name}~~  {prob_a:.0%}")
        else:
            if st.button(_btn_label(team_a, prob_a), key=f"pick_a_{game_id}", use_container_width=True):
                _make_pick(game_id, team_a, bracket, bracket_template, live_features, artifacts)

    with col_mid:
        st.markdown("<div style='text-align:center;padding-top:8px;color:gray;font-size:12px'>vs</div>",
                    unsafe_allow_html=True)

    with col_b:
        if picked_id == team_b.kaggle_id:
            is_upset = team_b.kaggle_id != fav_id
            icon = "🔴" if is_upset else "🟢"
            st.success(f"{icon} **({team_b.seed}) {team_b.name}** — {prob_b:.0%} ✓")
            if st.button("↩ Undo", key=f"undo_b_{game_id}", use_container_width=True):
                _undo_pick(game_id, bracket_template)
        elif picked_id is not None:
            st.markdown(f"~~({team_b.seed}) {team_b.name}~~  {prob_b:.0%}")
        else:
            if st.button(_btn_label(team_b, prob_b), key=f"pick_b_{game_id}", use_container_width=True):
                _make_pick(game_id, team_b, bracket, bracket_template, live_features, artifacts)

    # Probability bar (blue = team A, orange = team B)
    st.markdown(
        f"""<div style='display:flex;height:5px;border-radius:3px;overflow:hidden;margin:-4px 0 14px 0'>
          <div style='width:{prob_a*100:.1f}%;background:#1f77b4'></div>
          <div style='width:{prob_b*100:.1f}%;background:#ff7f0e'></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def render_bracket_builder():
    st.title("📋 Bracket Builder")
    st.markdown(
        "Pick winners round by round. Win probabilities come from the ensemble model.  \n"
        "🟢 = model's favorite &nbsp;|&nbsp; 🔴 = upset pick &nbsp;|&nbsp; ~~strikethrough~~ = eliminated"
    )

    artifacts = st.session_state.get("artifacts")
    fm = st.session_state.get("feature_matrix")

    if artifacts is None or fm is None:
        st.error("Models or feature matrix not loaded. Restart the app.")
        return

    seeds_path = MANUAL_DIR / "seeds_2026.csv"
    if not seeds_path.exists():
        st.warning("No bracket loaded. Place `data/raw/manual/seeds_2026.csv` and restart the app.")
        return

    bracket_template, team_map, live_features = _load_base(artifacts, fm)
    _init_state(bracket_template)

    # Store team_map in session state for undo replay
    st.session_state.bb_team_map = team_map

    bracket = st.session_state.bb_bracket
    picks   = st.session_state.bb_picks

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏆 Your Bracket")

        total_games = len(bracket_template)
        n_picked = len(picks)
        st.progress(n_picked / max(total_games, 1), text=f"{n_picked} / {total_games} games picked")

        # Champion
        champ_game = next((g for g in bracket if g.round_name == "NCG"), None)
        if champ_game and champ_game.game_id in picks:
            champ = team_map.get(picks[champ_game.game_id])
            if champ:
                st.success(f"🏆 Champion: **{champ.name}**")

        # Upsets picked
        n_upsets = 0
        for gid, wid in picks.items():
            g = next((x for x in bracket_template if x.game_id == gid), None)
            if g and g.team_A and g.team_B:
                winner = team_map.get(wid)
                if winner:
                    other = g.team_B if wid == g.team_A.kaggle_id else g.team_A
                    if winner.seed > other.seed:
                        n_upsets += 1
        st.metric("Upset picks", n_upsets)

        st.divider()
        if st.button("🔄 Reset Bracket", use_container_width=True):
            for key in ["bb_bracket", "bb_picks", "bb_team_map"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Round tabs ────────────────────────────────────────────────────────────
    tab_labels = [ROUND_LABELS[r] for r in ROUND_ORDER]
    tabs = st.tabs(tab_labels)

    for tab, round_code in zip(tabs, ROUND_ORDER):
        with tab:
            round_games = [g for g in bracket if g.round_name == round_code]
            available = [g for g in round_games if g.team_A is not None and g.team_B is not None]
            picked_count = sum(1 for g in round_games if g.game_id in picks)

            st.caption(f"{picked_count} / {len(round_games)} games picked")

            if not available and not any(g.game_id in picks for g in round_games):
                prev_idx = ROUND_ORDER.index(round_code) - 1
                if prev_idx >= 0:
                    st.info(f"Complete the {ROUND_LABELS[ROUND_ORDER[prev_idx]]} first.")
                else:
                    st.info("No games available yet.")
                continue

            # Early rounds: group by region
            if round_code in ("FF", "R64", "R32", "S16", "E8"):
                regions_present = sorted(
                    {g.region for g in round_games if g.team_A or g.game_id in picks},
                    key=lambda r: REGION_ORDER.index(r) if r in REGION_ORDER else 99
                )
                if len(regions_present) > 1:
                    rtabs = st.tabs(regions_present)
                    for rtab, region in zip(rtabs, regions_present):
                        with rtab:
                            for g in round_games:
                                if g.region != region:
                                    continue
                                _render_game(g, picks, team_map, live_features,
                                             artifacts, bracket, bracket_template)
                                st.divider()
                else:
                    for g in round_games:
                        _render_game(g, picks, team_map, live_features,
                                     artifacts, bracket, bracket_template)
                        st.divider()
            else:
                # Final Four & Championship
                for g in round_games:
                    _render_game(g, picks, team_map, live_features,
                                 artifacts, bracket, bracket_template)
                    st.divider()


render_bracket_builder()
