"""
March Madness Model 2026 — Streamlit Dashboard
Run with: streamlit run src/dashboard/app.py

Handles shared resource loading (cached per session) and page navigation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from config import APP_TITLE, FEATURES_DIR, ARTIFACTS_DIR

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    """Loads all model .pkl files once per session."""
    try:
        from src.models.train import load_artifacts
        return load_artifacts()
    except Exception as e:
        st.error(f"Could not load model artifacts: {e}")
        st.info("Run `python scripts/train_models.py` first to train the models.")
        return None


@st.cache_data(show_spinner="Loading feature matrix...")
def load_feature_matrix_cached():
    """Loads feature_matrix.parquet once per session."""
    try:
        from src.features.engineer import load_feature_matrix
        return load_feature_matrix()
    except Exception as e:
        st.warning(f"Feature matrix not available: {e}")
        return None


@st.cache_data(show_spinner="Loading simulation results...")
def load_simulation_results_cached():
    """Loads last saved simulation results."""
    try:
        from src.simulation.monte_carlo import load_simulation_results
        return load_simulation_results()
    except Exception:
        return None


@st.cache_data(show_spinner="Loading Torvik data...")
def load_torvik_current():
    """Loads current season Torvik data."""
    try:
        from src.ingestion.torvik import load_trank_season
        from config import LIVE_SEASON
        return load_trank_season(LIVE_SEASON)
    except Exception:
        return None


# ── Navigation ────────────────────────────────────────────────────────────────

pages = {
    "🏠 Home": "src/dashboard/pages/1_Home.py",
    "🎯 Matchup Predictor": "src/dashboard/pages/2_Matchup_Predictor.py",
    "🏆 Bracket Simulator": "src/dashboard/pages/3_Bracket_Simulator.py",
    "📊 Feature Analysis": "src/dashboard/pages/4_Feature_Analysis.py",
    "👥 Team Profiles": "src/dashboard/pages/5_Team_Profiles.py",
    "📈 Backtest": "src/dashboard/pages/6_Backtest.py",
}

# Sidebar navigation
with st.sidebar:
    st.title("🏀 March Madness 2026")
    st.markdown("---")
    page = st.radio("Navigate", list(pages.keys()))
    st.markdown("---")
    st.caption("Model: Stacking Ensemble (LogReg + XGB + LGBM)")
    st.caption("Data: Bart Torvik + Kaggle")

# Load the selected page
page_file = pages[page]

# Pre-load shared resources into session state
if "artifacts" not in st.session_state:
    st.session_state.artifacts = load_model_artifacts()
if "feature_matrix" not in st.session_state:
    st.session_state.feature_matrix = load_feature_matrix_cached()
if "torvik_current" not in st.session_state:
    st.session_state.torvik_current = load_torvik_current()

# Execute the selected page
try:
    exec(open(page_file).read())
except FileNotFoundError:
    st.error(f"Page not found: {page_file}")
except Exception as e:
    st.error(f"Page error: {e}")
    st.exception(e)
