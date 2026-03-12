"""
Central configuration for the March Madness Model.
All paths, constants, column names, and hyperparameters live here.
Every other module imports from this file.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
TORVIK_DIR      = RAW_DIR / "torvik"
SPORTS_REF_DIR  = RAW_DIR / "sports_ref"
KAGGLE_DIR      = RAW_DIR / "kaggle"
MANUAL_DIR      = RAW_DIR / "manual"   # for seeds_2026.csv etc.
PROCESSED_DIR   = DATA_DIR / "processed"
FEATURES_DIR    = DATA_DIR / "features"
ARTIFACTS_DIR   = ROOT / "src" / "models" / "artifacts"

# Create directories on import
for _d in [TORVIK_DIR, SPORTS_REF_DIR, KAGGLE_DIR, MANUAL_DIR,
           PROCESSED_DIR, FEATURES_DIR, ARTIFACTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Season range ──────────────────────────────────────────────────────────────
FIRST_SEASON   = 2010       # earliest training season (2009-10)
CURRENT_SEASON = 2026       # live season (2025-26)
ALL_SEASONS    = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

# Train / Val / Test / Live split
TRAIN_SEASONS = list(range(2010, 2023))  # 2010–2022 inclusive
VAL_SEASONS   = [2023, 2024]
TEST_SEASONS  = [2025]
LIVE_SEASON   = 2026

# ── Data source URLs ──────────────────────────────────────────────────────────
TORVIK_BASE_URL       = "https://barttorvik.com/trank.php"
TORVIK_GAME_URL       = "https://barttorvik.com/team-game.php"
SPORTS_REF_STATS_URL  = "https://www.sports-reference.com/cbb/seasons/men/{year}-school-stats.html"
SPORTS_REF_DELAY_SECS = 4.0
TORVIK_DELAY_SECS     = 1.5

# ── Kaggle expected filenames ─────────────────────────────────────────────────
KAGGLE_FILES = {
    "teams":            "MTeams.csv",
    "seasons":          "MSeasons.csv",
    "seeds":            "MNCAATourneySeeds.csv",
    "slots":            "MNCAATourneySlots.csv",
    "tourney_compact":  "MNCAATourneyCompactResults.csv",
    "tourney_detailed": "MNCAATourneyDetailedResults.csv",
    "reg_compact":      "MRegularSeasonCompactResults.csv",
    "reg_detailed":     "MRegularSeasonDetailedResults.csv",
}

KAGGLE_REQUIRED = ["teams", "seeds", "tourney_compact", "reg_compact"]
KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data"
)

# ── Raw column schema from Torvik trank.php JSON ──────────────────────────────
# Maps Torvik JSON field names → our canonical column names
TORVIK_COL_MAP = {
    "team":         "torvik_name",
    "conf":         "conf",
    "adjoe":        "AdjO",
    "adjde":        "AdjD",
    "barthag":      "barthag",      # power rating
    "efg_o":        "eFG_pct",
    "efg_d":        "eFG_pct_d",    # opponent eFG allowed
    "to_o":         "TO_pct",       # turnover rate (off)
    "to_d":         "TO_forced_pct",# forced turnover rate (def)
    "or_o":         "OR_pct",       # offensive rebound %
    "or_d":         "DR_pct",       # defensive rebound % (actually opp OR%)
    "ftr_o":        "FTR",
    "ftr_d":        "FTR_d",
    "twop_o":       "twop_pct",
    "twop_d":       "twop_pct_d",
    "threep_o":     "threep_pct",
    "threep_d":     "threep_pct_d",
    "blk":          "Blk_pct",
    "stl":          "Stl_pct",
    "adj_t":        "AdjT",
    "wins":         "W",
    "games":        "G",
    "year":         "year",
    "seed":         "seed",
    "rank":         "rank",
    "sos":          "SOS",
    "rk":           "rank",         # alternate key
}

# ── Team-level stat columns (season averages) ─────────────────────────────────
# These are per-team columns that get _A and _B suffixes in game_level.parquet
TEAM_STAT_COLS = [
    "AdjO",
    "AdjD",
    "AdjT",
    "eFG_pct",
    "eFG_pct_d",
    "TO_pct",
    "TO_forced_pct",
    "OR_pct",
    "DR_pct",
    "FTR",
    "Blk_pct",
    "Stl_pct",
    "W_pct",
    "SOS",
    "barthag",
]

# ── Engineered feature column names ──────────────────────────────────────────
# These are the final ML feature columns in feature_matrix.parquet
FEATURE_COLS = [
    # Efficiency cross-matchup differentials
    "AdjO_diff",             # AdjO_A - AdjD_B
    "AdjD_diff",             # AdjD_A - AdjO_B (positive = A has defensive disadvantage)
    "AdjNetRtg_diff",        # (AdjO_A-AdjD_A) - (AdjO_B-AdjD_B)
    "barthag_diff",          # overall power rating diff
    # Tempo
    "AdjT_diff",
    "AdjT_interaction",      # AdjT_A * AdjT_B
    # Four factors
    "eFG_diff",              # eFG_pct_A - eFG_pct_d_B (A offense vs B defense)
    "eFG_d_diff",            # eFG_pct_d_A - eFG_pct_B (A defense vs B offense)
    "TO_off_diff",           # TO_pct_A - TO_pct_B (lower is better off)
    "TO_forced_diff",        # TO_forced_pct_A - TO_forced_pct_B
    "OR_diff",               # OR_pct_A - DR_pct_B (offensive boards vs def boards)
    "FTR_diff",
    # Supplemental
    "Blk_diff",
    "Stl_diff",
    "W_pct_diff",
    "SOS_diff",
    # Style clusters
    "style_cluster_A",
    "style_cluster_B",
    "style_cluster_interaction",
    # Tournament-specific
    "seed_diff",
    "seed_upset_flag",
]

TARGET_COL = "team_A_win"

# ── Clustering ────────────────────────────────────────────────────────────────
N_STYLE_CLUSTERS = 5
CLUSTER_FEATURES = ["AdjO", "AdjD", "AdjT", "eFG_pct", "TO_pct", "OR_pct"]

# ── Model hyperparameters ─────────────────────────────────────────────────────
LOGREG_PARAMS = {
    "C": 0.1,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# ── Simulation ────────────────────────────────────────────────────────────────
N_SIMULATIONS  = 10_000
RANDOM_SEED    = 42

# ── Tournament bracket structure ──────────────────────────────────────────────
# Maps Kaggle DayNum ranges → round name
ROUND_DAY_MAP = {
    (134, 135): "FF",   # First Four
    (136, 137): "R64",
    (138, 139): "R32",
    (143, 144): "S16",
    (145, 146): "E8",
    (152, 152): "F4",
    (154, 154): "NCG",
}

ROUND_ORDER = ["FF", "R64", "R32", "S16", "E8", "F4", "NCG"]
ROUND_POINTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "NCG": 32}

REGIONS = ["East", "West", "South", "Midwest"]

# ── Dashboard ─────────────────────────────────────────────────────────────────
APP_TITLE  = "March Madness Model 2026"
TOP_N_TEAMS = 25
