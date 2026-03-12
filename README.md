# March Madness Predictor

ML-powered NCAA tournament bracket simulator. Trains on historical tournament results to predict win probabilities and simulate the current year's bracket via 10,000 Monte Carlo runs.

## Quick Start (Dashboard Only)

Clone the repo and launch the dashboard — models and data are included, no setup required:

```bash
git clone <repo-url>
cd march-madness-model
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## Dashboard Pages

1. **Home** — model metrics + top-25 championship leaderboard
2. **Matchup Predictor** — head-to-head win probability + SHAP explanation
3. **Bracket Simulator** — Monte Carlo advancement probabilities by round
4. **Feature Analysis** — SHAP importance, stat correlations, model comparison
5. **Team Profiles** — radar charts + percentile rankings
6. **Backtest** — historical tournament accuracy

## After Selection Sunday

Once the real bracket is announced, drop the seedings file and re-run:

```bash
# Place actual seeds at:
data/raw/manual/seeds_{YEAR}.csv   # columns: TeamName, Seed, Region

# Re-run simulation:
python scripts/run_simulation.py

# Relaunch dashboard:
streamlit run src/dashboard/app.py
```

## Full Pipeline (re-fetch data + retrain from scratch)

```bash
# 1. Fetch Sports-Reference stats + tournament results + team schedules
#    (~25 min for all years due to rate limiting)
python scripts/fetch_data.py --years 2010-2025 --schedules

# 2. Build feature matrix
python scripts/build_features.py --steps crosswalk,games,clusters,features

# 3. Train models (tournament-only for calibrated sim probabilities)
python scripts/train_models.py --tournament-only --evaluate

# 4. Run simulation
python scripts/run_simulation.py
```

## Model Performance

Trained on tournament games only (tournament-only mode for calibrated probabilities):

| Model | Test AUC | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 0.967 | 90.4% |
| XGBoost | 0.943 | 88.4% |
| LightGBM | 0.942 | 88.7% |
| **Stacking Ensemble** | **0.968** | **90.4%** |


## Data Sources

- **Stats**: [Sports-Reference CBB](https://www.sports-reference.com/cbb/) — Advanced stats (AdjO, AdjD, Pace, eFG%, TOV%, ORB%, etc.)
- **Tournament results**: Sports-Reference bracket pages (2010–2025)
- **Optional**: Bart Torvik — manual CSV upload for 2026 projections

## Stack

Python · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit · Plotly · Pandas · Parquet
