"""
Tournament backtesting: runs the model against historical brackets
where outcomes are known. Lets you evaluate:
  - Per-game prediction accuracy in tournament games
  - Stat correlation analysis (which features predict tournament winners)
  - Simulated bracket score vs actual results
  - Upset analysis by seed matchup
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURES_DIR, FEATURE_COLS, TARGET_COL, ROUND_ORDER, ROUND_POINTS

logger = logging.getLogger(__name__)


def backtest_tournament(
    year: int,
    artifacts: dict = None,
    model_name: str = "ensemble",
) -> dict:
    """
    Runs the model against a historical tournament bracket (known outcomes).

    Args:
        year:        tournament year (e.g., 2023, 2024, 2025)
        artifacts:   loaded model artifacts (loaded from disk if None)
        model_name:  which model to use ("logreg" | "xgb" | "lgbm" | "ensemble")

    Returns dict:
        year             int
        model_name       str
        games_df         DataFrame  — per-game predictions vs actual outcomes
        metrics          dict       — accuracy, AUC, Brier, log_loss
        by_round         DataFrame  — accuracy per tournament round
        by_seed_matchup  DataFrame  — accuracy by seed matchup (e.g., 1v16, 5v12)
        bracket_score    float      — simulated bracket score using actual winners
        upset_analysis   DataFrame  — predicted probability for actual upset games
    """
    if artifacts is None:
        from src.models.train import load_artifacts
        artifacts = load_artifacts()

    from src.features.engineer import load_feature_matrix, get_X_y
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

    fm = load_feature_matrix()
    tourney_games = fm[
        (fm["season"] == year) & (fm["is_tournament"] == True)
    ].copy()

    if tourney_games.empty:
        raise ValueError(
            f"No tournament games found for {year}. "
            f"Check that {year} is in the feature matrix."
        )

    X = tourney_games[FEATURE_COLS].fillna(0).values.astype("float32")
    y_true = tourney_games[TARGET_COL].values

    # Predict
    model = artifacts[model_name]
    scaler = artifacts.get("scaler")

    if model_name == "logreg" and scaler is not None:
        X_in = scaler.transform(X)
    else:
        X_in = X

    probs = model.predict_proba(X_in)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Build games DataFrame — only include columns that exist
    base_cols = ["game_id", "season", "tournament_round", "team_A_id", "team_B_id", TARGET_COL]
    optional_cols = ["day_num", "seed_A", "seed_B", "seed_diff", "seed_upset_flag"]
    present_cols = base_cols + [c for c in optional_cols if c in tourney_games.columns]
    games_df = tourney_games[present_cols].copy()
    games_df["pred_prob_A"] = probs.astype("float32")
    games_df["pred_win_A"] = preds
    games_df["correct"] = (preds == y_true).astype(int)

    # Derive upset flag: seed_diff > 0 means team_A is the underdog (higher seed number)
    if "seed_A" in games_df.columns and "seed_B" in games_df.columns:
        games_df["is_upset"] = (
            games_df["seed_A"].notna() & games_df["seed_B"].notna() &
            (games_df["seed_A"] > games_df["seed_B"]) & (y_true == 1)
        ).astype(int)
    elif "seed_diff" in games_df.columns:
        games_df["is_upset"] = (
            (games_df["seed_diff"] > 0) & (y_true == 1)
        ).astype(int)
    else:
        games_df["is_upset"] = 0

    # Overall metrics
    metrics = {
        "year": year,
        "model_name": model_name,
        "n_games": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "auc": round(float(roc_auc_score(y_true, probs)), 4),
        "log_loss": round(float(log_loss(y_true, probs)), 4),
        "brier": round(float(brier_score_loss(y_true, probs)), 4),
    }

    # Metrics by round
    by_round_rows = []
    for round_name in ROUND_ORDER:
        rnd = games_df[games_df["tournament_round"] == round_name]
        if rnd.empty:
            continue
        by_round_rows.append({
            "round": round_name,
            "n_games": len(rnd),
            "accuracy": round(rnd["correct"].mean(), 4),
            "avg_confidence": round(
                (rnd["pred_prob_A"] - 0.5).abs().mean() + 0.5, 4
            ),
            "n_upsets": int(rnd["is_upset"].sum()),
        })
    by_round = pd.DataFrame(by_round_rows)

    # Metrics by seed matchup
    seed_rows = []
    has_seeds = "seed_A" in games_df.columns and "seed_B" in games_df.columns
    if has_seeds and games_df["seed_A"].notna().any():
        for _, g in games_df.iterrows():
            if pd.notna(g["seed_A"]) and pd.notna(g["seed_B"]):
                s_lo = min(int(g["seed_A"]), int(g["seed_B"]))
                s_hi = max(int(g["seed_A"]), int(g["seed_B"]))
                lower_won = (g[TARGET_COL] == 1 and g["seed_A"] > g["seed_B"]) or \
                            (g[TARGET_COL] == 0 and g["seed_B"] > g["seed_A"])
                seed_rows.append({
                    "matchup": f"{s_lo}v{s_hi}",
                    "seed_lo": s_lo,
                    "seed_hi": s_hi,
                    "upset_occurred": int(lower_won),
                    "model_upset_prob": float(
                        g["pred_prob_A"] if g["seed_A"] > g["seed_B"] else 1 - g["pred_prob_A"]
                    ),
                    "correct": int(g["correct"]),
                })

        by_seed_matchup = (
            pd.DataFrame(seed_rows)
            .groupby("matchup")
            .agg(
                n_games=("correct", "count"),
                accuracy=("correct", "mean"),
                upset_rate=("upset_occurred", "mean"),
                avg_model_upset_prob=("model_upset_prob", "mean"),
            )
            .round(4)
            .reset_index()
        )
    else:
        by_seed_matchup = pd.DataFrame()

    # Simulated bracket score (pick winner = most likely per game)
    bracket_score = _compute_bracket_score(games_df)

    # Upset analysis: games where underdog actually won
    upsets = games_df[games_df["is_upset"] == 1].copy()
    upset_cols = ["tournament_round", "pred_prob_A", "team_A_id", "team_B_id", TARGET_COL]
    upset_cols += [c for c in ["seed_A", "seed_B"] if c in games_df.columns]
    upset_analysis = upsets[upset_cols].copy()

    return {
        "year": year,
        "model_name": model_name,
        "games_df": games_df,
        "metrics": metrics,
        "by_round": by_round,
        "by_seed_matchup": by_seed_matchup,
        "bracket_score": bracket_score,
        "upset_analysis": upset_analysis,
    }


def backtest_all_years(
    years: list[int] = None,
    model_name: str = "ensemble",
    artifacts: dict = None,
) -> pd.DataFrame:
    """
    Runs backtest for multiple years and returns a summary DataFrame.
    Useful for evaluating model stability across seasons.

    Returns DataFrame with one row per year: year, accuracy, AUC, Brier, bracket_score
    """
    if years is None:
        from config import VAL_SEASONS, TEST_SEASONS
        years = VAL_SEASONS + TEST_SEASONS

    if artifacts is None:
        from src.models.train import load_artifacts
        artifacts = load_artifacts()

    rows = []
    for year in years:
        try:
            result = backtest_tournament(year, artifacts=artifacts, model_name=model_name)
            row = dict(result["metrics"])
            row["bracket_score"] = result["bracket_score"]
            rows.append(row)
            logger.info(
                f"  {year}: acc={row['accuracy']:.3f}, "
                f"AUC={row['auc']:.3f}, bracket={row['bracket_score']:.0f}pts"
            )
        except Exception as e:
            logger.warning(f"  Backtest failed for {year}: {e}")

    return pd.DataFrame(rows)


def stat_correlation_with_wins(
    year: int = None,
    tourney_only: bool = True,
) -> pd.DataFrame:
    """
    Computes Pearson correlation of each feature with team_A_win.
    Shows which stats are most predictive of tournament vs regular season wins.

    Args:
        year:         if None, uses all data; otherwise filters to a year
        tourney_only: if True, only tournament games

    Returns DataFrame:
        feature         str
        correlation     float   Pearson r with team_A_win
        abs_correlation float
        p_value         float
    """
    from scipy import stats as scipy_stats
    from src.features.engineer import load_feature_matrix

    fm = load_feature_matrix()

    if year is not None:
        fm = fm[fm["season"] == year]
    if tourney_only:
        fm = fm[fm["is_tournament"] == True]

    if fm.empty:
        raise ValueError("No data available with the given filters.")

    rows = []
    for col in FEATURE_COLS:
        if col not in fm.columns:
            continue
        valid = fm[[col, TARGET_COL]].dropna()
        if len(valid) < 30:
            continue
        r, p = scipy_stats.pearsonr(valid[col].astype(float), valid[TARGET_COL].astype(float))
        rows.append({
            "feature": col,
            "correlation": round(float(r), 4),
            "abs_correlation": round(abs(float(r)), 4),
            "p_value": round(float(p), 6),
        })

    df = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return df


def _compute_bracket_score(games_df: pd.DataFrame) -> float:
    """
    Computes simulated ESPN-style bracket score.
    Points per round: R64=1, R32=2, S16=4, E8=8, F4=16, NCG=32
    Awards points when the model's predicted winner matches the actual winner.
    """
    score = 0.0
    for _, row in games_df.iterrows():
        round_pts = ROUND_POINTS.get(row["tournament_round"], 0)
        if row["correct"] == 1:
            score += round_pts
    return score
