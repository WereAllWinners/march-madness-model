"""
Feature Analysis: global SHAP, correlation heatmap, feature importance comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import FEATURE_COLS, FEATURES_DIR


def render_feature_analysis():
    st.title("📊 Feature Analysis")
    st.markdown(
        "Understand which statistics most strongly predict game outcomes. "
        "Global SHAP values show average feature importance across all games."
    )

    fm = st.session_state.get("feature_matrix")
    artifacts = st.session_state.get("artifacts")

    if fm is None:
        st.error("Feature matrix not loaded.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 SHAP Importance",
        "🔗 Correlations",
        "📊 Model Comparison",
        "🎓 Tournament vs Regular",
    ])

    with tab1:
        _render_shap_importance(artifacts, fm)

    with tab2:
        _render_correlation_heatmap(fm)

    with tab3:
        _render_model_importance_comparison(artifacts)

    with tab4:
        _render_context_comparison(fm)


def _render_shap_importance(artifacts, fm):
    st.subheader("Global SHAP Feature Importance")
    st.markdown(
        "**Mean absolute SHAP value** across all val+test games. "
        "Higher = this feature has more influence on predictions."
    )

    # Try to load cached SHAP
    for model_name in ["xgb", "lgbm"]:
        shap_path = FEATURES_DIR / f"shap_global_{model_name}.parquet"
        if shap_path.exists():
            shap_df = pd.read_parquet(shap_path)
            top20 = shap_df.head(20)

            fig = px.bar(
                top20,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title=f"Top 20 Features by Mean |SHAP| ({model_name.upper()})",
                color="mean_abs_shap",
                color_continuous_scale="Blues",
                text=[f"{v:.4f}" for v in top20["mean_abs_shap"]],
            )
            fig.update_layout(
                yaxis={"autorange": "reversed"},
                coloraxis_showscale=False,
                height=500,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
            break
    else:
        if artifacts is not None:
            if st.button("Compute Global SHAP (may take 30-60 seconds)"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        from src.models.shap_explainer import compute_and_cache_global_shap
                        summaries = compute_and_cache_global_shap(artifacts, fm)
                        st.success("SHAP computed! Reload this tab.")
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")
        else:
            st.info("Train models first to compute SHAP values.")


def _render_correlation_heatmap(fm: pd.DataFrame):
    st.subheader("Feature Correlation Heatmap")
    st.markdown(
        "Pearson correlation between all engineered features. "
        "Highly correlated features (|r| > 0.8) may be redundant."
    )

    avail = [c for c in FEATURE_COLS if c in fm.columns]
    subset = fm[avail].dropna(how="all")

    # Limit to numeric cols
    corr = subset.astype(float).corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=700,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_importance_comparison(artifacts):
    st.subheader("Feature Importance by Model")
    st.markdown(
        "Comparison of feature importances across LogReg coefficients, "
        "XGBoost gain, and LightGBM split gain."
    )

    if artifacts is None:
        st.info("No models loaded.")
        return

    importance_data = {}

    # LogReg coefficients
    logreg = artifacts.get("logreg")
    if logreg is not None:
        try:
            model = logreg
            # Unwrap CalibratedClassifierCV
            if hasattr(model, "calibrated_classifiers_"):
                model = model.calibrated_classifiers_[0].estimator
            coefs = np.abs(model.coef_[0])
            importance_data["LogReg |coef|"] = dict(zip(FEATURE_COLS[:len(coefs)], coefs))
        except Exception:
            pass

    # XGBoost
    xgb = artifacts.get("xgb")
    if xgb is not None:
        try:
            model = xgb
            if hasattr(model, "calibrated_classifiers_"):
                model = model.calibrated_classifiers_[0].estimator
            imp = model.get_booster().get_score(importance_type="gain")
            importance_data["XGB Gain"] = imp
        except Exception:
            pass

    # LightGBM
    lgbm = artifacts.get("lgbm")
    if lgbm is not None:
        try:
            model = lgbm
            if hasattr(model, "calibrated_classifiers_"):
                model = model.calibrated_classifiers_[0].estimator
            imp = dict(zip(
                [f"f{i}" if model.booster_.feature_name() is None else model.booster_.feature_name()[i]
                 for i in range(len(FEATURE_COLS))],
                model.booster_.feature_importance(importance_type="gain")
            ))
            importance_data["LGBM Gain"] = imp
        except Exception:
            pass

    if not importance_data:
        st.warning("Could not extract feature importances. Re-train models.")
        return

    # Build comparison DataFrame
    rows = []
    for feature in FEATURE_COLS:
        row = {"feature": feature}
        for model_name, imp_dict in importance_data.items():
            row[model_name] = imp_dict.get(feature, imp_dict.get(f"f{FEATURE_COLS.index(feature)}", 0))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Normalize each model's importances to 0-1
    for col in df.columns[1:]:
        max_val = df[col].max()
        if max_val > 0:
            df[col] = df[col] / max_val

    # Sort by average importance
    df["avg"] = df.iloc[:, 1:].mean(axis=1)
    df = df.sort_values("avg", ascending=False).head(20)

    model_cols = [c for c in df.columns if c not in ("feature", "avg")]
    fig = go.Figure()
    colors = {"LogReg |coef|": "#1f77b4", "XGB Gain": "#ff7f0e", "LGBM Gain": "#2ca02c"}
    for col in model_cols:
        fig.add_trace(go.Bar(
            name=col,
            x=df["feature"],
            y=df[col],
            marker_color=colors.get(col),
        ))

    fig.update_layout(
        barmode="group",
        height=450,
        xaxis_tickangle=-45,
        yaxis_title="Normalized Importance",
        legend_title="Model",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_context_comparison(fm: pd.DataFrame):
    st.subheader("Tournament vs Regular Season: Which Stats Matter More?")
    st.markdown(
        "Pearson correlation of each feature with game outcome (team_A_win), "
        "compared between tournament games and regular season games."
    )

    try:
        from scipy import stats as scipy_stats

        avail = [c for c in FEATURE_COLS if c in fm.columns]
        rows = []

        for is_tourney in [True, False]:
            subset = fm[fm["is_tournament"] == is_tourney].copy()
            label = "Tournament" if is_tourney else "Regular Season"

            for col in avail:
                valid = subset[[col, "team_A_win"]].dropna()
                if len(valid) < 30:
                    continue
                r, p = scipy_stats.pearsonr(
                    valid[col].astype(float),
                    valid["team_A_win"].astype(float)
                )
                rows.append({
                    "feature": col,
                    "context": label,
                    "correlation": round(float(r), 4),
                    "abs_correlation": round(abs(float(r)), 4),
                })

        df = pd.DataFrame(rows)

        # Top features by tournament correlation
        tourney_top = (
            df[df["context"] == "Tournament"]
            .sort_values("abs_correlation", ascending=False)
            .head(15)["feature"]
            .tolist()
        )

        plot_df = df[df["feature"].isin(tourney_top)]
        fig = px.bar(
            plot_df,
            x="abs_correlation",
            y="feature",
            color="context",
            orientation="h",
            barmode="group",
            title="Feature Correlation with Win (Top 15 Tournament Features)",
            color_discrete_map={"Tournament": "#d62728", "Regular Season": "#1f77b4"},
        )
        fig.update_layout(
            yaxis={"autorange": "reversed"},
            height=500,
            xaxis_title="|Pearson r| with team_A_win",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        pivot = df.pivot(index="feature", columns="context", values="correlation").reset_index()
        pivot["Diff (Tourney - Regular)"] = (
            pivot.get("Tournament", 0) - pivot.get("Regular Season", 0)
        ).round(4)
        st.dataframe(pivot.sort_values("Diff (Tourney - Regular)", ascending=False),
                     use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Context comparison failed: {e}")


render_feature_analysis()
