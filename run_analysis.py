import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_month_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m")
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    return df


def log_pos_growth(series: pd.Series) -> pd.Series:
    """
    Compute ln(x_t) - ln(x_{t-1}) safely.

    If x is <= 0, treat as missing (so ln(x) is undefined).
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)  # avoid log(0) and log(negative)
    return np.log(s).diff()


def make_month_dummies(idx: pd.DatetimeIndex, prefix: str = "m") -> pd.DataFrame:
    month = idx.month
    d = pd.get_dummies(month, prefix=prefix, drop_first=True)
    d.index = idx
    return d


def vif_table(exog: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for each column in exog (expects no intercept column).
    """
    if exog.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "VIF"])

    vifs = []
    x = exog.values
    for i, col in enumerate(exog.columns):
        vifs.append({"feature": col, "VIF": variance_inflation_factor(x, i)})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)


def ols_fit_and_diagnostics(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    *,
    add_const: bool = True,
) -> dict:
    y = df[y_col]
    X = df[x_cols].copy()

    if add_const:
        X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X, missing="drop")
    res = model.fit()

    # Coef table
    conf_int = res.conf_int()
    coef_table = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "t": res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_low": conf_int.iloc[:, 0].values,
            "ci_high": conf_int.iloc[:, 1].values,
        }
    )

    bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = (np.nan,) * 4
    try:
        # Breusch-Pagan wants residuals and exog without the constant handling issues.
        # Using res.resid and X used in fit.
        bp = het_breuschpagan(res.resid, X)
        bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = bp
    except Exception:
        pass

    dw = float(durbin_watson(res.resid))

    # VIF (exclude intercept/const)
    if "const" in coef_table["term"].values:
        x_for_vif = X.drop(columns=["const"], errors="ignore")
    else:
        x_for_vif = X
    vif = vif_table(x_for_vif)

    return {
        "results": res,
        "coef_table": coef_table,
        "fit_stats": {
            "r_squared": float(res.rsquared),
            "adj_r_squared": float(res.rsquared_adj),
            "n_obs": int(res.nobs),
            "durbin_watson": dw,
            "bp_lm_stat": float(bp_lm_stat) if not np.isnan(bp_lm_stat) else np.nan,
            "bp_lm_pvalue": float(bp_lm_pvalue) if not np.isnan(bp_lm_pvalue) else np.nan,
            "bp_f_stat": float(bp_f_stat) if not np.isnan(bp_f_stat) else np.nan,
            "bp_f_pvalue": float(bp_f_pvalue) if not np.isnan(bp_f_pvalue) else np.nan,
        },
        "vif": vif,
    }


def adf_table(df: pd.DataFrame, series_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in series_cols:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) < 10:
            rows.append({"series": col, "adf_stat": np.nan, "p_value": np.nan, "n_obs": len(x)})
            continue
        stat, pval, usedlag, nobs, crit, icbest = adfuller(x, autolag="AIC")
        rows.append(
            {
                "series": col,
                "adf_stat": float(stat),
                "p_value": float(pval),
                "used_lag": int(usedlag),
                "n_obs": int(nobs),
                "crit_1%": float(crit.get("1%")),
                "crit_5%": float(crit.get("5%")),
                "crit_10%": float(crit.get("10%")),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def run_granger(df: pd.DataFrame, x_col: str, y_col: str, *, maxlag: int = 6) -> pd.DataFrame:
    """
    Test whether x_col helps predict y_col (x -> y) in a Granger sense.
    """
    data = df[[y_col, x_col]].dropna()
    if data.shape[0] < 30:
        return pd.DataFrame(columns=["y_predicted_by_x", "maxlag", "lag", "test_stat", "p_value"])

    # statsmodels expects columns in order [y, x]
    test_out = []
    for lag in range(1, maxlag + 1):
        try:
            res = grangercausalitytests(data[[y_col, x_col]], maxlag=lag, verbose=False)
            # res is dict keyed by lag
            # Take the last computed lag's ssr_ftest
            ssr_ftest = res[lag][0]["ssr_ftest"]
            test_stat, p_value = float(ssr_ftest[0]), float(ssr_ftest[1])
            test_out.append(
                {
                    "y_predicted_by_x": f"{y_col} ~ {x_col} (x->y)",
                    "maxlag": lag,
                    "lag": lag,
                    "test_stat": test_stat,
                    "p_value": p_value,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(test_out)


@dataclass
class FeatureConfig:
    use_seasonal_dummies: bool = False
    include_lags: bool = False
    include_infra_controls: bool = False


def build_feature_df(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, str, list[str]]:
    y_col = "gdp_growth"

    # Core payment growth features
    features = ["upi_growth", "cc_growth", "dc_growth", "ppi_growth"]

    if cfg.include_lags:
        features += [f"{c}_lag1" for c in ["upi_growth", "cc_growth", "dc_growth", "ppi_growth"]]

    if cfg.include_infra_controls:
        features += ["atm_growth", "pos_growth", "cc_outstanding_growth", "dc_outstanding_growth"]

    # Validate existence
    missing = [c for c in ([y_col] + features) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    X = df[features].copy()
    if cfg.use_seasonal_dummies:
        month_d = make_month_dummies(df.index, prefix="m")
        X = pd.concat([X, month_d], axis=1)
        features = list(X.columns)

    X[y_col] = df[y_col]
    model_df = X.dropna()
    return model_df, y_col, features


def build_all_growth_feature_df(
    df: pd.DataFrame, original_numeric_cols: list[str]
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Build a model frame that uses ALL available *_growth predictors
    (except gdp_growth) as X and gdp_growth as y.
    """
    y_col = "gdp_growth"
    x_cols = [f"{c}_growth" for c in original_numeric_cols if c != "gdp_crore" and f"{c}_growth" in df.columns]
    model_df = df[[y_col] + x_cols].dropna()
    return model_df, y_col, x_cols


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def standardized_coef_table(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    """
    Run OLS on z-scored y and x for comparable effect magnitudes.
    """
    zdf = pd.DataFrame(index=df.index)
    zdf[y_col] = zscore(df[y_col])
    for c in x_cols:
        zdf[c] = zscore(df[c])
    zdf = zdf.dropna()
    if zdf.empty or len(x_cols) == 0:
        return pd.DataFrame(columns=["term", "std_beta", "std_err", "t", "p_value", "ci_low", "ci_high", "abs_std_beta"])
    fit = ols_fit_and_diagnostics(zdf, y_col, x_cols)
    out = fit["coef_table"].copy()
    out = out[out["term"] != "const"].copy()
    out = out.rename(columns={"coef": "std_beta"})
    out["abs_std_beta"] = out["std_beta"].abs()
    return out.sort_values("abs_std_beta", ascending=False)


def pretty_factor_name(term: str) -> str:
    if term.endswith("_growth"):
        return term[: -len("_growth")]
    return term


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="oooo.csv", help="Path to oooo.csv")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--seasonal-dummies", action="store_true", help="Include month dummies in OLS/XGB")
    parser.add_argument("--infra-controls", action="store_true", help="Add ATM/POS/outstanding controls")
    parser.add_argument("--max-granger-lag", type=int, default=6)
    parser.add_argument("--xgb-test-last-months", type=int, default=18, help="Test size (last N rows)")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    input_path = args.input
    results_dir = args.results_dir
    _ensure_dir(results_dir)
    _ensure_dir(os.path.join(results_dir, "plots"))
    # Avoid Matplotlib writing into user/system locations that may not be writable.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(results_dir, ".mplconfig"))
    _ensure_dir(os.environ["MPLCONFIGDIR"])

    raw = pd.read_csv(input_path)
    df = parse_month_index(raw, "date")

    # Clean up any unnamed/blank columns from CSV export artifacts.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)].copy()
    df = df.loc[:, df.columns.astype(str).str.strip() != ""].copy()

    # Keep raw columns for traceability
    required_raw = [
        "gdp_crore",
        "upi_value_cr",
        "cc_value_cr",
        "dc_value_cr",
        "ppi_value_cr",
        "atm_count",
        "pos_count",
        "cc_outstanding",
        "dc_outstanding",
    ]
    missing_raw = [c for c in required_raw if c not in df.columns]
    if missing_raw:
        raise ValueError(f"CSV is missing columns: {missing_raw}")

    # GDP missing handling (per prompt: drop months where GDP is missing)
    # NOTE: With log-growth (diff), missing GDP also affects the next month (needs lag).
    df = df[df["gdp_crore"].notna()].copy()
    # Capture numeric columns from original CSV (before adding derived growth/lag features).
    original_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Compute log-growth for target and main predictors
    df["gdp_growth"] = log_pos_growth(df["gdp_crore"])
    df["upi_growth"] = log_pos_growth(df["upi_value_cr"])
    df["cc_growth"] = log_pos_growth(df["cc_value_cr"])
    df["dc_growth"] = log_pos_growth(df["dc_value_cr"])
    df["ppi_growth"] = log_pos_growth(df["ppi_value_cr"])

    # Infra controls (optional)
    df["atm_growth"] = log_pos_growth(df["atm_count"])
    df["pos_growth"] = log_pos_growth(df["pos_count"])
    df["cc_outstanding_growth"] = log_pos_growth(df["cc_outstanding"])
    df["dc_outstanding_growth"] = log_pos_growth(df["dc_outstanding"])
    df["upi_banks_growth"] = log_pos_growth(df["upi_banks"])

    # Lags (optional)
    for c in ["upi_growth", "cc_growth", "dc_growth", "ppi_growth"]:
        df[f"{c}_lag1"] = df[c].shift(1)

    # Also compute growth for every numeric raw column (except target) so all
    # available X variables can be tested in one model.
    all_candidate_cols = [c for c in original_numeric_cols if c != "gdp_crore"]
    for c in all_candidate_cols:
        gcol = f"{c}_growth"
        if gcol not in df.columns:
            df[gcol] = log_pos_growth(df[c])

    # Stationarity tests on growth series
    growth_series_cols = ["gdp_growth", "upi_growth", "cc_growth", "dc_growth", "ppi_growth"]
    adf = adf_table(df, growth_series_cols)
    adf.to_csv(os.path.join(results_dir, "adf_growth_series.csv"), index=False)

    # Save cleaned growth dataset (for transparency)
    growth_out_cols = ["gdp_growth", "upi_growth", "cc_growth", "dc_growth", "ppi_growth",
                        "atm_growth", "pos_growth", "cc_outstanding_growth", "dc_outstanding_growth"]
    cleaned = df[growth_out_cols].copy()
    cleaned.to_csv(os.path.join(results_dir, "growth_data_cleaned.csv"), index=True)

    # ========== OLS Baseline ==========
    baseline_cfg = FeatureConfig(
        use_seasonal_dummies=bool(args.seasonal_dummies),
        include_lags=False,
        include_infra_controls=bool(args.infra_controls),
    )
    baseline_df, y_col, used_features = build_feature_df(df, baseline_cfg)

    baseline_x_cols = [c for c in used_features if c != y_col]
    baseline_fit = ols_fit_and_diagnostics(baseline_df, y_col, baseline_x_cols)
    baseline_fit["coef_table"].to_csv(os.path.join(results_dir, "ols_baseline_coef_table.csv"), index=False)
    pd.DataFrame([baseline_fit["fit_stats"]]).to_csv(
        os.path.join(results_dir, "ols_baseline_fit_stats.csv"), index=False
    )
    baseline_fit["vif"].to_csv(os.path.join(results_dir, "ols_baseline_vif.csv"), index=False)

    # ========== OLS With Lagged Effects ==========
    lag_cfg = FeatureConfig(
        use_seasonal_dummies=bool(args.seasonal_dummies),
        include_lags=True,
        include_infra_controls=bool(args.infra_controls),
    )
    lag_df, y_col2, used_features2 = build_feature_df(df, lag_cfg)
    lag_x_cols = [c for c in used_features2 if c != y_col2]
    lag_fit = ols_fit_and_diagnostics(lag_df, y_col2, lag_x_cols)
    lag_fit["coef_table"].to_csv(os.path.join(results_dir, "ols_lag_effects_coef_table.csv"), index=False)
    pd.DataFrame([lag_fit["fit_stats"]]).to_csv(os.path.join(results_dir, "ols_lag_effects_fit_stats.csv"), index=False)
    lag_fit["vif"].to_csv(os.path.join(results_dir, "ols_lag_effects_vif.csv"), index=False)

    # ========== Subsample Analysis (split at 2019) ==========
    subs = []
    split_date = pd.to_datetime("2019-12-01")
    # Keep only what baseline config uses (for interpretability)
    for name, subdf in [
        ("pre_2020", baseline_df[baseline_df.index <= split_date]),
        ("post_2020", baseline_df[baseline_df.index > split_date]),
    ]:
        if subdf.shape[0] < 20:
            continue
        x_cols = [c for c in baseline_x_cols if c in subdf.columns]
        try:
            fit = ols_fit_and_diagnostics(subdf, y_col, x_cols)
            fit_rows = fit["coef_table"].copy()
            fit_rows["subsample"] = name
            subs.append(fit_rows)
        except Exception:
            continue

    if subs:
        pd.concat(subs, ignore_index=True).to_csv(os.path.join(results_dir, "ols_subsample_coef_table.csv"), index=False)

    # ========== OLS All-Factor Model (all numeric X vars from CSV) ==========
    factor_impact = None
    all_df, all_y, all_x = build_all_growth_feature_df(df, original_numeric_cols)
    if len(all_x) > 0:
        all_fit = ols_fit_and_diagnostics(all_df, all_y, all_x)
        all_fit["coef_table"].to_csv(os.path.join(results_dir, "ols_all_factors_coef_table.csv"), index=False)
        pd.DataFrame([all_fit["fit_stats"]]).to_csv(
            os.path.join(results_dir, "ols_all_factors_fit_stats.csv"), index=False
        )
        all_fit["vif"].to_csv(os.path.join(results_dir, "ols_all_factors_vif.csv"), index=False)

        # Standardized coefficients make "which factor affects GDP most" easier to compare.
        std_table = standardized_coef_table(all_df, all_y, all_x)
        std_table.to_csv(os.path.join(results_dir, "ols_all_factors_standardized_effects.csv"), index=False)
        if not std_table.empty:
            factor_impact = std_table.copy()
            factor_impact["factor"] = factor_impact["term"].map(pretty_factor_name)
            factor_impact["direction"] = np.where(factor_impact["std_beta"] >= 0, "positive", "negative")
            factor_impact["significant_5pct"] = factor_impact["p_value"] < 0.05
            factor_impact = factor_impact[
                ["factor", "term", "std_beta", "abs_std_beta", "direction", "p_value", "significant_5pct"]
            ].sort_values("abs_std_beta", ascending=False)
            factor_impact.to_csv(os.path.join(results_dir, "factor_impact_ranked.csv"), index=False)

    # ========== Granger Causality (UPI -> GDP) ==========
    granger_df = run_granger(df, x_col="upi_growth", y_col="gdp_growth", maxlag=args.max_granger_lag)
    granger_df.to_csv(os.path.join(results_dir, "granger_upi_to_gdp.csv"), index=False)

    # ========== XGBoost + SHAP ==========
    # Imports here so the OLS can still run even if xgboost/shap are missing.
    import shap
    import matplotlib.pyplot as plt

    if factor_impact is not None and not factor_impact.empty:
        top_n = min(12, len(factor_impact))
        top = factor_impact.head(top_n).copy()
        colors = top["direction"].map({"positive": "#2ca02c", "negative": "#d62728"}).tolist()
        plt.figure(figsize=(9, max(4, 0.45 * top_n)))
        plt.barh(top["factor"], top["std_beta"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.gca().invert_yaxis()
        plt.title("Ranked Factor Impact on GDP Growth (Standardized Betas)")
        plt.xlabel("Standardized Coefficient (beta)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plots", "factor_impact_ranked.png"), dpi=220)
        plt.close()

    # Align features with baseline lag cfg (we'll use lagged features if requested)
    xgb_cfg = FeatureConfig(
        use_seasonal_dummies=bool(args.seasonal_dummies),
        include_lags=True,  # typically helpful for ML and specified in robustness checks
        include_infra_controls=bool(args.infra_controls),
    )
    xgb_df, y_col_xgb, used_features_xgb = build_feature_df(df, xgb_cfg)

    X = xgb_df[used_features_xgb].copy()
    # build_feature_df returns model_df with y_col included in index-based dropna;
    # remove target column if present (defensive)
    X = X.drop(columns=[y_col_xgb], errors="ignore")
    y = xgb_df[y_col_xgb].copy()

    # Time-based split: last N rows as test
    if args.xgb_test_last_months >= len(xgb_df):
        test_size = max(6, int(len(xgb_df) * 0.2))
    else:
        test_size = args.xgb_test_last_months
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    model = None
    model_used = None
    model_fit_error = None
    try:
        from xgboost import XGBRegressor

        model_used = "xgboost"
        model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=args.random_seed,
            objective="reg:squarederror",
            tree_method="hist",
        )
        model.fit(X_train, y_train)
    except Exception as e:
        model_fit_error = repr(e)
        # Fallback: tree-based sklearn model that SHAP can explain.
        # This is specifically to handle environments where xgboost's native library
        # cannot load (e.g., missing OpenMP runtime).
        from sklearn.ensemble import RandomForestRegressor

        model_used = "random_forest_fallback"
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=args.random_seed,
            n_jobs=1,  # reduce dependency on external threading/OpenMP runtimes
            max_depth=6,
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "model_used": model_used,
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "xgboost_import_or_load_error": model_fit_error,
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, f"ml_test_metrics_{model_used}.csv"), index=False
    )

    # SHAP
    # For speed, subsample the data used to compute SHAP values.
    shap_max_samples = 150
    if len(X_test) > shap_max_samples:
        shap_idx = np.random.default_rng(args.random_seed).choice(len(X_test), size=shap_max_samples, replace=False)
        X_shap = X_test.iloc[shap_idx]
    else:
        X_shap = X_test

    explainer = shap.Explainer(model, X_train, feature_names=list(X.columns))
    shap_values = explainer(X_shap)

    # SHAP summary (beeswarm)
    shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "plots", f"ml_shap_summary_beeswarm_{model_used}.png"),
        dpi=200,
    )
    plt.close()

    # SHAP bar plot
    shap.summary_plot(shap_values, X_shap, show=False, plot_type="bar", max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", f"ml_shap_summary_bar_{model_used}.png"), dpi=200)
    plt.close()

    # Export top features by mean(|SHAP|)
    try:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.DataFrame(
            {"feature": shap_values.feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(
            os.path.join(results_dir, f"ml_shap_feature_importance_{model_used}.csv"), index=False
        )
    except Exception:
        pass

    # ========== Elasticity interpretation helper ==========
    # For log-difference growth rates: elasticity approx equals coefficient in growth equation.
    try:
        baseline_coef = baseline_fit["coef_table"].set_index("term").loc["upi_growth", "coef"]
        baseline_p = baseline_fit["coef_table"].set_index("term").loc["upi_growth", "p_value"]
        elasticity_interpret = {"upi_elasticity_beta": float(baseline_coef), "upi_elasticity_p_value": float(baseline_p)}
        pd.DataFrame([elasticity_interpret]).to_csv(os.path.join(results_dir, "elasticity_interpretation.csv"), index=False)
    except Exception:
        pass

    print("Analysis complete. Outputs written to:", os.path.abspath(results_dir))


if __name__ == "__main__":
    main()

