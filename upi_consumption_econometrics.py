import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

# Ensure Matplotlib works in headless environments (no GUI) and can write caches.
# This prevents hard aborts in some sandboxed setups.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".mplconfig"))
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


@dataclass(frozen=True)
class ModelSpec:
    name: str
    y: str
    x: List[str]
    upi_var: str
    credit_vars: List[str]


def quarter_str_to_timestamp(quarter_series: pd.Series) -> pd.DatetimeIndex:
    """
    Convert strings like '2016Q2' to a quarterly DatetimeIndex (start of quarter).
    """
    periods = pd.PeriodIndex(quarter_series.astype(str), freq="Q")
    return periods.to_timestamp(how="start")


def safe_log(series: pd.Series) -> pd.Series:
    """
    Log-transform with basic protection against zeros/non-positives.
    Returns NaN where log is undefined so later dropna() removes them cleanly.
    """
    return np.log(series.where(series > 0))


def create_log_growth_features(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    For each numeric column x, compute log-growth: log(x_t) - log(x_{t-1})
    and store as x_g.
    """
    for col in numeric_cols:
        log_x = safe_log(df[col])
        # Naming convention: dependent variable uses `pfce_g` as requested.
        if col == "pfce_crore":
            df["pfce_g"] = log_x.diff()
        else:
            df[f"{col}_g"] = log_x.diff()
    return df


def create_lag_and_share_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create:
      - upi_g_lag1 = upi_value_cr_g shifted by 1-quarter
      - upi_share and its log-growth (and 1-quarter lag)
    """
    df["upi_g_lag1"] = df["upi_value_cr_g"].shift(1)

    denom = df["upi_value_cr"] + df["cc_value_cr"] + df["dc_value_cr"]
    df["upi_share"] = df["upi_value_cr"] / denom

    # Log-growth of share (NaN when share <= 0 by safe_log)
    df["upi_share_g"] = safe_log(df["upi_share"]).diff()
    df["upi_share_g_lag1"] = df["upi_share_g"].shift(1)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Convert `quarter` to a proper datetime index, sort, and set as index.
    Steps 2-4: Create log-growth features, create lag/share features, and drop NaNs.
    """
    df = df.copy()

    # Ensure expected columns are present
    required_cols = [
        "quarter",
        "pfce_crore",
        "upi_value_cr",
        "cc_value_cr",
        "dc_value_cr",
        "ppi_value_cr",
        "cc_outstanding",
        "dc_outstanding",
        "atm_count",
        "pos_count",
        "upi_banks",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric columns
    numeric_cols = [c for c in df.columns if c != "quarter"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Step 1: datetime index
    df["quarter_dt"] = quarter_str_to_timestamp(df["quarter"])
    df = df.sort_values("quarter_dt").set_index("quarter_dt")

    # Step 2: log-growth for every numeric column except quarter
    numeric_cols_no_quarter = [c for c in df.columns if c not in {"quarter", "quarter_dt"}]
    df = create_log_growth_features(df, numeric_cols_no_quarter)

    # Step 3: lag and upi share features
    df = create_lag_and_share_features(df)

    # Step 4: clean - drop all rows with NaNs (from diff/lag/log)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def fit_ols(df: pd.DataFrame, y: str, x_vars: List[str]):
    """
    Fit statsmodels OLS with intercept. Returns the fitted results.
    """
    d = df[[y] + x_vars].dropna()
    if d.empty:
        raise ValueError(f"No usable rows for model with y={y} and x={x_vars}")

    X = sm.add_constant(d[x_vars], has_constant="add")
    y_vec = d[y]
    return sm.OLS(y_vec, X).fit()


def coefficient_table(res) -> pd.DataFrame:
    """
    Coefficients table with coef, p-value, and 95% confidence intervals.
    """
    ci = res.conf_int(alpha=0.05)
    out = pd.DataFrame(
        {
            "coef": res.params,
            "p_value": res.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    return out


def compute_vif(df: pd.DataFrame, x_vars: List[str]) -> pd.DataFrame:
    """
    Compute VIF for each regressor (excluding the intercept).
    """
    d = df[x_vars].dropna()
    if d.shape[0] < (len(x_vars) + 2):
        # Still attempt, but warn via returned NaNs (handled by caller in CSV).
        pass

    X = sm.add_constant(d[x_vars], has_constant="add")
    X_vals = X.values
    vif_rows = []
    # X columns order: const + x_vars
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = variance_inflation_factor(X_vals, i)
        except Exception:
            vif = np.nan
        vif_rows.append({"variable": col, "vif": vif})

    return pd.DataFrame(vif_rows)


def save_residuals_vs_fitted(res, out_path: str) -> None:
    """
    Residuals vs fitted scatter plot.
    """
    fitted = res.fittedvalues
    resid = res.resid

    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, resid, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_upi_interpretation(res, model: ModelSpec, alpha: float = 0.05) -> None:
    """
    Programmatically print:
      - statistical significance of UPI variable (given `alpha`)
      - sign of its coefficient
      - relative importance vs credit variables
    """
    upi_coef = res.params[model.upi_var]
    upi_p = res.pvalues[model.upi_var]
    upi_sig = upi_p < alpha
    upi_sign = "positive" if upi_coef > 0 else "negative" if upi_coef < 0 else "zero"

    credit_abs = []
    for v in model.credit_vars:
        if v in res.params.index:
            credit_abs.append(abs(res.params[v]))
    credit_abs_max = max(credit_abs) if credit_abs else np.nan

    upi_abs = abs(upi_coef)
    if np.isfinite(credit_abs_max) and credit_abs_max > 0:
        importance = upi_abs / credit_abs_max
        importance_str = f"{importance:.3f}x larger than max(|credit coef|)"
    else:
        importance_str = "credit comparison unavailable"

    print(f"\n{model.name}: UPI effect")
    print(f"  Variable: {model.upi_var}")
    print(f"  Coef sign: {upi_sign} (coef={upi_coef:.6f})")
    print(f"  p-value: {upi_p:.6g} -> statistically {'significant' if upi_sig else 'not significant'} at alpha={alpha}")
    print(f"  Relative importance: {importance_str}")


def run_pipeline(data_path: str = "sparshm.csv", results_dir: str = "results") -> None:
    # Load data
    df_raw = pd.read_csv(data_path)

    # Preprocess and build features
    df = preprocess(df_raw)

    # Model specs (max 4 regressors per model, excluding intercept)
    models: List[ModelSpec] = [
        ModelSpec(
            name="model1",
            y="pfce_g",
            x=["upi_g_lag1", "cc_outstanding_g", "pos_count_g", "ppi_value_cr_g"],
            upi_var="upi_g_lag1",
            credit_vars=["cc_outstanding_g"],
        ),
        ModelSpec(
            name="model2",
            y="pfce_g",
            x=["upi_g_lag1", "cc_value_cr_g", "dc_value_cr_g"],
            upi_var="upi_g_lag1",
            credit_vars=["cc_value_cr_g", "dc_value_cr_g"],
        ),
        ModelSpec(
            name="model3",
            y="pfce_g",
            x=["upi_share_g_lag1", "cc_outstanding_g"],
            upi_var="upi_share_g_lag1",
            credit_vars=["cc_outstanding_g"],
        ),
    ]

    os.makedirs(results_dir, exist_ok=True)

    vif_frames: List[pd.DataFrame] = []

    for spec in models:
        res = fit_ols(df, spec.y, spec.x)

        # Outputs (summary)
        summary_path = os.path.join(results_dir, f"{spec.name}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {spec.name}\n")
            f.write(f"Dependent variable: {spec.y}\n")
            f.write(f"Regressors: {spec.x}\n\n")
            f.write(res.summary().as_text())
            f.write("\n\n")
            f.write("Coefficients table (coef, p-value, 95% CI):\n")
            f.write(coefficient_table(res).to_string())
            f.write("\n\n")
            f.write(f"R-squared: {res.rsquared:.6f}\n")
            f.write(f"Adjusted R-squared: {res.rsquared_adj:.6f}\n")
            dw = durbin_watson(res.resid)
            f.write(f"Durbin-Watson: {dw:.6f}\n")

        # Coefficient table saved to logs/plots not required by user; keep summary file as requested.
        # VIF
        vif_df = compute_vif(df, spec.x)
        vif_df.insert(0, "model", spec.name)
        vif_frames.append(vif_df)

        # Residual diagnostics plot
        plot_path = os.path.join(results_dir, f"{spec.name}_residuals_vs_fitted.png")
        save_residuals_vs_fitted(res, plot_path)

        # Print interpretation
        dw = durbin_watson(res.resid)
        print(f"\n{spec.name}: OLS diagnostics")
        print(f"  R-squared: {res.rsquared:.6f}")
        print(f"  Adjusted R-squared: {res.rsquared_adj:.6f}")
        print(f"  Durbin-Watson: {dw:.6f} (approx. 2 suggests no first-order autocorrelation; <2 positive; >2 negative)")

        print_upi_interpretation(res, spec, alpha=0.05)

    vif_all = pd.concat(vif_frames, ignore_index=True) if vif_frames else pd.DataFrame()
    vif_csv_path = os.path.join(results_dir, "vif_tables.csv")
    vif_all.to_csv(vif_csv_path, index=False)

    print(f"\nWrote results to: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    run_pipeline(data_path="sparshm.csv", results_dir="results")

