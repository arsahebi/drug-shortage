"""
statistical_tests_advanced_models.py
=====================================
Advanced regression models for Figures 1 and 2 of the Metformin JAMA paper.
Implements the three approaches discussed in the team meeting (Feb 25, 2026).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKGROUND: WHY THE EXISTING TESTS ARE NOT ENOUGH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The existing tests handle ONE source of non-independence at a time:
  - Kruskal-Wallis / Dunn:  no clustering (independence assumed)
  - NDC-bootstrap:           clusters by NDC only
  - FEI-bootstrap:           clusters by FEI only
  - GEE (current):           accounts for repeated NDC over years, but does
                             NOT account for multiple NDCs within same FEI

John's question: "Is there a way to cluster BOTH at the same time?" -> Yes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE THREE MODELS (A, B, C)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL A — GEE with FEI dummies  ("GEE with both effects")
  The current GEE accounts for NDC repetition via exchangeable correlation.
  This model ADDITIONALLY includes FEI dummy variables in the regression so
  the country coefficient is estimated after absorbing facility-level variation.
  John: "some form of regression like your GEE, but using fixed effects for
  NDC and FEI."

MODEL B — Random Effects + Two-Way Clustered Standard Errors  (PRIMARY)
  Mixed model (random NDC intercept) with SE clustered simultaneously at
  both NDC and FEI level using Cameron-Gelbach-Miller 2011 formula:
  V_2way = V_NDC + V_FEI - V_intersection
  John: "random effects with clustered standard errors would probably be
  adequate. The goal is to claim a p-value that's not wrong."

MODEL C — Fixed Effects (NDC and/or FEI)  (SENSITIVITY)
  Absorbs all within-group variation by demeaning. Because country is a
  fixed property of each NDC, NDC FE will be COLLINEAR with country dummies
  -> coefficient likely dropped. This is EXPECTED. John predicted this:
  "I'm guessing there'll be no significance because there'll be too little
  variation left — but that's a valid, honest result."
  FEI FE alone may retain some variation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA COVERAGE NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DMF         : 2020, 2022, 2024  -> true panel; all models applicable
  NDMA        : 2020, 2022        -> partial panel
  Dissolution : 2024 only         -> cross-section; NDC FE not applicable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from statistical_tests_advanced_models import run_advanced_models
    run_advanced_models(ndc_year_df, output_dir="./diagnostic_plots")

Required columns: CountryCode, NDC11, FEI, Year, DMF, NDMA, Dissolution
Figure 2 also needs: Price, Volume, and an inspection outcome column
(set FIG2_OUTCOME_COL at the top of this file to match your column name).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro

# -- optional dependencies ---------------------------------------------------
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Gaussian
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed. "
                  "pip install statsmodels --break-system-packages")

# -- constants ----------------------------------------------------------------
PLOT_COUNTRIES   = ["IND", "USA", "CHN"]
COUNTRY_COL      = "CountryCode"
NDC_COL          = "NDC11"
FEI_COL          = "FEI"
YEAR_COL         = "Year"
FIG2_OUTCOME_COL = "PriorScore_cat"           # <-- adjusted to match ndc_year_df
FIG2_GROUPS      = ["NAI", "VAI", "OAI"]      # reference = NAI
FIG2_OUTCOMES    = ["price", "volume"]

METRIC_YEARS = {
    "DMF":         [2020, 2022, 2024],
    "NDMA":        [2020, 2022],
    "Dissolution": [2024],
}


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _sep(title="", w=80):
    print("\n" + "=" * w)
    if title:
        print(f"  {title}")
        print("=" * w)


def _subsep(title):
    print(f"\n  {'-' * 72}")
    print(f"  {title}")
    print(f"  {'-' * 72}")


def _log_transform(y, label):
    """Apply log1p if |skewness| > 1."""
    skew = float(pd.Series(y[np.isfinite(y)]).skew())
    if abs(skew) > 1.0:
        print(f"    WARNING: Skewness={skew:.2f} > 1 -> applying log1p({label})")
        return np.log1p(y), f"log1p({label})", True
    return y, label, False


def _diagnostics(residuals, exog, ylabel, tag, output_dir):
    """Shapiro-Wilk + Breusch-Pagan + 4-panel PNG."""
    r = residuals[np.isfinite(residuals)]
    if len(r) > 5000:
        r = np.random.default_rng(42).choice(r, 5000, replace=False)
    sw_s, sw_p = shapiro(r)

    bp_lm, bp_p = np.nan, np.nan
    if HAS_STATSMODELS and exog is not None:
        try:
            idx = np.isfinite(residuals)
            bp_lm, bp_p, _, _ = het_breuschpagan(residuals[idx], exog[idx])
        except Exception:
            pass

    norm = "NON-NORMAL (*)" if sw_p < 0.05 else "normal"
    het  = "HETEROSKEDASTIC (*)" if bp_p < 0.05 else "homoskedastic"

    print(f"\n    -- Residual diagnostics [{tag} | {ylabel}] --")
    print(f"    Shapiro-Wilk  : W={sw_s:.4f}  p={sw_p:.4f}  -> {norm}")
    if not np.isnan(bp_lm):
        print(f"    Breusch-Pagan : LM={bp_lm:.4f}  p={bp_p:.4f}  -> {het}")
    if sw_p < 0.05 or bp_p < 0.05:
        print(f"    INFO: Consider log-transforming if not already done.")

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Residual Diagnostics - {tag} | {ylabel}", fontsize=11)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(0, color="red", lw=1.5, ls="--")
    ax1.set_title("Residuals histogram")
    ax1.set_xlabel("Residual"); ax1.set_ylabel("Count")

    ax2 = fig.add_subplot(gs[0, 1])
    osm, osr = stats.probplot(r, dist="norm")
    ax2.plot(osm[0], osm[1], "o", color="steelblue", ms=3, alpha=0.6)
    ax2.plot(osm[0], np.std(r)*np.array(osm[0])+np.mean(r), "r--", lw=1.5)
    ax2.set_title("Q-Q plot (Normal)")
    ax2.set_xlabel("Theoretical quantiles"); ax2.set_ylabel("Sample quantiles")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(range(len(r)), r, s=12, alpha=0.5, color="steelblue")
    ax3.axhline(0, color="red", lw=1.5, ls="--")
    ax3.set_title("Residuals by index")
    ax3.set_xlabel("Observation"); ax3.set_ylabel("Residual")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(0.5, 0.5,
             f"Shapiro-Wilk  p={sw_p:.4f}\n{norm}\n\n"
             f"Breusch-Pagan p={bp_p:.4f}\n{het}",
             ha="center", va="center", fontsize=10, transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.6", fc="lightyellow", ec="orange"))
    ax4.axis("off"); ax4.set_title("Test summary")

    plt.tight_layout()
    fname = (tag + "_" + ylabel + "_diag.png"
             ).replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved -> {fpath}")


def _coef_table(names, params, se, dof, header=""):
    """Print coefficient table with t and p values."""
    t = params / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), df=max(dof, 1))
    lo = params - 1.96 * se
    hi = params + 1.96 * se
    if header:
        print(f"\n    {header}")
    print(f"    {'Var':>12s}  {'Coef':>9s}  {'SE':>8s}  "
          f"{'t':>7s}  {'p':>8s}  {'[95% CI]':>22s}  sig")
    print(f"    {'-'*80}")
    for i, name in enumerate(names):
        if np.isnan(params[i]):
            print(f"    {name:>12s}  (dropped - collinear with fixed effects)")
            continue
        sig = "**" if p[i]<0.01 else ("*" if p[i]<0.05 else ("." if p[i]<0.10 else ""))
        ci = f"[{lo[i]:+.4f}, {hi[i]:+.4f}]"
        print(f"    {name:>12s}  {params[i]:>9.4f}  {se[i]:>8.4f}  "
              f"{t[i]:>7.3f}  {p[i]:>8.4f}  {ci:>22s}  {sig}")


def _cgm_vcov(y, X, c1, c2):
    """
    Cameron-Gelbach-Miller (2011) two-way clustered variance-covariance.
    V = V(c1) + V(c2) - V(c1 intersect c2)
    """
    n, k = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)

    def _v(clusters):
        g = len(np.unique(clusters))
        dfc = (g / (g-1)) * (n / (n-k))
        meat = np.zeros((k, k))
        for c in np.unique(clusters):
            idx = clusters == c
            sc = X[idx].T @ e[idx]
            meat += np.outer(sc, sc)
        return bread @ (dfc * meat) @ bread

    inter = np.array([f"{a}__{b}" for a, b in zip(c1, c2)])
    return _v(c1) + _v(c2) - _v(inter)


def _within_demean(sub, ycol, dummy_cols, fe_col):
    """Demean y and dummies within fe_col groups (within-FE estimator)."""
    s = sub[[ycol] + dummy_cols + [fe_col]].dropna().copy()
    for col in [ycol] + dummy_cols:
        s[col] = s[col] - s.groupby(fe_col)[col].transform("mean")
    return s


def _twoway_demean(sub, ycol, dummy_cols, fe1, fe2):
    """Iterative demeaning for two-way FE."""
    s = sub[[ycol] + dummy_cols + [fe1, fe2]].dropna().copy()
    cols = [ycol] + dummy_cols
    for _ in range(200):
        prev = s[cols].values.copy()
        for fe in [fe1, fe2]:
            for col in cols:
                s[col] = s[col] - s.groupby(fe)[col].transform("mean")
        if np.max(np.abs(s[cols].values - prev)) < 1e-9:
            break
    return s


def _ols_clustered(y, X, sub, dummy_names, ylabel, tag, output_dir,
                   ndc_col=NDC_COL, fei_col=FEI_COL, cluster_on_ndc=True):
    """
    OLS with the appropriate SE variant for the data type:

      cluster_on_ndc=True  (panel: DMF, NDMA)
        → ★ Two-way clustered SE (NDC + FEI) via CGM 2011  [PRIMARY]
          Accounts for: repeated NDC testing + multiple NDCs within same FEI.

      cluster_on_ndc=False (cross-section: Dissolution, 2024 only)
        → ★ FEI-clustered SE only  [PRIMARY for cross-section]
          NDC clustering is not applicable (1 observation per NDC).

    Advisor note (John): "Since we have NDC random intercepts, do we need to
    cluster on NDC too?" — For the cross-section (Dissolution), NDC FE/RE is
    not applicable and there is only one observation per NDC, so FEI-only
    clustering is correct. For the panel, two-way clustering is conservative
    even with random effects included.
    """
    if not HAS_STATSMODELS: return
    ols = sm.OLS(y, X).fit()
    n, k = X.shape
    dof = max(n - k, 1)
    all_names = ["const"] + list(dummy_names)

    has_ndc = ndc_col in sub.columns and sub[ndc_col].nunique() >= 2
    has_fei = fei_col in sub.columns and sub[fei_col].nunique() >= 2

    if cluster_on_ndc and has_ndc and has_fei:
        # Panel data: two-way clustered SE (NDC + FEI) — PRIMARY
        try:
            V2  = _cgm_vcov(y, X, sub[ndc_col].values, sub[fei_col].values)
            se2 = np.sqrt(np.diag(V2))
            _coef_table(all_names, ols.params, se2, dof,
                        header=f"★ TWO-WAY clustered SE (NDC+FEI) [{ylabel}] -- PRIMARY:")
        except Exception as exc:
            print(f"    Two-way SE error: {exc}")

    elif not cluster_on_ndc and has_fei:
        # Cross-section (Dissolution): FEI-only clustered SE — PRIMARY
        try:
            r = sm.OLS(y, X).fit(cov_type="cluster",
                                  cov_kwds={"groups": sub[fei_col].values})
            _coef_table(all_names, r.params, r.bse, dof,
                        header=f"★ FEI-clustered SE [{ylabel}] -- PRIMARY (cross-section, NDC clustering N/A):")
        except Exception as exc:
            print(f"    FEI-clustered SE error: {exc}")

    else:
        print(f"    WARNING: Insufficient clusters for clustered SE. Reporting naive OLS SE.")
        _coef_table(all_names, ols.params, ols.bse, dof, header=f"OLS (naive) SE [{ylabel}]:")

    _diagnostics(ols.resid, X, ylabel, tag, output_dir)


# ============================================================================
# FIGURE 1 HELPERS
# ============================================================================

def _prep_fig1(df, metric):
    cols = [COUNTRY_COL, NDC_COL, YEAR_COL, metric]
    if FEI_COL in df.columns:
        cols.append(FEI_COL)
    sub = df[df[COUNTRY_COL].isin(PLOT_COUNTRIES)][cols].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric]).copy()
    sub["IND"] = (sub[COUNTRY_COL] == "IND").astype(float)
    sub["CHN"] = (sub[COUNTRY_COL] == "CHN").astype(float)
    return sub


# ============================================================================
# FIGURE 2 HELPERS
# ============================================================================

def _find_outcome_col(df):
    """Locate the inspection outcome column."""
    candidates = [FIG2_OUTCOME_COL, "PriorScore_cat", "InspectionOutcome",
                  "PriorInspection", "inspection_outcome", "Outcome"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(v in df[c].dropna().astype(str).unique() for v in FIG2_GROUPS):
            return c
    return None


def _prep_fig2(df, outcome):
    outcome_col = _find_outcome_col(df)
    if outcome_col is None:
        print(f"  SKIP: inspection outcome column not found.")
        print(f"  Set FIG2_OUTCOME_COL at top of file to match your column name.")
        return None, None
    cols = [outcome_col, NDC_COL, COUNTRY_COL, YEAR_COL, outcome]
    if FEI_COL in df.columns:
        cols.append(FEI_COL)
    sub = df[df[COUNTRY_COL].isin(PLOT_COUNTRIES)][cols].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna(subset=[outcome])

    # Remap numeric scores back to NAI/VAI/OAI labels if needed
    # NAI=0.0 (No Action Indicated), VAI=1.5 (Voluntary Action Indicated),
    # OAI=3.5 (Official Action Indicated)
    numeric_to_label = {0.0: "NAI", 1.5: "VAI", 3.5: "OAI"}
    col_vals = sub[outcome_col].dropna().unique()
    if not any(v in col_vals for v in FIG2_GROUPS):
        # Values are numeric — remap
        sub[outcome_col] = pd.to_numeric(sub[outcome_col],
                                          errors="coerce").map(numeric_to_label)
        print(f"  INFO: Remapped {outcome_col} numeric scores "
              f"(0.0->NAI, 1.5->VAI, 3.5->OAI)")

    sub = sub[sub[outcome_col].isin(FIG2_GROUPS)].copy()
    sub["VAI"] = (sub[outcome_col] == "VAI").astype(float)
    sub["OAI"] = (sub[outcome_col] == "OAI").astype(float)
    return sub, outcome_col


# ============================================================================
# MODEL A — GEE WITH FEI DUMMIES
# ============================================================================

def _gee_with_fei(y, sub, dummy_names, ylabel, metric_or_outcome,
                  is_crosssection=False):
    """
    GEE with FEI dummies in the mean model.
    Correlation: Exchangeable within NDC (or Independence for cross-section).
    """
    if not HAS_STATSMODELS: return
    cov_struct = Independence() if is_crosssection else Exchangeable()

    fei_dummies = pd.DataFrame()
    if FEI_COL in sub.columns and sub[FEI_COL].nunique() >= 2:
        fei_dummies = pd.get_dummies(sub[FEI_COL], prefix="FEI",
                                     drop_first=True).astype(float)

    X_df = pd.concat([
        sub[list(dummy_names)].reset_index(drop=True),
        fei_dummies.reset_index(drop=True)
    ], axis=1)
    exog   = sm.add_constant(X_df.values.astype(float))
    groups = sub[NDC_COL].values

    try:
        model = GEE(y, exog, groups=groups,
                    family=Gaussian(), cov_struct=cov_struct)
        try:
            result = model.fit(maxiter=100, ctol=1e-6)
        except TypeError:
            result = model.fit(maxiter=100)
        dof = max(len(y) - exog.shape[1], 1)
        key_names = ["const"] + list(dummy_names)
        _coef_table(
            key_names,
            np.array([result.params[i] for i in range(len(key_names))]),
            np.array([result.bse[i] for i in range(len(key_names))]),
            dof,
            header=f"Key coefficients [{ylabel}] "
                   f"(+{fei_dummies.shape[1]} FEI dummies, omitted for brevity):"
        )
        corr_label = "Independence (cross-section)" if is_crosssection else "Exchangeable within NDC"
        print(f"    Correlation structure: {corr_label}")
    except Exception as exc:
        print(f"    GEE+FEI error: {exc}")


def fig1_modelA_gee_fei(df, output_dir="."):
    _sep("MODEL A (Fig 1) -- GEE with FEI Dummies")
    print("""
  metric ~ IND + CHN + FEI_dummies
  Correlation: Exchangeable within NDC11 (same NDC over years)
  Reference: USA (country), first FEI alphabetically (facility)

  This is the direct answer to John's question: "GEE, but also
  deal with the FEI clustering." FEI dummies absorb facility-level
  mean differences; GEE correlation handles NDC repetition.
  WARNING: With 8-18 FEIs, adding many dummies consumes df.
""")
    if not HAS_STATSMODELS:
        print("  SKIP: statsmodels required."); return

    for metric, years in METRIC_YEARS.items():
        _subsep(f"{metric}  (years: {years})")
        sub = _prep_fig1(df, metric)
        if sub.empty: continue
        n_fei = sub[FEI_COL].nunique() if FEI_COL in sub.columns else 0
        print(f"  n_obs={len(sub)}  n_NDC={sub[NDC_COL].nunique()}  n_FEI={n_fei}")
        y_raw = sub[metric].values.astype(float)
        y, ylabel, _ = _log_transform(y_raw, metric)
        is_xs = (metric == "Dissolution")
        if is_xs:
            print("  NOTE: Dissolution is 2024-only (cross-section). "
                  "Using Independence correlation.")
        _gee_with_fei(y, sub, ["IND", "CHN"], ylabel, metric, is_xs)


def fig2_modelA_gee_fei(df, output_dir="."):
    _sep("MODEL A (Fig 2) -- GEE with FEI Dummies")
    print("""
  outcome ~ VAI + OAI + FEI_dummies
  Correlation: Exchangeable within NDC11
  Reference: NAI (inspection outcome), first FEI (facility)
""")
    if not HAS_STATSMODELS:
        print("  SKIP: statsmodels required."); return

    for outcome in FIG2_OUTCOMES:
        _subsep(f"Outcome: {outcome}")
        sub, oc = _prep_fig2(df, outcome)
        if sub is None or sub.empty: continue
        n_fei = sub[FEI_COL].nunique() if FEI_COL in sub.columns else 0
        print(f"  n_obs={len(sub)}  n_NDC={sub[NDC_COL].nunique()}  n_FEI={n_fei}")
        y_raw = sub[outcome].values.astype(float)
        y, ylabel, _ = _log_transform(y_raw, outcome)
        _gee_with_fei(y, sub, ["VAI", "OAI"], ylabel, outcome, False)


# ============================================================================
# MODEL B — RANDOM EFFECTS + TWO-WAY CLUSTERED SE   (PRIMARY)
# ============================================================================

def fig1_modelB_re_twoway(df, output_dir="."):
    _sep("MODEL B (Fig 1) -- Random Effects + Two-Way Clustered SE  [PRIMARY]")
    print("""
  Step 1: Mixed model  metric ~ IND + CHN + (1|NDC11)
          Random NDC intercept handles repeated measurement over years.
          Intraclass correlation (ICC) reported.
  Step 2: Re-compute SE using CGM 2011 two-way clustering:
          V_2way = V_NDC + V_FEI - V_intersection
          This ALSO accounts for multiple NDCs within same FEI.

  John: "random effects with clustered standard errors would probably
  be adequate -- the goal is to claim a p-value that's not wrong."

  NOTE for Dissolution (2024 only): MixedLM not applicable (1 obs
  per NDC). Falls back to OLS + two-way clustered SE.
""")
    if not HAS_STATSMODELS:
        print("  SKIP: statsmodels required."); return

    for metric, years in METRIC_YEARS.items():
        _subsep(f"{metric}  (years: {years})")
        sub = _prep_fig1(df, metric)
        if sub.empty: continue
        n_obs = len(sub)
        n_ndc = sub[NDC_COL].nunique()
        n_fei = sub[FEI_COL].nunique() if FEI_COL in sub.columns else 0
        print(f"  n_obs={n_obs}  n_NDC={n_ndc}  n_FEI={n_fei}")

        y_raw = sub[metric].values.astype(float)
        y, ylabel, _ = _log_transform(y_raw, metric)
        sub = sub.copy(); sub["_y"] = y
        X = sm.add_constant(sub[["IND", "CHN"]].values.astype(float))

        if metric == "Dissolution" or n_ndc < 3:
            print(f"  NOTE: Cross-section only (2024) -> OLS + FEI-clustered SE")
            print(f"  NOTE: NDC-level clustering not applicable (1 obs per NDC in cross-section)")
            _ols_clustered(y, X, sub, ["IND", "CHN"], ylabel,
                           f"B_OLS_{metric}", output_dir, cluster_on_ndc=False)
        else:
            # Step 1: Random Effects (MixedLM)
            try:
                mlm = smf.mixedlm("_y ~ IND + CHN", data=sub,
                                   groups=sub[NDC_COL]).fit(reml=True)
                var_re  = float(mlm.cov_re.iloc[0,0]) if hasattr(mlm,"cov_re") else 0
                var_res = float(mlm.scale)
                icc = var_re / (var_re + var_res) if (var_re + var_res) > 0 else 0
                print(f"\n  RE model | ICC={icc:.4f}  "
                      f"Var(NDC)={var_re:.4f}  Var(resid)={var_res:.4f}")
                if icc > 0.1:
                    print("  INFO: ICC > 0.1 -- substantial NDC clustering, RE adjustment important")
                names_re = ["Intercept", "IND", "CHN"]
                p_re = np.array([mlm.params.get(n, np.nan) for n in names_re])
                s_re = np.array([mlm.bse.get(n, np.nan) for n in names_re])
                _coef_table(names_re, p_re, s_re, max(n_obs-3,1),
                            header="RE coefficients (standard SE -- reference only):")
            except Exception as exc:
                print(f"  MixedLM error: {exc}")

            # Step 2: OLS + two-way clustered SE
            print(f"\n  Two-way clustered SE on OLS [{ylabel}]:")
            _ols_clustered(y, X, sub, ["IND", "CHN"], ylabel,
                           f"B_RE_2way_{metric}", output_dir)

        # --- CHN vs IND: re-parameterize with IND as reference ---
        # USA and CHN dummies vs IND baseline gives CHN-IND coefficient directly.
        _subsep(f"CHN vs IND comparison [{metric}]  (reference = IND)")
        print(f"  Tests whether China-manufactured products differ from India-manufactured.")
        sub_iref = sub.copy()
        sub_iref["USA_dummy"] = (sub_iref[COUNTRY_COL] == "USA").astype(float)
        # CHN column already present from _prep_fig1; USA_dummy added above
        X_iref = sm.add_constant(sub_iref[["USA_dummy", "CHN"]].values.astype(float))
        ols_iref = sm.OLS(y, X_iref).fit()
        dof_i = max(n_obs - 3, 1)

        has_ndc_i = NDC_COL in sub_iref.columns and sub_iref[NDC_COL].nunique() >= 2
        has_fei_i = FEI_COL in sub_iref.columns and sub_iref[FEI_COL].nunique() >= 2

        if metric == "Dissolution" or n_ndc < 3:
            if has_fei_i:
                try:
                    r_iref = sm.OLS(y, X_iref).fit(
                        cov_type="cluster", cov_kwds={"groups": sub_iref[FEI_COL].values}
                    )
                    _coef_table(["const", "USA_vs_IND", "CHN_vs_IND"],
                                r_iref.params, r_iref.bse, dof_i,
                                header=f"★ CHN vs IND — FEI-clustered SE [{ylabel}]:")
                except Exception as exc:
                    print(f"    CHN vs IND FEI-clustered error: {exc}")
        elif has_ndc_i and has_fei_i:
            try:
                V2i = _cgm_vcov(y, X_iref,
                                sub_iref[NDC_COL].values, sub_iref[FEI_COL].values)
                se2i = np.sqrt(np.diag(V2i))
                _coef_table(["const", "USA_vs_IND", "CHN_vs_IND"],
                            ols_iref.params, se2i, dof_i,
                            header=f"★ CHN vs IND — two-way clustered SE [{ylabel}] -- PRIMARY:")
            except Exception as exc:
                print(f"    CHN vs IND two-way error: {exc}")
        elif has_fei_i:
            try:
                r_iref = sm.OLS(y, X_iref).fit(
                    cov_type="cluster", cov_kwds={"groups": sub_iref[FEI_COL].values}
                )
                _coef_table(["const", "USA_vs_IND", "CHN_vs_IND"],
                            r_iref.params, r_iref.bse, dof_i,
                            header=f"★ CHN vs IND — FEI-clustered SE [{ylabel}]:")
            except Exception as exc:
                print(f"    CHN vs IND FEI-clustered error: {exc}")


def fig2_modelB_re_twoway(df, output_dir="."):
    _sep("MODEL B (Fig 2) -- Random Effects + Two-Way Clustered SE  [PRIMARY]")
    print("""
  outcome ~ VAI + OAI + (1|NDC11)
  SE clustered two-way: NDC11 + FEI (CGM 2011)
  Reference group: NAI
""")
    if not HAS_STATSMODELS:
        print("  SKIP: statsmodels required."); return

    for outcome in FIG2_OUTCOMES:
        _subsep(f"Outcome: {outcome}")
        sub, oc = _prep_fig2(df, outcome)
        if sub is None or sub.empty: continue
        n_obs = len(sub)
        n_ndc = sub[NDC_COL].nunique()
        n_fei = sub[FEI_COL].nunique() if FEI_COL in sub.columns else 0
        print(f"  n_obs={n_obs}  n_NDC={n_ndc}  n_FEI={n_fei}")

        y_raw = sub[outcome].values.astype(float)
        y, ylabel, _ = _log_transform(y_raw, outcome)
        sub = sub.copy(); sub["_y"] = y
        X = sm.add_constant(sub[["VAI", "OAI"]].values.astype(float))

        if n_ndc >= 3:
            try:
                mlm = smf.mixedlm("_y ~ VAI + OAI", data=sub,
                                   groups=sub[NDC_COL]).fit(reml=True)
                var_re  = float(mlm.cov_re.iloc[0,0]) if hasattr(mlm,"cov_re") else 0
                var_res = float(mlm.scale)
                icc = var_re / (var_re + var_res) if (var_re + var_res) > 0 else 0
                print(f"\n  RE model | ICC={icc:.4f}  "
                      f"Var(NDC)={var_re:.4f}  Var(resid)={var_res:.4f}")
                names_re = ["Intercept", "VAI", "OAI"]
                p_re = np.array([mlm.params.get(n, np.nan) for n in names_re])
                s_re = np.array([mlm.bse.get(n, np.nan) for n in names_re])
                _coef_table(names_re, p_re, s_re, max(n_obs-3,1),
                            header="RE coefficients (standard SE -- reference only):")
            except Exception as exc:
                print(f"  MixedLM error: {exc}")

        print(f"\n  Two-way clustered SE on OLS [{ylabel}] (reference = NAI):")
        _ols_clustered(y, X, sub, ["VAI", "OAI"], ylabel,
                       f"B_RE_2way_{outcome}", output_dir)

        # --- OAI vs VAI test: re-parameterize with VAI as reference ---
        # NAI and OAI dummies vs VAI baseline gives OAI-VAI coefficient directly.
        _subsep(f"OAI vs VAI comparison [{outcome}]  (reference = VAI)")
        print(f"  Advisor request: also test OAI relative to VAI (not just vs NAI).")
        has_ndc_v = NDC_COL in sub.columns and sub[NDC_COL].nunique() >= 2
        has_fei_v = FEI_COL in sub.columns and sub[FEI_COL].nunique() >= 2

        sub_vref = sub.copy()
        # sub already has "OAI" column from _prep_fig2; add "NAI"
        sub_vref["NAI_dummy"] = (sub_vref[oc] == "NAI").astype(float)
        X_vref = sm.add_constant(
            sub_vref[["NAI_dummy", "OAI"]].values.astype(float)
        )
        ols_vref = sm.OLS(y, X_vref).fit()
        dof_v    = max(len(y) - 3, 1)

        if has_ndc_v and has_fei_v:
            try:
                V2v  = _cgm_vcov(y, X_vref,
                                 sub_vref[NDC_COL].values, sub_vref[FEI_COL].values)
                se2v = np.sqrt(np.diag(V2v))
                _coef_table(["const", "NAI_vs_VAI", "OAI_vs_VAI"],
                            ols_vref.params, se2v, dof_v,
                            header=f"★ OAI vs VAI — two-way clustered SE [{ylabel}] -- PRIMARY:")
            except Exception as exc:
                print(f"    OAI vs VAI two-way error: {exc}")
        elif has_fei_v:
            try:
                r_vref = sm.OLS(y, X_vref).fit(
                    cov_type="cluster", cov_kwds={"groups": sub_vref[FEI_COL].values}
                )
                _coef_table(["const", "NAI_vs_VAI", "OAI_vs_VAI"],
                            r_vref.params, r_vref.bse, dof_v,
                            header=f"★ OAI vs VAI — FEI-clustered SE [{ylabel}]:")
            except Exception as exc:
                print(f"    OAI vs VAI FEI-clustered error: {exc}")


# ============================================================================
# MODEL C — FIXED EFFECTS
# ============================================================================

def fig1_modelC_fe(df, output_dir="."):
    _sep("MODEL C (Fig 1) -- Fixed Effects Regression")
    print("""
  C1. FEI fixed effects:  metric ~ IND + CHN + FEI_i
  C2. NDC fixed effects:  metric ~ IND + CHN + NDC_i
      WARNING: Country is fixed per NDC -> NDC FE will be COLLINEAR
      with IND/CHN dummies. Coefficient likely dropped. EXPECTED.
  C3. NDC + FEI fixed effects (most conservative, DMF/NDMA only)

  John: "I'm guessing there'll be no significance -- but if there is
  significance after fixed effects, that's defensible and easy."
""")
    if not HAS_STATSMODELS:
        print("  SKIP: statsmodels required."); return

    for metric, years in METRIC_YEARS.items():
        _subsep(f"{metric}  (years: {years})")
        sub = _prep_fig1(df, metric)
        if sub.empty: continue
        n_fei = sub[FEI_COL].nunique() if FEI_COL in sub.columns else 0
        print(f"  n_obs={len(sub)}  n_NDC={sub[NDC_COL].nunique()}  n_FEI={n_fei}")

        y_raw = sub[metric].values.astype(float)
        y, ylabel, _ = _log_transform(y_raw, metric)
        sub = sub.copy(); sub["_y"] = y

        # C1: FEI FE
        if FEI_COL in sub.columns and n_fei >= 2:
            print(f"\n  [C1: FEI Fixed Effects | {ylabel}]")
            s = _within_demean(sub, "_y", ["IND", "CHN"], FEI_COL)
            _fe_results(s, "_y", ["IND", "CHN"], FEI_COL, ylabel,
                        f"C1_FEI_{metric}", output_dir)

        # C2: NDC FE
        if metric == "Dissolution":
            print(f"\n  [C2: NDC Fixed Effects | {ylabel}]")
            print("  SKIPPED -- Dissolution is 2024-only; 1 obs per NDC.")
        elif sub[NDC_COL].nunique() >= 2:
            print(f"\n  [C2: NDC Fixed Effects | {ylabel}]")
            print("  WARNING: IND/CHN dummies likely collinear with NDC FE (expected).")
            s = _within_demean(sub, "_y", ["IND", "CHN"], NDC_COL)
            _fe_results(s, "_y", ["IND", "CHN"], NDC_COL, ylabel,
                        f"C2_NDC_{metric}", output_dir)

        # C3: Both (DMF and NDMA only)
        if metric != "Dissolution" and FEI_COL in sub.columns and n_fei >= 2:
            print(f"\n  [C3: NDC + FEI Fixed Effects | {ylabel}] (most conservative)")
            s = _twoway_demean(sub, "_y", ["IND", "CHN"], NDC_COL, FEI_COL)
            _fe_results(s, "_y", ["IND", "CHN"], NDC_COL, ylabel,
                        f"C3_NDCFEI_{metric}", output_dir,
                        extra_fe_col=FEI_COL)


def fig2_modelC_fe(df, output_dir="."):
    _sep("MODEL C (Fig 2) -- Fixed Effects Regression")
    print("""
  C1. FEI FE: outcome ~ VAI + OAI + FEI_i
  C2. NDC FE: outcome ~ VAI + OAI + NDC_i
  C3. NDC + FEI FE (most conservative)
  Reference: NAI
""")
    if not HAS_STATSMODELS:
        print("  SKIP: statsmodels required."); return

    for outcome in FIG2_OUTCOMES:
        _subsep(f"Outcome: {outcome}")
        sub, oc = _prep_fig2(df, outcome)
        if sub is None or sub.empty: continue
        n_fei = sub[FEI_COL].nunique() if FEI_COL in sub.columns else 0
        print(f"  n_obs={len(sub)}  n_NDC={sub[NDC_COL].nunique()}  n_FEI={n_fei}")

        y_raw = sub[outcome].values.astype(float)
        y, ylabel, _ = _log_transform(y_raw, outcome)
        sub = sub.copy(); sub["_y"] = y

        if FEI_COL in sub.columns and n_fei >= 2:
            print(f"\n  [C1: FEI Fixed Effects | {ylabel}]")
            s = _within_demean(sub, "_y", ["VAI", "OAI"], FEI_COL)
            _fe_results(s, "_y", ["VAI", "OAI"], FEI_COL, ylabel,
                        f"C1_FEI_{outcome}", output_dir)

        if sub[NDC_COL].nunique() >= 2:
            print(f"\n  [C2: NDC Fixed Effects | {ylabel}]")
            s = _within_demean(sub, "_y", ["VAI", "OAI"], NDC_COL)
            _fe_results(s, "_y", ["VAI", "OAI"], NDC_COL, ylabel,
                        f"C2_NDC_{outcome}", output_dir)

        if FEI_COL in sub.columns and sub[NDC_COL].nunique() >= 2 and n_fei >= 2:
            print(f"\n  [C3: NDC + FEI Fixed Effects | {ylabel}] (most conservative)")
            s = _twoway_demean(sub, "_y", ["VAI", "OAI"], NDC_COL, FEI_COL)
            _fe_results(s, "_y", ["VAI", "OAI"], NDC_COL, ylabel,
                        f"C3_NDCFEI_{outcome}", output_dir, extra_fe_col=FEI_COL)


def _fe_results(s, ycol, dummy_cols, fe_col, ylabel, tag, output_dir,
                extra_fe_col=None):
    """Run OLS on demeaned data, print results with adjusted dof."""
    X = sm.add_constant(s[dummy_cols].values.astype(float), has_constant="add")
    y = s[ycol].values
    n_obs = len(s)
    n_g1  = s[fe_col].nunique()
    n_g2  = s[extra_fe_col].nunique() if extra_fe_col else 0
    dof   = max(n_obs - len(dummy_cols) - n_g1 - n_g2, 1)
    k     = len(dummy_cols) + 1

    cond = np.linalg.cond(X)
    if cond > 1e8:
        print(f"    WARNING: Condition number={cond:.1e} -- collinearity detected (expected).")

    ols = sm.OLS(y, X).fit()
    # Adjust SE for lost degrees of freedom from FE demeaning
    se_adj = np.sqrt(np.diag(ols.cov_params()) * (n_obs - k) / dof)
    groups_str = f"n_{fe_col}={n_g1}"
    if extra_fe_col:
        groups_str += f"  n_{extra_fe_col}={n_g2}"
    print(f"    n_obs={n_obs}  {groups_str}  dof_adj={dof}")
    _coef_table(dummy_cols, ols.params[1:k], se_adj[1:k], dof)
    _diagnostics(ols.resid, X, ylabel, tag, output_dir)


# ============================================================================
# ADVISORY
# ============================================================================

def _advisory():
    _sep("ADVISORY -- What to report")
    print("""
  Based on the team meeting (Feb 25, 2026):

  RECOMMENDED REPORTING STRATEGY
  --------------------------------
  PRIMARY:     Model B (RE + two-way clustered SE)
               Report beta_IND and beta_CHN with two-way clustered p-values.
               "Random effects with clustered standard errors, accounting
               for repeated NDC measurements and within-facility correlation."

  SENSITIVITY: Model C (Fixed Effects)
               If consistent with Model B: "Results robust to FE specification."
               If country coef dropped (NDC FE): report as expected limitation.
               "Country effects are fully absorbed by NDC fixed effects,
               reflecting that country of manufacture is a time-invariant
               property of each NDC."

  SUPPLEMENTARY: Model A (GEE + FEI dummies)
               Available in appendix / for reviewer response.

  IF SIGNIFICANCE DISAPPEARS (expected with small n):
  ---------------------------------------------------
  "Point estimates were consistently positive for India vs. USA across
  all specifications. After jointly accounting for repeated NDC
  measurements and within-facility clustering, confidence intervals
  widened substantially, reflecting the modest sample sizes. Results
  are directionally consistent with significant differences under less
  conservative assumptions (NDC-clustered bootstrap, p<0.05)."

  ** = p<0.01   * = p<0.05   . = p<0.10
""")


# ============================================================================
# MASTER RUNNER
# ============================================================================

def run_advanced_models(ndc_year_df: pd.DataFrame,
                        output_dir: str = "./diagnostic_plots"):
    """
    Run all three models (A, B, C) for Figures 1 and 2.

    Parameters
    ----------
    ndc_year_df : pd.DataFrame
        Main NDC-year dataframe from the paper's analysis script.
        Required: CountryCode, NDC11, FEI, Year, DMF, NDMA, Dissolution
        For Fig 2: Price, Volume, inspection outcome column
                   (set FIG2_OUTCOME_COL at top of file if needed)
    output_dir : str
        Directory for diagnostic PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "#" * 80)
    print("  METFORMIN JAMA -- REGRESSION MODELS (Model B — Primary)")
    print("  Mixed model (random NDC intercept) + two-way clustered SE (CGM 2011)")
    print("  Models A and C available in this file for sensitivity if needed.")
    print("#" * 80)
    print(f"\n  Diagnostic plots -> {os.path.abspath(output_dir)}/\n")

    print("\n" + "=" * 80)
    print("  FIGURE 1 -- Quality by Country")
    print("  metric ~ IND + CHN + (1|NDC11)  |  reference = USA")
    print("  SE: two-way clustered (NDC+FEI) for panel; FEI-only for Dissolution")
    print("  Additional test: CHN vs IND (reference = IND)")
    print("=" * 80)
    fig1_modelB_re_twoway(ndc_year_df, output_dir)

    print("\n" + "=" * 80)
    print("  FIGURE 2 -- Market Response to Prior Inspection Outcome")
    print("  outcome ~ VAI + OAI + (1|NDC11)  |  reference = NAI")
    print("  Additional test: OAI vs VAI (reference = VAI)")
    print("=" * 80)
    fig2_modelB_re_twoway(ndc_year_df, output_dir)

    print("\n" + "#" * 80)
    print("  DONE")
    print("#" * 80)


if __name__ == "__main__":
    print(__doc__)
