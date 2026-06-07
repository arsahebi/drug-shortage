"""
statistical_tests_figs1_2.py
============================
Offline statistical tests for Figures 1–4 of the Metformin JAMA paper.

STRUCTURE
---------
Figure 1  – Quality (DMF / NDMA / Dissolution) by COUNTRY
            Tests: pairwise Kruskal-Wallis + Dunn post-hoc (independent assumption)
                   GEE-based approach accounting for repeated NDCs across years

Figure 2  – Price / Volume by PRIOR INSPECTION OUTCOME (NAI / VAI / OAI)
            Tests: Kruskal-Wallis + Dunn post-hoc (independent)
                   GEE-based approach accounting for same NDC in multiple years
                   Mixed-effects approach accounting for same facility across NDCs

Figures 3 & 4 – Scatter correlations (already use Spearman in main code)
            Additional: cluster-robust Spearman via bootstrap by NDC
            Note: market data, so no facility clustering; but same NDC repeated
                  across years may not be independent. We handle with NDC-level
                  block bootstrap and report alongside plain Spearman.

ASSUMPTIONS / LIMITATIONS (to mention if tests are reported)
--------------------------------------------------------------
* Independent approach: each (NDC, Year) row treated as an independent observation.
  Underestimates SEs if the same NDC appears in multiple years.
* NDC-cluster robust: accounts for repeated measurement of the same NDC across years
  but treats different NDCs as independent (reasonable for market data, Figs 3/4).
* Facility-cluster robust (Figs 1/2): different NDCs from the same facility share
  manufacturing environment; we account for this via facility (FEI) clustering.
* For Fig 1 country comparison, clustering by NDC (same NDC over years) and by
  facility are both implemented.

USAGE
-----
    # assumes ndc_year_df is already built by the main script
    run_all_tests(ndc_year_df)

    # or call individual sections:
    test_fig1_country(ndc_year_df)
    test_fig2_inspection_outcome(ndc_year_df)
    test_fig3_fig4_scatter_correlations(ndc_year_df)
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, spearmanr, ConstantInputWarning
from itertools import combinations

# Optional imports – GEE / mixed models
try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Gaussian, Gamma
    from statsmodels.genmod.cov_struct import Independence, Exchangeable
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed; GEE tests will be skipped.")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _dunn_posthoc(data_groups: dict, adjust: str = "bonferroni") -> pd.DataFrame:
    """
    Dunn (1964) post-hoc pairwise comparisons after Kruskal-Wallis.
    data_groups: {label: array-like of values}
    adjust: 'bonferroni' | 'holm'

    Returns a DataFrame with columns: group1, group2, z, p_raw, p_adj
    """
    labels = list(data_groups.keys())
    all_values = []
    all_group = []
    for lbl, vals in data_groups.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        all_values.extend(v.tolist())
        all_group.extend([lbl] * len(v))

    combined = np.array(all_values, dtype=float)
    groups = np.array(all_group)
    ranks = stats.rankdata(combined)

    n_total = len(combined)
    ns = {lbl: int((groups == lbl).sum()) for lbl in labels}
    mean_ranks = {lbl: float(ranks[groups == lbl].mean()) for lbl in labels}

    # tie correction
    _, tie_counts = np.unique(combined, return_counts=True)
    T_ties = np.sum(tie_counts ** 3 - tie_counts)
    C = 1 - T_ties / (n_total ** 3 - n_total)

    rows = []
    for g1, g2 in combinations(labels, 2):
        n1, n2 = ns[g1], ns[g2]
        mr1, mr2 = mean_ranks[g1], mean_ranks[g2]
        se = np.sqrt((n_total * (n_total + 1) / 12.0 - T_ties / (12.0 * (n_total - 1))) *
                     (1.0 / n1 + 1.0 / n2))
        se = max(se, 1e-12)
        z = (mr1 - mr2) / se
        p_raw = 2 * stats.norm.sf(abs(z))
        rows.append({"group1": g1, "group2": g2, "z": z,
                     "mean_rank_1": mr1, "mean_rank_2": mr2,
                     "p_raw": p_raw})

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    p_vals = result["p_raw"].values
    k = len(p_vals)

    if adjust == "bonferroni":
        result["p_adj"] = np.minimum(p_vals * k, 1.0)
    elif adjust == "holm":
        order = np.argsort(p_vals)
        p_adj = p_vals.copy()
        for rank, idx in enumerate(order):
            p_adj[idx] = min(p_vals[idx] * (k - rank), 1.0)
        result["p_adj"] = p_adj
    else:
        result["p_adj"] = p_vals

    result["significant_adj"] = result["p_adj"] < 0.05
    return result.round({"z": 3, "p_raw": 5, "p_adj": 5,
                         "mean_rank_1": 2, "mean_rank_2": 2})


def _block_bootstrap_spearman(x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                               n_boot: int = 2000, seed: int = 42) -> dict:
    """
    Cluster (block) bootstrap for Spearman correlation.
    Resamples whole clusters (NDCs or FEIs) with replacement.
    Returns: {'rho': float, 'p': float, 'ci_lo': float, 'ci_hi': float, 'n_clusters': int}
    """
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[mask], y[mask], clusters[mask]

    rho_obs, p_obs = spearmanr(x, y)
    unique_clusters = np.unique(clusters)
    n_cl = len(unique_clusters)

    boot_rhos = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=n_cl, replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in sampled])
        if len(idx) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress ConstantInputWarning silently
            r, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boot_rhos.append(r)

    boot_rhos = np.array(boot_rhos)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    # p-value: shift distribution to 0, see what fraction exceeds |rho_obs|
    shifted = boot_rhos - np.mean(boot_rhos)
    p_boot = float(np.mean(np.abs(shifted) >= abs(rho_obs)))

    return {
        "rho": rho_obs,
        "p_naive": p_obs,
        "p_boot": p_boot,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_obs": int(mask.sum()),
        "n_clusters": n_cl,
    }


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _print_kw(metric: str, groups: dict):
    """Run and print Kruskal-Wallis + Dunn for a single metric."""
    clean = {k: np.asarray(v, dtype=float) for k, v in groups.items()}
    clean = {k: v[np.isfinite(v)] for k, v in clean.items() if len(v) > 0}

    ns = {k: len(v) for k, v in clean.items()}
    medians = {k: float(np.median(v)) if len(v) > 0 else np.nan for k, v in clean.items()}
    means   = {k: float(np.mean(v))   if len(v) > 0 else np.nan for k, v in clean.items()}

    print(f"\n  [{metric}]")
    for lbl in clean:
        print(f"    {lbl:>12s}  n={ns[lbl]:3d}  median={medians[lbl]:>12.4g}  mean={means[lbl]:>12.4g}")

    arrays = [v for v in clean.values() if len(v) >= 1]
    labels = [k for k, v in clean.items() if len(v) >= 1]
    if len(arrays) < 2:
        print("    [SKIP] fewer than 2 groups with data")
        return

    try:
        stat, p = kruskal(*arrays)
        print(f"    Kruskal-Wallis  H={stat:.4f}  p={p:.5f}  {'*' if p < 0.05 else ''}")
    except Exception as e:
        print(f"    Kruskal-Wallis ERROR: {e}")
        return

    if p < 0.05 or True:  # always show pairwise for reporting
        dunn = _dunn_posthoc({lbl: clean[lbl] for lbl in labels}, adjust="bonferroni")
        if not dunn.empty:
            print("    Dunn post-hoc (Bonferroni-adjusted):")
            print(dunn[["group1", "group2", "z", "p_raw", "p_adj", "significant_adj"]]
                  .to_string(index=False))


# ---------------------------------------------------------------------------
# FIGURE 1 – Quality by COUNTRY
# ---------------------------------------------------------------------------

def test_fig1_country(df: pd.DataFrame,
                      country_col: str = "CountryCode",
                      countries: list = ("IND", "CHN", "USA"),
                      metrics: list = ("DMF", "NDMA", "Dissolution"),
                      cluster_col: str = "NDC11",
                      facility_col: str = "FEI"):
    """
    Statistical tests for Figure 1: quality metric differences across countries.

    Approach 1 – Independent (treating each NDC-Year row as independent).
    Approach 2 – Cluster-robust: bootstrap resampling by NDC11 (same NDC over years).
    Approach 3 – Cluster-robust: bootstrap resampling by FEI (same facility across NDCs).
    """
    _print_section("FIGURE 1 – Quality by Country")

    d = df[df[country_col].isin(countries)].copy()

    # ---- Approach 1: Independent (Kruskal-Wallis + Dunn) ----
    print("\n[Approach 1: Independent – each NDC-Year row treated independently]")
    print("  LIMITATION: same NDC in multiple years inflates n; same FEI across NDCs ignored.")
    for metric in metrics:
        groups = {cc: d.loc[d[country_col] == cc, metric].dropna().values
                  for cc in countries}
        _print_kw(metric, groups)

    # ---- Approach 2: Cluster-robust (bootstrap by NDC11) ----
    print("\n[Approach 2: Cluster-robust Bootstrap – resampling by NDC11]")
    print("  Accounts for same NDC appearing in multiple test years.")
    for metric in metrics:
        sub = d[[country_col, metric, cluster_col]].dropna(subset=[metric, cluster_col]).copy()
        if sub.empty:
            print(f"  [{metric}] no data")
            continue
        # Pairwise bootstrap
        print(f"\n  [{metric}]")
        for c1, c2 in combinations(countries, 2):
            g1 = sub[sub[country_col] == c1]
            g2 = sub[sub[country_col] == c2]
            if len(g1) < 3 or len(g2) < 3:
                continue
            # Create a binary indicator variable and correlate (or use Mann-Whitney
            # with cluster bootstrap)
            # Stack into x=metric, cluster=NDC11, group=c1/c2 dummy
            combined = pd.concat([g1[[metric, cluster_col]], g2[[metric, cluster_col]]])
            dummy = np.array([0]*len(g1) + [1]*len(g2), dtype=float)
            res = _block_bootstrap_spearman(
                combined[metric].values.astype(float),
                dummy,
                combined[cluster_col].values,
                n_boot=1000,
            )
            # Use p_boot as cluster-robust p-value analog
            print(f"    {c1} vs {c2}: "
                  f"n_obs={res['n_obs']}  n_clusters={res['n_clusters']}  "
                  f"p_naive={res['p_naive']:.4f}  p_boot_clustered={res['p_boot']:.4f}")

    # ---- Approach 3: Cluster-robust (bootstrap by FEI) ----
    if facility_col in d.columns:
        print("\n[Approach 3: Cluster-robust Bootstrap – resampling by FEI (facility)]")
        print("  Accounts for multiple NDCs from the same manufacturing facility.")
        for metric in metrics:
            sub = d[[country_col, metric, facility_col]].dropna(subset=[metric, facility_col]).copy()
            if sub.empty:
                continue
            print(f"\n  [{metric}]")
            for c1, c2 in combinations(countries, 2):
                g1 = sub[sub[country_col] == c1]
                g2 = sub[sub[country_col] == c2]
                if len(g1) < 3 or len(g2) < 3:
                    continue
                combined = pd.concat([g1[[metric, facility_col]], g2[[metric, facility_col]]])
                dummy = np.array([0]*len(g1) + [1]*len(g2), dtype=float)
                res = _block_bootstrap_spearman(
                    combined[metric].values.astype(float),
                    dummy,
                    combined[facility_col].values,
                    n_boot=1000,
                )
                print(f"    {c1} vs {c2}: "
                      f"n_obs={res['n_obs']}  n_clusters_fei={res['n_clusters']}  "
                      f"p_naive={res['p_naive']:.4f}  p_boot_clustered={res['p_boot']:.4f}")

    # ---- GEE approach (if statsmodels available) ----
    if HAS_STATSMODELS:
        print("\n[Approach 4: GEE – accounts for repeated NDC11 across years]")
        print("  Model: metric ~ C(country) | groups = NDC11 | cov_struct = Exchangeable")
        for metric in metrics:
            sub = d[["NDC11", country_col, metric]].dropna().copy()
            if sub.empty or sub["NDC11"].nunique() < 5:
                print(f"  [{metric}] insufficient clusters for GEE")
                continue
            # encode country as dummies (reference = USA)
            sub["country_enc"] = pd.Categorical(sub[country_col], categories=list(countries))
            dummies = pd.get_dummies(sub["country_enc"], drop_first=True).astype(float)
            sub = pd.concat([sub.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            exog_cols = [c for c in dummies.columns]
            endog = sub[metric].values.astype(float)
            exog = sm.add_constant(sub[exog_cols].values.astype(float))
            groups = sub["NDC11"].values

            try:
                model = GEE(endog, exog, groups=groups,
                            family=Gaussian(),
                            cov_struct=Exchangeable())
                # statsmodels >=0.14 uses ctol; older uses tol
                try:
                    result = model.fit(maxiter=100, ctol=1e-6)
                except TypeError:
                    result = model.fit(maxiter=100)
                print(f"\n  [{metric}] GEE result (reference group: {countries[0]}):")
                print(result.summary().tables[1])
            except Exception as e:
                print(f"  [{metric}] GEE ERROR: {e}")


# ---------------------------------------------------------------------------
# FIGURE 2 – Price / Volume by Prior Inspection Outcome
# ---------------------------------------------------------------------------

def test_fig2_inspection_outcome(df: pd.DataFrame,
                                  score_col: str = "PriorScore",
                                  outcomes: list = (0.0, 1.5, 3.5),
                                  outcome_labels: dict = None,
                                  price_col: str = "price",
                                  vol_col: str = "volume",
                                  cluster_col: str = "NDC11",
                                  facility_col: str = "FEI"):
    """
    Statistical tests for Figure 2: price/volume differences by prior inspection outcome.

    Approach 1 – Independent (Kruskal-Wallis + Dunn).
    Approach 2 – Cluster-robust bootstrap by NDC11 (same NDC year-over-year).
    Approach 3 – Cluster-robust bootstrap by FEI (same facility, multiple NDCs).
    Approach 4 – GEE with NDC11 grouping (if statsmodels available).
    """
    if outcome_labels is None:
        outcome_labels = {0.0: "NAI", 1.5: "VAI", 3.5: "OAI"}

    _print_section("FIGURE 2 – Price & Volume by Prior Inspection Outcome")

    # Filter to IND/USA/CHN only — matches plot_obs2 which uses COUNTRY_CODE_ORDER
    PLOT_COUNTRIES = ["IND", "USA", "CHN"]
    d = df[df["CountryCode"].isin(PLOT_COUNTRIES)].copy()
    print(f"  [Country filter applied: {PLOT_COUNTRIES} — matches Figure 2 plot]")

    d[score_col] = pd.to_numeric(d.get(score_col), errors="coerce")
    d[price_col] = pd.to_numeric(d.get(price_col), errors="coerce")
    d[vol_col]   = pd.to_numeric(d.get(vol_col),   errors="coerce")

    price_df = d[d[score_col].notna() & d[price_col].notna()].copy()
    vol_df   = d[d[score_col].notna() & d[vol_col].notna() & (d[vol_col] > 0)].copy()

    for metric_col, metric_name, sub in [
        (price_col, "PRICE", price_df),
        (vol_col,   "VOLUME (log-transformed for GEE)", vol_df),
    ]:
        print(f"\n{'─'*70}")
        print(f"  Outcome: {metric_name}")
        print(f"{'─'*70}")

        groups_ind = {outcome_labels.get(s, str(s)):
                      sub.loc[sub[score_col] == s, metric_col].values
                      for s in outcomes}

        # ---- Approach 1 ----
        print("\n  [Approach 1: Independent – Kruskal-Wallis + Dunn (Bonferroni)]")
        print("  LIMITATION: same NDC across years, same facility across NDCs.")
        _print_kw(metric_name, groups_ind)

        # ---- Approach 2: NDC11 bootstrap ----
        if cluster_col in sub.columns:
            print(f"\n  [Approach 2: Cluster-robust Bootstrap by {cluster_col}]")
            sub_cl = sub[[score_col, metric_col, cluster_col]].dropna().copy()
            for s1, s2 in combinations(outcomes, 2):
                l1 = outcome_labels.get(s1, str(s1))
                l2 = outcome_labels.get(s2, str(s2))
                g1 = sub_cl[sub_cl[score_col] == s1]
                g2 = sub_cl[sub_cl[score_col] == s2]
                if len(g1) < 3 or len(g2) < 3:
                    print(f"    {l1} vs {l2}: insufficient data")
                    continue
                combined = pd.concat([g1[[metric_col, cluster_col]],
                                      g2[[metric_col, cluster_col]]])
                dummy = np.array([0]*len(g1) + [1]*len(g2), dtype=float)
                res = _block_bootstrap_spearman(
                    combined[metric_col].values.astype(float),
                    dummy,
                    combined[cluster_col].values,
                    n_boot=1000,
                )
                print(f"    {l1} vs {l2}: "
                      f"n_obs={res['n_obs']}  n_clusters={res['n_clusters']}  "
                      f"p_naive={res['p_naive']:.4f}  p_boot_clustered={res['p_boot']:.4f}")

        # ---- Approach 3: FEI bootstrap ----
        if facility_col in sub.columns:
            print(f"\n  [Approach 3: Cluster-robust Bootstrap by {facility_col} (facility)]")
            n_fei = sub[facility_col].nunique()
            print(f"  NOTE: Only {n_fei} unique FEIs – bootstrap unreliable if <20 clusters.")
            print(f"  p=0.000 results here may be artifacts of small cluster count, not true significance.")
            sub_fei = sub[[score_col, metric_col, facility_col]].dropna().copy()
            for s1, s2 in combinations(outcomes, 2):
                l1 = outcome_labels.get(s1, str(s1))
                l2 = outcome_labels.get(s2, str(s2))
                g1 = sub_fei[sub_fei[score_col] == s1]
                g2 = sub_fei[sub_fei[score_col] == s2]
                if len(g1) < 3 or len(g2) < 3:
                    continue
                combined = pd.concat([g1[[metric_col, facility_col]],
                                      g2[[metric_col, facility_col]]])
                dummy = np.array([0]*len(g1) + [1]*len(g2), dtype=float)
                res = _block_bootstrap_spearman(
                    combined[metric_col].values.astype(float),
                    dummy,
                    combined[facility_col].values,
                    n_boot=1000,
                )
                print(f"    {l1} vs {l2}: "
                      f"n_obs={res['n_obs']}  n_fei_clusters={res['n_clusters']}  "
                      f"p_naive={res['p_naive']:.4f}  p_boot_clustered={res['p_boot']:.4f}")

        # ---- Approach 4: GEE ----
        if HAS_STATSMODELS and cluster_col in sub.columns:
            print(f"\n  [Approach 4: GEE by {cluster_col} | Exchangeable within-NDC correlation]")
            sub_gee = sub[[score_col, metric_col, cluster_col]].dropna().copy()
            if sub_gee[cluster_col].nunique() < 5:
                print("    Insufficient clusters for GEE")
                continue
            # encode outcomes as ordinal or dummies
            sub_gee["score_enc"] = sub_gee[score_col].map(
                {s: i for i, s in enumerate(sorted(outcomes))}
            ).astype(float)
            endog = sub_gee[metric_col].values.astype(float)
            if "VOLUME" in metric_name:
                endog = np.log10(endog)  # log-transform volume
            exog = sm.add_constant(sub_gee["score_enc"].values.astype(float))
            groups = sub_gee[cluster_col].values
            try:
                model = GEE(endog, exog, groups=groups,
                            family=Gaussian(),
                            cov_struct=Exchangeable())
                try:
                    result = model.fit(maxiter=100, ctol=1e-6)
                except TypeError:
                    result = model.fit(maxiter=100)
                print(f"    GEE coef on score (0=NAI,1=VAI,2=OAI):")
                print(result.summary().tables[1])
            except Exception as e:
                print(f"    GEE ERROR: {e}")


# ---------------------------------------------------------------------------
# FIGURES 3 & 4 – Scatter: Quality vs Volume / Price
# ---------------------------------------------------------------------------

def test_fig3_fig4_scatter_correlations(df: pd.DataFrame,
                                         xcols: list = ("DMF", "NDMA", "Dissolution"),
                                         ycols: list = ("volume", "price"),
                                         cluster_col: str = "NDC11"):
    """
    Statistical tests for Figures 3 & 4 scatter plots.

    The main code already reports plain Spearman.
    Here we add:
      - NDC-level block bootstrap Spearman (cluster-robust)
        Rationale: same NDC tracked in 2020, 2022, 2024 → not independent.
        Facility clustering less important here because we use market-level data
        (IQVIA) aggregated at NDC level, not raw inspection rows.

    NOTE: plain Spearman is already reasonable if most NDCs appear in only 1 year.
    If many appear in 2–3 years, the naive p-value will be anti-conservative.
    """
    _print_section("FIGURES 3 & 4 – Scatter Correlations (Quality vs Volume / Price)")
    print("  Approach: Naive Spearman (as in main code) + NDC-cluster-bootstrap Spearman")
    print(f"  Cluster col: {cluster_col}")

    # Filter to IND/USA/CHN only — matches plot_obs3 which uses COUNTRY_CODE_ORDER
    PLOT_COUNTRIES = ["IND", "USA", "CHN"]
    d = df[df["CountryCode"].isin(PLOT_COUNTRIES)].copy()
    print(f"  [Country filter applied: {PLOT_COUNTRIES} — matches Figures 3 & 4 plots]")
    for xcol in xcols:
        d[xcol] = pd.to_numeric(d.get(xcol), errors="coerce")
    for ycol in ycols:
        d[ycol] = pd.to_numeric(d.get(ycol), errors="coerce")

    for ycol in ycols:
        print(f"\n  Y = {ycol}")
        for xcol in xcols:
            sub = d[[xcol, ycol, cluster_col]].dropna().copy()
            if ycol == "volume":
                sub = sub[sub[ycol] > 0]
            n = len(sub)
            if n < 3:
                print(f"    {xcol}: insufficient data (n={n})")
                continue

            # Naive Spearman
            rho_naive, p_naive = spearmanr(sub[xcol].values, sub[ycol].values)

            # Cluster bootstrap
            res = _block_bootstrap_spearman(
                sub[xcol].values.astype(float),
                sub[ycol].values.astype(float),
                sub[cluster_col].values,
                n_boot=2000,
            )
            n_clusters = res["n_clusters"]

            print(f"    {xcol}:")
            print(f"      Naive Spearman: rho={rho_naive:+.4f}  p={p_naive:.5f}  n={n}")
            print(f"      Clustered (NDC bootstrap): rho={res['rho']:+.4f}  "
                  f"p_boot={res['p_boot']:.5f}  "
                  f"95%CI=[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
                  f"n_ndcs={n_clusters}")
            if abs(p_naive - res["p_boot"]) > 0.05:
                print(f"      *** Note: naive and clustered p-values differ notably – "
                      f"consider reporting clustered version as sensitivity.")


# ---------------------------------------------------------------------------
# MASTER RUNNER
# ---------------------------------------------------------------------------

def run_all_tests(ndc_year_df: pd.DataFrame):
    """
    Run all three test blocks.
    Call this function after building ndc_year_df in the main script.
    """
    print("\n" + "#" * 80)
    print("  METFORMIN JAMA – OFFLINE STATISTICAL TESTS")
    print("  (Results are offline / not modifying the paper directly)")
    print("#" * 80)

    test_fig1_country(ndc_year_df)
    test_fig2_inspection_outcome(ndc_year_df)
    test_fig3_fig4_scatter_correlations(ndc_year_df)

    print("\n" + "#" * 80)
    print("  DONE")
    print("#" * 80)


# ---------------------------------------------------------------------------
# STANDALONE USAGE NOTES (printed at import)
# ---------------------------------------------------------------------------
_USAGE = """
HOW TO USE
----------
In the main script (20260123-MetforminJAMAGraphs.py), after the line:
    ndc_year_df = build_ndc_year_table(df)

Add:
    from statistical_tests_figs1_2 import run_all_tests
    run_all_tests(ndc_year_df)

Or call individual functions:
    from statistical_tests_figs1_2 import test_fig1_country, test_fig2_inspection_outcome
    test_fig1_country(ndc_year_df)
    test_fig2_inspection_outcome(ndc_year_df)

REQUIRED PACKAGES
-----------------
    scipy         (always required)
    statsmodels   (optional, for GEE – install with: pip install statsmodels)

LIMITATIONS TO MENTION IN PAPER (if tests are reported)
---------------------------------------------------------
1. Independent approach: each (NDC, Year) row treated as independent.
   Likely underestimates SEs if same NDC tested in multiple years.

2. NDC-cluster bootstrap: corrects for repeated testing of same NDC,
   but treats different NDCs sharing a facility as independent.

3. Facility (FEI) bootstrap: corrects for multiple NDCs per facility,
   but assumes independence across facilities and across years within NDC.

4. GEE (Figs 1/2): correct approach for correlated data, but requires
   normally distributed residuals (met better for price than volume).
   For volume (log-transformed), GEE results should be interpreted on
   log10 scale.

5. Figures 3/4 use market-level (IQVIA) data; no facility clustering needed,
   but same NDC across 3 test years introduces within-NDC dependence.
   Block bootstrap by NDC is the recommended sensitivity analysis.
"""

if __name__ == "__main__":
    print(_USAGE)
