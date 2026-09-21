# %%
"""
Build max-sample figures for both NDC-FEI linkage methods (manual, rule-based),
each pooled and split by dosage form (IR / ER).

Per John's guidance (Sept 17 2026 meeting): each figure uses whatever sample its
own axes require, with no additional restriction beyond the standing Canada /
Bangladesh exclusion. Figures 1 and 4 still need prior_outcome / CountryCode
respectively and so are naturally limited to facility-linked rows. Figures 2 and 3
color points by country, so they also require an actual matched facility (not just
a non-null CountryCode -- that field can come from an old Q&A spreadsheet fallback
independent of any real facility match); S1 uses every row with the fields it plots.

Inputs: variants/step5_{manual,rulebased}.csv (built by running step2-step5
with REQUIRE_REDICA_HISTORY=0 DROP_NDCS_WITHOUT_FEI=0 against each step1 map;
see the run commands in the Sept 20 2026 session).

Dosage form (IR/ER) comes from Valisure's own per-year testing sheets
(Data/08 - Valisure/raw/Valisure_2024_raw_prices_20260728_f1-and-formulation_
20260813.xlsx -- the formulation column we asked Valisure to add), joined on
NDC11 x TestYear via _valisure_formulation_lookup(), not from the step5 CSV's
static IR_ER column (which was a one-time NDC8 -> FDA product.csv
DOSAGEFORMNAME join, constant across years). Checked against that static join
before switching: 126/126 overlapping rows agreed exactly; the Valisure file
also resolves 14 NDCs the static join couldn't match.

Output: processed/outputs/variants/<map>_<dosage>/ — PNGs + a stats log.
"""

import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, kruskal, mannwhitneyu

warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
PROC = BASE / "Data/99 - Outputs - Metformin Analysis/processed"
VDIR = PROC / "variants"
OUT  = PROC / "outputs" / "variants"

DMF_COL, NDMA_COL, DIFF_COL = "DMF (ng/DAY) Valisure", "NDMA (ng/DAY) Valisure", "Difference Factor"
VOL_COL, PRICE_COL = "iqvia_extended_units", "price"
OUTCOME_ORDER  = ["NAI", "VAI", "OAI"]
COUNTRY_ORDER  = ["IND", "CHN", "USA"]
COUNTRY_COLORS = {"IND": "#ef4444", "CHN": "#f59e0b", "USA": "#3b82f6"}
OUTCOME_COLORS = {"NAI": "#22c55e", "VAI": "#f59e0b", "OAI": "#ef4444"}


# ── stat helpers (self-contained; mirror step6_graphs_july26.py) ─────────────
def _kruskal_p(groups: dict):
    valid = {k: np.asarray(v, dtype=float) for k, v in groups.items() if len(v) >= 2}
    if len(valid) < 2:
        return None
    try:
        return kruskal(*valid.values())[1]
    except Exception:
        return None


def _dunn_posthoc(groups: dict, adjust="bonferroni") -> pd.DataFrame:
    labels = list(groups.keys())
    all_vals, all_grp = [], []
    for lbl, vals in groups.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        all_vals.extend(v.tolist())
        all_grp.extend([lbl] * len(v))
    combined = np.array(all_vals, dtype=float)
    grp_arr = np.array(all_grp)
    rnks = stats.rankdata(combined)
    n_total = len(combined)
    _, tie_counts = np.unique(combined, return_counts=True)
    T_ties = np.sum(tie_counts ** 3 - tie_counts)
    rows = []
    for g1, g2 in combinations(labels, 2):
        n1, n2 = int((grp_arr == g1).sum()), int((grp_arr == g2).sum())
        if n1 < 2 or n2 < 2:
            continue
        mr1, mr2 = float(rnks[grp_arr == g1].mean()), float(rnks[grp_arr == g2].mean())
        se = np.sqrt((n_total * (n_total + 1) / 12.0 - T_ties / (12.0 * (n_total - 1)))
                     * (1.0 / n1 + 1.0 / n2))
        se = max(se, 1e-12)
        z = (mr1 - mr2) / se
        p_raw = 2 * stats.norm.sf(abs(z))
        rows.append({"group1": g1, "group2": g2, "z": round(z, 3), "p_raw": round(p_raw, 5)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    k = len(result)
    result["p_adj"] = np.minimum(result["p_raw"].values * k, 1.0).round(5) if adjust == "bonferroni" else result["p_raw"]
    result["sig"] = result["p_adj"].apply(lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else ""))
    return result


def _cluster_permutation_p(x, y, clusters, n_perm=5000, seed=42):
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym, cm = x[mask], y[mask], np.asarray(clusters, dtype=str)[mask]
    ranks = stats.rankdata(xm)
    uc = np.unique(cm)
    idx = [np.where(cm == c)[0] for c in uc]
    if any(len(np.unique(ym[i])) > 1 for i in idx):
        return np.nan  # clusters straddle groups; permutation invalid
    lab = np.array([ym[i][0] for i in idx])
    def st(l):
        sel = np.concatenate([idx[k] for k in range(len(uc)) if l[k] == 1])
        return abs(ranks[sel].mean() - ranks.mean())
    obs = st(lab)
    rng = np.random.default_rng(seed)
    c = 0
    for _ in range(n_perm):
        if st(rng.permutation(lab)) >= obs - 1e-12:
            c += 1
    return (c + 1) / (n_perm + 1)


def _block_bootstrap_spearman(x, y, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym, cm = x[mask], y[mask], np.asarray(clusters, dtype=str)[mask]
    rho_obs, p_obs = spearmanr(xm, ym)
    unique_cl = np.unique(cm)
    n_cl = len(unique_cl)
    boot_rhos = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_cl, size=n_cl, replace=True)
        idx = np.concatenate([np.where(cm == c)[0] for c in sampled])
        if len(idx) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = spearmanr(xm[idx], ym[idx])
        boot_rhos.append(r if np.isfinite(r) else 0.0)
    if len(boot_rhos) < 10:
        return {"rho": rho_obs, "p_naive": p_obs, "p_boot": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_obs": int(mask.sum()), "n_clusters": n_cl}
    boot_rhos = np.array(boot_rhos)
    shifted = boot_rhos - np.mean(boot_rhos)
    p_boot = max(float(np.mean(np.abs(shifted) >= abs(rho_obs))), 1.0 / n_boot)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    return {"rho": rho_obs, "p_naive": p_obs, "p_boot": p_boot, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "n_obs": int(mask.sum()), "n_clusters": n_cl}


# ── Model B: RE + two-way CGM clustered SE (NDC x FEI) ────────────────────────
# Ported verbatim from step6_graphs_july26.py, the established PRIMARY
# specification for Figures 1 and 4 per Metformin JAMA 2026 02 27_StatTests.docx
# (Model B there). Figures 2 and 3 keep Spearman + NDC-cluster bootstrap, which
# that memo already validated as the appropriate test for NDC-year scatter data
# and does not extend two-way clustering to.
def _cgm_vcov(y, X, c1, c2, beta=None):
    n, k = X.shape
    b = np.asarray(beta) if beta is not None else np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    def _v(clusters):
        g = len(np.unique(clusters))
        dfc = (g / (g - 1)) * (n / (n - k))
        meat = np.zeros((k, k))
        for c in np.unique(clusters):
            idx = clusters == c
            sc = X[idx].T @ e[idx]
            meat += np.outer(sc, sc)
        return bread @ (dfc * meat) @ bread
    inter = np.array([f"{a}__{b}" for a, b in zip(c1, c2)])
    return _v(c1) + _v(c2) - _v(inter)


def _coef_table(log, names, params, se, dof, header="", group_support=None, ref_support=None):
    """group_support: optional {name: (n_obs, n_fei)} to flag dummies resting on
    too few observations or too few facilities. A coefficient with e.g. one
    China observation can still return a tiny SE and p<0.001 -- that is a
    single-point artifact, not evidence, and must not be reported as a finding.

    ref_support: optional (n_obs, n_fei) for the omitted/reference category.
    Every dummy in the table is a contrast against that baseline, so if the
    baseline itself is thin (e.g. a 3-observation, 1-facility reference group),
    every coefficient in the table is just as unreliable even when its own
    group_support looks fine -- this flags all of them in that case."""
    from scipy.stats import t as t_dist
    t_vals = params / np.where(se > 0, se, np.nan)
    p_vals = 2 * t_dist.sf(np.abs(t_vals), df=max(dof, 1))
    lo, hi = params - 1.96 * se, params + 1.96 * se
    ref_thin = ref_support is not None and (ref_support[0] < 3 or ref_support[1] < 2)
    if header:
        log(f"\n  {header}")
    for i, name in enumerate(names):
        if name == "const" or np.isnan(params[i]):
            continue
        sig = "**" if p_vals[i] < 0.01 else ("*" if p_vals[i] < 0.05 else ("." if p_vals[i] < 0.10 else ""))
        p_str = f"p={p_vals[i]:.4f}" if p_vals[i] >= 0.001 else "p<0.001"
        flag = ""
        if group_support and name in group_support:
            n_obs, n_fei = group_support[name]
            if n_obs < 3 or n_fei < 2:
                flag = f"  ** UNRELIABLE: only {n_obs} obs from {n_fei} facility(ies), not a real estimate **"
        if not flag and ref_thin:
            flag = (f"  ** UNRELIABLE: reference group has only {ref_support[0]} obs "
                     f"from {ref_support[1]} facility(ies), not a real baseline **")
        log(f"    {name}: beta={params[i]:+.3f}, SE={se[i]:.3f}, "
            f"95% CI [{lo[i]:+.3f}, {hi[i]:+.3f}], {p_str}{sig}{flag}")


def modelB_re_twoway(log, sub, y_col, dummy_names, ndc_col, fei_col, tag, cross_section=False):
    """Model B (PRIMARY for Fig 1 / Fig 4), matching
    Metformin_2026 03 10_Appendix.docx: OLS point estimate + Cameron-Gelbach-
    Miller (2011) two-way clustered SE on NDC x FEI is PRIMARY. A random-NDC-
    intercept MixedLM is fit only as a diagnostic, to report the ICC that
    justifies clustering -- it does not supply the reported beta.

    (Changed 2026-09-21 to match step6_graphs_july26.py: an earlier version
    used the MixedLM beta as primary. On this data the India coefficient is
    essentially unaffected either way; OLS is what every paper draft from
    March 10 onward has described, and it is what CGM (2011) itself derives
    the clustering formula for.)

    cross_section=True (Difference Factor, 2024 only) switches to FEI-only
    clustered SE with no MixedLM diagnostic step at all, matching
    step6_graphs_july26.py's is_xs branch. A single-year metric gives exactly
    one observation per NDC, so n_NDC == n_obs and there is no repeated-measures
    structure for an ICC to describe in the first place."""
    if not HAS_STATSMODELS:
        log("  statsmodels not available; Model B skipped"); return
    sub = sub.dropna(subset=[y_col, ndc_col, fei_col] + dummy_names).copy()
    if sub.empty:
        log(f"  [{tag}] n=0, skipped"); return
    y = sub[y_col].values.astype(float)
    X = sm.add_constant(sub[dummy_names].values.astype(float))
    n_obs, n_ndc = len(sub), sub[ndc_col].nunique()
    n_fei = sub[fei_col].nunique()
    dof = max(n_obs - len(dummy_names) - 1, 1)
    log(f"  [{tag}] n_obs={n_obs}  n_NDC={n_ndc}  n_FEI={n_fei}")

    group_support = {}
    for d in dummy_names:
        m = sub[d] == 1
        group_support[d] = (int(m.sum()), int(sub.loc[m, fei_col].nunique()))
    ref_mask = (sub[dummy_names] == 0).all(axis=1)
    ref_support = (int(ref_mask.sum()), int(sub.loc[ref_mask, fei_col].nunique()))

    if cross_section:
        log("    cross-section (single year): FEI-only clustered SE, NDC clustering N/A")
        if n_fei < 2:
            log("    too few FEI clusters (need >=2)"); return
        try:
            ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub[fei_col].values})
            _coef_table(log, ["const"] + dummy_names, ols.params, ols.bse, dof,
                        header=f"OLS + FEI-clustered SE -- PRIMARY (cross-section) [{tag}]:",
                        group_support=group_support, ref_support=ref_support)
        except Exception as exc:
            log(f"    FEI-clustered error: {exc}")
        return

    if n_ndc < 2 or n_fei < 2:
        log("    too few NDC or FEI clusters for two-way clustering (need >=2 each)"); return

    # Diagnostic only: MixedLM ICC, to document within-NDC correlation and
    # justify clustering. Its beta is not used below.
    try:
        formula = f"{y_col} ~ " + " + ".join(dummy_names)
        mlm = smf.mixedlm(formula, data=sub, groups=sub[ndc_col]).fit(reml=True)
        var_re = float(mlm.cov_re.iloc[0, 0]) if hasattr(mlm, "cov_re") else 0
        var_res = float(mlm.scale)
        icc = var_re / (var_re + var_res) if (var_re + var_res) > 0 else 0
        log(f"    MixedLM (diagnostic only): ICC={icc:.4f}")
    except Exception as exc:
        log(f"    MixedLM diagnostic error: {exc}")

    # PRIMARY: OLS point estimate + CGM two-way clustered SE (NDC x FEI)
    ols = sm.OLS(y, X).fit()
    try:
        V2 = _cgm_vcov(y, X, sub[ndc_col].values, sub[fei_col].values, beta=ols.params)
        se2 = np.sqrt(np.diag(V2))
        _coef_table(log, ["const"] + dummy_names, ols.params, se2, dof,
                    header=f"OLS + TWO-WAY clustered SE (NDC x FEI) -- PRIMARY [{tag}]:",
                    group_support=group_support, ref_support=ref_support)
    except Exception as exc:
        log(f"    CGM error: {exc}")


def pairwise_group_tests(log, sub, val_col, group_col, order, cluster_col="NDC11", fei_col=None):
    groups = {g: sub.loc[sub[group_col] == g, val_col].dropna().values for g in order}
    kw = _kruskal_p(groups)
    log(f"  Kruskal-Wallis across {[g for g in order if len(groups[g])]}: "
        f"p={kw:.4f}" if kw is not None else "  Kruskal-Wallis: n/a")
    valid = {k: v for k, v in groups.items() if len(v) >= 2}
    if kw is not None and len(valid) >= 2:
        dunn = _dunn_posthoc(valid)
        if not dunn.empty:
            log(dunn.to_string(index=False))
    for g1, g2 in combinations(order, 2):
        d1, d2 = sub[sub[group_col] == g1], sub[sub[group_col] == g2]
        if len(d1) < 3 or len(d2) < 3:
            log(f"  {g1} vs {g2}: insufficient data (n1={len(d1)}, n2={len(d2)})")
            continue
        x = np.concatenate([d1[val_col].values, d2[val_col].values]).astype(float)
        y = np.array([0] * len(d1) + [1] * len(d2), dtype=float)
        cl = np.concatenate([d1[cluster_col].astype(str).values, d2[cluster_col].astype(str).values])
        p_perm = _cluster_permutation_p(x, y, cl)
        p_mw = mannwhitneyu(d1[val_col].astype(float), d2[val_col].astype(float), alternative="two-sided").pvalue
        if np.isfinite(p_perm):
            p_primary, tag = p_perm, f"p_perm={p_perm:.4f}"
        else:
            p_primary, tag = p_mw, f"p_mw={p_mw:.4f} (clusters straddle; perm N/A)"
        sig = " **" if p_primary < 0.01 else (" *" if p_primary < 0.05 else "")
        extra = ""
        if fei_col is not None:
            d1f, d2f = d1[d1[fei_col].notna()], d2[d2[fei_col].notna()]
            if len(d1f) >= 3 and len(d2f) >= 3:
                xf = np.concatenate([d1f[val_col].values, d2f[val_col].values]).astype(float)
                yf = np.array([0] * len(d1f) + [1] * len(d2f), dtype=float)
                clf = np.concatenate([d1f[fei_col].astype(str).values, d2f[fei_col].astype(str).values])
                p_fei = _cluster_permutation_p(xf, yf, clf)
                extra = (f"  | FEI-clustered (n1={len(d1f)} n2={len(d2f)}, "
                         f"{len(np.unique(clf))} facilities): "
                         f"p_perm={p_fei:.4f}" if np.isfinite(p_fei) else
                         "  | FEI-clustered: clusters straddle groups, N/A")
            else:
                extra = "  | FEI-clustered: insufficient facility-linked rows"
        log(f"  {g1} vs {g2}: n1={len(d1)} n2={len(d2)}  NDC-clustered {tag}{sig}  [p_mw={p_mw:.4f}]{extra}")


def correlation_tests(log, sub, x_col, y_col, cluster_col="NDC11"):
    d = sub[[x_col, y_col, cluster_col]].dropna()
    if len(d) < 5:
        log(f"  n={len(d)}: insufficient data")
        return None
    res = _block_bootstrap_spearman(d[x_col].values.astype(float), d[y_col].values.astype(float), d[cluster_col].values)
    sig = " **" if res["p_boot"] < 0.01 else (" *" if res["p_boot"] < 0.05 else "")
    log(f"  n={res['n_obs']} (NDCs={res['n_clusters']})  rho={res['rho']:+.4f}  "
        f"95% CI [{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]  "
        f"p_naive={res['p_naive']:.4f}  p_boot={res['p_boot']:.4f}{sig}")
    return res


COUNTRY_FULL = {"IND": "India", "USA": "United States of America", "CHN": "China"}


def _country_legend(fig, ax_ref, title="Country"):
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COUNTRY_COLORS[cc],
                           markersize=7, label=COUNTRY_FULL[cc]) for cc in COUNTRY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(COUNTRY_ORDER), title=title,
               bbox_to_anchor=(0.5, -0.06), frameon=True, fontsize=8, title_fontsize=8)


def _n_label(ax, x_pos, n, y_frac=0.03):
    ymin, ymax = ax.get_ylim()
    if ax.get_yscale() == "log":
        y = ymin * (ymax / ymin) ** y_frac
    else:
        y = ymin + (ymax - ymin) * y_frac
    ax.text(x_pos, y, f"n={n}", ha="center", va="top", fontsize=8, color="#444444")


# ── figure builders ───────────────────────────────────────────────────────────
def fig1(df, outdir, log):
    d = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, ylab in [(axes[0], PRICE_COL, "Price per Unit ($)"),
                           (axes[1], VOL_COL, "Market Volume (Extended Units)")]:
        sub = d[d["prior_outcome"].notna() & d[col].notna() & (d[col] > 0)]
        if col == PRICE_COL:
            sub = sub[sub.get("price_outlier", 0) == 0]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("n=0")
            log(f"\n[{ylab}] n=0"); continue
        data = [sub.loc[sub.prior_outcome == o, col].values for o in OUTCOME_ORDER]
        bp = ax.boxplot(data, labels=OUTCOME_ORDER, showfliers=False, patch_artist=True,
                         boxprops=dict(facecolor="none", edgecolor="black"),
                         medianprops=dict(color="#f59e0b", linewidth=1.5))
        rng = np.random.default_rng(0)
        for i, o in enumerate(OUTCOME_ORDER):
            s = sub.loc[sub.prior_outcome == o]
            jitter = rng.uniform(-0.06, 0.06, len(s))
            colors = [COUNTRY_COLORS.get(cc, "#9ca3af") for cc in s.CountryCode]
            ax.scatter(i + 1 + jitter, s[col], s=16, alpha=0.75, color=colors, edgecolor="none")
        if col == VOL_COL:
            ax.set_yscale("log")
        ax.set_xlabel("Prior Inspection Outcome")
        ax.set_ylabel(ylab)
        for i, o in enumerate(OUTCOME_ORDER):
            _n_label(ax, i + 1, int((sub.prior_outcome == o).sum()))
        log(f"\n[{ylab}] n={len(sub)}, by outcome: "
            f"{ {o: int((sub.prior_outcome==o).sum()) for o in OUTCOME_ORDER} }")
        pairwise_group_tests(log, sub, col, "prior_outcome", OUTCOME_ORDER, fei_col="prior_fei")
        m = sub.copy()
        m["VAI"] = (m.prior_outcome == "VAI").astype(float)
        m["OAI"] = (m.prior_outcome == "OAI").astype(float)
        m["_y"] = np.log(m[col].astype(float))
        modelB_re_twoway(log, m, "_y", ["VAI", "OAI"], "NDC11", "prior_fei", f"log({ylab}), ref=NAI")
        # OAI vs VAI: re-parameterize with VAI as the omitted (reference) group
        m2 = sub.copy()
        m2["NAI_d"] = (m2.prior_outcome == "NAI").astype(float)
        m2["OAI_d"] = (m2.prior_outcome == "OAI").astype(float)
        m2["_y"] = np.log(m2[col].astype(float))
        modelB_re_twoway(log, m2, "_y", ["NAI_d", "OAI_d"], "NDC11", "prior_fei", f"log({ylab}), ref=VAI")
    _country_legend(fig, axes[0], title="Country")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(outdir / "Figure1_Price_Volume_by_Outcome.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig2_3(df, outdir, log, x_col, label, fname):
    d = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [(DMF_COL, "DMF (ng/day)", False), (NDMA_COL, "NDMA (ng/day)", False),
               (DIFF_COL, "Dissolution Difference", True)]
    for ax, (col, xlab, linear_x) in zip(axes, metrics):
        # Only rows with an actual matched facility (matched_fei notna), not just
        # a non-null CountryCode. CountryCode alone is not reliable: 28 of the
        # 124 "known country" DMF-vs-volume rows in the rulebased/all variant
        # have zero matched FEIs and got their country from the old Q&A
        # spreadsheet's independent CountryCode column, not from either linkage
        # method. Since the figure visually encodes country by color, showing
        # those points (colored or grey) would misrepresent them as having a
        # confirmed country when they don't.
        sub = d[d[col].notna() & d[x_col].notna() & (d[x_col] > 0)
                & d.CountryCode.isin(COUNTRY_ORDER) & d.matched_fei.notna()]
        if x_col == PRICE_COL:
            sub = sub[sub.get("price_outlier", 0) == 0]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            log(f"\n[{xlab} vs {label}] n=0")
            continue
        for cc in COUNTRY_ORDER:
            s = sub[sub.CountryCode == cc]
            if len(s):
                ax.scatter(s[col], s[x_col], s=22, alpha=0.75, color=COUNTRY_COLORS[cc])
        if not linear_x:
            ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel(xlab); ax.set_ylabel(label)
        log(f"\n[{xlab} vs {label}] n={len(sub)} (NDCs={sub.NDC11.nunique()}, "
            f"all facility-matched, country confirmed)")
        res = correlation_tests(log, sub, col, x_col)
        if res is not None:
            sig = " **" if res["p_boot"] < 0.01 else (" *" if res["p_boot"] < 0.05 else "")
            ci = f"[{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}]" if np.isfinite(res["ci_lo"]) else "[n/a]"
            txt = f"n={res['n_obs']}\nrho={res['rho']:+.3f} {ci}\np={res['p_boot']:.4f}{sig}"
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                     bbox=dict(boxstyle="round", facecolor="#f5eedc", edgecolor="#c9b98a", alpha=0.9))
            # Fit in the same space the axes are drawn in (x on symlog, y on log)
            # so the line renders straight rather than bending on the plot. A
            # linear-space fit plotted on log axes looks curved even though the
            # fit itself is a straight line -- it is straight in the wrong space.
            xs = sub[col].values.astype(float)
            ys = sub[x_col].values.astype(float)
            if len(xs) >= 3 and xs.max() > xs.min():
                zx = np.linspace(xs.min(), xs.max(), 50)
                b, a = np.polyfit(np.log1p(xs), np.log(ys), 1)
                ax.plot(zx, np.exp(a + b * np.log1p(zx)), "--", color="#f4777f", linewidth=1.5)
    _country_legend(fig, axes[0], title="Country")
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(outdir / f"{fname}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig4(df, outdir, log):
    # Country here means the manufacturing facility's country, so a row only
    # belongs in this figure if a facility was actually matched (matched_fei
    # notna). Some rows carry CountryCode from the old Q&A spreadsheet's own
    # country field even with zero matched FEIs (n_feis == 0) -- an
    # independent, untraceable source, not evidence of a facility link -- and
    # are excluded here even though the max-sample policy keeps them for
    # Figures 2/3, which don't use country at all.
    d = df[df.CountryCode.isin(COUNTRY_ORDER) & df.matched_fei.notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [(DMF_COL, "DMF (ng/day)"), (NDMA_COL, "NDMA (ng/day)"), (DIFF_COL, "Dissolution Difference")]
    for ax, (col, title) in zip(axes, metrics):
        sub = d[d[col].notna()]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            log(f"\n[{title} by country] n=0"); continue
        means = [sub.loc[sub.CountryCode == cc, col].mean() for cc in COUNTRY_ORDER]
        ns    = [int((sub.CountryCode == cc).sum()) for cc in COUNTRY_ORDER]
        bars = ax.bar(range(len(COUNTRY_ORDER)), means, color="#93c5fd", edgecolor="black", width=0.6)
        ax.set_xticks(range(len(COUNTRY_ORDER)))
        ax.set_xticklabels([COUNTRY_FULL[cc].replace("United States of America", "USA") for cc in COUNTRY_ORDER])
        ax.set_xlabel("Country")
        ax.set_title(title)
        ymax = max(means) if means else 1
        for i, (m, n) in enumerate(zip(means, ns)):
            ax.text(i, m + ymax * 0.02, f"{m:,.2f}" if m < 100 else f"{m:,.0f}",
                    ha="center", va="bottom", fontsize=9)
            ax.text(i, -ymax * 0.06, f"n={n}", ha="center", va="top", fontsize=8, color="#444444")
        ax.set_ylim(bottom=-ymax * 0.12 if ymax else -0.1, top=ymax * 1.15 if ymax else 1)
        log(f"\n[{title} by country] n={len(sub)}, by country: "
            f"{ {cc: int((sub.CountryCode==cc).sum()) for cc in COUNTRY_ORDER} }")
        pairwise_group_tests(log, sub, col, "CountryCode", COUNTRY_ORDER, fei_col="matched_fei")
        m = sub.copy()
        m["IND"] = (m.CountryCode == "IND").astype(float)
        m["CHN"] = (m.CountryCode == "CHN").astype(float)
        m["_y"] = np.log1p(m[col].astype(float))
        modelB_re_twoway(log, m, "_y", ["IND", "CHN"], "NDC11", "matched_fei", f"log1p({title}), ref=USA",
                          cross_section=(col == DIFF_COL))
        # CHN vs IND: re-parameterize with IND as the omitted (reference) group
        m2 = sub.copy()
        m2["USA_d"] = (m2.CountryCode == "USA").astype(float)
        m2["CHN_d"] = (m2.CountryCode == "CHN").astype(float)
        m2["_y"] = np.log1p(m2[col].astype(float))
        modelB_re_twoway(log, m2, "_y", ["USA_d", "CHN_d"], "NDC11", "matched_fei", f"log1p({title}), ref=IND",
                          cross_section=(col == DIFF_COL))
    fig.tight_layout()
    fig.savefig(outdir / "Figure4_Quality_by_Country.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig5(df, outdir, log):
    """Market volume by country of manufacture -- Figure 1's box-plot layout
    with the axes swapped: x is CountryCode instead of prior_outcome, points
    are colored by prior inspection outcome instead of country. Uses
    matched_fei (always available once a facility is matched) rather than
    prior_fei, matching Figure 4's clustering choice for the same reason: a
    country figure should not additionally require inspection history."""
    d = df[df.CountryCode.isin(COUNTRY_ORDER) & df.matched_fei.notna()
           & df[VOL_COL].notna() & (df[VOL_COL] > 0)].copy()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        log("\n[Market Volume (Extended Units) by country] n=0")
        fig.savefig(outdir / "Figure5_Volume_by_Country.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        return
    labels = [COUNTRY_FULL[cc].replace("United States of America", "USA") for cc in COUNTRY_ORDER]
    data = [d.loc[d.CountryCode == cc, VOL_COL].values for cc in COUNTRY_ORDER]
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor="black"),
               medianprops=dict(color="#f59e0b", linewidth=1.5))
    rng = np.random.default_rng(0)
    for i, cc in enumerate(COUNTRY_ORDER):
        s = d[d.CountryCode == cc]
        jitter = rng.uniform(-0.06, 0.06, len(s))
        colors = [OUTCOME_COLORS.get(o, "#9ca3af") for o in s.prior_outcome]
        ax.scatter(i + 1 + jitter, s[VOL_COL], s=16, alpha=0.75, color=colors, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Country of Manufacture")
    ax.set_ylabel("Market Volume (Extended Units)")
    for i, cc in enumerate(COUNTRY_ORDER):
        _n_label(ax, i + 1, int((d.CountryCode == cc).sum()))
    log(f"\n[Market Volume (Extended Units) by country] n={len(d)}, by country: "
        f"{ {cc: int((d.CountryCode==cc).sum()) for cc in COUNTRY_ORDER} }")
    pairwise_group_tests(log, d, VOL_COL, "CountryCode", COUNTRY_ORDER, fei_col="matched_fei")
    m = d.copy()
    m["IND"] = (m.CountryCode == "IND").astype(float)
    m["CHN"] = (m.CountryCode == "CHN").astype(float)
    m["_y"] = np.log(m[VOL_COL].astype(float))
    modelB_re_twoway(log, m, "_y", ["IND", "CHN"], "NDC11", "matched_fei",
                      "log(Market Volume (Extended Units)), ref=USA")
    m2 = d.copy()
    m2["USA_d"] = (m2.CountryCode == "USA").astype(float)
    m2["CHN_d"] = (m2.CountryCode == "CHN").astype(float)
    m2["_y"] = np.log(m2[VOL_COL].astype(float))
    modelB_re_twoway(log, m2, "_y", ["USA_d", "CHN_d"], "NDC11", "matched_fei",
                      "log(Market Volume (Extended Units)), ref=IND")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=OUTCOME_COLORS[o],
                           markersize=7, label=o) for o in OUTCOME_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(OUTCOME_ORDER), title="Prior Inspection Outcome",
               bbox_to_anchor=(0.5, -0.06), frameon=True, fontsize=8, title_fontsize=8)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(outdir / "Figure5_Volume_by_Country.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def figS1(df, outdir, log):
    d = df[df.months_since_inspection.notna() & df.prior_outcome.notna()].copy()
    if len(d) < 3:
        log("\n[Figure S1] insufficient data"); return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(d.months_since_inspection.astype(float), bins=15, color="#3b82f6", alpha=0.75, edgecolor="white")
    ax.axvline(36, color="#ef4444", linestyle="--", label="36 months")
    ax.set_xlabel("Months since last inspection"); ax.set_ylabel("Count")
    ax.set_title(f"Figure S1 — n={len(d)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "FigureS1_Months_Since_Inspection.png", dpi=150); plt.close(fig)
    log(f"\n[Figure S1] n={len(d)}, mean={d.months_since_inspection.astype(float).mean():.1f}, "
        f"median={d.months_since_inspection.astype(float).median():.1f}, "
        f">36mo: {int((d.months_since_inspection.astype(float)>36).sum())}")


# ── driver ────────────────────────────────────────────────────────────────────
def _redica_country_lookup():
    """FEI -> country code, derived only from Redica's FDA inspection Site
    Display Name -- never the Q&A spreadsheet's independent CountryCode column.

    That spreadsheet column traces to Valisure's own self-reported "Mfr
    location" field on the raw testing sheets (e.g. Valisure_2024_raw_prices_
    20260728.xlsx, "2022 Testing Data - Actual" sheet), filled from the
    product label at test time -- not derived from any facility match. A
    facility with zero Redica inspection events (e.g. Chartwell Congers,
    FEI 3008897678) has no entry here even if step1 matched it to an NDC, so
    using this lookup rather than the panel's own CountryCode column is the
    only way to guarantee every plotted country is Redica-derived. Mirrors
    step2_build_panel_july26.py's fei_to_country_code construction exactly."""
    import ast, re
    ev = pd.read_excel(BASE / "Data/07 - Redica/raw/MetfrmoinValisure_Red_Flag_Events_RedicaJuly26.xlsx")
    fm = pd.read_excel(BASE / "Data/07 - Redica/raw/MetfrmoinValisure_FEI_RedicaID_Mapping_RedicaJuly26.xlsx", dtype=str)
    id_to_fei = dict(zip(fm["Redica ID"].str.strip(), fm["All FEIs"].str.strip()))
    ev["FEI"] = ev["Site Redica Id"].map(id_to_fei)
    def parse_list(x):
        try:
            return ast.literal_eval(x) if pd.notna(x) and str(x).strip().startswith("[") else []
        except Exception:
            return []
    ev["agency"] = ev["Agency List"].apply(parse_list)
    ev["event_dt"] = pd.to_datetime(ev["Event Date"], errors="coerce")
    df_fda = ev[(ev["Event Type"] == "Inspection") & ev["agency"].apply(lambda a: "US - FDA" in a)
                & ev["event_dt"].notna()]
    cmap = {"India": "IND", "China": "CHN", "United States": "USA", "United States of America": "USA",
            "Canada": "CAN", "Bangladesh": "BGD"}
    def extract(s):
        if not isinstance(s, str):
            return None
        m = re.search(r"\[.+?\s*/\s*(.+?)\]", s)
        return cmap.get(m.group(1).strip()) if m else None
    out = {}
    for fei, grp in df_fda.dropna(subset=["FEI"]).groupby("FEI"):
        out[fei] = extract(grp["Site Display Name"].iloc[0])
    return out


def _matched_fei_lookup(map_label):
    """NDC11 -> one manufacturing FEI, independent of inspection history.
    prior_fei only exists where a prior inspection was found (null for the 31
    country-known-but-no-history rows the max-sample policy now includes), so
    Figure 4's FEI-clustering needs this separate, always-available key.
    Multi-plant NDCs (11 of them) get their first assigned FEI; the choice of
    which of the two plants only affects which cluster the row falls in, not
    whether it is clustered at all."""
    m = pd.read_csv(PROC / f"step1_ndc_fei_map_{map_label}.csv", dtype=str)
    m = m.dropna(subset=["FEI"]).sort_values(["NDC11", "FEI"])
    def n11(x):
        p = str(x).split("-"); return f"{p[0].zfill(5)}-{p[1].zfill(4)}-{p[2].zfill(2)}"
    m["NDC11"] = m["NDC11"].map(n11)
    return m.drop_duplicates("NDC11", keep="first").set_index("NDC11")["FEI"].to_dict()


VALISURE_FORM_FILE = BASE / "Data/08 - Valisure/raw/Valisure_2024_raw_prices_20260728_f1-and-formulation_20260813.xlsx"


def _valisure_formulation_lookup():
    """(NDC11, TestYear) -> 'IR'/'ER', from Valisure's own per-year testing
    sheets (the formulation column added at our request) rather than the
    static NDC8 -> FDA product.csv DOSAGEFORMNAME join used previously. The
    2024 sheet's real header is on the second row. Verified before switching:
    every NDC11 that appears in more than one year keeps the same formulation
    (no reformulation in this data), and every row present in both the old
    static join and this file agreed."""
    sheets = {
        "Copy of 2020 Testing Data": (2020, 0, "Formulation"),
        "2022 Testing Data - Actual": (2022, 0, "Formulation"),
        "2024 Testing Data": (2024, 1, "formulation"),
    }
    frames = []
    for sheet, (year, header, col) in sheets.items():
        d = pd.read_excel(VALISURE_FORM_FILE, sheet_name=sheet, header=header)
        d = d[["NDC11", col]].rename(columns={col: "IR_ER"})
        d["TestYear"] = year
        frames.append(d)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["IR_ER"])
    out["NDC11"] = out["NDC11"].astype(str).str.strip()
    out = out.drop_duplicates(["NDC11", "TestYear"])
    return {(r.NDC11, r.TestYear): r.IR_ER for r in out.itertuples()}


_REDICA_COUNTRY_CACHE = None
_VALISURE_FORM_CACHE = None


def build(map_label, dosage):
    global _REDICA_COUNTRY_CACHE, _VALISURE_FORM_CACHE
    if _REDICA_COUNTRY_CACHE is None:
        _REDICA_COUNTRY_CACHE = _redica_country_lookup()
    if _VALISURE_FORM_CACHE is None:
        _VALISURE_FORM_CACHE = _valisure_formulation_lookup()

    df = pd.read_csv(VDIR / f"step5_{map_label}.csv")
    df = df[df[[DMF_COL, NDMA_COL, DIFF_COL]].notna().any(axis=1)]  # Valisure-tested rows only
    df["matched_fei"] = df["NDC11"].map(_matched_fei_lookup(map_label))
    # Overwrite the step5 CSV's static IR_ER (NDC8 -> product.csv, one value
    # per NDC, no year granularity) with Valisure's own per-year formulation.
    df["IR_ER"] = list(zip(df["NDC11"].astype(str).str.strip(), df["TestYear"]))
    df["IR_ER"] = df["IR_ER"].map(_VALISURE_FORM_CACHE)
    # Overwrite CountryCode with a Redica-derived-only value. The panel's own
    # CountryCode (built in step2) falls back to the Q&A spreadsheet's
    # independent CountryCode column whenever the matched FEI has no Redica
    # inspection coverage, and that spreadsheet field traces to Valisure's own
    # self-reported "Mfr location" on the raw testing sheets -- not to any
    # facility match. Every figure that colors or groups by country must use
    # this column, or it can silently plot a country nothing here actually
    # confirmed.
    df["CountryCode"] = df["matched_fei"].map(_REDICA_COUNTRY_CACHE)
    if dosage != "all":
        df = df[df.IR_ER == dosage]
    outdir = OUT / f"{map_label}_{dosage}"
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "stats_log.txt"
    lines = [f"=== {map_label} / {dosage} ===",
             f"rows={len(df)}  NDC11s={df.NDC11.nunique()}  "
             f"Redica-confirmed country={df.CountryCode.notna().sum()}  not={df.CountryCode.isna().sum()}"]
    def log(s): lines.append(str(s))
    fig1(df, outdir, log)
    fig2_3(df, outdir, log, VOL_COL, "Market Volume (Extended Units)", "Figure2_Volume_vs_Quality")
    fig2_3(df, outdir, log, PRICE_COL, "Price per Unit ($)", "Figure3_Price_vs_Quality")
    fig4(df, outdir, log)
    fig5(df, outdir, log)
    figS1(df, outdir, log)
    logpath.write_text("\n".join(lines))
    print(f"{map_label:10s} {dosage:4s}  rows={len(df):4d}  NDC11s={df.NDC11.nunique():3d}  -> {outdir}")


if __name__ == "__main__":
    for m in ["rulebased", "manual"]:
        for dosage in ["all", "IR", "ER"]:
            build(m, dosage)
# %%
