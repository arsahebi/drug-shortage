# %%
"""
Build max-sample figures for both NDC-FEI linkage methods (manual, rule-based),
each pooled and split by dosage form (IR / ER).

Per John's guidance (Sept 17 2026 meeting): each figure uses whatever sample its
own axes require, with no additional restriction beyond the standing Canada /
Bangladesh exclusion. Figures 1 and 4 still need prior_outcome / CountryCode
respectively and so are naturally limited to facility-linked rows; Figures 2, 3
and S1 use every row that has the fields they plot, facility-linked or not.

Inputs: variants/step5_{manual,rulebased}.csv (built by running step2-step5
with REQUIRE_REDICA_HISTORY=0 DROP_NDCS_WITHOUT_FEI=0 against each step1 map;
see the run commands in the Sept 20 2026 session). Each carries an IR_ER column
joined from the FDA NDC directory's DOSAGEFORMNAME.

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
        return {"rho": rho_obs, "p_naive": p_obs, "p_boot": np.nan, "n_obs": int(mask.sum()), "n_clusters": n_cl}
    boot_rhos = np.array(boot_rhos)
    shifted = boot_rhos - np.mean(boot_rhos)
    p_boot = max(float(np.mean(np.abs(shifted) >= abs(rho_obs))), 1.0 / n_boot)
    return {"rho": rho_obs, "p_naive": p_obs, "p_boot": p_boot, "n_obs": int(mask.sum()), "n_clusters": n_cl}


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
        return
    res = _block_bootstrap_spearman(d[x_col].values.astype(float), d[y_col].values.astype(float), d[cluster_col].values)
    sig = " **" if res["p_boot"] < 0.01 else (" *" if res["p_boot"] < 0.05 else "")
    log(f"  n={res['n_obs']} (NDCs={res['n_clusters']})  rho={res['rho']:+.4f}  "
        f"p_naive={res['p_naive']:.4f}  p_boot={res['p_boot']:.4f}{sig}")


# ── figure builders ───────────────────────────────────────────────────────────
def fig1(df, outdir, log):
    d = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in [(axes[0], PRICE_COL, "Medicaid price ($/unit)"),
                            (axes[1], VOL_COL, "IQVIA extended units")]:
        sub = d[d["prior_outcome"].notna() & d[col].notna() & (d[col] > 0)]
        if col == PRICE_COL:
            sub = sub[sub.get("price_outlier", 0) == 0]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}  (n=0)")
            log(f"\n[{title}] n=0"); continue
        data = [sub.loc[sub.prior_outcome == o, col].values for o in OUTCOME_ORDER]
        bp = ax.boxplot(data, labels=OUTCOME_ORDER, showfliers=False, patch_artist=True)
        for patch, o in zip(bp["boxes"], OUTCOME_ORDER):
            patch.set_facecolor(OUTCOME_COLORS[o]); patch.set_alpha(0.35)
        for i, o in enumerate(OUTCOME_ORDER):
            y = sub.loc[sub.prior_outcome == o, col].values
            ax.scatter(np.random.default_rng(0).normal(i + 1, 0.05, len(y)), y, s=14, alpha=0.6, color=OUTCOME_COLORS[o])
        if col == VOL_COL:
            ax.set_yscale("log")
        ax.set_title(f"{title}  (n={len(sub)})")
        log(f"\n[{title}] n={len(sub)}, by outcome: "
            f"{ {o: int((sub.prior_outcome==o).sum()) for o in OUTCOME_ORDER} }")
        pairwise_group_tests(log, sub, col, "prior_outcome", OUTCOME_ORDER, fei_col="prior_fei")
    fig.suptitle("Figure 1 — Price and volume by prior inspection outcome (facility-linked rows only)")
    fig.tight_layout()
    fig.savefig(outdir / "Figure1_Price_Volume_by_Outcome.png", dpi=150); plt.close(fig)


def fig2_3(df, outdir, log, x_col, label, fname):
    d = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [(DMF_COL, "DMF (ng/day)", "symlog"), (NDMA_COL, "NDMA (ng/day)", "symlog"),
               (DIFF_COL, "Difference Factor", "linear")]
    for ax, (col, ylab, yscale) in zip(axes, metrics):
        sub = d[d[col].notna() & d[x_col].notna() & (d[x_col] > 0)]
        if x_col == PRICE_COL:
            sub = sub[sub.get("price_outlier", 0) == 0]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("n=0")
            log(f"\n[{ylab} vs {label}] n=0")
            continue
        known = sub[sub.CountryCode.notna()]
        unknown = sub[sub.CountryCode.isna()]
        for cc in COUNTRY_ORDER:
            s = known[known.CountryCode == cc]
            if len(s):
                ax.scatter(s[x_col], s[col], s=18, alpha=0.7, color=COUNTRY_COLORS[cc], label=cc)
        if len(unknown):
            ax.scatter(unknown[x_col], unknown[col], s=18, alpha=0.5, color="#9ca3af", label="unmatched facility")
        ax.set_xscale("log")
        if (sub[col] > 0).any():
            ax.set_yscale(yscale)
        ax.set_xlabel(label); ax.set_ylabel(ylab)
        ax.set_title(f"n={len(sub)}")
        log(f"\n[{ylab} vs {label}] n={len(sub)} (NDCs={sub.NDC11.nunique()}, "
            f"{sub.CountryCode.notna().sum()} facility-linked, {sub.CountryCode.isna().sum()} not)")
        correlation_tests(log, sub, x_col, col)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{fname.split('_')[0]} — Quality vs {label} (maximum available sample; grey = no facility match)")
    fig.tight_layout()
    fig.savefig(outdir / f"{fname}.png", dpi=150); plt.close(fig)


def fig4(df, outdir, log):
    # Country here means the manufacturing facility's country, so a row only
    # belongs in this figure if a facility was actually matched (matched_fei
    # notna). Some rows carry CountryCode from the old Q&A spreadsheet's own
    # country field even with zero matched FEIs (n_feis == 0) -- an
    # independent, untraceable source, not evidence of a facility link -- and
    # are excluded here even though the max-sample policy keeps them for
    # Figures 2/3, which don't use country at all.
    d = df[df.CountryCode.isin(COUNTRY_ORDER) & df.matched_fei.notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [(DMF_COL, "DMF (ng/day)"), (NDMA_COL, "NDMA (ng/day)"), (DIFF_COL, "Difference Factor")]
    for ax, (col, ylab) in zip(axes, metrics):
        sub = d[d[col].notna()]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{ylab}  (n=0)")
            log(f"\n[{ylab} by country] n=0"); continue
        data = [sub.loc[sub.CountryCode == cc, col].values for cc in COUNTRY_ORDER]
        bp = ax.boxplot(data, labels=COUNTRY_ORDER, showfliers=False, patch_artist=True)
        for patch, cc in zip(bp["boxes"], COUNTRY_ORDER):
            patch.set_facecolor(COUNTRY_COLORS[cc]); patch.set_alpha(0.35)
        for i, cc in enumerate(COUNTRY_ORDER):
            y = sub.loc[sub.CountryCode == cc, col].values
            ax.scatter(np.random.default_rng(0).normal(i + 1, 0.05, len(y)), y, s=14, alpha=0.6, color=COUNTRY_COLORS[cc])
        ax.set_title(f"{ylab}  (n={len(sub)})")
        log(f"\n[{ylab} by country] n={len(sub)}, by country: "
            f"{ {cc: int((sub.CountryCode==cc).sum()) for cc in COUNTRY_ORDER} }")
        pairwise_group_tests(log, sub, col, "CountryCode", COUNTRY_ORDER, fei_col="matched_fei")
    fig.suptitle("Figure 4 — Quality by country of manufacture (facility-linked rows only)")
    fig.tight_layout()
    fig.savefig(outdir / "Figure4_Quality_by_Country.png", dpi=150); plt.close(fig)


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


def build(map_label, dosage):
    df = pd.read_csv(VDIR / f"step5_{map_label}.csv")
    df = df[df[[DMF_COL, NDMA_COL, DIFF_COL]].notna().any(axis=1)]  # Valisure-tested rows only
    df["matched_fei"] = df["NDC11"].map(_matched_fei_lookup(map_label))
    if dosage != "all":
        df = df[df.IR_ER == dosage]
    outdir = OUT / f"{map_label}_{dosage}"
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "stats_log.txt"
    lines = [f"=== {map_label} / {dosage} ===",
             f"rows={len(df)}  NDC11s={df.NDC11.nunique()}  "
             f"facility-linked={df.CountryCode.notna().sum()}  not={df.CountryCode.isna().sum()}"]
    def log(s): lines.append(str(s))
    fig1(df, outdir, log)
    fig2_3(df, outdir, log, VOL_COL, "IQVIA extended units", "Figure2_Volume_vs_Quality")
    fig2_3(df, outdir, log, PRICE_COL, "Medicaid price ($/unit)", "Figure3_Price_vs_Quality")
    fig4(df, outdir, log)
    figS1(df, outdir, log)
    logpath.write_text("\n".join(lines))
    print(f"{map_label:10s} {dosage:4s}  rows={len(df):4d}  NDC11s={df.NDC11.nunique():3d}  -> {outdir}")


if __name__ == "__main__":
    for m in ["rulebased", "manual"]:
        for dosage in ["all", "IR", "ER"]:
            build(m, dosage)
# %%
