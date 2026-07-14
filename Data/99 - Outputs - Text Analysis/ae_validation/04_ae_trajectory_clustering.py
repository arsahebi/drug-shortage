"""
04_ae_trajectory_clustering.py
────────────────────────────────────────────────────────────────────────────
Reverse-flow analysis: cluster facilities by their AE time series shape,
then characterize each cluster by inspection outcomes and text signals.

Forward flow (scripts 01–03): text signals → predict AE outcome
Reverse flow (this script):   AE trajectory shape → characterize by inspection

Why reverse?
  The forward model asks "do 483 signals predict harm?" This script asks
  "given that a facility has a concerning AE pattern, what does its
  regulatory history look like?" This catches cases where:
  - AEs were rising before the inspection (pre-distribution)
  - AEs persisted after inspection (no forced correction / strategic leniency)
  - AEs spiked then dropped (OAI + correction working)

Trajectory features per FEI (from FAERS FEI×year):
  slope_pre   : OLS slope of AEs over t-2 → t0 (rising = positive)
  slope_post  : OLS slope of AEs over t0 → t+2 (declining = negative)
  peak_at_t0  : whether max AE is at inspection year
  persist     : n_ae_t2 / n_ae_t0 (>1 = sustained or rising post-inspection)
  pre_rise    : n_ae_t0 / n_ae_tm2 (>1 = elevated at inspection vs 2yr prior)

Clusters (KMeans, k=4):
  - Spike-correct:  high pre-rise, strong post-decline (OAI working)
  - Flat-high:      elevated across all years, slow decline (strategic leniency?)
  - Flat-low:       low AEs throughout, low signals
  - Rising:         AEs increasing through t+2 (possibly missed by FDA)

Outputs
───────
  outputs/tables/ae_cluster_summary.csv     — cluster × mean features
  outputs/tables/ae_cluster_fei_list.csv    — every FEI with cluster label + signals
  outputs/figures/ae_cluster_trajectories.png — trajectory plot per cluster
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

HERE    = Path(__file__).resolve().parent
OUT     = HERE / "outputs"
OUT_TABS = OUT / "tables"
OUT_FIGS = OUT / "figures"
PANEL   = OUT / "fei_ae_panel.parquet"

TEXT_FEATURES = [
    "severity_critmajor_share",
    "contamination_llm_share",
    "data_integrity_llm_share",
    "patient_risk_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "scope_facilitywide_share",
    "cultural_root_cause_share",
    "vc_labcontrols_share",
    "vc_qualitysystem_share",
    "n_labcontrols_obs",
    "n_qualitysystem_obs",
    "joint_labcontrols_qualitysystem",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "joint_qualitysystem_production",
    "multi_domain_insp",
]

_TECH_FEATURES = [
    "vc_labcontrols_share",
    "data_integrity_llm_share",
    "joint_labcontrols_dataintegrity",
    "joint_contamination_labcontrols",
    "n_labcontrols_obs",
]

SEED = 42


# ── Build per-FEI trajectory features ────────────────────────────────────────

def _slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope; returns NaN if fewer than 2 valid points."""
    pairs = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
    if len(pairs) < 2:
        return np.nan
    xs_, ys_ = zip(*pairs)
    return linregress(xs_, ys_).slope


def _build_fei_trajectories(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse panel to one row per FEI.
    Trajectory features are computed from the median inspection snapshot
    (most FEIs appear in multiple panel years; we want a stable estimate).
    """
    ae_cols = ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2"]
    # keep only ae and text columns + identifiers
    keep = ["fei", "any_oai", "n_oai", "n_vai"] + ae_cols + TEXT_FEATURES
    keep = [c for c in keep if c in panel.columns]
    df = panel[keep].copy()

    rows = []
    for fei, grp in df.groupby("fei"):
        # median AE at each lag point across all panel years for this FEI
        m = {c: grp[c].median() for c in ae_cols if c in grp.columns}

        # trajectory shape features
        slope_pre  = _slope([-2, -1, 0],
                            [m.get("n_ae_tm2", np.nan),
                             m.get("n_ae_tm1", np.nan),
                             m.get("n_ae_t0",  np.nan)])
        slope_post = _slope([0, 1, 2],
                            [m.get("n_ae_t0", np.nan),
                             m.get("n_ae_t1", np.nan),
                             m.get("n_ae_t2", np.nan)])

        t0   = m.get("n_ae_t0",  np.nan)
        tm2  = m.get("n_ae_tm2", np.nan)
        t2   = m.get("n_ae_t2",  np.nan)
        persist  = (t2  / t0)  if (t0  and t0  > 0) else np.nan
        pre_rise = (t0  / tm2) if (tm2 and tm2 > 0) else np.nan

        # regulatory
        any_oai = grp["any_oai"].max() if "any_oai" in grp.columns else 0
        n_oai   = grp["n_oai"].sum()   if "n_oai"   in grp.columns else 0

        # text signal composite (mean of available tech features, median across years)
        tech_vals = [grp[c].median() for c in _TECH_FEATURES if c in grp.columns]
        tech_score = np.nanmean(tech_vals) if tech_vals else np.nan

        # mean of each text feature across panel years for this FEI
        text_means = {f: grp[f].median() for f in TEXT_FEATURES if f in grp.columns}

        row = {
            "fei":        fei,
            "n_panel_years": len(grp),
            "any_oai":    any_oai,
            "n_oai":      n_oai,
            **{c: round(v, 1) for c, v in m.items()},
            "slope_pre":  round(slope_pre,  1) if not np.isnan(slope_pre)  else np.nan,
            "slope_post": round(slope_post, 1) if not np.isnan(slope_post) else np.nan,
            "persist":    round(persist,    3) if not np.isnan(persist)    else np.nan,
            "pre_rise":   round(pre_rise,   3) if not np.isnan(pre_rise)   else np.nan,
            "tech_score": round(tech_score, 3) if not np.isnan(tech_score) else np.nan,
            **{f: round(v, 3) for f, v in text_means.items()},
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Cluster by trajectory shape ───────────────────────────────────────────────

_CLUSTER_FEATURES = ["slope_pre", "slope_post", "persist", "pre_rise"]
_CLUSTER_NAMES    = {
    0: "Spike-correct",
    1: "Flat-high",
    2: "Flat-low",
    3: "Rising",
}


def _cluster(fei_traj: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = fei_traj.copy()
    feat_cols = [c for c in _CLUSTER_FEATURES if c in df.columns]
    X = df[feat_cols].copy()

    # impute with median before clustering
    for c in feat_cols:
        X[c] = X[c].fillna(X[c].median())

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = km.fit_predict(Xs)
        df["cluster_raw"] = labels

        # Name clusters by their slope_post and persist profile
        cluster_profiles = (
            df.groupby("cluster_raw")[feat_cols].mean()
              .assign(mean_ae_t0=df.groupby("cluster_raw")["n_ae_t0"].mean())
        )
        print("\nRaw cluster profiles:")
        print(cluster_profiles.round(2).to_string())

        # Sort clusters by persist descending so cluster 0 = most persistent
        persist_order = cluster_profiles["persist"].sort_values(ascending=False).index.tolist()
        name_map = {raw: name for raw, name in zip(persist_order, ["Flat/rising", "Spike-correct", "Slow-decline", "Flat-low"])}
        df["cluster"] = df["cluster_raw"].map(name_map)

    except ModuleNotFoundError:
        print("scikit-learn not found — assigning rule-based clusters instead.")
        # Rule-based fallback: segment by slope_post and persist
        def _rule(row):
            if pd.isna(row.get("persist")) or pd.isna(row.get("slope_post")):
                return "Unknown"
            if row["slope_post"] < -100 and row.get("pre_rise", 1) > 1.1:
                return "Spike-correct"
            if row["persist"] > 0.95:
                return "Flat/rising"
            if row["slope_post"] > 0:
                return "Rising"
            return "Slow-decline"
        df["cluster"] = df.apply(_rule, axis=1)

    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cluster_trajectories(fei_traj: pd.DataFrame, out_path: Path) -> None:
    ae_cols  = ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2"]
    xlabels  = ["t−2", "t−1", "t=0\n(inspection)", "t+1", "t+2"]
    x        = np.arange(len(ae_cols))
    clusters = fei_traj["cluster"].unique()
    colors   = plt.cm.tab10(np.linspace(0, 0.8, len(clusters)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvline(x=2, color="gray", linewidth=1.0, linestyle="--", alpha=0.5, label="Inspection")

    for clr, cluster in zip(colors, sorted(clusters)):
        sub = fei_traj[fei_traj["cluster"] == cluster]
        means = [sub[c].mean() if c in sub.columns else np.nan for c in ae_cols]
        n = len(sub)
        ax.plot(x, means, marker="o", linewidth=2, color=clr,
                label=f"{cluster} (n={n})")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Median serious AEs per FEI")
    ax.set_title("AE trajectory clusters (centered on inspection year)\n"
                 "Clustered by slope_pre, slope_post, persist, pre_rise", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster trajectory plot → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PANEL.exists():
        raise FileNotFoundError(f"Panel not found: {PANEL}\nRun 01_build_fei_ae_panel.py first.")

    print("Loading panel…")
    panel = pd.read_parquet(PANEL)
    print(f"  {len(panel)} rows, {panel['fei'].nunique()} FEIs")

    print("\nBuilding per-FEI trajectory features…")
    fei_traj = _build_fei_trajectories(panel)
    print(f"  {len(fei_traj)} FEIs with trajectory data")
    print(f"\n  Overall AE window (median across FEIs):")
    for c in ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2"]:
        if c in fei_traj.columns:
            print(f"    {c}: {fei_traj[c].median():.0f}")

    print("\nClustering by trajectory shape…")
    fei_traj = _cluster(fei_traj, k=4)

    # Summary by cluster
    summary_cols = (
        ["cluster", "n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2",
         "slope_pre", "slope_post", "persist", "pre_rise",
         "tech_score", "any_oai", "n_oai"]
    )
    summary_cols = [c for c in summary_cols if c in fei_traj.columns]
    cluster_summary = (
        fei_traj.groupby("cluster")[summary_cols[1:]]
                .mean()
                .round(3)
                .reset_index()
    )
    cluster_summary.insert(1, "n_feis", fei_traj.groupby("cluster").size().values)

    print(f"\nCluster summary:")
    print(cluster_summary.to_string(index=False))

    # Flat/rising cluster deep-dive (strategic leniency candidates)
    flat_rising = fei_traj[fei_traj["cluster"] == "Flat/rising"].sort_values(
        "tech_score", ascending=False
    )
    if not flat_rising.empty:
        print(f"\nFlat/rising cluster FEIs (sorted by technical signal):")
        show_cols = ["fei", "any_oai", "n_ae_t0", "n_ae_t2", "persist",
                     "tech_score", "vc_labcontrols_share", "data_integrity_llm_share"]
        show_cols = [c for c in show_cols if c in flat_rising.columns]
        print(flat_rising[show_cols].to_string(index=False))

    OUT_TABS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    cluster_summary.to_csv(OUT_TABS / "ae_cluster_summary.csv", index=False)
    fei_traj.to_csv(OUT_TABS / "ae_cluster_fei_list.csv", index=False)
    print(f"\nSaved cluster tables → {OUT_TABS}/")

    plot_cluster_trajectories(fei_traj, OUT_FIGS / "ae_cluster_trajectories.png")


if __name__ == "__main__":
    main()
