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

Usage:
  python 04_ae_trajectory_clustering.py --granularity inspection   # default
  python 04_ae_trajectory_clustering.py --granularity yearly

Granularity:
  inspection  : uses fei_ae_panel_inspection_centered.parquet
                t in quarters (tm4..tp4 = ±4 quarters = ±1 year)
                consistent with correlation and model slides
  yearly      : uses fei_ae_panel.parquet
                t in years (tm2..t2 = ±2 years)

Outputs (suffix _insp or _yearly added to filenames)
───────
  outputs/tables/ae_cluster_summary_{gran}.csv
  outputs/tables/ae_cluster_fei_list_{gran}.csv
  outputs/figures/ae_cluster_trajectories_{gran}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

HERE     = Path(__file__).resolve().parent
OUT      = HERE / "outputs"
OUT_TABS = OUT / "tables"
OUT_FIGS = OUT / "figures"

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


def _panel_config(granularity: str, anda_ae: bool = False) -> dict:
    """Return panel path and column config for the given granularity."""
    if granularity == "inspection":
        fname = "fei_ae_panel_inspection_centered_anda.parquet" if anda_ae else \
                "fei_ae_panel_inspection_centered.parquet"
        return {
            "path":     OUT / fname,
            "ae_cols":  ["n_ae_tm4", "n_ae_tm3", "n_ae_tm2", "n_ae_tm1",
                         "n_ae_t0",
                         "n_ae_tp1", "n_ae_tp2", "n_ae_tp3", "n_ae_tp4"],
            "col_start":  "n_ae_tm4",
            "col_end":    "n_ae_tp4",
            "x_pre":    [-4, -3, -2, -1, 0],
            "x_post":   [0, 1, 2, 3, 4],
            "xlabels":  ["t−4", "t−2", "t=0\n(insp.)", "t+2", "t+4"],
            "xl_idx":   [0, 2, 4, 6, 8],   # indices into ae_cols for plot ticks
            "suffix":   "insp",
            "unit":     "quarters",
        }
    else:
        return {
            "path":     OUT / "fei_ae_panel.parquet",
            "ae_cols":  ["n_ae_tm2", "n_ae_tm1", "n_ae_t0", "n_ae_t1", "n_ae_t2"],
            "col_start":  "n_ae_tm2",
            "col_end":    "n_ae_t2",
            "x_pre":    [-2, -1, 0],
            "x_post":   [0, 1, 2],
            "xlabels":  ["t−2", "t−1", "t=0\n(insp.)", "t+1", "t+2"],
            "xl_idx":   [0, 1, 2, 3, 4],
            "suffix":   "yearly",
            "unit":     "years",
        }


# ── Build per-FEI trajectory features ────────────────────────────────────────

def _slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope; returns NaN if fewer than 2 valid points."""
    pairs = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
    if len(pairs) < 2:
        return np.nan
    xs_, ys_ = zip(*pairs)
    return linregress(xs_, ys_).slope


def _build_fei_trajectories(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Collapse panel to one row per FEI.
    Trajectory features computed from median across all inspection events
    (or panel years) for each FEI.
    """
    ae_cols    = [c for c in cfg["ae_cols"] if c in panel.columns]
    col_start  = cfg["col_start"]
    col_end    = cfg["col_end"]
    x_pre      = cfg["x_pre"]
    x_post     = cfg["x_post"]

    keep = ["fei", "any_oai", "n_oai", "n_vai"] + ae_cols + TEXT_FEATURES
    keep = [c for c in keep if c in panel.columns]
    df = panel[keep].copy()

    rows = []
    for fei, grp in df.groupby("fei"):
        m = {c: grp[c].median() for c in ae_cols if c in grp.columns}

        # pre-inspection slope (all points from start to t0)
        pre_ys = [m.get(c, np.nan) for c in ae_cols if c in ae_cols[:ae_cols.index("n_ae_t0") + 1]]
        slope_pre = _slope(x_pre, pre_ys[-len(x_pre):])

        # post-inspection slope (all points from t0 to end)
        post_ys = [m.get(c, np.nan) for c in ae_cols if c in ae_cols[ae_cols.index("n_ae_t0"):]]
        slope_post = _slope(x_post, post_ys[:len(x_post)])

        t0    = m.get("n_ae_t0",   np.nan)
        t_s   = m.get(col_start,   np.nan)
        t_e   = m.get(col_end,     np.nan)

        persist  = (t_e / t0) if (t0  and t0  > 0) else np.nan
        pre_rise = (t0  / t_s) if (t_s and t_s > 0) else np.nan

        any_oai = grp["any_oai"].max() if "any_oai" in grp.columns else 0
        n_oai   = grp["n_oai"].sum()   if "n_oai"   in grp.columns else 0

        tech_vals  = [grp[c].median() for c in _TECH_FEATURES if c in grp.columns]
        tech_score = np.nanmean(tech_vals) if tech_vals else np.nan

        text_means = {f: grp[f].median() for f in TEXT_FEATURES if f in grp.columns}

        row = {
            "fei":           fei,
            "n_rows":        len(grp),
            "any_oai":       any_oai,
            "n_oai":         n_oai,
            **{c: round(v, 1) for c, v in m.items()},
            "slope_pre":     round(slope_pre,  1) if not np.isnan(slope_pre)  else np.nan,
            "slope_post":    round(slope_post, 1) if not np.isnan(slope_post) else np.nan,
            "persist":       round(persist,    3) if not np.isnan(persist)    else np.nan,
            "pre_rise":      round(pre_rise,   3) if not np.isnan(pre_rise)   else np.nan,
            "tech_score":    round(tech_score, 3) if not np.isnan(tech_score) else np.nan,
            **{f: round(v, 3) for f, v in text_means.items()},
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Cluster by trajectory shape ───────────────────────────────────────────────

_CLUSTER_FEATURES = ["slope_pre", "slope_post", "persist", "pre_rise"]


def _cluster(fei_traj: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = fei_traj.copy()
    feat_cols = [c for c in _CLUSTER_FEATURES if c in df.columns]
    X = df[feat_cols].copy()

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

        cluster_profiles = (
            df.groupby("cluster_raw")[feat_cols].mean()
              .assign(mean_ae_t0=df.groupby("cluster_raw")["n_ae_t0"].mean())
        )
        print("\nRaw cluster profiles:")
        print(cluster_profiles.round(2).to_string())

        persist_order = cluster_profiles["persist"].sort_values(ascending=False).index.tolist()
        name_map = {raw: name for raw, name in zip(
            persist_order, ["Flat/rising", "Spike-correct", "Slow-decline", "Flat-low"]
        )}
        df["cluster"] = df["cluster_raw"].map(name_map)

    except ModuleNotFoundError:
        print("scikit-learn not found — assigning rule-based clusters instead.")
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

def plot_cluster_trajectories(fei_traj: pd.DataFrame, cfg: dict, out_path: Path) -> None:
    ae_cols  = [c for c in cfg["ae_cols"] if c in fei_traj.columns]
    xl_idx   = cfg["xl_idx"]
    xlabels  = cfg["xlabels"]
    x        = np.arange(len(ae_cols))
    clusters = fei_traj["cluster"].dropna().unique()
    colors   = plt.cm.tab10(np.linspace(0, 0.8, len(clusters)))

    fig, ax = plt.subplots(figsize=(7, 4))
    t0_idx = ae_cols.index("n_ae_t0") if "n_ae_t0" in ae_cols else 0
    ax.axvline(x=t0_idx, color="gray", linewidth=1.0, linestyle="--",
               alpha=0.5, label="Inspection")

    for clr, cluster in zip(colors, sorted(clusters)):
        sub  = fei_traj[fei_traj["cluster"] == cluster]
        means = [sub[c].mean() if c in sub.columns else np.nan for c in ae_cols]
        ax.plot(x, means, marker="o", linewidth=2, color=clr,
                label=f"{cluster} (n={len(sub)})")

    ax.set_xticks([x[i] for i in xl_idx])
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Median serious AEs per FEI")
    ax.set_title(
        f"AE trajectory clusters ({cfg['unit']}, centered on inspection)\n"
        "Clustered by slope_pre, slope_post, persist, pre_rise", fontsize=9
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster trajectory plot → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularity", choices=["yearly", "inspection"], default="inspection",
        help="Panel to use: inspection-centered quarterly (default) or yearly"
    )
    parser.add_argument(
        "--anda-ae", dest="anda_ae", action="store_true",
        help="Use ANDA-specific AE panel (only applies to inspection granularity)"
    )
    args = parser.parse_args()

    cfg = _panel_config(args.granularity, anda_ae=args.anda_ae)
    suffix = cfg["suffix"] + ("_anda" if args.anda_ae else "")

    if not cfg["path"].exists():
        raise FileNotFoundError(
            f"Panel not found: {cfg['path']}\n"
            "Run 01_build_fei_ae_panel.py first."
        )

    print(f"Granularity: {args.granularity}  (t in {cfg['unit']})")
    print(f"Loading panel: {cfg['path'].name}")
    panel = pd.read_parquet(cfg["path"])
    print(f"  {len(panel)} rows, {panel['fei'].nunique()} FEIs")

    print("\nBuilding per-FEI trajectory features…")
    fei_traj = _build_fei_trajectories(panel, cfg)
    print(f"  {len(fei_traj)} FEIs with trajectory data")
    print(f"\n  Median AE window across FEIs:")
    for c in cfg["ae_cols"]:
        if c in fei_traj.columns:
            print(f"    {c}: {fei_traj[c].median():.0f}")

    print("\nClustering by trajectory shape…")
    fei_traj = _cluster(fei_traj, k=4)

    summary_cols = (
        ["cluster"] +
        [c for c in cfg["ae_cols"] if c in fei_traj.columns] +
        ["slope_pre", "slope_post", "persist", "pre_rise", "tech_score", "any_oai", "n_oai"]
    )
    summary_cols = [c for c in summary_cols if c in fei_traj.columns]
    cluster_summary = (
        fei_traj.groupby("cluster")[summary_cols[1:]]
                .mean()
                .round(3)
                .reset_index()
    )
    cluster_summary.insert(1, "n_feis", fei_traj.groupby("cluster").size().values)

    print(f"\nCluster summary ({args.granularity}):")
    print(cluster_summary.to_string(index=False))

    flat_rising = fei_traj[fei_traj["cluster"] == "Flat/rising"].sort_values(
        "tech_score", ascending=False
    )
    if not flat_rising.empty:
        print(f"\nFlat/rising cluster FEIs (sorted by technical signal):")
        show_cols = ["fei", "any_oai", "n_ae_t0", cfg["col_end"], "persist",
                     "tech_score", "vc_labcontrols_share", "data_integrity_llm_share"]
        show_cols = [c for c in show_cols if c in flat_rising.columns]
        print(flat_rising[show_cols].to_string(index=False))

    OUT_TABS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    cluster_summary.to_csv(OUT_TABS / f"ae_cluster_summary_{suffix}.csv", index=False)
    fei_traj.to_csv(OUT_TABS / f"ae_cluster_fei_list_{suffix}.csv", index=False)
    print(f"\nSaved cluster tables → {OUT_TABS}/")

    plot_cluster_trajectories(
        fei_traj, cfg, OUT_FIGS / f"ae_cluster_trajectories_{suffix}.png"
    )


if __name__ == "__main__":
    main()
