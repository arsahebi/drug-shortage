"""
Module M13 — Chronological case studies for top text-covered FEIs.

Selects the 2 best candidates (text coverage × OAI history) and produces
three outputs per FEI to show the text signal is temporally honest:

  1. Signal trajectory figure: text severity/contamination scores at each
     snapshot date, overlaid with inspection outcome markers and shortage bands.
     Shows whether signals were elevated *before* OAI events.

  2. Before/after OAI comparison table: text feature mean in the snapshot
     immediately before vs after each OAI event at that facility.

  3. Lead table (all FEIs): for every FEI with a snapshot before an OAI,
     record snapshot date → OAI date → gap in months → key signal values.

Outputs:
  outputs/figures/m13_trajectory_<fei>.png      (one per case study FEI)
  outputs/tables/m13_before_after_oai.csv
  outputs/tables/m13_lead_table.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import DATA, OUT_DATA, OUT_FIGS, OUT_TABS, OUT_LOGS, VALISURE_FEI, REDICA_CSV  # noqa
from utils import get_logger, read_table  # noqa

log = get_logger("m13_case_study", OUT_LOGS / "m13_case_study.log")

TIMESERIES_CSV = DATA / "99 - Outputs - Text Analysis" / "483_fei_text_features_timeseries.csv"
INSP_XLSX      = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
UUTAH_MONTHLY  = OUT_DATA / "master_panel_monthly.parquet"

# Features shown on the trajectory figure
TRAJ_FEATS = {
    "severity_critmajor_share":    "Critical+Major severity",
    "contamination_llm_share":     "Contamination flag",
    "scope_facilitywide_share":    "Facility-wide scope",
    "remediation_none_share":      "No remediation",
}

# Features included in the lead table and before/after comparison
LEAD_FEATS = [
    "severity_critmajor_share", "contamination_llm_share",
    "scope_facilitywide_share", "remediation_none_share",
    "repeat_llm_only_share", "repeat_cross_insp_share",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    bridge = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping",
                           usecols=["API", "FEI_NUMBER"])
    bridge = bridge.dropna(subset=["FEI_NUMBER"])
    bridge["FEI_NUMBER"] = bridge["FEI_NUMBER"].astype(int)
    our_feis = set(bridge["FEI_NUMBER"].unique())
    fei_drugs = (bridge.groupby("FEI_NUMBER")["API"]
                 .apply(lambda x: ", ".join(sorted(x.unique())))
                 .to_dict())

    ts = pd.read_csv(TIMESERIES_CSV)
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    ts["fei"] = ts["fei"].astype(int)
    ts = ts[ts["fei"].isin(our_feis)].copy()

    insp = pd.read_excel(INSP_XLSX)
    insp["Inspection End Date"] = pd.to_datetime(insp["Inspection End Date"], errors="coerce")
    insp = insp[insp["FEI Number"].isin(our_feis) &
                insp["Product Type"].str.contains("Drug", na=False)].copy()

    # Redica for warning letters / additional OAI events
    redica = pd.read_csv(REDICA_CSV)
    redica["Event Date"] = pd.to_datetime(redica["Event Date"], errors="coerce")

    monthly = read_table(UUTAH_MONTHLY) if UUTAH_MONTHLY.exists() else None

    return ts, insp, redica, monthly, fei_drugs, our_feis


# ─────────────────────────────────────────────────────────────────────────────
# FEI ranking: select best case study candidates
# ─────────────────────────────────────────────────────────────────────────────

def rank_feis(ts: pd.DataFrame, insp: pd.DataFrame) -> pd.DataFrame:
    snap_counts = ts.groupby("fei").agg(
        n_snaps=("snapshot_date", "count"),
        first_snap=("snapshot_date", "min"),
        last_snap=("snapshot_date", "max"),
    )
    insp_counts = insp.groupby("FEI Number").agg(
        n_insp=("Inspection ID", "nunique"),
        n_oai=("Classification", lambda x: (x == "Official Action Indicated (OAI)").sum()),
    ).rename_axis("fei")
    merged = snap_counts.join(insp_counts, how="left").fillna(0)
    merged["coverage"] = merged["n_snaps"] / merged["n_insp"].replace(0, np.nan)
    merged["score"] = merged["n_snaps"] * (1 + merged["n_oai"])
    return merged.sort_values("score", ascending=False).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory(fei: int, ts: pd.DataFrame, insp: pd.DataFrame,
                    monthly: pd.DataFrame | None, fei_drugs: dict, out_dir: Path):
    snaps = ts[ts["fei"] == fei].sort_values("snapshot_date").copy()
    insp_fei = (insp[insp["FEI Number"] == fei]
                .sort_values("Inspection End Date").copy())
    drugs = fei_drugs.get(fei, "")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E07A5F", "#1C7293", "#2E8B57", "#8B4513"]

    for (col, label), color in zip(TRAJ_FEATS.items(), colors):
        if col not in snaps.columns:
            continue
        vals = snaps[col].fillna(0)
        ax.plot(snaps["snapshot_date"], vals, "o-", color=color,
                label=label, linewidth=1.8, markersize=7, zorder=3)

    # Vertical lines for inspection outcomes
    cls_styles = {
        "Official Action Indicated (OAI)": ("#C0392B", "--", "OAI"),
        "Voluntary Action Indicated (VAI)": ("#E67E22", ":",  "VAI"),
        "No Action Indicated (NAI)":        ("#27AE60", "-.", "NAI"),
    }
    added_labels: set = set()
    for _, row in insp_fei.iterrows():
        cls = row["Classification"]
        if cls not in cls_styles:
            continue
        color, ls, short = cls_styles[cls]
        lbl = short if short not in added_labels else "_"
        ax.axvline(row["Inspection End Date"], color=color, linestyle=ls,
                   linewidth=1.5, alpha=0.8, label=lbl, zorder=2)
        added_labels.add(short)

    # Shortage bands (from monthly panel via drug names)
    if monthly is not None and drugs:
        drug_list = [d.strip() for d in drugs.split(",")]
        mp = monthly[monthly["drug_norm"].isin(drug_list)].copy()
        if "shortage_start" in mp.columns:
            onsets = mp[mp["shortage_start"] == 1].copy()
            onsets["date"] = pd.to_datetime(
                onsets["year"].astype(str) + "-" + onsets["month"].astype(str).str.zfill(2) + "-01")
            for _, sr in onsets.iterrows():
                ax.axvspan(sr["date"], sr["date"] + pd.DateOffset(months=1),
                           alpha=0.12, color="gray", zorder=1)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of observations (0–1)")
    ax.set_title(f"FEI {fei} — {drugs}\nText signals over time vs inspection outcomes",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"m13_trajectory_{fei}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Trajectory saved: %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Before/after OAI comparison
# ─────────────────────────────────────────────────────────────────────────────

def before_after_oai(ts: pd.DataFrame, insp: pd.DataFrame) -> pd.DataFrame:
    oai_insp = insp[insp["Classification"] == "Official Action Indicated (OAI)"].copy()
    rows = []
    for _, oai_row in oai_insp.iterrows():
        fei = oai_row["FEI Number"]
        oai_date = oai_row["Inspection End Date"]
        snaps_fei = ts[ts["fei"] == fei].sort_values("snapshot_date")
        if snaps_fei.empty:
            continue
        before = snaps_fei[snaps_fei["snapshot_date"] < oai_date]
        after  = snaps_fei[snaps_fei["snapshot_date"] >= oai_date]
        if before.empty:
            continue
        snap_before = before.iloc[-1]  # most recent snapshot before OAI
        snap_after  = after.iloc[0] if not after.empty else None
        rec = {
            "fei": fei,
            "oai_date": oai_date.date(),
            "snapshot_before": snap_before["snapshot_date"].date(),
            "gap_months_before": round(
                (oai_date - snap_before["snapshot_date"]).days / 30.44, 1),
            "snapshot_after": snap_after["snapshot_date"].date() if snap_after is not None else None,
        }
        for feat in LEAD_FEATS:
            rec[f"before_{feat}"] = round(float(snap_before.get(feat, np.nan)), 3) \
                if pd.notna(snap_before.get(feat)) else None
            rec[f"after_{feat}"] = round(float(snap_after.get(feat, np.nan)), 3) \
                if snap_after is not None and pd.notna(snap_after.get(feat)) else None
        rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Lead table: all FEIs with snapshot before OAI
# ─────────────────────────────────────────────────────────────────────────────

def build_lead_table(ts: pd.DataFrame, insp: pd.DataFrame, fei_drugs: dict) -> pd.DataFrame:
    oai_insp = insp[insp["Classification"] == "Official Action Indicated (OAI)"].copy()
    rows = []
    for _, oai_row in oai_insp.iterrows():
        fei = oai_row["FEI Number"]
        oai_date = oai_row["Inspection End Date"]
        snaps_fei = ts[ts["fei"] == fei].sort_values("snapshot_date")
        before = snaps_fei[snaps_fei["snapshot_date"] < oai_date]
        if before.empty:
            continue
        snap = before.iloc[-1]
        rec = {
            "fei": fei,
            "drugs": fei_drugs.get(fei, ""),
            "snapshot_date": snap["snapshot_date"].date(),
            "oai_date": oai_date.date(),
            "gap_months": round((oai_date - snap["snapshot_date"]).days / 30.44, 1),
        }
        for feat in LEAD_FEATS:
            rec[feat] = round(float(snap.get(feat, np.nan)), 3) \
                if pd.notna(snap.get(feat)) else None
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("gap_months")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data…")
    ts, insp, redica, monthly, fei_drugs, our_feis = load_data()

    ranked = rank_feis(ts, insp)
    top2 = ranked["fei"].head(2).tolist()
    log.info("Top case study FEIs: %s", top2)
    for fei in top2:
        r = ranked[ranked["fei"] == fei].iloc[0]
        log.info("  FEI %d: %d snaps, %.0f%% coverage, %d OAIs, drugs: %s",
                 fei, r["n_snaps"], r["coverage"] * 100, r["n_oai"],
                 fei_drugs.get(fei, ""))

    # Trajectory figures
    for fei in top2:
        plot_trajectory(fei, ts, insp, monthly, fei_drugs, OUT_FIGS)

    # Before/after OAI
    ba = before_after_oai(ts, insp)
    ba_path = OUT_TABS / "m13_before_after_oai.csv"
    ba.to_csv(ba_path, index=False)
    log.info("Before/after OAI table: %d rows → %s", len(ba), ba_path)

    # Lead table
    lt = build_lead_table(ts, insp, fei_drugs)
    lt_path = OUT_TABS / "m13_lead_table.csv"
    lt.to_csv(lt_path, index=False)
    log.info("Lead table: %d FEI-OAI pairs → %s", len(lt), lt_path)

    print(f"\nCase studies: {[int(f) for f in top2]}")
    print(f"Before/after OAI: {len(ba)} events  →  {ba_path}")
    print(f"Lead table:       {len(lt)} rows     →  {lt_path}")
    if not lt.empty:
        print("\n=== Lead table (gap months = snapshot → OAI) ===")
        print(lt[["fei", "drugs", "snapshot_date", "oai_date", "gap_months",
                   "severity_critmajor_share", "contamination_llm_share",
                   "repeat_llm_only_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
