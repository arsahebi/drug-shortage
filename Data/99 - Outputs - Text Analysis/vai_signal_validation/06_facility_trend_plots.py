"""
06_facility_trend_plots.py
----------------------------------------------------------------------------
Per-facility diagnostic: full quarterly AE trend over time, with every FDA
inspection marked (colored by OAI/VAI/NAI) and the 483 severity signal
annotated at each inspection. Built at the user's request as a visual
robustness check after the AE-outcome confound fix (see 02_vai_signal_model.py) --
lets a human eyeball whether a facility's AE trend actually moves around its
inspections, rather than trusting a single summary statistic.

Unlike the -4..+4 quarter window used elsewhere in this folder (centered on
one inspection at a time), this plots each facility's ENTIRE observed
quarterly AE history in one line, with ALL of that facility's inspections
marked on it -- so multiple inspections and any drift between them are
visible together.

Usage
-----
  python 06_facility_trend_plots.py                 default: the 6 Hi-sig
                                                      VAI facilities flagged
                                                      in 05_silent_problem.py
  python 06_facility_trend_plots.py --fei 3009843207 3002949099
                                                      specific FEIs

Outputs
-------
  outputs/figures/trend_<fei>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DATA = ROOT / "Data"
OUT      = HERE / "outputs"
OUT_FIGS = OUT / "figures"
OUT_TABS = OUT / "tables"

ANDA_AE_QTR_CSV = DATA / "08 - Valisure" / "processed" / "valisure_anda_faers_ae_counts_quarterly.csv"
FDA_INSP_XLSX   = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
TEXT_TS_CSV     = DATA / "99 - Outputs - Text Analysis" / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
FLAGGED_CSV     = OUT_TABS / "silent_problem_flagged_facilities.csv"

_CLASS_COLOR = {"OAI": "#dc2626", "VAI": "#d97706", "NAI": "#059669"}


def _period_to_float(period: str) -> float:
    year, q = period.split("Q")
    return int(year) + (int(q) - 1) / 4.0


def _load_ae_quarterly(fei: int) -> pd.DataFrame:
    df = pd.read_csv(ANDA_AE_QTR_CSV, low_memory=False)
    df["fei"] = pd.to_numeric(df["fei"], errors="coerce")
    df = df[df["fei"] == fei].copy()
    df["t"] = df["period"].map(_period_to_float)
    return df.sort_values("t")


def _load_inspections(fei: int) -> pd.DataFrame:
    fda = pd.read_excel(FDA_INSP_XLSX,
                        usecols=["FEI Number", "Inspection End Date", "Classification", "Project Area"])
    fda = fda[fda["Project Area"] == "Drug Quality Assurance"].copy()
    fda["fei"] = pd.to_numeric(fda["FEI Number"], errors="coerce")
    fda = fda[fda["fei"] == fei].copy()
    fda["insp_date"] = pd.to_datetime(fda["Inspection End Date"], errors="coerce")
    fda = fda.dropna(subset=["insp_date"])
    cls = fda["Classification"].astype(str).str.upper()
    fda["cls"] = np.select(
        [cls.str.contains("OFFICIAL ACTION"), cls.str.contains("VOLUNTARY ACTION"), cls.str.contains("NO ACTION")],
        ["OAI", "VAI", "NAI"], default="?"
    )
    fda["t"] = fda["insp_date"].dt.year + (fda["insp_date"].dt.quarter - 1) / 4.0
    return fda[["insp_date", "cls", "t"]].sort_values("t")


def _load_severity_at_inspections(fei: int) -> pd.DataFrame:
    ts = pd.read_csv(TEXT_TS_CSV, low_memory=False)
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce")
    ts = ts[ts["fei"] == fei].copy()
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    ts["t"] = ts["snapshot_date"].dt.year + (ts["snapshot_date"].dt.quarter - 1) / 4.0
    return ts[["snapshot_date", "t", "severity_majmod_share", "n_obs_total"]].sort_values("t")


def plot_facility(fei: int, label: str = "") -> None:
    ae = _load_ae_quarterly(fei)
    insp = _load_inspections(fei)
    sev = _load_severity_at_inspections(fei)

    if ae.empty:
        print(f"  [SKIP] FEI {fei}: no ANDA-matched AE data")
        return

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(ae["t"], ae["n_ae_serious"], color="#2563eb", linewidth=1.6, marker="o", markersize=3)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Serious AE reports / quarter (ANDA-matched)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    for _, row in insp.iterrows():
        color = _CLASS_COLOR.get(row["cls"], "gray")
        ax1.axvline(row["t"], color=color, linestyle="--", linewidth=1.2, alpha=0.8)

    if not sev.empty:
        ax2 = ax1.twinx()
        ax2.scatter(sev["t"], sev["severity_majmod_share"], color="#7c3aed", marker="D", s=45, zorder=5,
                    label="severity_majmod_share at inspection")
        ax2.set_ylabel("severity_majmod_share (at inspection)", color="#7c3aed")
        ax2.tick_params(axis="y", labelcolor="#7c3aed")
        ax2.set_ylim(-0.05, 1.05)

    handles = [plt.Line2D([0], [0], color=c, linestyle="--", label=cls) for cls, c in _CLASS_COLOR.items()]
    handles.append(plt.Line2D([0], [0], color="#2563eb", marker="o", label="Serious AE / quarter"))
    if not sev.empty:
        handles.append(plt.Line2D([0], [0], color="#7c3aed", marker="D", linestyle="None",
                                   label="severity_majmod_share"))
    ax1.legend(handles=handles, fontsize=7, loc="upper left")

    title = f"FEI {fei}"
    if label:
        title += f" -- {label}"
    ax1.set_title(title, fontsize=10)
    fig.tight_layout()

    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    out_path = OUT_FIGS / f"trend_{fei}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fei", nargs="+", type=int, default=None,
                        help="Specific FEIs to plot. Default: the flagged Hi-sig VAI facilities.")
    args = parser.parse_args()

    if args.fei:
        for fei in args.fei:
            plot_facility(fei)
        return

    if not FLAGGED_CSV.exists():
        raise FileNotFoundError(f"{FLAGGED_CSV} not found -- run 05_silent_problem.py first, or pass --fei.")

    flagged = pd.read_csv(FLAGGED_CSV)
    print(f"Plotting {len(flagged)} flagged Hi-sig VAI facilities...")
    for _, row in flagged.iterrows():
        label = f"{row.get('labeler', '')}/{row.get('api', '')} (persist={row.get('persist_tp4_t0', 'NA')})"
        plot_facility(int(row["fei"]), label)


if __name__ == "__main__":
    main()
