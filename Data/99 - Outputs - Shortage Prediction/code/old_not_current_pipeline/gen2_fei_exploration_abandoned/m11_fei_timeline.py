"""
FEI event timeline: link 483 text-risk snapshots to inspection outcomes,
recalls, and drug shortages for the Valisure 14-drug FEI universe.

Exploratory temporal co-occurrence only — a shortage may have multiple causes
and a facility may manufacture many drugs; no causal attribution is implied.

Usage:
    python m11_fei_timeline.py --fei 1021343   # single-FEI test mode
    python m11_fei_timeline.py                 # full run, all FEIs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (  # noqa: E402
    DATA,
    OUT_DATA,
    OUT_FIGS,
    PANEL_END_YEAR,
    PANEL_START_YEAR,
    REDICA_AGG,
    REDICA_CSV,
    VALISURE_FEI,
)

TIMESERIES_CSV = DATA / "99 - Outputs - Text Analysis" / "483_fei_text_features_timeseries.csv"
CITATIONS_XLSX = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Citations Details.xlsx"
RECALL_XLSX = DATA / "22 - FDA - Recall" / "raw" / "Recall Data.xlsx"
SHORTAGE_XLSX = DATA / "24 - UUtah - Drug Shortage" / "raw" / "efox shortages small file through 2025 final.xlsx"
OUT_DIR = OUT_FIGS / "timelines"
OUT_DIR.mkdir(exist_ok=True)

RISK_COLS = [
    "severity_critmajor_share", "scope_facilitywide_share",
    "contamination_llm_only_share", "repeat_llm_share", "repeat_cross_insp_share",
    "cultural_root_cause_share", "vc_buildingsequipment_share",
    "remediation_none_share", "wl_ref_regex_share", "oos_oot_regex_share",
]

HIGH_RISK_CFRS = {
    "sterility": "211.113",
    "investigation": "211.192",
    "data_integrity": "211.68",
}


# ---------------------------------------------------------------- Step 0
def load_valisure_universe() -> dict[int, list[str]]:
    b = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    b = b[["API", "FEI_NUMBER"]].dropna(subset=["FEI_NUMBER"])
    b["FEI_NUMBER"] = b["FEI_NUMBER"].astype("int64")
    valisure_feis: dict[int, list[str]] = {}
    for fei, grp in b.groupby("FEI_NUMBER"):
        valisure_feis[int(fei)] = sorted(grp["API"].str.strip().unique())
    api_counts = b.groupby("API")["FEI_NUMBER"].nunique().sort_values(ascending=False)
    print(f"[step0] {len(valisure_feis)} unique FEIs across {b['API'].nunique()} APIs")
    print("[step0] APIs with most FEIs:")
    print(api_counts.head(8).to_string())
    return valisure_feis


# ---------------------------------------------------------------- Step 1a
def load_483_timeseries(feis: set[int]) -> pd.DataFrame:
    ts = pd.read_csv(TIMESERIES_CSV)
    ts = ts[ts["fei"].isin(feis)].copy()
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"])
    ts["outside_panel"] = ~ts["snapshot_date"].dt.year.between(PANEL_START_YEAR, PANEL_END_YEAR)
    # Red flag: >=2 of the 4 markers the m12 validation grid showed to be
    # forward-predictive of OAI/Warning-Letter escalation and recalls
    # (repeat, contamination, OOS/OOT, severity-high). Median split mirrors
    # the m12 test design; repeat uses >0 because its median is 0.
    med_contam = ts["contamination_llm_only_share"].median()
    med_oos = ts["oos_oot_regex_share"].median()
    med_sev = ts["severity_critmajor_share"].median()
    n_criteria = (
        (ts["repeat_llm_only_share"] > 0).astype(int)
        + (ts["contamination_llm_only_share"] > med_contam).astype(int)
        + (ts["oos_oot_regex_share"] > med_oos).astype(int)
        + (ts["severity_critmajor_share"] > med_sev).astype(int)
    )
    ts["n_risk_criteria"] = n_criteria
    ts["high_risk_483"] = n_criteria >= 2
    print(f"[step1a] flag thresholds: contamination>{med_contam:.2f}, "
          f"oos_oot>{med_oos:.2f}, sev_critmajor>{med_sev:.2f}, repeat>0")
    print(f"[step1a] 483 snapshots: {len(ts)} rows, {ts['fei'].nunique()} FEIs, "
          f"{int(ts['high_risk_483'].sum())} high-risk, "
          f"{int(ts['outside_panel'].sum())} outside panel window")
    return ts.sort_values(["fei", "snapshot_date"]).reset_index(drop=True)


# ---------------------------------------------------------------- Step 1b
def load_redica_events(feis: set[int]) -> pd.DataFrame:
    """One row per inspection event from redica_all_drugs_combined.csv
    (127/129 Valisure FEIs — far broader than SITE_RED_FLAG_EVENTS, which
    only covers the 16 deep-dive site exports)."""
    ev = pd.read_csv(REDICA_CSV)
    ev = ev[ev["FEI"].isin(feis)].copy()
    ev["Event Start Date"] = pd.to_datetime(ev["Event Date"], errors="coerce")
    ev["Event End Date"] = ev["Event Start Date"]

    def classify(r) -> str:
        c = r["Classification"]
        if pd.notna(c) and c in ("OAI", "VAI", "NAI"):
            return c
        if r.get("483") == 1:
            return "483_issued"
        return "other"

    ev["classification"] = ev.apply(classify, axis=1)
    ev = ev[ev["classification"] != "other"]
    ev["n_483_critical"] = ev.get("483 critical")
    counts = ev.groupby("classification").size()
    print(f"[step1b] Redica events: {len(ev)} rows, {ev['FEI'].nunique()} FEIs")
    print(counts.to_string())
    return ev.rename(columns={"FEI": "fei"}).reset_index(drop=True)


# ---------------------------------------------------------------- Step 1c
def load_redica_agg(feis: set[int]) -> pd.DataFrame:
    ag = pd.read_excel(REDICA_AGG)
    ag = ag[ag["Fei"].isin(feis)].copy()
    ag = ag.rename(columns={"Fei": "fei", "Total Score": "redica_total_score",
                            "L 5 Y Score": "redica_l5y_score"})
    ag = ag.groupby("fei").agg(
        firm_name=("Site Display Name", "first"),
        redica_total_score=("redica_total_score", "max"),
        redica_l5y_score=("redica_l5y_score", "max"),
    ).reset_index()
    print(f"[step1c] Redica agg scores: {len(ag)} FEIs")
    return ag


# ---------------------------------------------------------------- Step 1d
def load_citations(feis: set[int]) -> pd.DataFrame:
    cit = pd.read_excel(CITATIONS_XLSX)
    cit = cit[cit["FEI Number"].isin(feis)].copy()
    pa = cit["Program Area"].astype(str)
    cit = cit[pa.str.contains("Drug", case=False) | pa.str.contains("Pharmaceutical", case=False)]
    cit["Inspection End Date"] = pd.to_datetime(cit["Inspection End Date"], errors="coerce")
    cfr = cit["Act/CFR Number"].astype(str)

    grouped = cit.groupby(["FEI Number", "Inspection End Date"]).agg(
        n_cfr_citations=("Act/CFR Number", "nunique"),
        short_descriptions=("Short Description", lambda s: "; ".join(s.dropna().astype(str).unique()[:5])),
    ).reset_index()
    for flag, code in HIGH_RISK_CFRS.items():
        has = (cit.assign(_h=cfr.str.contains(code, regex=False))
               .groupby(["FEI Number", "Inspection End Date"])["_h"].any().reset_index(name=f"has_{flag}_cfr"))
        grouped = grouped.merge(has, on=["FEI Number", "Inspection End Date"], how="left")
    grouped = grouped.rename(columns={"FEI Number": "fei", "Inspection End Date": "insp_end_date"})
    names = (cit.groupby("FEI Number")["Legal Name"].first()
             .reset_index().rename(columns={"FEI Number": "fei", "Legal Name": "legal_name"}))
    print(f"[step1d] Citation inspection-groups: {len(grouped)} rows, {grouped['fei'].nunique()} FEIs")
    return grouped, names


def join_citations_to_events(ev: pd.DataFrame, cit_grp: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["match_date"] = ev["Event End Date"].fillna(ev["Event Start Date"])
    ev = ev.dropna(subset=["match_date"]).sort_values("match_date").reset_index(drop=True)
    cit_grp = cit_grp.dropna(subset=["insp_end_date"]).sort_values("insp_end_date").reset_index(drop=True)
    merged = pd.merge_asof(
        ev, cit_grp,
        left_on="match_date", right_on="insp_end_date",
        by="fei", direction="nearest", tolerance=pd.Timedelta(days=7),
    )
    n_enriched = merged["n_cfr_citations"].notna().sum()
    print(f"[step1d] Events enriched with citations (±7 days): {n_enriched}/{len(merged)}")
    return merged


# ---------------------------------------------------------------- Step 1e
def load_recalls(feis: set[int]) -> pd.DataFrame:
    rec = pd.read_excel(RECALL_XLSX)
    rec["FEI Number"] = pd.to_numeric(rec["FEI Number"], errors="coerce").astype("Int64")
    rec = rec[(rec["FEI Number"].isin(feis)) & (rec["Product Type"] == "Drugs")].copy()
    rec["Center Classification Date"] = pd.to_datetime(rec["Center Classification Date"], errors="coerce")
    rec["recall_class"] = rec["Event Classification"].astype(str).str.extract(r"(Class I{1,3})", expand=False)
    rec["reason_excerpt"] = rec["Reason for Recall"].astype(str).str.slice(0, 40)
    rec["product_excerpt"] = rec["Product Description"].astype(str).str.slice(0, 40)
    rec = rec.rename(columns={"FEI Number": "fei"})
    print(f"[step1e] Drug recalls: {len(rec)} rows, {rec['fei'].nunique()} FEIs; "
          f"class counts:\n{rec['recall_class'].value_counts().to_string()}")
    return rec.reset_index(drop=True)


# ---------------------------------------------------------------- Step 1f
def load_shortages(valisure_feis: dict[int, list[str]]) -> pd.DataFrame:
    sh = pd.read_excel(SHORTAGE_XLSX, header=None, skiprows=2,
                       names=["drug_name", "status", "ahfs_code", "reason", "year",
                              "date_notified", "date_resolved", "sole_source",
                              "parenteral", "controlled_substance"])
    sh = sh.dropna(subset=["drug_name"]).copy()
    sh["date_notified"] = pd.to_datetime(sh["date_notified"], errors="coerce")
    sh["date_resolved"] = pd.to_datetime(sh["date_resolved"], errors="coerce")
    sh["year"] = pd.to_numeric(sh["year"], errors="coerce")

    all_apis = sorted({a for apis in valisure_feis.values() for a in apis})
    api_lower = {a.lower(): a for a in all_apis}

    def match_api(drug: str) -> tuple[str | None, str]:
        d = str(drug).lower().strip()
        if d in api_lower:
            return api_lower[d], "exact"
        for al, a in api_lower.items():
            if al in d:
                return a, "partial"
        return None, "none"

    matched = sh["drug_name"].apply(match_api)
    sh["matched_api"] = matched.apply(lambda t: t[0])
    sh["match_confidence"] = matched.apply(lambda t: t[1])

    yr = sh["date_notified"].dt.year.fillna(sh["year"])
    sh = sh[(yr >= PANEL_START_YEAR - 2) & (yr <= PANEL_END_YEAR + 1)]

    print(f"[step1f] Shortages in window {PANEL_START_YEAR-2}-{PANEL_END_YEAR+1}: {len(sh)} rows")
    print("[step1f] Match confidence counts:")
    print(sh["match_confidence"].value_counts().to_string())
    m = sh[sh["matched_api"].notna()]
    print(f"[step1f] Matched shortage drugs ({m['matched_api'].nunique()} APIs):")
    for api, grp in m.groupby("matched_api"):
        names = grp["drug_name"].unique()[:3]
        print(f"  {api}: {len(grp)} shortages, e.g. {list(names)}")
    return sh[sh["matched_api"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------- Step 2
def build_fei_timeline(fei: int, apis: list[str], ts: pd.DataFrame, ev: pd.DataFrame,
                       ag: pd.DataFrame, rec: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    f_ts = ts[ts["fei"] == fei]
    for _, r in f_ts.iterrows():
        rows.append({
            "event_date": r["snapshot_date"], "event_type": "483_snapshot",
            "event_label": f"483 snapshot: {r['n_obs_total']} obs, sev_critmajor={r['severity_critmajor_share']:.2f}",
            "high_risk_483": bool(r["high_risk_483"]), "n_obs_total": r["n_obs_total"],
            **{c: r[c] for c in RISK_COLS},
        })
    f_ev = ev[ev["fei"] == fei]
    for _, r in f_ev.iterrows():
        if pd.isna(r["Event Start Date"]):
            continue
        et = "483_issued" if r["classification"] == "483_issued" else "inspection_outcome"
        lab = r["classification"]
        if pd.notna(r.get("n_cfr_citations")):
            lab += f" ({int(r['n_cfr_citations'])} CFR citations)"
        rows.append({
            "event_date": r["Event Start Date"], "event_type": et,
            "event_label": lab[:80], "classification": r["classification"],
            "n_cfr_citations": r.get("n_cfr_citations"),
            "has_sterility_cfr": r.get("has_sterility_cfr"),
            "has_investigation_cfr": r.get("has_investigation_cfr"),
            "has_data_integrity_cfr": r.get("has_data_integrity_cfr"),
            "short_descriptions": r.get("short_descriptions"),
        })
    f_rec = rec[rec["fei"] == fei]
    for _, r in f_rec.iterrows():
        if pd.isna(r["Center Classification Date"]):
            continue
        rows.append({
            "event_date": r["Center Classification Date"], "event_type": "recall",
            "event_label": f"Recall {r['recall_class']}: {r['product_excerpt']}"[:80],
            "recall_class": r["recall_class"], "recall_reason": r["reason_excerpt"],
        })
    f_sh = sh[sh["matched_api"].isin(apis)]
    for _, r in f_sh.iterrows():
        if pd.notna(r["date_notified"]):
            rows.append({
                "event_date": r["date_notified"], "event_type": "shortage_start",
                "event_label": f"Shortage start: {str(r['drug_name'])[:55]}"[:80],
                "shortage_drug": r["drug_name"], "shortage_reason": r["reason"],
                "shortage_resolved_date": r["date_resolved"],
                "match_confidence": r["match_confidence"],
            })
        if pd.notna(r["date_resolved"]):
            rows.append({
                "event_date": r["date_resolved"], "event_type": "shortage_resolved",
                "event_label": f"Shortage resolved: {str(r['drug_name'])[:50]}"[:80],
                "shortage_drug": r["drug_name"],
            })
    if not rows:
        return pd.DataFrame()
    tl = pd.DataFrame(rows)
    for c in ["high_risk_483", "months_since_nearest_high_risk_483", "classification"] + RISK_COLS:
        if c not in tl.columns:
            tl[c] = np.nan
    tl["fei"] = fei
    a = ag[ag["fei"] == fei]
    tl["firm_name"] = a["firm_name"].iloc[0] if len(a) else ""
    tl["redica_total_score"] = a["redica_total_score"].iloc[0] if len(a) else np.nan
    tl["redica_l5y_score"] = a["redica_l5y_score"].iloc[0] if len(a) else np.nan
    tl["year_month"] = tl["event_date"].dt.strftime("%Y-%m")
    tl["outside_panel"] = ~tl["event_date"].dt.year.between(PANEL_START_YEAR, PANEL_END_YEAR)
    tl = tl.sort_values("event_date").reset_index(drop=True)

    # carry-forward risk signals from most recent snapshot to each event
    snaps = tl[tl["event_type"] == "483_snapshot"][["event_date"] + RISK_COLS].copy()
    if len(snaps):
        snaps = snaps.sort_values("event_date")
        non_snap = tl["event_type"] != "483_snapshot"
        ff = pd.merge_asof(
            tl.loc[non_snap, ["event_date"]].sort_values("event_date").reset_index(),
            snaps, on="event_date", direction="backward",
        ).set_index("index")
        for c in RISK_COLS:
            tl.loc[ff.index, c] = ff[c]

    # months since nearest prior high-risk 483 for recall/shortage events
    hr_dates = tl.loc[(tl["event_type"] == "483_snapshot") & (tl["high_risk_483"] == True),  # noqa: E712
                      "event_date"].sort_values().to_list()
    def months_since(d):
        prior = [h for h in hr_dates if h <= d]
        if not prior:
            return np.nan
        return round((d - prior[-1]).days / 30.44, 1)
    target = tl["event_type"].isin(["recall", "shortage_start"])
    if target.any():
        tl.loc[target, "months_since_nearest_high_risk_483"] = (
            tl.loc[target, "event_date"].apply(months_since).astype(float))
    tl["within_24mo_of_high_risk"] = tl["months_since_nearest_high_risk_483"].between(0, 24)
    return tl


# ---------------------------------------------------------------- Step 3
MARKER_STYLE = {
    "OAI": dict(marker="^", color="red"), "VAI": dict(marker=">", color="orange"),
    "NAI": dict(marker="v", color="green"), "483_issued": dict(marker="x", color="gray"),
}
RECALL_COLOR = {"Class I": "darkred", "Class II": "orange", "Class III": "gold"}


def plot_fei_timeline(fei: int, apis: list[str], tl: pd.DataFrame) -> Path | None:
    if tl.empty:
        return None
    firm = tl["firm_name"].iloc[0] or str(fei)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    LANES = {"483_snapshot": 4, "inspection": 3, "recall": 2, "shortage": 1}

    dmin, dmax = tl["event_date"].min(), tl["event_date"].max()
    dmin = min(dmin, pd.Timestamp(f"{PANEL_START_YEAR}-01-01"))
    dmax = max(dmax, pd.Timestamp(f"{PANEL_END_YEAR}-12-31"))
    pad = pd.Timedelta(days=180)
    ax.axvspan(pd.Timestamp(f"{PANEL_START_YEAR}-01-01"), pd.Timestamp(f"{PANEL_END_YEAR}-12-31"),
               color="lightblue", alpha=0.25, zorder=0)
    ax.axvspan(dmin - pad, pd.Timestamp(f"{PANEL_START_YEAR}-01-01"),
               color="lightgray", alpha=0.35, zorder=0)
    ax.text(pd.Timestamp(f"{PANEL_START_YEAR}-06-01"), 4.65, f"panel window {PANEL_START_YEAR}-{PANEL_END_YEAR}",
            fontsize=8, color="steelblue")
    if (tl["event_date"] < pd.Timestamp(f"{PANEL_START_YEAR}-01-01")).any():
        ax.text(dmin, 4.65, "pre-panel", fontsize=8, color="gray")

    # Lane 1: 483 snapshots
    snaps = tl[tl["event_type"] == "483_snapshot"]
    for _, r in snaps.iterrows():
        size = np.clip(40 + 160 * (r["n_obs_total"] / max(snaps["n_obs_total"].max(), 1)), 40, 200)
        col = "red" if r["high_risk_483"] else "steelblue"
        ax.scatter(r["event_date"], LANES["483_snapshot"], s=size, color=col, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(f"{r['severity_critmajor_share']:.2f}", (r["event_date"], LANES["483_snapshot"]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7)
        if r["high_risk_483"]:
            ax.axvline(r["event_date"], color="red", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            ax.axvspan(r["event_date"], r["event_date"] + pd.Timedelta(days=730),
                       color="red", alpha=0.06, zorder=0)

    # Lane 2: inspection outcomes
    insp = tl[tl["event_type"].isin(["inspection_outcome", "483_issued"])]
    for _, r in insp.iterrows():
        st = MARKER_STYLE.get(r.get("classification"), dict(marker="o", color="black"))
        ax.scatter(r["event_date"], LANES["inspection"], s=70, zorder=3, **st)
        lbl = str(r.get("classification", ""))
        if pd.notna(r.get("n_cfr_citations")):
            flags = "".join(k[4].upper() for k in ["has_sterility_cfr", "has_investigation_cfr",
                                                    "has_data_integrity_cfr"] if r.get(k) is True)
            lbl += f"\n{int(r['n_cfr_citations'])} cit" + (f" [{flags}]" if flags else "")
        ax.annotate(lbl, (r["event_date"], LANES["inspection"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=6.5)

    # Lane 3: recalls
    recs = tl[tl["event_type"] == "recall"]
    for _, r in recs.iterrows():
        col = RECALL_COLOR.get(r.get("recall_class"), "black")
        ax.scatter(r["event_date"], LANES["recall"], s=80, marker="D", color=col, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(f"{r.get('recall_class','')}\n{str(r.get('recall_reason',''))[:22]}",
                    (r["event_date"], LANES["recall"]),
                    textcoords="offset points", xytext=(0, -22), ha="center", fontsize=6)

    # Lane 4: shortages
    sh_rows = tl[tl["event_type"] == "shortage_start"]
    for i, (_, r) in enumerate(sh_rows.iterrows()):
        end = r.get("shortage_resolved_date")
        end = end if pd.notna(end) else dmax
        y = LANES["shortage"] - 0.12 + 0.08 * (i % 4)
        ax.plot([r["event_date"], end], [y, y], color="dimgray", linewidth=4, alpha=0.7,
                solid_capstyle="butt", zorder=2)
        ax.annotate(str(r.get("shortage_drug", ""))[:28], (r["event_date"], y),
                    textcoords="offset points", xytext=(2, 4), fontsize=6, color="dimgray")

    ax.set_yticks(list(LANES.values()))
    ax.set_yticklabels(["483 snapshots", "Inspections", "Recalls", "Shortages (API-linked)"])
    ax.set_ylim(0.4, 4.9)
    ax.set_xlim(dmin - pad, dmax + pad)
    apis_s = ", ".join(apis)
    ax.set_title(f"FEI {fei} — {firm} ({apis_s})", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"fei_{fei}_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overview(timelines: dict[int, pd.DataFrame], summary: pd.DataFrame) -> Path:
    order = summary.sort_values(["n_class_I_recalls", "n_high_risk_483_snapshots"],
                                ascending=False)["fei"].tolist()
    order = [f for f in order if f in timelines and not timelines[f].empty]
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(13, max(0.42 * n, 3)), sharex=True)
    if n == 1:
        axes = [axes]
    dmin = pd.Timestamp("2009-01-01")
    dmax = pd.Timestamp(f"{PANEL_END_YEAR + 1}-12-31")
    type_style = {
        "483_snapshot": dict(marker="o", color="steelblue"),
        "inspection_outcome": dict(marker="^", color="orange"),
        "483_issued": dict(marker="x", color="gray"),
        "recall": dict(marker="D", color="darkred"),
        "shortage_start": dict(marker="s", color="dimgray"),
    }
    for ax, fei in zip(axes, order):
        tl = timelines[fei]
        for et, st in type_style.items():
            sub = tl[tl["event_type"] == et]
            if et == "483_snapshot":
                hr = sub[sub["high_risk_483"] == True]  # noqa: E712
                nm = sub[sub["high_risk_483"] != True]  # noqa: E712
                ax.scatter(nm["event_date"], [0.5] * len(nm), s=12, **st)
                ax.scatter(hr["event_date"], [0.5] * len(hr), s=14, marker="o", color="red")
            else:
                ax.scatter(sub["event_date"], [0.5] * len(sub), s=12, **st)
        ax.set_yticks([])
        ax.set_ylabel(str(fei), rotation=0, ha="right", va="center", fontsize=7)
        ax.axvspan(pd.Timestamp(f"{PANEL_START_YEAR}-01-01"),
                   pd.Timestamp(f"{PANEL_END_YEAR}-12-31"), color="lightblue", alpha=0.2)
        ax.set_xlim(dmin, dmax)
    axes[0].set_title("All FEIs — event overview (sorted by Class I recalls)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "all_feis_timeline_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- Step 4
def narrative(fei: int, apis: list[str], tl: pd.DataFrame) -> str:
    L = []
    firm = tl["firm_name"].iloc[0] if len(tl) else ""
    score = tl["redica_total_score"].iloc[0] if len(tl) else np.nan
    snaps = tl[tl["event_type"] == "483_snapshot"]
    L.append(f"FEI {fei}: {firm}")
    L.append(f"APIs manufactured: {', '.join(apis)}")
    if len(snaps):
        L.append(f"Panel coverage: {snaps['event_date'].min().date()} to "
                 f"{snaps['event_date'].max().date()} | Redica total score: {score}")
    else:
        L.append(f"Panel coverage: no 483 snapshots | Redica total score: {score}")

    L.append(f"\n483 history ({len(snaps)} snapshots):")
    for _, r in snaps.iterrows():
        tag = "[HIGH RISK]" if r["high_risk_483"] else "[normal]"
        L.append(f"  - {r['event_date'].date()}: {int(r['n_obs_total'])} observations, "
                 f"sev_critmajor={r['severity_critmajor_share']:.2f}, "
                 f"scope_fw={r.get('scope_facilitywide_share', float('nan')):.2f}, "
                 f"cross_repeat={r.get('repeat_cross_insp_share', float('nan')):.2f} {tag}")

    insp = tl[tl["event_type"].isin(["inspection_outcome", "483_issued"])]
    L.append(f"\nInspection outcomes (Redica): {len(insp)}")
    for _, r in insp.iterrows():
        cit = (f", {int(r['n_cfr_citations'])} citations, key CFRs: {str(r.get('short_descriptions'))[:90]}"
               if pd.notna(r.get("n_cfr_citations")) else "")
        L.append(f"  - {r['event_date'].date()}: {r.get('classification')}{cit}")

    recs = tl[tl["event_type"] == "recall"]
    L.append(f"\nRecalls: {len(recs)}")
    for _, r in recs.iterrows():
        L.append(f"  - {r['event_date'].date()}: {r.get('recall_class')}, {r.get('recall_reason')}")

    shs = tl[tl["event_type"] == "shortage_start"]
    L.append(f"\nShortages (matched via API name): {len(shs)}")
    for _, r in shs.iterrows():
        res = r.get("shortage_resolved_date")
        dur = f"{(res - r['event_date']).days} days" if pd.notna(res) else "unresolved"
        res_s = res.date() if pd.notna(res) else "—"
        L.append(f"  - {str(r.get('shortage_drug'))[:50]}: notified {r['event_date'].date()}, "
                 f"resolved {res_s} ({dur}), reason: {str(r.get('shortage_reason'))[:50]}")

    cooc = tl[tl["event_type"].isin(["recall", "shortage_start"])
              & tl["months_since_nearest_high_risk_483"].notna()]
    L.append("\nTemporal co-occurrence:")
    if len(cooc) == 0:
        L.append("  - none (no recall/shortage events follow a high-risk 483 snapshot)")
    for _, r in cooc.iterrows():
        L.append(f"  - {r['event_type']} {r['event_date'].date()}: "
                 f"{r['months_since_nearest_high_risk_483']} months after nearest high-risk 483. "
                 f"within_24mo={bool(r['within_24mo_of_high_risk'])}")
    L.append("  - Caution: temporal co-occurrence only. FEI may produce multiple drugs; "
             "shortage/recall may have causes unrelated to this facility's quality history.")
    return "\n".join(L)


# ---------------------------------------------------------------- Step 5
def summarize_fei(fei: int, apis: list[str], tl: pd.DataFrame) -> dict:
    snaps = tl[tl["event_type"] == "483_snapshot"] if len(tl) else pd.DataFrame()
    insp = tl[tl["event_type"] == "inspection_outcome"] if len(tl) else pd.DataFrame()
    recs = tl[tl["event_type"] == "recall"] if len(tl) else pd.DataFrame()
    shs = tl[tl["event_type"] == "shortage_start"] if len(tl) else pd.DataFrame()
    rec_after = recs[recs["months_since_nearest_high_risk_483"].notna()] if len(recs) else pd.DataFrame()
    return {
        "fei": fei,
        "firm_name": tl["firm_name"].iloc[0] if len(tl) else "",
        "apis_made": ", ".join(apis),
        "n_483_snapshots": len(snaps),
        "n_high_risk_483_snapshots": int(snaps["high_risk_483"].sum()) if len(snaps) else 0,
        "has_pre_panel_483": bool((snaps["event_date"].dt.year < PANEL_START_YEAR).any()) if len(snaps) else False,
        "n_483_issued_redica": int((tl["event_type"] == "483_issued").sum()) if len(tl) else 0,
        "n_oai_inspections": int((insp["classification"] == "OAI").sum()) if len(insp) else 0,
        "n_vai_inspections": int((insp["classification"] == "VAI").sum()) if len(insp) else 0,
        "redica_total_score": tl["redica_total_score"].iloc[0] if len(tl) else np.nan,
        "redica_l5y_score": tl["redica_l5y_score"].iloc[0] if len(tl) else np.nan,
        "n_drug_recalls": len(recs),
        "n_class_I_recalls": int((recs["recall_class"] == "Class I").sum()) if len(recs) else 0,
        "n_shortage_drugs_linked": shs["shortage_drug"].nunique() if len(shs) else 0,
        "any_recall_within_24mo": bool(recs["within_24mo_of_high_risk"].any()) if len(recs) else False,
        "any_shortage_within_24mo": bool(shs["within_24mo_of_high_risk"].any()) if len(shs) else False,
        "months_to_nearest_recall_after_high_risk": (
            float(rec_after["months_since_nearest_high_risk_483"].min()) if len(rec_after) else np.nan),
    }


# ---------------------------------------------------------------- Step 6
def export_drug_month_feature(ts: pd.DataFrame, valisure_feis: dict[int, list[str]]) -> Path:
    """Drug-month trailing-24-month counts of (a) red-flagged 483s and
    (b) 483s containing repeat-violation findings — time-varying text features."""
    months = [(y, m) for y in range(PANEL_START_YEAR, PANEL_END_YEAR + 1) for m in range(1, 13)]

    def trailing_counts(snap: pd.DataFrame, colname: str) -> pd.DataFrame:
        snap = snap.copy()
        snap["smidx"] = snap["snapshot_date"].dt.year * 12 + snap["snapshot_date"].dt.month
        rows = []
        for fei, apis in valisure_feis.items():
            s = snap[snap["fei"] == fei]["smidx"].to_list()
            if not s:
                continue
            for y, m in months:
                midx = y * 12 + m
                n = sum(1 for sm in s if 0 <= midx - sm < 24)
                if n:
                    for api in apis:
                        rows.append({"drug_norm": api, "year": y, "month": m, "n": n})
        if not rows:
            return pd.DataFrame(columns=["drug_norm", "year", "month", colname])
        return (pd.DataFrame(rows).groupby(["drug_norm", "year", "month"])["n"].sum()
                .reset_index().rename(columns={"n": colname}))

    flagged = trailing_counts(ts[ts["high_risk_483"]], "n_flagged_483_last_24mo")
    repeat = trailing_counts(ts[ts["repeat_llm_only_share"] > 0], "n_repeat_483_last_24mo")
    out = flagged.merge(repeat, on=["drug_norm", "year", "month"], how="outer").fillna(0)
    out[["n_flagged_483_last_24mo", "n_repeat_483_last_24mo"]] = (
        out[["n_flagged_483_last_24mo", "n_repeat_483_last_24mo"]].astype(int))
    path = OUT_DATA / "text_highrisk_483_monthly.csv"
    out.to_csv(path, index=False)
    print(f"[step6] drug-month trailing text features saved: {path} ({len(out)} rows, "
          f"{out['drug_norm'].nunique()} drugs)")
    return path


def export_cooccurrence_stats(timelines: dict[int, pd.DataFrame], ts: pd.DataFrame) -> Path:
    """Within-24-months co-occurrence stats, restricted to text-covered FEIs
    (those with >=1 483 snapshot)."""
    covered = set(ts["fei"].unique())
    rows = []
    for etype in ["recall", "shortage_start"]:
        evs = []
        for fei, tl in timelines.items():
            if fei not in covered or tl.empty:
                continue
            evs.append(tl[tl["event_type"] == etype])
        if not evs:
            continue
        allev = pd.concat(evs)
        n_total = len(allev)
        n_within = int(allev["within_24mo_of_high_risk"].fillna(False).sum())
        rows.append({"event_type": etype, "n_events_text_covered_feis": n_total,
                     "n_within_24mo_of_high_risk_483": n_within,
                     "share": round(n_within / n_total, 3) if n_total else np.nan})
    stats = pd.DataFrame(rows)
    path = OUT_DATA / "text_cooccurrence_stats.csv"
    stats.to_csv(path, index=False)
    print(f"[step6] co-occurrence stats saved: {path}")
    print(stats.to_string(index=False))
    return path


# 483 text features tested for forward predictive power (median split).
# Outcome: severe regulatory action (OAI classification or Warning Letter)
# at the same FEI within 24 months of the snapshot.
FORWARD_FEATURES = {
    "contamination_llm_only_share":  "Contamination (LLM)",
    "repeat_llm_share":              "Repeat violations (LLM)",
    "repeat_cross_insp_share":       "Cross-inspection repeat (algo)",
    "oos_oot_regex_share":           "OOS/OOT references (regex)",
    "severity_critmajor_share":      "Critical+Major obs. (LLM)",
    "scope_facilitywide_share":      "Facility-wide scope (LLM)",
    "cultural_root_cause_share":     "Cultural root cause (LLM)",
    "vc_buildingsequipment_share":   "Buildings/equipment violations",
    "remediation_none_share":        "No remediation (LLM)",
    "n_obs_total":                   "Number of observations",
}


def export_snapshot_forward_stats(ts: pd.DataFrame) -> Path:
    """Forward validation: given a 483 was issued, does its TEXT content
    predict severe regulatory action (OAI or Warning Letter) at the same
    facility within 24 months? Median split per feature."""
    H = pd.Timedelta(days=730)
    ev = pd.read_csv(REDICA_CSV)
    ev["Event Date"] = pd.to_datetime(ev["Event Date"], errors="coerce")
    bad = ev[(ev["Classification"] == "OAI") | (ev["Warning Letter"] == 1)][["FEI", "Event Date"]]

    def fwd_bad(fei, d0) -> bool:
        f = bad[bad["FEI"] == fei]["Event Date"].dropna()
        return bool(((f > d0) & (f <= d0 + H)).any())

    df = ts.copy()
    df["bad_24mo"] = [fwd_bad(r.fei, r.snapshot_date) for r in df.itertuples()]

    rows = []
    for col, label in FORWARD_FEATURES.items():
        x = df[col].astype(float)
        med = x.median()
        hi, lo = df[x > med], df[x <= med]
        hi_rate = float(hi["bad_24mo"].mean()) if len(hi) else np.nan
        lo_rate = float(lo["bad_24mo"].mean()) if len(lo) else np.nan
        rows.append({
            "feature": col, "label": label, "median": round(float(med), 3),
            "n_hi": len(hi), "n_lo": len(lo),
            "hi_rate": round(hi_rate, 3), "lo_rate": round(lo_rate, 3),
            "lift": round(hi_rate / lo_rate, 2) if lo_rate else np.nan,
        })
    # composite high-risk flag for reference
    hr, nr = df[df["high_risk_483"]], df[~df["high_risk_483"]]
    rows.append({
        "feature": "high_risk_483", "label": "Composite high-risk flag (>=3 criteria)",
        "median": np.nan, "n_hi": len(hr), "n_lo": len(nr),
        "hi_rate": round(float(hr["bad_24mo"].mean()), 3),
        "lo_rate": round(float(nr["bad_24mo"].mean()), 3),
        "lift": round(float(hr["bad_24mo"].mean()) / max(float(nr["bad_24mo"].mean()), 1e-9), 2),
    })
    stats = pd.DataFrame(rows)
    stats["base_rate"] = round(float(df["bad_24mo"].mean()), 3)
    stats["n_snapshots"] = len(df)
    path = OUT_DATA / "text_snapshot_forward_stats.csv"
    stats.to_csv(path, index=False)
    print(f"[step6] snapshot forward-validation stats saved: {path}")
    print(stats[["label", "n_hi", "n_lo", "hi_rate", "lo_rate", "lift"]].to_string(index=False))
    return path


# ---------------------------------------------------------------- Step 7 (export)
def _export_fei_events(timelines: dict[int, pd.DataFrame],
                       valisure_feis: dict[int, list[str]]) -> None:
    """Export compact event-level data for the dashboard interactive timeline viewer.
    Only FEIs with ≥1 483 snapshot are exported (text-covered subset)."""
    ETYPES = {"483_snapshot", "inspection_outcome", "recall", "shortage_start"}
    KEEP = [
        "fei", "firm_name", "event_date", "event_type", "event_label",
        "classification", "high_risk_483", "n_obs_total",
        "severity_critmajor_share", "recall_class", "shortage_drug",
        "shortage_resolved_date",
    ]
    chunks = []
    for fei_id, tl in timelines.items():
        if tl.empty:
            continue
        if not (tl["event_type"] == "483_snapshot").any():
            continue
        sub = tl[tl["event_type"].isin(ETYPES)].copy()
        for c in KEEP:
            if c not in sub.columns:
                sub[c] = np.nan
        sub = sub[KEEP].copy()
        sub["apis"] = ", ".join(valisure_feis.get(fei_id, []))
        chunks.append(sub)
    if not chunks:
        print("[step7] No FEI events to export")
        return
    df = pd.concat(chunks, ignore_index=True)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["shortage_resolved_date"] = pd.to_datetime(
        df["shortage_resolved_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    path = OUT_DATA / "fei_events_all.csv"
    df.to_csv(path, index=False)
    print(f"[step7] FEI event timeline data: {path} ({len(df)} rows, {df['fei'].nunique()} FEIs)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fei", type=int, default=None, help="run for a single FEI (test mode)")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip per-FEI charts whose PNG already exists")
    args = ap.parse_args()

    valisure_feis = load_valisure_universe()
    feis = set(valisure_feis)

    ts = load_483_timeseries(feis)
    ev = load_redica_events(feis)
    ag = load_redica_agg(feis)
    cit_grp, legal_names = load_citations(feis)
    site_names = (ev.groupby("fei")["Site Display Name"].first().reset_index()
                  .rename(columns={"Site Display Name": "site_name"}))
    ev = join_citations_to_events(ev, cit_grp)
    ag = ag.merge(site_names, on="fei", how="outer").merge(legal_names, on="fei", how="outer")
    ag["firm_name"] = ag["firm_name"].fillna(ag["site_name"]).fillna(ag["legal_name"])
    rec = load_recalls(feis)
    sh = load_shortages(valisure_feis)

    run_feis = [args.fei] if args.fei else sorted(feis)
    timelines: dict[int, pd.DataFrame] = {}
    summaries, narratives = [], []

    for fei in run_feis:
        apis = valisure_feis.get(fei, [])
        tl = build_fei_timeline(fei, apis, ts, ev, ag, rec, sh)
        timelines[fei] = tl
        summaries.append(summarize_fei(fei, apis, tl))
        if len(tl):
            nar = narrative(fei, apis, tl)
            narratives.append(nar)
            if args.fei:
                print("\n" + "=" * 70)
                print(nar)
        has_outcome = len(tl) and tl["event_type"].isin(["recall", "shortage_start"]).any()
        if args.skip_existing and (OUT_DIR / f"fei_{fei}_timeline.png").exists():
            has_outcome = False
        if not args.no_charts and has_outcome:
            out = plot_fei_timeline(fei, apis, tl)
            if out and args.fei:
                print(f"\n[step3] chart saved: {out}")

    summary = pd.DataFrame(summaries).sort_values(
        ["n_class_I_recalls", "n_high_risk_483_snapshots"], ascending=False)
    summary.to_csv(OUT_DIR / "fei_timeline_summary.csv", index=False)
    (OUT_DIR / "fei_timeline_narratives.txt").write_text("\n\n" + ("\n\n" + "=" * 70 + "\n\n").join(narratives))
    print(f"\n[step5] summary saved: {OUT_DIR / 'fei_timeline_summary.csv'} ({len(summary)} FEIs)")
    print(summary.head(15).to_string(index=False))

    _export_fei_events(timelines, valisure_feis)

    if not args.fei:
        export_drug_month_feature(ts, valisure_feis)
        export_cooccurrence_stats(timelines, ts)
        export_snapshot_forward_stats(ts)
        if not args.no_charts:
            out = plot_overview(timelines, summary)
            print(f"[step3] overview chart saved: {out}")


if __name__ == "__main__":
    main()
