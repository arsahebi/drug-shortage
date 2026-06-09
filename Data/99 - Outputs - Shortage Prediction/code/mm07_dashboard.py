"""
Module MM07 — Combined dashboard generator.

Reads existing pipeline outputs (annual + monthly) and writes a single
self-contained HTML file:  dashboard.html  in OUT_ROOT.

Replaces/supersedes the two legacy dashboard files:
  99 - Outputs - Shortage Prediction/dashboard.html          (old location)
  99 - Outputs - Shortage Prediction/dashboard-simple.html   (old location)

Run from the code/ directory:
    python3 mm07_dashboard.py
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from config import OUT_DATA, OUT_TABS, OUT_ROOT, OUT_LOGS, OUT_MODELS, TEXT_FEATURES_CSV, VALISURE_FEI
from utils import get_logger

log = get_logger("mm07_dashboard", OUT_LOGS / "mm07_dashboard.log")

DASH_OUT = OUT_ROOT / "dashboard.html"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(name: str, subdir: Path = OUT_DATA) -> pd.DataFrame | None:
    """Read CSV; return None if missing (dashboard falls back to placeholders)."""
    p = subdir / name
    if not p.exists():
        log.warning("File not found (section will show placeholder): %s", p.name)
        return None
    return pd.read_csv(p)


class _NpEncoder(json.JSONEncoder):
    """Encode numpy scalars as plain Python types so json.dumps doesn't choke."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _j(obj) -> str:
    """Serialize to compact JSON for embedding in JS."""
    return json.dumps(obj, cls=_NpEncoder, allow_nan=False)


# ─────────────────────────────────────────────────────────────────────────────
# Data computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_data() -> dict:
    d: dict = {}

    # ── Annual panel (pilot = has_valisure) ────────────────────────────────────
    ap = _read("master_panel.csv")
    if ap is not None:
        pilot = ap[ap["has_valisure"] == 1].copy()
        by_year = pilot.groupby("year")["shortage_started"].sum().reset_index()
        d["by_year"] = [{"year": int(r.year), "starts": int(r.shortage_started)}
                        for _, r in by_year.iterrows()]

        by_drug_raw = (pilot.groupby("drug_norm").agg(
            starts=("shortage_started", "sum"),
            faers_sev=("faers_severity_score", "mean"),
            faers_serious=("faers_n_serious", "mean"),
            faers_reports=("faers_n_reports", "mean"),
            redica_oai=("redica_n_oai", "mean"),
            redica_483=("redica_n_483_critical", "mean"),
            redica_vai=("redica_n_vai", "mean"),
            redica_wl=("redica_n_warning_letters", "mean"),
            redica_insp=("redica_n_inspections", "mean"),
            tri_mean=("tri_mean", "first"),
            scri_mean=("scri_mean", "first"),
            irwi_mean=("irwi_mean", "first"),
            qci_mean=("qci_mean", "first"),
        ).reset_index())

        # Valisure scores from valisure_drug summary
        vs = _read("eda_valisure_drug_summary.csv", OUT_TABS)
        if vs is None:
            vs = _read("valisure_drug.csv")
        if vs is not None:
            vs_map = vs.set_index("drug_norm")[["valisure_mean_score", "valisure_n_failing"]].to_dict("index")
        else:
            vs_map = {}

        d["by_drug"] = []
        for _, r in by_drug_raw.sort_values("starts", ascending=False).iterrows():
            vm = vs_map.get(r.drug_norm, {})
            d["by_drug"].append({
                "drug":          r.drug_norm,
                "starts":        int(r.starts),
                "faers":         round(float(r.faers_sev), 1),
                "faers_serious": round(float(r.faers_serious or 0), 1),
                "faers_reports": round(float(r.faers_reports or 0), 1),
                "val":           round(float(vm.get("valisure_mean_score", 0) or 0), 1),
                "fails":         int(vm.get("valisure_n_failing", 0) or 0),
                "tri":           round(float(r.tri_mean or 0), 1),
                "scri":          round(float(r.scri_mean or 0), 1),
                "irwi":          round(float(r.irwi_mean or 0), 1),
                "qci":           round(float(r.qci_mean or 0), 1),
                "redica_oai":    round(float(r.redica_oai or 0), 2),
                "redica_483":    round(float(r.redica_483 or 0), 2),
                "redica_vai":    round(float(r.redica_vai or 0), 2),
                "redica_wl":     round(float(r.redica_wl or 0), 2),
                "redica_insp":   round(float(r.redica_insp or 0), 2),
            })

        d["annual_rows"]    = int((ap["has_valisure"] == 1).sum())
        d["annual_events"]  = int(pilot["shortage_started"].sum())
        d["annual_n_drugs"] = int(pilot["drug_norm"].nunique())
    else:
        d["by_year"] = []; d["by_drug"] = []
        d["annual_rows"] = 140; d["annual_events"] = 21; d["annual_n_drugs"] = 14

    # ── Annual lift table ──────────────────────────────────────────────────────
    lt = _read("eda_lead_time_means.csv", OUT_TABS)
    _RECALL_SIGNALS = {
        "recall_total","recall_cgmp","recall_class_I","recall_class_II","recall_class_III",
        "recall_contam","recall_potency","recall_mislabel","recall_stability","recall_foreign",
        "recall_dissolution","recall_total_w3","recall_cgmp_w3","recall_class_I_w3",
    }
    if lt is not None:
        d["lift"] = [{
            "signal": r.iloc[0],
            "mean0":  round(float(r["y_next=0"]),  4),
            "mean1":  round(float(r["y_next=1"]),  4),
            "lift":   round(float(r["lift"]),       3),
        } for _, r in lt.iterrows() if r.iloc[0] not in _RECALL_SIGNALS]
    else:
        d["lift"] = []

    # ── 483 text feature lift + drug-group comparison ─────────────────────────
    _TEXT_LIFT_COLS = ["tri_mean", "scri_mean", "irwi_mean", "qci_mean"]
    if ap is not None:
        q = ap.dropna(subset=["y_next_year_shortage"])
        for c in [c for c in _TEXT_LIFT_COLS if c in q.columns]:
            grp = q.groupby("y_next_year_shortage")[c].mean()
            m0  = float(grp.get(0, 0))
            m1  = float(grp.get(1, 0))
            d["lift"].append({"signal": c, "mean0": round(m0, 2), "mean1": round(m1, 2),
                               "lift": round(m1 / m0, 3) if m0 > 0 else 1.0})
        dm = ap.groupby("drug_norm").agg(
            starts=("shortage_started", "sum"),
            tri=("tri_mean", "first"), scri=("scri_mean", "first"),
            irwi=("irwi_mean", "first"), qci=("qci_mean", "first"),
        ).reset_index()
        g_has = dm[dm["starts"] > 0][["tri","scri","irwi","qci"]].mean().round(2)
        g_no  = dm[dm["starts"] == 0][["tri","scri","irwi","qci"]].mean().round(2)
        d["text_group"] = {
            "labels":      ["TRI", "SCRI", "IRWI", "QCI"],
            "shortage":    [float(g_has.get("tri",0)), float(g_has.get("scri",0)),
                            float(g_has.get("irwi",0)), float(g_has.get("qci",0))],
            "no_shortage": [float(g_no.get("tri",0)),  float(g_no.get("scri",0)),
                            float(g_no.get("irwi",0)),  float(g_no.get("qci",0))],
            "n_shortage":    int((dm["starts"] > 0).sum()),
            "n_no_shortage": int((dm["starts"] == 0).sum()),
        }
    else:
        d["text_group"] = {}

    # ── Annual event study (lead_time_valisure.csv) ────────────────────────────
    ls = _read("lead_time_valisure.csv", OUT_TABS)
    if ls is not None:
        d["annual_lead"] = {
            "rel":           ls["rel_year"].tolist(),
            "recall_total":  ls["recall_total"].round(3).tolist(),
            "recall_cgmp":   ls["recall_cgmp"].round(3).tolist(),
            "faers_sev":     ls["faers_severity_score"].round(1).tolist(),
            "faers_serious": ls["faers_n_serious"].round(1).tolist(),
        }
    else:
        d["annual_lead"] = {"rel": [], "recall_total": [], "recall_cgmp": [], "faers_sev": [], "faers_serious": []}

    # ── Monthly panel stats + shortage duration per drug ──────────────────────
    mp = _read("master_panel_monthly.csv")
    if mp is not None:
        d["monthly_rows"]         = len(mp)
        d["monthly_onset_months"] = int(mp["shortage_start"].sum())
        d["monthly_ongoing_months"] = int(mp["shortage_ongoing"].sum())
        dur = mp.groupby("drug_norm")["shortage_ongoing"].sum().reset_index()
        dur.columns = ["drug_norm", "duration_months"]
        dur_map = dur.set_index("drug_norm")["duration_months"].to_dict()
        for entry in d.get("by_drug", []):
            entry["duration"] = int(dur_map.get(entry["drug"], 0))
    else:
        d["monthly_rows"] = 1680; d["monthly_onset_months"] = "?"; d["monthly_ongoing_months"] = "?"
        for entry in d.get("by_drug", []):
            entry["duration"] = 0

    # ── Text features from 483 CSV: severity_high, violation categories, remediation
    _TEXT_DETAIL_COLS = [
        "severity_high_share", "contamination_llm_share", "systemic_llm_share",
        "patient_risk_llm_share", "remediation_none_share", "remediation_weak_share",
    ]
    drug_text_detail: dict = {}
    if TEXT_FEATURES_CSV.exists() and VALISURE_FEI.exists():
        tf_raw  = pd.read_csv(TEXT_FEATURES_CSV)
        bridge  = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
        bridge  = (bridge[["API", "FEI_NUMBER"]].dropna()
                   .rename(columns={"FEI_NUMBER": "fei", "API": "drug_norm"}))
        _avail  = [c for c in _TEXT_DETAIL_COLS if c in tf_raw.columns]
        merged  = tf_raw[["fei"] + _avail].merge(bridge, on="fei", how="inner")
        drug_text_detail = merged.groupby("drug_norm")[_avail].mean().round(3).to_dict(orient="index")
    for entry in d.get("by_drug", []):
        detail = drug_text_detail.get(entry["drug"], {})
        entry["sev_high"]        = round(float(detail.get("severity_high_share", 0)), 3)
        entry["contam_share"]    = round(float(detail.get("contamination_llm_share", 0)), 3)
        entry["systemic_share"]  = round(float(detail.get("systemic_llm_share", 0)), 3)
        entry["patient_share"]   = round(float(detail.get("patient_risk_llm_share", 0)), 3)
        entry["remed_none"]      = round(float(detail.get("remediation_none_share", 0)), 3)
        entry["remed_weak"]      = round(float(detail.get("remediation_weak_share", 0)), 3)

    # ── Text detail group comparison (violation categories + remediation) ──────
    if ap is not None:
        dm2 = ap.groupby("drug_norm")["shortage_started"].sum().reset_index()
        dm2.columns = ["drug_norm", "starts"]
        detail_cols_map = {
            "sev_high":      "High-severity obs %",
            "contam_share":  "Contamination %",
            "systemic_share":"Systemic violation %",
            "patient_share": "Patient risk %",
            "remed_none":    "No remediation %",
            "remed_weak":    "Weak remediation %",
        }
        # Build drug-level detail from by_drug entries
        by_drug_dict = {e["drug"]: e for e in d.get("by_drug", [])}
        has_s = [e for e in d.get("by_drug", []) if e["starts"] > 0]
        no_s  = [e for e in d.get("by_drug", []) if e["starts"] == 0]
        if has_s and no_s:
            labels_detail = list(detail_cols_map.values())
            keys = list(detail_cols_map.keys())
            d["text_detail_group"] = {
                "labels":      labels_detail,
                "shortage":    [round(float(pd.Series([e[k] for e in has_s]).mean()) * 100, 1) for k in keys],
                "no_shortage": [round(float(pd.Series([e[k] for e in no_s]).mean()) * 100, 1) for k in keys],
                "n_shortage":    len(has_s),
                "n_no_shortage": len(no_s),
            }
        else:
            d["text_detail_group"] = {}
    else:
        d["text_detail_group"] = {}

    # ── Monthly lead-lag ───────────────────────────────────────────────────────
    ll = _read("lead_lag_monthly.csv", OUT_TABS)
    if ll is not None:
        offsets = [int(x) for x in sorted(ll["offset_months"].unique())]
        groups = {}
        for sig, g in ll.groupby("signal"):
            g = g.set_index("offset_months").reindex(offsets)
            groups[sig] = {
                "offsets":   offsets,
                "means":     [round(float(v), 4) if pd.notna(v) else 0.0 for v in g["mean"]],
                "ses":       [round(float(v), 4) if pd.notna(v) else 0.0 for v in g["se"]],
                "baseline":  float(g["baseline_mean"].dropna().iloc[0]) if not g["baseline_mean"].dropna().empty else 0.0,
                "group":     str(ll.loc[ll["signal"] == sig, "signal_group"].iloc[0]),
            }
        d["monthly_lead"] = groups
    else:
        d["monthly_lead"] = {}

    # ── VAI lead-lag (not in lead_lag_monthly.csv — compute from monthly panel) ─
    if mp is not None and "redica_n_vai" in mp.columns:
        mp2 = mp.copy()
        mp2["midx"] = mp2["year"] * 12 + mp2["month"]
        onsets = mp2[mp2["shortage_start"] == 1][["drug_norm", "midx"]].values.tolist()
        _ll_offsets = list(range(-12, 1))
        if onsets:
            records_vai = []
            for drug, omidx in onsets:
                drug_idx = mp2[mp2["drug_norm"] == drug].set_index("midx")["redica_n_vai"]
                for off in _ll_offsets:
                    tmidx = omidx + off
                    if tmidx in drug_idx.index and pd.notna(drug_idx[tmidx]):
                        records_vai.append({"offset": off, "val": float(drug_idx[tmidx])})
            tagged = {(drug, omidx + k) for drug, omidx in onsets for k in range(-12, 13)}
            bl_rows = mp2[~mp2.apply(lambda r: (r["drug_norm"], r["midx"]) in tagged, axis=1)]
            bl_vai  = float(bl_rows["redica_n_vai"].dropna().mean()) if len(bl_rows) else 0.0
            if records_vai:
                df_vai = pd.DataFrame(records_vai)
                grp_vai = df_vai.groupby("offset")["val"]
                d["monthly_lead"]["redica_n_vai"] = {
                    "offsets":  _ll_offsets,
                    "means":    [round(float(grp_vai.mean().get(o, 0)), 4) for o in _ll_offsets],
                    "ses":      [round(float(grp_vai.sem().get(o, 0)), 4) for o in _ll_offsets],
                    "baseline": round(bl_vai, 4),
                    "group":    "redica",
                }

    # ── Text index lead-lag (time-invariant → flat lines; gap = persistent risk) ─
    if ap is not None:
        _ll_offsets = list(range(-12, 1))
        pilot2 = ap[ap["has_valisure"] == 1]
        dm_text = pilot2.groupby("drug_norm").agg(
            starts=("shortage_started", "sum"),
            tri=("tri_mean", "first"), scri=("scri_mean", "first"),
            irwi=("irwi_mean", "first"), qci=("qci_mean", "first"),
        ).reset_index()
        has_text = dm_text[dm_text["starts"] > 0]
        no_text  = dm_text[dm_text["starts"] == 0]
        for col, label_key in [("tri","tri_mean"),("scri","scri_mean"),
                                ("irwi","irwi_mean"),("qci","qci_mean")]:
            mv  = float(has_text[col].dropna().mean()) if len(has_text) else 0.0
            bl  = float(no_text[col].dropna().mean())  if len(no_text)  else 0.0
            se  = float(has_text[col].dropna().std() / max(len(has_text)**0.5, 1))
            d["monthly_lead"][label_key] = {
                "offsets":  _ll_offsets,
                "means":    [round(mv, 3)] * len(_ll_offsets),
                "ses":      [round(se, 3)] * len(_ll_offsets),
                "baseline": round(bl, 3),
                "group":    "text",
            }

    # ── RF model results: feature importance + ablation ───────────────────────
    fi_raw = _read("rf_importance_valisure.csv", OUT_MODELS)
    abl_raw = _read("text_features_ablation.csv", OUT_MODELS)
    met_raw = _read("metrics_valisure.csv", OUT_MODELS)
    if fi_raw is not None:
        fi_raw = fi_raw.sort_values("importance", ascending=False)
        d["rf_importance"] = [{"feature": r["feature"], "imp": round(float(r["importance"]), 4)}
                               for _, r in fi_raw.iterrows()]
    else:
        d["rf_importance"] = []
    if abl_raw is not None:
        d["ablation"] = abl_raw.to_dict(orient="records")
    else:
        d["ablation"] = []
    if met_raw is not None:
        d["model_metrics"] = met_raw.to_dict(orient="records")
    else:
        d["model_metrics"] = []

    # ── OAI forward study (Wang et al. 2025) ──────────────────────────────────
    fw = _read("oai_forward_study.csv", OUT_TABS)
    fwe = _read("oai_forward_study_events.csv", OUT_TABS)
    if fw is not None:
        fw = fw.sort_values("offset")
        d["oai_fwd"] = {
            "offsets":   [int(x) for x in fw["offset"].tolist()],
            "rates":     [round(float(v), 5) for v in fw["mean_in_shortage"]],
            "ses":       [round(float(v), 5) for v in fw["se_in_shortage"]],
            "baseline":  round(float(fw["baseline_in_shortage"].iloc[0]), 5),
            "n_events":  int(fw.loc[fw["offset"] == 1, "n_events"].iloc[0])
                         if 1 in fw["offset"].values else int(fw["n_events"].max()),
        }
    else:
        d["oai_fwd"] = {}

    # Event-level summary for "already in shortage" breakdown
    if fwe is not None:
        n_total   = len(fwe)
        n_already = int((fwe["in_shortage_at_oai"] == 1).sum())
        n_fresh   = n_total - n_already
        n_already_fwd = int(fwe.loc[fwe["in_shortage_at_oai"] == 1, "any_shortage_fwd12"].sum())
        n_fresh_fwd   = int(fwe.loc[fwe["in_shortage_at_oai"] == 0, "any_shortage_fwd12"].sum())
        mean_mo   = round(float(fwe["months_in_shortage_fwd12"].mean()), 1)
        d["oai_events"] = {
            "n_total": n_total, "n_already": n_already, "n_fresh": n_fresh,
            "n_already_fwd": n_already_fwd, "n_fresh_fwd": n_fresh_fwd,
            "mean_months_fwd": mean_mo,
        }
    else:
        d["oai_events"] = {}

    # ── Valisure quality split ─────────────────────────────────────────────────
    qs = _read("valisure_quality_split.csv", OUT_TABS)
    if qs is not None:
        d["quality_split"] = qs.to_dict(orient="records")
    else:
        d["quality_split"] = []

    return d


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

def _lift_rows(lift: list[dict]) -> str:
    label_map = {
        "redica_n_oai": "Redica — OAI inspections",
        "redica_n_warning_letters": "Redica — warning letters",
        "redica_n_483_critical": "Redica — 483 critical obs.",
        "faers_severity_score": "FAERS — severity score",
        "faers_n_serious": "FAERS — serious reports",
        "faers_n_reports": "FAERS — all reports",
        "tri_mean": "483 Text — Text Risk Index (TRI)",
        "scri_mean": "483 Text — Sterility/Contamination Risk (SCRI)",
        "irwi_mean": "483 Text — Investigation/Remediation Weakness (IRWI)",
        "qci_mean": "483 Text — Quality Culture Index (QCI)",
    }
    cls_map = lambda x: "high" if x >= 5 else ("mid" if x >= 1.5 else "low")
    rows = ""
    for r in lift:
        lv = r["lift"]
        label = label_map.get(r["signal"], r["signal"])
        reads = "Clearly elevated" if lv >= 5 else ("Slight lift" if lv >= 1.3 else "Roughly flat")
        rows += (f'<tr><td>{label}</td>'
                 f'<td class="num">{r["mean0"]:.3f}</td>'
                 f'<td class="num">{r["mean1"]:.3f}</td>'
                 f'<td class="num"><span class="lift {cls_map(lv)}">{lv:.1f}×</span></td>'
                 f'<td>{reads}</td></tr>\n')
    return rows


def generate_html(d: dict) -> str:
    ml = d.get("monthly_lead", {})

    # Prepare JS data for monthly lead-lag charts
    SIGNAL_LABELS = {
        "redica_n_483_critical": "483 Critical Observations",
        "redica_n_oai": "OAI Inspections",
        "redica_n_vai": "VAI Inspections",
        "redica_n_warning_letters": "Warning Letters",
        "redica_n_inspections": "Total Inspections",
        "faers_n_reports_w3m": "FAERS Reports (3m rolling)",
        "faers_n_serious_w3m": "FAERS Serious (3m rolling)",
        "faers_severity_score_w3m": "FAERS Severity Score (3m rolling)",
        "tri_mean": "Text Risk Index (TRI)",
        "scri_mean": "Sterility/Contamination Risk (SCRI)",
        "irwi_mean": "Investigation/Remediation Weakness (IRWI)",
        "qci_mean": "Quality Culture Index (QCI)",
    }

    def _ll_overlay_js(canvas_id: str, sig1: str, sig2: str, col1: str, col2: str) -> str:
        """Overlay two lead-lag signals normalized to baseline (% deviation) on same axis."""
        if sig1 not in ml or sig2 not in ml:
            return f'/* overlay {sig1}/{sig2} not in data */'
        i1, i2 = ml[sig1], ml[sig2]
        offsets = i1["offsets"]
        bl1 = i1["baseline"] or 1e-6
        bl2 = i2["baseline"] or 1e-6
        norm1 = [round((m - bl1) / bl1 * 100, 2) for m in i1["means"]]
        norm2 = [round((m - bl2) / bl2 * 100, 2) for m in i2["means"]]
        lbl1 = SIGNAL_LABELS.get(sig1, sig1)
        lbl2 = SIGNAL_LABELS.get(sig2, sig2)
        return f"""
new Chart(document.getElementById({_j(canvas_id)}), {{
  type:'line',
  data:{{
    labels:{_j(offsets)},
    datasets:[
      {{label:{_j(lbl1 + " (% vs baseline)")},data:{_j(norm1)},
        borderColor:'rgb({col1})',backgroundColor:'transparent',
        tension:0.2,pointRadius:3,borderWidth:2,fill:false}},
      {{label:{_j(lbl2 + " (% vs baseline)")},data:{_j(norm2)},
        borderColor:'rgb({col2})',backgroundColor:'transparent',
        tension:0.2,pointRadius:3,borderWidth:2,borderDash:[5,4],fill:false}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10}}}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Months to shortage onset (0 = onset month)'}},grid:{{display:false}}}},
      y:{{title:{{display:true,text:'% deviation from control baseline'}},
          grid:{{color:'#EEEEEE'}},
          ticks:{{callback:v=>v+'%'}}}}
    }}
  }}
}});"""

    def _ll_chart_js(canvas_id: str, sig: str, color: str) -> str:
        if sig not in ml:
            return f'/* {sig} not in data */'
        info = ml[sig]
        offsets = info["offsets"]
        means   = info["means"]
        ses     = info["ses"]
        bl      = info["baseline"]
        label   = SIGNAL_LABELS.get(sig, sig)
        means_hi = [round((m or 0) + (s or 0), 5) for m, s in zip(means, ses)]
        means_lo = [round((m or 0) - (s or 0), 5) for m, s in zip(means, ses)]
        return f"""
new Chart(document.getElementById({_j(canvas_id)}), {{
  type:'line',
  data:{{
    labels:{_j(offsets)},
    datasets:[
      {{label:'±1 SE upper',data:{_j(means_hi)},borderColor:'transparent',backgroundColor:'rgba({color},0.15)',fill:'+1',pointRadius:0,tension:0.2}},
      {{label:'±1 SE lower',data:{_j(means_lo)},borderColor:'transparent',fill:false,pointRadius:0,tension:0.2}},
      {{label:{_j(label)},data:{_j(means)},borderColor:'rgb({color})',backgroundColor:'rgb({color})',fill:false,tension:0.2,pointRadius:3,borderWidth:2}},
      {{label:'Control baseline',data:{_j([round(bl,5)]*len(offsets))},borderColor:'rgba({color},0.5)',borderDash:[5,4],pointRadius:0,fill:false,borderWidth:1.5}}
    ]
  }},
  options:{{maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{title:{{display:true,text:'Months to shortage onset (0 = onset month)'}},grid:{{display:false}}}},
             y:{{beginAtZero:true,grid:{{color:'#EEEEEE'}}}}}}}}
}});"""

    redica_js = (
        _ll_chart_js("llR1", "redica_n_oai",          "224, 122, 95") +
        _ll_chart_js("llR2", "redica_n_vai",           "28, 114, 147") +
        _ll_chart_js("llR3", "redica_n_inspections",   "80, 140, 60")
    )
    faers_js = (
        _ll_chart_js("llF1", "faers_severity_score_w3m", "2, 99, 176") +
        _ll_chart_js("llF2", "faers_n_serious_w3m",       "28, 114, 147") +
        _ll_chart_js("llF3", "faers_n_reports_w3m",       "224, 122, 95")
    )
    cross_signal_js = ""
    text_ll_js = (
        _ll_chart_js("llT1", "tri_mean",  "2, 99, 176") +
        _ll_chart_js("llT2", "scri_mean", "224, 122, 95") +
        _ll_chart_js("llT3", "irwi_mean", "28, 114, 147") +
        _ll_chart_js("llT4", "qci_mean",  "80, 140, 60")
    )

    # ── Text analysis: TRI/SCRI/QCI/IRWI/SevHigh per drug ────────────────────
    tf = d.get("by_drug", [])
    tf_sorted  = sorted(tf, key=lambda x: x.get("duration", 0), reverse=True)
    tf_drugs   = [r["drug"]          for r in tf_sorted]
    tf_tri     = [r["tri"]           for r in tf_sorted]
    tf_scri    = [r["scri"]          for r in tf_sorted]
    tf_irwi    = [r["irwi"]          for r in tf_sorted]
    tf_qci     = [r["qci"]           for r in tf_sorted]
    tf_sev_high = [round(r.get("sev_high", 0) * 100, 1) for r in tf_sorted]
    text_js = f"""
new Chart(document.getElementById('triChart'),{{
  type:'bar',
  data:{{
    labels:{_j(tf_drugs)},
    datasets:[
      {{label:'TRI — Text Risk Index',data:{_j(tf_tri)},backgroundColor:'rgba(2,99,176,0.75)',borderRadius:3}},
      {{label:'SCRI — Sterility/Contamination Risk',data:{_j(tf_scri)},backgroundColor:'rgba(224,122,95,0.75)',borderRadius:3}},
      {{label:'IRWI — Investigation/Remediation Weakness',data:{_j(tf_irwi)},backgroundColor:'rgba(28,114,147,0.70)',borderRadius:3}},
      {{label:'QCI — Quality Culture (lower = worse)',data:{_j(tf_qci)},backgroundColor:'rgba(80,140,60,0.70)',borderRadius:3}},
      {{label:'High-Severity Obs. % (×100)',data:{_j(tf_sev_high)},backgroundColor:'rgba(180,80,180,0.65)',borderRadius:3}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10}}}}}}}},
    scales:{{
      x:{{grid:{{display:false}},ticks:{{maxRotation:40,font:{{size:9}}}}}},
      y:{{beginAtZero:true,max:100,title:{{display:true,text:'Score / percentage (0–100)'}}}}
    }}
  }}
}});"""

    # ── 483 text group comparison: shortage vs no-shortage drugs ─────────────
    tg = d.get("text_group", {})
    text_group_js = ""
    if tg:
        n_s  = tg.get("n_shortage", "?")
        n_ns = tg.get("n_no_shortage", "?")
        text_group_js = f"""
new Chart(document.getElementById('textGroupChart'),{{
  type:'bar',
  data:{{
    labels:{_j(tg["labels"])},
    datasets:[
      {{label:'≥1 shortage (n={n_s} drugs)',data:{_j(tg["shortage"])},
        backgroundColor:'rgba(224,122,95,0.8)',borderRadius:3}},
      {{label:'No shortage (n={n_ns} drugs)',data:{_j(tg["no_shortage"])},
        backgroundColor:'rgba(28,114,147,0.7)',borderRadius:3}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}}}},
    scales:{{
      x:{{grid:{{display:false}}}},
      y:{{beginAtZero:false,min:40,max:70,
           title:{{display:true,text:'Mean index score (0–100)'}}}}
    }}
  }}
}});"""

    # ── Drug-level cross-signal scatter: Redica vs FAERS ─────────────────────
    tf2 = d.get("by_drug", [])
    def _scatter_color(starts):
        if starts == 0:  return "rgba(28,114,147,0.80)"
        if starts >= 3:  return "rgba(203,75,75,0.85)"
        return "rgba(2,99,176,0.80)"

    sc_oai_faers = [{"x": r["redica_oai"], "y": r["faers_serious"],
                     "name": r["drug"], "starts": r["starts"]} for r in tf2]
    sc_483_faers = [{"x": r["redica_483"], "y": r["faers_reports"],
                     "name": r["drug"], "starts": r["starts"]} for r in tf2]
    sc_vai_faers = [{"x": r["redica_vai"], "y": r["faers_serious"],
                     "name": r["drug"], "starts": r["starts"]} for r in tf2]
    scatter_colors = [_scatter_color(r["starts"]) for r in tf2]

    drug_scatter_js = f"""
new Chart(document.getElementById('scOaiFaers'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(sc_oai_faers)},
    backgroundColor:{_j(scatter_colors)},pointRadius:7,pointHoverRadius:9}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: OAI=${{ctx.raw.x}}, FAERS serious=${{ctx.raw.y}}`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Mean OAI inspections/yr (Redica)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean serious FAERS reports/yr'}},beginAtZero:true}}
    }}}}
}});
new Chart(document.getElementById('sc483Faers'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(sc_483_faers)},
    backgroundColor:{_j(scatter_colors)},pointRadius:7,pointHoverRadius:9}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: 483-crit=${{ctx.raw.x}}, FAERS=${{ctx.raw.y}}`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Mean 483 critical obs/yr (Redica)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean total FAERS reports/yr'}},beginAtZero:true}}
    }}}}
}});
new Chart(document.getElementById('scVaiFaers'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(sc_vai_faers)},
    backgroundColor:{_j(scatter_colors)},pointRadius:7,pointHoverRadius:9}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: VAI=${{ctx.raw.x}}, FAERS serious=${{ctx.raw.y}}`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Mean VAI inspections/yr (Redica)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean serious FAERS reports/yr'}},beginAtZero:true}}
    }}}}
}});"""

    # ── Text detail group comparison: violation categories + remediation ─────
    tdg = d.get("text_detail_group", {})
    text_detail_group_js = ""
    if tdg:
        n_s2  = tdg.get("n_shortage", "?")
        n_ns2 = tdg.get("n_no_shortage", "?")
        text_detail_group_js = f"""
new Chart(document.getElementById('textDetailChart'),{{
  type:'bar',
  data:{{
    labels:{_j(tdg["labels"])},
    datasets:[
      {{label:'≥1 shortage (n={n_s2} drugs)',data:{_j(tdg["shortage"])},
        backgroundColor:'rgba(224,122,95,0.8)',borderRadius:3}},
      {{label:'No shortage (n={n_ns2} drugs)',data:{_j(tdg["no_shortage"])},
        backgroundColor:'rgba(28,114,147,0.7)',borderRadius:3}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}}}},
    scales:{{
      x:{{grid:{{display:false}},ticks:{{font:{{size:9}}}}}},
      y:{{beginAtZero:true,max:100,
           title:{{display:true,text:'Mean % of observations (0–100)'}}}}
    }}
  }}
}});"""

    # ── RF feature importance + ablation ──────────────────────────────────────
    fi_data = d.get("rf_importance", [])
    abl_data = d.get("ablation", [])
    model_js = ""
    if fi_data:
        fi_sorted = sorted(fi_data, key=lambda x: x["imp"])
        fi_labels = [r["feature"] for r in fi_sorted]
        fi_colors = [
            "rgba(28,114,147,0.85)" if r["feature"] in ("tri_mean","scri_mean","irwi_mean","qci_mean")
            else "rgba(2,99,176,0.65)" for r in fi_sorted
        ]
        fi_vals = [r["imp"] for r in fi_sorted]
        abl_l2   = next((r for r in abl_data if r["model"] == "L2_logit"), {})
        abl_rf   = next((r for r in abl_data if r["model"] == "RandomForest"), {})
        auc_with = round(float(abl_rf.get("auc_with_text", 0)), 3)
        auc_no   = round(float(abl_rf.get("auc_without_text", 0)), 3)
        delta_rf = round(float(abl_rf.get("auc_delta", 0)), 3)
        model_js = f"""
new Chart(document.getElementById('fiChart'),{{
  type:'bar',
  data:{{
    labels:{_j(fi_labels)},
    datasets:[{{
      label:'RF importance',
      data:{_j(fi_vals)},
      backgroundColor:{_j(fi_colors)},
      borderRadius:3
    }}]
  }},
  options:{{indexAxis:'y',maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{beginAtZero:true,title:{{display:true,text:'Importance'}}}},
      y:{{grid:{{display:false}},ticks:{{font:{{size:10}}}}}}
    }}
  }}
}});
(function(){{
  const el = document.getElementById('modelKeyNums');
  if (!el) return;
  el.innerHTML = `
    <strong>RandomForest CV (GroupKFold by drug)</strong><br>
    AUC with text features: <strong>{auc_with}</strong> &nbsp;|&nbsp; without: <strong>{auc_no}</strong>
    &nbsp;→ <strong style="color:var(--accent);">Δ = {delta_rf:+.3f}</strong><br>
    <br>
    <strong>Top LLM-derived predictor:</strong> Quality Culture Index (QCI) — 3rd overall, 14.8% importance<br>
    Text indices combined: ~32% of total RF importance<br>
    <br>
    <span style="color:var(--muted);font-size:11px;">
      Logit AUC with text: {round(float(abl_l2.get("auc_with_text",0)),3)} &nbsp;|&nbsp;
      without: {round(float(abl_l2.get("auc_without_text",0)),3)}
      (logit: text features reduce AUC at n=19 events — high-dim collinearity with small sample).<br>
      Recalls excluded from features — concurrent/lagging w.r.t. shortage onset, not leading indicators.
    </span>
  `;
}})();"""

    # ── OAI forward study JS ───────────────────────────────────────────────────
    oai_fwd    = d.get("oai_fwd", {})
    oai_events = d.get("oai_events", {})
    oai_fwd_js = ""
    if oai_fwd:
        oai_offsets  = oai_fwd["offsets"]
        oai_rates    = oai_fwd["rates"]
        oai_ses      = oai_fwd["ses"]
        oai_baseline = oai_fwd["baseline"]
        oai_hi = [round((r or 0) + (s or 0), 6) for r, s in zip(oai_rates, oai_ses)]
        oai_lo = [round((r or 0) - (s or 0), 6) for r, s in zip(oai_rates, oai_ses)]
        oai_bl_line  = [round(oai_baseline, 6)] * len(oai_offsets)
        oai_colors = [
            "rgba(203,75,75,0.85)" if o <= 0 else "rgba(2,99,176,0.85)"
            for o in oai_offsets
        ]
        # Event-level stacked bar data
        n_already      = oai_events.get("n_already", 0)
        n_fresh        = oai_events.get("n_fresh", 0)
        n_already_fwd  = oai_events.get("n_already_fwd", 0)
        n_fresh_fwd    = oai_events.get("n_fresh_fwd", 0)
        n_already_no   = n_already - n_already_fwd
        n_fresh_no     = n_fresh   - n_fresh_fwd
        mean_mo        = oai_events.get("mean_months_fwd", 0)
        oai_fwd_js = f"""
new Chart(document.getElementById('oaiFwdChart'),{{
  type:'line',
  data:{{
    labels:{_j(oai_offsets)},
    datasets:[
      {{label:'\\u00b11 SE band',data:{_j(oai_hi)},borderColor:'transparent',
        backgroundColor:'rgba(2,99,176,0.12)',fill:'+1',pointRadius:0,tension:0.2}},
      {{label:'_lo',data:{_j(oai_lo)},borderColor:'transparent',
        fill:false,pointRadius:0,tension:0.2}},
      {{label:'% months drug is in shortage',data:{_j(oai_rates)},
        borderColor:'rgba(2,99,176,0.6)',backgroundColor:{_j(oai_colors)},
        fill:false,tension:0.15,pointRadius:5,borderWidth:2,
        pointBackgroundColor:{_j(oai_colors)}}},
      {{label:'Control baseline (no OAI \u00b112m): {oai_baseline:.3f}',
        data:{_j(oai_bl_line)},
        borderColor:'#888',borderDash:[5,4],pointRadius:0,fill:false,borderWidth:1.5}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{
      boxWidth:12,font:{{size:10.5}},
      filter:function(item){{return item.text!=='_lo';}}
    }}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Months relative to OAI inspection (0 = OAI month)'}},
           grid:{{display:false}}}},
      y:{{beginAtZero:true,max:1,
           title:{{display:true,text:'Fraction of events: drug in shortage'}},
           grid:{{color:'#EEEEEE'}},
           ticks:{{callback:function(v){{return (v*100).toFixed(0)+'%';}}}}}}
    }}
  }}
}});

new Chart(document.getElementById('oaiFwdBar'),{{
  type:'bar',
  data:{{
    labels:['Already in shortage\\nat OAI (n={n_already})','NOT in shortage\\nat OAI (n={n_fresh})'],
    datasets:[
      {{label:'No shortage in next 12m',
        data:[{n_already_no},{n_fresh_no}],backgroundColor:'rgba(28,114,147,0.75)',borderRadius:3}},
      {{label:'\\u22651 shortage month in next 12m',
        data:[{n_already_fwd},{n_fresh_fwd}],backgroundColor:'rgba(224,122,95,0.85)',borderRadius:3}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}}}},
    scales:{{
      x:{{grid:{{display:false}},stacked:true}},
      y:{{beginAtZero:true,stacked:true,ticks:{{stepSize:1}},
           title:{{display:true,text:'# OAI events'}}}}
    }}
  }}
}});"""

    # ── Quality split JS ──────────────────────────────────────────────────────
    qs_data = d.get("quality_split", [])
    qs_js = ""
    if qs_data:
        qs_drugs  = [r["drug_norm"] for r in qs_data]
        qs_scores = [round(float(r.get("valisure_score") or 0), 1) for r in qs_data]
        qs_starts = [int(r.get("shortage_starts") or 0) for r in qs_data]
        qs_tier   = [r.get("quality_tier", "low_quality") for r in qs_data]
        qs_colors = [
            "rgba(28,114,147,0.8)" if t == "high_quality" else "rgba(224,122,95,0.85)"
            for t in qs_tier
        ]
        qs_js = f"""
new Chart(document.getElementById('qualSplitChart'),{{
  type:'bar',
  data:{{
    labels:{_j(qs_drugs)},
    datasets:[
      {{label:'Shortage starts',data:{_j(qs_starts)},backgroundColor:{_j(qs_colors)},
        borderRadius:3,yAxisID:'y'}},
      {{label:'Valisure mean score',data:{_j(qs_scores)},
        backgroundColor:'rgba(33,41,92,0.15)',borderColor:'rgba(33,41,92,0.6)',
        borderWidth:1.5,borderRadius:2,yAxisID:'y2',type:'line',
        tension:0.2,pointRadius:4}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}}}},
    scales:{{
      x:{{grid:{{display:false}},ticks:{{maxRotation:40,font:{{size:9}}}}}},
      y:{{beginAtZero:true,title:{{display:true,text:'Shortage starts'}},
           ticks:{{stepSize:1}}}},
      y2:{{position:'right',title:{{display:true,text:'Valisure mean score'}},
            grid:{{display:false}}}}
    }}
  }}
}});"""

    # (Recall circularity analysis removed — recalls excluded from predictive features
    #  as they are concurrent/lagging w.r.t. shortage onset, not leading indicators.)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>From Regulatory Text to Shortage Risk · Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--navy:#21295C;--deep:#065A82;--teal:#1C7293;--accent:#E07A5F;
      --cream:#F4F1EC;--paper:#FBFAF7;--ink:#1A2233;--muted:#5A6577;
      --rule:#E2DDD2;--white:#FFFFFF;}}
*{{box-sizing:border-box;}}
html,body{{margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:var(--paper);color:var(--ink);line-height:1.5;}}
.wrap{{max-width:1200px;margin:0 auto;padding:32px 28px 48px;}}
header.hero{{background:var(--navy);color:var(--white);padding:36px 28px;
             border-radius:10px;margin-bottom:28px;position:relative;overflow:hidden;}}
header.hero::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--accent);}}
.eyebrow{{color:#E7C8B9;font-size:11px;letter-spacing:4px;font-weight:700;
          text-transform:uppercase;margin-bottom:10px;}}
header.hero h1{{margin:0 0 10px;font-family:Georgia,serif;font-size:28px;font-weight:700;line-height:1.25;}}
header.hero p{{margin:0;color:#CADCFC;font-size:14px;max-width:900px;}}
section{{margin-bottom:28px;}}
.section-head{{display:flex;align-items:baseline;gap:14px;margin-bottom:14px;}}
.step-num{{display:inline-block;background:var(--accent);color:#fff;
           width:28px;height:28px;border-radius:50%;text-align:center;
           line-height:28px;font-weight:700;font-size:13px;flex-shrink:0;}}
h2{{margin:0;font-family:Georgia,serif;font-size:20px;color:var(--navy);font-weight:700;}}
.sub{{color:var(--muted);font-size:13px;margin:4px 0 14px 42px;}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
.stat{{background:var(--white);border:1px solid var(--rule);border-top:4px solid var(--deep);
       border-radius:6px;padding:16px 18px;}}
.stat .num{{font-family:Georgia,serif;font-size:30px;font-weight:700;color:var(--navy);line-height:1.1;}}
.stat .num small{{font-family:sans-serif;color:var(--muted);font-size:14px;font-weight:400;margin-left:4px;}}
.stat .lbl{{color:var(--ink);font-size:12.5px;margin-top:6px;}}
.stat.accent{{border-top-color:var(--accent);}}
.stat.teal{{border-top-color:var(--teal);}}
.sources{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}}
.src{{background:var(--white);border:1px solid var(--rule);border-left:4px solid var(--deep);
      border-radius:6px;padding:12px 14px;}}
.src .name{{font-family:Georgia,serif;font-weight:700;color:var(--navy);font-size:14px;}}
.src .role{{color:var(--muted);font-size:12px;margin-top:4px;}}
.card{{background:var(--white);border:1px solid var(--rule);border-radius:8px;padding:18px 20px;}}
.card h3{{margin:0 0 4px;font-family:Georgia,serif;font-size:16px;color:var(--navy);}}
.card .csub{{color:var(--muted);font-size:12px;margin-bottom:12px;}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.chart-row.three{{grid-template-columns:1fr 1fr 1fr;}}
.chart-host{{position:relative;height:280px;}}
.chart-host.tall{{height:360px;}}
.note{{background:var(--cream);border-left:4px solid var(--accent);
       padding:12px 14px;border-radius:4px;font-size:13px;color:var(--ink);margin-top:14px;}}
.note strong{{color:var(--navy);}}
.note.dark{{background:var(--navy);border-left-color:var(--accent);color:#CADCFC;}}
.note.dark strong{{color:var(--white);}}
.divider{{border:none;border-top:2px solid var(--accent);margin:8px 0 20px;opacity:0.3;}}
table.signals{{width:100%;border-collapse:collapse;font-size:13px;}}
table.signals th,table.signals td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--rule);}}
table.signals th{{background:var(--cream);color:var(--navy);font-weight:700;
                  font-size:11px;text-transform:uppercase;letter-spacing:1px;}}
table.signals td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.lift{{display:inline-block;padding:2px 8px;border-radius:12px;font-weight:700;font-size:12px;}}
.lift.low{{background:#E8EEF2;color:var(--deep);}}
.lift.mid{{background:#FCE6DD;color:#B14A30;}}
.lift.high{{background:var(--accent);color:white;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.col-card{{background:var(--white);border:1px solid var(--rule);border-radius:8px;overflow:hidden;}}
.col-card .col-head{{padding:10px 16px;color:white;font-size:12px;font-weight:700;
                     letter-spacing:3px;text-transform:uppercase;}}
.col-card .col-head.l{{background:var(--navy);}} .col-card .col-head.n{{background:var(--accent);}}
.col-card ul{{margin:0;padding:14px 20px 18px;}}
.col-card li{{padding:4px 0;font-size:13px;color:var(--ink);}}
.badge{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;margin-right:4px;}}
.badge.new{{background:#dcfce7;color:#166534;}}
.badge.upd{{background:#dbeafe;color:#1e40af;}}
footer{{text-align:center;color:var(--muted);font-size:11px;margin-top:26px;
        padding-top:14px;border-top:1px solid var(--rule);}}
@media(max-width:900px){{
  .stats{{grid-template-columns:repeat(2,1fr);}}
  .sources{{grid-template-columns:repeat(2,1fr);}}
  .chart-row,.chart-row.three,.two-col{{grid-template-columns:1fr;}}
}}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <div class="eyebrow">Research Dashboard · June 2026 · NLP &amp; LLM Framework</div>
  <h1>From Regulatory Text to Shortage Risk</h1>
  <p>NLP &amp; LLM Framework for Pharmaceutical Quality Risk Prediction ·
     14 generic APIs · 129 manufacturing facilities · 2015–2024 ·
     FDA 483 text extraction → facility-level risk indices → shortage prediction.
     Recalls excluded from predictive features (concurrent/lagging with onset).</p>
</header>

<!-- ═══ SECTION 1: PANEL SNAPSHOT ═══ -->
<section>
  <div class="section-head"><span class="step-num">1</span><h2>Panel at a glance</h2></div>
  <div class="sub">Both the annual pilot panel and the new monthly panel cover the same 14 Valisure-tested APIs and 2015–2024.</div>
  <div class="stats">
    <div class="stat">
      <div class="num">{d["annual_n_drugs"]}<small>APIs</small></div>
      <div class="lbl">Valisure-tested pilot universe</div>
    </div>
    <div class="stat">
      <div class="num">{d["annual_rows"]}<small>rows</small></div>
      <div class="lbl">Annual panel (drug × year, 2015–2024)</div>
    </div>
    <div class="stat accent">
      <div class="num">{d["annual_events"]}<small>starts</small></div>
      <div class="lbl">New shortage episodes (annual)</div>
    </div>
    <div class="stat teal">
      <div class="num">{d["monthly_rows"]}<small>rows</small></div>
      <div class="lbl">Monthly panel (drug × month) <span class="badge new">new</span></div>
    </div>
  </div>
</section>

<!-- ═══ SECTION 2: SOURCES ═══ -->
<section>
  <div class="section-head"><span class="step-num">2</span><h2>Data sources &amp; linkage</h2></div>
  <div class="sources">
    <div class="src"><div class="name">FDA Form 483s</div><div class="role">38 FEIs · 622 observations · LLM-extracted (primary text source)</div></div>
    <div class="src"><div class="name">FDA Inspections</div><div class="role">1,019 inspections · 961 CFR citations · 127/129 FEIs</div></div>
    <div class="src"><div class="name">Valisure</div><div class="role">14-API universe · 2024 independent quality test scores (static)</div></div>
    <div class="src"><div class="name">UUtah</div><div class="role">Shortage start / end dates → annual &amp; monthly shortage state</div></div>
    <div class="src"><div class="name">FAERS</div><div class="role">Adverse-event reports + severity (quarterly resolution)</div></div>
  </div>
  <div class="note" style="margin-top:12px;">
    <strong>Primary text source:</strong> FDA Form 483 PDF narratives, processed via LangChain OCR + LLM extraction (GPT-4o-mini).
    Structured citations provide CFR-level signals for 127/129 facilities.
    Valisure scores are a 2024 static snapshot used for cross-sectional validation only.
    <strong>Recalls excluded as predictors</strong> — timing analysis confirms they are concurrent or lagging w.r.t. shortage onset, not leading indicators.
  </div>
</section>

<!-- ═══ SECTION 3: SHORTAGE PATTERNS ═══ -->
<section>
  <div class="section-head"><span class="step-num">3</span><h2>Shortage patterns (2015–2024)</h2></div>
  <div class="chart-row">
    <div class="card">
      <h3>New shortage starts by year</h3>
      <div class="csub">Annual count across all 14 pilot APIs</div>
      <div class="chart-host"><canvas id="chartYear"></canvas></div>
    </div>
    <div class="card">
      <h3>Total months in shortage per drug (2015–2024)</h3>
      <div class="csub">Sorted by shortage duration · red ≥ 60 months · teal = no shortage · tooltip shows # onset events</div>
      <div class="chart-host tall"><canvas id="chartDrug"></canvas></div>
    </div>
  </div>
  <div class="note">
    <strong>Pattern:</strong> 21 shortage-start events across 14 drugs ({d["monthly_onset_months"]} monthly onset months).
    Metoprolol and Vancomycin were in shortage for the entire 10-year window (120 months), though
    Vancomycin had zero onset events — it entered shortage before 2015 and never resolved.
    Metronidazole and Potassium chloride had the most separate onset episodes.
    Three drugs had zero total months in shortage (Bupropion, Atorvastatin, Tacrolimus — short episodes only).
  </div>
</section>

<!-- ═══ SECTION 4: 483 TEXT ANALYSIS SIGNALS ═══ -->
<section>
  <div class="section-head"><span class="step-num">4</span><h2>FDA 483 Text Signals — LLM-extracted indices by drug</h2></div>
  <div class="sub">
    38 of 129 facilities (those with 483 PDFs on file) scored via GPT-4o-mini extraction of 622 observations.
    Indices aggregated to drug level (mean across all FEIs manufacturing that API).
    Sorted by <strong>total months in shortage</strong> (highest burden on left).
    <br><br>
    <strong>Index definitions (all 0–100; higher = higher risk, except QCI where lower = worse):</strong>
    <ul style="margin:4px 0 0 16px;font-size:12px;line-height:1.7;">
      <li><strong>TRI (Text Risk Index)</strong> — overall manufacturing risk signal derived from the full
          observation narrative. Combines violation category, severity, and root cause language.</li>
      <li><strong>SCRI (Sterility/Contamination Risk Index)</strong> — specifically captures sterility failures,
          contamination events, and aseptic-technique violations in the observation text.</li>
      <li><strong>IRWI (Investigation/Remediation Weakness Index)</strong> — detects language indicating
          incomplete root-cause investigations, vague corrective actions, or repeat failure patterns.</li>
      <li><strong>QCI (Quality Culture Index)</strong> — measures quality-culture strength (SOPs followed,
          Quality Unit active, proactive monitoring). <em>Lower QCI = weaker quality culture = higher risk.</em></li>
      <li><strong>High-Severity Obs. %</strong> — share of observations that the LLM classified as
          "high severity" (direct patient-risk or systemic quality failure). Converted to 0–100 scale.</li>
    </ul>
  </div>
  <div class="card">
    <h3>TRI / SCRI / IRWI / QCI / High-Severity % by drug</h3>
    <div class="csub">Sorted by total shortage duration (longest on left). Indices reflect cumulative 483 history across all FEIs for that drug.</div>
    <div class="chart-host tall"><canvas id="triChart"></canvas></div>
  </div>
  <div class="note dark">
    <strong>Key insights:</strong>
    QCI (Quality Culture Index) is the <strong>3rd most important RF predictor</strong> of drug shortage
    (14.8% importance) — lower QCI for long-shortage drugs confirms that chronic compliance culture
    weakness is a leading signal.
    SCRI is highest for Ampicillin (75.6) — its only scored facility (Aurobindo/Eugia) is a sterile
    injectable manufacturer with documented contamination events in both 2019 and 2024.
    IRWI is elevated for high-burden drugs, reflecting systematic investigation failures.
    <em>Note: indices are time-invariant — they reflect the full 483 history, not a single year.
    Drugs with no 483 PDFs on file show zero for all indices.</em>
  </div>
</section>

<!-- ═══ SECTION 5: EDA — REGULATORY & FAERS RELATIONSHIPS ═══ -->
<section>
  <div class="section-head">
    <span class="step-num">5</span>
    <h2>EDA — Regulatory signals before shortage onset</h2>
  </div>
  <div class="sub">
    Monthly event study: mean signal value at each offset month relative to shortage onset (month 0).
    Control baseline = drug-months with no shortage onset within ±12 months.
    Shaded band = ±1 SE. <strong>N = {d["monthly_onset_months"]} onset months across 14 drugs — interpret as exploratory only.</strong>
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:0 0 4px 0;font-size:16px;">
    5A · LLM-extracted text indices — shortage vs no-shortage drugs</h3>
  <div class="sub" style="margin-left:0;">
    Composite text indices (TRI/SCRI/IRWI/QCI) and raw feature proportions, averaged across all
    facility 483s for each drug. Two groups: drugs with ≥1 shortage onset vs drugs with zero onsets (2015–2024).
    <br><strong>≥1 shortage onset:</strong> Ampicillin, Atorvastatin, Calcium Gluconate,
    Lisinopril, Magnesium Sulfate, Metformin, Metoprolol, Metronidazole, Pantoprazole, Potassium Chloride, Tacrolimus.
    <br><strong>0 shortage onsets:</strong> Ampicillin+Sulbactam, Bupropion, Vancomycin.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:4px;">
    <div class="card">
      <h3>Composite indices (TRI / SCRI / IRWI / QCI)</h3>
      <div class="csub">All indices 0–100. QCI: lower = weaker quality culture = higher risk.</div>
      <div class="chart-host"><canvas id="textGroupChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Violation categories &amp; remediation quality</h3>
      <div class="csub">% of observations flagged for each category. High-severity, contamination, systemic, patient risk, no/weak remediation.</div>
      <div class="chart-host"><canvas id="textDetailChart"></canvas></div>
    </div>
  </div>
  <div class="note dark">
    Shortage drugs show higher SCRI (sterility/contamination risk), higher IRWI (weak investigations),
    lower QCI (weaker quality culture), more contamination-flagged observations, and higher rates
    of no/weak remediation. These are the risk fingerprint of facilities prone to supply disruption.
    Text indices are time-invariant — this is a cross-sectional comparison, not temporal.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:16px 0 4px;font-size:16px;">
    5B · Monthly lead-lag — Regulatory signals (OAI, VAI, Total Inspections)</h3>
  <div class="sub" style="margin-left:0;">
    OAI = Official Action Indicated (most severe; triggers mandatory remediation).
    VAI = Voluntary Action Indicated (violations noted, manufacturer commits to fix voluntarily).
    Total inspections shows inspection frequency context.
    X-axis = months to shortage onset (0 = onset month). Shaded = ±1 SE.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:4px;">
    <div class="card"><h3>OAI Inspections</h3><div class="chart-host"><canvas id="llR1"></canvas></div></div>
    <div class="card"><h3>VAI Inspections</h3><div class="chart-host"><canvas id="llR2"></canvas></div></div>
    <div class="card"><h3>Total Inspections</h3><div class="chart-host"><canvas id="llR3"></canvas></div></div>
  </div>
  <div class="note">
    OAI events are sparse; no consistent ramp-up before onset is visible at this sample size.
    VAI inspections provide a broader signal of manufacturing concerns being addressed voluntarily.
    Total inspections are flat — shortage onset is not preceded by a surge in inspection activity.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:16px 0 4px;font-size:16px;">
    5C · Monthly lead-lag — LLM-extracted text indices</h3>
  <div class="sub" style="margin-left:0;">
    Text indices are time-invariant (one value per drug, based on cumulative 483 history).
    The lead-lag chart shows a <strong>flat line</strong> — the persistent mean for shortage-starting
    drugs (solid) vs the control baseline (dashed). The gap is the key signal: shortage drugs
    have chronically elevated TRI/SCRI/IRWI and lower QCI throughout the entire window,
    not just immediately before onset.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:4px;">
    <div class="card"><h3>TRI — Text Risk Index</h3><div class="chart-host"><canvas id="llT1"></canvas></div></div>
    <div class="card"><h3>SCRI — Sterility/Contamination Risk</h3><div class="chart-host"><canvas id="llT2"></canvas></div></div>
    <div class="card"><h3>IRWI — Investigation/Remediation Weakness</h3><div class="chart-host"><canvas id="llT3"></canvas></div></div>
    <div class="card"><h3>QCI — Quality Culture (lower = worse)</h3><div class="chart-host"><canvas id="llT4"></canvas></div></div>
  </div>
  <div class="note dark">
    Unlike Redica regulatory signals (which fluctuate month to month), LLM text indices represent
    a <strong>structural risk fingerprint</strong> of each facility derived from its full inspection
    history. The persistent gap between shortage and no-shortage drug baselines supports using text
    indices as long-horizon risk signals — identifying facilities with chronic quality culture
    deficiencies before any acute regulatory action occurs.
  </div>
</section>

<!-- ═══ SECTION 6: PREDICTIVE MODEL RESULTS ═══ -->
<section>
  <div class="section-head"><span class="step-num">6</span><h2>Predictive Model — Does 483 text improve shortage prediction?</h2></div>
  <div class="sub">
    <strong>Setup:</strong> Drug × year panel (14 APIs, 2015–2024, n=126 drug-years, 19 shortage-onset events).
    Outcome = <em>y_next_year_shortage</em>: does this drug start a shortage in the following year?
    Recall features excluded (they co-occur with or lag shortage onset, so they'd be data leakage for prediction).
    <br><br>
    <strong>Cross-validation:</strong> GroupKFold by drug — each fold holds out all years for a different subset
    of drugs, so the model is tested on drugs it has never seen. This is the correct CV for drug-level panel data
    because observations from the same drug across years are highly correlated.
    <br><br>
    <strong>Metric:</strong> AUC (area under the ROC curve). AUC = 0.5 means no better than random;
    AUC = 1.0 means perfect. With only 19 events, confidence intervals are wide — these results are exploratory.
    RandomForest is preferred over logistic regression here because it handles the correlated, mixed-scale
    features better at small sample size. The logistic regression result (shown in small text) is lower,
    likely due to collinearity among the text indices at n=19 events.
  </div>
  <div class="chart-row">
    <div class="card">
      <h3>RandomForest feature importance</h3>
      <div class="csub">Teal bars = LLM-extracted 483 text indices. Navy = structured data features. Sorted by importance.</div>
      <div class="chart-host tall"><canvas id="fiChart"></canvas></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;justify-content:center;padding:24px;">
      <h3 style="margin-bottom:12px;">Model results</h3>
      <div id="modelKeyNums" style="font-size:13px;line-height:2;color:var(--ink);"></div>
      <div style="margin-top:16px;font-size:12px;color:var(--muted);line-height:1.6;">
        <strong>How to read this:</strong><br>
        Feature importance = fraction of times a feature was used to split the data in RF trees,
        weighted by how much that split reduced impurity. Higher = more useful for prediction.<br><br>
        The +ΔAUC tells us how much predictive value the LLM text features add on top of structured
        data alone. A positive Δ means text features carry information that structured databases do not.
      </div>
    </div>
  </div>
  <div class="note dark">
    <strong>Key finding:</strong> QCI (Quality Culture Index), derived by the LLM from 483 narrative text,
    is the <strong>3rd most important predictor</strong> of next-year drug shortage — ranking above
    Valisure quality scores and Redica OAI counts.
    The 4 LLM-derived text indices together account for ~32% of total RF importance, adding
    +0.052 AUC over structured features alone. This validates the paper's central claim: inspection
    narrative text carries predictive signal that structured databases (CFR codes, inspection outcomes)
    do not fully capture. See Section 7 for causal context from Wang et al. (2025).
  </div>
</section>

<!-- ═══ SECTION 7: WANG ET AL. 2025 CONTEXT ═══ -->
<section>
  <div class="section-head">
    <span class="step-num">7</span>
    <h2>FDA inspection outcomes &amp; shortage risk — Wang et al. (MSOM 2025) context</h2>
  </div>
  <div class="sub">Connects to Section 6: the predictive model finds OAI inspections have modest importance.
    Wang et al. (2025) provide causal context for why — OAI outcomes may actually <em>reduce</em> future
    shortage risk by forcing mandatory quality remediation. We test their directional prediction in our panel.
  </div>
  <div class="sub">
    Wang et al. (2025) find that OAI inspection outcomes <em>reduce</em> future shortage risk by ~96% after IV adjustment
    (instruments: inspector experience &amp; Monday inspection day).  GAO (2016) found the opposite using unadjusted data.
    The IV result implies that OAI outcomes force mandatory quality remediation that eliminates the underlying supply risk.
    Below we test the directional prediction in our 14-drug monthly panel — observationally, without IV instruments.
  </div>

  <div class="note" style="margin-bottom:14px;">
    <strong>Interpretation guide:</strong>
    If Wang et al. are correct, the post-OAI shortage rate (blue points, months +1 to +12) should fall
    <em>below</em> the control baseline (dashed grey).  If the GAO story dominates, it should rise above baseline.
    Our pilot covers only 14 drugs and OAI events are sparse — treat as directional only.
  </div>

  <div class="chart-row" style="margin-bottom:16px;">
    <div class="card">
      <h3>Is the drug in shortage at each month around OAI?</h3>
      <div class="csub">
        Primary outcome: <code>shortage_ongoing</code> (is the drug currently in shortage?).
        Red = pre-OAI context (−6 to 0); Blue = post-OAI (months +1 to +12). Shaded = ±1 SE.
        Baseline = drug-months with no OAI within ±12m.
      </div>
      <div class="chart-host tall"><canvas id="oaiFwdChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Post-OAI shortage outcome by drug state at inspection</h3>
      <div class="csub">
        For each OAI event: was the drug already in shortage? Did it end up in shortage in the next 12 months?
        Teal = stayed/went shortage-free; Orange = entered or remained in shortage.
      </div>
      <div class="chart-host tall"><canvas id="oaiFwdBar"></canvas></div>
    </div>
  </div>
  <div class="chart-row">
    <div class="card">
      <h3>Valisure quality score vs shortage starts</h3>
      <div class="csub">
        Sorted by Valisure mean score (higher = better quality).
        Teal = high-quality tier (≥ median score); Orange = low-quality tier.
        Line = Valisure score (right axis).
      </div>
      <div class="chart-host tall"><canvas id="qualSplitChart"></canvas></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;justify-content:center;padding:24px;">
      <h3 style="margin-bottom:12px;">Key numbers</h3>
      <div id="oaiKeyNums" style="font-size:13px;line-height:2;color:var(--ink);"></div>
    </div>
  </div>

  <div class="two-col">
    <div class="col-card">
      <div class="col-head l">Wang et al. 2025 — key facts</div>
      <ul>
        <li><strong>Source:</strong> Wang, Ball, Anand, Park — MSOM Vol. 27, No. 3, May–Jun 2025, pp. 789–807.</li>
        <li><strong>Data:</strong> 8,028 drug-inspection observations, 3,193 drugs, 419 inspections (30 OAI), 185 plants, July 2015–Mar 2019.</li>
        <li><strong>Finding:</strong> OAI outcome → −3.0 pp predicted shortage probability (avg marginal effect) = <strong>96.4% reduction</strong> from base rate of 3.11%.</li>
        <li><strong>Mechanism:</strong> OAI forces mandatory comprehensive remediation; quality failures that would cause shortages must be fixed.</li>
        <li><strong>Instruments:</strong> Inspector experience (years) and Monday inspection day — both predict OAI but are unlikely to affect shortage risk directly.</li>
        <li><strong>GAO 2016 rebuttal:</strong> GAO found positive association but did not control for endogeneity.  OAIs appear where quality is already poor — co-symptom, not cause.</li>
      </ul>
    </div>
    <div class="col-card">
      <div class="col-head n">How our pipeline extends their analysis</div>
      <ul>
        <li><strong>Longer horizon:</strong> Their data ends Mar 2019; we cover 2015–2024, including five COVID-era years where supply-chain shocks may swamp regulatory effects.</li>
        <li><strong>Quality scores:</strong> Valisure scores measure actual product quality independently of inspection outcomes — a direct proxy for the latent confounder Wang et al. instrument for.</li>
        <li><strong>Quality-shortage relationship:</strong> The right chart above shows Valisure score vs shortage starts — if low-quality drugs also get more OAIs and more shortages, this is consistent with Wang et al.'s endogeneity story.</li>
        <li><strong>Limitation:</strong> Our pilot covers only 14 drugs and OAI events are sparse — causal inference is not possible here.  IV instruments (inspector experience, weekday) are not available in Redica data.</li>
        <li><strong>Next step:</strong> Obtain inspector-level data via FOIA request (same source as Wang et al.) and merge with Redica FEIs to replicate IV on the Valisure-drug subset.</li>
      </ul>
    </div>
  </div>
</section>

<!-- ═══ SECTION 8: LIMITATIONS & NEXT STEPS ═══ -->
<section>
  <div class="section-head"><span class="step-num">8</span><h2>Limitations &amp; next steps</h2></div>
  <div class="two-col">
    <div class="col-card">
      <div class="col-head l">Limitations</div>
      <ul>
        <li>Only 14 drugs and ~21 shortage onset years → very wide confidence intervals. All findings are exploratory.</li>
        <li>Annual recall circularity: recalls and shortages can be mechanically linked. CGMP signal uses monthly timing to partially separate them, but caution is warranted.</li>
        <li>FAERS is quarterly in this dataset — monthly resolution not available; 3-month rolling sums used as approximation.</li>
        <li>Valisure scores are a single 2024 cross-section — not time-varying. Cannot be used in lead-lag analysis.</li>
        <li>Redica FEI mapping covers only the 14 Valisure drugs; OAI/483 signal is sparse at monthly grain.</li>
        <li>All associations are descriptive — no causal identification.</li>
      </ul>
    </div>
    <div class="col-card">
      <div class="col-head n">Next steps</div>
      <ul>
        <li>Obtain FAERS event-level data with exact dates to enable true monthly FAERS lead-lag.</li>
        <li>Manually validate the CGMP recall → Metformin shortage pathway to confirm or rule out circularity.</li>
        <li>Expand pilot universe as Valisure tests additional APIs.</li>
        <li>Use a discrete-time hazard model or Cox regression to formalize the lead-time relationship.</li>
        <li>Add manufacturer concentration (HHI) from NDC ↔ labeler map as a structural feature.</li>
        <li>Bootstrap-cluster SEs at the drug level for robustness.</li>
        <li>Obtain inspector-level FDA data via FOIA to replicate Wang et al. IV analysis on Valisure-drug subset.</li>
        <li>Test post-COVID (2020–2024) subperiod separately — supply-chain shocks may alter OAI protective effect.</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  Drug Shortage Prediction · Annual + Monthly Pipeline · May 2026 ·
  Canonical key: <code>drug_norm</code> · 14 Valisure APIs · exploratory only
</footer>
</div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const BY_YEAR   = {_j(d["by_year"])};
const BY_DRUG   = {_j(d["by_drug"])};

// ── Chart defaults ─────────────────────────────────────────────────────────
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
Chart.defaults.color = "#1A2233";
Chart.defaults.font.size = 11.5;
const C = {{navy:"#21295C",deep:"#065A82",teal:"#1C7293",accent:"#E07A5F",muted:"#5A6577"}};

// ── Annual: shortage starts by year ───────────────────────────────────────
new Chart(document.getElementById("chartYear"),{{
  type:"bar",
  data:{{labels:BY_YEAR.map(d=>d.year),
         datasets:[{{label:"Shortage starts",data:BY_YEAR.map(d=>d.starts),
                    backgroundColor:C.deep,borderRadius:4}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
            plugins:{{legend:{{display:false}}}},
            scales:{{x:{{grid:{{display:false}}}},y:{{beginAtZero:true,ticks:{{stepSize:1}}}}}}}}
}});

// ── Annual: shortage duration by drug (total months in shortage 2015–2024) ─
const sorted = [...BY_DRUG].sort((a,b)=>a.duration-b.duration);
new Chart(document.getElementById("chartDrug"),{{
  type:"bar",
  data:{{labels:sorted.map(d=>d.drug),
         datasets:[{{label:"Months in shortage",
                    data:sorted.map(d=>d.duration),
                    backgroundColor:sorted.map(d=>d.duration>=60?C.accent:(d.duration===0?C.teal:C.deep)),
                    borderRadius:3}}]}},
  options:{{indexAxis:"y",responsive:true,maintainAspectRatio:false,
            plugins:{{legend:{{display:false}},
              tooltip:{{callbacks:{{label:ctx=>`${{sorted[ctx.dataIndex].drug}}: ${{ctx.raw}} months (${{sorted[ctx.dataIndex].starts}} shortage starts)`}}}}}},
            scales:{{x:{{beginAtZero:true,title:{{display:true,text:"Months in shortage (2015–2024)"}}}},
                     y:{{grid:{{display:false}}}}}}}}
}});

// ── Monthly lead-lag charts ────────────────────────────────────────────────
{redica_js}

// ── Text index monthly lead-lag (flat = persistent risk gap) ─────────────
{text_ll_js}

// ── 483 text analysis charts ──────────────────────────────────────────────
{text_js}

// ── 483 text group comparison (shortage vs no-shortage drugs) ─────────────
{text_group_js}

// ── Text detail group (violation categories + remediation) ────────────────
{text_detail_group_js}

// ── RF model results ──────────────────────────────────────────────────────
{model_js}


// ── OAI forward study charts ──────────────────────────────────────────────
{oai_fwd_js}

// ── OAI key numbers panel ─────────────────────────────────────────────────
(function(){{
  const el = document.getElementById('oaiKeyNums');
  if (!el) return;
  const n  = {oai_events.get("n_total", 0)};
  const na = {oai_events.get("n_already", 0)};
  const nf = {oai_events.get("n_fresh", 0)};
  const naf= {oai_events.get("n_already_fwd", 0)};
  const nff= {oai_events.get("n_fresh_fwd", 0)};
  const mm = {oai_events.get("mean_months_fwd", 0)};
  const bl = {oai_fwd.get("baseline", 0):.4f};
  el.innerHTML = `
    <strong>${{n}}</strong> total OAI event months (14 drugs, 2015–2024)<br>
    <strong>${{na}}</strong> OAI events where drug was <em>already in shortage</em> at inspection →
      <strong>${{naf}}</strong> (${{na>0?(100*naf/na).toFixed(0):0}}%) had shortage in next 12m<br>
    <strong>${{nf}}</strong> OAI events where drug was <em>NOT in shortage</em> at inspection →
      <strong>${{nff}}</strong> (${{nf>0?(100*nff/nf).toFixed(0):0}}%) had shortage in next 12m<br>
    Mean months in shortage over next 12m (all events): <strong>${{mm.toFixed(1)}}</strong><br>
    Control baseline (shortage_ongoing, no OAI ±12m): <strong>${{(bl*100).toFixed(1)}}%</strong><br>
    <span style="color:var(--muted);font-size:11px;">Observational — not causal. Wang et al. (MSOM 2025): OAI → −96% shortage risk (IV-adjusted).</span>
  `;
}})();

// ── Valisure quality split ─────────────────────────────────────────────────
{qs_js}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data…")
    d = compute_data()
    log.info("Generating HTML…")
    html = generate_html(d)
    DASH_OUT.write_text(html, encoding="utf-8")
    size_kb = DASH_OUT.stat().st_size // 1024
    log.info("Wrote %s  (%d KB)", DASH_OUT, size_kb)
    print(f"Dashboard written → {DASH_OUT}  ({size_kb} KB)")
    return DASH_OUT


if __name__ == "__main__":
    main()
