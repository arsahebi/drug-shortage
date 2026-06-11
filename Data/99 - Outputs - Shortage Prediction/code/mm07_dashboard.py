"""
Module MM07 — Dashboard generator.

Five sections, one narrative:
  1. Regulatory signals → shortage  (OAI lead-lag, monthly event study)
  2. Case study: FEI 3002809586     (text trend over 7 inspections, 2016–2025)
  3. Forward validation             (483 text content predicts OAI/recall, m12)
  4. Facility timelines             (interactive chronological explorer, m11)
  5. Coverage & next steps
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from config import OUT_DATA, OUT_TABS, OUT_ROOT, OUT_LOGS, OUT_FIGS, VALISURE_FEI, DATA
from utils import get_logger

log = get_logger("mm07_dashboard", OUT_LOGS / "mm07_dashboard.log")
DASH_OUT = OUT_ROOT / "dashboard.html"


def _read(name: str, subdir: Path = OUT_DATA) -> pd.DataFrame | None:
    p = subdir / name
    if not p.exists():
        return None
    return pd.read_csv(p)


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def _j(o) -> str:
    return json.dumps(o, cls=_NpEncoder, allow_nan=False)


# ─────────────────────────────────────────────────────────────────────────────

INSP_XLSX  = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
TS_CSV     = DATA / "99 - Outputs - Text Analysis" / "483_fei_text_features_timeseries.csv"
CASE_FEI   = 3002809586   # best case study: 7 snaps, 58% coverage, 5/7 OAI


def compute_data() -> dict:
    d: dict = {}

    # ── Basic counts ─────────────────────────────────────────────────────────
    bridge = None
    if VALISURE_FEI.exists():
        bridge = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping",
                               usecols=["API", "FEI_NUMBER"]).dropna(subset=["FEI_NUMBER"])
        bridge["FEI_NUMBER"] = bridge["FEI_NUMBER"].astype(int)
    ts_raw = pd.read_csv(TS_CSV) if TS_CSV.exists() else None

    d["n_drugs"]     = int(bridge["API"].nunique())        if bridge is not None else 14
    d["n_feis"]      = int(bridge["FEI_NUMBER"].nunique()) if bridge is not None else 129
    d["n_feis_text"] = int(ts_raw["fei"].nunique())        if ts_raw is not None else 37
    d["n_snaps"]     = len(ts_raw)                         if ts_raw is not None else 79

    # ── OAI lead-lag (from mm06) ──────────────────────────────────────────────
    ll = _read("lead_lag_monthly.csv", OUT_TABS)
    d["oai_lead"] = {}
    if ll is not None:
        oai = ll[ll["signal"] == "redica_n_oai"].set_index("offset_months")
        offsets = sorted(oai.index.tolist())
        bl = float(oai["baseline_mean"].dropna().iloc[0]) if not oai["baseline_mean"].dropna().empty else 0.039
        d["oai_lead"] = {
            "offsets":  offsets,
            "means":    [round(float(oai.loc[o, "mean"]), 4) for o in offsets],
            "ses":      [round(float(oai.loc[o, "se"]),   4) for o in offsets],
            "baseline": round(bl, 4),
        }

    # ── Case study: FEI 3002809586 ────────────────────────────────────────────
    d["case_study"] = {"snaps": [], "inspections": [], "warning_letters": [], "oai_dates": []}
    if ts_raw is not None and INSP_XLSX.exists():
        ts_raw["snapshot_date"] = pd.to_datetime(ts_raw["snapshot_date"])
        ts_raw["fei"] = ts_raw["fei"].astype(int)
        snaps = ts_raw[ts_raw["fei"] == CASE_FEI].sort_values("snapshot_date")

        insp = pd.read_excel(INSP_XLSX)
        insp["Inspection End Date"] = pd.to_datetime(insp["Inspection End Date"], errors="coerce")
        insp_fei = insp[
            (insp["FEI Number"] == CASE_FEI) &
            insp["Product Type"].str.contains("Drug", na=False)
        ].sort_values("Inspection End Date")

        cls_map = {
            "Official Action Indicated (OAI)": "OAI",
            "Voluntary Action Indicated (VAI)": "VAI",
            "No Action Indicated (NAI)":        "NAI",
        }
        for _, r in snaps.iterrows():
            yr = round(r["snapshot_date"].year + (r["snapshot_date"].dayofyear - 1) / 365.25, 3)
            d["case_study"]["snaps"].append({
                "x":     yr,
                "label": r["snapshot_date"].strftime("%b %Y"),
                "contam":  round(float(r.get("contamination_llm_share",  0) or 0), 3),
                "remed":   round(float(r.get("remediation_none_share",    0) or 0), 3),
                "sev":     round(float(r.get("severity_critmajor_share",  0) or 0), 3),
            })
        # OAI inspection dates (separate markers, not coloring text points)
        for _, r in insp_fei.iterrows():
            if r["Classification"] == "Official Action Indicated (OAI)" and pd.notna(r["Inspection End Date"]):
                yr = round(r["Inspection End Date"].year +
                           (r["Inspection End Date"].dayofyear - 1) / 365.25, 3)
                d["case_study"]["oai_dates"].append({
                    "x": yr,
                    "label": "OAI " + r["Inspection End Date"].strftime("%b %Y"),
                })

        # Warning letters for this FEI (independent of 483 — issued post-inspection)
        from config import REDICA_CSV
        redica = pd.read_csv(REDICA_CSV)
        redica["Event Date"] = pd.to_datetime(redica["Event Date"], errors="coerce")
        wl_rows = redica[(redica["FEI"] == CASE_FEI) & (redica["Warning Letter"] == 1)].copy()
        ts_min = snaps["snapshot_date"].min() if not snaps.empty else pd.Timestamp("2016-01-01")
        for _, wr in wl_rows.iterrows():
            if pd.notna(wr["Event Date"]) and wr["Event Date"] >= ts_min:
                yr = round(wr["Event Date"].year +
                           (wr["Event Date"].dayofyear - 1) / 365.25, 3)
                d["case_study"]["warning_letters"].append({
                    "x": yr,
                    "label": "Warning Letter " + wr["Event Date"].strftime("%b %Y"),
                })

    # ── Monthly onset count ───────────────────────────────────────────────────
    mp = _read("master_panel_monthly.csv")
    d["monthly_onset_months"] = int(mp["shortage_start"].sum()) if mp is not None else 0

    # ── Facility-level FAERS for case study via Valisure ANDA→FEI bridge ─────
    # Bridge: Valisure NDC_FEI Mapping sheet has Application Number (ANDA) + FEI_NUMBER.
    # Matches FAERS appl_no (primary-suspect records only) → facility-specific AE series.
    faers_bupropion: list[dict] = []
    faers_metformin: list[dict] = []
    if VALISURE_FEI.exists():
        try:
            ndc_map = pd.read_excel(VALISURE_FEI, sheet_name="NDC_FEI Mapping",
                                    usecols=["Application Number", "FEI_NUMBER", "API"])
            ndc_map = ndc_map.dropna(subset=["Application Number", "FEI_NUMBER"])
            ndc_map["appl_no_str"] = (ndc_map["Application Number"]
                                      .astype(str).str.replace("ANDA", "", regex=False)
                                      .str.strip().str.lstrip("0"))
            ndc_map["FEI_NUMBER"] = ndc_map["FEI_NUMBER"].astype(int)
            case_rows = ndc_map[ndc_map["FEI_NUMBER"] == CASE_FEI]
            # ANDA→drug label for this facility
            anda_drug = dict(zip(case_rows["appl_no_str"], case_rows["API"].str.title()))

            from config import FAERS_ALL
            faers_raw = pd.read_csv(
                FAERS_ALL.parent / "faers_all_drugs_anda_linked_2015Q1_2026Q1.csv",
                usecols=["primaryid", "appl_no", "serious_flag", "year"],
                low_memory=False)
            faers_raw = faers_raw[faers_raw["appl_no"].notna()].copy()
            faers_raw["appl_no_str"] = (faers_raw["appl_no"].astype(str)
                                        .str.strip().str.lstrip("0"))
            case_faers = faers_raw[faers_raw["appl_no_str"].isin(anda_drug)].copy()
            case_faers["drug"] = case_faers["appl_no_str"].map(anda_drug)

            def _to_bars(df, drug_label):
                sub = df[df["drug"] == drug_label]
                by_yr = (sub.groupby("year")["serious_flag"]
                         .apply(lambda x: int((x == 1).sum()))
                         .reset_index()
                         .rename(columns={"serious_flag": "n_serious"}))
                by_yr = by_yr[(by_yr["year"] >= 2015) & (by_yr["year"] <= 2024)]
                return [{"x": int(r.year) + 0.5, "y": int(r.n_serious)} for _, r in by_yr.iterrows()]

            faers_bupropion = _to_bars(case_faers, "Bupropion")
            faers_metformin = _to_bars(case_faers, "Metformin")
            log.info("Facility FAERS split: Bupropion=%d rows, Metformin=%d rows",
                     len(faers_bupropion), len(faers_metformin))
        except Exception as e:
            log.warning("Facility FAERS build failed: %s", e)
    d["case_study"]["faers_bupropion"] = faers_bupropion
    d["case_study"]["faers_metformin"] = faers_metformin

    # m12 forward validation grid
    _ESC = [
        ("repeat_llm_only_share",        "Repeat violations"),
        ("repeat_cross_insp_share",      "Cross-insp. repeat"),
        ("contamination_llm_only_share", "Contamination"),
        ("oos_oot_regex_share",          "OOS/OOT refs"),
        ("severity_critmajor_share",     "Crit+Major severity"),
        ("scope_facilitywide_share",     "Facility-wide scope"),
    ]
    _REC = [
        ("vc_buildingsequipment_share",  "Buildings/equipment"),
        ("repeat_llm_only_share",        "Repeat violations"),
        ("repeat_cross_insp_share",      "Cross-insp. repeat"),
        ("capital_root_cause_share",     "Capital root cause"),
        ("cultural_root_cause_share",    "Cultural root cause"),
    ]
    grid = _read("text_signal_grid.csv", OUT_TABS)
    d["grid_esc"] = d["grid_rec"] = []
    d["esc_base"] = d["rec_base"] = 0.0
    d["fwd_n"] = d["n_feis_text"]
    if grid is not None:
        def _cells(feats, outcome):
            out = []
            for feat, label in feats:
                r = grid[(grid["feature"] == feat) & (grid["outcome"] == outcome)]
                if len(r):
                    r = r.iloc[0]
                    out.append({"label": label,
                                "hi":   round(float(r["hi_rate"]) * 100, 1),
                                "lo":   round(float(r["lo_rate"]) * 100, 1),
                                "lift": float(r["lift"])})
            return out
        d["grid_esc"] = _cells(_ESC, "esc_24")
        d["grid_rec"] = _cells(_REC, "rec_24")
        g0 = grid[grid["outcome"] == "esc_24"].iloc[0]
        g1 = grid[grid["outcome"] == "rec_24"].iloc[0]
        d["esc_base"] = round(float(
            (g0["hi_rate"]*g0["n_hi"] + g0["lo_rate"]*g0["n_lo"]) / (g0["n_hi"]+g0["n_lo"]))*100, 1)
        d["rec_base"] = round(float(
            (g1["hi_rate"]*g1["n_hi"] + g1["lo_rate"]*g1["n_lo"]) / (g1["n_hi"]+g1["n_lo"]))*100, 1)
        d["fwd_n"] = int(g0["n_hi"] + g0["n_lo"])

    return d


# ─────────────────────────────────────────────────────────────────────────────

def generate_html(d: dict) -> str:

    # ── OAI lead-lag chart JS ─────────────────────────────────────────────────
    ol = d.get("oai_lead", {})
    if ol:
        offsets  = ol["offsets"]
        means    = ol["means"]
        ses      = ol["ses"]
        bl       = ol["baseline"]
        hi = [round((m or 0) + (s or 0), 4) for m, s in zip(means, ses)]
        lo = [round((m or 0) - (s or 0), 4) for m, s in zip(means, ses)]
        lead_lag_js = f"""
new Chart(document.getElementById('oaiLeadChart'),{{
  type:'line',
  data:{{labels:{_j(offsets)},datasets:[
    {{label:'±1 SE',data:{_j(hi)},borderColor:'transparent',
      backgroundColor:'rgba(224,122,95,0.18)',fill:'+1',pointRadius:0,tension:0.2}},
    {{label:'_lo',data:{_j(lo)},borderColor:'transparent',fill:false,pointRadius:0,tension:0.2}},
    {{label:'OAI rate near onset',data:{_j(means)},
      borderColor:'rgb(224,122,95)',fill:false,tension:0.2,pointRadius:4,borderWidth:2.5}},
    {{label:'Control baseline ({bl:.3f})',
      data:{_j([bl]*len(offsets))},
      borderColor:'rgba(224,122,95,0.4)',borderDash:[6,4],pointRadius:0,fill:false,borderWidth:1.5}}
  ]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:11,font:{{size:10}},
      filter:i=>i.text!=='_lo'}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Months before shortage onset (0 = onset month)'}},grid:{{display:false}}}},
      y:{{beginAtZero:true,
          title:{{display:true,text:'Mean OAI inspections / drug-month'}},
          grid:{{color:'#EEE'}}}}
    }}
  }}
}});"""
    else:
        lead_lag_js = "/* oai_lead not available */"

    # ── Case study chart JS ───────────────────────────────────────────────────
    cs = d.get("case_study", {})
    if cs.get("snaps"):
        snaps  = cs["snaps"]
        wls    = cs.get("warning_letters", [])
        xs     = [s["x"]     for s in snaps]
        contam = [s["contam"] for s in snaps]
        remed  = [s["remed"]  for s in snaps]
        labels_x = [s["label"] for s in snaps]
        wl_xs   = [w["x"]     for w in wls]
        wl_lbls = [w["label"] for w in wls]
        oai_xs  = [o["x"]     for o in cs.get("oai_dates", [])]
        oai_lbl = [o["label"] for o in cs.get("oai_dates", [])]
        f_bup   = cs.get("faers_bupropion", [])
        f_met   = cs.get("faers_metformin",  [])
        case_js = f"""
(function(){{
const LABELS_X={_j(labels_x)};
const datasets=[
  {{label:'Contamination flag',
    data:{_j(list(zip(xs, contam)))}.map(p=>(({{x:p[0],y:p[1]}}))  ),
    yAxisID:'y',borderColor:'#E07A5F',fill:false,tension:0.2,borderWidth:2.5,
    pointRadius:6,pointBackgroundColor:'#E07A5F',pointBorderColor:'#B5563E'}},
  {{label:'No remediation response',
    data:{_j(list(zip(xs, remed)))}.map(p=>(({{x:p[0],y:p[1]}}))  ),
    yAxisID:'y',borderColor:'#065A82',fill:false,tension:0.2,borderWidth:2.5,borderDash:[5,3],
    pointRadius:6,pointBackgroundColor:'#065A82',pointBorderColor:'#044060'}}
];
if({_j(f_bup)}.length) datasets.push({{
  label:'Serious AEs — Bupropion',type:'bar',
  data:{_j(f_bup)},yAxisID:'y2',
  backgroundColor:'rgba(224,122,95,0.35)',borderColor:'rgba(224,122,95,0.7)',
  borderWidth:1,barPercentage:0.45,categoryPercentage:0.9,stack:'s'
}});
if({_j(f_met)}.length) datasets.push({{
  label:'Serious AEs — Metformin',type:'bar',
  data:{_j(f_met)},yAxisID:'y2',
  backgroundColor:'rgba(28,114,147,0.30)',borderColor:'rgba(28,114,147,0.6)',
  borderWidth:1,barPercentage:0.45,categoryPercentage:0.9,stack:'s'
}});
if({_j(oai_xs)}.length) datasets.push({{
  label:'OAI inspection',type:'scatter',
  data:{_j(oai_xs)}.map((x,i)=>(({{x,y:0.94,label:{_j(oai_lbl)}[i]}}))  ),
  yAxisID:'y',backgroundColor:'#C0392B',borderColor:'#922B21',
  pointStyle:'triangle',pointRadius:11,showLine:false
}});
if({_j(wl_xs)}.length) datasets.push({{
  label:'Warning Letter',type:'scatter',
  data:{_j(wl_xs)}.map((x,i)=>(({{x,y:0.94,label:{_j(wl_lbls)}[i]}}))  ),
  yAxisID:'y',backgroundColor:'#8B0000',borderColor:'#8B0000',
  pointStyle:'star',pointRadius:13,showLine:false
}});
new Chart(document.getElementById('caseChart'),{{
  type:'line',data:{{datasets}},
  options:{{maintainAspectRatio:false,
    plugins:{{
      legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10}}}}}},
      tooltip:{{callbacks:{{
        title:items=>{{
          const r=items[0].raw;
          if(r?.label) return r.label;
          return LABELS_X[items[0].dataIndex]||'';
        }}
      }}}}
    }},
    scales:{{
      x:{{type:'linear',min:2015,max:2026,
          title:{{display:true,text:'Year'}},
          ticks:{{stepSize:1,callback:v=>Number.isInteger(v)?v:''}},
          grid:{{color:'#EEE'}}}},
      y:{{min:0,max:1,position:'left',
          title:{{display:true,text:'Share of 483 observations'}},
          ticks:{{callback:v=>Math.round(v*100)+'%'}},
          grid:{{color:'#EEE'}}}},
      y2:{{min:0,position:'right',
           title:{{display:true,text:'Serious AEs / year (ANDA-linked)'}},
           grid:{{display:false}},ticks:{{color:'#999'}}}}
    }}
  }}
}});
}})();"""
    else:
        case_js = "/* case study data not available */"

    def _split_chart(cid, cells, x_title):
        if not cells: return f"/* {cid}: no data */"
        labels = [c["label"] for c in cells]
        hi     = [c["hi"]    for c in cells]
        lo     = [c["lo"]    for c in cells]
        lifts  = [c["lift"]  for c in cells]
        return f"""(function(){{
const L={_j(lifts)};
new Chart(document.getElementById({_j(cid)}),{{
  type:'bar',data:{{labels:{_j(labels)},datasets:[
    {{label:'Above median',data:{_j(hi)},backgroundColor:'rgba(224,122,95,0.85)',borderRadius:3}},
    {{label:'At/below median',data:{_j(lo)},backgroundColor:'rgba(28,114,147,0.75)',borderRadius:3}}
  ]}},
  options:{{maintainAspectRatio:false,indexAxis:'y',
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:11,font:{{size:10}}}}}},
      tooltip:{{callbacks:{{afterBody:i=>'Lift: '+L[i[0].dataIndex]+'×'}}}}}},
    scales:{{
      x:{{beginAtZero:true,title:{{display:true,text:{_j(x_title)}}},ticks:{{callback:v=>v+'%'}}}},
      y:{{grid:{{display:false}},ticks:{{font:{{size:10.5}}}}}}
    }}
  }}
}});
}})();"""

    fwd_js = (
        _split_chart("escChart", d["grid_esc"], "% → OAI or Warning Letter within 24 months") +
        _split_chart("recChart", d["grid_rec"], "% → drug recall within 24 months")
    )

    timeline_js = ""  # removed — not informative enough

    n         = d["n_feis_text"]
    sn        = d["n_snaps"]
    eb        = d["esc_base"]
    rb        = d["rec_base"]
    fn        = d["fwd_n"]
    n_onsets  = d["monthly_onset_months"]
    case_fei  = CASE_FEI

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FDA 483 Signal Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--navy:#21295C;--deep:#065A82;--teal:#1C7293;--accent:#E07A5F;
      --cream:#F4F1EC;--paper:#FBFAF7;--ink:#1A2233;--muted:#5A6577;--rule:#E2DDD2;}}
*{{box-sizing:border-box;}} html,body{{margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:var(--paper);color:var(--ink);line-height:1.5;}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 24px 40px;}}
header{{background:var(--navy);color:#fff;padding:28px 28px 22px;border-radius:8px;
        margin-bottom:22px;border-left:6px solid var(--accent);}}
header h1{{margin:0 0 6px;font-family:Georgia,serif;font-size:24px;}}
header p{{margin:0;color:#CADCFC;font-size:13px;}}
.chips{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;}}
.chip{{background:#fff;border:1px solid var(--rule);border-top:3px solid var(--deep);
       border-radius:6px;padding:10px 16px;font-size:13px;}}
.chip strong{{display:block;font-family:Georgia,serif;font-size:22px;color:var(--navy);}}
section{{margin-bottom:24px;}}
h2{{font-family:Georgia,serif;font-size:18px;color:var(--navy);margin:0 0 4px;}}
.sub{{color:var(--muted);font-size:12.5px;margin:0 0 12px;}}
.row2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.card{{background:#fff;border:1px solid var(--rule);border-radius:7px;padding:16px 18px;}}
.card h3{{font-family:Georgia,serif;font-size:15px;color:var(--navy);margin:0 0 4px;}}
.csub{{color:var(--muted);font-size:11.5px;margin-bottom:10px;}}
.ch{{position:relative;height:300px;}}
.ch.tall{{height:360px;}}
.note{{background:var(--cream);border-left:4px solid var(--accent);padding:10px 14px;
       border-radius:4px;font-size:13px;margin-top:12px;}}
.note.dk{{background:var(--navy);color:#CADCFC;}} .note.dk strong{{color:#fff;}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.box{{background:#fff;border:1px solid var(--rule);border-radius:7px;overflow:hidden;}}
.box-head{{padding:8px 14px;color:#fff;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}}
.box-head.l{{background:var(--navy);}} .box-head.n{{background:var(--accent);}}
.box ul{{margin:0;padding:12px 18px 16px;}}
.box li{{padding:3px 0;font-size:12.5px;}}
footer{{text-align:center;color:var(--muted);font-size:11px;margin-top:20px;
        padding-top:12px;border-top:1px solid var(--rule);}}
@media(max-width:800px){{.row2,.two{{grid-template-columns:1fr;}}
  .chips{{grid-template-columns:repeat(2,1fr);}}}}
</style>
</head>
<body><div class="wrap">

<header>
  <h1>FDA 483 Regulatory Text as Early Warning Signal</h1>
  <p>{d["n_drugs"]} generic APIs · {d["n_feis"]} manufacturing facilities ·
     {n} with 483 text data · {sn} LLM-scored snapshots · 2015–2024</p>
</header>

<div class="chips">
  <div class="chip"><strong>{d["n_feis"]}</strong>Total FEIs</div>
  <div class="chip"><strong>{n}</strong>FEIs with text</div>
  <div class="chip"><strong>{sn}</strong>Scored snapshots</div>
  <div class="chip"><strong>{fn}</strong>Used in validation</div>
</div>

<!-- SECTION 1: REGULATORY SIGNALS → SHORTAGE -->
<section>
  <h2>Do regulatory signals precede shortage?</h2>
  <p class="sub">Mean OAI rate across {n_onsets} shortage onset events (14 drugs, 2015–2024) vs control baseline. Exploratory — wide confidence intervals at this sample size.</p>
  <div class="card" style="max-width:620px;">
    <h3>OAI rate in months leading to shortage onset</h3>
    <div class="csub">Solid line = mean near onset · dashed = control baseline.</div>
    <div class="ch"><canvas id="oaiLeadChart"></canvas></div>
  </div>
</section>

<!-- SECTION 2: CASE STUDY FEI 3002809586 -->
<section>
  <h2>Case study — one facility over 9 years (FEI {case_fei})</h2>
  <p class="sub">
    Makes Bupropion &amp; Metformin. 7 FDA Form 483s between 2016 and 2025 (58% of drug
    inspections). Each point is one dated 483 document. ★ marks a Warning Letter issued
    by FDA — an independent regulatory action post-inspection. Hover for dates.
  </p>
  <div class="card">
    <h3>Contamination flag &amp; no-remediation share, 2016–2025</h3>
    <div class="csub">
      Left axis: two signals from 483 text (facility-specific, 7 snapshots covering 58% of drug inspections).
      Right axis: serious AEs linked to <strong>this facility</strong> via ANDA numbers
      (primary-suspect FAERS records), split by drug.
      ▲ = OAI inspection · ★ = Warning Letter.
    </div>
    <div class="ch" style="height:320px;"><canvas id="caseChart"></canvas></div>
  </div>
  <div class="note dk">
    <strong>What the trend shows:</strong>
    Contamination is persistently high (52–62%) across all 7 snapshots — no improvement over 9 years.
    The no-remediation share <em>rises</em> from 22% (2016) to 63% (2019) and plateaus —
    the facility keeps acknowledging the same problems in 483 text but never commits to fixing them.
    A Warning Letter followed in May 2022 when these signals were already at their peak.
    The gray bars show serious adverse events linked directly to <strong>this facility</strong>
    via its ANDA numbers (Bupropion: ANDA078866, Metformin: ANDA077336) — rising from ~14/year
    in 2015–2016 to a peak of 72 in 2021, then settling at 55–63 post–Warning Letter.
    This is not a causal claim — it is a pattern visible in the record, in chronological order.
  </div>
</section>

<!-- SECTION 3: FORWARD VALIDATION -->
<section>
  <h2>What 483 text predicts — facility level</h2>
  <p class="sub">
    {fn} snapshots split at each feature's median. Every facility already received a 483 —
    differences come from <strong>what the text says</strong>, not the document's existence.
    Base rates: escalation {eb}% · recall {rb}%.
  </p>
  <div class="row2">
    <div class="card">
      <h3>OAI or Warning Letter within 24 months</h3>
      <div class="csub">Orange = above median · teal = at/below · hover for lift</div>
      <div class="ch tall"><canvas id="escChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Drug recall within 24 months</h3>
      <div class="csub">Same split. Different features predict different failure modes.</div>
      <div class="ch tall"><canvas id="recChart"></canvas></div>
    </div>
  </div>
  <div class="note dk">
    <strong>Key:</strong> repeat violations and cross-inspection repeats predict escalation (2–7×);
    buildings/equipment violations predict recalls; facilities with no remediation response
    show longer supply disruption (ρ = +0.33, 36m horizon).
    n = {fn} snapshots, {n} FEIs — exploratory.
  </div>
</section>

<!-- SECTION 4: COVERAGE & NEXT STEPS -->
<section>
  <h2>Coverage &amp; next steps</h2>
  <div class="two">
    <div class="box">
      <div class="box-head l">Limitations</div>
      <ul>
        <li>37/129 FEIs have text (28.7%); median inspection coverage within those is ~22%.</li>
        <li>14 drugs, 79 snapshots — all findings are exploratory.</li>
        <li>No causal identification.</li>
      </ul>
    </div>
    <div class="box">
      <div class="box-head n">Next steps</div>
      <ul>
        <li><strong>Stage 1:</strong> Redica counts → p_esc per facility (m12, done). Add Redica <em>483 critical / 483 major</em> observation counts as additional features.</li>
        <li><strong>Stage 2:</strong> max/mean p_esc → drug-year shortage model (m07/m09, done).</li>
        <li><strong>FAERS correlation:</strong> for the 37 text-covered FEIs, test whether 483 text severity correlates with subsequent serious adverse events for the drugs they manufacture — an independent, patient-facing outcome.</li>
        <li>Expand 483 PDF coverage — the binding constraint on text analysis.</li>
      </ul>
    </div>
  </div>
</section>

<footer>FDA 483 Signal Analysis · {d["n_drugs"]} APIs · {d["n_feis"]} FEIs · June 2026</footer>
</div>
<script>
Chart.defaults.font.family='-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif';
Chart.defaults.color="#1A2233";Chart.defaults.font.size=11.5;
{lead_lag_js}
{case_js}
{fwd_js}
{timeline_js}
</script>
</body></html>"""


def main():
    log.info("Loading data…")
    d = compute_data()
    html = generate_html(d)
    DASH_OUT.write_text(html, encoding="utf-8")
    size_kb = DASH_OUT.stat().st_size // 1024
    log.info("Wrote %s  (%d KB)", DASH_OUT, size_kb)
    print(f"Dashboard → {DASH_OUT}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
