"""
Module MM07 — Dashboard generator.

Sections:
  1. Data at a glance
  2. Data sources
  3. Forward validation — 483 text content → escalation / recall (facility level)
  4. Monthly lead-lag — regulatory + text signals before shortage onset
  5. Interactive FEI timeline explorer
  6. Facility summary table
  7. Limitations & next steps
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from config import OUT_DATA, OUT_TABS, OUT_ROOT, OUT_LOGS, OUT_FIGS, TEXT_TIMESERIES_CSV, VALISURE_FEI
from utils import get_logger

log = get_logger("mm07_dashboard", OUT_LOGS / "mm07_dashboard.log")

DASH_OUT = OUT_ROOT / "dashboard.html"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(name: str, subdir: Path = OUT_DATA) -> pd.DataFrame | None:
    p = subdir / name
    if not p.exists():
        log.warning("Missing (section will show placeholder): %s", p.name)
        return None
    return pd.read_csv(p)


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def _j(obj) -> str:
    return json.dumps(obj, cls=_NpEncoder, allow_nan=False)


# ─────────────────────────────────────────────────────────────────────────────
# Data computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_data() -> dict:
    d: dict = {}

    # ── Basic counts ──────────────────────────────────────────────────────────
    bridge = None
    if VALISURE_FEI.exists():
        bridge = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping",
                               usecols=["API", "FEI_NUMBER"])
        bridge = bridge.dropna(subset=["FEI_NUMBER"])
        bridge["FEI_NUMBER"] = bridge["FEI_NUMBER"].astype(int)

    ts_raw = None
    if TEXT_TIMESERIES_CSV.exists():
        ts_raw = pd.read_csv(TEXT_TIMESERIES_CSV)

    d["n_drugs"]      = int(bridge["API"].nunique())       if bridge is not None else 14
    d["n_feis"]       = int(bridge["FEI_NUMBER"].nunique()) if bridge is not None else 129
    d["n_feis_text"]  = int(ts_raw["fei"].nunique())        if ts_raw  is not None else 37
    d["n_snapshots"]  = len(ts_raw)                         if ts_raw  is not None else 79

    # ── Monthly lead-lag ──────────────────────────────────────────────────────
    ll = _read("lead_lag_monthly.csv", OUT_TABS)
    d["monthly_lead"] = {}
    if ll is not None:
        offsets = [int(x) for x in sorted(ll["offset_months"].unique())]
        for sig, g in ll.groupby("signal"):
            g = g.set_index("offset_months").reindex(offsets)
            d["monthly_lead"][sig] = {
                "offsets":  offsets,
                "means":    [round(float(v), 4) if pd.notna(v) else 0.0 for v in g["mean"]],
                "ses":      [round(float(v), 4) if pd.notna(v) else 0.0 for v in g["se"]],
                "baseline": float(g["baseline_mean"].dropna().iloc[0])
                             if not g["baseline_mean"].dropna().empty else 0.0,
            }

    mp = _read("master_panel_monthly.csv")
    d["monthly_onset_months"] = int(mp["shortage_start"].sum()) if mp is not None else 0

    # VAI lead-lag (not in lead_lag_monthly.csv — compute from monthly panel)
    if mp is not None and "redica_n_vai" in mp.columns:
        mp2 = mp.copy()
        mp2["midx"] = mp2["year"] * 12 + mp2["month"]
        onsets = mp2[mp2["shortage_start"] == 1][["drug_norm", "midx"]].values.tolist()
        _off = list(range(-12, 1))
        if onsets:
            recs = []
            for drug, omidx in onsets:
                di = mp2[mp2["drug_norm"] == drug].set_index("midx")["redica_n_vai"]
                for off in _off:
                    t = omidx + off
                    if t in di.index and pd.notna(di[t]):
                        recs.append({"offset": off, "val": float(di[t])})
            tagged = {(drug, omidx + k) for drug, omidx in onsets for k in range(-12, 13)}
            bl_rows = mp2[~mp2.apply(lambda r: (r["drug_norm"], r["midx"]) in tagged, axis=1)]
            bl = float(bl_rows["redica_n_vai"].dropna().mean()) if len(bl_rows) else 0.0
            if recs:
                dfv = pd.DataFrame(recs).groupby("offset")["val"]
                d["monthly_lead"]["redica_n_vai"] = {
                    "offsets":  _off,
                    "means":    [round(float(dfv.mean().get(o, 0)), 4) for o in _off],
                    "ses":      [round(float(dfv.sem().get(o, 0)), 4) for o in _off],
                    "baseline": round(bl, 4),
                }

    # Time-varying text signals lead-lag
    hr = _read("text_highrisk_483_monthly.csv")
    if hr is not None and mp is not None and "n_repeat_483_last_24mo" in hr.columns:
        mp3 = mp.merge(hr, on=["drug_norm", "year", "month"], how="left")
        mp3["n_repeat_483_last_24mo"] = mp3["n_repeat_483_last_24mo"].fillna(0)
        mp3["midx"] = mp3["year"] * 12 + mp3["month"]
        onsets = mp3[mp3["shortage_start"] == 1][["drug_norm", "midx"]].values.tolist()
        _off = list(range(-12, 1))
        recs = []
        for drug, om in onsets:
            di = mp3[mp3["drug_norm"] == drug].set_index("midx")["n_repeat_483_last_24mo"]
            for off in _off:
                t = om + off
                if t in di.index and pd.notna(di[t]):
                    recs.append({"offset": off, "val": float(di[t])})
        tagged = {(dr, om + k) for dr, om in onsets for k in range(-12, 13)}
        bl_rows = mp3[~mp3.apply(lambda r: (r["drug_norm"], r["midx"]) in tagged, axis=1)]
        bl = float(bl_rows["n_repeat_483_last_24mo"].mean()) if len(bl_rows) else 0.0
        if recs:
            dfh = pd.DataFrame(recs).groupby("offset")["val"]
            d["monthly_lead"]["n_repeat_483_last_24mo"] = {
                "offsets":  _off,
                "means":    [round(float(dfh.mean().get(o, 0)), 4) for o in _off],
                "ses":      [round(float(dfh.sem().get(o, 0)), 4) for o in _off],
                "baseline": round(bl, 4),
            }

    # ── m12: text-signal forward validation grid ──────────────────────────────
    _ESC_FEATS = [
        ("repeat_llm_only_share",        "Repeat violations (LLM)"),
        ("repeat_cross_insp_share",      "Cross-insp. repeat (algo)"),
        ("contamination_llm_only_share", "Contamination (LLM)"),
        ("oos_oot_regex_share",          "OOS/OOT references"),
        ("severity_critmajor_share",     "Critical+Major severity"),
        ("scope_facilitywide_share",     "Facility-wide scope"),
    ]
    _REC_FEATS = [
        ("vc_buildingsequipment_share",  "Buildings/equipment violations"),
        ("repeat_llm_only_share",        "Repeat violations (LLM)"),
        ("repeat_cross_insp_share",      "Cross-insp. repeat (algo)"),
        ("capital_root_cause_share",     "Capital root cause"),
        ("cultural_root_cause_share",    "Cultural root cause"),
    ]
    grid = _read("text_signal_grid.csv", OUT_TABS)
    d["grid_esc"], d["grid_rec"], d["grid_extra"] = [], [], {}
    if grid is not None:
        def _cells(feats, outcome):
            out = []
            for feat, label in feats:
                row = grid[(grid["feature"] == feat) & (grid["outcome"] == outcome)]
                if len(row):
                    r = row.iloc[0]
                    out.append({"label": label,
                                "hi":    round(float(r["hi_rate"]) * 100, 1),
                                "lo":    round(float(r["lo_rate"]) * 100, 1),
                                "lift":  float(r["lift"]),
                                "n_hi":  int(r["n_hi"]), "n_lo": int(r["n_lo"])})
            return out

        d["grid_esc"] = _cells(_ESC_FEATS, "esc_24")
        d["grid_rec"] = _cells(_REC_FEATS, "rec_24")

        def _cell1(feat, outcome, col="effect"):
            row = grid[(grid["feature"] == feat) & (grid["outcome"] == outcome)]
            return float(row.iloc[0][col]) if len(row) else None

        g0 = grid[grid["outcome"] == "esc_24"].iloc[0]
        g1 = grid[grid["outcome"] == "rec_24"].iloc[0]
        d["grid_extra"] = {
            "sev_esc12_lift":     _cell1("severity_critmajor_share",  "esc_12", "lift"),
            "remed_none_shdur":   _cell1("remediation_none_share",    "sh_dur_36"),
            "cross_repeat_esc24": _cell1("repeat_cross_insp_share",   "esc_24", "lift"),
            "scope_fw_esc24":     _cell1("scope_facilitywide_share",  "esc_24", "lift"),
            "esc24_base": round(float(
                (g0["hi_rate"] * g0["n_hi"] + g0["lo_rate"] * g0["n_lo"])
                / (g0["n_hi"] + g0["n_lo"])) * 100, 1),
            "rec24_base": round(float(
                (g1["hi_rate"] * g1["n_hi"] + g1["lo_rate"] * g1["n_lo"])
                / (g1["n_hi"] + g1["n_lo"])) * 100, 1),
            "n_snapshots": int(g0["n_hi"] + g0["n_lo"]),
        }

    # ── FEI timeline data ─────────────────────────────────────────────────────
    fs = _read("fei_timeline_summary.csv", OUT_FIGS / "timelines")
    if fs is not None:
        fs["firm_name"] = fs["firm_name"].fillna("")
        d["fei_summary"] = fs.replace({np.nan: None}).to_dict(orient="records")
    else:
        d["fei_summary"] = []

    ev_all = _read("fei_events_all.csv")
    d["fei_events"] = {}
    if ev_all is not None:
        ev_all["event_date"] = pd.to_datetime(ev_all["event_date"], errors="coerce")
        ev_all["year_dec"] = (ev_all["event_date"].dt.year +
                              (ev_all["event_date"].dt.dayofyear - 1) / 365.25).round(3)
        for fei_id, grp in ev_all.groupby("fei"):
            firm_nm = grp["firm_name"].dropna().iloc[0] if grp["firm_name"].notna().any() else str(fei_id)
            apis_v  = grp["apis"].dropna().iloc[0] if "apis" in grp.columns and grp["apis"].notna().any() else ""
            ev483, evinsp, evrec, shbands = [], [], [], []
            for _, r in grp.sort_values("event_date").iterrows():
                if pd.isna(r.get("year_dec")):
                    continue
                yr    = float(r["year_dec"])
                etype = str(r["event_type"])
                lbl   = str(r.get("event_label", ""))[:70]
                if etype == "483_snapshot":
                    ev483.append({
                        "yr": yr,
                        "hr": bool(r["high_risk_483"]) if pd.notna(r.get("high_risk_483")) else False,
                        "n_obs": int(r["n_obs_total"]) if pd.notna(r.get("n_obs_total")) else 1,
                        "sev":  round(float(r["severity_critmajor_share"]), 2)
                                if pd.notna(r.get("severity_critmajor_share")) else 0.0,
                        "label": lbl,
                    })
                elif etype == "inspection_outcome":
                    evinsp.append({"yr": yr,
                                   "cls": str(r["classification"]) if pd.notna(r.get("classification")) else "?",
                                   "label": lbl})
                elif etype == "recall":
                    evrec.append({"yr": yr,
                                  "cls": str(r["recall_class"]) if pd.notna(r.get("recall_class")) else "?",
                                  "label": lbl})
                elif etype == "shortage_start":
                    drug   = str(r.get("shortage_drug", ""))[:40]
                    end_yr = None
                    resolved = r.get("shortage_resolved_date")
                    if pd.notna(resolved) and str(resolved) not in ("nan", "NaT", ""):
                        try:
                            rd = pd.to_datetime(resolved, errors="coerce")
                            if pd.notna(rd):
                                end_yr = round(float(rd.year + (rd.dayofyear - 1) / 365.25), 3)
                        except Exception:
                            pass
                    shbands.append({"start": yr, "end": end_yr or 2025.5, "drug": drug})
            d["fei_events"][str(fei_id)] = {
                "firm": str(firm_nm)[:55], "apis": str(apis_v)[:60],
                "events_483": ev483, "events_insp": evinsp,
                "events_recall": evrec, "shortage_bands": shbands,
                "n_483":    len(ev483),
                "n_hr":     sum(1 for e in ev483 if e["hr"]),
                "n_oai":    sum(1 for e in evinsp if e["cls"] == "OAI"),
                "n_recalls": len(evrec),
            }
        log.info("Timeline events loaded: %d FEIs", len(d["fei_events"]))

    return d


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_html(d: dict) -> str:
    ml = d.get("monthly_lead", {})

    SIGNAL_LABELS = {
        "redica_n_oai":            "OAI Inspections",
        "redica_n_vai":            "VAI Inspections",
        "redica_n_inspections":    "Total Inspections",
        "n_repeat_483_last_24mo":  "Repeat-violation 483s (trailing 24m)",
        "n_flagged_483_last_24mo": "Red-flagged 483s ≥2 risk markers (trailing 24m)",
    }

    def _ll_chart_js(canvas_id: str, sig: str, color: str) -> str:
        if sig not in ml:
            return f"/* {sig} not in data */"
        info    = ml[sig]
        offsets = info["offsets"]
        means   = info["means"]
        ses     = info["ses"]
        bl      = info["baseline"]
        label   = SIGNAL_LABELS.get(sig, sig)
        hi = [round((m or 0) + (s or 0), 5) for m, s in zip(means, ses)]
        lo = [round((m or 0) - (s or 0), 5) for m, s in zip(means, ses)]
        return f"""
new Chart(document.getElementById({_j(canvas_id)}),{{
  type:'line',
  data:{{labels:{_j(offsets)},datasets:[
    {{label:'\\u00b11 SE',data:{_j(hi)},borderColor:'transparent',
      backgroundColor:'rgba({color},0.15)',fill:'+1',pointRadius:0,tension:0.2}},
    {{label:'_lo',data:{_j(lo)},borderColor:'transparent',fill:false,pointRadius:0,tension:0.2}},
    {{label:{_j(label)},data:{_j(means)},borderColor:'rgb({color})',fill:false,
      tension:0.2,pointRadius:3,borderWidth:2}},
    {{label:'Baseline',data:{_j([round(bl,5)]*len(offsets))},
      borderColor:'rgba({color},0.45)',borderDash:[5,4],pointRadius:0,fill:false,borderWidth:1.5}}
  ]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{title:{{display:true,text:'Months to shortage onset (0 = onset)'}},grid:{{display:false}}}},
      y:{{beginAtZero:true,grid:{{color:'#EEE'}}}}
    }}
  }}
}});"""

    redica_js = (
        _ll_chart_js("llR1", "redica_n_oai",         "224, 122, 95") +
        _ll_chart_js("llR2", "redica_n_vai",          "28, 114, 147") +
        _ll_chart_js("llR3", "redica_n_inspections",  "80, 140, 60")
    )
    text_ll_js = (
        _ll_chart_js("llT1", "n_repeat_483_last_24mo",   "180, 80, 180") +
        _ll_chart_js("llT2", "n_flagged_483_last_24mo",  "224, 122, 95")
    )

    # ── Forward validation charts (m12 grid) ──────────────────────────────────
    gx        = d.get("grid_extra", {})
    fwd_n     = int(gx.get("n_snapshots") or 0)
    esc_base  = gx.get("esc24_base", 0)
    rec_base  = gx.get("rec24_base", 0)

    def _split_chart_js(canvas_id: str, cells: list, x_title: str) -> str:
        if not cells:
            return f"/* {canvas_id}: no data */"
        labels = [c["label"] for c in cells]
        hi     = [c["hi"]    for c in cells]
        lo     = [c["lo"]    for c in cells]
        lifts  = [c["lift"]  for c in cells]
        return f"""
(function(){{
const LIFTS={_j(lifts)};
new Chart(document.getElementById({_j(canvas_id)}),{{
  type:'bar',
  data:{{labels:{_j(labels)},datasets:[
    {{label:'Above median',data:{_j(hi)},backgroundColor:'rgba(224,122,95,0.85)',borderRadius:3}},
    {{label:'At/below median',data:{_j(lo)},backgroundColor:'rgba(28,114,147,0.75)',borderRadius:3}}
  ]}},
  options:{{maintainAspectRatio:false,indexAxis:'y',
    plugins:{{
      legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}},
      tooltip:{{callbacks:{{afterBody:items=>'Lift: '+LIFTS[items[0].dataIndex]+'×'}}}}
    }},
    scales:{{
      x:{{beginAtZero:true,title:{{display:true,text:{_j(x_title)}}},ticks:{{callback:v=>v+'%'}}}},
      y:{{grid:{{display:false}},ticks:{{font:{{size:10.5}}}}}}
    }}
  }}
}});
}})();"""

    fwd_js = (
        _split_chart_js("escChart", d.get("grid_esc", []),
                        "% followed by OAI or Warning Letter within 24 months") +
        _split_chart_js("recChart", d.get("grid_rec", []),
                        "% followed by a drug recall within 24 months")
    )

    # Narrative bullets from grid values
    def _fmt_cell(cells, label):
        for c in cells:
            if c["label"] == label:
                return f"{c['hi']}% vs {c['lo']}% ({c['lift']}×)"
        return "—"

    esc_repeat  = _fmt_cell(d.get("grid_esc", []), "Repeat violations (LLM)")
    esc_contam  = _fmt_cell(d.get("grid_esc", []), "Contamination (LLM)")
    esc_cross   = _fmt_cell(d.get("grid_esc", []), "Cross-insp. repeat (algo)")
    esc_scope   = _fmt_cell(d.get("grid_esc", []), "Facility-wide scope")
    rec_bldg    = _fmt_cell(d.get("grid_rec", []), "Buildings/equipment violations")
    sev12_lift  = gx.get("sev_esc12_lift")
    sev12_txt   = f"{sev12_lift}×" if sev12_lift else "—"
    remed_rho   = gx.get("remed_none_shdur")
    remed_txt   = f"ρ = +{remed_rho}" if remed_rho else "—"

    # ── FEI summary table ─────────────────────────────────────────────────────
    fei_sum = d.get("fei_summary", [])
    cov = [r for r in fei_sum if (r.get("n_483_snapshots") or 0) > 0]
    cov.sort(key=lambda r: ((r.get("n_high_risk_483_snapshots") or 0),
                            (r.get("n_oai_inspections") or 0)), reverse=True)
    n_cov_feis = len(cov)
    fei_rows = []
    for r in cov:
        hr_n     = r.get("n_high_risk_483_snapshots") or 0
        hr_style = ' style="color:#C0392B;font-weight:700;"' if hr_n > 0 else ""
        fei_rows.append(
            f"<tr>"
            f"<td>{r['fei']}</td>"
            f"<td>{r.get('firm_name','')}</td>"
            f"<td>{r.get('apis_made','')}</td>"
            f"<td class='num'>{r.get('n_483_snapshots', 0)}</td>"
            f"<td class='num'{hr_style}>{hr_n}</td>"
            f"<td class='num'>{r.get('n_oai_inspections', 0)}</td>"
            f"<td class='num'>{r.get('n_vai_inspections', 0)}</td>"
            f"<td class='num'>{r.get('n_drug_recalls', 0)}</td>"
            f"</tr>"
        )
    fei_table_rows = "\n".join(fei_rows)

    # ── Interactive FEI timeline JS ───────────────────────────────────────────
    fei_events_data = d.get("fei_events", {})
    timeline_js = f"""
const FEI_EVENTS = {_j(fei_events_data)};
let _tChart = null;
(function(){{
  const sel = document.getElementById('feiSelector');
  if (!sel || !Object.keys(FEI_EVENTS).length) {{
    if (sel) {{ sel.innerHTML='<option>No data — run m11 first</option>'; sel.disabled=true; }}
    return;
  }}
  const feiList = Object.keys(FEI_EVENTS).sort((a,b)=>{{
    const da=FEI_EVENTS[a],db=FEI_EVENTS[b];
    return (db.n_hr+db.n_oai*2+db.n_recalls)-(da.n_hr+da.n_oai*2+da.n_recalls);
  }});
  feiList.forEach(fei=>{{
    const ev=FEI_EVENTS[fei];
    const opt=document.createElement('option');
    opt.value=fei;
    opt.textContent=ev.firm+' (FEI '+fei+')'+(ev.apis?' · '+ev.apis:'');
    sel.appendChild(opt);
  }});
  function buildTimeline(fei){{
    const ev=FEI_EVENTS[fei]; if(!ev) return;
    const stats=document.getElementById('feiStats');
    if(stats) stats.innerHTML=
      '<strong>'+ev.n_483+'</strong> 483s (<span style="color:#C0392B;font-weight:700;">'
      +ev.n_hr+' high-risk</span>) · <strong>'+ev.n_oai+'</strong> OAI · <strong>'
      +ev.n_recalls+'</strong> recalls';
    const datasets=[];
    const ns=ev.shortage_bands.length;
    ev.shortage_bands.forEach((s,i)=>{{
      const y=0.07*(i-(ns-1)/2);
      datasets.push({{label:i===0?'Shortage period':'_hide_',type:'line',
        data:[{{x:s.start,y:y,label:s.drug+' — shortage start'}},
              {{x:s.end,  y:y,label:s.drug+' — resolved'}}],
        borderColor:'rgba(100,100,100,0.5)',borderWidth:10,pointRadius:2,fill:false,order:0}});
    }});
    const hr=ev.events_483.filter(e=>e.hr);
    if(hr.length) datasets.push({{
      label:'483 Red-flagged',type:'scatter',
      data:hr.map(e=>({{x:e.yr,y:2,label:e.label,n_obs:e.n_obs,sev:e.sev}})),
      backgroundColor:'rgba(192,57,43,0.88)',borderColor:'#922B21',borderWidth:1,
      pointRadius:ctx=>Math.max(6,Math.min(18,(ctx.raw?.n_obs||1)*1.8)),
      pointHoverRadius:14,order:2}});
    const ok=ev.events_483.filter(e=>!e.hr);
    if(ok.length) datasets.push({{
      label:'483 Normal',type:'scatter',
      data:ok.map(e=>({{x:e.yr,y:2,label:e.label,n_obs:e.n_obs,sev:e.sev}})),
      backgroundColor:'rgba(28,114,147,0.7)',borderColor:'#065A82',borderWidth:1,
      pointRadius:ctx=>Math.max(5,Math.min(14,(ctx.raw?.n_obs||1)*1.5)),
      pointHoverRadius:10,order:2}});
    const clsC={{OAI:'rgba(192,57,43,0.9)',VAI:'rgba(230,126,34,0.9)',NAI:'rgba(39,174,96,0.9)'}};
    ['OAI','VAI','NAI'].forEach(cls=>{{
      const evs=ev.events_insp.filter(e=>e.cls===cls); if(!evs.length) return;
      datasets.push({{label:'Inspection: '+cls,type:'scatter',
        data:evs.map(e=>({{x:e.yr,y:3,label:e.label}})),
        backgroundColor:clsC[cls],borderColor:clsC[cls],
        pointStyle:'triangle',pointRadius:9,pointHoverRadius:12,order:1}});
    }});
    const rclsC={{'Class I':'rgba(139,0,0,0.9)','Class II':'rgba(204,85,0,0.9)',
                  'Class III':'rgba(184,134,11,0.9)'}};
    Object.entries(rclsC).forEach(([cls,color])=>{{
      const evs=ev.events_recall.filter(e=>e.cls===cls); if(!evs.length) return;
      datasets.push({{label:'Recall: '+cls,type:'scatter',
        data:evs.map(e=>({{x:e.yr,y:1,label:e.label}})),
        backgroundColor:color,borderColor:color,
        pointStyle:'rectRot',pointRadius:9,pointHoverRadius:12,order:1}});
    }});
    if(_tChart) _tChart.destroy();
    const ctx=document.getElementById('timelineChart'); if(!ctx) return;
    _tChart=new Chart(ctx,{{
      data:{{datasets}},
      options:{{maintainAspectRatio:false,
        scales:{{
          x:{{type:'linear',min:2008,max:2026,
              title:{{display:true,text:'Year'}},
              ticks:{{stepSize:1,callback:v=>Number.isInteger(v)?v:''}},
              grid:{{color:'#EEE'}}}},
          y:{{min:-0.7,max:3.7,
              ticks:{{stepSize:1,callback:v=>{{
                const m={{0:'Shortages',1:'Recalls',2:'483 Snapshots',3:'Inspections'}};
                return m[Math.round(v)]||'';
              }}}},grid:{{color:'#EEE'}}}}
        }},
        plugins:{{
          legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}},
            filter:i=>!i.text.startsWith('_')}}}},
          tooltip:{{callbacks:{{label:ctx=>{{
            const r=ctx.raw;
            if(r.label) return r.label;
            if(r.n_obs!==undefined) return r.n_obs+' obs, sev='+r.sev;
            return '';
          }}}}}}
        }}
      }}
    }});
    const story=[];
    if(ev.n_hr>0){{
      const pct=Math.round(ev.n_hr/ev.n_483*100);
      story.push('<strong>'+ev.n_hr+'/'+ev.n_483+' ('+pct+'%)</strong> of 483 snapshots are red-flagged.');
    }}
    if(ev.n_oai>0) story.push('<strong>'+ev.n_oai+' OAI</strong> inspection outcome(s).');
    if(ev.n_recalls>0) story.push('<strong>'+ev.n_recalls+'</strong> recall event(s).');
    if(ev.n_hr>0&&ev.n_oai>0)
      story.push('Look for red circles (●) preceding OAI triangles (▲).');
    const stEl=document.getElementById('feiStory');
    if(stEl) stEl.innerHTML=story.length?story.join(' '):'No high-risk events for this facility.';
  }}
  buildTimeline(feiList[0]);
  sel.addEventListener('change',()=>buildTimeline(sel.value));
}})();"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FDA 483 Signal Analysis · Dashboard</title>
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
.step-num{{display:inline-block;background:var(--accent);color:#fff;width:28px;height:28px;
           border-radius:50%;text-align:center;line-height:28px;font-weight:700;
           font-size:13px;flex-shrink:0;}}
h2{{margin:0;font-family:Georgia,serif;font-size:20px;color:var(--navy);font-weight:700;}}
h3{{margin:0 0 4px;font-family:Georgia,serif;font-size:16px;color:var(--navy);}}
.sub{{color:var(--muted);font-size:13px;margin:4px 0 14px 42px;}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
.stat{{background:var(--white);border:1px solid var(--rule);border-top:4px solid var(--deep);
       border-radius:6px;padding:16px 18px;}}
.stat .num{{font-family:Georgia,serif;font-size:30px;font-weight:700;color:var(--navy);line-height:1.1;}}
.stat .num small{{font-family:sans-serif;color:var(--muted);font-size:14px;font-weight:400;margin-left:4px;}}
.stat .lbl{{color:var(--ink);font-size:12.5px;margin-top:6px;}}
.stat.accent{{border-top-color:var(--accent);}} .stat.teal{{border-top-color:var(--teal);}}
.sources{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}}
.src{{background:var(--white);border:1px solid var(--rule);border-left:4px solid var(--deep);
      border-radius:6px;padding:12px 14px;}}
.src .name{{font-family:Georgia,serif;font-weight:700;color:var(--navy);font-size:14px;}}
.src .role{{color:var(--muted);font-size:12px;margin-top:4px;}}
.card{{background:var(--white);border:1px solid var(--rule);border-radius:8px;padding:18px 20px;}}
.card .csub{{color:var(--muted);font-size:12px;margin-bottom:12px;}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.chart-row.three{{grid-template-columns:1fr 1fr 1fr;}}
.chart-host{{position:relative;height:280px;}}
.chart-host.tall{{height:360px;}}
.note{{background:var(--cream);border-left:4px solid var(--accent);padding:12px 14px;
       border-radius:4px;font-size:13px;color:var(--ink);margin-top:14px;}}
.note strong{{color:var(--navy);}}
.note.dark{{background:var(--navy);border-left-color:var(--accent);color:#CADCFC;}}
.note.dark strong{{color:var(--white);}}
.divider{{border:none;border-top:2px solid var(--accent);margin:8px 0 20px;opacity:0.3;}}
table.signals{{width:100%;border-collapse:collapse;font-size:13px;}}
table.signals th,table.signals td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--rule);}}
table.signals th{{background:var(--cream);color:var(--navy);font-weight:700;
                  font-size:11px;text-transform:uppercase;letter-spacing:1px;}}
table.signals td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.col-card{{background:var(--white);border:1px solid var(--rule);border-radius:8px;overflow:hidden;}}
.col-card .col-head{{padding:10px 16px;color:white;font-size:12px;font-weight:700;
                     letter-spacing:3px;text-transform:uppercase;}}
.col-card .col-head.l{{background:var(--navy);}} .col-card .col-head.n{{background:var(--accent);}}
.col-card ul{{margin:0;padding:14px 20px 18px;}}
.col-card li{{padding:4px 0;font-size:13px;color:var(--ink);}}
footer{{text-align:center;color:var(--muted);font-size:11px;margin-top:26px;
        padding-top:14px;border-top:1px solid var(--rule);}}
@media(max-width:900px){{
  .stats,.sources{{grid-template-columns:repeat(2,1fr);}}
  .chart-row,.chart-row.three,.two-col{{grid-template-columns:1fr;}}
}}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <div class="eyebrow">Research Dashboard · June 2026</div>
  <h1>FDA 483 Regulatory Text as Early Warning Signal</h1>
  <p>{d["n_drugs"]} generic APIs · {d["n_feis"]} manufacturing facilities ·
     {d["n_feis_text"]} with text data · 2015–2024</p>
</header>

<section>
  <div class="note dark" style="margin-top:0;font-size:13.5px;line-height:1.9;">
    <strong>Three findings:</strong><br>
    <strong>1.</strong> What the 483 text says predicts what happens next at the facility:
    <strong>repeat violations, cross-inspection repeats, contamination, facility-wide scope</strong>
    → 2–7× higher rate of OAI or Warning Letter within 24 months (Section 3).<br>
    <strong>2.</strong> 483s with <strong>no remediation response</strong> in the text are followed by
    more months of supply disruption over 3 years ({remed_txt}) — the quality of the response matters,
    not just the violation count.<br>
    <strong>3.</strong> Text coverage is partial (37/129 FEIs, ~22% of inspections).
    The next step is a two-stage model: facility-level escalation risk (Redica, complete) →
    drug-level shortage model — text features as a separate characterization layer.
  </div>
</section>

<!-- 1: DATA SNAPSHOT -->
<section>
  <div class="section-head"><span class="step-num">1</span><h2>Data at a glance</h2></div>
  <div class="stats">
    <div class="stat">
      <div class="num">{d["n_drugs"]}<small>APIs</small></div>
      <div class="lbl">Valisure-tested pilot universe</div>
    </div>
    <div class="stat">
      <div class="num">{d["n_feis"]}<small>FEIs</small></div>
      <div class="lbl">Manufacturing facilities (FDA-registered)</div>
    </div>
    <div class="stat accent">
      <div class="num">{d["n_feis_text"]}<small>FEIs</small></div>
      <div class="lbl">With public 483 PDFs (text coverage)</div>
    </div>
    <div class="stat teal">
      <div class="num">{d["n_snapshots"]}<small>snapshots</small></div>
      <div class="lbl">483 documents scored by LLM</div>
    </div>
  </div>
  <div class="note" style="margin-top:12px;">
    <strong>Coverage caveat:</strong> 37/129 FEIs have public 483 PDFs (28.7%); among those,
    median inspection coverage is ~22%. Text-based findings apply to this subset only.
    Structured inspection outcomes (OAI/VAI/NAI) are available for 127/129 facilities from FDA OASIS.
  </div>
</section>

<!-- 2: DATA SOURCES -->
<section>
  <div class="section-head"><span class="step-num">2</span><h2>Data sources</h2></div>
  <div class="sources">
    <div class="src">
      <div class="name">FDA Form 483s</div>
      <div class="role">{d["n_feis_text"]} FEIs · {d["n_snapshots"]} snapshots · LLM-extracted observations</div>
    </div>
    <div class="src">
      <div class="name">FDA Inspections (OASIS)</div>
      <div class="role">Complete inspection history · OAI/VAI/NAI outcomes · 127/129 FEIs</div>
    </div>
    <div class="src">
      <div class="name">Redica</div>
      <div class="role">Red-flag events, inspection ratings, warning letter tracking</div>
    </div>
    <div class="src">
      <div class="name">FDA Recalls</div>
      <div class="role">Drug recall records, class and reason, FEI-linked</div>
    </div>
    <div class="src">
      <div class="name">UUtah</div>
      <div class="role">Shortage onset dates — event anchor for lead-lag analysis</div>
    </div>
  </div>
</section>

<!-- 3: FORWARD VALIDATION -->
<section>
  <div class="section-head"><span class="step-num">3</span>
    <h2>Does 483 text content predict what happens next?</h2></div>
  <div class="sub">
    Each of the {fwd_n} snapshots is split at the median of each text feature.
    Every facility already received a 483 — differences come from <strong>what the text says</strong>,
    not the mere existence of the document.
    Base rates: escalation {esc_base}% · recall {rec_base}%.
  </div>
  <div class="chart-row">
    <div class="card">
      <h3>OAI or Warning Letter within 24 months</h3>
      <div class="csub">Orange = above feature median · teal = at/below · hover for lift</div>
      <div class="chart-host tall"><canvas id="escChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Drug recall at the facility within 24 months</h3>
      <div class="csub">Same median split. Recall@12m excluded (only 2 events in sample).</div>
      <div class="chart-host tall"><canvas id="recChart"></canvas></div>
    </div>
  </div>
  <div class="note dark">
    <em>Repeat violations</em> ({esc_repeat}) and <em>cross-inspection repeats</em> ({esc_cross})
    are the strongest escalation predictors. <em>Contamination</em> ({esc_contam}) and
    <em>facility-wide scope</em> ({esc_scope}) also predict escalation.
    <em>Buildings/equipment violations</em> ({rec_bldg}) predict recalls.
    <em>Critical+Major severity</em> is strongest at 12 months ({sev12_txt}) — FDA acts quickly on
    documented defects. n = {fwd_n} snapshots, {n_cov_feis} FEIs — exploratory.
  </div>
</section>

<!-- 4: MONTHLY LEAD-LAG -->
<section>
  <div class="section-head"><span class="step-num">4</span>
    <h2>Signal accumulation before shortage onset</h2></div>
  <div class="sub">
    Monthly event study: mean signal at each offset month relative to shortage onset (month 0).
    Control baseline = drug-months with no onset within ±12 months. Shaded = ±1 SE.
    N = {d["monthly_onset_months"]} onset months · 14 drugs · exploratory.
  </div>

  <hr class="divider"/>
  <h3 style="margin:0 0 4px;">4A · Regulatory signals (OAI / VAI / total inspections)</h3>
  <div class="chart-row three" style="margin:10px 0 14px;">
    <div class="card"><h3>OAI Inspections</h3>
      <div class="chart-host"><canvas id="llR1"></canvas></div></div>
    <div class="card"><h3>VAI Inspections</h3>
      <div class="chart-host"><canvas id="llR2"></canvas></div></div>
    <div class="card"><h3>Total Inspections</h3>
      <div class="chart-host"><canvas id="llR3"></canvas></div></div>
  </div>
  <div class="note">
    OAI events are sparse at n=14 drugs; no clear ramp-up before onset.
    VAI inspections reflect voluntary remediation activity. Total inspections are flat.
  </div>

  <hr class="divider" style="margin-top:20px;"/>
  <h3 style="margin:16px 0 4px;">4B · Time-varying text signals (trailing 24 months)</h3>
  <div style="color:var(--muted);font-size:13px;margin-bottom:10px;">
    Count of 483 documents meeting risk criteria at the drug's facilities, trailing 24 months,
    aggregated to drug-month. Solid = mean near onset; dashed = control baseline.
  </div>
  <div class="chart-row" style="margin-bottom:4px;">
    <div class="card">
      <h3>Repeat-violation 483s</h3>
      <div class="csub">Count of 483 documents with ≥1 repeat-violation finding.</div>
      <div class="chart-host"><canvas id="llT1"></canvas></div>
    </div>
    <div class="card">
      <h3>Red-flagged 483s (≥2 risk markers)</h3>
      <div class="csub">Documents meeting ≥2 of 4 markers: repeat, contamination, OOS/OOT, Critical+Major severity.</div>
      <div class="chart-host"><canvas id="llT2"></canvas></div>
    </div>
  </div>
</section>

<!-- 5: INTERACTIVE TIMELINE -->
<section>
  <div class="section-head"><span class="step-num">5</span>
    <h2>Facility timeline explorer</h2></div>
  <div class="sub">
    Select any text-covered facility. Four event lanes:
    <strong>▲ Inspections</strong> (red=OAI · orange=VAI · green=NAI) ·
    <strong>● 483 Snapshots</strong> (red=red-flagged · blue=normal · size = observation count) ·
    <strong>◆ Recalls</strong> · <strong>── Shortages</strong>.
    Hover any point for details. Sorted by risk score (high-risk 483s + OAI).
  </div>
  <div class="card" style="margin-bottom:4px;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;">
      <label style="font-weight:700;color:var(--navy);white-space:nowrap;">Facility:</label>
      <select id="feiSelector" style="font-size:13px;padding:6px 10px;border:1px solid var(--rule);
        border-radius:4px;background:var(--white);color:var(--ink);
        flex:1;min-width:260px;max-width:560px;"></select>
      <div id="feiStats" style="font-size:12px;color:var(--muted);"></div>
    </div>
    <div style="position:relative;height:310px;"><canvas id="timelineChart"></canvas></div>
    <div id="feiStory" class="note" style="margin-top:12px;font-size:13px;"></div>
  </div>
  <div class="note" style="margin-top:8px;">
    Temporal co-occurrence only. Timelines illustrate the chronological mechanism;
    they do not establish attribution between a specific facility and a shortage.
  </div>
</section>

<!-- 6: FEI SUMMARY TABLE -->
<section>
  <div class="section-head"><span class="step-num">6</span>
    <h2>Text-covered facilities</h2></div>
  <div class="sub">
    {n_cov_feis} of {d["n_feis"]} facilities have public 483 PDFs.
    Sorted by red-flagged snapshot count, then OAI inspections.
  </div>
  <table class="signals">
    <thead><tr>
      <th>FEI</th><th>Facility</th><th>APIs</th>
      <th>Snapshots</th><th>Red-flagged</th><th>OAI</th><th>VAI</th><th>Recalls</th>
    </tr></thead>
    <tbody>{fei_table_rows}</tbody>
  </table>
</section>

<!-- 7: LIMITATIONS & NEXT STEPS -->
<section>
  <div class="section-head"><span class="step-num">7</span>
    <h2>Limitations &amp; next steps</h2></div>
  <div class="two-col">
    <div class="col-card">
      <div class="col-head l">Limitations</div>
      <ul>
        <li>37/129 FEIs have text (28.7%); median inspection coverage within those is ~22%. Text findings apply to this subset only — absence of text ≠ absence of risk.</li>
        <li>14 drugs, 79 snapshots — wide confidence intervals throughout. All findings are exploratory.</li>
        <li>Shortage–facility links are via API name only; a shortage may involve a facility not in this set.</li>
        <li>All associations are descriptive — no causal identification.</li>
      </ul>
    </div>
    <div class="col-card">
      <div class="col-head n">Next steps</div>
      <ul>
        <li><strong>Stage 1 (facility):</strong> logistic model predicting OAI/Warning Letter from Redica inspection counts (complete, 127 FEIs) → <em>p_esc</em> per facility.</li>
        <li><strong>Stage 2 (drug):</strong> max/mean <em>p_esc</em> across a drug's facilities as input to drug-year shortage model — replaces raw text averaging.</li>
        <li><strong>m13 case studies:</strong> chronological signal trajectory for the 2 best-covered FEIs — dated 483 content preceding OAI and shortage, in temporal order.</li>
        <li>Expand 483 PDF coverage; systematic collection is the binding constraint on text-based analysis.</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  FDA 483 Signal Analysis · {d["n_drugs"]} APIs · {d["n_feis"]} FEIs · Exploratory · June 2026
</footer>
</div>

<script>
Chart.defaults.font.family = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif';
Chart.defaults.color = "#1A2233";
Chart.defaults.font.size = 11.5;

{redica_js}
{text_ll_js}
{fwd_js}
{timeline_js}
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
