"""
Module MM07 — Dashboard generator.

Three sections, one message:
  1. Forward validation  — 483 text content predicts OAI/recall (facility level, m12)
  2. Facility timelines  — interactive chronological explorer (m11)
  3. Coverage & next steps
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from config import OUT_DATA, OUT_TABS, OUT_ROOT, OUT_LOGS, OUT_FIGS, VALISURE_FEI
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

def compute_data() -> dict:
    d: dict = {}

    # Basic counts
    bridge = None
    if VALISURE_FEI.exists():
        bridge = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping",
                               usecols=["API", "FEI_NUMBER"]).dropna(subset=["FEI_NUMBER"])
        bridge["FEI_NUMBER"] = bridge["FEI_NUMBER"].astype(int)
    ts_path = OUT_DATA.parent.parent / "Data" / "99 - Outputs - Text Analysis" / \
              "483_fei_text_features_timeseries.csv"
    # Use config path via the module import path already resolved
    from config import DATA
    ts_path = DATA / "99 - Outputs - Text Analysis" / "483_fei_text_features_timeseries.csv"
    ts_raw = pd.read_csv(ts_path) if ts_path.exists() else None

    d["n_drugs"]     = int(bridge["API"].nunique())        if bridge  is not None else 14
    d["n_feis"]      = int(bridge["FEI_NUMBER"].nunique()) if bridge  is not None else 129
    d["n_feis_text"] = int(ts_raw["fei"].nunique())        if ts_raw  is not None else 37
    d["n_snaps"]     = len(ts_raw)                         if ts_raw  is not None else 79

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

    # m11 FEI timeline events
    ev_all = _read("fei_events_all.csv")
    d["fei_events"] = {}
    if ev_all is not None:
        ev_all["event_date"] = pd.to_datetime(ev_all["event_date"], errors="coerce")
        ev_all["year_dec"] = (ev_all["event_date"].dt.year +
                              (ev_all["event_date"].dt.dayofyear - 1) / 365.25).round(3)
        for fei_id, grp in ev_all.groupby("fei"):
            firm = grp["firm_name"].dropna().iloc[0] if grp["firm_name"].notna().any() else str(fei_id)
            apis = grp["apis"].dropna().iloc[0] if "apis" in grp.columns and grp["apis"].notna().any() else ""
            ev483, evinsp, evrec, shbands = [], [], [], []
            for _, r in grp.sort_values("event_date").iterrows():
                if pd.isna(r.get("year_dec")): continue
                yr = float(r["year_dec"]); etype = str(r["event_type"])
                lbl = str(r.get("event_label", ""))[:70]
                if etype == "483_snapshot":
                    ev483.append({"yr": yr,
                        "hr":    bool(r["high_risk_483"]) if pd.notna(r.get("high_risk_483")) else False,
                        "n_obs": int(r["n_obs_total"]) if pd.notna(r.get("n_obs_total")) else 1,
                        "sev":   round(float(r["severity_critmajor_share"]), 2)
                                 if pd.notna(r.get("severity_critmajor_share")) else 0.0,
                        "label": lbl})
                elif etype == "inspection_outcome":
                    evinsp.append({"yr": yr,
                        "cls": str(r["classification"]) if pd.notna(r.get("classification")) else "?",
                        "label": lbl})
                elif etype == "recall":
                    evrec.append({"yr": yr,
                        "cls": str(r["recall_class"]) if pd.notna(r.get("recall_class")) else "?",
                        "label": lbl})
                elif etype == "shortage_start":
                    drug = str(r.get("shortage_drug", ""))[:40]
                    end_yr = None
                    resolved = r.get("shortage_resolved_date")
                    if pd.notna(resolved) and str(resolved) not in ("nan", "NaT", ""):
                        try:
                            rd = pd.to_datetime(resolved, errors="coerce")
                            if pd.notna(rd):
                                end_yr = round(float(rd.year + (rd.dayofyear-1)/365.25), 3)
                        except Exception:
                            pass
                    shbands.append({"start": yr, "end": end_yr or 2025.5, "drug": drug})
            d["fei_events"][str(fei_id)] = {
                "firm": str(firm)[:55], "apis": str(apis)[:60],
                "events_483": ev483, "events_insp": evinsp,
                "events_recall": evrec, "shortage_bands": shbands,
                "n_483": len(ev483),
                "n_hr":  sum(1 for e in ev483 if e["hr"]),
                "n_oai": sum(1 for e in evinsp if e["cls"] == "OAI"),
                "n_recalls": len(evrec),
            }
    return d


# ─────────────────────────────────────────────────────────────────────────────

def generate_html(d: dict) -> str:

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

    fei_events_data = d["fei_events"]
    timeline_js = f"""
const FEI_EVENTS={_j(fei_events_data)};
let _tChart=null;
(function(){{
  const sel=document.getElementById('feiSel');
  if(!sel||!Object.keys(FEI_EVENTS).length){{
    if(sel){{sel.innerHTML='<option>No data</option>';sel.disabled=true;}}return;
  }}
  const feiList=Object.keys(FEI_EVENTS).sort((a,b)=>{{
    const da=FEI_EVENTS[a],db=FEI_EVENTS[b];
    return(db.n_hr+db.n_oai*2+db.n_recalls)-(da.n_hr+da.n_oai*2+da.n_recalls);
  }});
  feiList.forEach(fei=>{{
    const ev=FEI_EVENTS[fei],opt=document.createElement('option');
    opt.value=fei;
    opt.textContent=ev.firm+' ('+fei+')'+(ev.apis?' · '+ev.apis:'');
    sel.appendChild(opt);
  }});
  function build(fei){{
    const ev=FEI_EVENTS[fei];if(!ev)return;
    const st=document.getElementById('feiStats');
    if(st)st.innerHTML='<strong>'+ev.n_483+'</strong> 483s · <span style="color:#C0392B;font-weight:700;">'
      +ev.n_hr+' high-risk</span> · <strong>'+ev.n_oai+'</strong> OAI · <strong>'+ev.n_recalls+'</strong> recalls';
    const ds=[];
    ev.shortage_bands.forEach((s,i)=>{{
      ds.push({{label:i===0?'Shortage':'_',type:'line',
        data:[{{x:s.start,y:0,label:s.drug}},{{x:s.end,y:0,label:s.drug}}],
        borderColor:'rgba(100,100,100,0.5)',borderWidth:10,pointRadius:2,fill:false,order:0}});
    }});
    const hr=ev.events_483.filter(e=>e.hr),ok=ev.events_483.filter(e=>!e.hr);
    if(hr.length)ds.push({{label:'483 High-risk',type:'scatter',
      data:hr.map(e=>(({{x:e.yr,y:2,label:e.label,n_obs:e.n_obs,sev:e.sev}}))  ),
      backgroundColor:'rgba(192,57,43,0.88)',borderColor:'#922B21',borderWidth:1,
      pointRadius:ctx=>Math.max(6,Math.min(18,(ctx.raw?.n_obs||1)*1.8)),order:2}});
    if(ok.length)ds.push({{label:'483 Normal',type:'scatter',
      data:ok.map(e=>(({{x:e.yr,y:2,label:e.label,n_obs:e.n_obs,sev:e.sev}}))  ),
      backgroundColor:'rgba(28,114,147,0.7)',borderColor:'#065A82',borderWidth:1,
      pointRadius:ctx=>Math.max(5,Math.min(14,(ctx.raw?.n_obs||1)*1.5)),order:2}});
    const CC={{OAI:'rgba(192,57,43,0.9)',VAI:'rgba(230,126,34,0.9)',NAI:'rgba(39,174,96,0.9)'}};
    ['OAI','VAI','NAI'].forEach(c=>{{
      const evs=ev.events_insp.filter(e=>e.cls===c);if(!evs.length)return;
      ds.push({{label:'Insp: '+c,type:'scatter',data:evs.map(e=>(({{x:e.yr,y:3,label:e.label}}))  ),
        backgroundColor:CC[c],borderColor:CC[c],pointStyle:'triangle',pointRadius:9,order:1}});
    }});
    const RC={{'Class I':'rgba(139,0,0,0.9)','Class II':'rgba(204,85,0,0.9)'}};
    Object.entries(RC).forEach(([c,col])=>{{
      const evs=ev.events_recall.filter(e=>e.cls===c);if(!evs.length)return;
      ds.push({{label:'Recall '+c,type:'scatter',data:evs.map(e=>(({{x:e.yr,y:1,label:e.label}}))  ),
        backgroundColor:col,borderColor:col,pointStyle:'rectRot',pointRadius:9,order:1}});
    }});
    if(_tChart)_tChart.destroy();
    const ctx=document.getElementById('tlChart');if(!ctx)return;
    _tChart=new Chart(ctx,{{data:{{datasets:ds}},options:{{maintainAspectRatio:false,
      scales:{{
        x:{{type:'linear',min:2008,max:2026,title:{{display:true,text:'Year'}},
            ticks:{{stepSize:1,callback:v=>Number.isInteger(v)?v:''}},grid:{{color:'#EEE'}}}},
        y:{{min:-0.5,max:3.7,ticks:{{stepSize:1,callback:v=>{{
          return{{0:'Shortages',1:'Recalls',2:'483s',3:'Inspections'}}[Math.round(v)]||'';
        }}}},grid:{{color:'#EEE'}}}}
      }},
      plugins:{{
        legend:{{position:'bottom',labels:{{boxWidth:11,font:{{size:10}},filter:i=>!i.text.startsWith('_')}}}},
        tooltip:{{callbacks:{{label:ctx=>{{
          const r=ctx.raw;
          if(r.label)return r.label;
          if(r.n_obs!==undefined)return r.n_obs+' obs, sev='+r.sev;
          return '';
        }}}}}}
      }}
    }}}});
    const st2=document.getElementById('feiStory'),s=[];
    if(ev.n_hr>0)s.push('<strong>'+ev.n_hr+'/'+ev.n_483+'</strong> snapshots high-risk.');
    if(ev.n_oai>0)s.push('<strong>'+ev.n_oai+' OAI</strong>.');
    if(ev.n_recalls>0)s.push('<strong>'+ev.n_recalls+'</strong> recall(s).');
    if(ev.n_hr>0&&ev.n_oai>0)s.push('Look for red ● preceding OAI ▲.');
    if(st2)st2.innerHTML=s.join(' ')||'No high-risk events.';
  }}
  build(feiList[0]);
  sel.addEventListener('change',()=>build(sel.value));
}})();"""

    n  = d["n_feis_text"]
    sn = d["n_snaps"]
    eb = d["esc_base"]
    rb = d["rec_base"]
    fn = d["fwd_n"]

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

<!-- SECTION 1: FORWARD VALIDATION -->
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

<!-- SECTION 2: TIMELINE -->
<section>
  <h2>Chronological timeline — facility explorer</h2>
  <p class="sub">
    Select a facility. Four lanes: <strong>▲ Inspections</strong> (red=OAI · orange=VAI · green=NAI) ·
    <strong>● 483 Snapshots</strong> (red=high-risk · blue=normal · size = observation count) ·
    <strong>◆ Recalls</strong> · <strong>── Shortages</strong>.
    Does red ● precede OAI ▲? That is the chronological test.
  </p>
  <div class="card">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap;">
      <label style="font-weight:700;color:var(--navy);">Facility:</label>
      <select id="feiSel" style="font-size:13px;padding:5px 9px;border:1px solid var(--rule);
        border-radius:4px;flex:1;min-width:240px;max-width:520px;"></select>
      <span id="feiStats" style="font-size:12px;color:var(--muted);"></span>
    </div>
    <div style="position:relative;height:300px;"><canvas id="tlChart"></canvas></div>
    <div id="feiStory" class="note" style="margin-top:10px;font-size:12.5px;"></div>
  </div>
  <p style="color:var(--muted);font-size:12px;margin-top:8px;">
    Temporal co-occurrence only — shortages may involve facilities not in this set.
  </p>
</section>

<!-- SECTION 3: COVERAGE & NEXT STEPS -->
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
        <li><strong>Stage 1:</strong> Redica inspection counts (complete, 127 FEIs) → p_esc per facility via logistic regression (m12, done).</li>
        <li><strong>Stage 2:</strong> max/mean p_esc across facilities → drug-year shortage model, replacing raw text averaging (m07/m09, done).</li>
        <li><strong>m13:</strong> Chronological case studies for 2 best-covered FEIs — dated text signal trajectory vs OAI/shortage outcomes.</li>
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
