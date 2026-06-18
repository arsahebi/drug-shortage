"""
Module 16 — FEI-level facility drill-down dashboard.

For each drug the user selects a specific FEI (manufacturing facility) and sees:
  - Quality signal timeseries (483 LLM signals over time)
  - FDA inspection timeline (OAI / VAI / NAI outcomes)
  - Recall events for that facility
  - Drug-level shortage periods as shaded regions

Output:
  outputs/figures/fei_facility_dashboard.html

Run:
  python m16_fei_dashboard.py
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    DATA, OUT_FIGS, OUT_LOGS,
    TEXT_TIMESERIES_REDICA_CSV, RECALL_FILT, VALISURE_FEI,
    PANEL_START_YEAR, PANEL_END_YEAR,
)
from utils import get_logger

log = get_logger("m16_fei_dashboard", OUT_LOGS / "m16_fei_dashboard.log")

UUTAH_FILE  = DATA / "24 - UUtah - Drug Shortage" / "raw" / "efox shortages small file through 2025 final.xlsx"
INSP_FILE   = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
OUT_HTML    = OUT_FIGS / "fei_facility_dashboard.html"

DATE_MIN = f"{PANEL_START_YEAR}-01-01"
DATE_MAX = f"{PANEL_END_YEAR}-12-31"

# Canonical drug name → display name
TARGET_DRUGS = {
    "metformin":      "Metformin",
    "atorvastatin":   "Atorvastatin",
    "bupropion":      "Bupropion",
    "pantoprazole":   "Pantoprazole",
    "vancomycin":     "Vancomycin",
    "lisinopril":     "Lisinopril",
    "metoprolol":     "Metoprolol",
    "metronidazole":  "Metronidazole",
    "potassium chloride": "Potassium Chloride",
    "tacrolimus":     "Tacrolimus",
    "magnesium sulfate":  "Magnesium Sulfate",
    "calcium gluconate":  "Calcium Gluconate",
    "ampicillin":     "Ampicillin",
}

SIGNAL_COLS = {
    "severity_critmajor_share": ("Critical/Major severity", "#E57200"),
    "contamination_llm_share":  ("Contamination",           "#7B2D8B"),
    "data_integrity_llm_share": ("Data integrity",          "#006666"),
    "investigation_llm_share":  ("Investigation failures",  "#1A4480"),
}

CLASS_COLORS = {
    "Class I":   "#C41E3A",
    "Class II":  "#E57200",
    "Class III": "#D4A017",
}

INSP_COLORS = {
    "Out of Acceptable (OAI)":     "#C41E3A",
    "Voluntary Action (VAI)":      "#E57200",
    "No Action Indicated (NAI)":   "#2E7D32",
}

# Normalize classification strings
def _norm_class(c: str) -> str:
    c = str(c).strip()
    if "Out of Acceptable" in c or c.startswith("OAI"):
        return "Out of Acceptable (OAI)"
    if "Voluntary Action" in c or c.startswith("VAI"):
        return "Voluntary Action (VAI)"
    if "No Action" in c or c.startswith("NAI"):
        return "No Action Indicated (NAI)"
    return c


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_fei_drug_map() -> pd.DataFrame:
    vfei = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    vfei = vfei[["API", "FEI_NUMBER"]].dropna(subset=["FEI_NUMBER"])
    vfei.columns = ["drug_raw", "fei"]
    vfei["fei"] = vfei["fei"].astype(int)
    vfei["drug_raw"] = vfei["drug_raw"].astype(str).str.strip()
    # normalize to canonical names
    def _match(name: str) -> str | None:
        nl = name.lower()
        for k, v in sorted(TARGET_DRUGS.items(), key=lambda x: -len(x[0])):
            if k in nl:
                return v
        # pass through as-is if not in map
        return name.strip()
    vfei["drug"] = vfei["drug_raw"].map(_match)
    return vfei[["fei", "drug"]].drop_duplicates()


def _load_quality_ts() -> pd.DataFrame:
    if not TEXT_TIMESERIES_REDICA_CSV.exists():
        log.warning("Quality timeseries CSV not found: %s", TEXT_TIMESERIES_REDICA_CSV)
        return pd.DataFrame(columns=["fei", "snapshot_date"] + list(SIGNAL_COLS.keys()))
    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"]           = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    ts = ts.dropna(subset=["fei", "snapshot_date"])
    ts = ts[(ts["snapshot_date"] >= DATE_MIN) & (ts["snapshot_date"] <= DATE_MAX)]
    cols = ["fei", "snapshot_date"] + [c for c in SIGNAL_COLS if c in ts.columns]
    return ts[cols]


def _load_inspections(valisure_feis: set) -> pd.DataFrame:
    insp = pd.read_excel(INSP_FILE, usecols=[
        "FEI Number", "Legal Name", "City", "Country/Area",
        "Inspection End Date", "Classification",
    ])
    insp.columns = ["fei", "name", "city", "country", "insp_date", "classification"]
    insp["fei"]       = pd.to_numeric(insp["fei"], errors="coerce").astype("Int64")
    insp["insp_date"] = pd.to_datetime(insp["insp_date"], errors="coerce")
    insp = insp[insp["fei"].isin(valisure_feis) & insp["insp_date"].notna()]
    insp = insp[(insp["insp_date"] >= DATE_MIN) & (insp["insp_date"] <= DATE_MAX)]
    insp["classification"] = insp["classification"].map(_norm_class)
    log.info("Inspections for Valisure FEIs: %d rows", len(insp))
    return insp.reset_index(drop=True)


def _load_recalls(valisure_feis: set) -> pd.DataFrame:
    r = pd.read_csv(RECALL_FILT, low_memory=False)
    r.columns = [c.strip() for c in r.columns]
    r["fei"]        = pd.to_numeric(r["FEI Number"], errors="coerce").astype("Int64")
    r["recall_date"] = pd.to_datetime(r["Recall_Date"], errors="coerce")
    r = r[r["fei"].isin(valisure_feis) & r["recall_date"].notna()]
    r = r[(r["recall_date"] >= DATE_MIN) & (r["recall_date"] <= DATE_MAX)]
    r["class_norm"] = r["Event Classification"].astype(str).str.strip()
    # shorten product description for hover
    r["product_short"] = r["Product Description"].astype(str).str[:80]
    return r[["fei", "recall_date", "class_norm", "product_short", "Reason for Recall"]].reset_index(drop=True)


def _load_shortages() -> dict[str, list[dict]]:
    """Returns {drug: [{start, end, reason}, ...]}"""
    sh = pd.read_excel(UUTAH_FILE, header=1)
    sh.columns = [c.strip() for c in sh.columns]
    sh = sh.rename(columns={
        "Drug Shortages": "drug_raw",
        "Date Notified": "start",
        "Date Resolved": "end",
        "Reason": "reason",
    })
    sh["drug_raw"] = sh["drug_raw"].astype(str).str.lower()
    sh["start"]    = pd.to_datetime(sh["start"], errors="coerce")
    sh["end"]      = pd.to_datetime(sh["end"],   errors="coerce")

    result: dict[str, list[dict]] = {}
    for key, display in TARGET_DRUGS.items():
        sub = sh[sh["drug_raw"].str.contains(key, na=False)].copy()
        sub = sub.dropna(subset=["start"])
        sub["end"] = sub["end"].fillna(pd.Timestamp(DATE_MAX))
        sub = sub[(sub["start"] >= DATE_MIN) & (sub["start"] <= DATE_MAX)]
        rows = []
        for _, row in sub.iterrows():
            rows.append({
                "start":  row["start"].strftime("%Y-%m-%d"),
                "end":    row["end"].strftime("%Y-%m-%d"),
                "reason": str(row.get("reason", "")).strip() or "Unknown",
            })
        if rows:
            result[display] = rows
    return result


def _fei_name_map(insp: pd.DataFrame) -> dict[int, str]:
    """First occurrence of each FEI → 'Firm Name, City, Country'"""
    first = insp.sort_values("insp_date").groupby("fei").first().reset_index()
    out: dict[int, str] = {}
    for _, row in first.iterrows():
        parts = [str(row["name"]).strip()]
        if row["city"] and str(row["city"]).strip() not in ("-", "nan"):
            parts.append(str(row["city"]).strip())
        if row["country"] and str(row["country"]).strip() not in ("-", "nan"):
            parts.append(str(row["country"]).strip())
        out[int(row["fei"])] = ", ".join(parts)
    return out


# ── Build per-FEI JSON data ───────────────────────────────────────────────────

def _build_fei_data(
    fdmap: pd.DataFrame,
    quality: pd.DataFrame,
    inspections: pd.DataFrame,
    recalls: pd.DataFrame,
    shortages: dict[str, list[dict]],
    name_map: dict[int, str],
) -> dict:
    """
    Returns nested dict:
    {drug: {fei_int: {name, stats, quality, inspections, recalls}}}
    plus top-level "shortages" key.
    """
    fei_data: dict = {}

    for drug, grp in fdmap.groupby("drug"):
        drug_feis = grp["fei"].unique().tolist()
        if not drug_feis:
            continue

        drug_dict: dict[str, dict] = {}
        for fei in sorted(drug_feis):
            fei_int = int(fei)

            # quality signals
            q_fei = quality[quality["fei"] == fei_int].sort_values("snapshot_date")
            q_dates = q_fei["snapshot_date"].dt.strftime("%Y-%m-%d").tolist()
            q_signals: dict[str, list] = {}
            for col in SIGNAL_COLS:
                if col in q_fei.columns:
                    q_signals[col] = q_fei[col].fillna(0).round(4).tolist()

            # inspections
            i_fei = inspections[inspections["fei"] == fei_int].sort_values("insp_date")
            i_rows = []
            for _, row in i_fei.iterrows():
                i_rows.append({
                    "date":  row["insp_date"].strftime("%Y-%m-%d"),
                    "class": str(row["classification"]),
                })

            # stats from inspections
            n_oai = sum(1 for r in i_rows if "OAI" in r["class"])
            n_vai = sum(1 for r in i_rows if "VAI" in r["class"])
            n_nai = sum(1 for r in i_rows if "NAI" in r["class"])

            # recalls
            r_fei = recalls[recalls["fei"] == fei_int].sort_values("recall_date")
            r_rows = []
            for _, row in r_fei.iterrows():
                r_rows.append({
                    "date":    row["recall_date"].strftime("%Y-%m-%d"),
                    "class":   str(row["class_norm"]),
                    "product": str(row["product_short"]),
                    "reason":  str(row["Reason for Recall"])[:100],
                })

            drug_dict[str(fei_int)] = {
                "name":       name_map.get(fei_int, f"FEI {fei_int}"),
                "stats":      {"n_oai": n_oai, "n_vai": n_vai, "n_nai": n_nai, "n_recalls": len(r_rows)},
                "q_dates":    q_dates,
                "q_signals":  q_signals,
                "inspections": i_rows,
                "recalls":    r_rows,
            }

        fei_data[drug] = drug_dict

    return {"drugs": fei_data, "shortages": shortages}


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(fei_data: dict) -> str:
    data_json = json.dumps(fei_data, default=str, ensure_ascii=False)

    signal_labels_js = json.dumps({k: v[0] for k, v in SIGNAL_COLS.items()})
    signal_colors_js = json.dumps({k: v[1] for k, v in SIGNAL_COLS.items()})
    class_colors_js  = json.dumps(CLASS_COLORS)
    insp_colors_js   = json.dumps(INSP_COLORS)

    drugs_sorted = sorted(fei_data["drugs"].keys())
    drugs_js = json.dumps(drugs_sorted)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FEI Facility Dashboard — Drug Shortage Project</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
         background: #F5F7FA; color: #1A1A2E; }}
  .page-wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px; }}

  h1 {{ font-size: 1.55rem; font-weight: 700; color: #1A1A2E; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.92rem; color: #5A6475; margin-bottom: 24px; }}

  /* Drug selector */
  .drug-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
  .drug-btn {{
    padding: 6px 14px; border-radius: 20px; border: 1.5px solid #C8D0DC;
    background: white; font-size: 0.85rem; cursor: pointer; font-weight: 500;
    transition: all 0.15s; color: #3A4560;
  }}
  .drug-btn:hover {{ border-color: #1A4480; color: #1A4480; }}
  .drug-btn.active {{
    background: #1A4480; border-color: #1A4480; color: white; font-weight: 600;
  }}

  /* FEI selector row */
  .fei-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }}
  .fei-row label {{ font-size: 0.9rem; font-weight: 600; color: #3A4560; white-space: nowrap; }}
  #fei-select {{
    flex: 1; max-width: 520px; padding: 8px 12px; border-radius: 6px;
    border: 1.5px solid #C8D0DC; font-size: 0.88rem; background: white;
    color: #1A1A2E; cursor: pointer;
  }}

  /* Stats card */
  .stats-card {{
    background: white; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #E2E8F0; margin-bottom: 20px;
    display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start;
  }}
  .stat-block {{ display: flex; flex-direction: column; }}
  .stat-label {{ font-size: 0.75rem; font-weight: 600; color: #7A8499; text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat-value {{ font-size: 1.35rem; font-weight: 700; color: #1A1A2E; }}
  .stat-value.oai {{ color: #C41E3A; }}
  .stat-value.vai {{ color: #E57200; }}
  .stat-value.nai {{ color: #2E7D32; }}
  .stat-value.recall {{ color: #7B2D8B; }}
  .fei-name {{ font-size: 0.97rem; font-weight: 600; color: #1A4480; flex: 1 0 100%; }}

  /* Chart panel */
  .chart-card {{
    background: white; border-radius: 10px; padding: 16px;
    border: 1px solid #E2E8F0; margin-bottom: 20px;
  }}
  .chart-title {{ font-size: 0.9rem; font-weight: 600; color: #3A4560; margin-bottom: 8px; }}

  /* No-data state */
  .no-data-msg {{
    text-align: center; color: #9AA3B2; font-size: 0.95rem; padding: 60px 0;
  }}
  .legend-note {{ font-size: 0.78rem; color: #9AA3B2; margin-top: 6px; }}
</style>
</head>
<body>
<div class="page-wrap">

  <h1>FEI Facility Explorer</h1>
  <p class="subtitle">Select a drug, then a manufacturing facility (FEI) to trace its quality history,
     inspection outcomes, recalls, and shortage periods.</p>

  <div class="drug-bar" id="drug-bar"></div>

  <div class="fei-row">
    <label for="fei-select">Facility (FEI):</label>
    <select id="fei-select"><option value="">— select drug first —</option></select>
  </div>

  <div class="stats-card" id="stats-card">
    <div class="fei-name" id="fei-name-label">Select a drug and facility above</div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Quality signals &amp; inspection outcomes</div>
    <div id="chart-quality"></div>
    <div class="no-data-msg" id="quality-empty" style="display:none">No quality signal data for this facility</div>
    <p class="legend-note">Vertical lines = inspections (OAI red · VAI orange · NAI green).
       Shaded bands = drug shortage periods (UUtah database).</p>
  </div>

  <div class="chart-card">
    <div class="chart-title">Recall events</div>
    <div id="chart-recalls"></div>
    <div class="no-data-msg" id="recalls-empty" style="display:none">No recalls recorded for this facility in the study period</div>
    <p class="legend-note">Bars colored by recall class (I = most severe). Shaded bands = shortage periods.</p>
  </div>

</div>

<script>
const FEI_DATA    = {data_json};
const SIG_LABELS  = {signal_labels_js};
const SIG_COLORS  = {signal_colors_js};
const CLASS_COLORS= {class_colors_js};
const INSP_COLORS = {insp_colors_js};
const DRUGS       = {drugs_js};

// ── State ──────────────────────────────────────────────────────────────────
let activeDrug = null;
let activeFei  = null;

// ── Drug buttons ───────────────────────────────────────────────────────────
function buildDrugBar() {{
  const bar = document.getElementById('drug-bar');
  DRUGS.forEach(drug => {{
    const btn = document.createElement('button');
    btn.className = 'drug-btn';
    btn.textContent = drug;
    btn.onclick = () => selectDrug(drug);
    bar.appendChild(btn);
  }});
}}

function selectDrug(drug) {{
  activeDrug = drug;
  document.querySelectorAll('.drug-btn').forEach(b => {{
    b.classList.toggle('active', b.textContent === drug);
  }});
  populateFeiDropdown(drug);
}}

// ── FEI dropdown ───────────────────────────────────────────────────────────
function populateFeiDropdown(drug) {{
  const sel  = document.getElementById('fei-select');
  const feis = FEI_DATA.drugs[drug] || {{}};
  sel.innerHTML = '';
  Object.keys(feis).sort().forEach(fei => {{
    const info = feis[fei];
    const opt = document.createElement('option');
    opt.value = fei;
    const oai = info.stats.n_oai;
    const tag = oai > 0 ? ` ⚠ ${{oai}} OAI` : '';
    opt.textContent = `${{fei}} — ${{info.name}}${{tag}}`;
    sel.appendChild(opt);
  }});
  if (sel.options.length > 0) {{
    sel.value = sel.options[0].value;
    selectFei(sel.value);
  }}
  sel.onchange = () => selectFei(sel.value);
}}

function selectFei(fei) {{
  activeFei = fei;
  const info = FEI_DATA.drugs[activeDrug][fei];
  if (!info) return;
  renderStats(info);
  renderQuality(info);
  renderRecalls(info);
}}

// ── Stats card ─────────────────────────────────────────────────────────────
function renderStats(info) {{
  const s = info.stats;
  document.getElementById('fei-name-label').textContent = info.name;
  document.getElementById('stats-card').innerHTML = `
    <div class="fei-name">${{info.name}}</div>
    <div class="stat-block">
      <span class="stat-label">OAI Inspections</span>
      <span class="stat-value oai">${{s.n_oai}}</span>
    </div>
    <div class="stat-block">
      <span class="stat-label">VAI Inspections</span>
      <span class="stat-value vai">${{s.n_vai}}</span>
    </div>
    <div class="stat-block">
      <span class="stat-label">NAI Inspections</span>
      <span class="stat-value nai">${{s.n_nai}}</span>
    </div>
    <div class="stat-block">
      <span class="stat-label">Recalls (study period)</span>
      <span class="stat-value recall">${{s.n_recalls}}</span>
    </div>
  `;
}}

// ── Shortage shapes helper ─────────────────────────────────────────────────
function getShortageShapes(yref) {{
  const periods = (FEI_DATA.shortages || {{}})[activeDrug] || [];
  return periods.map(p => ({{
    type: 'rect', xref: 'x', yref: yref,
    x0: p.start, x1: p.end, y0: 0, y1: 1,
    fillcolor: 'rgba(123,45,139,0.08)',
    line: {{ width: 0 }},
    layer: 'below',
  }}));
}}

// ── Inspection shapes helper ──────────────────────────────────────────────
function getInspShapes(inspections) {{
  return inspections.map(i => {{
    const col = INSP_COLORS[i.class] || '#888';
    return {{
      type: 'line', xref: 'x', yref: 'paper',
      x0: i.date, x1: i.date, y0: 0, y1: 1,
      line: {{ color: col + '99', width: 1.5, dash: 'dot' }},
      layer: 'above',
    }};
  }});
}}

// ── Quality chart ──────────────────────────────────────────────────────────
function renderQuality(info) {{
  const div   = document.getElementById('chart-quality');
  const empty = document.getElementById('quality-empty');

  if (!info.q_dates || info.q_dates.length === 0) {{
    div.style.display  = 'none';
    empty.style.display = 'block';
    return;
  }}
  div.style.display   = 'block';
  empty.style.display = 'none';

  const traces = [];
  Object.keys(SIG_LABELS).forEach(col => {{
    const vals = (info.q_signals || {{}})[col];
    if (!vals || vals.length === 0) return;
    traces.push({{
      type: 'scatter', mode: 'lines+markers',
      x: info.q_dates, y: vals,
      name: SIG_LABELS[col],
      line: {{ color: SIG_COLORS[col], width: 2 }},
      marker: {{ size: 7 }},
      hovertemplate: '<b>%{{x}}</b><br>' + SIG_LABELS[col] + ': %{{y:.0%}}<extra></extra>',
    }});
  }});

  // inspection scatter (colored dots at y=1.05)
  const inspByClass = {{}};
  (info.inspections || []).forEach(i => {{
    if (!inspByClass[i.class]) inspByClass[i.class] = [];
    inspByClass[i.class].push(i.date);
  }});
  Object.entries(inspByClass).forEach(([cls, dates]) => {{
    const col = INSP_COLORS[cls] || '#888';
    traces.push({{
      type: 'scatter', mode: 'markers',
      x: dates, y: Array(dates.length).fill(1.05),
      name: cls,
      marker: {{ color: col, size: 10, symbol: 'diamond' }},
      hovertemplate: '<b>%{{x}}</b><br>Inspection: ' + cls + '<extra></extra>',
    }});
  }});

  const shapes = [
    ...getShortageShapes('paper'),
    ...getInspShapes(info.inspections || []),
  ];

  const layout = {{
    height: 360,
    margin: {{ t: 20, r: 20, l: 60, b: 50 }},
    plot_bgcolor: '#FAFBFC', paper_bgcolor: 'white',
    xaxis: {{ type: 'date', range: ['{DATE_MIN}', '{DATE_MAX}'],
              showgrid: true, gridcolor: '#EEF0F4' }},
    yaxis: {{ title: '% of 483 observations flagged', tickformat: '.0%',
              range: [0, 1.15], showgrid: true, gridcolor: '#EEF0F4' }},
    legend: {{ orientation: 'h', y: -0.22, x: 0 }},
    shapes: shapes,
    hovermode: 'x unified',
  }};

  Plotly.react(div, traces, layout, {{responsive: true, displayModeBar: false}});
}}

// ── Recalls chart ──────────────────────────────────────────────────────────
function renderRecalls(info) {{
  const div   = document.getElementById('chart-recalls');
  const empty = document.getElementById('recalls-empty');

  const recalls = info.recalls || [];
  if (recalls.length === 0) {{
    div.style.display   = 'none';
    empty.style.display = 'block';
    return;
  }}
  div.style.display   = 'block';
  empty.style.display = 'none';

  // Group by class
  const byClass = {{}};
  recalls.forEach(r => {{
    if (!byClass[r.class]) byClass[r.class] = [];
    byClass[r.class].push(r);
  }});

  const traces = Object.entries(byClass).map(([cls, rows]) => {{
    const color = CLASS_COLORS[cls] || '#888';
    return {{
      type: 'bar',
      x: rows.map(r => r.date),
      y: Array(rows.length).fill(1),
      name: cls,
      marker: {{ color: color, opacity: 0.85 }},
      customdata: rows.map(r => [r.product, r.reason]),
      hovertemplate:
        '<b>%{{x}}</b> — ' + cls + '<br>%{{customdata[0]}}<br><i>%{{customdata[1]}}</i><extra></extra>',
    }};
  }});

  const shapes = getShortageShapes('paper');

  const layout = {{
    height: 200,
    margin: {{ t: 20, r: 20, l: 60, b: 50 }},
    plot_bgcolor: '#FAFBFC', paper_bgcolor: 'white',
    barmode: 'stack',
    xaxis: {{ type: 'date', range: ['{DATE_MIN}', '{DATE_MAX}'],
              showgrid: true, gridcolor: '#EEF0F4' }},
    yaxis: {{ title: 'Recall events', showticklabels: false, fixedrange: true }},
    legend: {{ orientation: 'h', y: -0.35, x: 0 }},
    shapes: shapes,
    hovermode: 'x unified',
  }};

  Plotly.react(div, traces, layout, {{responsive: true, displayModeBar: false}});
}}

// ── Init ───────────────────────────────────────────────────────────────────
buildDrugBar();
if (DRUGS.length > 0) selectDrug(DRUGS[0]);
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading data…")
    fdmap     = _load_fei_drug_map()
    valisure_feis = set(fdmap["fei"].unique())

    quality    = _load_quality_ts()
    inspections = _load_inspections(valisure_feis)
    recalls    = _load_recalls(valisure_feis)
    shortages  = _load_shortages()
    name_map   = _fei_name_map(inspections)

    log.info("FEI-drug pairs: %d | FEIs with quality: %d",
             len(fdmap), quality["fei"].nunique())

    fei_data = _build_fei_data(fdmap, quality, inspections, recalls, shortages, name_map)

    n_drugs = len(fei_data["drugs"])
    n_feis  = sum(len(v) for v in fei_data["drugs"].values())
    log.info("Built data for %d drugs, %d FEIs", n_drugs, n_feis)

    log.info("Building HTML…")
    html = build_html(fei_data)
    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


if __name__ == "__main__":
    main()
