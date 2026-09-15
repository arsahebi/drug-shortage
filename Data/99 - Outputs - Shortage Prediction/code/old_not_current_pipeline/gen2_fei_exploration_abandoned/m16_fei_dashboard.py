"""
Module 16 — FEI-level facility drill-down dashboard.

For each drug → FEI selection the dashboard shows:
  - Quality signal timeseries (483 LLM signals, one at a time via dropdown)
  - FDA inspection outcomes (OAI / VAI / NAI markers)
  - Recall events
  - Manufacturer's Medicaid market volume (SDUD, FEI-level via labeler code)
  - Drug-level shortage periods as shaded regions

Plotly.js is embedded inline (no CDN dependency).

Output:
  outputs/figures/fei_facility_dashboard.html
"""

from __future__ import annotations
import json
import re
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

import plotly as _plotly
PLOTLY_JS_PATH = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"

UUTAH_FILE  = DATA / "24 - UUtah - Drug Shortage" / "raw" / "efox shortages small file through 2025 final.xlsx"
INSP_FILE   = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
NDC_PRODUCT = DATA / "03 - FDA - NDC" / "product.csv"
SDUD_PANEL  = DATA / "04_11 - Build - Monthly Panel (SDUD+NADAC)" / "processed" / "2026-03-26-sdud_nadac_panel.csv"
OUT_HTML    = OUT_FIGS / "fei_facility_dashboard.html"

DATE_MIN = f"{PANEL_START_YEAR}-01-01"
DATE_MAX = f"{PANEL_END_YEAR}-12-31"

TARGET_DRUGS = {
    "metformin":          "Metformin",
    "atorvastatin":       "Atorvastatin",
    "bupropion":          "Bupropion",
    "pantoprazole":       "Pantoprazole",
    "vancomycin":         "Vancomycin",
    "lisinopril":         "Lisinopril",
    "metoprolol":         "Metoprolol",
    "metronidazole":      "Metronidazole",
    "potassium chloride": "Potassium Chloride",
    "tacrolimus":         "Tacrolimus",
    "magnesium sulfate":  "Magnesium Sulfate",
    "calcium gluconate":  "Calcium Gluconate",
    "ampicillin":         "Ampicillin",
}

# Signals to show in the quality chart
SIGNAL_COLS = {
    "severity_critmajor_share": ("Critical/Major severity", "#E57200"),
    "contamination_llm_share":  ("Contamination",           "#7B2D8B"),
    "data_integrity_llm_share": ("Data integrity",           "#006666"),
    "investigation_llm_share":  ("Investigation failures",   "#1A4480"),
}

CLASS_COLORS = {"Class I": "#C41E3A", "Class II": "#E57200", "Class III": "#D4A017"}
INSP_COLORS  = {
    "Out of Acceptable (OAI)":   "#C41E3A",
    "Voluntary Action (VAI)":    "#E57200",
    "No Action Indicated (NAI)": "#2E7D32",
}

SKIP_WORDS = {
    "limited","corporation","company","pharmaceuticals","pharmaceutical",
    "healthcare","manufacturing","laboratories","laboratory","private",
    "llc","ltd","inc","pvt","gmbh","industries","industry",
    "international","national","holdings","group","labs","pharma","usa",
    "america","americas","north","south","east","west","the","and",
    "products","product","drug","drugs","sciences","science",
}


def _norm_insp_class(c: str) -> str:
    c = str(c).strip()
    if "Out of Acceptable" in c or c.upper().startswith("OAI"):
        return "Out of Acceptable (OAI)"
    if "Voluntary Action" in c or c.upper().startswith("VAI"):
        return "Voluntary Action (VAI)"
    if "No Action" in c or c.upper().startswith("NAI"):
        return "No Action Indicated (NAI)"
    return c


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_fei_drug_map() -> pd.DataFrame:
    vfei = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    vfei = vfei[["API", "FEI_NUMBER"]].dropna(subset=["FEI_NUMBER"])
    vfei.columns = ["drug_raw", "fei"]
    vfei["fei"] = vfei["fei"].astype(int)
    vfei["drug_raw"] = vfei["drug_raw"].astype(str).str.strip()
    def _match(name: str) -> str:
        nl = name.lower()
        for k, v in sorted(TARGET_DRUGS.items(), key=lambda x: -len(x[0])):
            if k in nl:
                return v
        return name.strip()
    vfei["drug"] = vfei["drug_raw"].map(_match)
    return vfei[["fei", "drug"]].drop_duplicates()


def _load_quality_ts() -> pd.DataFrame:
    if not TEXT_TIMESERIES_REDICA_CSV.exists():
        log.warning("Quality timeseries not found: %s", TEXT_TIMESERIES_REDICA_CSV)
        return pd.DataFrame(columns=["fei", "snapshot_date"] + list(SIGNAL_COLS))
    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"]           = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    ts = ts.dropna(subset=["fei", "snapshot_date"])
    ts = ts[(ts["snapshot_date"] >= DATE_MIN) & (ts["snapshot_date"] <= DATE_MAX)]
    keep = ["fei", "snapshot_date"] + [c for c in SIGNAL_COLS if c in ts.columns]
    return ts[keep]


def _load_inspections(valisure_feis: set) -> pd.DataFrame:
    insp = pd.read_excel(INSP_FILE, usecols=[
        "FEI Number", "Legal Name", "City", "Country/Area",
        "Inspection End Date", "Classification",
    ])
    insp.columns = ["fei", "name", "city", "country", "insp_date", "classification"]
    insp["fei"]            = pd.to_numeric(insp["fei"], errors="coerce").astype("Int64")
    insp["insp_date"]      = pd.to_datetime(insp["insp_date"], errors="coerce")
    insp["classification"] = insp["classification"].map(_norm_insp_class)
    insp = insp[insp["fei"].isin(valisure_feis) & insp["insp_date"].notna()]
    insp = insp[(insp["insp_date"] >= DATE_MIN) & (insp["insp_date"] <= DATE_MAX)]
    return insp.reset_index(drop=True)


def _load_recalls(valisure_feis: set) -> pd.DataFrame:
    r = pd.read_csv(RECALL_FILT, low_memory=False)
    r.columns = [c.strip() for c in r.columns]
    r["fei"]         = pd.to_numeric(r["FEI Number"], errors="coerce").astype("Int64")
    r["recall_date"] = pd.to_datetime(r["Recall_Date"], errors="coerce")
    r = r[r["fei"].isin(valisure_feis) & r["recall_date"].notna()]
    r = r[(r["recall_date"] >= DATE_MIN) & (r["recall_date"] <= DATE_MAX)]
    r["class_norm"]    = r["Event Classification"].astype(str).str.strip()
    r["product_short"] = r["Product Description"].astype(str).str[:80]
    return r[["fei", "recall_date", "class_norm", "product_short", "Reason for Recall"]].reset_index(drop=True)


def _load_shortages() -> dict[str, list[dict]]:
    sh = pd.read_excel(UUTAH_FILE, header=1)
    sh.columns = [c.strip() for c in sh.columns]
    sh = sh.rename(columns={"Drug Shortages": "drug_raw", "Date Notified": "start",
                             "Date Resolved": "end", "Reason": "reason"})
    sh["drug_raw"] = sh["drug_raw"].astype(str).str.lower()
    sh["start"]    = pd.to_datetime(sh["start"], errors="coerce")
    sh["end"]      = pd.to_datetime(sh["end"],   errors="coerce")
    result: dict = {}
    for key, display in TARGET_DRUGS.items():
        sub = sh[sh["drug_raw"].str.contains(key, na=False)].dropna(subset=["start"])
        sub = sub.copy()
        sub["end"] = sub["end"].fillna(pd.Timestamp(DATE_MAX))
        sub = sub[(sub["start"] >= DATE_MIN) & (sub["start"] <= DATE_MAX)]
        rows = [{"start": r["start"].strftime("%Y-%m-%d"),
                 "end":   r["end"].strftime("%Y-%m-%d"),
                 "reason": str(r.get("reason","")).strip() or "Unknown"}
                for _, r in sub.iterrows()]
        if rows:
            result[display] = rows
    return result


def _load_ndc_labelers() -> pd.DataFrame:
    ndc = pd.read_csv(NDC_PRODUCT, sep=",", encoding="latin1", dtype=str, on_bad_lines="skip")
    ndc["labeler_code"] = ndc["PRODUCTNDC"].str.split("-").str[0].str.zfill(5)
    ndc["labeler_name"] = ndc["LABELERNAME"].astype(str).str.strip().str.lower()
    return ndc[["labeler_code", "labeler_name"]].drop_duplicates()


def _build_fei_labeler_map(insp: pd.DataFrame, labelers: pd.DataFrame) -> dict[int, list[str]]:
    """Map FEI → labeler codes via firm name keyword matching."""
    first = insp.sort_values("insp_date").groupby("fei").first().reset_index()
    result: dict[int, list[str]] = {}
    for _, row in first.iterrows():
        fei  = int(row["fei"])
        name = str(row.get("name", "")).strip().lower()
        words = [w for w in re.findall(r"\b[a-z]{4,}\b", name) if w not in SKIP_WORDS]
        if not words:
            continue
        for word in words[:3]:
            matches = labelers[labelers["labeler_name"].str.contains(word, na=False, regex=False)]
            if 0 < len(matches) <= 20:
                result[fei] = matches["labeler_code"].unique().tolist()
                break
    return result


def _load_sdud_by_labeler(fdmap: pd.DataFrame, fei_labeler: dict[int, list[str]]) -> dict:
    """Returns {drug: {fei_str: [{date, units}, ...]}}"""
    if not SDUD_PANEL.exists():
        log.warning("SDUD panel not found: %s", SDUD_PANEL)
        return {}
    sdud = pd.read_csv(SDUD_PANEL, low_memory=False,
                       usecols=["ndc11", "date", "drug_name", "sdud_units_reimbursed"])
    sdud["labeler_code"] = sdud["ndc11"].astype(str).str.zfill(11).str[:5]
    sdud["date"]         = pd.to_datetime(sdud["date"], errors="coerce")
    sdud = sdud.dropna(subset=["date"])
    sdud = sdud[(sdud["date"] >= DATE_MIN) & (sdud["date"] <= DATE_MAX)]
    sdud["sdud_units_reimbursed"] = pd.to_numeric(sdud["sdud_units_reimbursed"], errors="coerce").fillna(0)

    # For each drug, pre-aggregate by labeler × date
    result: dict = {}
    for drug, grp in fdmap.groupby("drug"):
        drug_feis = grp["fei"].unique().tolist()
        drug_sdud = sdud[sdud["drug_name"] == drug]
        if drug_sdud.empty:
            continue
        drug_dict: dict = {}
        for fei in drug_feis:
            fei_int = int(fei)
            lab_codes = fei_labeler.get(fei_int, [])
            if not lab_codes:
                continue
            sub = drug_sdud[drug_sdud["labeler_code"].isin(lab_codes)]
            if sub.empty:
                continue
            monthly = sub.groupby("date")["sdud_units_reimbursed"].sum().reset_index()
            monthly = monthly.sort_values("date")
            rows = [{"date": r["date"].strftime("%Y-%m-%d"), "units": round(r["sdud_units_reimbursed"])}
                    for _, r in monthly.iterrows()]
            if rows:
                drug_dict[str(fei_int)] = rows
        if drug_dict:
            result[drug] = drug_dict
    log.info("SDUD data: %d drugs, %d FEI-drug pairs",
             len(result), sum(len(v) for v in result.values()))
    return result


def _fei_name_map(insp: pd.DataFrame) -> dict[int, str]:
    first = insp.sort_values("insp_date").groupby("fei").first().reset_index()
    out: dict[int, str] = {}
    for _, row in first.iterrows():
        parts = [str(row["name"]).strip()]
        if str(row["city"]).strip() not in ("-", "nan", ""):
            parts.append(str(row["city"]).strip())
        if str(row["country"]).strip() not in ("-", "nan", ""):
            parts.append(str(row["country"]).strip())
        out[int(row["fei"])] = ", ".join(parts)
    return out


# ── Build per-FEI JSON payload ────────────────────────────────────────────────

def _build_fei_data(
    fdmap:       pd.DataFrame,
    quality:     pd.DataFrame,
    inspections: pd.DataFrame,
    recalls:     pd.DataFrame,
    shortages:   dict,
    name_map:    dict[int, str],
    sdud_data:   dict,
) -> dict:
    drugs_out: dict = {}
    for drug, grp in fdmap.groupby("drug"):
        drug_dict: dict = {}
        for fei in sorted(grp["fei"].unique()):
            fei_int = int(fei)
            q_fei = quality[quality["fei"] == fei_int].sort_values("snapshot_date")
            q_dates   = q_fei["snapshot_date"].dt.strftime("%Y-%m-%d").tolist()
            q_signals = {col: q_fei[col].fillna(0).round(4).tolist()
                         for col in SIGNAL_COLS if col in q_fei.columns}
            i_fei = inspections[inspections["fei"] == fei_int].sort_values("insp_date")
            i_rows = [{"date": r["insp_date"].strftime("%Y-%m-%d"), "class": str(r["classification"])}
                      for _, r in i_fei.iterrows()]
            r_fei = recalls[recalls["fei"] == fei_int].sort_values("recall_date")
            r_rows = [{"date": r["recall_date"].strftime("%Y-%m-%d"), "class": str(r["class_norm"]),
                       "product": str(r["product_short"]), "reason": str(r["Reason for Recall"])[:100]}
                      for _, r in r_fei.iterrows()]
            n_oai = sum(1 for r in i_rows if "OAI" in r["class"])
            n_vai = sum(1 for r in i_rows if "VAI" in r["class"])
            n_nai = sum(1 for r in i_rows if "NAI" in r["class"])
            sdud_rows = sdud_data.get(drug, {}).get(str(fei_int), [])
            drug_dict[str(fei_int)] = {
                "name":        name_map.get(fei_int, f"FEI {fei_int}"),
                "stats":       {"n_oai": n_oai, "n_vai": n_vai, "n_nai": n_nai,
                                 "n_recalls": len(r_rows), "has_sdud": len(sdud_rows) > 0},
                "q_dates":     q_dates,
                "q_signals":   q_signals,
                "inspections": i_rows,
                "recalls":     r_rows,
                "sdud":        sdud_rows,
            }
        drugs_out[drug] = drug_dict
    return {"drugs": drugs_out, "shortages": shortages}


# ── HTML ──────────────────────────────────────────────────────────────────────

def build_html(fei_data: dict) -> str:
    data_json        = json.dumps(fei_data, default=str, ensure_ascii=False)
    signal_labels_js = json.dumps({k: v[0] for k, v in SIGNAL_COLS.items()})
    signal_colors_js = json.dumps({k: v[1] for k, v in SIGNAL_COLS.items()})
    class_colors_js  = json.dumps(CLASS_COLORS)
    insp_colors_js   = json.dumps(INSP_COLORS)
    drugs_sorted     = json.dumps(sorted(fei_data["drugs"].keys()))
    plotly_js        = PLOTLY_JS_PATH.read_text(encoding="utf-8")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FEI Facility Explorer — Drug Shortage Project</title>
<script>{plotly_js}</script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:#F5F7FA;color:#1A1A2E}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px 20px}}
h1{{font-size:1.5rem;font-weight:700;margin-bottom:4px}}
.subtitle{{font-size:.9rem;color:#5A6475;margin-bottom:20px}}
.drug-bar{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}}
.drug-btn{{padding:6px 14px;border-radius:20px;border:1.5px solid #C8D0DC;background:white;
           font-size:.85rem;cursor:pointer;font-weight:500;color:#3A4560;transition:all .15s}}
.drug-btn:hover{{border-color:#1A4480;color:#1A4480}}
.drug-btn.active{{background:#1A4480;border-color:#1A4480;color:white;font-weight:600}}
.fei-row{{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}}
.fei-row label{{font-size:.9rem;font-weight:600;color:#3A4560;white-space:nowrap}}
#fei-select{{flex:1;max-width:540px;padding:7px 11px;border-radius:6px;border:1.5px solid #C8D0DC;
             font-size:.87rem;background:white;color:#1A1A2E;cursor:pointer}}
.stats-card{{background:white;border-radius:10px;padding:14px 18px;border:1px solid #E2E8F0;
             margin-bottom:18px;display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start}}
.stat-block{{display:flex;flex-direction:column}}
.stat-label{{font-size:.72rem;font-weight:600;color:#7A8499;text-transform:uppercase;letter-spacing:.04em}}
.stat-value{{font-size:1.3rem;font-weight:700}}
.oai{{color:#C41E3A}}.vai{{color:#E57200}}.nai{{color:#2E7D32}}
.recall-stat{{color:#7B2D8B}}.sdud-stat{{color:#1A4480}}
.fei-name-label{{font-size:.95rem;font-weight:600;color:#1A4480;flex:1 0 100%}}
.chart-card{{background:white;border-radius:10px;padding:16px;border:1px solid #E2E8F0;margin-bottom:16px}}
.chart-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px}}
.chart-title{{font-size:.9rem;font-weight:600;color:#3A4560}}
.sig-select{{padding:5px 9px;border-radius:5px;border:1px solid #C8D0DC;font-size:.82rem;background:white;cursor:pointer}}
.note{{font-size:.75rem;color:#9AA3B2;margin-top:5px}}
.empty-msg{{text-align:center;color:#9AA3B2;font-size:.9rem;padding:48px 0;display:none}}
</style>
</head>
<body>
<div class="wrap">
  <h1>FEI Facility Explorer</h1>
  <p class="subtitle">Select a drug and a manufacturing facility to trace its quality signals,
     inspection history, recalls, and Medicaid market volume.</p>

  <div class="drug-bar" id="drug-bar"></div>

  <div class="fei-row">
    <label>Facility (FEI):</label>
    <select id="fei-select"><option>— select drug first —</option></select>
  </div>

  <div class="stats-card" id="stats-card">
    <div class="fei-name-label">Select a drug and facility above</div>
  </div>

  <!-- Quality signals -->
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Quality signals (% of 483 observations flagged)</span>
      <select class="sig-select" id="sig-select" onchange="redrawQuality()">
        <option value="all">All signals</option>
      </select>
    </div>
    <div id="chart-quality"></div>
    <div class="empty-msg" id="quality-empty">No quality signal data for this facility</div>
    <p class="note">Vertical dotted lines = inspections (red=OAI · orange=VAI · green=NAI).
       Purple shading = shortage periods (UUtah database).</p>
  </div>

  <!-- Recalls -->
  <div class="chart-card">
    <div class="chart-title">Recall events</div>
    <div id="chart-recalls"></div>
    <div class="empty-msg" id="recalls-empty">No recalls linked to this facility in the study period</div>
    <p class="note">Bars colored by recall class (I = most severe). Purple shading = shortage periods.</p>
  </div>

  <!-- SDUD volume -->
  <div class="chart-card">
    <div class="chart-title">Manufacturer's Medicaid volume (SDUD)</div>
    <div id="chart-sdud"></div>
    <div class="empty-msg" id="sdud-empty">No SDUD data found — labeler code not matched for this facility</div>
    <p class="note">Monthly Medicaid units dispensed attributed to this facility's manufacturer
       (NDC labeler code matched by firm name). Volume reaching zero = market exit.</p>
  </div>

</div><!-- /wrap -->

<script>
// ── Data & constants ──────────────────────────────────────────────────────
const FEI_DATA     = {data_json};
const SIG_LABELS   = {signal_labels_js};
const SIG_COLORS   = {signal_colors_js};
const CLASS_COLORS = {class_colors_js};
const INSP_COLORS  = {insp_colors_js};
const DRUGS        = {drugs_sorted};

let activeDrug   = null;
let activeFeiKey = null;
let activeInfo   = null;

// ── Drug buttons ──────────────────────────────────────────────────────────
(function buildDrugBar() {{
  const bar = document.getElementById('drug-bar');
  DRUGS.forEach(function(drug) {{
    var btn = document.createElement('button');
    btn.className = 'drug-btn';
    btn.textContent = drug;
    btn.addEventListener('click', function() {{ selectDrug(drug); }});
    bar.appendChild(btn);
  }});
}})();

function selectDrug(drug) {{
  activeDrug = drug;
  document.querySelectorAll('.drug-btn').forEach(function(b) {{
    b.classList.toggle('active', b.textContent === drug);
  }});
  buildFeiDropdown(drug);
}}

// ── FEI dropdown ──────────────────────────────────────────────────────────
function buildFeiDropdown(drug) {{
  var sel  = document.getElementById('fei-select');
  var feis = (FEI_DATA.drugs[drug] || {{}});
  sel.innerHTML = '';
  Object.keys(feis).sort().forEach(function(fei) {{
    var info = feis[fei];
    var opt  = document.createElement('option');
    opt.value = fei;
    var oaiTag = info.stats.n_oai > 0 ? ' ⚠ ' + info.stats.n_oai + ' OAI' : '';
    var sdudTag = info.stats.has_sdud ? ' ● SDUD' : '';
    opt.textContent = fei + ' — ' + info.name + oaiTag + sdudTag;
    sel.appendChild(opt);
  }});
  sel.onchange = function() {{ selectFei(sel.value); }};
  if (sel.options.length > 0) selectFei(sel.options[0].value);
}}

// ── FEI selection ─────────────────────────────────────────────────────────
function selectFei(feiKey) {{
  activeFeiKey = feiKey;
  activeInfo   = (FEI_DATA.drugs[activeDrug] || {{}})[feiKey];
  if (!activeInfo) return;
  renderStats(activeInfo);
  populateSigSelect(activeInfo);
  redrawQuality();
  renderRecalls(activeInfo);
  renderSdud(activeInfo);
}}

// ── Stats card ────────────────────────────────────────────────────────────
function renderStats(info) {{
  var s = info.stats;
  document.getElementById('stats-card').innerHTML =
    '<div class="fei-name-label">' + info.name + '</div>' +
    '<div class="stat-block"><span class="stat-label">OAI</span>' +
      '<span class="stat-value oai">' + s.n_oai + '</span></div>' +
    '<div class="stat-block"><span class="stat-label">VAI</span>' +
      '<span class="stat-value vai">' + s.n_vai + '</span></div>' +
    '<div class="stat-block"><span class="stat-label">NAI</span>' +
      '<span class="stat-value nai">' + s.n_nai + '</span></div>' +
    '<div class="stat-block"><span class="stat-label">Recalls</span>' +
      '<span class="stat-value recall-stat">' + s.n_recalls + '</span></div>' +
    (s.has_sdud ? '<div class="stat-block"><span class="stat-label">SDUD</span>' +
      '<span class="stat-value sdud-stat">available</span></div>' : '');
}}

// ── Signal selector ───────────────────────────────────────────────────────
function populateSigSelect(info) {{
  var sel = document.getElementById('sig-select');
  sel.innerHTML = '<option value="all">All signals</option>';
  Object.keys(SIG_LABELS).forEach(function(col) {{
    var vals = (info.q_signals || {{}})[col];
    if (!vals || vals.length === 0) return;
    var opt = document.createElement('option');
    opt.value = col;
    opt.textContent = SIG_LABELS[col];
    sel.appendChild(opt);
  }});
}}

function redrawQuality() {{
  if (activeInfo) renderQuality(activeInfo);
}}

// ── Shortage shapes ───────────────────────────────────────────────────────
function shortageShapes() {{
  var periods = ((FEI_DATA.shortages || {{}})[activeDrug] || []);
  return periods.map(function(p) {{
    return {{
      type:'rect', xref:'x', yref:'paper',
      x0:p.start, x1:p.end, y0:0, y1:1,
      fillcolor:'rgba(123,45,139,0.07)', line:{{width:0}}, layer:'below'
    }};
  }});
}}

// ── Inspection shapes ─────────────────────────────────────────────────────
function inspShapes(inspections) {{
  return (inspections || []).map(function(i) {{
    var col = INSP_COLORS[i.class] || '#888888';
    return {{
      type:'line', xref:'x', yref:'paper',
      x0:i.date, x1:i.date, y0:0, y1:1,
      line:{{color:col+'AA', width:1.5, dash:'dot'}}, layer:'above'
    }};
  }});
}}

// ── Base layout ───────────────────────────────────────────────────────────
function baseLayout(height, yTitle, extraShapes) {{
  return {{
    height: height,
    margin: {{t:10, r:20, l:60, b:50}},
    plot_bgcolor:'#FAFBFC', paper_bgcolor:'white',
    xaxis: {{type:'date', range:['{DATE_MIN}','{DATE_MAX}'],
             showgrid:true, gridcolor:'#EEF0F4'}},
    yaxis: {{title:yTitle, showgrid:true, gridcolor:'#EEF0F4'}},
    legend: {{orientation:'h', y:-0.25, x:0, font:{{size:11}}}},
    shapes: (extraShapes || []),
    hovermode:'x unified',
  }};
}}

// ── Quality chart ─────────────────────────────────────────────────────────
function renderQuality(info) {{
  var div   = document.getElementById('chart-quality');
  var empty = document.getElementById('quality-empty');

  var selectedSig = document.getElementById('sig-select').value;
  var sigKeys = selectedSig === 'all' ? Object.keys(SIG_LABELS) : [selectedSig];

  var traces = [];
  sigKeys.forEach(function(col) {{
    var vals = (info.q_signals || {{}})[col];
    if (!vals || vals.length === 0) return;
    traces.push({{
      type:'scatter', mode:'lines+markers',
      x: info.q_dates, y: vals,
      name: SIG_LABELS[col],
      line:  {{color: SIG_COLORS[col], width:2.5}},
      marker:{{size:8}},
      hovertemplate:'<b>%{{x}}</b><br>' + SIG_LABELS[col] + ': %{{y:.0%}}<extra></extra>',
    }});
  }});

  // Inspection markers as scatter at y=1.08
  var byClass = {{}};
  (info.inspections || []).forEach(function(i) {{
    if (!byClass[i.class]) byClass[i.class] = [];
    byClass[i.class].push(i.date);
  }});
  Object.keys(byClass).forEach(function(cls) {{
    var dates = byClass[cls];
    var col   = INSP_COLORS[cls] || '#888888';
    traces.push({{
      type:'scatter', mode:'markers',
      x:dates, y:Array(dates.length).fill(1.08),
      name:cls,
      marker:{{color:col, size:11, symbol:'diamond'}},
      hovertemplate:'<b>%{{x}}</b><br>Inspection: ' + cls + '<extra></extra>',
    }});
  }});

  if (traces.length === 0) {{
    div.style.display   = 'none';
    empty.style.display = 'block';
    return;
  }}
  div.style.display   = 'block';
  empty.style.display = 'none';

  var shapes = shortageShapes().concat(inspShapes(info.inspections));
  var layout = baseLayout(340, '% observations flagged', shapes);
  layout.yaxis.tickformat = '.0%';
  layout.yaxis.range = [0, 1.18];

  Plotly.react(div, traces, layout, {{responsive:true, displayModeBar:false}});
}}

// ── Recalls chart ─────────────────────────────────────────────────────────
function renderRecalls(info) {{
  var div   = document.getElementById('chart-recalls');
  var empty = document.getElementById('recalls-empty');
  var rows  = info.recalls || [];

  if (rows.length === 0) {{
    div.style.display   = 'none';
    empty.style.display = 'block';
    return;
  }}
  div.style.display   = 'block';
  empty.style.display = 'none';

  var byClass = {{}};
  rows.forEach(function(r) {{
    if (!byClass[r.class]) byClass[r.class] = [];
    byClass[r.class].push(r);
  }});

  var traces = Object.keys(byClass).map(function(cls) {{
    var items = byClass[cls];
    return {{
      type:'bar',
      x: items.map(function(r){{return r.date;}}),
      y: Array(items.length).fill(1),
      name: cls,
      marker: {{color: CLASS_COLORS[cls] || '#888888', opacity:0.85}},
      customdata: items.map(function(r){{return [r.product, r.reason];}}),
      hovertemplate:'<b>%{{x}}</b> — ' + cls + '<br>%{{customdata[0]}}<br><i>%{{customdata[1]}}</i><extra></extra>',
    }};
  }});

  var shapes  = shortageShapes();
  var layout  = baseLayout(180, 'Recall events', shapes);
  layout.barmode = 'stack';
  layout.yaxis.showticklabels = false;

  Plotly.react(div, traces, layout, {{responsive:true, displayModeBar:false}});
}}

// ── SDUD chart ────────────────────────────────────────────────────────────
function renderSdud(info) {{
  var div   = document.getElementById('chart-sdud');
  var empty = document.getElementById('sdud-empty');
  var rows  = info.sdud || [];

  if (rows.length === 0) {{
    div.style.display   = 'none';
    empty.style.display = 'block';
    return;
  }}
  div.style.display   = 'block';
  empty.style.display = 'none';

  var dates = rows.map(function(r){{return r.date;}});
  var units = rows.map(function(r){{return r.units;}});

  var maxUnits = Math.max.apply(null, units);
  var scaleM   = maxUnits > 1e6;
  var displayY = units.map(function(u){{return scaleM ? u/1e6 : u/1e3;}});
  var yTitle   = scaleM ? 'Units dispensed (millions)' : 'Units dispensed (thousands)';

  // Find first zero after peak
  var peakIdx = units.indexOf(maxUnits);
  var exitAnnot = null;
  for (var i = peakIdx; i < units.length; i++) {{
    if (units[i] === 0) {{
      exitAnnot = dates[i];
      break;
    }}
  }}
  // Or find first near-zero (< 1% of peak)
  if (!exitAnnot) {{
    for (var i = peakIdx; i < units.length; i++) {{
      if (units[i] < maxUnits * 0.01) {{
        exitAnnot = dates[i];
        break;
      }}
    }}
  }}

  var traces = [{{
    type:'scatter', mode:'lines',
    x: dates, y: displayY,
    name:'Medicaid units',
    fill:'tozeroy',
    line:  {{color:'#1A4480', width:2}},
    fillcolor:'rgba(26,68,128,0.10)',
    hovertemplate:'<b>%{{x}}</b><br>Units: %{{y:,.1f}}' + (scaleM?'M':'K') + '<extra></extra>',
  }}];

  var shapes  = shortageShapes();
  var layout  = baseLayout(220, yTitle, shapes);

  if (exitAnnot) {{
    layout.annotations = [{{
      x: exitAnnot, y: 0, xref:'x', yref:'y',
      text: 'Market exit', showarrow:true,
      arrowhead:2, arrowcolor:'#C41E3A', font:{{color:'#C41E3A', size:11}},
      ay: -40
    }}];
  }}

  Plotly.react(div, traces, layout, {{responsive:true, displayModeBar:false}});
}}

// ── Init ──────────────────────────────────────────────────────────────────
if (DRUGS.length > 0) selectDrug(DRUGS[0]);
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading data…")
    fdmap          = _load_fei_drug_map()
    valisure_feis  = set(fdmap["fei"].unique())
    quality        = _load_quality_ts()
    inspections    = _load_inspections(valisure_feis)
    recalls        = _load_recalls(valisure_feis)
    shortages      = _load_shortages()
    name_map       = _fei_name_map(inspections)
    labelers       = _load_ndc_labelers()
    fei_labeler    = _build_fei_labeler_map(inspections, labelers)
    sdud_data      = _load_sdud_by_labeler(fdmap, fei_labeler)

    log.info("FEI-drug pairs: %d | with quality: %d | with SDUD: %d",
             len(fdmap), quality["fei"].nunique(),
             sum(len(v) for v in sdud_data.values()))

    fei_data = _build_fei_data(fdmap, quality, inspections, recalls,
                               shortages, name_map, sdud_data)
    n_drugs = len(fei_data["drugs"])
    n_feis  = sum(len(v) for v in fei_data["drugs"].values())
    log.info("Payload: %d drugs, %d FEIs", n_drugs, n_feis)

    log.info("Building HTML (embedding plotly.js inline)…")
    html = build_html(fei_data)
    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


if __name__ == "__main__":
    main()
