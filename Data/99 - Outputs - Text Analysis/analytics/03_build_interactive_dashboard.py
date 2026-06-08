"""
03_build_interactive_dashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Generates fei_dashboard.html — the main interactive research output.
  Self-contained single HTML file; open directly in Chrome or Firefox.

  LEFT panel (65%)  : vis.js network — all 129 FEI nodes, colored by severity.
  RIGHT panel (35%) : 3-tab detail panel on node click:
    Tab 1 — Overview   : SVG event timeline (zoom/pan) + 483 regex signal badges
    Tab 2 — Events     : full scrollable event table with inspection detail columns
    Tab 3 — CFR        : per-FEI CFR frequency bars, domain breakdown, co-occurrence

  If scripts 04-07 (LLM pipeline) have been run, a 4th "Risk Signals" tab appears
  automatically showing the Text Risk Index and per-observation LLM signal cards.

WHEN TO RUN
  Run after 01_build_combined_dataset.py (required).
  Also re-run after 06_aggregate_score.py if you want the LLM Risk Signals tab.

REQUIRED FOR COMBINED DATASET?  YES — primary research visualization.

INPUTS (required — produced by 01)
  fei_events_timeline.csv
  fei_node_summary.csv
  fei_edge_list.csv
  fei_cfr_data.json

INPUTS (optional — enriches the dashboard if present)
  Data/12 - FDA - 483/processed/483_fei_features.csv   → regex signal badges (Overview tab)
  Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv
  483_fei_context_features.csv          → LLM Text Risk Index (produced by 05)
  483_observation_context_signals.csv   → per-observation LLM cards (produced by 04)

OUTPUTS
  fei_dashboard.html  — open in Chrome / Firefox / Safari (no server needed)

DEPENDENCIES
  pip install pandas openpyxl
  JavaScript libs (vis.js, etc.) are bundled in lib/ or loaded from CDN.
"""

import json
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parents[2]
OUT     = Path(__file__).parent

EVENTS_CSV   = OUT / "fei_events_timeline.csv"
NODES_CSV    = OUT / "fei_node_summary.csv"
EDGES_CSV    = OUT / "fei_edge_list.csv"
CFR_JSON     = OUT / "fei_cfr_data.json"
SIGNALS_483  = BASE / "Data/12 - FDA - 483/processed/483_fei_features.csv"
WL_REC_CSV   = BASE / "Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv"
VALISURE     = BASE / "Data/08 - Valisure/raw/FEIs_March 2026.xlsx"
RISK_CSV     = OUT / "483_fei_context_features.csv"          # produced by 05
OBS_SIGNALS  = OUT / "483_observation_context_signals.csv"   # produced by 04
HTML_OUT     = OUT / "fei_dashboard.html"

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")
events_df = pd.read_csv(EVENTS_CSV, parse_dates=["event_date"])
nodes_df  = pd.read_csv(NODES_CSV)
edges_df  = pd.read_csv(EDGES_CSV)

events_df["fei"] = events_df["fei"].astype(int)
nodes_df["fei"]  = nodes_df["fei"].astype(int)

# API mapping from Valisure
valisure = pd.read_excel(VALISURE, sheet_name="API Only_FEI Mapping")
api_map  = valisure.groupby("FEI_NUMBER")["API"].apply(
    lambda x: list(set(x.dropna()))
).to_dict()

# 483 text signals
sig483 = {}
if SIGNALS_483.exists():
    df483 = pd.read_csv(SIGNALS_483)
    df483["fei"] = pd.to_numeric(df483["fei"], errors="coerce").astype("Int64")
    for _, r in df483.iterrows():
        fei = str(int(r["fei"]))
        sig483[fei] = {
            "n_483s":              int(r.get("n_483s_total", 0)),
            "n_obs":               int(r.get("n_observations_total", 0)),
            "avg_chars":           int(r.get("avg_obs_body_chars", 0)),
            "ever_repeat":         bool(r.get("ever_repeat", False)),
            "ever_data_integrity": bool(r.get("ever_data_integrity", False)),
            "ever_contamination":  bool(r.get("ever_contamination", False)),
            "ever_systemic":       bool(r.get("ever_systemic", False)),
            "ever_oos_oot":        bool(r.get("ever_oos_oot", False)),
            "ever_patient_risk":   bool(r.get("ever_patient_risk", False)),
            "ever_wl_ref":         bool(r.get("ever_wl_ref", False)),
        }

# WL text signals
sigWL = {}
if WL_REC_CSV.exists():
    wl_df = pd.read_csv(WL_REC_CSV)
    fei_col = "search_fei" if "search_fei" in wl_df.columns else "primary_fei"
    wl_df["_fei"] = pd.to_numeric(wl_df[fei_col], errors="coerce").astype("Int64")
    for fei_val, grp in wl_df.groupby("_fei"):
        fei = str(int(fei_val))
        sigWL[fei] = {
            "n_wls":             len(grp),
            "n_violations":      int(grp["n_violations"].sum()),
            "n_cfr_unique":      int(grp["n_cfr_unique"].sum()),
            "cfr_list":          "; ".join(
                                    sorted(set(
                                        c for s in grp["cfr_list"].fillna("").tolist()
                                        for c in s.split("; ") if c.strip()
                                    ))[:6]
                                 ),
            "repeat_facility":   bool(grp["has_repeat_at_facility"].any()),
            "repeat_multi":      bool(grp["has_repeat_multi_site"].any()),
            "mgmt_oversight":    bool(grp["has_management_oversight"].any()),
            "corp_failure":      bool(grp["has_corporate_failure_lang"].any()),
            "n_prior_wl_refs":   int(grp["n_prior_wl_refs"].sum()),
            "n_repeat_sections": int(grp["n_repeat_sections"].sum()),
        }

# CFR data
cfr_data = {}
if CFR_JSON.exists():
    with open(CFR_JSON) as f:
        cfr_data = json.load(f)
    print(f"  CFR data: {len(cfr_data)} FEIs with citation records")
else:
    print("  CFR data not found — run 01_build_combined_dataset.py first")

print(f"  483 text signals: {len(sig483)} FEIs")
print(f"  WL  text signals: {len(sigWL)}  FEIs")

# LLM risk signals (Phase 3 output — optional, graceful if missing)
sigRisk = {}
obs_by_fei: dict[str, list] = {}
if RISK_CSV.exists() and OBS_SIGNALS.exists():
    risk_df = pd.read_csv(RISK_CSV)
    risk_df["fei"] = pd.to_numeric(risk_df["fei"], errors="coerce").astype("Int64")
    for _, r in risk_df.iterrows():
        fei = str(int(r["fei"]))
        sigRisk[fei] = {
            "n_obs_scored":              int(r.get("n_obs_scored",  0)),
            "n_483_obs":                 int(r.get("n_483_obs",     0)),
            "n_wl_obs":                  int(r.get("n_wl_obs",      0)),
            "severity_high_share":       float(r.get("severity_high_share",    0)),
            "severity_mod_share":        float(r.get("severity_mod_share",     0)),
            "severity_low_share":        float(r.get("severity_low_share",     0)),
            "dominant_root_cause":       str(r.get("dominant_root_cause",      "Unclear")),
            "capital_share":             float(r.get("capital_share",           0)),
            "cultural_share":            float(r.get("cultural_share",          0)),
            "mixed_share":               float(r.get("mixed_share",             0)),
            "unclear_share":             float(r.get("unclear_share",           0)),
            "remediation_strong_share":  float(r.get("remediation_strong_share",0)),
            "remediation_partial_share": float(r.get("remediation_partial_share",0)),
            "remediation_weak_share":    float(r.get("remediation_weak_share",  0)),
            "remediation_none_share":    float(r.get("remediation_none_share",  0)),
            "repeat_flag_share":         float(r.get("repeat_llm_share",        0)),
            "systemic_flag_share":       float(r.get("systemic_llm_share",      0)),
            "patient_risk_share":        float(r.get("patient_risk_llm_share",  0)),
            "dominant_violation_category": str(r.get("dominant_violation_category","Other")),
            "mean_confidence":           float(r.get("mean_confidence",          0)),
            "text_risk_index":           float(r.get("text_risk_index",          0)),
        }

    obs_df = pd.read_csv(OBS_SIGNALS)
    obs_df["fei"] = pd.to_numeric(obs_df["fei"], errors="coerce").astype("Int64")
    for fei_val, grp in obs_df.groupby("fei"):
        fei = str(int(fei_val))
        obs_list = []
        for _, o in grp.sort_values("confidence", ascending=False).head(25).iterrows():
            obs_list.append({
                "obs_id":    str(o.get("obs_num", ""))[-40:],
                "src":       "483",
                "cat":       str(o.get("violation_category", "Other")),
                "sev":       str(o.get("severity_tier", "Low")),
                "rc":        str(o.get("root_cause_type", "Unclear")),
                "rem":       str(o.get("remediation_signal", "None")),
                "repeat":    bool(str(o.get("repeat_flag_llm", "false")).lower() == "true"),
                "systemic":  bool(str(o.get("systemic_flag_llm", "false")).lower() == "true"),
                "patient":   bool(str(o.get("patient_risk_flag_llm", "false")).lower() == "true"),
                "quote":     str(o.get("evidence_quote", ""))[:250],
                "conf":      float(o.get("confidence", 0)),
            })
        obs_by_fei[fei] = obs_list

    print(f"  LLM risk signals: {len(sigRisk)} FEIs  |  obs cards: {sum(len(v) for v in obs_by_fei.values())}")
elif RISK_CSV.exists():
    print("  [INFO] 483_fei_context_features.csv found but 483_observation_context_signals.csv missing — no obs cards")
else:
    print("  LLM risk signals: not found (run 04→05 pipeline to generate)")


# ══════════════════════════════════════════════════════════════════════════
# 2. BUILD vis.js DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════
OUTCOME_COLORS = {
    "Warning Letter":       "#8B0000",
    "OAI":                  "#C0392B",
    "Class I Recall":       "#7D3C98",
    "VAI":                  "#E67E22",
    "NAI":                  "#27AE60",
    "Import Refusal Only":  "#1A5276",
    "No Regulatory Events": "#AEB6BF",
}

sev_min = nodes_df["severity_score"].min()
sev_max = max(nodes_df["severity_score"].max(), 1)

def scale_size(s, lo=14, hi=52):
    return round(lo + (s - sev_min) / (sev_max - sev_min) * (hi - lo), 1)

nodes_list = []
fei_info   = {}
for _, r in nodes_df.iterrows():
    fei   = int(r["fei"])
    firm  = str(r["firm_name"])
    ctry  = str(r["country"])
    worst = str(r["worst_outcome"])
    color = OUTCOME_COLORS.get(worst, "#AEB6BF")
    size  = scale_size(float(r["severity_score"]))
    apis  = api_map.get(fei, [])

    hover_text = f"FEI {fei}\n{firm}\n{ctry}"

    nodes_list.append({
        "id":    fei,
        "label": firm[:16] + ("…" if len(firm) > 16 else ""),
        "title": hover_text,
        "color": {
            "background": color,
            "border":     "#ffffff",
            "highlight":  {"background": color, "border": "#FFD700"},
            "hover":      {"background": color, "border": "#FFD700"},
        },
        "size":  size,
        "shape": "dot",
        "font":  {"color": "#ffffff", "size": 9, "face": "Arial",
                  "strokeWidth": 2, "strokeColor": color},
    })

    fei_info[str(fei)] = {
        "firm":         firm,
        "country":      ctry,
        "worst":        worst,
        "apis":         apis,
        "n_insp":       int(r["n_inspections"]),
        "n_oai":        int(r["n_oai"]),
        "n_vai":        int(r["n_vai"]),
        "n_nai":        int(r["n_nai"]),
        "n_483":        int(r["n_483s"]),
        "n_wl":         int(r["n_warning_letters"]),
        "n_recall":     int(r["n_recalls"]),
        "n_class1":     int(r["n_class_I_recalls"]),
        "n_refusal":    int(r["n_import_refusals"]),
        "severity":     round(float(r["severity_score"]), 1),
        "first_event":  str(r["first_event_date"])[:10],
        "last_event":   str(r["last_event_date"])[:10],
    }

EDGE_COLORS = {
    "wl_cross_site":   {"color": "#8B0000", "width": 4},
    "wl_repeat_multi": {"color": "#C0392B", "width": 3},
    "same_company":    {"color": "#AAB7B8", "width": 1.5},
}
edges_list = []
for _, r in edges_df.iterrows():
    etype = str(r["edge_type"])
    ec    = EDGE_COLORS.get(etype, {"color": "#AAB7B8", "width": 1})
    label = str(r.get("label", "")) if "wl" in etype else ""
    edges_list.append({
        "from":   int(r["fei_a"]),
        "to":     int(r["fei_b"]),
        "color":  {"color": ec["color"], "opacity": 0.8},
        "width":  ec["width"],
        "dashes": (etype == "same_company"),
        "label":  label,
        "title":  str(r.get("description", "")),
        "font":   {"size": 9, "color": ec["color"]},
    })

# Events: ALL records, descending date, with enriched inspection fields
def sv(row, col):
    """Safe string value from row."""
    v = row.get(col, "")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s in ("nan", "None", "-") else s

events_dict = {}
for fei_val, grp in events_df.groupby("fei"):
    grp_sorted = grp.sort_values("event_date", ascending=False)
    recs = []
    for _, r in grp_sorted.iterrows():
        ev = {
            "date":    str(r["event_date"])[:10],
            "year":    int(r["event_year"]),
            "type":    str(r["event_type"]),
            "sub":     str(r["event_subtype"]),
            "details": str(r["key_details"])[:160],
        }
        # Enriched inspection fields (from updated 01_build_combined_dataset.py)
        for col, key in [("inspection_id",   "insp_id"),
                         ("city",            "city"),
                         ("state",           "state"),
                         ("product_type",    "prod_type"),
                         ("program_area",    "prog_area"),
                         ("posted_citations","posted_cit"),
                         ("fiscal_year",     "fiscal_yr")]:
            ev[key] = sv(r, col) if col in r.index else ""
        recs.append(ev)
    events_dict[str(fei_val)] = recs

# JSON for embedding
nodes_json   = json.dumps(nodes_list)
edges_json   = json.dumps(edges_list)
events_json  = json.dumps(events_dict)
info_json    = json.dumps(fei_info)
sig483_json  = json.dumps(sig483)
sigWL_json   = json.dumps(sigWL)
cfr_json     = json.dumps(cfr_data)
sigRisk_json = json.dumps(sigRisk)
obsRisk_json = json.dumps(obs_by_fei)


# ══════════════════════════════════════════════════════════════════════════
# 3. GENERATE HTML
# ══════════════════════════════════════════════════════════════════════════
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FDA FEI Regulatory Network Dashboard</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Arial, sans-serif; background: #f0f2f6; overflow: hidden; }

/* ── Header ── */
#header {
  position: fixed; top: 0; left: 0; right: 0; height: 52px; z-index: 100;
  background: linear-gradient(135deg, #1F3564, #007A87);
  display: flex; align-items: center; padding: 0 20px; gap: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
#header .title { color: white; font-size: 16px; font-weight: bold; white-space: nowrap; }
#header .subtitle { color: rgba(255,255,255,0.7); font-size: 11px; white-space: nowrap; }
.legend { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-left: auto; }
.leg-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.leg-item { display: flex; align-items: center; gap: 4px; color: white; font-size: 10px; }

/* ── Main layout ── */
#main { display: flex; height: 100vh; padding-top: 52px; }

/* ── Network panel ── */
#network-panel { flex: 1; position: relative; transition: flex 0.3s ease; }
#network-container { width: 100%; height: 100%; }
#hint {
  position: absolute; bottom: 14px; left: 14px;
  background: rgba(31,53,100,0.82); color: white;
  border-radius: 6px; padding: 7px 13px; font-size: 11px; pointer-events: none;
}

/* ── Detail panel ── */
#detail-panel {
  width: 0; overflow: hidden;
  background: white; border-left: 2px solid #e0e0e0;
  display: flex; flex-direction: column;
  transition: width 0.3s ease; min-height: 0;
}
#detail-panel.open { width: 50%; min-width: 480px; }

/* ── Panel header ── */
#panel-header {
  background: #1F3564; padding: 12px 16px 10px; flex-shrink: 0; position: relative;
}
#panel-header .fei-num { color: #BDC3C7; font-size: 11px; margin-bottom: 2px; }
#panel-header .firm    { color: white; font-size: 15px; font-weight: bold; line-height: 1.2; }
#panel-header .country { color: #BDC3C7; font-size: 11px; margin-top: 2px; }
#panel-header .apis    { color: #00C8D4; font-size: 10px; margin-top: 3px; font-style: italic; }
.badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.badge  { padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: bold; color: white; }
#close-btn {
  position: absolute; top: 8px; right: 12px;
  background: rgba(255,255,255,0.2); border: none; color: white;
  border-radius: 50%; width: 26px; height: 26px; cursor: pointer; font-size: 15px;
  display: flex; align-items: center; justify-content: center;
}
#close-btn:hover { background: rgba(255,255,255,0.35); }

/* ── Tab navigation ── */
#tab-nav {
  display: flex; background: #F0F3F8;
  border-bottom: 2px solid #ddd; flex-shrink: 0;
}
.tab-btn {
  flex: 1; padding: 8px 4px; border: none; background: transparent;
  font-size: 11px; font-weight: bold; color: #888; cursor: pointer;
  border-bottom: 3px solid transparent; margin-bottom: -2px;
  transition: all 0.15s; letter-spacing: 0.3px;
}
.tab-btn:hover { background: #E8ECF8; color: #1F3564; }
.tab-btn.active { color: #1F3564; border-bottom-color: #1F3564; background: white; }

/* ── Tab contents ── */
.tab-content { display: none; flex: 1; flex-direction: column; overflow: hidden; min-height: 0; }
.tab-content.active { display: flex; }

/* ── OVERVIEW TAB ── */
#tab-overview { overflow-y: auto; }

/* Timeline */
#tl-header-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px 4px; border-bottom: 1px solid #f0f0f0; flex-shrink: 0;
}
.tl-title {
  font-size: 11px; font-weight: bold; color: #1F3564;
  text-transform: uppercase; letter-spacing: 0.5px;
  display: flex; align-items: center; gap: 6px;
}
.tl-hint { font-size: 9px; color: #aaa; margin-top: 1px; }
#tl-zoom-btns { display: flex; gap: 3px; }
.tl-zoom-btn {
  width: 22px; height: 22px; border: 1px solid #ddd; border-radius: 4px;
  background: white; cursor: pointer; font-size: 13px; font-weight: bold;
  color: #555; display: flex; align-items: center; justify-content: center;
  line-height: 1;
}
.tl-zoom-btn:hover { background: #E8ECF8; border-color: #1F3564; color: #1F3564; }
#tl-range-label { font-size: 9px; color: #888; margin-top: 1px; text-align: right; }

#timeline-wrap { padding: 4px 14px 6px; flex-shrink: 0; border-bottom: 1px solid #f0f0f0; }
#timeline-svg {
  width: 100%; height: 260px; display: block; border-radius: 4px;
  cursor: default; background: #f8f9fa;
}

/* Text signals */
#text-signals { padding: 8px 14px 8px; border-bottom: 1px solid #f0f0f0; flex-shrink: 0; }
.section-title {
  font-size: 11px; font-weight: bold; color: #1F3564;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
  display: flex; align-items: center; gap: 6px;
}
.new-badge { background: #27AE60; color: white; font-size: 8px; padding: 1px 6px; border-radius: 8px; font-weight: normal; }
.fda-badge { background: #7D3C98; color: white; font-size: 8px; padding: 1px 6px; border-radius: 8px; font-weight: normal; }
.signal-source { margin-bottom: 6px; }
.signal-source-label { font-size: 10px; font-weight: bold; color: #555; margin-bottom: 3px; }
.signals-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 10px; }
.sig-item { font-size: 10px; display: flex; align-items: flex-start; gap: 4px; padding: 1px 0; }
.sig-yes { color: #C0392B; font-weight: bold; flex-shrink: 0; }
.sig-no  { color: #AEB6BF; flex-shrink: 0; }
.sig-stat { font-size: 9px; color: #666; margin-top: 3px; font-style: italic; }
.no-text-data { font-size: 10px; color: #BDC3C7; font-style: italic; }

/* ── EVENTS TAB ── */
#tab-events { overflow-y: auto; }
#event-table-wrap { flex: 1; padding: 8px 14px 14px; }
.ev-table { width: 100%; border-collapse: collapse; font-size: 10px; }
.ev-table thead tr { background: #E8ECF0; position: sticky; top: 0; z-index: 1; }
.ev-table th { padding: 5px 6px; text-align: left; font-weight: bold; color: #555; white-space: nowrap; }
.ev-table td { padding: 3px 6px; border-bottom: 1px solid #f5f5f5; vertical-align: top; }
.ev-table tr:hover td { background: #fafafa; }
.ev-type { font-weight: bold; white-space: nowrap; }
.ev-sub  { color: #888; font-size: 9px; white-space: nowrap; }
.ev-detail { color: #666; font-size: 9px; line-height: 1.3; }
.ev-date { color: #999; white-space: nowrap; font-size: 9px; }
.ev-insp-extra { color: #aaa; font-size: 8px; line-height: 1.4; margin-top: 1px; }
.ev-count-bar {
  padding: 6px 14px 4px; font-size: 10px; color: #666; flex-shrink: 0;
  border-bottom: 1px solid #f0f0f0;
  display: flex; gap: 8px; flex-wrap: wrap;
}
.ev-filter-btn {
  padding: 2px 7px; border-radius: 8px; border: 1px solid #ddd;
  font-size: 9px; cursor: pointer; background: white; color: #666;
}
.ev-filter-btn.active { background: #1F3564; color: white; border-color: #1F3564; }

/* ── CFR TAB ── */
#tab-cfr { overflow-y: auto; }
#cfr-body { padding: 10px 14px 20px; }
.cfr-section { margin-bottom: 16px; }
.cfr-bar-row { margin-bottom: 6px; }
.cfr-bar-label { font-size: 10px; font-weight: bold; color: #1F3564; margin-bottom: 1px; display: flex; justify-content: space-between; }
.cfr-bar-short { font-size: 9px; color: #888; margin-bottom: 2px; }
.cfr-bar-track { height: 12px; background: #f0f0f0; border-radius: 3px; position: relative; overflow: hidden; }
.cfr-bar-fill  { height: 100%; border-radius: 3px; transition: width 0.3s; }
.cfr-cooccur-tbl { width: 100%; border-collapse: collapse; font-size: 10px; margin-top: 4px; }
.cfr-cooccur-tbl thead tr { background: #E8ECF0; }
.cfr-cooccur-tbl th { padding: 4px 6px; text-align: left; font-weight: bold; color: #555; }
.cfr-cooccur-tbl td { padding: 3px 6px; border-bottom: 1px solid #f5f5f5; }
.cfr-cooccur-tbl tr:hover td { background: #fafafa; }
.cfr-count-pill {
  display: inline-block; background: #C0392B; color: white;
  border-radius: 8px; padding: 1px 7px; font-size: 9px; font-weight: bold;
}
.domain-chip {
  display: inline-block; font-size: 8px; padding: 1px 5px; border-radius: 3px;
  color: white; margin-left: 3px; vertical-align: middle;
}

/* ── RISK SIGNALS TAB ── */
#tab-risk { overflow-y: auto; }
#risk-body { padding: 10px 14px 20px; }
.risk-summary-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px;
}
.risk-kpi {
  background: #F0F3F8; border-radius: 6px; padding: 8px 10px;
  border-left: 3px solid #1F3564;
}
.risk-kpi-label { font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: 0.4px; }
.risk-kpi-value { font-size: 18px; font-weight: bold; color: #1F3564; line-height: 1.1; }
.risk-kpi-sub   { font-size: 9px; color: #aaa; margin-top: 1px; }
.risk-tri-badge {
  display: inline-block; border-radius: 8px; padding: 3px 10px;
  font-size: 13px; font-weight: bold; color: white; margin-bottom: 12px;
}
.risk-bar-row { margin-bottom: 5px; display: flex; align-items: center; gap: 8px; }
.risk-bar-label { font-size: 10px; min-width: 90px; color: #555; flex-shrink: 0; }
.risk-bar-track { flex: 1; background: #f0f0f0; border-radius: 3px; height: 11px; }
.risk-bar-fill  { height: 100%; border-radius: 3px; }
.risk-bar-pct   { font-size: 9px; color: #888; min-width: 34px; text-align: right; }
.obs-card {
  border: 1px solid #e8e8e8; border-radius: 6px; padding: 8px 10px;
  margin-bottom: 8px; background: white;
}
.obs-card-header {
  display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap;
}
.obs-pill {
  font-size: 8px; padding: 1px 6px; border-radius: 8px; color: white; font-weight: bold;
}
.obs-quote {
  font-size: 9px; color: #555; line-height: 1.4;
  background: #fafafa; border-left: 2px solid #ddd;
  padding: 4px 7px; border-radius: 0 3px 3px 0; margin-top: 4px;
  font-style: italic;
}
.obs-flags { font-size: 9px; color: #888; margin-top: 3px; }
.obs-conf  { font-size: 8px; color: #aaa; }

/* ── Tooltip for timeline events ── */
#ev-tooltip {
  display: none; position: fixed; z-index: 9999;
  background: rgba(31,53,100,0.94); color: white;
  border-radius: 6px; padding: 7px 11px; font-size: 11px;
  pointer-events: none; max-width: 300px; line-height: 1.45;
  box-shadow: 0 3px 10px rgba(0,0,0,0.3);
}
</style>
</head>
<body>

<!-- Header -->
<div id="header">
  <div>
    <div class="title">FDA Facility Cross-Site Regulatory Network</div>
    <div class="subtitle">129 FEIs · 14 APIs · Inspections · 483s · Warning Letters · Recalls · Import Refusals</div>
  </div>
  <div class="legend">
    <span style="color:rgba(255,255,255,0.6);font-size:10px">NODES:</span>
    <span class="leg-item"><span class="leg-dot" style="background:#8B0000"></span>Warning Letter</span>
    <span class="leg-item"><span class="leg-dot" style="background:#C0392B"></span>OAI</span>
    <span class="leg-item"><span class="leg-dot" style="background:#7D3C98"></span>Class I Recall</span>
    <span class="leg-item"><span class="leg-dot" style="background:#E67E22"></span>VAI</span>
    <span class="leg-item"><span class="leg-dot" style="background:#27AE60"></span>NAI</span>
    <span class="leg-item"><span class="leg-dot" style="background:#1A5276"></span>Refusal</span>
    <span class="leg-item"><span class="leg-dot" style="background:#AEB6BF"></span>No Data</span>
    <span style="color:rgba(255,255,255,0.6);font-size:10px;margin-left:6px">EDGES:</span>
    <span class="leg-item"><span style="display:inline-block;width:22px;height:3px;background:#C0392B;border-radius:2px"></span>WL Cross-Site</span>
    <span class="leg-item"><span style="display:inline-block;width:22px;height:2px;background:#AAB7B8;border-radius:2px;border-top:1px dashed #AAB7B8"></span>Same Company</span>
  </div>
</div>

<!-- Main -->
<div id="main">

  <!-- Network -->
  <div id="network-panel">
    <div id="network-container"></div>
    <div id="hint">
      <b>Click any node</b> to explore &nbsp;·&nbsp; <b>Scroll</b> to zoom &nbsp;·&nbsp; <b>Drag</b> to pan
    </div>
  </div>

  <!-- Detail Panel -->
  <div id="detail-panel">

    <!-- Fixed header -->
    <div id="panel-header">
      <div class="fei-num" id="d-fei">FEI —</div>
      <div class="firm"    id="d-firm">—</div>
      <div class="country" id="d-country">—</div>
      <div class="apis"    id="d-apis"></div>
      <div class="badges"  id="d-badges"></div>
      <button id="close-btn" onclick="closePanel()">✕</button>
    </div>

    <!-- Tab navigation -->
    <div id="tab-nav">
      <button class="tab-btn active" data-tab="overview" onclick="showTab('overview')">📈 Overview</button>
      <button class="tab-btn"        data-tab="events"   onclick="showTab('events')">📋 Events</button>
      <button class="tab-btn"        data-tab="cfr"      onclick="showTab('cfr')">⚗️ CFR Analysis</button>
      <button class="tab-btn"        data-tab="risk"     onclick="showTab('risk')">🔬 Risk Signals</button>
    </div>

    <!-- ── TAB 1: Overview ── -->
    <div id="tab-overview" class="tab-content active">
      <!-- Timeline header row -->
      <div id="tl-header-row">
        <div>
          <div class="tl-title">
            Event Timeline
            <span class="fda-badge">Also on FDA Dashboard</span>
          </div>
          <div class="tl-hint">🖱 Scroll to zoom · Drag to pan</div>
        </div>
        <div>
          <div id="tl-zoom-btns">
            <button class="tl-zoom-btn" title="Zoom in"   onclick="tlZoomBtn(1)">+</button>
            <button class="tl-zoom-btn" title="Zoom out"  onclick="tlZoomBtn(-1)">−</button>
            <button class="tl-zoom-btn" title="Reset view" onclick="tlReset()" style="font-size:11px">↺</button>
          </div>
          <div id="tl-range-label"></div>
        </div>
      </div>

      <!-- SVG Timeline -->
      <div id="timeline-wrap">
        <svg id="timeline-svg"></svg>
      </div>

      <!-- Text Analysis Signals -->
      <div id="text-signals">
        <div class="section-title">
          Text Analysis Signals
          <span class="new-badge">✨ Not on FDA Dashboard</span>
        </div>
        <div id="signals-body"></div>
      </div>
    </div>

    <!-- ── TAB 2: Events ── -->
    <div id="tab-events" class="tab-content">
      <div class="ev-count-bar" id="ev-count-bar">
        <span id="ev-count-label">All Events</span>
        <span id="ev-filter-btns"></span>
      </div>
      <div id="event-table-wrap">
        <table class="ev-table">
          <thead><tr>
            <th>Date</th><th>Type</th><th>Sub-type</th><th>Details</th>
          </tr></thead>
          <tbody id="ev-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- ── TAB 3: CFR Analysis ── -->
    <div id="tab-cfr" class="tab-content">
      <div id="cfr-body">
        <div class="no-text-data" style="padding:20px">Select a facility to view CFR data.</div>
      </div>
    </div>

    <!-- ── TAB 4: Risk Signals (LLM pipeline) ── -->
    <div id="tab-risk" class="tab-content">
      <div id="risk-body">
        <div class="no-text-data" style="padding:20px">Select a facility to view LLM-extracted risk signals.</div>
      </div>
    </div>

  </div><!-- end detail-panel -->
</div><!-- end main -->

<!-- Floating tooltip for SVG timeline -->
<div id="ev-tooltip"></div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────
""" + f"""
const NODES_DATA  = {nodes_json};
const EDGES_DATA  = {edges_json};
const EVENTS_DATA = {events_json};
const FEI_INFO    = {info_json};
const SIG_483     = {sig483_json};
const SIG_WL      = {sigWL_json};
const CFR_DATA    = {cfr_json};
const SIG_RISK    = {sigRisk_json};
const OBS_RISK    = {obsRisk_json};
""" + r"""

// ── Event color map ───────────────────────────────────────────────────────
function evColor(ev) {
  if (ev.type === 'Inspection') {
    if (ev.sub === 'OAI') return '#C0392B';
    if (ev.sub === 'VAI') return '#E67E22';
    return '#27AE60';
  }
  if (ev.type === 'Warning Letter') return '#8B0000';
  if (ev.type === '483')            return '#F39C12';
  if (ev.type === 'Recall') {
    if (ev.sub.includes('Class I') && !ev.sub.includes('II') && !ev.sub.includes('III')) return '#7D3C98';
    if (ev.sub.includes('Class II')) return '#9B59B6';
    return '#D7BDE2';
  }
  if (ev.type === 'Import Refusal') return '#1A5276';
  return '#888';
}

// ── CFR domain color ──────────────────────────────────────────────────────
const DOMAIN_COLORS = {
  'org_personnel':  '#3498DB',
  'bldg_equipment': '#E67E22',
  'production':     '#27AE60',
  'pkg_labeling':   '#9B59B6',
  'lab_controls':   '#E74C3C',
  'records_reports':'#1A5276',
  'other_211':      '#95A5A6',
  'non_211':        '#BDC3C7',
};
function domainColor(d) { return DOMAIN_COLORS[d] || '#95A5A6'; }

// ── vis.js Network ────────────────────────────────────────────────────────
const container = document.getElementById('network-container');
const visNodes  = new vis.DataSet(NODES_DATA);
const visEdges  = new vis.DataSet(EDGES_DATA);
const network   = new vis.Network(container, {nodes: visNodes, edges: visEdges}, {
  nodes: {
    shape: 'dot',
    font:  { size: 10, face: 'Arial', strokeWidth: 2, strokeColor: 'rgba(0,0,0,0.4)' },
    borderWidth: 1.5,
    borderWidthSelected: 3,
    shadow: { enabled: true, size: 5, x: 2, y: 2, color: 'rgba(0,0,0,0.15)' },
  },
  edges: {
    smooth:     { type: 'continuous' },
    selectionWidth: 2,
    font:       { size: 9, align: 'middle' },
  },
  physics: {
    barnesHut: {
      gravitationalConstant: -9000,
      centralGravity:        0.3,
      springLength:          165,
      springConstant:        0.025,
      damping:               0.09,
      avoidOverlap:          0.55
    },
    stabilization: { iterations: 300, updateInterval: 20 }
  },
  interaction: {
    hover:             true,
    tooltipDelay:      150,
    navigationButtons: false,
    keyboard:          true,
    hideEdgesOnDrag:   true,
  }
});

let currentFei = null;
network.on('click', function(params) {
  if (params.nodes.length > 0) openPanel(String(params.nodes[0]));
});

// ── Panel open / close ────────────────────────────────────────────────────
function openPanel(fei) {
  currentFei = fei;
  const info   = FEI_INFO[fei]    || {};
  const events = EVENTS_DATA[fei] || [];
  const s483   = SIG_483[fei]     || null;
  const sWL    = SIG_WL[fei]      || null;

  // Header
  document.getElementById('d-fei').textContent     = 'FEI ' + fei;
  document.getElementById('d-firm').textContent    = info.firm    || '—';
  document.getElementById('d-country').textContent = info.country || '—';
  document.getElementById('d-apis').textContent    =
    (info.apis && info.apis.length) ? 'APIs: ' + info.apis.join(', ') : '';

  // Badges
  const badges = [];
  if (info.n_wl    > 0) badges.push(['Warning Letter × ' + info.n_wl, '#8B0000']);
  if (info.n_oai   > 0) badges.push(['OAI × ' + info.n_oai, '#C0392B']);
  if (info.n_class1> 0) badges.push(['Class I Recall × ' + info.n_class1, '#7D3C98']);
  if (info.n_vai   > 0) badges.push(['VAI × ' + info.n_vai, '#E67E22']);
  if (info.n_nai   > 0) badges.push(['NAI × ' + info.n_nai, '#27AE60']);
  if (info.n_refusal>0) badges.push(['Refusal × ' + Math.min(info.n_refusal, 9999), '#1A5276']);
  document.getElementById('d-badges').innerHTML = badges.map(([t,c]) =>
    `<span class="badge" style="background:${c}">${t}</span>`
  ).join('');

  // Render all tabs' content (non-timeline tabs are instant)
  renderSignals(s483, sWL);
  renderEventTable(events, null);  // null = show all types
  renderCfrTab(fei);
  renderRiskTab(fei);

  // Open panel, default to Overview tab
  showTab('overview');
  const panel = document.getElementById('detail-panel');
  const wasOpen = panel.classList.contains('open');
  panel.classList.add('open');
  // Timeline must wait for CSS transition (300ms) so clientWidth is valid
  setTimeout(() => renderTimeline(events), wasOpen ? 40 : 360);
}

function closePanel() {
  document.getElementById('detail-panel').classList.remove('open');
  currentFei = null;
}

function showTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name)
  );
  document.querySelectorAll('.tab-content').forEach(t =>
    t.classList.toggle('active', t.id === 'tab-' + name)
  );
  // Re-render timeline if switching to overview (size may have changed)
  if (name === 'overview' && currentFei) {
    setTimeout(() => renderTimeline(EVENTS_DATA[currentFei] || []), 30);
  }
}

// ══════════════════════════════════════════════════════════════════════════
// TIMELINE  (zoom/pan via viewBox manipulation)
// ══════════════════════════════════════════════════════════════════════════
const LANE_MAP    = { 'Inspection':1, '483':2, 'Warning Letter':2, 'Recall':3, 'Import Refusal':3 };
const LANE_LABELS = ['', 'Inspections', '483 / WL', 'Recall / Refusal'];

// ViewBox state — updated by zoom/pan; reset by renderTimeline() on each fresh draw
let tlVB = null;  // { x, w, fullW, h }

function renderTimeline(events) {
  const svgEl = document.getElementById('timeline-svg');
  svgEl.innerHTML = '';

  // Use actual clientWidth; fall back to 50% of window if panel is still transitioning (clientWidth ≈ 0)
  const rawW = svgEl.parentElement ? svgEl.parentElement.clientWidth : 0;
  const W    = rawW > 100 ? rawW - 28 : Math.max(420, Math.round(window.innerWidth * 0.48));
  const H  = 260;
  svgEl.setAttribute('width',  W);
  svgEl.setAttribute('height', H);

  const ML = 78, MR = 10, MT = 16, MB = 26;
  const plotW = W - ML - MR;

  // Initialize viewBox to full range
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  tlVB = { x: 0, w: W, fullW: W, h: H };

  const FULL_MIN = 2009, FULL_MAX = 2026.5;
  // x-position in SVG coords for a date string
  function xOfDate(dateStr) {
    const d   = new Date(dateStr);
    const yr  = d.getFullYear() + d.getMonth() / 12 + d.getDate() / 365;
    return ML + (yr - FULL_MIN) / (FULL_MAX - FULL_MIN) * plotW;
  }

  const LANE_Y = [0, MT + 46, MT + 118, MT + 190];
  const ns     = 'http://www.w3.org/2000/svg';
  function mk(tag, attrs, text) {
    const el = document.createElementNS(ns, tag);
    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));
    if (text !== undefined) el.textContent = text;
    return el;
  }

  // Background
  svgEl.appendChild(mk('rect', {x:0, y:0, width:W, height:H, fill:'#f8f9fa'}));

  // Lane backgrounds
  [[LANE_Y[1]-22, '#fff7f7'], [LANE_Y[2]-22, '#fffff5'], [LANE_Y[3]-22, '#f7f9ff']].forEach(([ly, fill]) => {
    svgEl.appendChild(mk('rect', {x:ML, y:ly, width:plotW, height:60, fill}));
  });

  // Year gridlines + labels (draw all years; zoom clips via viewBox)
  for (let yr = FULL_MIN + 1; yr <= Math.ceil(FULL_MAX); yr++) {
    const x = xOfDate(`${yr}-01-01`);
    svgEl.appendChild(mk('line', {x1:x, y1:MT, x2:x, y2:H-MB, stroke:'#e0e0e0', 'stroke-width':1}));
    svgEl.appendChild(mk('text', {
      x: x, y: H - 8, 'text-anchor':'middle', 'font-size': 9, fill:'#bbb', 'font-family':'Arial'
    }, String(yr)));
  }

  // Lane separator lines + labels
  svgEl.appendChild(mk('line', {x1:ML, y1:MT, x2:ML, y2:H-MB, stroke:'#ccc', 'stroke-width':1}));
  for (let i = 1; i <= 3; i++) {
    svgEl.appendChild(mk('text', {
      x: ML - 5, y: LANE_Y[i] + 4,
      'text-anchor':'end', 'font-size':8, fill:'#aaa', 'font-family':'Arial'
    }, LANE_LABELS[i]));
    if (i < 3) {
      const sepY = LANE_Y[i] + 30;
      svgEl.appendChild(mk('line', {
        x1:ML, y1:sepY, x2:W-MR, y2:sepY,
        stroke:'#e4e4e4', 'stroke-width':0.5, 'stroke-dasharray':'4,3'
      }));
    }
  }

  // Event circles
  const tooltip = document.getElementById('ev-tooltip');
  const placed  = {};

  // Sort events chronologically for proper jitter tracking
  const sorted = [...events].sort((a,b) => a.date.localeCompare(b.date));
  sorted.forEach(ev => {
    const lane = LANE_MAP[ev.type];
    if (!lane) return;
    const x  = xOfDate(ev.date);
    const ly = LANE_Y[lane];

    // Jitter to reduce overlap
    const xBin = Math.round(x / 4) * 4 + '_' + lane;
    placed[xBin] = (placed[xBin] || 0) + 1;
    const jitter = (placed[xBin] - 1) * 8;

    const r     = 7;
    const color = evColor(ev);
    const cy    = ly - jitter;

    const g = document.createElementNS(ns, 'g');
    g.style.cursor = 'pointer';

    const circle = mk('circle', {
      cx: x, cy, r, fill: color, opacity: 0.85, stroke:'white', 'stroke-width': 1.5
    });

    let tip = `<b>${ev.type}</b>${ev.sub ? ' (' + ev.sub + ')' : ''}<br>${ev.date}`;
    if (ev.details) tip += `<br><span style="opacity:.85;font-size:10px">${ev.details}</span>`;
    if (ev.type === 'Inspection') {
      if (ev.city)     tip += `<br><span style="opacity:.7">📍 ${ev.city}${ev.state ? ', '+ev.state : ''}</span>`;
      if (ev.insp_id)  tip += `<br><span style="opacity:.6">ID: ${ev.insp_id}</span>`;
      if (ev.posted_cit === 'Yes') tip += `<br><span style="opacity:.7">📋 483 Posted</span>`;
    }

    circle.addEventListener('mouseenter', e => {
      tooltip.innerHTML = tip; tooltip.style.display = 'block'; moveTooltip(e);
    });
    circle.addEventListener('mousemove',  moveTooltip);
    circle.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    g.appendChild(circle);
    svgEl.appendChild(g);
  });

  updateTlRangeLabel();
}

function moveTooltip(e) {
  const t = document.getElementById('ev-tooltip');
  t.style.left = (e.clientX + 14) + 'px';
  t.style.top  = (e.clientY - 10) + 'px';
}

function updateTlRangeLabel() {
  if (!tlVB) return;
  const FULL_MIN = 2009, FULL_MAX = 2026.5;
  const ML = 78, W = tlVB.fullW, plotW = W - ML - 10;
  const frac  = (tlVB.x + (tlVB.fullW - tlVB.w) / 2) / tlVB.fullW;
  const range = (tlVB.w / tlVB.fullW) * (FULL_MAX - FULL_MIN);
  const lo    = FULL_MIN + (tlVB.x / plotW) * (FULL_MAX - FULL_MIN);
  const hi    = lo + range;
  const label = `${Math.max(2009, Math.round(lo))} – ${Math.min(2027, Math.ceil(hi))}`;
  const el = document.getElementById('tl-range-label');
  if (el) el.textContent = range < 16 ? label : '';
}


// ── Timeline zoom/pan — initialised ONCE at page load ────────────────────
// Handlers read/write the module-level `tlVB` object.
// renderTimeline() simply resets tlVB after each fresh draw.
(function initTimelineInteraction() {
  const svg  = document.getElementById('timeline-svg');
  let dragging = false, dragStartX = 0, dragStartVBX = 0;

  // ── Mouse wheel → zoom centred on mouse X ──────────────────────────────
  svg.addEventListener('wheel', function(e) {
    e.preventDefault();
    if (!tlVB || tlVB.fullW < 10) return;

    const rect   = svg.getBoundingClientRect();
    const mx     = (e.clientX - rect.left) / rect.width;   // 0 → 1
    const factor = e.deltaY > 0 ? 1.3 : 0.77;              // out / in
    const newW   = Math.max(tlVB.fullW / 12, Math.min(tlVB.fullW, tlVB.w * factor));
    const anchor = tlVB.x + mx * tlVB.w;  // SVG coord under cursor stays fixed
    tlVB.x = Math.max(0, Math.min(tlVB.fullW - newW, anchor - mx * newW));
    tlVB.w = newW;
    svg.setAttribute('viewBox', tlVB.x + ' 0 ' + tlVB.w + ' ' + tlVB.h);
    updateTlRangeLabel();
  }, { passive: false });

  // ── Mouse drag → pan ───────────────────────────────────────────────────
  svg.addEventListener('mousedown', function(e) {
    if (e.button !== 0 || !tlVB) return;
    dragging     = true;
    dragStartX   = e.clientX;
    dragStartVBX = tlVB.x;
    svg.style.cursor = 'grabbing';
    e.preventDefault();
  });

  window.addEventListener('mousemove', function(e) {
    if (!dragging || !tlVB || tlVB.fullW < 10) return;
    const rect   = svg.getBoundingClientRect();
    if (rect.width < 1) return;
    const dxCoord = -(e.clientX - dragStartX) / rect.width * tlVB.w;
    tlVB.x = Math.max(0, Math.min(tlVB.fullW - tlVB.w, dragStartVBX + dxCoord));
    svg.setAttribute('viewBox', tlVB.x + ' 0 ' + tlVB.w + ' ' + tlVB.h);
    updateTlRangeLabel();
  });

  window.addEventListener('mouseup', function() {
    if (dragging) { dragging = false; svg.style.cursor = 'default'; }
  });
})();

// ── Zoom buttons (called from HTML onclick) ───────────────────────────────
function tlZoomBtn(dir) {
  if (!tlVB || tlVB.fullW < 10) return;
  const svg    = document.getElementById('timeline-svg');
  const factor = dir > 0 ? 0.7 : 1.4;
  const center = tlVB.x + tlVB.w / 2;
  const newW   = Math.max(tlVB.fullW / 12, Math.min(tlVB.fullW, tlVB.w * factor));
  tlVB.x = Math.max(0, Math.min(tlVB.fullW - newW, center - newW / 2));
  tlVB.w = newW;
  svg.setAttribute('viewBox', tlVB.x + ' 0 ' + tlVB.w + ' ' + tlVB.h);
  updateTlRangeLabel();
}
function tlReset() {
  if (!tlVB || tlVB.fullW < 10) return;
  tlVB.x = 0; tlVB.w = tlVB.fullW;
  document.getElementById('timeline-svg')
    .setAttribute('viewBox', '0 0 ' + tlVB.fullW + ' ' + tlVB.h);
  updateTlRangeLabel();
}

// ══════════════════════════════════════════════════════════════════════════
// TEXT ANALYSIS SIGNALS
// ══════════════════════════════════════════════════════════════════════════
function renderSignals(s483, sWL) {
  const body = document.getElementById('signals-body');
  if (!s483 && !sWL) {
    body.innerHTML = '<div class="no-text-data">No 483 or Warning Letter text extracted for this facility.</div>';
    return;
  }
  let html = '';

  if (s483) {
    const items = [
      ['Repeat violations in obs. text', s483.ever_repeat],
      ['Data integrity language',        s483.ever_data_integrity],
      ['Contamination mentioned',        s483.ever_contamination],
      ['Systemic failure noted',         s483.ever_systemic],
      ['OOS / OOT results cited',        s483.ever_oos_oot],
      ['Patient safety concern',         s483.ever_patient_risk],
      ['Ref. to prior Warning Letter',   s483.ever_wl_ref],
    ];
    html += `<div class="signal-source">
      <div class="signal-source-label">📋 From Form 483 PDF Text (${s483.n_483s} PDF${s483.n_483s!==1?'s':''}, ${s483.n_obs} observations)</div>
      <div class="signals-grid">`;
    items.forEach(([label, val]) => {
      html += `<div class="sig-item">
        <span class="${val?'sig-yes':'sig-no'}">${val?'✓':'✗'}</span>
        <span style="color:${val?'#333':'#aaa'}">${label}</span>
      </div>`;
    });
    html += '</div>';
    if (s483.avg_chars > 0) {
      html += `<div class="sig-stat">Avg observation body: ${s483.avg_chars.toLocaleString()} chars
        — ${Math.round(s483.avg_chars/158)}× richer than Citation DB</div>`;
    }
    html += '</div>';
  } else {
    html += '<div class="signal-source"><div class="no-text-data">No 483 PDF extracted for this facility.</div></div>';
  }

  if (sWL) {
    const items = [
      ['Repeat violations at this facility',  sWL.repeat_facility],
      ['Repeat violations at multiple sites', sWL.repeat_multi],
      ['Management oversight critique',       sWL.mgmt_oversight],
      ['Corporate failure language',          sWL.corp_failure],
    ];
    html += `<div class="signal-source" style="margin-top:6px">
      <div class="signal-source-label">⚠️ From Warning Letter Text (${sWL.n_wls} WL${sWL.n_wls!==1?'s':''})</div>
      <div class="signals-grid">`;
    items.forEach(([label, val]) => {
      html += `<div class="sig-item">
        <span class="${val?'sig-yes':'sig-no'}">${val?'✓':'✗'}</span>
        <span style="color:${val?'#333':'#aaa'}">${label}</span>
      </div>`;
    });
    html += '</div>';
    const cfr = sWL.cfr_list ? sWL.cfr_list.split('; ').slice(0,4).join(', ') : '';
    html += `<div class="sig-stat">${sWL.n_violations} violation${sWL.n_violations!==1?'s':''} ·
      ${sWL.n_prior_wl_refs} prior WL ref${sWL.n_prior_wl_refs!==1?'s':''}
      ${cfr ? ' · CFRs: ' + cfr : ''}</div>`;
    html += '</div>';
  } else {
    html += '<div class="signal-source" style="margin-top:4px"><div class="no-text-data">No Warning Letter text extracted for this facility.</div></div>';
  }

  body.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════════════
// EVENT TABLE  (all events, with enriched inspection details)
// ══════════════════════════════════════════════════════════════════════════
let evFilterType = null;  // null = all

function renderEventTable(events, filterType) {
  evFilterType = filterType;
  const filtered = filterType ? events.filter(e => e.type === filterType) : events;

  // Count bar
  const typeCounts = {};
  events.forEach(e => typeCounts[e.type] = (typeCounts[e.type] || 0) + 1);
  const typeOrder = ['Inspection','483','Warning Letter','Recall','Import Refusal'];
  let filterBtns = `<button class="ev-filter-btn ${!filterType?'active':''}"
      onclick="renderEventTable(EVENTS_DATA['${currentFei}']||[], null)">All (${events.length})</button>`;
  typeOrder.forEach(t => {
    if (!typeCounts[t]) return;
    filterBtns += `<button class="ev-filter-btn ${filterType===t?'active':''}"
      onclick="renderEventTable(EVENTS_DATA['${currentFei}']||[], '${t}')">${t} (${typeCounts[t]})</button>`;
  });
  document.getElementById('ev-filter-btns').innerHTML = filterBtns;
  document.getElementById('ev-count-label').textContent =
    filterType ? `Showing: ${filterType} (${filtered.length} of ${events.length})` : `All Events (${events.length})`;

  // Table
  const tbody = document.getElementById('ev-tbody');
  tbody.innerHTML = filtered.map(ev => {
    const color = evColor(ev);
    let extra = '';
    if (ev.type === 'Inspection') {
      const parts = [];
      if (ev.city)        parts.push('📍 ' + ev.city + (ev.state ? ', ' + ev.state : ''));
      if (ev.insp_id)     parts.push('ID: ' + ev.insp_id);
      if (ev.prod_type)   parts.push(ev.prod_type);
      if (ev.fiscal_yr)   parts.push('FY' + ev.fiscal_yr);
      if (ev.posted_cit === 'Yes') parts.push('📋 483 Posted');
      if (parts.length)   extra = `<div class="ev-insp-extra">${parts.join(' · ')}</div>`;
    }
    return `<tr>
      <td class="ev-date">${ev.date}</td>
      <td><span class="ev-type" style="color:${color}">${ev.type}</span></td>
      <td class="ev-sub">${ev.sub || ''}</td>
      <td class="ev-detail">${ev.details}${extra}</td>
    </tr>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════════════════
// CFR ANALYSIS TAB
// ══════════════════════════════════════════════════════════════════════════
function renderCfrTab(fei) {
  const body = document.getElementById('cfr-body');
  const data = CFR_DATA[fei];

  if (!data) {
    body.innerHTML = `<div style="padding:20px">
      <div class="no-text-data">No citation records found for this facility.</div>
      <div style="font-size:10px;color:#888;margin-top:8px">
        This may mean all inspections were NAI (No Action Indicated), which is a positive signal.
      </div>
    </div>`;
    return;
  }

  const maxCfrCount = data.cfrs && data.cfrs.length > 0
    ? Math.max(...data.cfrs.map(c => c.count)) : 1;

  let html = `
  <div style="background:#F0F3F8;padding:8px 14px;border-bottom:1px solid #e0e0e0;margin-bottom:10px">
    <span style="font-size:11px;color:#555">
      <b>${data.n_total_cit}</b> total citations ·
      <b>${data.n_insp_with_cit}</b> inspections with violations ·
      <b>${data.n_unique_cfr}</b> unique CFRs
    </span>
  </div>

  <!-- Top CFRs bar chart -->
  <div class="cfr-section" style="padding:0 14px">
    <div class="section-title">
      Top Cited CFR Sections
      <span class="new-badge">✨ Not on FDA Dashboard</span>
    </div>`;

  if (data.cfrs && data.cfrs.length > 0) {
    data.cfrs.forEach(item => {
      const pct   = Math.round(item.count / maxCfrCount * 100);
      const color = domainColor(item.domain);
      html += `<div class="cfr-bar-row">
        <div class="cfr-bar-label">
          <span>${item.cfr}
            <span class="domain-chip" style="background:${color}">${item.domain.replace('_',' ')}</span>
          </span>
          <span style="color:${color};font-weight:bold">${item.count}×</span>
        </div>
        <div class="cfr-bar-short">${item.short}</div>
        <div class="cfr-bar-track">
          <div class="cfr-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
      </div>`;
    });
  }

  html += `</div>`;

  // Domain breakdown
  if (data.domains && data.domains.length > 0) {
    const maxDom = Math.max(...data.domains.map(d => d.count));
    html += `<div class="cfr-section" style="padding:0 14px;margin-top:14px">
      <div class="section-title">Regulatory Domain Breakdown</div>`;
    data.domains.forEach(d => {
      const pct   = Math.round(d.count / maxDom * 100);
      const color = domainColor(
        Object.keys(DOMAIN_COLORS).find(k => d.domain.toLowerCase().includes(k.split('_')[0])) || 'non_211'
      );
      html += `<div style="margin-bottom:5px;display:flex;align-items:center;gap:8px">
        <span style="font-size:9px;min-width:160px;color:#555;flex-shrink:0">${d.domain}</span>
        <div style="flex:1;background:#f0f0f0;border-radius:3px;height:10px">
          <div style="width:${pct}%;background:${color};height:100%;border-radius:3px;opacity:0.8"></div>
        </div>
        <span style="font-size:9px;color:#888;min-width:24px;text-align:right">${d.count}</span>
      </div>`;
    });
    html += `</div>`;
  }

  // Co-occurrence pairs
  if (data.cooccurrence && data.cooccurrence.length > 0) {
    const maxCooc = data.cooccurrence[0].count;
    html += `<div class="cfr-section" style="padding:0 14px;margin-top:14px">
      <div class="section-title">CFR Co-Occurrence Pairs</div>
      <div style="font-size:9px;color:#888;margin-bottom:6px">
        Pairs cited together in the same inspection — signals systemic issues
      </div>
      <table class="cfr-cooccur-tbl">
        <thead><tr>
          <th>CFR A</th><th>CFR B</th><th style="text-align:center">Together</th>
        </tr></thead>
        <tbody>`;
    data.cooccurrence.forEach(p => {
      const colorA = domainColor(p.a_dom || 'non_211');
      const colorB = domainColor(p.b_dom || 'non_211');
      html += `<tr>
        <td><span style="color:#1F3564;font-weight:bold">${p.a}</span>
          <span class="domain-chip" style="background:${colorA}">${(p.a_dom||'').replace('_',' ')}</span></td>
        <td><span style="color:#1F3564;font-weight:bold">${p.b}</span>
          <span class="domain-chip" style="background:${colorB}">${(p.b_dom||'').replace('_',' ')}</span></td>
        <td style="text-align:center"><span class="cfr-count-pill">${p.count}</span></td>
      </tr>`;
    });
    html += `</tbody></table></div>`;
  }

  // Per-inspection breakdown (most recent)
  if (data.inspections && data.inspections.length > 0) {
    html += `<div class="cfr-section" style="padding:0 14px;margin-top:14px">
      <div class="section-title">Per-Inspection CFR Summary</div>
      <table class="cfr-cooccur-tbl">
        <thead><tr>
          <th>Date</th><th>Insp. ID</th><th style="text-align:center"># CFRs</th><th>CFRs Cited</th>
        </tr></thead>
        <tbody>`;
    data.inspections.forEach(ins => {
      html += `<tr>
        <td style="white-space:nowrap">${ins.date}</td>
        <td style="color:#888">${ins.insp_id}</td>
        <td style="text-align:center;font-weight:bold;color:#C0392B">${ins.n_cfr}</td>
        <td style="font-size:9px;color:#666">${ins.cfrs}</td>
      </tr>`;
    });
    html += `</tbody></table></div>`;
  }

  html += `
  <div style="padding:10px 14px;font-size:9px;color:#aaa;border-top:1px solid #f0f0f0;margin-top:10px">
    Source: FDA Inspection Citations Details DB · Filtered to 129 FEIs in our study universe
  </div>`;

  body.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════════════
// RISK SIGNALS TAB  (LLM-extracted, Phase 3 pipeline)
// ══════════════════════════════════════════════════════════════════════════
function renderRiskTab(fei) {
  const body = document.getElementById('risk-body');
  const data = SIG_RISK[fei];
  const obs  = OBS_RISK[fei] || [];

  if (!data) {
    body.innerHTML = `<div style="padding:20px">
      <div class="no-text-data">No LLM-extracted risk signals for this facility.</div>
      <div style="font-size:10px;color:#888;margin-top:8px">
        Run the text analysis pipeline (scripts 04 → 05 → 06) to generate signals.
      </div>
    </div>`;
    return;
  }

  // TRI badge colour
  const tri = data.text_risk_index;
  const triColor = tri >= 60 ? '#C0392B' : tri >= 35 ? '#E67E22' : '#27AE60';

  function pct(v) { return Math.round((v || 0) * 100); }
  function bar(label, val, color, maxVal) {
    const w = maxVal > 0 ? Math.round(val / maxVal * 100) : pct(val);
    return `<div class="risk-bar-row">
      <span class="risk-bar-label">${label}</span>
      <div class="risk-bar-track"><div class="risk-bar-fill" style="width:${w}%;background:${color}"></div></div>
      <span class="risk-bar-pct">${pct(val)}%</span>
    </div>`;
  }

  // Severity colours
  const SEV_C  = { High: '#C0392B', Moderate: '#E67E22', Low: '#27AE60' };
  // Root cause colours
  const RC_C   = { Capital: '#3498DB', Cultural: '#E67E22', Mixed: '#7D3C98', Unclear: '#95A5A6' };
  // Remediation colours
  const REM_C  = { Strong: '#27AE60', Partial: '#F1C40F', Weak: '#E67E22', None: '#C0392B' };
  // Violation category colours (reuse domain palette)
  const CAT_C  = {
    LabControls: '#E74C3C', ProductionControls: '#27AE60',
    BuildingsEquipment: '#E67E22', OrgPersonnel: '#3498DB',
    PackagingLabeling: '#9B59B6', RecordsReports: '#1A5276',
    QualitySystem: '#C0392B', Other: '#95A5A6',
  };

  let html = `
  <!-- TRI summary row -->
  <div style="padding:10px 14px 8px;border-bottom:1px solid #f0f0f0">
    <div style="font-size:10px;color:#888;margin-bottom:4px">
      <b>${data.n_obs_scored}</b> observations scored
      (${data.n_483_obs} from 483s · ${data.n_wl_obs} from WLs)
      · avg confidence ${Math.round((data.mean_confidence||0)*100)}%
    </div>
    <span class="risk-tri-badge" style="background:${triColor}">
      Text Risk Index: ${tri.toFixed(1)} / 100
    </span>
    <span style="font-size:9px;color:#aaa;margin-left:8px">
      Dominant root cause: <b style="color:#555">${data.dominant_root_cause}</b>
      · Dominant violation: <b style="color:#555">${data.dominant_violation_category}</b>
    </span>
  </div>

  <div style="padding:8px 14px 0">

  <!-- KPI grid -->
  <div class="risk-summary-grid">
    <div class="risk-kpi">
      <div class="risk-kpi-label">High Severity</div>
      <div class="risk-kpi-value" style="color:#C0392B">${pct(data.severity_high_share)}%</div>
      <div class="risk-kpi-sub">of scored observations</div>
    </div>
    <div class="risk-kpi">
      <div class="risk-kpi-label">Repeat Violations</div>
      <div class="risk-kpi-value" style="color:#E67E22">${pct(data.repeat_flag_share)}%</div>
      <div class="risk-kpi-sub">have repeat flag</div>
    </div>
    <div class="risk-kpi">
      <div class="risk-kpi-label">Capital Root Cause</div>
      <div class="risk-kpi-value" style="color:#3498DB">${pct(data.capital_share)}%</div>
      <div class="risk-kpi-sub">equipment / process gap</div>
    </div>
    <div class="risk-kpi">
      <div class="risk-kpi-label">Weak Remediation</div>
      <div class="risk-kpi-value" style="color:#C0392B">${pct(data.remediation_weak_share + data.remediation_none_share)}%</div>
      <div class="risk-kpi-sub">weak or no corrective action</div>
    </div>
  </div>

  <!-- Severity distribution -->
  <div class="section-title" style="margin-top:10px">Severity Tier Distribution</div>
  ${bar('High',     data.severity_high_share, '#C0392B', 1)}
  ${bar('Moderate', data.severity_mod_share,  '#E67E22', 1)}
  ${bar('Low',      data.severity_low_share,  '#27AE60', 1)}

  <!-- Root cause breakdown -->
  <div class="section-title" style="margin-top:12px">Root Cause Breakdown</div>
  ${bar('Capital',  data.capital_share,  '#3498DB', 1)}
  ${bar('Cultural', data.cultural_share, '#E67E22', 1)}
  ${bar('Mixed',    data.mixed_share,    '#7D3C98', 1)}
  ${bar('Unclear',  data.unclear_share,  '#95A5A6', 1)}
  <div style="font-size:9px;color:#aaa;margin-top:3px">
    Capital = equipment/facility/SOP gap &nbsp;·&nbsp;
    Cultural = training/oversight/data-integrity failure
  </div>

  <!-- Remediation mix -->
  <div class="section-title" style="margin-top:12px">Remediation Signal</div>
  ${bar('Strong',  data.remediation_strong_share,  '#27AE60', 1)}
  ${bar('Partial', data.remediation_partial_share, '#F1C40F', 1)}
  ${bar('Weak',    data.remediation_weak_share,    '#E67E22', 1)}
  ${bar('None',    data.remediation_none_share,    '#C0392B', 1)}

  <!-- TRI formula note -->
  <div style="font-size:9px;color:#aaa;background:#f8f9fa;border-radius:4px;padding:6px 8px;margin-top:12px;line-height:1.6">
    <b style="color:#555">TRI formula:</b>
    0.35×High + 0.20×Mod + 0.20×(1−Strong rem) + 0.15×Repeat + 0.10×Systemic
  </div>

  </div><!-- end padding -->`;

  // Per-observation cards
  if (obs.length > 0) {
    html += `<div style="padding:8px 14px 0;border-top:1px solid #f0f0f0;margin-top:10px">
      <div class="section-title">
        Observation Cards
        <span class="new-badge">top ${obs.length} by confidence</span>
      </div>`;
    obs.forEach(o => {
      const sevColor = SEV_C[o.sev]  || '#888';
      const rcColor  = RC_C[o.rc]    || '#888';
      const remColor = REM_C[o.rem]  || '#888';
      const catColor = CAT_C[o.cat]  || '#888';
      const flags = [];
      if (o.repeat)   flags.push('🔁 Repeat');
      if (o.systemic) flags.push('⚠️ Systemic');
      if (o.patient)  flags.push('🏥 Patient risk');
      html += `<div class="obs-card">
        <div class="obs-card-header">
          <span class="obs-pill" style="background:${catColor}">${o.cat}</span>
          <span class="obs-pill" style="background:${sevColor}">${o.sev}</span>
          <span class="obs-pill" style="background:${rcColor}">${o.rc}</span>
          <span class="obs-pill" style="background:${remColor}">${o.rem} rem.</span>
          <span style="font-size:8px;color:#aaa;margin-left:auto">${o.src}</span>
        </div>
        ${o.quote ? `<div class="obs-quote">"${o.quote.replace(/"/g,'&quot;')}"</div>` : ''}
        ${flags.length ? `<div class="obs-flags">${flags.join(' &nbsp; ')}</div>` : ''}
        <div class="obs-conf">conf ${Math.round(o.conf*100)}%</div>
      </div>`;
    });
    html += `</div>`;
  }

  html += `<div style="padding:6px 14px 14px;font-size:9px;color:#aaa;border-top:1px solid #f0f0f0;margin-top:8px">
    Source: LLM extraction pipeline (Claude) · scripts 04–06
  </div>`;

  body.innerHTML = html;
}

</script>
</body>
</html>"""

# ── Save ───────────────────────────────────────────────────────────────────
with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = HTML_OUT.stat().st_size // 1024
print(f"\n✓ Saved: {HTML_OUT}  ({size_kb} KB)")
print("  Open in Chrome / Firefox — no server required.")
print(f"\nEmbedded:")
print(f"  {len(nodes_list)} nodes  ·  {len(edges_list)} edges")
print(f"  {sum(len(v) for v in events_dict.values())} events across {len(events_dict)} FEIs")
print(f"  {len(sig483)} FEIs with 483 text signals")
print(f"  {len(sigWL)} FEIs with WL text signals")
print(f"  {len(cfr_data)} FEIs with CFR citation data")
print(f"  {len(sigRisk)} FEIs with LLM risk signals  "
      f"(run 04→05→06 pipeline to populate)"
      if not sigRisk else
      f"  {len(sigRisk)} FEIs with LLM risk signals  ·  "
      f"{sum(len(v) for v in obs_by_fei.values())} observation cards")
