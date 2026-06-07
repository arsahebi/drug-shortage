# %%
"""
01_build_combined_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Combines five FDA data sources into a unified FEI-level event timeline,
  node summary, cross-site edge list, and CFR citation JSON.
  These four output files feed every downstream script in this folder.

WHEN TO RUN
  Run first, before any other script in this folder.
  Re-run whenever a source file from folders 12, 14, 21, 22, or 23 changes.
  Takes ~1-2 minutes for 129 FEIs.

REQUIRED FOR COMBINED DATASET?  YES — this is the root of the pipeline.

INPUTS (all read-only — this script does not modify source files)
  Data/08 - Valisure/raw/FEIs_March 2026.xlsx   → 129 reference FEIs
  Data/14 - FDA - Inspection/raw/Inspections Details.xlsx
  Data/14 - FDA - Inspection/raw/Inspections Citations Details.xlsx
  Data/12 - FDA - 483/processed/483_pdf_inventory.csv  → 483 dates + n_observations
  Data/12 - FDA - 483/processed/483_fei_features.csv   → FEI-level regex flags
    NOTE: reads 483_fei_features.csv (FEI-level aggregates), NOT 483_observations.csv.
    The per-observation 483_observations.csv is used by the LLM pipeline (scripts 04-07).
  Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv
  Data/21 - FDA - Warning Letter/processed/wl_fei_network.csv
  Data/22 - FDA - Recall/processed/recall_filtered.csv
  Data/23 - FDA - Import Refusal/processed/import_refusal_filtered.csv

OUTPUTS (written to this folder)
  fei_events_timeline.csv  — one row per regulatory event per FEI
  fei_node_summary.csv     — one row per FEI with aggregate counts and severity score
  fei_edge_list.csv        — cross-FEI edges (WL cross-site + same-company name match)
  fei_cfr_data.json        — per-FEI CFR citation frequencies, domains, co-occurrence

DOWNSTREAM CONSUMERS
  02_build_interactive_network.py   reads all four outputs
  03_build_interactive_dashboard.py reads all four outputs
  07_merge_text_signals.py          reads fei_node_summary.csv
"""

import json
import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
OUT  = Path(__file__).parent

VALISURE   = BASE / "Data/08 - Valisure/raw/FEIs_March 2026.xlsx"
INSP_RAW   = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"
CIT_RAW    = BASE / "Data/14 - FDA - Inspection/raw/Inspections Citations Details.xlsx"
INV_483    = BASE / "Data/12 - FDA - 483/processed/483_pdf_inventory.csv"
OBS_483    = BASE / "Data/12 - FDA - 483/processed/483_fei_features.csv"
WL_REC     = BASE / "Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv"
WL_NET     = BASE / "Data/21 - FDA - Warning Letter/processed/wl_fei_network.csv"
REC_FILT   = BASE / "Data/22 - FDA - Recall/processed/recall_filtered.csv"
IMP_FILT   = BASE / "Data/23 - FDA - Import Refusal/processed/import_refusal_filtered.csv"


# ══════════════════════════════════════════════════════════════════════════
# 1. REFERENCE FEIs (129 from Valisure mapping)
# ══════════════════════════════════════════════════════════════════════════
print("="*65)
print("STEP 1 — Loading reference FEIs")
print("="*65)

valisure = pd.read_excel(VALISURE, sheet_name="API Only_FEI Mapping")
ref_feis = set(valisure["FEI_NUMBER"].dropna().astype(int))
print(f"Reference FEIs: {len(ref_feis)}")


# ══════════════════════════════════════════════════════════════════════════
# 2. FIRM NAME + COUNTRY LOOKUP (from Inspection DB)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 2 — Loading Inspection DB for firm names & countries")
print("="*65)

insp_raw = pd.read_excel(INSP_RAW)
insp_raw["FEI Number"] = insp_raw["FEI Number"].astype(str).str.strip()
insp_raw["_fei_int"] = pd.to_numeric(insp_raw["FEI Number"], errors="coerce").astype("Int64")

# Filter to our FEIs
insp_our = insp_raw[insp_raw["_fei_int"].isin(ref_feis)].copy()
print(f"Inspection rows for our FEIs: {len(insp_our)}")

# Build firm name & country lookup (most common per FEI)
firm_lookup    = insp_our.groupby("_fei_int")["Legal Name"].agg(lambda x: x.mode()[0] if len(x) else "Unknown")
country_lookup = insp_our.groupby("_fei_int")["Country/Area"].agg(lambda x: x.mode()[0] if len(x) else "Unknown")

def get_firm(fei):
    return firm_lookup.get(fei, f"FEI {fei}")

def get_country(fei):
    return country_lookup.get(fei, "Unknown")

def safe_str(v):
    """Return empty string for NaN / 'nan' / '-' values."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s in ("nan", "None", "-", "NaN", "") else s


# ══════════════════════════════════════════════════════════════════════════
# 3. BUILD EVENTS TIMELINE
# Each event: fei, firm_name, country, event_date, event_year,
#             event_type, event_subtype, severity_num, key_details
#             + inspection-specific: inspection_id, city, state,
#               product_type, program_area, posted_citations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 3 — Building events timeline")
print("="*65)

SEV = {
    "Warning Letter":   5,
    "Inspection_OAI":   4,
    "Recall_ClassI":    4,
    "Inspection_VAI":   2,
    "Recall_ClassII":   2,
    "483":              2,
    "Import Refusal":   1,
    "Inspection_NAI":   0,
    "Recall_ClassIII":  0,
}

events = []

# ── A: Inspections ─────────────────────────────────────────────────────────
insp_our["Inspection End Date"] = pd.to_datetime(insp_our["Inspection End Date"], errors="coerce")
insp_our = insp_our.dropna(subset=["Inspection End Date"])

for _, row in insp_our.iterrows():
    fei  = int(row["_fei_int"])
    clf  = str(row["Classification"])
    if "Official Action" in clf or "OAI" in clf:
        subtype = "OAI"; sev_key = "Inspection_OAI"
    elif "Voluntary Action" in clf or "VAI" in clf:
        subtype = "VAI"; sev_key = "Inspection_VAI"
    else:
        subtype = "NAI"; sev_key = "Inspection_NAI"

    area        = safe_str(row.get("Project Area", ""))
    prod        = safe_str(row.get("Product Type", ""))
    city        = safe_str(row.get("City", ""))
    state       = safe_str(row.get("State", ""))
    insp_id     = safe_str(row.get("Inspection ID", ""))
    posted_cit  = safe_str(row.get("Posted Citations", ""))
    fiscal_yr   = safe_str(row.get("Fiscal Year", ""))

    detail_parts = [subtype, area]
    if prod and prod not in ("nan",):
        detail_parts.append(prod)

    events.append({
        "fei":              fei,
        "firm_name":        get_firm(fei),
        "country":          get_country(fei),
        "event_date":       row["Inspection End Date"].date(),
        "event_year":       row["Inspection End Date"].year,
        "event_type":       "Inspection",
        "event_subtype":    subtype,
        "severity_num":     SEV[sev_key],
        "key_details":      " · ".join(p for p in detail_parts if p),
        "source":           "Inspection DB",
        # Inspection-specific enrichment
        "inspection_id":    insp_id,
        "city":             city,
        "state":            state,
        "product_type":     prod,
        "program_area":     area,
        "posted_citations": posted_cit,
        "fiscal_year":      fiscal_yr,
    })

print(f"  Inspection events: {len(events)}")


# ── B: 483 PDFs ──────────────────────────────────────────────────────────
inv483 = pd.read_csv(INV_483)
inv483["fei"] = inv483["fei"].astype("Int64")
inv483 = inv483[inv483["fei"].isin(ref_feis)].copy()
inv483["insp_date"] = pd.to_datetime(inv483["insp_date"], errors="coerce")
inv483 = inv483.dropna(subset=["insp_date"])

fei_483_feats = pd.read_csv(OBS_483) if OBS_483.exists() else pd.DataFrame()
if not fei_483_feats.empty:
    fei_483_feats["fei"] = pd.to_numeric(fei_483_feats["fei"], errors="coerce").astype("Int64")

n_before = len(events)
for _, row in inv483.iterrows():
    fei   = int(row["fei"])
    n_obs = int(row.get("n_observations", 0))
    firm  = safe_str(row.get("firm_name", "")) or get_firm(fei)

    details = f"483 · {n_obs} observations"
    if not fei_483_feats.empty:
        frow = fei_483_feats[fei_483_feats["fei"] == fei]
        if not frow.empty:
            sig = []
            for col, label in [("ever_data_integrity","DataIntegrity"),
                                ("ever_repeat","Repeat"),
                                ("ever_contamination","Contamination"),
                                ("ever_systemic","Systemic")]:
                if col in frow.columns and bool(frow.iloc[0][col]):
                    sig.append(label)
            if sig:
                details += " · " + ", ".join(sig)

    events.append({
        "fei":              fei,
        "firm_name":        firm if firm and firm != "nan" else get_firm(fei),
        "country":          get_country(fei),
        "event_date":       row["insp_date"].date(),
        "event_year":       row["insp_date"].year,
        "event_type":       "483",
        "event_subtype":    f"{n_obs} obs",
        "severity_num":     SEV["483"],
        "key_details":      details,
        "source":           "483 PDF",
        "inspection_id": "", "city": "", "state": "",
        "product_type": "", "program_area": "",
        "posted_citations": "", "fiscal_year": "",
    })

print(f"  483 events: {len(events) - n_before}")


# ── C: Warning Letters ───────────────────────────────────────────────────
wl_rec = pd.read_csv(WL_REC)
fei_col_wl = "search_fei" if "search_fei" in wl_rec.columns else "primary_fei"
wl_rec["_fei"] = pd.to_numeric(wl_rec[fei_col_wl], errors="coerce").astype("Int64")
wl_rec = wl_rec[wl_rec["_fei"].isin(ref_feis)].copy()
wl_rec["wl_date"] = pd.to_datetime(wl_rec["wl_date"], errors="coerce")
wl_rec = wl_rec.dropna(subset=["wl_date"])

n_before = len(events)
for _, row in wl_rec.iterrows():
    fei     = int(row["_fei"])
    n_viol  = int(row.get("n_violations", 0))
    rep_fac = bool(row.get("has_repeat_at_facility", False))
    rep_mul = bool(row.get("has_repeat_multi_site", False))
    details = f"WL · {n_viol} violations"
    if rep_fac: details += " · RepeatAtFacility"
    if rep_mul: details += " · RepeatMultiSite"
    if bool(row.get("has_management_oversight", False)):
        details += " · MgmtOversight"

    events.append({
        "fei":              fei,
        "firm_name":        get_firm(fei),
        "country":          get_country(fei),
        "event_date":       row["wl_date"].date(),
        "event_year":       row["wl_date"].year,
        "event_type":       "Warning Letter",
        "event_subtype":    f"WL {row.get('wl_number','?')}",
        "severity_num":     SEV["Warning Letter"],
        "key_details":      details,
        "source":           "Warning Letter",
        "inspection_id": "", "city": "", "state": "",
        "product_type": "", "program_area": "",
        "posted_citations": "", "fiscal_year": "",
    })

print(f"  Warning Letter events: {len(events) - n_before}")


# ── D: Recalls ───────────────────────────────────────────────────────────
# Note: recall_filtered.csv contains API-matched recalls only (filtered by
# Product Description to our 14 study APIs in 20260316_recall_features.py).
# Facility-level drug recalls (all products) are retained in that script for
# reference but only API-matched rows are written to this CSV.
rec_filt = pd.read_csv(REC_FILT)
rec_filt["_fei"] = pd.to_numeric(rec_filt["FEI Number"], errors="coerce").astype("Int64")
rec_filt = rec_filt[rec_filt["_fei"].isin(ref_feis)].copy()

date_col_rec = "Recall_Date" if "Recall_Date" in rec_filt.columns else "Center Classification Date"
rec_filt["_date"] = pd.to_datetime(rec_filt[date_col_rec], errors="coerce")
rec_filt = rec_filt.dropna(subset=["_date"])

n_before = len(events)
for _, row in rec_filt.iterrows():
    fei     = int(row["_fei"])
    clf     = str(row.get("Event Classification", ""))
    reason  = str(row.get("Reason for Recall", ""))[:100]

    clf_strip = clf.strip()
    if clf_strip == "Class I":
        subtype = "Class I";   sev_key = "Recall_ClassI"
    elif clf_strip == "Class II":
        subtype = "Class II";  sev_key = "Recall_ClassII"
    else:
        subtype = "Class III"; sev_key = "Recall_ClassIII"

    events.append({
        "fei":              fei,
        "firm_name":        str(row.get("Recalling Firm Name", get_firm(fei)))[:60],
        "country":          str(row.get("Recalling Firm Country", get_country(fei))),
        "event_date":       row["_date"].date(),
        "event_year":       row["_date"].year,
        "event_type":       "Recall",
        "event_subtype":    subtype,
        "severity_num":     SEV[sev_key],
        "key_details":      f"Recall {subtype} · {reason[:80]}",
        "source":           "Recall DB",
        "inspection_id": "", "city": "", "state": "",
        "product_type": "", "program_area": "",
        "posted_citations": "", "fiscal_year": "",
    })

print(f"  Recall events: {len(events) - n_before}")


# ── E: Import Refusals ───────────────────────────────────────────────────
# Note: import_refusal_filtered.csv contains API-matched refusals only (filtered
# by Product Code and Description to our 14 study APIs in 20260316_import_refusal_features.py).
# Facility-level drug refusals (all products) are retained in that script for reference.
imp_filt = pd.read_csv(IMP_FILT)
imp_filt["_fei"] = pd.to_numeric(imp_filt["FEI Number"], errors="coerce").astype("Int64")
imp_filt = imp_filt[imp_filt["_fei"].isin(ref_feis)].copy()

date_col_imp = "Refused_Date" if "Refused_Date" in imp_filt.columns else "Refused Date"
imp_filt["_date"] = pd.to_datetime(imp_filt[date_col_imp], errors="coerce")
imp_filt = imp_filt.dropna(subset=["_date"])

n_before = len(events)
for _, row in imp_filt.iterrows():
    fei     = int(row["_fei"])
    charges = str(row.get("Refusal Charges", ""))
    lab_flag = any(
        str(row.get(c, "")).strip().lower() not in ["nan", "", "none", "no"]
        for c in ["FDA Sample Analysis", "Private Lab Analysis"]
    )
    details = f"Refusal · Charges: {charges}"
    if lab_flag:
        details += " · LabConfirmed"

    events.append({
        "fei":              fei,
        "firm_name":        str(row.get("Firm Legal Name", get_firm(fei)))[:60],
        "country":          get_country(fei),
        "event_date":       row["_date"].date(),
        "event_year":       row["_date"].year,
        "event_type":       "Import Refusal",
        "event_subtype":    charges[:30],
        "severity_num":     SEV["Import Refusal"],
        "key_details":      details[:120],
        "source":           "Import Refusal DB",
        "inspection_id": "", "city": "", "state": "",
        "product_type": "", "program_area": "",
        "posted_citations": "", "fiscal_year": "",
    })

print(f"  Import Refusal events: {len(events) - n_before}")


# ── Finalize events dataframe ─────────────────────────────────────────────
events_df = pd.DataFrame(events)
events_df["event_date"] = pd.to_datetime(events_df["event_date"])
events_df = events_df.sort_values(["fei", "event_date"]).reset_index(drop=True)
events_df.to_csv(OUT / "fei_events_timeline.csv", index=False)
print(f"\nTotal events: {len(events_df)}")
print(events_df.groupby("event_type").size().to_string())


# ══════════════════════════════════════════════════════════════════════════
# 4. BUILD NODE SUMMARY  (one row per FEI)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 4 — Building node summary")
print("="*65)

node_rows = []
for fei in sorted(ref_feis):
    ev = events_df[events_df["fei"] == fei]

    by_type = ev.groupby("event_type").size().to_dict()
    insp_ev = ev[ev["event_type"] == "Inspection"]
    rec_ev  = ev[ev["event_type"] == "Recall"]

    n_oai = int((insp_ev["event_subtype"] == "OAI").sum())
    n_vai = int((insp_ev["event_subtype"] == "VAI").sum())
    n_nai = int((insp_ev["event_subtype"] == "NAI").sum())

    has_wl    = by_type.get("Warning Letter", 0) > 0
    has_oai   = n_oai > 0
    has_class1= bool((rec_ev["event_subtype"] == "Class I").any())

    if has_wl:
        worst = "Warning Letter"
    elif has_oai:
        worst = "OAI"
    elif by_type.get("Recall", 0) > 0 and has_class1:
        worst = "Class I Recall"
    elif n_vai > 0:
        worst = "VAI"
    elif n_nai > 0:
        worst = "NAI"
    elif by_type.get("Import Refusal", 0) > 0:
        worst = "Import Refusal Only"
    else:
        worst = "No Regulatory Events"

    sev_score = (
        by_type.get("Warning Letter", 0) * 10
        + n_oai * 6
        + has_class1 * 5
        + n_vai * 2
        + by_type.get("483", 0) * 1
        + min(by_type.get("Recall", 0), 10) * 0.5
        + min(by_type.get("Import Refusal", 0), 20) * 0.1
    )

    first_date = ev["event_date"].min() if len(ev) else None
    last_date  = ev["event_date"].max() if len(ev) else None

    node_rows.append({
        "fei":                  fei,
        "firm_name":            get_firm(fei),
        "country":              get_country(fei),
        "n_events_total":       len(ev),
        "n_inspections":        by_type.get("Inspection", 0),
        "n_oai":                n_oai,
        "n_vai":                n_vai,
        "n_nai":                n_nai,
        "n_483s":               by_type.get("483", 0),
        "n_warning_letters":    by_type.get("Warning Letter", 0),
        "n_recalls":            by_type.get("Recall", 0),
        "n_class_I_recalls":    int((rec_ev["event_subtype"] == "Class I").sum()),
        "n_import_refusals":    by_type.get("Import Refusal", 0),
        "has_wl":               has_wl,
        "has_oai":              has_oai,
        "has_class_I_recall":   has_class1,
        "worst_outcome":        worst,
        "severity_score":       round(sev_score, 2),
        "first_event_date":     str(first_date)[:10] if first_date else "",
        "last_event_date":      str(last_date)[:10]  if last_date  else "",
    })

nodes_df = pd.DataFrame(node_rows)
nodes_df.to_csv(OUT / "fei_node_summary.csv", index=False)
print(f"Node summary: {len(nodes_df)} FEIs")
print(nodes_df.groupby("worst_outcome").size().sort_values(ascending=False).to_string())


# ══════════════════════════════════════════════════════════════════════════
# 5. BUILD EDGE LIST
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 5 — Building edge list")
print("="*65)

edges = []

if WL_NET.exists():
    wl_net = pd.read_csv(WL_NET)
    for _, row in wl_net.iterrows():
        fei_a = int(row["fei_a"])
        fei_b = int(row["fei_b"])
        etype = str(row.get("edge_type", "wl_link"))
        edate = str(row.get("wl_date", ""))[:10]
        desc  = str(row.get("description", ""))
        edges.append({
            "fei_a":       fei_a,
            "fei_b":       fei_b,
            "edge_type":   "wl_cross_site" if "cross_site" in etype else "wl_repeat_multi",
            "edge_date":   edate,
            "weight":      3,
            "label":       f"WL {edate[:7]}",
            "description": desc,
        })
    print(f"  WL cross-site edges: {len(wl_net)}")
else:
    print("  WL network file not found — skipping WL edges")

firms_in_insp = insp_our.groupby("_fei_int")["Legal Name"].agg(
    lambda x: x.mode()[0] if len(x) else ""
).reset_index()
firms_in_insp.columns = ["fei", "firm_name"]
firms_in_insp = firms_in_insp[firms_in_insp["fei"].isin(ref_feis)]

def normalize_name(name):
    name = str(name).lower()
    for suffix in [" ltd", " limited", " inc", " corp", " llc", " pvt",
                   " co.", " pharmaceuticals", " pharma", " labs",
                   " laboratories", " industries", " healthcare"]:
        name = name.replace(suffix, "")
    return re.sub(r"[^a-z0-9]", "", name)

same_co_pairs = set()
fei_firm_list = [(int(r["fei"]), normalize_name(r["firm_name"]), r["firm_name"])
                 for _, r in firms_in_insp.iterrows() if r["firm_name"]]

for i in range(len(fei_firm_list)):
    for j in range(i + 1, len(fei_firm_list)):
        fei_a, norm_a, full_a = fei_firm_list[i]
        fei_b, norm_b, full_b = fei_firm_list[j]
        if fei_a == fei_b:
            continue
        min_len = min(len(norm_a), len(norm_b))
        if min_len < 5:
            continue
        shorter = norm_a if len(norm_a) <= len(norm_b) else norm_b
        longer  = norm_b if len(norm_a) <= len(norm_b) else norm_a
        if len(shorter) >= 8 and shorter in longer:
            pair = tuple(sorted([fei_a, fei_b]))
            if pair not in same_co_pairs:
                same_co_pairs.add(pair)
                edges.append({
                    "fei_a":       fei_a,
                    "fei_b":       fei_b,
                    "edge_type":   "same_company",
                    "edge_date":   "",
                    "weight":      1,
                    "label":       "Same Co.",
                    "description": f"{full_a} ↔ {full_b}",
                })

print(f"  Same-company edges: {len(same_co_pairs)}")

edges_df = pd.DataFrame(edges)
edges_df = edges_df[edges_df["fei_a"] != edges_df["fei_b"]]
edges_df = (edges_df
            .sort_values("weight", ascending=False)
            .drop_duplicates(subset=["fei_a", "fei_b"])
            .reset_index(drop=True))

edges_df.to_csv(OUT / "fei_edge_list.csv", index=False)
print(f"  Total edges (deduplicated): {len(edges_df)}")
print(edges_df.groupby("edge_type").size().to_string())


# ══════════════════════════════════════════════════════════════════════════
# 6. BUILD CFR DATA PER FEI
#    Source: Inspections Citations Details (CFR Number, Short Description)
#    Output: fei_cfr_data.json with frequencies, domains, co-occurrence
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 6 — Building CFR citation data")
print("="*65)

DOMAIN_LABELS = {
    "org_personnel":    "Org. & Personnel (211.22–34)",
    "bldg_equipment":   "Buildings & Equipment (211.42–72)",
    "production":       "Production Controls (211.80–115)",
    "pkg_labeling":     "Packaging & Labeling (211.122–137)",
    "lab_controls":     "Lab Controls (211.160–176)",
    "records_reports":  "Records & Reports (211.180–198)",
    "other_211":        "Other 21 CFR 211",
    "non_211":          "Other Regulations",
}

def cfr_domain(cfr_str):
    """Map Act/CFR Number to official FDA Part 211 subpart domain."""
    m = re.search(r"211\.(\d+)", str(cfr_str))
    if not m:
        return "non_211"
    num = int(m.group(1))
    if 22  <= num <= 34:  return "org_personnel"
    if 42  <= num <= 72:  return "bldg_equipment"
    if 80  <= num <= 115: return "production"
    if 122 <= num <= 137: return "pkg_labeling"
    if 160 <= num <= 176: return "lab_controls"
    if 180 <= num <= 198: return "records_reports"
    return "other_211"

if CIT_RAW.exists():
    cit_raw = pd.read_excel(CIT_RAW)
    cit_raw["FEI Number"] = pd.to_numeric(cit_raw["FEI Number"], errors="coerce").astype("Int64")
    cit_our = cit_raw[cit_raw["FEI Number"].isin(ref_feis)].copy()
    print(f"  Citation rows for our FEIs: {len(cit_our)}")

    # Add domain column
    cit_our["domain"] = cit_our["Act/CFR Number"].apply(cfr_domain)
    cit_our["domain_label"] = cit_our["domain"].map(DOMAIN_LABELS)

    cfr_data = {}
    n_feis_with_cit = 0

    for fei_val, grp in cit_our.groupby("FEI Number"):
        fei = str(int(fei_val))
        n_feis_with_cit += 1

        # Top 15 most-cited CFRs
        cfr_agg = (
            grp.groupby("Act/CFR Number")
               .agg(count=("Act/CFR Number", "count"),
                    short=("Short Description", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
                    domain=("domain", "first"))
               .sort_values("count", ascending=False)
               .reset_index()
        )
        cfrs = [
            {
                "cfr":   str(r["Act/CFR Number"]),
                "short": str(r["short"])[:70],
                "count": int(r["count"]),
                "domain": str(r["domain"]),
            }
            for _, r in cfr_agg.head(15).iterrows()
        ]

        # Domain-level counts
        domain_agg = (
            grp.groupby("domain_label")
               .size()
               .reset_index(name="count")
               .sort_values("count", ascending=False)
        )
        domains = [
            {"domain": str(r["domain_label"]), "count": int(r["count"])}
            for _, r in domain_agg.iterrows()
        ]

        # CFR co-occurrence: which CFR pairs appear together in same inspection
        cooccur = {}
        for insp_id, igrp in grp.groupby("Inspection ID"):
            cfr_list = sorted(igrp["Act/CFR Number"].dropna().astype(str).unique().tolist())
            for i in range(len(cfr_list)):
                for j in range(i + 1, len(cfr_list)):
                    pair = (cfr_list[i], cfr_list[j])
                    cooccur[pair] = cooccur.get(pair, 0) + 1

        cooccur_sorted = sorted(cooccur.items(), key=lambda x: -x[1])[:12]
        cooccur_out = [
            {
                "a": p[0], "b": p[1], "count": c,
                "a_dom": cfr_domain(p[0]), "b_dom": cfr_domain(p[1])
            }
            for p, c in cooccur_sorted
        ]

        # Per-inspection summary (for the CFR inspector-level view)
        insp_summary = []
        for insp_id, igrp in grp.groupby("Inspection ID"):
            end_date = igrp["Inspection End Date"].iloc[0] if "Inspection End Date" in igrp.columns else ""
            insp_summary.append({
                "insp_id": str(insp_id),
                "date":    str(end_date)[:10],
                "n_cfr":   int(len(igrp)),
                "cfrs":    ", ".join(igrp["Act/CFR Number"].dropna().astype(str).unique().tolist()[:6]),
            })
        insp_summary.sort(key=lambda x: x["date"], reverse=True)

        cfr_data[fei] = {
            "n_insp_with_cit":  int(grp["Inspection ID"].nunique()),
            "n_total_cit":      int(len(grp)),
            "n_unique_cfr":     int(grp["Act/CFR Number"].nunique()),
            "cfrs":             cfrs,
            "domains":          domains,
            "cooccurrence":     cooccur_out,
            "inspections":      insp_summary[:20],  # top 20 most recent
        }

    with open(OUT / "fei_cfr_data.json", "w") as f:
        json.dump(cfr_data, f)
    print(f"  CFR data saved: {n_feis_with_cit} FEIs with citation records")
else:
    print(f"  Citation Details file not found: {CIT_RAW}")
    print("  Skipping CFR data export")

print(f"\nAll outputs saved to: {OUT}")
print("  fei_events_timeline.csv")
print("  fei_node_summary.csv")
print("  fei_edge_list.csv")
print("  fei_cfr_data.json")
