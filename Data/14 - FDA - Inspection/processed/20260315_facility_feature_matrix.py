"""
20260315_facility_feature_matrix.py

Active FDA inspection/citation pipeline, step 2 of 2.

Purpose
-------
Build a per-facility (FEI) feature matrix from FDA Dashboard inspections and
citations for use in downstream quality-signal analysis.

Inputs
------
Produced by 20260315_cfr_cooccurrence_analysis.py:
- inspection_details_filtered.csv
  Drug inspection rows for project FEIs.
- citations_with_classification.csv
  Citation rows already mapped to project areas and inspection classes.
- cfr_by_classification.csv
  Drug QA CFR frequency table with NAI/VAI/OAI counts, Total, and OAI_share.

External input:
- ../../08 - Valisure/raw/FEIs_March 2026.xlsx
  sheet "API Only_FEI Mapping": maps FEIs to API names.

Output
------
- facility_feature_matrix.csv
  One row per FEI with Drug QA inspection history, Bioresearch inspection
  history, Drug QA citation-domain counts, OAI-predictive CFR citation counts,
  repeat-CFR rate, and API labels.

Pipeline order
--------------
1. Run 20260315_cfr_cooccurrence_analysis.py.
2. Run this script.
3. Downstream users include 06 - Metformin Data quality-signal correlation and
   99 - Outputs - MQRI prompt/materials.

Citation-to-project-area mapping
--------------------------------
This script relies on the upstream Mapped_Area column. Upstream rules are:
- 21 CFR 211.xxx -> Drug Quality Assurance
- 21 CFR 312/314/320.xxx -> Bioresearch Monitoring
- unmapped FDCA records are excluded.

Inspection classification assignment
------------------------------------
This script uses the upstream citation Classification/Class values, where each
citation was matched to the classification for its own project area within the
inspection. For inspection-history features, it separately summarizes the
Drug Quality Assurance and Bioresearch Monitoring inspection rows rather than
collapsing an inspection to a single worst-class row.

OAI-predictive CFR feature
--------------------------
Do not maintain a manual hard-coded CFR list here. OAI_PREDICTIVE_CFRS is
derived from cfr_by_classification.csv, which is Drug QA only. A CFR is counted
as OAI-predictive when:
- Total >= 5
- OAI_share >= 0.33

The resulting n_oai_predictive_cfrs feature counts Drug QA citation rows whose
Act/CFR Number is in that derived set.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).parent
OAI_PREDICTIVE_MIN_TOTAL = 5
OAI_PREDICTIVE_MIN_SHARE = 0.33

# ── Load filtered data ──────────────────────────────────────────────────────
insp = pd.read_csv(OUT / "inspection_details_filtered.csv", parse_dates=["Inspection End Date"])
cit  = pd.read_csv(OUT / "citations_with_classification.csv")   # output of cfr_cooccurrence_analysis

# FEI → API mapping
fei_raw = pd.read_excel(
    BASE / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx",
    sheet_name="API Only_FEI Mapping"
)
fei_df = (
    fei_raw[["API", "FEI_NUMBER"]]
    .dropna(subset=["FEI_NUMBER"])
    .query("API != 'Grand Total'")
    .assign(FEI_NUMBER=lambda x: x["FEI_NUMBER"].astype(int))
)
fei_to_api = fei_df.groupby("FEI_NUMBER")["API"].apply(
    lambda x: "; ".join(sorted(x.unique()))
).to_dict()

# ── Per-area inspection tables ──────────────────────────────────────────────
def simplify(c):
    if "OAI" in str(c): return "OAI"
    if "VAI" in str(c): return "VAI"
    if "NAI" in str(c): return "NAI"
    return "Unknown"

AREAS = ["Drug Quality Assurance", "Bioresearch Monitoring"]

insp_by_area = {}
for area in AREAS:
    sub = (
        insp[insp["Project Area"] == area]
        .drop_duplicates(subset=["Inspection ID"])
        .copy()
    )
    sub["Class"] = sub["Classification"].apply(simplify)
    insp_by_area[area] = sub[sub["Class"] != "Unknown"]

# Separate citation tables by area (already mapped in cfr_cooccurrence_analysis)
cit_dqa = cit[cit["Mapped_Area"] == "Drug Quality Assurance"].copy()
cit_bio = cit[cit["Mapped_Area"] == "Bioresearch Monitoring"].copy()

# ── CFR domain mapping (Drug QA / 21 CFR 211 only) ─────────────────────────
def cfr_domain(cfr):
    cfr = str(cfr)
    if "211.2" in cfr:                                                  return "org_personnel"
    if any(s in cfr for s in ["211.4", "211.5", "211.6"]):             return "buildings_equipment"
    if any(s in cfr for s in ["211.8", "211.9", "211.10", "211.11"]):  return "production_process"
    if any(s in cfr for s in ["211.12", "211.13", "211.14"]):          return "lab_qc"
    if any(s in cfr for s in ["211.15", "211.16", "211.17", "211.18", "211.19"]): return "lab_records"
    if "211.7" in cfr:                                                  return "components"
    if "211.22" in cfr or "211.68" in cfr:                             return "quality_data_systems"
    return "other"

cit_dqa = cit_dqa.copy()
cit_dqa["Domain"] = cit_dqa["Act/CFR Number"].apply(cfr_domain)

# OAI-predictive CFR codes are derived from the upstream Drug QA summary.
# Threshold: Drug QA only, Total >= 5, OAI_share >= 0.33.
cfr_by_class = pd.read_csv(OUT / "cfr_by_classification.csv")
required_cols = {"Act/CFR Number", "Total", "OAI_share"}
missing_cols = required_cols - set(cfr_by_class.columns)
if missing_cols:
    raise ValueError(
        "cfr_by_classification.csv is missing required columns: "
        + ", ".join(sorted(missing_cols))
    )

oai_predictive_rows = cfr_by_class[
    (cfr_by_class["Total"] >= OAI_PREDICTIVE_MIN_TOTAL)
    & (cfr_by_class["OAI_share"] >= OAI_PREDICTIVE_MIN_SHARE)
].copy()
OAI_PREDICTIVE_CFRS = set(oai_predictive_rows["Act/CFR Number"])
print(
    "Derived OAI-predictive CFRs: "
    f"{len(OAI_PREDICTIVE_CFRS)} unique Act/CFR Numbers "
    f"(Drug QA; Total >= {OAI_PREDICTIVE_MIN_TOTAL}; "
    f"OAI_share >= {OAI_PREDICTIVE_MIN_SHARE})"
)

# ── Build per-FEI feature matrix ────────────────────────────────────────────
all_feis = set(insp["FEI Number"].unique())
rows = []

for fei in all_feis:
    # ── Drug QA inspection history ──────────────────────────────────────────
    dqa_grp = insp_by_area["Drug Quality Assurance"]
    dqa_grp = dqa_grp[dqa_grp["FEI Number"] == fei].sort_values("Inspection End Date")

    n_dqa        = len(dqa_grp)
    n_dqa_oai    = (dqa_grp["Class"] == "OAI").sum()
    n_dqa_vai    = (dqa_grp["Class"] == "VAI").sum()
    n_dqa_nai    = (dqa_grp["Class"] == "NAI").sum()
    oai_rate_dqa = n_dqa_oai / n_dqa if n_dqa > 0 else np.nan
    latest_dqa   = dqa_grp.iloc[-1]["Class"] if n_dqa > 0 else "None"
    latest_date  = dqa_grp.iloc[-1]["Inspection End Date"] if n_dqa > 0 else pd.NaT

    # Avg gap between Drug QA inspections
    if n_dqa > 1:
        avg_gap_dqa = dqa_grp["Inspection End Date"].diff().dt.days.dropna().mean()
    else:
        avg_gap_dqa = np.nan

    # ── Bioresearch inspection history ──────────────────────────────────────
    bio_grp = insp_by_area["Bioresearch Monitoring"]
    bio_grp = bio_grp[bio_grp["FEI Number"] == fei]

    n_bio        = len(bio_grp)
    n_bio_oai    = (bio_grp["Class"] == "OAI").sum()
    n_bio_vai    = (bio_grp["Class"] == "VAI").sum()
    oai_rate_bio = n_bio_oai / n_bio if n_bio > 0 else np.nan
    has_bio      = 1 if n_bio > 0 else 0

    # ── Drug QA citations ───────────────────────────────────────────────────
    fei_cit = cit_dqa[cit_dqa["FEI Number"] == fei]
    n_cit_total  = len(fei_cit)
    domain_counts = fei_cit["Domain"].value_counts()

    n_oai_pred   = fei_cit["Act/CFR Number"].isin(OAI_PREDICTIVE_CFRS).sum()

    cfr_per_insp  = fei_cit.groupby("Act/CFR Number")["Inspection ID"].nunique()
    n_cfr_unique  = len(cfr_per_insp)
    n_cfr_repeat  = (cfr_per_insp > 1).sum()
    repeat_rate   = n_cfr_repeat / n_cfr_unique if n_cfr_unique > 0 else 0

    # ── Bioresearch citations ───────────────────────────────────────────────
    bio_cit = cit_bio[cit_bio["FEI Number"] == fei]
    n_bio_cit = len(bio_cit)

    rows.append({
        "FEI_NUMBER":               fei,
        "API":                      fei_to_api.get(fei, "Unknown"),
        # Drug QA inspection features
        "n_dqa_inspections":        n_dqa,
        "n_dqa_OAI":                n_dqa_oai,
        "n_dqa_VAI":                n_dqa_vai,
        "n_dqa_NAI":                n_dqa_nai,
        "OAI_rate_dqa":             round(oai_rate_dqa, 3) if not np.isnan(oai_rate_dqa) else np.nan,
        "latest_dqa_class":         latest_dqa,
        "latest_dqa_date":          latest_date,
        "avg_dqa_gap_days":         round(avg_gap_dqa, 1) if not np.isnan(avg_gap_dqa) else np.nan,
        # Bioresearch inspection features
        "has_bioresearch_insp":     has_bio,
        "n_bio_inspections":        n_bio,
        "n_bio_OAI":                int(n_bio_oai),
        "n_bio_VAI":                int(n_bio_vai),
        "OAI_rate_bio":             round(oai_rate_bio, 3) if not np.isnan(oai_rate_bio) else np.nan,
        # Drug QA citation features (21 CFR 211)
        "n_dqa_citations":          n_cit_total,
        "n_cit_buildings_equip":    domain_counts.get("buildings_equipment", 0),
        "n_cit_production":         domain_counts.get("production_process", 0),
        "n_cit_lab_qc":             domain_counts.get("lab_qc", 0),
        "n_cit_lab_records":        domain_counts.get("lab_records", 0),
        "n_cit_org_personnel":      domain_counts.get("org_personnel", 0),
        "n_cit_quality_data":       domain_counts.get("quality_data_systems", 0),
        "n_oai_predictive_cfrs":    int(n_oai_pred),
        "n_unique_cfrs":            n_cfr_unique,
        "repeat_cfr_rate":          round(repeat_rate, 3),
        # Bioresearch citation features (21 CFR 314/312)
        "n_bio_citations":          n_bio_cit,
    })

feat_df = pd.DataFrame(rows).sort_values("OAI_rate_dqa", ascending=False)
feat_df.to_csv(OUT / "facility_feature_matrix.csv", index=False)

print("Facility Feature Matrix")
print(f"Shape: {feat_df.shape}")
print()
print(feat_df[[
    "FEI_NUMBER", "API",
    "n_dqa_inspections", "n_dqa_OAI", "OAI_rate_dqa",
    "has_bioresearch_insp", "n_bio_OAI",
    "n_dqa_citations", "n_oai_predictive_cfrs", "repeat_cfr_rate"
]].head(20).to_string(index=False))

print()
print("=== Correlation with Drug QA OAI rate ===")
num_cols = [
    "n_dqa_inspections", "n_dqa_citations", "n_oai_predictive_cfrs",
    "repeat_cfr_rate", "n_cit_buildings_equip", "n_cit_lab_qc",
    "has_bioresearch_insp", "n_bio_OAI", "n_bio_citations"
]
valid = feat_df[num_cols + ["OAI_rate_dqa"]].dropna(subset=["OAI_rate_dqa"])
corr = valid[num_cols + ["OAI_rate_dqa"]].corr()["OAI_rate_dqa"].drop("OAI_rate_dqa").sort_values(ascending=False)
print(corr.round(3).to_string())
print()
print(f"Saved to: {OUT / 'facility_feature_matrix.csv'}")
