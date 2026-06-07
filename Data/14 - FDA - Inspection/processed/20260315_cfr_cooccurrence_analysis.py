# %%
"""
20260315_cfr_cooccurrence_analysis.py

Active FDA inspection/citation pipeline, step 1 of 2.

Purpose
-------
Build the cleaned inspection/citation tables used by downstream warning-letter
and quality-signal analyses, then summarize Drug QA CFR frequency and
co-occurrence patterns.

Inputs
------
- ../../08 - Valisure/raw/FEIs_March 2026.xlsx
  sheet "API Only_FEI Mapping": API-to-FEI universe for this project.
- ../raw/Inspections Details.xlsx
  FDA Dashboard inspection rows.
- ../raw/Inspections Citations Details.xlsx
  FDA Dashboard citation rows.

Outputs
-------
- inspection_details_filtered.csv
  Drug inspection rows for project FEIs, before project-area filtering.
- citation_details_filtered.csv
  Drug citation rows for project FEIs, before CFR-area mapping.
- citations_with_classification.csv
  Mapped citations with the classification of their own project area.
  This is consumed by 21 - FDA - Warning Letter.
- citations_bioresearch.csv
  Reference subset for Bioresearch Monitoring citations.
- cfr_by_classification.csv
  Drug QA CFR counts by NAI/VAI/OAI with Total and OAI_share.
  This is the only source for OAI-predictive CFRs in step 2.
- cfr_cooccurrence_pairs_with_labels.csv
  Drug QA CFR pair counts within inspections, with short descriptions.
- api_classification_summary.csv
  API-level inspection classification summary by project area.

Pipeline order
--------------
1. Run this script from this folder or any working directory.
2. Run 20260315_facility_feature_matrix.py, which consumes this script's
   filtered citation/inspection outputs and cfr_by_classification.csv.

Citation-to-project-area mapping
--------------------------------
The CFR prefix determines the project area:
- 21 CFR 211.xxx -> Drug Quality Assurance
- 21 CFR 312/314/320.xxx -> Bioresearch Monitoring
- FDCA 582 / 505-1 records are unmapped and excluded from mapped citation
  outputs.

Inspection classification assignment
------------------------------------
Each citation is matched to the inspection classification for its own mapped
project area: (Inspection ID, Mapped_Area) -> Classification. If that exact
area row is missing, the script falls back to the most severe classification
available for the same inspection, ordered OAI > VAI > NAI.

OAI-share CFR analysis
----------------------
cfr_by_classification.csv is computed for Drug Quality Assurance citations
only. For each (Act/CFR Number, Short Description), the script counts NAI, VAI,
and OAI citations, computes Total = NAI + VAI + OAI, and computes
OAI_share = OAI / Total. Step 2 derives OAI-predictive CFR features from this
file using Drug QA only, Total >= 5, and OAI_share >= 0.33.

Notes on project areas
----------------------
Drug Quality Assurance covers manufacturing quality/cGMP observations.
Bioresearch Monitoring covers NDA/IND/post-market observations. Unapproved
and Misbranded Drugs rows are retained in inspection_details_filtered.csv but
excluded from mapped citation analysis because they do not contribute mapped
citations in this project data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

BASE = Path(__file__).resolve().parents[2]  # Data/
OUT  = Path(__file__).parent                # processed/

# ── 1. Load FEI list ────────────────────────────────────────────────────────
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
our_feis = set(fei_df["FEI_NUMBER"].unique())
print(f"Our FEIs: {len(our_feis)} unique, {fei_df['API'].nunique()} APIs")

# ── 2. Load and filter inspection / citation data ───────────────────────────
insp = pd.read_excel(BASE / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx")
cit  = pd.read_excel(BASE / "14 - FDA - Inspection" / "raw" / "Inspections Citations Details.xlsx")

insp_f = insp[insp["FEI Number"].isin(our_feis)].copy()
cit_f  = cit[cit["FEI Number"].isin(our_feis)].copy()

# Keep only Drug product type / program
insp_drug = insp_f[insp_f["Product Type"] == "Drugs"].copy()
cit_drug  = cit_f[cit_f["Program Area"] == "Drugs"].copy()

# Save filtered files (raw, before any area logic)
insp_drug.to_csv(OUT / "inspection_details_filtered.csv", index=False)
cit_drug.to_csv(OUT / "citation_details_filtered.csv", index=False)

print(f"\nInspection rows (drug, all areas): {len(insp_drug)}")
print(insp_drug["Project Area"].value_counts().to_string())
print(f"\nCitation rows (drug): {len(cit_drug)}")

# ── 3. Per-area inspection summary ─────────────────────────────────────────
# Keep all project area rows; do NOT collapse to one row per inspection.
# "Unapproved and Misbranded Drugs" has zero citations and covers marketing
# violations (not manufacturing quality) → excluded from citation analysis.
AREAS_OF_INTEREST = ["Drug Quality Assurance", "Bioresearch Monitoring"]

def simplify(c):
    if "OAI" in str(c): return "OAI"
    if "VAI" in str(c): return "VAI"
    if "NAI" in str(c): return "NAI"
    return "Unknown"

insp_areas = insp_drug[insp_drug["Project Area"].isin(AREAS_OF_INTEREST)].copy()
insp_areas["Class"] = insp_areas["Classification"].apply(simplify)

print("\n=== Unique inspections per Project Area ===")
for area in AREAS_OF_INTEREST:
    sub = insp_areas[insp_areas["Project Area"] == area]
    n_unique = sub["Inspection ID"].nunique()
    vc = sub.drop_duplicates("Inspection ID")["Class"].value_counts()
    print(f"\n  {area}: {n_unique} unique inspections")
    print(f"    " + "  ".join(f"{k}: {v}" for k, v in vc.items()))

# ── 4. Map each citation to its project area via CFR prefix ─────────────────
# The CFR number itself encodes which project area the citation belongs to.
def cfr_to_area(cfr):
    cfr = str(cfr)
    if "21 CFR 211" in cfr:
        return "Drug Quality Assurance"
    if any(x in cfr for x in ["21 CFR 312", "21 CFR 314", "21 CFR 320"]):
        return "Bioresearch Monitoring"
    return None   # FDCA 582, 505-1 → exclude (2 records)

cit_drug = cit_drug.copy()
cit_drug["Mapped_Area"] = cit_drug["Act/CFR Number"].apply(cfr_to_area)

# Drop unmapped (2 FDCA records)
cit_mapped = cit_drug[cit_drug["Mapped_Area"].notna()].copy()
print(f"\nCitations after area mapping: {len(cit_mapped)}")
print(cit_mapped["Mapped_Area"].value_counts().to_string())

# ── 5. Join each citation to the classification of its own project area ──────
# Build lookup: (Inspection ID, Project Area) → Classification
area_class = (
    insp_areas[["Inspection ID", "Project Area", "Classification"]]
    .drop_duplicates(subset=["Inspection ID", "Project Area"])
    .set_index(["Inspection ID", "Project Area"])["Classification"]
    .to_dict()
)

# Fallback: worst class across all areas for that inspection
CLASS_SEVERITY = {
    "Official Action Indicated (OAI)":  3,
    "Voluntary Action Indicated (VAI)": 2,
    "No Action Indicated (NAI)":        1,
}
fallback_class = (
    insp_areas
    .assign(sev=insp_areas["Classification"].map(CLASS_SEVERITY).fillna(0))
    .sort_values("sev", ascending=False)
    .groupby("Inspection ID")["Classification"]
    .first()
    .to_dict()
)

def get_classification(inspection_id, mapped_area):
    # First try exact (inspection, area) match
    key = (inspection_id, mapped_area)
    if key in area_class:
        return area_class[key]
    # Fallback: worst class across all areas
    return fallback_class.get(inspection_id, None)

cit_mapped["Classification"] = cit_mapped.apply(
    lambda r: get_classification(r["Inspection ID"], r["Mapped_Area"]), axis=1
)
cit_mapped["Class"] = cit_mapped["Classification"].apply(simplify)
cit_mapped = cit_mapped[cit_mapped["Class"] != "Unknown"]

print(f"\nCitations with matched classification: {len(cit_mapped)}")
print(cit_mapped["Class"].value_counts().to_string())
print("\nBreakdown by area:")
print(cit_mapped.groupby(["Mapped_Area", "Class"]).size().unstack(fill_value=0).to_string())

cit_mapped.to_csv(OUT / "citations_with_classification.csv", index=False)

# ── 6. Focus co-occurrence on Drug QA (21 CFR 211) ──────────────────────────
# These are the manufacturing quality citations most relevant to our goal.
# Bioresearch (21 CFR 314) citations are saved separately for reference.
cit_dqa = cit_mapped[cit_mapped["Mapped_Area"] == "Drug Quality Assurance"].copy()
cit_bio = cit_mapped[cit_mapped["Mapped_Area"] == "Bioresearch Monitoring"].copy()
cit_bio.to_csv(OUT / "citations_bioresearch.csv", index=False)

# ── 7. CFR frequency by classification (Drug QA only) ───────────────────────
cfr_class = (
    cit_dqa.groupby(["Act/CFR Number", "Short Description", "Class"])
    .size().reset_index(name="count")
)
cfr_pivot = cfr_class.pivot_table(
    index=["Act/CFR Number", "Short Description"],
    columns="Class", values="count", fill_value=0
).reset_index()

for col in ["NAI", "VAI", "OAI"]:
    if col not in cfr_pivot.columns:
        cfr_pivot[col] = 0

cfr_pivot["Total"]     = cfr_pivot["NAI"] + cfr_pivot["VAI"] + cfr_pivot["OAI"]
cfr_pivot["OAI_share"] = cfr_pivot["OAI"] / cfr_pivot["Total"].replace(0, np.nan)
cfr_pivot = cfr_pivot.sort_values("Total", ascending=False)

cfr_pivot.to_csv(OUT / "cfr_by_classification.csv", index=False)
print(f"\nTop 20 Drug QA CFRs by total frequency:")
print(cfr_pivot.head(20)[
    ["Act/CFR Number", "Short Description", "NAI", "VAI", "OAI", "Total", "OAI_share"]
].to_string(index=False))

# ── 8. OAI-predictive CFRs (Drug QA, min 5 occurrences) ─────────────────────
oai_predictive = cfr_pivot[cfr_pivot["Total"] >= 5].sort_values("OAI_share", ascending=False)
print(f"\nTop Drug QA CFRs by OAI share (min 5 occurrences):")
print(oai_predictive.head(15)[
    ["Act/CFR Number", "Short Description", "OAI", "Total", "OAI_share"]
].to_string(index=False))

# ── 9. Co-occurrence within inspections (Drug QA citations only) ─────────────
insp_cfr = (
    cit_dqa.groupby("Inspection ID")["Act/CFR Number"]
    .apply(list).reset_index(name="cfr_list")
)
# Attach classification (use Drug QA classification per inspection)
dqa_class = (
    insp_areas[insp_areas["Project Area"] == "Drug Quality Assurance"]
    [["Inspection ID", "Classification"]]
    .drop_duplicates("Inspection ID")
    .assign(Class=lambda x: x["Classification"].apply(simplify))
)
insp_cfr = insp_cfr.merge(dqa_class[["Inspection ID", "Class"]], on="Inspection ID", how="left")

cooccur = {}
for _, row in insp_cfr.iterrows():
    cfrs = sorted(set(row["cfr_list"]))
    for a, b in combinations(cfrs, 2):
        cooccur[(a, b)] = cooccur.get((a, b), 0) + 1

cooccur_df = pd.DataFrame(
    [(a, b, cnt) for (a, b), cnt in cooccur.items()],
    columns=["CFR_A", "CFR_B", "co_count"]
).sort_values("co_count", ascending=False)

# Add Short Description labels (take first match per CFR)
cfr_to_short = (
    cit_dqa[["Act/CFR Number", "Short Description"]]
    .drop_duplicates(subset=["Act/CFR Number"])
    .set_index("Act/CFR Number")["Short Description"]
    .to_dict()
)
cooccur_df["Short_A"] = cooccur_df["CFR_A"].map(cfr_to_short)
cooccur_df["Short_B"] = cooccur_df["CFR_B"].map(cfr_to_short)
cooccur_df.to_csv(OUT / "cfr_cooccurrence_pairs_with_labels.csv", index=False)

print(f"\nTop 20 co-occurring Drug QA CFR pairs:")
print(cooccur_df.head(20)[["Short_A", "Short_B", "co_count"]].to_string(index=False))

# ── 10. API-level summary (per project area) ─────────────────────────────────
fei_to_api = fei_df.set_index("FEI_NUMBER")["API"].to_dict()

api_rows = []
for area in AREAS_OF_INTEREST:
    sub = insp_areas[insp_areas["Project Area"] == area].drop_duplicates("Inspection ID")
    sub = sub.copy()
    sub["API"] = sub["FEI Number"].map(fei_to_api)
    grp = sub.groupby(["API", "Class"]).size().unstack(fill_value=0).reset_index()
    for col in ["NAI", "VAI", "OAI"]:
        if col not in grp.columns: grp[col] = 0
    grp["Total"]    = grp["NAI"] + grp["VAI"] + grp["OAI"]
    grp["OAI_rate"] = grp["OAI"] / grp["Total"].replace(0, np.nan)
    grp["Area"]     = area
    api_rows.append(grp)

api_summary = pd.concat(api_rows, ignore_index=True).sort_values(["Area", "OAI_rate"], ascending=[True, False])
api_summary.to_csv(OUT / "api_classification_summary.csv", index=False)
print(f"\nAPI-level summary (Drug QA):")
print(api_summary[api_summary["Area"] == "Drug Quality Assurance"].to_string(index=False))

print("\nAll outputs saved to:", OUT)
