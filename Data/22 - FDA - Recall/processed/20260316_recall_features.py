# %%
"""
20260316_recall_features.py

Extracts per-FEI recall features from FDA Recall database.

Filters to our 129 reference FEIs, then computes:
  - Recall counts by severity class (I / II / III)
  - Recall status distribution
  - Text analysis of Reason for Recall (contamination categories)
  - Temporal features (recency, frequency)

Outputs:
  - recall_filtered.csv        — all recall rows for our FEIs
  - recall_fei_features.csv    — aggregated per-FEI features
  - recall_reason_categories.csv — category distribution
"""

import pandas as pd
import re
from pathlib import Path

BASE  = Path(__file__).parents[3]
RAW   = BASE / "Data/22 - FDA - Recall/raw/Recall Data.xlsx"
FEI_F = BASE / "Data/08 - Valisure/raw/FEIs_March 2026.xlsx"
OUT   = Path(__file__).parent

# ── Load reference FEIs ─────────────────────────────────────────────────────
fei_df = pd.read_excel(FEI_F, sheet_name="API Only_FEI Mapping")
fei_df = fei_df[fei_df["API"] != "Grand Total"]
our_feis = set(fei_df["FEI_NUMBER"].dropna().astype(int).astype(str))
print(f"Reference FEIs: {len(our_feis)}")

# ── Load recall data ─────────────────────────────────────────────────────────
print("Loading Recall Data.xlsx …")
rec = pd.read_excel(RAW)
rec["FEI Number"] = rec["FEI Number"].astype(str).str.strip()
print(f"Total recall rows: {len(rec):,}  |  Unique FEIs: {rec['FEI Number'].nunique():,}")

# ── Filter to our FEIs ───────────────────────────────────────────────────────
our_rec = rec[rec["FEI Number"].isin(our_feis)].copy()
print(f"Rows for our FEIs: {len(our_rec):,}  |  Unique FEIs: {our_rec['FEI Number'].nunique()}")

# Keep only drug recalls (Product Type = DRUG or Center = CDER)
drug_mask = (
    our_rec["Product Type"].str.upper().str.contains("DRUG", na=False) |
    our_rec["Center"].str.upper().str.contains("CDER", na=False)
)
drug_rec_all = our_rec[drug_mask].copy()
print(f"Drug-type recall rows (all products):  {len(drug_rec_all):,}  |  FEIs: {drug_rec_all['FEI Number'].nunique()}")

# Further filter to recalls for our 14 API products specifically.
# Facilities manufacture many drugs; we want only recalls relevant to our study APIs.
api_names = fei_df[fei_df["API"] != "Grand Total"]["API"].dropna().unique().tolist()
api_pattern = "|".join([re.escape(a) for a in api_names])
api_mask = drug_rec_all["Product Description"].str.contains(api_pattern, case=False, na=False)
drug_rec = drug_rec_all[api_mask].copy()
print(f"API-matched recall rows (our 14 APIs): {len(drug_rec):,}  |  FEIs: {drug_rec['FEI Number'].nunique()}")
print(f"  (Facility-level count retained in drug_rec_all for reference: {len(drug_rec_all)} rows)")

# Parse classification date
drug_rec["Recall_Date"] = pd.to_datetime(drug_rec["Center Classification Date"], errors="coerce")

drug_rec.to_csv(OUT / "recall_filtered.csv", index=False)

# ── Reason for Recall text categorisation ───────────────────────────────────
REASON_CATS = {
    "cgmp_failure":    re.compile(r"cGMP|current good manufacturing|manufacturing\s+(deficiency|issue|error|problem)", re.I),
    "contamination":   re.compile(r"contaminat|impurit|NDMA|nitrosamine|benzene|particulate|microbial|mold|sterility", re.I),
    "wrong_potency":   re.compile(r"subpotent|superpotent|potency|out[\s\-]of[\s\-]spec|OOS|incorrect\s+strength", re.I),
    "mislabeled":      re.compile(r"mislabeled|misbranded|incorrect\s+label|wrong\s+label|label\s+error|undeclared", re.I),
    "foreign_material":re.compile(r"foreign\s+(material|particle|body|object)|glass|metal|rubber", re.I),
    "packaging_defect":re.compile(r"packaging|seal\s+failure|cap\s+defect|container|leaking", re.I),
    "stability":       re.compile(r"stability|degraded|out[\s\-]of[\s\-]date|expir", re.I),
    "color_odor":      re.compile(r"discolorat|colour|odor|off[\s\-]colour|discolour", re.I),
    "data_integrity":  re.compile(r"data\s+integrit|audit\s+trail|ALCOA|falsif", re.I),
}

reasons = drug_rec["Reason for Recall"].fillna("")
for cat, pat in REASON_CATS.items():
    drug_rec[f"reason_{cat}"] = reasons.str.contains(pat).astype(int)

# Overall category distribution
cat_cols = [f"reason_{c}" for c in REASON_CATS]
cat_summary = drug_rec[cat_cols].sum().sort_values(ascending=False)
cat_summary.name = "n_recalls"
cat_summary.index = [c.replace("reason_","") for c in cat_summary.index]
cat_summary.to_csv(OUT / "recall_reason_categories.csv")
print("\nRecall reason categories (all our drug recalls):")
print(cat_summary)

# ── Aggregate to FEI level ───────────────────────────────────────────────────
print("\n" + "="*60)
print("FEI-LEVEL AGGREGATION")
print("="*60)

CLASS_SEV = {"Class I": 3, "Class II": 2, "Class III": 1}

fei_feat = []
for fei in sorted(our_feis):
    sub = drug_rec[drug_rec["FEI Number"] == fei]
    n   = len(sub)

    row = {"fei": int(fei)}
    row["n_recalls_drug"]     = n
    row["has_drug_recall"]    = int(n > 0)

    if n == 0:
        row.update({
            "n_recall_class_I":    0,
            "n_recall_class_II":   0,
            "n_recall_class_III":  0,
            "has_class_I_recall":  0,
            "n_recalls_ongoing":   0,
            "n_recalls_terminated":0,
            "latest_recall_date":  None,
            "earliest_recall_date":None,
            "recall_severity_max": 0,
        })
        for cat in REASON_CATS:
            row[f"has_recall_{cat}"] = 0
    else:
        ec = sub["Event Classification"].str.strip()
        row["n_recall_class_I"]    = int((ec == "Class I").sum())
        row["n_recall_class_II"]   = int((ec == "Class II").sum())
        row["n_recall_class_III"]  = int((ec == "Class III").sum())
        row["has_class_I_recall"]  = int(row["n_recall_class_I"] > 0)
        row["n_recalls_ongoing"]   = int((sub["Status"].str.strip() == "Ongoing").sum())
        row["n_recalls_terminated"]= int((sub["Status"].str.strip() == "Terminated").sum())
        dates = sub["Recall_Date"].dropna()
        row["latest_recall_date"]  = dates.max().strftime("%Y-%m-%d") if len(dates) else None
        row["earliest_recall_date"]= dates.min().strftime("%Y-%m-%d") if len(dates) else None
        sev_map = ec.map(CLASS_SEV).fillna(0)
        row["recall_severity_max"] = int(sev_map.max())
        for cat in REASON_CATS:
            row[f"has_recall_{cat}"] = int(sub[f"reason_{cat}"].max())

    fei_feat.append(row)

feat_df = pd.DataFrame(fei_feat)
feat_df.to_csv(OUT / "recall_fei_features.csv", index=False)

# Summary
have_recalls = feat_df[feat_df["has_drug_recall"] == 1]
print(f"\nFEIs with ≥1 drug recall: {len(have_recalls)} / {len(feat_df)}")
print(f"FEIs with Class I recall:  {feat_df['has_class_I_recall'].sum()}")
print(f"FEIs with contamination recall: {feat_df['has_recall_contamination'].sum()}")
print(f"FEIs with cGMP-failure recall:  {feat_df['has_recall_cgmp_failure'].sum()}")

print("\nTop FEIs by recall count:")
top = feat_df.sort_values("n_recalls_drug", ascending=False).head(15)
print(top[["fei","n_recalls_drug","n_recall_class_I","n_recall_class_II",
           "has_class_I_recall","recall_severity_max",
           "has_recall_contamination","has_recall_cgmp_failure"]].to_string())

print(f"\nAll outputs saved to: {OUT}")

# %%
