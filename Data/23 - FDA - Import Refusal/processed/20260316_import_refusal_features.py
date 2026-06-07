# %%
"""
20260316_import_refusal_features.py

Extracts per-FEI import refusal features from FDA Import Refusal database.

Filters to our 129 reference FEIs, maps refusal charge codes to categories,
and computes FEI-level signal features.

FDA charge codes relevant for pharmaceutical manufacturers:
  3xxx = Drugs (3842=misbranded, 3843=adulterated, 3847=no registration,
                3854=cGMP, 3851=NDA/ANDA missing, 3886=counterfeit)
  Codes < 1000 without 3-prefix may be general FDA import charges
  Charge 75 = "Goods may be adulterated - 21 USC 342" (food/drug general)
  Charge 27 = Labeling/marking deficiency
  Charge 118 = Adulterated/misbranded drug

Outputs:
  - import_refusal_filtered.csv   — API-matched refusal rows (our 14 APIs)
  - import_refusal_fei_features.csv — aggregated per-FEI features
  - import_refusal_charge_summary.csv — charge code frequency
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter

BASE  = Path(__file__).parents[3]
RAW   = BASE / "Data/23 - FDA - Import Refusal/raw/Import Refusal Data.xlsx"
FEI_F = BASE / "Data/08 - Valisure/raw/FEIs_March 2026.xlsx"
OUT   = Path(__file__).parent

# ── Reference FEIs ──────────────────────────────────────────────────────────
fei_df = pd.read_excel(FEI_F, sheet_name="API Only_FEI Mapping")
fei_df = fei_df[fei_df["API"] != "Grand Total"]
our_feis  = set(fei_df["FEI_NUMBER"].dropna().astype(int).astype(str))
api_names = fei_df["API"].dropna().unique().tolist()
print(f"Reference FEIs: {len(our_feis)}  |  APIs: {len(api_names)}")

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading Import Refusal Data.xlsx …")
imp = pd.read_excel(RAW)
imp["FEI Number"] = imp["FEI Number"].astype(str).str.strip()
print(f"Total rows: {len(imp):,}  |  Unique FEIs: {imp['FEI Number'].nunique():,}")

our_imp = imp[imp["FEI Number"].isin(our_feis)].copy()
print(f"Rows for our FEIs: {len(our_imp):,}  |  Unique FEIs: {our_imp['FEI Number'].nunique()}")

# Parse date
our_imp["Refused_Date"] = pd.to_datetime(our_imp["Refused Date"], errors="coerce")

# (import_refusal_filtered.csv saved after API-product filter below)

# ── Charge code mapping ─────────────────────────────────────────────────────
# FDA drug-specific charges (3xxx)
DRUG_CHARGES_MAP = {
    "3842": "misbranded_drug",
    "3843": "adulterated_drug",
    "3847": "unregistered_facility",
    "3851": "unapproved_new_drug",
    "3854": "cgmp_violation",
    "3886": "counterfeit",
    "3846": "no_approval",
    # General drug adulteration/misbranding
    "118":  "drug_adulteration_general",
    "27":   "labeling_deficiency",
    "75":   "general_adulteration",
    # Other general import violation codes observed in our FEI data
    "179":  "other_general_violation",  # ~4% of rows; broad/non-specific import code
    "16":   "other_general_violation",  # ~3% of rows; general import charge
}

DRUG_CHARGE_CODES = set(DRUG_CHARGES_MAP.keys())

def parse_charges(charge_str):
    """Return list of individual charge codes from a comma-separated string."""
    if pd.isna(charge_str):
        return []
    return [c.strip() for c in str(charge_str).split(",") if c.strip()]

def has_drug_charge(charge_str):
    codes = parse_charges(charge_str)
    return any(c in DRUG_CHARGE_CODES for c in codes)

def categorise_charges(charge_str):
    codes = parse_charges(charge_str)
    cats = set()
    for c in codes:
        if c in DRUG_CHARGES_MAP:
            cats.add(DRUG_CHARGES_MAP[c])
        elif c.startswith("38"):
            cats.add("other_drug_charge")
    return cats

our_imp["charge_list"]       = our_imp["Refusal Charges"].apply(parse_charges)
our_imp["has_drug_charge"]   = our_imp["Refusal Charges"].apply(has_drug_charge)
our_imp["charge_categories"] = our_imp["Refusal Charges"].apply(categorise_charges)

# FDA Sample Analysis flag
our_imp["fda_sample_flag"]     = our_imp["FDA Sample Analysis"].str.strip().str.upper() == "YES"
our_imp["private_lab_flag"]    = our_imp["Private Lab Analysis"].str.strip().str.upper() == "YES"
our_imp["lab_analysis_flag"]   = our_imp["fda_sample_flag"] | our_imp["private_lab_flag"]

# Drug product filter: product code starts with 5x, 6x (drugs in FDA system)
our_imp["is_drug_product"] = our_imp["Product Code and Description"].fillna("").str.match(r"[56]\d", na=False)

drug_imp = our_imp[our_imp["is_drug_product"]].copy()
print(f"Drug-product refusal rows (all):         {len(drug_imp):,}  |  FEIs: {drug_imp['FEI Number'].nunique()}")

# API-specific filter: match Product Code and Description to our 14 study APIs
# (mirrors the recall filter on Product Description in 20260316_recall_features.py)
api_pattern = "|".join([re.escape(a) for a in api_names])
api_mask    = drug_imp["Product Code and Description"].str.contains(api_pattern, case=False, na=False)
api_imp     = drug_imp[api_mask].copy()
print(f"API-matched refusal rows (our 14 APIs):  {len(api_imp):,}  |  FEIs: {api_imp['FEI Number'].nunique()}")
print(f"  (Facility-level drug refusals retained in drug_imp for reference: {len(drug_imp)} rows)")

# Save API-matched rows for downstream pipeline (combined dataset / dashboard)
api_imp.to_csv(OUT / "import_refusal_filtered.csv", index=False)

# Charge code frequency for API-matched drug products
all_charges = []
for codes in api_imp["charge_list"]:
    all_charges.extend(codes)
charge_counts = pd.Series(Counter(all_charges)).sort_values(ascending=False)
charge_summary = charge_counts.reset_index()
charge_summary.columns = ["charge_code","count"]
charge_summary["category"] = charge_summary["charge_code"].map(DRUG_CHARGES_MAP).fillna("other")
charge_summary.to_csv(OUT / "import_refusal_charge_summary.csv", index=False)
print("\nTop charge codes (drug products):")
print(charge_summary.head(20).to_string())

# ── Aggregate to FEI level ───────────────────────────────────────────────────
print("\n" + "="*60)
print("FEI-LEVEL AGGREGATION")
print("="*60)

fei_feat = []
for fei in sorted(our_feis):
    sub_all = api_imp[api_imp["FEI Number"] == fei]
    sub_drug = api_imp[api_imp["FEI Number"] == fei]

    n_all  = len(sub_all)
    n_drug = len(sub_drug)

    row = {"fei": int(fei)}
    row["n_import_refusals_total"]  = n_all
    row["n_import_refusals_drug"]   = n_drug
    row["has_import_refusal"]       = int(n_all > 0)
    row["has_drug_import_refusal"]  = int(n_drug > 0)

    if n_all > 0:
        dates = sub_all["Refused_Date"].dropna()
        row["latest_refusal_date"]   = dates.max().strftime("%Y-%m-%d") if len(dates) else None
        row["earliest_refusal_date"] = dates.min().strftime("%Y-%m-%d") if len(dates) else None
        row["n_refusal_years"]       = dates.dt.year.nunique() if len(dates) else 0
        row["has_drug_charge"]       = int(sub_all["has_drug_charge"].any())
        row["n_lab_analysis_refusals"]= int(sub_all["lab_analysis_flag"].sum())
        row["has_lab_analysis_flag"] = int(sub_all["lab_analysis_flag"].any())
        # Specific drug charge categories
        all_cats = set()
        for cats in sub_all["charge_categories"]:
            all_cats |= cats
        for cat in ["cgmp_violation","adulterated_drug","misbranded_drug",
                    "unregistered_facility","unapproved_new_drug","counterfeit",
                    "drug_adulteration_general","labeling_deficiency","general_adulteration"]:
            row[f"has_charge_{cat}"] = int(cat in all_cats)
        # Severity composite: cgmp or adulteration are most serious
        row["import_refusal_severity"] = (
            row.get("has_charge_cgmp_violation", 0) * 3
            + row.get("has_charge_adulterated_drug", 0) * 3
            + row.get("has_charge_counterfeit", 0) * 4
            + row.get("has_charge_drug_adulteration_general", 0) * 2
            + row.get("has_lab_analysis_flag", 0) * 2
            + min(n_all, 20)           # volume signal, capped
        )
    else:
        row.update({
            "latest_refusal_date": None,
            "earliest_refusal_date": None,
            "n_refusal_years": 0,
            "has_drug_charge": 0,
            "n_lab_analysis_refusals": 0,
            "has_lab_analysis_flag": 0,
            "import_refusal_severity": 0,
        })
        for cat in ["cgmp_violation","adulterated_drug","misbranded_drug",
                    "unregistered_facility","unapproved_new_drug","counterfeit",
                    "drug_adulteration_general","labeling_deficiency","general_adulteration"]:
            row[f"has_charge_{cat}"] = 0

    fei_feat.append(row)

feat_df = pd.DataFrame(fei_feat)
feat_df.to_csv(OUT / "import_refusal_fei_features.csv", index=False)

have = feat_df[feat_df["has_import_refusal"] == 1]
have_drug = feat_df[feat_df["has_drug_import_refusal"] == 1]
print(f"\nFEIs with any import refusal:        {len(have)} / {len(feat_df)}")
print(f"FEIs with drug-product refusal:      {len(have_drug)} / {len(feat_df)}")
print(f"FEIs with cGMP charge:               {feat_df['has_charge_cgmp_violation'].sum()}")
print(f"FEIs with adulteration charge:       {feat_df['has_charge_adulterated_drug'].sum()}")
print(f"FEIs with lab-analysis refusal:      {feat_df['has_lab_analysis_flag'].sum()}")

print("\nTop FEIs by import refusal count:")
top = feat_df.sort_values("n_import_refusals_drug", ascending=False).head(15)
print(top[["fei","n_import_refusals_drug","has_drug_charge","has_charge_cgmp_violation",
           "has_charge_adulterated_drug","n_lab_analysis_refusals",
           "import_refusal_severity"]].to_string())

print(f"\nAll outputs saved to: {OUT}")

# %%
