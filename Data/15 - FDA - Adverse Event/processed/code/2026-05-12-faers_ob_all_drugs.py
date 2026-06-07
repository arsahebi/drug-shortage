# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ---------------- Configuration ----------------
DATA_DIR = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data")
FAERS_DIR = DATA_DIR / "15 - FDA - Adverse Event/raw"
OUT_DIR = DATA_DIR / "15 - FDA - Adverse Event/processed"
ORANGEBOOK_FILE = DATA_DIR / "01 - Orange Book/output_data/products.csv"

# Severity Priority Map [cite: 1, 9]
SEVERITY_MAP = {
    "DE": "Death", "LT": "Life-threatening", "HO": "Hospitalization",
    "DS": "Disability", "CA": "Congenital anomaly", "RI": "Required intervention",
    "OT": "Other serious"
}

# ---------------- Helpers ----------------
def load_faers_file(path: Path):
    """Loads a FAERS $ delimited file and normalizes headers."""
    df = pd.read_csv(path, sep="$", dtype=str, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip().str.lower()
    return df

def get_best_severity(codes_str):
    """Pick highest priority severity from comma-separated string."""
    if pd.isna(codes_str): return "No outcome reported"
    codes = set(codes_str.split(","))
    for key, val in SEVERITY_MAP.items():
        if key in codes: return val
    return "Other / unknown"

# ---------------- Load Orange Book ----------------
ob = pd.read_csv(ORANGEBOOK_FILE)
_ob_raw = pd.to_numeric(ob['Appl_No'], errors='coerce')
ob['Appl_No'] = pd.array(
    [int(v) if pd.notna(v) and v % 1 == 0 else pd.NA for v in _ob_raw],
    dtype="Int64"
)
# EXTRA SAFETY CHECK: Map Appl_No to its most frequent Appl_Type. 
# While current data shows 0 conflicts (1-to-1 mapping), this prevents errors 
# if future datasets include duplicate IDs with different strengths or entries.
ob_map = ob.groupby("Appl_No")["Appl_Type"].agg(lambda x: x.mode()[0] if not x.empty else pd.NA)

# ---------------- Main Processing ----------------
# 1. Flexible Discovery
pat = re.compile(r"(\d{4})[qQ]([1-4])")
quarters_to_process = []

for folder in FAERS_DIR.iterdir():
    if folder.is_dir():
        m = pat.search(folder.name)
        if m:
            year = int(m.group(1))
            q = int(m.group(2))
            quarters_to_process.append((year, q, folder))

# Sort chronologically (important for trend analysis!)
quarters_to_process.sort()

# %%

all_quarters_data = []

for year, q, folder in quarters_to_process:
    yy = str(year)[-2:]
    print(f"--- Processing {year} Q{q} ---")
    
    try:
        # File Discovery: rglob finds all .txt files, then case-insensitive string match.
        # pathlib.rglob uses fnmatchcase (case-sensitive) even on macOS, so glob patterns
        # like *[Dd]emo* fail against all-caps filenames like DEMO15Q1.txt.
        def load_flexible(table_name):
            all_txt = list(folder.rglob("*.txt"))
            t = table_name.lower()          # e.g. "demo"
            yq = f"{yy}q{q}"               # e.g. "15q1" — matches "15Q1" after .lower()

            matches = [
                f for f in all_txt
                if t in f.name.lower() and yq in f.name.lower()
            ]

            if not matches:
                raise FileNotFoundError(f"Missing {table_name} for {year} Q{q} in {folder}")

            # Prefer shorter names (avoids temp/backup files)
            file_path = sorted(matches, key=lambda x: len(x.name))[0]

            print(f"  Successfully matched: {file_path.name}")
            df = pd.read_csv(file_path, sep="$", dtype=str, low_memory=False, encoding="latin-1")
            df.columns = [c.strip().lower() for c in df.columns]
            return df

        # Load core tables
        demo = load_flexible("demo")
        drug = load_flexible("drug")
        outc = load_flexible("outc")
        print(f"  Total drug records: {len(drug)}")
                
        # 1. Process OUTC: Aggregate outcomes
        # Normalize column name to `outc_codes` and drop NA values before joining
        outc_agg = (
            outc.groupby("primaryid")["outc_cod"]
            .agg(lambda x: ",".join(pd.Series(x).dropna().unique()))
            .reset_index()
            .rename(columns={"outc_cod": "outc_codes"})
        )
        
        # 2. Process DRUG: Filter to Primary Suspect (PS)
        # Use str.upper() for robustness
        drug_ps = drug[drug["role_cod"].str.upper() == "PS"].copy()
        _raw = pd.to_numeric(drug_ps["nda_num"], errors='coerce')
        _I64 = 2**63 - 1
        drug_ps["appl_no"] = pd.array(
            [int(v) if pd.notna(v) and v % 1 == 0 and abs(v) <= _I64 else pd.NA for v in _raw],
            dtype="Int64"
        )
        print(f"  PS drug records: {len(drug_ps)}")
        
        # 3. Merge
        demo_subset = demo
        
        merged = drug_ps.merge(demo_subset, on="primaryid", how="left", suffixes=("", "_demo"))
        print(f"  After demo merge: {len(merged)}")
        merged = merged.merge(outc_agg, on="primaryid", how="left")
        print(f"  After outc merge: {len(merged)}")
        
        # Cleanup caseid duplicates
        if "caseid_demo" in merged.columns:
            merged["caseid"] = merged["caseid"].fillna(merged["caseid_demo"])
            merged = merged.drop(columns=["caseid_demo"])

        # 4. Link Orange Book & Metadata
        # EXTRA SAFETY CHECK: ob_map ensures 1-to-1 mapping even if OB has duplicates
        merged["ob_appl_type"] = merged["appl_no"].map(ob_map)
        merged["is_anda"] = merged["ob_appl_type"].str.strip().str.upper() == "A"
        
        merged["file_year"] = year
        merged["file_quarter"] = q
        
        all_quarters_data.append(merged)
        
    except (StopIteration, FileNotFoundError):
        print(f"  [SKIPPED] Missing required .txt files in {folder}")
    except Exception as e:
        print(f"  [ERROR] {year}Q{q}: {e}")

# %%
# ---------------- Final Export ----------------
if all_quarters_data:
    drugs_all = pd.concat(all_quarters_data, ignore_index=True)
    print(f"Total records across all quarters: {len(drugs_all)}")
    # Filter for ANDA linked records
    drugs_anda = drugs_all[drugs_all["is_anda"] == True].copy()
    print(f"ANDA-linked records: {len(drugs_anda)}")
    
    # ---------------- Stats (rows + unique appl_no + %) ----------------
    n_total_rows = len(drugs_all)

    def uniq_appl(df, mask):
        return df.loc[mask, "appl_no"].dropna().nunique()

    # Stage A: appl_no is non-missing (numeric)
    mask_has_appl = drugs_all["appl_no"].notna()
    n_has_appl_rows = mask_has_appl.sum()
    n_has_appl_uniq = uniq_appl(drugs_all, mask_has_appl)

    # Stage B: matched to OB (has Appl_Type from OB)
    mask_matched_ob = drugs_all["ob_appl_type"].notna()
    n_matched_rows = mask_matched_ob.sum()
    n_matched_uniq = uniq_appl(drugs_all, mask_matched_ob)

    # Stage C: among matched, generic ANDA (Appl_Type == 'A')
    mask_generic_A = mask_matched_ob & (drugs_all["ob_appl_type"] == "A")
    n_generic_rows = mask_generic_A.sum()
    n_generic_uniq = uniq_appl(drugs_all, mask_generic_A)

    # Stage D: among matched, non-generic (not 'A' — likely 'N' or others)
    mask_nongeneric = mask_matched_ob & (drugs_all["ob_appl_type"] != "A")
    n_nongeneric_rows = mask_nongeneric.sum()
    n_nongeneric_uniq = uniq_appl(drugs_all, mask_nongeneric)

    # Stage E: numeric appl_no but NOT matched in OB
    mask_unmatched_numeric = mask_has_appl & (~mask_matched_ob)
    n_unmatched_rows = mask_unmatched_numeric.sum()
    n_unmatched_uniq = uniq_appl(drugs_all, mask_unmatched_numeric)

    # Stage F: appl_no is missing
    mask_appl_missing = ~mask_has_appl
    n_appl_missing_rows = mask_appl_missing.sum()

    print("\n=== FAERS ALL DRUGS (DRUG rows) + ORANGE BOOK LINKING STATS ===")
    print(f"Total FAERS DRUG rows: {n_total_rows:,}")

    print("\n[1] appl_no present (nda_num numeric):")
    print(f"  rows: {n_has_appl_rows:,} ({n_has_appl_rows/n_total_rows:.1%})")
    print(f"  unique appl_no: {n_has_appl_uniq:,}")

    print("\n[2] matched to Orange Book (has ob_appl_type):")
    print(f"  rows: {n_matched_rows:,} ({n_matched_rows/n_total_rows:.1%})")
    print(f"  unique appl_no: {n_matched_uniq:,} ({(n_matched_uniq/n_has_appl_uniq if n_has_appl_uniq else 0):.1%} of appl_no present)")

    print("\n[3] among matched: generic ANDA (A):")
    print(f"  rows: {n_generic_rows:,} ({n_generic_rows/n_total_rows:.1%})")
    print(f"  unique appl_no: {n_generic_uniq:,}")

    print("\n[4] among matched: non-generic (not A):")
    print(f"  rows: {n_nongeneric_rows:,} ({n_nongeneric_rows/n_total_rows:.1%})")
    print(f"  unique appl_no: {n_nongeneric_uniq:,}")

    print("\n[5] appl_no present but NOT matched in OB:")
    print(f"  rows: {n_unmatched_rows:,} ({n_unmatched_rows/n_total_rows:.1%})")
    print(f"  unique appl_no: {n_unmatched_uniq:,}")

    print("\n[6] appl_no missing (nda_num NA/non-numeric):")
    print(f"  rows: {n_appl_missing_rows:,} ({n_appl_missing_rows/n_total_rows:.1%})")

    # Optional: show which numeric appl_no did not match OB (for debugging)
    nonmatching_list = sorted(int(x) for x in drugs_all.loc[mask_unmatched_numeric, "appl_no"].dropna().unique())
    print("\nNon-matching appl_no values (numeric) [first 60]:")
    print(nonmatching_list[:60])


    # ---------------- 5) Severity & time variables ----------------

    # Severity classification based on OUTC codes
    drugs_anda["severity"] = drugs_anda["outc_codes"].apply(get_best_severity)

    # Parse FDA receipt date (fda_dt) as datetime
    drugs_anda["fda_date"] = pd.to_datetime(
        drugs_anda["fda_dt"], format="%Y%m%d", errors="coerce"
    )

    # Year from fda_date; fall back to file_year if missing
    drugs_anda["year"] = drugs_anda["fda_date"].dt.year
    drugs_anda["year"] = drugs_anda["year"].fillna(drugs_anda["file_year"]).astype(int)

    # Period label like "2025Q1"
    drugs_anda["period"] = (
        drugs_anda["file_year"].astype(str)
        + "Q"
        + drugs_anda["file_quarter"].astype(str)
    )

    # For counting, deduplicate by (primaryid, appl_no)
    case_level = (
        drugs_anda[["primaryid", "appl_no", "severity", "year", "period"]]
        .drop_duplicates()
    )

    print("\nCase-level rows (unique primaryid + appl_no):", len(case_level))
    print("Unique primaryid (cases):", case_level["primaryid"].nunique())
    print("Unique appl_no (ANDAs):", case_level["appl_no"].nunique())
    print("\nSeverity counts (case_level):")
    print(case_level["severity"].value_counts(dropna=False))

    SERIOUS_LEVELS = {
        "Death",
        "Life-threatening",
        "Hospitalization",
        "Disability",
        "Congenital anomaly",
        "Required intervention",
        "Other serious",
    }

    drugs_anda["serious_flag"] = drugs_anda["severity"].isin(SERIOUS_LEVELS)

    drugs_anda = drugs_anda.reset_index(drop=True)

    print(f"Final ANDA dataset shape: {drugs_anda.shape}")
# %%
# 6) save data
# derive first/last quarter from discovered folders (quarters_to_process)
if quarters_to_process:
    start_y, start_q, _ = quarters_to_process[0]
    end_y, end_q, _ = quarters_to_process[-1]
    start_label = f"{start_y}q{start_q}"
    end_label = f"{end_y}q{end_q}"
    fname = f"faers_all_drugs_anda_linked_{start_label}_{end_label}.csv"
else:
    fname = "faers_all_drugs_anda_linked_unknown_range.csv"

out_path = OUT_DIR / fname
drugs_anda.to_csv(out_path, index=False)
print("Saved:", out_path)

# %%
