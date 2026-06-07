# %%
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# =============================================================================
# 1. PATHS & DICTS
# =============================================================================
BASE_DIR = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
INPUT_FILE = BASE_DIR / "Data/08 - Valisure/raw/Testing Data_DoD First 13 Drug Scores with ANDAs & NDCs.xlsx"
OUT_DIR = BASE_DIR / "Data/08 - Valisure/processed"
FEI_FILE = BASE_DIR / "Data/08 - Valisure/raw/FEIs_March 2026.xlsx"

RENAME_QUALITY = {
    "LCMS DMF (ng/day)":  "DMF (ng/DAY) Valisure",
    "LCMS NDMA (ng/day)": "NDMA (ng/DAY) Valisure",
    "LCMS NMBA (ng/day)": "NMBA (ng/DAY) Valisure",
    "LCMS NDEA (ng/day)": "NDEA (ng/DAY) Valisure",
    "LCMS NMPA (ng/day)": "NMPA (ng/DAY) Valisure",
    "As (ug/day)": "As (ug/day) Valisure",
    "Pb (ug/day)": "Pb (ug/day) Valisure",
    "Tl (ug/day)": "Tl (ug/day) Valisure",
    "Cd (ug/day)": "Cd (ug/day) Valisure",
    "Hg (ug/day)": "Hg (ug/day) Valisure",
    "Li (ug/day)": "Li (ug/day) Valisure",
    "Cr (ug/day)": "Cr (ug/day) Valisure",
    "Ni (ug/day)": "Ni (ug/day) Valisure",
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def format_ndc(val):
    """Converts hyphenated or numeric NDC to strict 11-digit string (5-4-2)."""
    s = str(val).strip()
    if "-" in s:
        p = s.split("-")
        if len(p) == 3:
            return p[0].zfill(5) + p[1].zfill(4) + p[2].zfill(2)
    digits = "".join(filter(str.isdigit, s))
    return digits.zfill(11) if digits else ""

# =============================================================================
# 3. MAIN PROCESSING
# =============================================================================
all_data = []
xf = pd.ExcelFile(INPUT_FILE)

for sheet_name in xf.sheet_names:
    # Read sheet (skipping the section header row)
    df = pd.read_excel(xf, sheet_name=sheet_name, header=1)
    
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Skip sheets that aren't data (must have NDC or API column)
    if "NDC" not in df.columns and "API" not in df.columns:
        continue

    # Fill missing API names (constant for each sheet)
    if "API" in df.columns:
        df["API"] = df["API"].ffill()

    # Clean NDC: format to 11 digits
    if "NDC" in df.columns:
        df["NDC11"] = df["NDC"].apply(format_ndc)

    # Apply Renaming
    df = df.rename(columns=RENAME_QUALITY)
    
    # Convert metric columns to numeric
    for new_col in RENAME_QUALITY.values():
        if new_col in df.columns:
            df[new_col] = pd.to_numeric(df[new_col], errors="coerce")

    # Drop placeholder/empty/Unnamed columns
    bad_cols = ["Elemental", "Carcinogens", "Dissolution"]
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c or c in bad_cols], errors="ignore")

    # Drop rows without an NDC (removes footer/summary rows)
    df = df.dropna(subset=["NDC"])
    
    all_data.append(df)
    print(f"Processed: {sheet_name}")

# Combine all sheets
final_df = pd.concat(all_data, ignore_index=True, sort=False)
final_df["Year"] = 2024

# =============================================================================
# 4.1 LOAD FEI AND MERGE
# =============================================================================
if FEI_FILE.exists():
    try:
        fei_xf = pd.ExcelFile(FEI_FILE)
        fei_df = pd.read_excel(fei_xf, sheet_name=0, header=0, dtype=str)
        fei_df.columns = [str(c).strip() for c in fei_df.columns]

        # find NDC and FEI-like columns
        ndc_cols = [c for c in fei_df.columns if "NDC" in c.upper()]
        fei_cols = [c for c in fei_df.columns if "FEI" in c.upper()]

        if ndc_cols:
            fei_df["NDC11"] = fei_df[ndc_cols[0]].apply(format_ndc)
            if fei_cols:
                fei_merge = fei_df[["NDC11", fei_cols[0]]].rename(columns={fei_cols[0]: "FEI"})
                final_df = final_df.merge(fei_merge, on="NDC11", how="left")
            else:
                print("FEI column not found in FEIs_March 2026.xlsx; skipping FEI merge")
        else:
            print("NDC column not found in FEIs_March 2026.xlsx; skipping FEI merge")
    except Exception as e:
        print(f"Error reading FEI file: {e}; skipping FEI merge")
else:
    print(f"FEI file not found: {FEI_FILE}; skipping FEI merge")

# =============================================================================
# 4. COLUMN REORDERING
# =============================================================================
# Target: [..., NDC, Year, Difference Factor, ...]
cols = list(final_df.columns)

# Safely remove to re-insert
if "NDC11" in cols: cols.remove("NDC11")
if "Year" in cols: cols.remove("Year")

if "Difference Factor" in cols:
    idx = cols.index("Difference Factor")
    # Insert NDC before Difference Factor, and Year after NDC
    cols.insert(idx, "Year")
    cols.insert(idx, "NDC11")
else:
    # Fallback if the column is missing: put them at the beginning
    cols = ["NDC11", "Year"] + cols

# If FEI was merged in, place it after the Labeler column (if present)
if "FEI" in final_df.columns:
    # remove any existing occurrence to re-insert
    cols = [c for c in cols if c != "FEI"]
    # find a labeler-like column (case-insensitive)
    labeler_idx = None
    for i, c in enumerate(cols):
        if str(c).lower() == "labeler":
            labeler_idx = i
            break

    if labeler_idx is not None:
        cols.insert(labeler_idx + 1, "FEI")
    else:
        # fallback: insert after NDC11 if present, else append at end
        if "NDC11" in cols:
            ndc_pos = cols.index("NDC11")
            cols.insert(ndc_pos + 1, "FEI")
        else:
            cols.append("FEI")

final_df = final_df[cols]

# =============================================================================
# 5. SAVE
# =============================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_DIR / "valisure_all_drugs_combined.csv", index=False)
print(f"\nDone! Files saved in: {OUT_DIR}")
# %%
