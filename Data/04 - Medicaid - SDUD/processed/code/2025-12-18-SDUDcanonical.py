# %%
import os
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd

# --- helpers ---------------------------------------------------------------

def _norm(s: str) -> str:
    """Normalize a column name for matching (lower, strip, collapse punctuation)."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Canonical column names we want
CANON = [
    "utilization_type",
    "state",
    "ndc",
    "labeler_code",
    "product_code",
    "package_size",
    "year",
    "quarter",
    "suppression_used",
    "product_name",
    "units_reimbursed",
    "num_prescriptions",
    "total_amount_reimbursed",
    "medicaid_amount_reimbursed",
    "non_medicaid_amount_reimbursed",
    "record_id",
]

# Map *many* possible variants → canonical
VARIANTS: Dict[str, List[str]] = {
    "record_id": ["record id", "record_id", "rec id"],
    "utilization_type": ["utilization type", "utilization_type", "utilizationtype"],
    "state": ["state", "state code", "state_cd", "st"],
    "ndc": ["ndc", "national drug code"],
    "labeler_code": ["labeler code", "labeler_code", "labeler"],
    "product_code": ["product code", "product_code", "prod code", "prod_code"],
    "package_size": ["package size", "package_size", "pkg size", "pkg_size"],
    "year": ["year", "yr"],
    "quarter": ["quarter", "qtr", "qrt", "q"],
    "suppression_used": ["suppression used", "supression used", "supression_used",
                         "suppression_used", "suppresion used"],
    "product_name": ["product name", "product fda list name", "fda list name",
                     "product_fda_list_name", "product list name"],
    "units_reimbursed": ["units reimbursed", "units_reimbursed", "units"],
    "num_prescriptions": ["number of prescriptions", "no. of prescriptions",
                          "no of prescriptions", "num prescriptions",
                          "num_prescriptions", "rx count"],
    "total_amount_reimbursed": ["total amount reimbursed", "total_amount_reimbursed",
                                "total reimbursed", "amount reimbursed"],
    "medicaid_amount_reimbursed": ["medicaid amount reimbursed",
                                   "medicaid_amount_reimbursed",
                                   "medicaid reimbursed"],
    "non_medicaid_amount_reimbursed": ["non medicaid amount reimbursed",
                                       "non-medicaid amount reimbursed",
                                       "non_medicaid_amount_reimbursed",
                                       "non medicaid reimbursed"],
}

# Build a fast reverse map of normalized variant → canonical
REV: Dict[str, str] = {}
for canon, alts in VARIANTS.items():
    for alt in alts:
        REV[_norm(alt)] = canon

def _coalesce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each canonical column, coalesce data from all matching variant columns
    (first non-null / non-empty wins), then drop the variant columns.
    """
    # group existing columns by canonical target
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        key = REV.get(_norm(col))
        if key:
            groups.setdefault(key, []).append(col)

    for canon, cols in groups.items():
        # start with the first column, fill from the rest
        ser = df[cols[0]]
        for c in cols[1:]:
            ser = ser.where(ser.notna() & (ser.astype(str).str.len() > 0), df[c])
        df[canon] = ser
        # drop the original variant columns (except the newly created canon if same name)
        to_drop = [c for c in cols if c != canon]
        df.drop(columns=to_drop, inplace=True, errors="ignore")

    return df

# --- main loader -----------------------------------------------------------

def build_sdud_master(sdud_dir: str, pattern: str = "SDUD_*.csv") -> pd.DataFrame:
    """
    Load and append all SDUD CSVs in a folder into a single, column-normalized DataFrame.
    - Reads as strings to preserve leading zeros.
    - Handles many header variants and coalesces into canonical names.
    - Adds `source_file`.
    """
    sdud_path = Path(sdud_dir)
    files = sorted(sdud_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {sdud_dir} matching {pattern}")

    frames = []
    for f in files:
        print(f"Loading {f.name} ...")
        df = pd.read_csv(f, dtype=str, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        # Coalesce variants → canonical
        df = _coalesce_columns(df)

        # Add filename and infer year from filename if needed
        df["source_file"] = f.name
        if "year" not in df.columns or df["year"].isna().all():
            m = re.search(r"(\d{4})", f.stem)
            if m:
                df["year"] = df.get("year", m.group(1))
                df["year"] = df["year"].fillna(m.group(1))

        frames.append(df)

    sdud_master = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure all canonical columns exist, even if missing in some years
    for c in CANON:
        if c not in sdud_master.columns:
            sdud_master[c] = pd.NA

    # Reorder (keep canon first, then everything else, then source_file at end)
    other_cols = [c for c in sdud_master.columns if c not in CANON + ["source_file"]]
    sdud_master = sdud_master[CANON + other_cols + ["source_file"]]

    print(
        f"Combined {len(files)} files → {len(sdud_master):,} rows, "
        f"{sdud_master.shape[1]} columns."
    )
    return sdud_master


# ---------------- run ----------------
# Chnage directory accordingly
DATA_DIR = "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data/04 - Medicaid - SDUD/raw"
OUT_DIR = "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data/04 - Medicaid - SDUD/processed"
sdud_canonical = build_sdud_master(DATA_DIR)
output_path = os.path.join(OUT_DIR, "2025-12-18-SDUDcanonical.parquet")
sdud_canonical.to_parquet(output_path, index=False)
print(f"Parquet file written to: {output_path}")



# %%
