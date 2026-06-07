# %%
# -*- coding: utf-8 -*-

"""
Build monthly NDC11 panel by merging:
  - SDUD monthly (utilization + reimbursement), filtered to drugs in Drug List
  - NADAC monthly (pricing)

No IQVIA. Drug scope is the full list from Data/Drugs List.xlsx (42 drugs).

Data sources (relative to Data/ root):
  - Drug list:      Drugs List.xlsx
  - SDUD canonical: 04 - Medicaid - SDUD/processed/2025-12-18-SDUDcanonical.parquet
  - SDUD monthly:   04 - Medicaid - SDUD/processed/2025-12-18-SDUDmonthly.parquet
  - NADAC raw:      11 - Medicaid - NADAC/raw/

All outputs are written under:
  04_11 - Build - Monthly Panel (SDUD+NADAC)/processed/
"""

import re
import warnings
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------- Paths -----------------------
DATA_ROOT = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/Data"
)

RUN_TAG = "2026-03-26"

OUTPUT_DIR = DATA_ROOT / "04_11 - Build - Monthly Panel (SDUD+NADAC)" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUGS_LIST_PATH = DATA_ROOT / "Drugs List.xlsx"

SDUD_CANONICAL_PATH = DATA_ROOT / "04 - Medicaid - SDUD/processed/2025-12-18-SDUDcanonical.parquet"

SDUD_MONTHLY_CANDIDATES = [
    "04 - Medicaid - SDUD/processed/2026-02-24-SDUDmonthly.parquet",
    "04 - Medicaid - SDUD/processed/2025-12-18-SDUDmonthly.parquet",
    "04 - Medicaid - SDUD/processed/sdud_monthly_national.parquet",
]

NADAC_DIR_CANDIDATES = [
    "11 - Medicaid - NADAC/raw",
    "11 - Medicaid - NADAC",
]

# US 50 states + DC (exclude territories: PR, VI, GU, AS, MP, XX)
US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
}

QUARTER_MONTHS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}


# ----------------------------- Helpers ------------------------------------
def _pick_existing(root: Path, rel_candidates: list) -> Path:
    for rel in rel_candidates:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these input paths exist:\n  - " + "\n  - ".join(str(root / r) for r in rel_candidates)
    )


def _pick_existing_dir(root: Path, rel_candidates: list) -> Path:
    for rel in rel_candidates:
        p = root / rel
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "None of these input directories exist:\n  - " + "\n  - ".join(str(root / r) for r in rel_candidates)
    )


def _month_end(x) -> pd.Series:
    return pd.to_datetime(x).dt.to_period("M").dt.to_timestamp("M")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _mode_nonnull(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0] if not s.mode().empty else s.iloc[0]


def _nadac_to_ndc11(s: pd.Series) -> pd.Series:
    """
    Robust NADAC NDC normalizer.
    Strips non-digits, pads to 11 with leading zeros, excludes invalid.
    """
    digits = s.astype("string").fillna("").str.replace(r"\D+", "", regex=True)
    out = digits.where(digits.str.len().between(8, 11), pd.NA)
    out = out.str.zfill(11)
    out = out.where(out.str.fullmatch(r"\d{11}"), pd.NA)
    out = out.where(out != "00000000000", pd.NA)
    return out


# ----------------------------- Drug List ------------------------------------
def load_drug_names(path: Path) -> list:
    """
    Load drug names from Drugs List.xlsx.
    The file has the first drug as the column header and the rest as rows.
    Returns a sorted list of unique drug names (title-cased).
    """
    df = pd.read_excel(path, dtype=str)
    # Column 1 (index 0) holds drug names; column header is also a drug name
    col = df.columns[0]
    names = [col] + df[col].dropna().tolist()
    # Title-case and deduplicate (preserve order)
    seen, result = set(), []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _build_drug_keys(drug_names: list) -> list:
    """
    Build (drug_name, search_key) pairs for matching against SDUD product_name.
    search_key is uppercase; 'INJECTION' suffix stripped since SDUD uses abbreviations.
    Sorted longest-key-first for greedy matching.
    """
    pairs = []
    for drug in drug_names:
        key = drug.upper()
        key = re.sub(r"\bINJECTION\b", "", key).strip()
        key = re.sub(r"\s+", " ", key).strip()
        pairs.append((drug, key))
    return sorted(pairs, key=lambda x: len(x[1]), reverse=True)


def assign_drug_name(product_name: str, drug_keys: list, min_match_len: int = 6) -> Optional[str]:
    """
    Match a SDUD product_name to a drug in the Drug List.

    SDUD product_name is a fixed-width 10-char field — long drug names are
    truncated (e.g. 'ATORVASTATIN' → 'ATORVASTAT'). We therefore compare only
    the shorter of (len(product_name), len(drug_key)) characters, which lets us
    match truncated names while still avoiding false positives like 'Epinephrine'
    matching a product_name that starts with 'NOREPINEPH'.

    min_match_len guards against spuriously short matches (< 6 chars).
    """
    pn = str(product_name).strip().upper()
    n = len(pn)
    if n < min_match_len:
        return None
    for drug, key in drug_keys:
        compare_len = min(n, len(key))
        if compare_len < min_match_len:
            continue
        if pn[:compare_len] == key[:compare_len]:
            return drug
    return None


# ----------------------------- SDUD ------------------------------------------
def build_ndc_drug_mapping(canonical_path: Path, drug_keys: list) -> pd.DataFrame:
    """
    Load only ndc + product_name from SDUDcanonical.parquet, assign drug labels,
    and return a DataFrame with columns [ndc11, drug_name].
    Uses modal drug_name per ndc11 in case of ambiguity.
    """
    print(f"Loading NDC↔drug mapping from: {canonical_path}")
    canon = pd.read_parquet(canonical_path, columns=["ndc", "product_name"])

    canon["ndc11"] = (
        canon["ndc"].astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(11)
    )
    canon = canon[
        canon["ndc11"].str.fullmatch(r"\d{11}") & (canon["ndc11"] != "00000000000")
    ]

    print("  Assigning drug names …")
    canon["drug_name"] = canon["product_name"].apply(
        lambda x: assign_drug_name(x, drug_keys)
    )
    canon = canon[canon["drug_name"].notna()]

    mapping = (
        canon.groupby("ndc11", as_index=False)["drug_name"]
        .agg(_mode_nonnull)
    )
    n_ndc = len(mapping)
    n_drugs = mapping["drug_name"].nunique()
    print(f"  Matched {n_ndc:,} unique NDC11s across {n_drugs} drugs.")
    return mapping


def load_sdud_monthly(parquet_path: Path, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Load SDUD monthly parquet, filter to drug NDCs, add drug_name, return
    DataFrame keyed by (ndc11, date, drug_name).
    """
    print(f"Loading SDUD monthly: {parquet_path}")
    m = pd.read_parquet(parquet_path).copy()

    # Normalize date column
    if "month_start" in m.columns:
        m["date"] = _month_end(m["month_start"])
    elif "date" in m.columns:
        m["date"] = _month_end(m["date"])
    else:
        raise ValueError("SDUD monthly parquet lacks month_start or date column.")

    # Normalize ndc11
    if "ndc11" not in m.columns:
        if "ndc" in m.columns:
            m["ndc11"] = m["ndc"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11)
        else:
            raise ValueError("SDUD monthly parquet lacks ndc11 or ndc column.")

    # Filter to drug NDCs
    drug_ndcs = set(mapping["ndc11"])
    m = m[m["ndc11"].isin(drug_ndcs)].copy()
    print(f"  After drug filter: {len(m):,} rows, {m['ndc11'].nunique():,} NDC11s")

    # Join drug_name
    m = m.merge(mapping, on="ndc11", how="left")

    # Aggregate in case multiple rows share (ndc11, date) after join
    agg_dict = {}
    for col, rename in [
        ("units_reimbursed",          "sdud_units_reimbursed"),
        ("num_prescriptions",         "sdud_num_prescriptions"),
        ("total_amount_reimbursed",   "total_amount_reimbursed"),
        ("medicaid_amount_reimbursed","medicaid_amount_reimbursed"),
    ]:
        if col in m.columns:
            agg_dict[rename] = (col, "sum")

    g = m.groupby(["ndc11", "date", "drug_name"], as_index=False).agg(**agg_dict)

    # Fill missing reimbursement columns with NaN
    for col in ("total_amount_reimbursed", "medicaid_amount_reimbursed",
                "sdud_units_reimbursed", "sdud_num_prescriptions"):
        if col not in g.columns:
            g[col] = np.nan

    return g


# ----------------------------- NADAC -----------------------------------------
def load_nadac_monthly(
    nadac_dir: Path,
    pricing_units: Optional[Set[str]] = None,
):
    """
    Build monthly NADAC table keyed by (ndc11, month_end_date).

    Returns:
        nadac_m      – monthly aggregation (mean of weekly prices per month)
        nadac_weekly – weekly-level table for annual averaging downstream
    """
    files = sorted(nadac_dir.glob("NADAC*.csv"))
    if not files:
        raise FileNotFoundError(f"No NADAC*.csv under {nadac_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, low_memory=False)

        # Normalize column names
        df.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in df.columns]

        col_ndc = next((c for c in ("ndc", "ndc_code", "ndc11") if c in df.columns), None)
        col_date = "as_of_date" if "as_of_date" in df.columns else None
        col_price = next((c for c in ("nadac_per_unit", "nadac") if c in df.columns), None)
        col_unit = "pricing_unit" if "pricing_unit" in df.columns else None

        if not (col_ndc and col_date and col_price):
            continue

        tmp = pd.DataFrame({
            "ndc11":        _nadac_to_ndc11(df[col_ndc]),
            "week_date":    pd.to_datetime(df[col_date], errors="coerce"),
            "nadac_price":  pd.to_numeric(df[col_price], errors="coerce"),
            "pricing_unit": (df[col_unit].astype(str) if col_unit else np.nan),
        }).dropna(subset=["ndc11", "week_date", "nadac_price"])

        if pricing_units is not None and col_unit:
            tmp = tmp[tmp["pricing_unit"].isin(pricing_units)]

        # Deduplicate to avoid double-counting if files overlap
        tmp = tmp.drop_duplicates(subset=["ndc11", "week_date", "nadac_price", "pricing_unit"])

        # Month-end alignment for panel merge
        tmp["date"] = tmp["week_date"].dt.to_period("M").dt.to_timestamp("M")

        frames.append(tmp[["ndc11", "date", "week_date", "nadac_price", "pricing_unit"]])

    if not frames:
        raise ValueError(f"Could not parse any NADAC files under: {nadac_dir}")

    all_ = pd.concat(frames, ignore_index=True)

    # Weekly table (for downstream annual averaging)
    nadac_weekly = all_[["ndc11", "week_date", "date", "nadac_price", "pricing_unit"]].copy()

    # Monthly aggregation
    nadac_m = (
        all_.groupby(["ndc11", "date"], as_index=False)
        .agg(
            nadac_price      = ("nadac_price", "mean"),
            n_obs            = ("nadac_price", "size"),
            pricing_unit_mode= ("pricing_unit", _mode_nonnull),
        )
    )
    return nadac_m, nadac_weekly


# ----------------------------- Runner -----------------------------------------
def build_panel(nadac_units: Optional[str] = None) -> pd.DataFrame:

    # ── 1. Drug list ─────────────────────────────────────────────────────────
    drug_names = load_drug_names(DRUGS_LIST_PATH)
    print(f"Drug list loaded: {len(drug_names)} drugs")
    drug_keys = _build_drug_keys(drug_names)

    # ── 2. NDC → drug_name mapping (from canonical parquet) ──────────────────
    mapping = build_ndc_drug_mapping(SDUD_CANONICAL_PATH, drug_keys)
    mapping.to_csv(OUTPUT_DIR / f"{RUN_TAG}-ndc_drug_mapping.csv", index=False)
    print(f"Saved NDC→drug mapping: {OUTPUT_DIR}/{RUN_TAG}-ndc_drug_mapping.csv")

    # ── 3. SDUD monthly ───────────────────────────────────────────────────────
    sdud_path = _pick_existing(DATA_ROOT, SDUD_MONTHLY_CANDIDATES)
    sdud = load_sdud_monthly(sdud_path, mapping)

    # ── 4. NADAC monthly ──────────────────────────────────────────────────────
    nadac_dir = _pick_existing_dir(DATA_ROOT, NADAC_DIR_CANDIDATES)
    print(f"Loading NADAC: {nadac_dir}")
    pricing_units = {nadac_units} if nadac_units else None
    nadac_m, nadac_weekly = load_nadac_monthly(nadac_dir, pricing_units=pricing_units)

    # Save weekly NADAC for downstream annual averaging
    nadac_weekly_path = OUTPUT_DIR / f"{RUN_TAG}-nadac_weekly.csv"
    nadac_weekly["week_date"] = pd.to_datetime(nadac_weekly["week_date"]).dt.strftime("%Y-%m-%d")
    nadac_weekly["date"]      = pd.to_datetime(nadac_weekly["date"]).dt.strftime("%Y-%m-%d")
    nadac_weekly.to_csv(nadac_weekly_path, index=False)
    print(f"Saved NADAC weekly: {nadac_weekly_path}  ({len(nadac_weekly):,} rows)")

    # ── 5. Merge ──────────────────────────────────────────────────────────────
    panel = sdud.merge(nadac_m, on=["ndc11", "date"], how="left")

    # Derived price columns
    panel["sdud_price_total_per_unit"]   = _safe_div(panel["total_amount_reimbursed"],    panel["sdud_units_reimbursed"])
    panel["sdud_price_medicaid_per_unit"]= _safe_div(panel["medicaid_amount_reimbursed"], panel["sdud_units_reimbursed"])

    # Validate NDC11
    panel["ndc11"] = panel["ndc11"].astype("string")
    panel = panel[panel["ndc11"].str.fullmatch(r"\d{11}")]
    panel = panel[panel["ndc11"] != "00000000000"]

    # ── 6. Column order & output ──────────────────────────────────────────────
    col_order = [
        "ndc11", "date", "drug_name",
        "sdud_num_prescriptions",
        "sdud_units_reimbursed",
        "total_amount_reimbursed",
        "medicaid_amount_reimbursed",
        "sdud_price_total_per_unit",
        "sdud_price_medicaid_per_unit",
        "nadac_price",
        "pricing_unit_mode",
        "n_obs",
    ]
    extra = [c for c in panel.columns if c not in col_order]
    panel = panel.sort_values(["drug_name", "ndc11", "date"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")

    out_csv = OUTPUT_DIR / f"{RUN_TAG}-sdud_nadac_panel.csv"
    panel.to_csv(out_csv, index=False, columns=col_order + extra)

    # Summary
    print(f"\n{'='*60}")
    print(f"Panel saved: {out_csv}")
    print(f"Rows:        {len(panel):,}")
    print(f"NDC11s:      {panel['ndc11'].nunique():,}")
    print(f"Drugs:       {panel['drug_name'].nunique()}")
    print(f"Date range:  {panel['date'].min()} – {panel['date'].max()}")
    print(f"\nRows per drug (top 10):")
    print(panel.groupby("drug_name").size().sort_values(ascending=False).head(10).to_string())
    print(f"{'='*60}\n")

    return panel


# -------------------- RUN --------------------
panel = build_panel()

# %%
