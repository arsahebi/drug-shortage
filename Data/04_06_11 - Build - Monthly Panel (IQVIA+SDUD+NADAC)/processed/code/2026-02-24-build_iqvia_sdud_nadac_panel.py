# %%
# -*- coding: utf-8 -*-

"""
Build monthly NDC11 panel by merging (old logic, path-updated only):
- IQVIA NDC-level (TRx, Extended Units) 2019–2025
- IQVIA pre-2019 (signature-only) allocated to NDC11 using learned signature→NDC shares
- SDUD monthly (utilization + reimbursement)
- NADAC monthly (pricing)

All outputs are written under:
  <DEFAULT_DATASET_DIR>/processed/

No figures are created.
"""

import os, re, warnings
from pathlib import Path
from typing import Optional, Set, List

import numpy as np
import pandas as pd


# ----------------------- Defaults -----------------------
DEFAULT_DATASET_DIR = (
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)"
)

RUN_TAG = "2026-02-24"

IQVIA_POST2019_CANDIDATES = [
    "06 - IQVIA/raw/Metformin Jul 2019 - Jun 2025 NDC Level.xlsx",
    "06 - Metformin Data/IQVIA/raw/Metformin Jul 2019 - Jun 2025 NDC Level.xlsx",
    "06 - Metformin Data/IQVIA/Metformin Jul 2019 - Jun 2025 NDC Level.xlsx",
]

IQVIA_PRE2019_CANDIDATES = [
    "06 - IQVIA/raw/Metformin Jan 2015 - Mar 2025 No NDC.xlsx",
    "06 - Metformin Data/IQVIA/raw/Metformin Jan 2015 - Mar 2025 No NDC.xlsx",
    "06 - Metformin Data/IQVIA/Metformin Jan 2015 - Mar 2025 No NDC.xlsx",
]

SDUD_MONTHLY_PARQUET_CANDIDATES = [
    "04 - Medicaid - SDUD/processed/2026-02-24-SDUDmonthly.parquet",
    "04 - Medicaid - SDUD/processed/2025-12-18-SDUDmonthly.parquet",
    "04 - Medicaid - SDUD/processed/sdud_monthly_national.parquet",
    "04 - Medicaid - SDUD/sdud_monthly_national.parquet",
]

NADAC_DIR_CANDIDATES = [
    "11 - Medicaid - NADAC/raw",
    "11 - Medicaid - NADAC",
]

META_FIELDS = ["Combined Molecule", "Manufacturer", "Prod Form2", "Product Sum", "Strength"]


# ----------------------------- Helpers ------------------------------------
def _pick_existing(data_root: Path, rel_candidates: List[str]) -> Path:
    for rel in rel_candidates:
        p = data_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these input paths exist:\n  - " + "\n  - ".join(str(data_root / r) for r in rel_candidates)
    )

def _pick_existing_dir(data_root: Path, rel_candidates: List[str]) -> Path:
    for rel in rel_candidates:
        p = data_root / rel
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "None of these input directories exist:\n  - " + "\n  - ".join(str(data_root / r) for r in rel_candidates)
    )

def _month_end(x) -> pd.Series:
    return pd.to_datetime(x).dt.to_period("M").dt.to_timestamp("M")

def _find_month_cols(cols) -> list:
    pat = re.compile(
        r"^(?:(?:TRx|Extended Units|EUTRx)\s*)?"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
        re.I,
    )
    return [c for c in cols if pat.match(str(c).strip())]

def _strip_metric_prefix(col: str) -> str:
    return re.sub(r"^(TRx|Extended Units|EUTRx)\s*", "", str(col).strip(), flags=re.I).strip()

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def _mode_nonnull(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0] if not s.mode().empty else s.iloc[0]

def _norm_str(x) -> str:
    return str(x).strip().upper() if pd.notna(x) else ""

def _make_signature_row(row: pd.Series) -> str:
    parts = []
    for c in META_FIELDS:
        parts.append(_norm_str(row.get(c, "")))
    return " || ".join(parts)

# ---------- NEW: robust NADAC NDC normalizer ----------
def _nadac_to_ndc11(s: pd.Series) -> pd.Series:
    """
    NADAC sometimes has NDC as 9/10 digits (leading zeros dropped), or hyphenated.
    Rule:
      - strip all non-digits
      - if length in [8..11], left-pad to 11
      - else -> NA
      - keep only exact 11-digit results and exclude all-zeros
    """
    digits = s.astype("string").fillna("").str.replace(r"\D+", "", regex=True)
    out = digits.where(digits.str.len().between(8, 11), pd.NA)
    out = out.str.zfill(11)
    out = out.where(out.str.fullmatch(r"\d{11}"), pd.NA)
    out = out.where(out != "00000000000", pd.NA)
    return out


# ----------------------------- IQVIA loaders ------------------------------
def load_iqvia_ndc11_monthly_post2019(
    xlsx_path: str,
    start: str = "2019-07-01",
    end: str = "2025-06-30",
) -> pd.DataFrame:
    book = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    frames = []

    for sheet in ("TRx", "Extended Units"):
        if sheet not in book:
            continue

        wide = book[sheet].copy()
        wide.columns = (
            wide.columns.astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )

        if "NDC" not in wide.columns:
            raise ValueError(f"'NDC' column not found on sheet {sheet}")

        wide["ndc11"] = wide["NDC"].astype(str).str.extract(r"(?<!\d)(\d{11})(?!\d)", expand=False)
        wide["ndc11"] = wide["ndc11"].fillna("").str.zfill(11)
        wide = wide[wide["ndc11"] != ""]

        for c in META_FIELDS:
            if c not in wide.columns:
                wide[c] = ""

        month_cols = _find_month_cols(wide.columns)
        if not month_cols:
            raise ValueError(f"IQVIA sheet '{sheet}' has no month columns like 'Jul 2019'.")

        rename = {c: _strip_metric_prefix(c) for c in month_cols}
        wide = wide.rename(columns=rename)

        long = (
            wide[["ndc11"] + META_FIELDS + list(rename.values())]
            .melt(
                id_vars=["ndc11"] + META_FIELDS,
                value_vars=list(rename.values()),
                var_name="month",
                value_name="volume",
            )
            .dropna(subset=["volume"])
        )
        long["date"] = _month_end(pd.to_datetime(long["month"], format="%b %Y"))
        long["metric"] = sheet
        frames.append(long[["ndc11", "date", "metric", "volume"] + META_FIELDS])

    if not frames:
        raise ValueError("No TRx/Extended Units sheets found in the IQVIA NDC-level file.")

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby(["ndc11", "date", "metric"] + META_FIELDS, as_index=False).agg(volume=("volume", "sum"))
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out


def load_iqvia_no_ndc_2015_2019(
    xlsx_path: str,
    end: str = "2019-06-30",
) -> pd.DataFrame:
    book = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    keep_metrics = {"TRx", "Extended Units"}
    frames = []

    for sheet_name, wide in book.items():
        if sheet_name not in keep_metrics:
            continue

        wide = wide.copy()
        wide.columns = (
            wide.columns.astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )

        month_cols = _find_month_cols(wide.columns)
        if not month_cols:
            continue

        rename = {c: _strip_metric_prefix(c) for c in month_cols}
        wide = wide.rename(columns=rename)

        for c in META_FIELDS:
            if c not in wide.columns:
                wide[c] = ""

        wide["signature"] = wide.apply(_make_signature_row, axis=1)

        long = (
            wide[["signature"] + META_FIELDS + list(rename.values())]
            .melt(
                id_vars=["signature"] + META_FIELDS,
                value_vars=list(rename.values()),
                var_name="month",
                value_name="volume",
            )
            .dropna(subset=["volume"])
        )
        long["date"] = _month_end(pd.to_datetime(long["month"], format="%b %Y"))
        long = long[long["date"] <= end]
        long["metric"] = sheet_name
        frames.append(long[["signature", "date", "metric", "volume"] + META_FIELDS])

    if not frames:
        return pd.DataFrame(columns=["signature", "date", "metric", "volume"] + META_FIELDS)

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby(["signature", "date", "metric"] + META_FIELDS, as_index=False).agg(volume=("volume", "sum"))
    return out


def learn_signature_ndc_shares(iq_post: pd.DataFrame, share_min: float = 0.0) -> pd.DataFrame:
    tmp = iq_post.copy()
    tmp["signature"] = tmp.apply(_make_signature_row, axis=1)

    w = (
        tmp.pivot_table(
            index=["ndc11", "signature", "date"],
            columns="metric",
            values="volume",
            aggfunc="sum",
        )
        .reset_index()
    )

    w["weight"] = w.get("TRx", 0.0).fillna(0.0) + w.get("Extended Units", 0.0).fillna(0.0)

    sig_ndc = w.groupby(["signature", "ndc11"], as_index=False)["weight"].sum()

    sig_tot = (
        sig_ndc.groupby("signature", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "total_weight"})
    )

    shares = sig_ndc.merge(sig_tot, on="signature", how="left")
    shares["alloc_share"] = np.where(
        shares["total_weight"] > 0, shares["weight"] / shares["total_weight"], np.nan
    )
    shares = shares.dropna(subset=["alloc_share"]).copy()

    if share_min and share_min > 0:
        shares = shares[shares["alloc_share"] >= share_min].copy()
        ren = (
            shares.groupby("signature", as_index=False)["alloc_share"]
            .sum()
            .rename(columns={"alloc_share": "renorm"})
        )
        shares = shares.merge(ren, on="signature", how="left")
        shares["alloc_share"] = np.where(
            shares["renorm"] > 0, shares["alloc_share"] / shares["renorm"], shares["alloc_share"]
        )
        shares = shares.drop(columns=["renorm"])

    return shares[["signature", "ndc11", "alloc_share"]].sort_values(
        ["signature", "alloc_share"], ascending=[True, False]
    )


def backfill_pre2019_ndc(
    pre_path: str,
    iq_post: pd.DataFrame,
    strategy: str = "proportional",
    share_min: float = 0.0,
) -> pd.DataFrame:
    pre = load_iqvia_no_ndc_2015_2019(pre_path)
    if pre.empty:
        return pd.DataFrame(columns=["ndc11", "date", "metric", "volume", "signature"] + META_FIELDS)

    shares = learn_signature_ndc_shares(iq_post, share_min=share_min).copy()

    if strategy.lower() == "modal":
        shares = (
            shares.sort_values(["signature", "alloc_share"], ascending=[True, False])
            .groupby(["signature"], as_index=False)
            .head(1)
        )
        shares["alloc_share"] = 1.0

    alloc = pre.merge(shares, on=["signature"], how="left")
    alloc = alloc.dropna(subset=["alloc_share"]).copy()
    alloc["volume"] = alloc["volume"] * alloc["alloc_share"]

    out = (
        alloc.groupby(["ndc11", "date", "metric", "signature"] + META_FIELDS, as_index=False)
        .agg(volume=("volume", "sum"))
    )
    return out


# ----------------------------- SDUD / NADAC loaders ------------------------
def load_sdud_monthly(parquet_path: str) -> pd.DataFrame:
    m = pd.read_parquet(parquet_path).copy()

    if "month_start" in m.columns:
        m["date"] = _month_end(m["month_start"])
    elif "date" in m.columns:
        m["date"] = _month_end(m["date"])
    else:
        raise ValueError("SDUD monthly parquet lacks a month column (month_start or date).")

    if "ndc11" not in m.columns:
        if "ndc" in m.columns:
            m["ndc11"] = m["ndc"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11)
        else:
            raise ValueError("SDUD monthly parquet lacks ndc11 or ndc.")

    g = (
        m.groupby(["ndc11", "date"], as_index=False)
        .agg(
            sdud_units_reimbursed=("units_reimbursed", "sum"),
            sdud_num_prescriptions=("num_prescriptions", "sum"),
            total_amount_reimbursed=("total_amount_reimbursed", "sum") if "total_amount_reimbursed" in m.columns else ("units_reimbursed", "size"),
            medicaid_amount_reimbursed=("medicaid_amount_reimbursed", "sum") if "medicaid_amount_reimbursed" in m.columns else ("units_reimbursed", "size"),
        )
    )
    if "total_amount_reimbursed" not in m.columns:
        g["total_amount_reimbursed"] = np.nan
    if "medicaid_amount_reimbursed" not in m.columns:
        g["medicaid_amount_reimbursed"] = np.nan

    return g


# def load_nadac_avg(nadac_dir: str, freq: str = "M", pricing_units: Optional[Set[str]] = None) -> pd.DataFrame:
#     """
#     NADAC monthly table keyed by (ndc11, month_end_date).
#     Robust to NADAC files where NDC is 9/10/11 digits (leading zeros dropped) or hyphenated.
#     """
#     nadac_dir = Path(nadac_dir)
#     files = sorted(nadac_dir.glob("NADAC*.csv"))
#     if not files:
#         raise FileNotFoundError(f"No NADAC*.csv under {nadac_dir}")

#     frames = []
#     for f in files:
#         df = pd.read_csv(f, dtype=str, low_memory=False)

#         # normalize columns
#         df.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in df.columns]

#         # identify columns
#         col_ndc = None
#         for cand in ("ndc", "ndc11", "ndc_code"):
#             if cand in df.columns:
#                 col_ndc = cand
#                 break

#         col_price = None
#         for cand in ("nadac_per_unit", "nadac"):
#             if cand in df.columns:
#                 col_price = cand
#                 break

#         col_date = None
#         for cand in ("effective_date", "as_of_date"):
#             if cand in df.columns:
#                 col_date = cand
#                 break

#         col_unit = "pricing_unit" if "pricing_unit" in df.columns else None

#         if not (col_ndc and col_price and col_date):
#             continue

#         tmp = pd.DataFrame({
#             "ndc11": _nadac_to_ndc11(df[col_ndc]),
#             "date": pd.to_datetime(df[col_date], errors="coerce"),
#             "nadac_price": pd.to_numeric(df[col_price], errors="coerce"),
#             "pricing_unit": (df[col_unit].astype(str) if col_unit else np.nan),
#         }).dropna(subset=["ndc11", "date", "nadac_price"])

#         # month-end alignment (must match IQVIA/SDUD date keys)
#         tmp["date"] = tmp["date"].dt.to_period(freq).dt.to_timestamp(freq)

#         if pricing_units is not None and col_unit:
#             tmp = tmp[tmp["pricing_unit"].isin(pricing_units)]

#         frames.append(tmp[["ndc11", "date", "nadac_price", "pricing_unit"]])

#     if not frames:
#         raise ValueError(f"Could not parse any NADAC files under: {nadac_dir}")

#     all_ = pd.concat(frames, ignore_index=True)

#     g = (
#         all_.groupby(["ndc11", "date"], as_index=False)
#             .agg(
#                 nadac_price=("nadac_price", "mean"),
#                 n_obs=("nadac_price", "size"),
#                 pricing_unit_mode=("pricing_unit", _mode_nonnull),
#             )
#     )
#     return g

from pathlib import Path
import re
import numpy as np
import pandas as pd
from typing import Optional, Set

def load_nadac_monthly(nadac_dir: str, pricing_units: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Build monthly NADAC table keyed by (ndc11, month_end_date).

    UPDATED 2026-02-24: The previous version averaged weekly prices within each
    month first, then averaged those monthly values annually (two-step average).
    This implicitly up-weighted 4-week months vs 5-week months. The fix: we now
    compute nadac_price at the monthly level the same way (mean of weeks in month,
    needed for the monthly panel merge), but n_obs now reflects the total weekly
    count so downstream annual scripts can instead compute a direct mean across
    ALL weekly records in a year (group by ndc11 + year(week_date), then mean of
    nadac_price at weekly level). This handles 52- or 53-week years automatically.

    - nadac_price: mean of all weekly NADAC prices in that month (for monthly panel)
    - n_obs: count of weekly records in that month
    - pricing_unit_mode: most common pricing unit in that month
    """
    nadac_dir = Path(nadac_dir)
    files = sorted(nadac_dir.glob("NADAC*.csv"))
    if not files:
        raise FileNotFoundError(f"No NADAC*.csv under {nadac_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, low_memory=False)

        # normalize column names
        df.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in df.columns]

        # identify columns robustly
        col_ndc = None
        for cand in ("ndc", "ndc_code", "ndc11"):
            if cand in df.columns:
                col_ndc = cand
                break

        col_date = "as_of_date"

        col_price = None
        for cand in ("nadac_per_unit", "nadac"):
            if cand in df.columns:
                col_price = cand
                break

        col_unit = "pricing_unit" if "pricing_unit" in df.columns else None

        if not (col_ndc and col_date and col_price):
            continue

        tmp = pd.DataFrame({
            "ndc11": _nadac_to_ndc11(df[col_ndc]),
            "week_date": pd.to_datetime(df[col_date], errors="coerce"),
            "nadac_price": pd.to_numeric(df[col_price], errors="coerce"),
            "pricing_unit": (df[col_unit].astype(str) if col_unit else np.nan),
        }).dropna(subset=["ndc11", "week_date", "nadac_price"])

        # Optional filter to a specific pricing unit set
        if pricing_units is not None and col_unit:
            tmp = tmp[tmp["pricing_unit"].isin(pricing_units)]

        # Deduplicate weekly records in case downloaded files overlap
        # (prevents double-counting n_obs across files)
        tmp = tmp.drop_duplicates(subset=["ndc11", "week_date", "nadac_price", "pricing_unit"])

        # month-end alignment for merge with IQVIA monthly panel
        tmp["date"] = tmp["week_date"].dt.to_period("M").dt.to_timestamp("M")

        # Retain week_date so downstream annual scripts can average directly
        # across all weekly records in a year (bypassing monthly intermediate).
        frames.append(tmp[["ndc11", "date", "week_date", "nadac_price", "pricing_unit"]])

    if not frames:
        raise ValueError(f"Could not parse any NADAC files under: {nadac_dir}")

    all_ = pd.concat(frames, ignore_index=True)

    # ── Weekly table (Option A) ───────────────────────────────────────────
    # Saved separately by build_panel as nadac_weekly.parquet.
    # Downstream QA/figure scripts load this and group by
    # (ndc11, year(week_date)) then take mean of nadac_price to get a true
    # annual average giving every week equal weight (52 or 53 weeks).
    nadac_weekly = all_[["ndc11", "week_date", "date", "nadac_price", "pricing_unit"]].copy()

    # ── Monthly aggregation (unchanged — for main panel merge) ────────────
    # nadac_price = mean of weekly prices in that calendar month.
    # Used only for the monthly panel CSV. Do NOT average these monthly
    # values to get annual price — use nadac_weekly.parquet instead.
    nadac_m = (
        all_.groupby(["ndc11", "date"], as_index=False)
            .agg(
                nadac_price=("nadac_price", "mean"),          # mean of weekly prices in month
                n_obs=("nadac_price", "size"),                # count of weekly records in month
                pricing_unit_mode=("pricing_unit", _mode_nonnull),
            )
    )
    return nadac_m, nadac_weekly

# ----------------------------- Runner -------------------------------------
def build_panel(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    run_tag: str = RUN_TAG,
    nadac_units: Optional[str] = None,
    share_min: float = 0.0,
    pre_mapping_strategy: str = "proportional",  # or "modal"
) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    data_root = dataset_dir.parent
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    iqvia_path   = _pick_existing(data_root, IQVIA_POST2019_CANDIDATES)
    pre_path     = _pick_existing(data_root, IQVIA_PRE2019_CANDIDATES)
    sdud_parquet = _pick_existing(data_root, SDUD_MONTHLY_PARQUET_CANDIDATES)
    nadac_dir    = _pick_existing_dir(data_root, NADAC_DIR_CANDIDATES)

    out_csv = processed_dir / f"{run_tag}-iqvia_with_sdud_nadac.cleaned.csv"
    map_csv = processed_dir / f"{run_tag}-signature_to_ndc_mapping.csv"
    audit_path = processed_dir / f"{run_tag}-pre_allocation_audit.csv"

    print(f"Loading IQVIA 2019+: {iqvia_path}")
    iq_post = load_iqvia_ndc11_monthly_post2019(str(iqvia_path))

    shares = learn_signature_ndc_shares(iq_post, share_min=share_min)
    shares.to_csv(map_csv, index=False)
    print(f"Saved signature→NDC shares: {map_csv}")

    print(f"Backfilling 2015–Jun 2019 ({pre_mapping_strategy}) from: {pre_path}")
    iq_pre = backfill_pre2019_ndc(
        str(pre_path),
        iq_post,
        strategy=pre_mapping_strategy,
        share_min=share_min,
    )

    pre_src = load_iqvia_no_ndc_2015_2019(str(pre_path))
    pre_tot = pre_src.groupby(["signature", "date", "metric"], as_index=False).agg(pre_total=("volume", "sum"))
    alloc_tot = iq_pre.groupby(["signature", "date", "metric"], as_index=False).agg(allocated_total=("volume", "sum"))
    audit = pre_tot.merge(alloc_tot, on=["signature", "date", "metric"], how="outer").fillna(0.0)
    audit["delta"] = audit["allocated_total"] - audit["pre_total"]
    audit.to_csv(audit_path, index=False)
    print(f"Saved pre-allocation audit: {audit_path}")

    iq_long = pd.concat(
        [
            iq_post[["ndc11", "date", "metric", "volume"] + META_FIELDS],
            iq_pre[["ndc11", "date", "metric", "volume"] + META_FIELDS],
        ],
        ignore_index=True,
    )

    iq_wide = (
        iq_long.pivot_table(index=["ndc11", "date"], columns="metric", values="volume", aggfunc="sum")
        .reset_index()
        .rename(columns={"TRx": "iqvia_trx", "Extended Units": "iqvia_extended_units"})
    )
    for c in ("iqvia_trx", "iqvia_extended_units"):
        if c not in iq_wide.columns:
            iq_wide[c] = np.nan

    meta_map = (
        iq_long.groupby("ndc11", as_index=False)
        .agg({c: _mode_nonnull for c in META_FIELDS if c in iq_long.columns})
    )

    print(f"Loading SDUD:  {sdud_parquet}")
    sdud = load_sdud_monthly(str(sdud_parquet))

    print(f"Loading NADAC: {nadac_dir}")
    pricing_units = {nadac_units} if nadac_units else None
    # nadac = load_nadac_avg(str(nadac_dir), freq="M", pricing_units=pricing_units)
    nadac, nadac_weekly = load_nadac_monthly(str(nadac_dir), pricing_units=pricing_units)

    # Save weekly-level NADAC for downstream annual averaging (Option A).
    # QA scripts should load this and group by (ndc11, year(week_date)),
    # then take mean of nadac_price — giving each week equal weight.
    nadac_weekly_path = processed_dir / f"{run_tag}-nadac_weekly.csv"
    nadac_weekly["week_date"] = pd.to_datetime(nadac_weekly["week_date"]).dt.strftime("%Y-%m-%d")
    nadac_weekly["date"] = pd.to_datetime(nadac_weekly["date"]).dt.strftime("%Y-%m-%d")
    nadac_weekly.to_csv(nadac_weekly_path, index=False)
    print(f"Saved NADAC weekly table: {nadac_weekly_path}  ({len(nadac_weekly):,} rows)")

    merged = (
        iq_wide.merge(sdud, on=["ndc11", "date"], how="left")
        .merge(nadac, on=["ndc11", "date"], how="left")
        .merge(meta_map, on="ndc11", how="left")
    )

    merged["ndc11"] = merged["ndc11"].astype("string")
    merged = merged[merged["ndc11"].str.fullmatch(r"\d{11}")]
    merged = merged[merged["ndc11"] != "00000000000"]

    merged["sdud_price_total_per_unit"] = _safe_div(merged["total_amount_reimbursed"], merged["sdud_units_reimbursed"])
    merged["sdud_price_medicaid_per_unit"] = _safe_div(
        merged["medicaid_amount_reimbursed"], merged["sdud_units_reimbursed"]
    )

    cols_front = ["ndc11", "date"] + META_FIELDS
    cols_rest = [
        "iqvia_trx",
        "iqvia_extended_units",
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
    order = [c for c in cols_front + cols_rest if c in merged.columns]

    merged = merged.sort_values(["ndc11", "date"]).reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")

    merged.to_csv(out_csv, index=False, columns=order + [c for c in merged.columns if c not in order])
    print(f"\nSaved cleaned panel: {out_csv}\nRows: {len(merged):,}")

    return merged


# -------------------- RUN --------------------
panel = build_panel(share_min=0.01)

# %%
