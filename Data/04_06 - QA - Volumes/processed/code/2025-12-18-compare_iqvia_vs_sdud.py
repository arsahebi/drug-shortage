# %%
# 04_06 - QA - Volumes
# 2025-12-18-compare_iqvia_vs_sdud.py
#
# Put this file in:
#   .../Data/04_06 - QA - Volumes/processed/code/2025-12-18-compare_iqvia_vs_sdud.py
#
# It writes ONLY to:
#   .../Data/04_06 - QA - Volumes/raw/source.txt
#   .../Data/04_06 - QA - Volumes/processed/<TAG>-*.{csv,parquet}
#
# Best-practice NDC11 comparison uses:
#   IQVIA: Metformin Jul 2019 - Jun 2025 NDC Level.xlsx  (raw IQVIA NDC-level)
#   SDUD:  2025-12-18-SDUDmonthly.parquet (processed monthly)
#
# Optional/extra: a 2015–2025 "wide enriched" IQVIA file can be loaded if you want to
# (kept here only as a reference; best-practice does NOT depend on it).

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- CONFIG (edit if needed) --------------------

DATA_ROOT = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/"
    "North Carolina State University/Project - Drug Shortage/Data"
)

QA_DIR = DATA_ROOT / "04_06 - QA - Volumes"
RAW_DIR = QA_DIR / "raw"
PROCESSED_DIR = QA_DIR / "processed"

# ---------- BEST PRACTICE INPUTS (NDC11 apples-to-apples, 2019–2025) ----------
IQVIA_NDC11_XLSX = DATA_ROOT / "06 - IQVIA" / "raw" / "Metformin Jul 2019 - Jun 2025 NDC Level.xlsx"
SDUD_MONTHLY_PARQUET = DATA_ROOT / "04 - Medicaid - SDUD" / "processed" / "2025-12-18-SDUDmonthly.parquet"

# ---------- OPTIONAL: 2015–2025 IQVIA NDC-level/enriched workbook ----------
# Keep here only if you want to run other (non-best-practice) checks later.
IQVIA_2015_2025_XLSX = DATA_ROOT / "06 - IQVIA" / "processed" / "2025-12-18-Metformin20152025NDClevel.xlsx"

# Tag used in output filenames
TAG = "2025-12-18"

# Best-practice date window (matches the IQVIA NDC-level file coverage)
DATE_MIN = "2019-07-01"
DATE_MAX = "2025-06-30"

# Optional: write merged CSV as well (parquet always written)
WRITE_MERGED_CSV = True

MOLECULE = "METFORMIN"

# -------------------- helpers --------------------

MONTH_PAT_GENERIC = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", re.I)
MONTH_PAT_WITH_PREFIX = re.compile(
    r"^(TRx|Extended Units|EUTRx)?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
    re.I,
)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _month_end(x) -> pd.Series:
    return pd.to_datetime(x).dt.to_period("M").dt.to_timestamp("M")

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )
    return out

def _corr_safe(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(a.corr(b))

def _write_source_txt():
    _ensure_dir(RAW_DIR)
    script_path = Path(__file__).resolve() if "__file__" in globals() else Path("(interactive)")
    lines = [
        "QA module: 04_06 - QA - Volumes",
        "Purpose: Compare IQVIA vs SDUD monthly volumes (TRx and Extended Units).",
        "",
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Script used: {script_path}",
        "",
        "BEST PRACTICE INPUTS (NDC11 apples-to-apples):",
        f"  IQVIA NDC-level Excel (raw): {IQVIA_NDC11_XLSX}",
        f"  SDUD monthly parquet (processed): {SDUD_MONTHLY_PARQUET}",
        "",
        "Optional (not used for best-practice outputs unless you call it):",
        f"  IQVIA 2015–2025 workbook: {IQVIA_2015_2025_XLSX}",
        "",
        "Metric mapping:",
        "  IQVIA 'Extended Units'  <->  SDUD 'units_reimbursed'",
        "  IQVIA 'TRx'             <->  SDUD 'num_prescriptions'",
        "",
        f"Date window: DATE_MIN={DATE_MIN}, DATE_MAX={DATE_MAX}",
    ]
    (RAW_DIR / "source.txt").write_text("\n".join(lines), encoding="utf-8")


# -------------------- loaders --------------------

def load_sdud_monthly_long_ndc11(sdud_monthly_parquet: Path) -> pd.DataFrame:
    """
    SDUD monthly parquet -> tidy long:
      ndc11, date (month-end), metric in {TRx, Extended Units}, volume
    """
    m = pd.read_parquet(sdud_monthly_parquet)

    if "month_start" in m.columns:
        m["date"] = _month_end(m["month_start"])
    elif "date" in m.columns:
        m["date"] = _month_end(m["date"])
    else:
        raise ValueError("SDUD monthly parquet must have 'month_start' or 'date'.")

    if "ndc11" not in m.columns:
        if "ndc" in m.columns:
            m["ndc11"] = m["ndc"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11)
        else:
            raise ValueError("SDUD monthly parquet must have 'ndc11' or 'ndc'.")

    g = (
        m.groupby(["ndc11", "date"], as_index=False)
        .agg(
            units_reimbursed=("units_reimbursed", "sum"),
            num_prescriptions=("num_prescriptions", "sum"),
        )
    )

    eu = g[["ndc11", "date", "units_reimbursed"]].rename(columns={"units_reimbursed": "volume"})
    eu["metric"] = "Extended Units"

    trx = g[["ndc11", "date", "num_prescriptions"]].rename(columns={"num_prescriptions": "volume"})
    trx["metric"] = "TRx"

    return pd.concat([eu, trx], ignore_index=True)[["ndc11", "date", "metric", "volume"]]


def load_iqvia_ndc11_monthly_best_practice(iqvia_xlsx: Path, molecule: str = "METFORMIN") -> pd.DataFrame:
    """
    BEST PRACTICE IQVIA loader for:
      'Metformin Jul 2019 - Jun 2025 NDC Level.xlsx'

    Handles month headers like:
      'TRx Jul 2019' or 'Extended Units Jul 2019' or just 'Jul 2019'
    """
    book = pd.read_excel(iqvia_xlsx, sheet_name=None, engine="openpyxl")
    frames = []

    # These sheets exist in the NDC-level file
    for metric in ("TRx", "Extended Units"):
        if metric not in book:
            continue

        wide = _clean_headers(book[metric])

        # Optional molecule filter if present
        if "Combined Molecule" in wide.columns:
            wide = wide[wide["Combined Molecule"].astype(str).str.upper() == molecule]

        # Extract ndc11 from the messy NDC field (this file uses an 'NDC' column)
        if "ndc11" in wide.columns:
            wide["ndc11"] = wide["ndc11"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11)
            wide = wide[wide["ndc11"].str.len() == 11]
        elif "NDC" in wide.columns:
            ndc = wide["NDC"].astype(str).str.extract(r"(?<!\d)(\d{11})(?!\d)", expand=False).fillna("")
            wide["ndc11"] = ndc.str.zfill(11)
            wide = wide[wide["ndc11"].str.len() == 11]
        else:
            raise ValueError(f"IQVIA sheet '{metric}' missing 'NDC' or 'ndc11' column.")

        # Month columns: allow optional prefix like "TRx Jul 2019"
        month_cols = [c for c in wide.columns if re.match(
            r"^(TRx|Extended Units|EUTRx)?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$",
            str(c).strip(), flags=re.I
        )]
        if not month_cols:
            sample = list(map(str, wide.columns[:40]))
            raise ValueError(
                f"IQVIA sheet '{metric}' has no recognizable month columns. First columns: {sample}"
            )

        # Strip metric prefix so var_name is plain "Jul 2019"
        rename = {c: re.sub(r"^(TRx|Extended Units|EUTRx)\s*", "", str(c).strip(), flags=re.I) for c in month_cols}
        wide = wide.rename(columns=rename)
        month_cols_clean = list(rename.values())

        long = (
            wide.melt(id_vars=["ndc11"], value_vars=month_cols_clean, var_name="month", value_name="volume")
            .dropna(subset=["volume"])
        )
        long["date"] = _month_end(pd.to_datetime(long["month"], format="%b %Y"))
        long["metric"] = metric
        frames.append(long[["ndc11", "date", "metric", "volume"]])

    if not frames:
        raise ValueError("No 'TRx' or 'Extended Units' sheets found in IQVIA NDC-level workbook.")

    iqvia = pd.concat(frames, ignore_index=True)
    iqvia = iqvia.groupby(["ndc11", "date", "metric"], as_index=False).agg(volume=("volume", "sum"))

    # Clamp to file coverage window (best-practice)
    iqvia = iqvia[(iqvia["date"] >= "2019-07-01") & (iqvia["date"] <= "2025-06-30")]
    return iqvia


# -------------------- compare + write outputs --------------------

def run_best_practice_ndc11_compare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    BEST PRACTICE OUTPUTS (these match your older 'comparison_ndc11_2019_2025.csv' logic):
      - <TAG>-iqvia_vs_sdud_ndc11.parquet
      - <TAG>-iqvia_vs_sdud_ndc11.csv (optional)
      - <TAG>-iqvia_vs_sdud_summary.csv
      - <TAG>-iqvia_vs_sdud_per_ndc11_corr.csv
    """
    _ensure_dir(PROCESSED_DIR)
    # _write_source_txt()

    iqvia = load_iqvia_ndc11_monthly_best_practice(IQVIA_NDC11_XLSX, molecule=MOLECULE)
    sdud  = load_sdud_monthly_long_ndc11(SDUD_MONTHLY_PARQUET)

    # Apply same date window on SDUD to align exactly
    if DATE_MIN is not None:
        sdud = sdud[sdud["date"] >= pd.to_datetime(DATE_MIN)]
    if DATE_MAX is not None:
        sdud = sdud[sdud["date"] <= pd.to_datetime(DATE_MAX)]

    comp = (
        iqvia.merge(sdud, on=["ndc11", "date", "metric"], suffixes=("_iqvia", "_sdud"))
        .sort_values(["ndc11", "metric", "date"])
        .reset_index(drop=True)
    )

    # Summary by metric
    rows = []
    for metric in sorted(comp["metric"].unique()):
        d = comp[comp["metric"] == metric]
        rows.append({
            "metric": metric,
            "n_rows": len(d),
            "corr_pooled": _corr_safe(d["volume_iqvia"], d["volume_sdud"]),
            "iqvia_sum": float(d["volume_iqvia"].sum()),
            "sdud_sum": float(d["volume_sdud"].sum()),
        })
    summary = pd.DataFrame(rows)

    # Per-NDC correlation
    per_ndc = (
        comp.groupby(["ndc11", "metric"])
        .apply(lambda g: pd.Series({"corr": _corr_safe(g["volume_iqvia"], g["volume_sdud"]), "n_months": len(g)}))
        .reset_index()
    )

    # Write outputs (ONLY inside QA module)
    out_comp_parquet = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_ndc11.parquet"
    out_comp_csv     = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_ndc11.csv"
    out_summary_csv  = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_summary.csv"
    out_perndc_csv   = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_per_ndc11_corr.csv"

    comp.to_parquet(out_comp_parquet, index=False)
    summary.to_csv(out_summary_csv, index=False)
    per_ndc.to_csv(out_perndc_csv, index=False)
    if WRITE_MERGED_CSV:
        comp.to_csv(out_comp_csv, index=False)

    # Console diagnostics
    print("=== BEST PRACTICE (NDC11 2019–2025) ===")
    print("Saved outputs:")
    print(f"- {out_comp_parquet}")
    if WRITE_MERGED_CSV:
        print(f"- {out_comp_csv}")
    print(f"- {out_summary_csv}")
    print(f"- {out_perndc_csv}")
    print(f"- {RAW_DIR / 'source.txt'}")

    print("\nQuick diagnostics:")
    print(f"Matched rows: {len(comp):,}")
    for _, r in summary.iterrows():
        print(f"{r['metric']:14s} | n={int(r['n_rows']):8d} | pooled corr={r['corr_pooled']:.3f}")

    return comp, summary, per_ndc

# -------------------- NDC9 loaders + compare (2015–2025) --------------------

def _ndc11_only(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"(?<!\d)(\d{11})(?!\d)", text)

def _to_ndc9_list(text: str) -> list[str]:
    ndc11s = _ndc11_only(text)
    return sorted({n[:9] for n in ndc11s})

def load_iqvia_monthly_long_ndc9_from_2015_2025(iqvia_xlsx: Path, molecule: str = "METFORMIN"
                                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    NDC9 analysis (NOT best-practice):
      - Uses the 2015–2025 IQVIA workbook which may contain multiple NDC11s in a single row.
      - Creates monthly long at NDC9 and evenly splits volumes if multiple NDC9s found.
    Returns:
      (iqvia_ndc9_long, flags_wide_rows)
        iqvia_ndc9_long columns: ndc9, date, metric, volume, needs_review
        flags_wide_rows: original wide rows that had >1 NDC9 (for manual review)
    """
    book = pd.read_excel(iqvia_xlsx, sheet_name=None, engine="openpyxl")
    out = []
    flags = []

    keep_metrics = {"TRx", "Extended Units"}

    for metric, wide in book.items():
        if metric not in keep_metrics:
            continue

        wide = _clean_headers(wide)

        if "Combined Molecule" in wide.columns:
            wide = wide[wide["Combined Molecule"].astype(str).str.upper() == molecule]

        if "NDC" not in wide.columns:
            raise ValueError(f"IQVIA 2015–2025 sheet '{metric}' missing 'NDC' column.")

        # month columns like "TRxJan 2015" / "TRx Jan 2015" / "Jan 2015"
        month_cols = [c for c in wide.columns if MONTH_PAT_WITH_PREFIX.match(str(c).strip())]
        if not month_cols:
            sample = list(map(str, wide.columns[:50]))
            raise ValueError(
                f"IQVIA 2015–2025 sheet '{metric}' has no recognizable month columns. First cols: {sample}"
            )

        # normalize month headers to "Jan 2015"
        rename = {c: re.sub(r"^(TRx|Extended Units|EUTRx)\s*", "", str(c).strip(), flags=re.I) for c in month_cols}
        wide = wide.rename(columns=rename)
        month_cols_clean = list(rename.values())

        # parse NDC9 list per row
        wide["ndc9_list"] = wide["NDC"].apply(_to_ndc9_list)
        wide["n_ndc9"] = wide["ndc9_list"].map(len)

        # flag multi-ndc rows
        multi = wide[wide["n_ndc9"] > 1].copy()
        if not multi.empty:
            multi["metric"] = metric
            flags.append(multi)

        # SINGLE
        single = wide[wide["n_ndc9"] == 1].copy()
        if not single.empty:
            single["ndc9"] = single["ndc9_list"].str[0]
            long = (
                single.melt(id_vars=["ndc9"], value_vars=month_cols_clean,
                            var_name="month", value_name="volume")
                .dropna(subset=["volume"])
            )
            long["date"] = _month_end(pd.to_datetime(long["month"], format="%b %Y"))
            long["metric"] = metric
            long["needs_review"] = False
            out.append(long[["ndc9", "date", "metric", "volume", "needs_review"]])

        # MULTI (explode + even split)
        if not multi.empty:
            multi["share"] = 1.0 / multi["n_ndc9"]
            exploded = (
                multi[["ndc9_list", "share"] + month_cols_clean]
                .explode("ndc9_list")
                .rename(columns={"ndc9_list": "ndc9"})
            )
            long_m = (
                exploded.melt(id_vars=["ndc9", "share"], value_vars=month_cols_clean,
                              var_name="month", value_name="raw_volume")
                .dropna(subset=["raw_volume"])
            )
            long_m["volume"] = long_m["raw_volume"] * long_m["share"]
            long_m["date"] = _month_end(pd.to_datetime(long_m["month"], format="%b %Y"))
            long_m["metric"] = metric
            long_m["needs_review"] = True
            out.append(long_m[["ndc9", "date", "metric", "volume", "needs_review"]])

    iqvia_ndc9 = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["ndc9", "date", "metric", "volume", "needs_review"]
    )
    iqvia_ndc9 = (
        iqvia_ndc9.groupby(["ndc9", "date", "metric", "needs_review"], as_index=False)
        .agg(volume=("volume", "sum"))
    )

    flags_wide = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()
    return iqvia_ndc9, flags_wide


def load_sdud_monthly_long_ndc9(sdud_monthly_parquet: Path) -> pd.DataFrame:
    """
    SDUD monthly parquet -> tidy long at NDC9:
      ndc9, date, metric, volume
    """
    m = pd.read_parquet(sdud_monthly_parquet)

    if "month_start" in m.columns:
        m["date"] = _month_end(m["month_start"])
    elif "date" in m.columns:
        m["date"] = _month_end(m["date"])
    else:
        raise ValueError("SDUD monthly parquet must have 'month_start' or 'date'.")

    if "ndc11" in m.columns:
        m["ndc9"] = m["ndc11"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11).str.slice(0, 9)
    elif "ndc" in m.columns:
        m["ndc9"] = m["ndc"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 9)
    else:
        raise ValueError("SDUD monthly parquet must have 'ndc11' or 'ndc'.")

    g = (
        m.groupby(["ndc9", "date"], as_index=False)
        .agg(
            units_reimbursed=("units_reimbursed", "sum"),
            num_prescriptions=("num_prescriptions", "sum"),
        )
    )
    eu = g[["ndc9", "date", "units_reimbursed"]].rename(columns={"units_reimbursed": "volume"})
    eu["metric"] = "Extended Units"
    trx = g[["ndc9", "date", "num_prescriptions"]].rename(columns={"num_prescriptions": "volume"})
    trx["metric"] = "TRx"

    return pd.concat([eu, trx], ignore_index=True)[["ndc9", "date", "metric", "volume"]]


def run_ndc9_compare_2015_2025() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    NDC9 compare output (2015–2025, NOT best-practice).
    Writes:
      - <TAG>-iqvia_vs_sdud_ndc9.parquet (+ optional csv)
      - <TAG>-iqvia_vs_sdud_ndc9_per_ndc_corr.csv
      - <TAG>-iqvia_rows_with_multiple_ndc9_for_review.csv (if needed)
    """
    _ensure_dir(PROCESSED_DIR)

    iqvia_ndc9, flags_wide = load_iqvia_monthly_long_ndc9_from_2015_2025(IQVIA_2015_2025_XLSX, molecule=MOLECULE)
    sdud_ndc9 = load_sdud_monthly_long_ndc9(SDUD_MONTHLY_PARQUET)

    comp = (
        iqvia_ndc9.merge(sdud_ndc9, on=["ndc9", "date", "metric"], suffixes=("_iqvia", "_sdud"))
        .sort_values(["ndc9", "metric", "date"])
        .reset_index(drop=True)
    )

    # pooled summary by metric and needs_review
    rows = []
    for metric in sorted(comp["metric"].unique()):
        d = comp[comp["metric"] == metric]
        rows.append({
            "view": "NDC9 pooled (2015–2025)",
            "metric": metric,
            "n": len(d),
            "corr": _corr_safe(d["volume_iqvia"], d["volume_sdud"]),
        })
        for flag, label in [(False, "single NDC9 rows"), (True, "multi NDC9 rows")]:
            dd = d[d["needs_review"] == flag]
            rows.append({
                "view": f"NDC9 {label}",
                "metric": metric,
                "n": len(dd),
                "corr": _corr_safe(dd["volume_iqvia"], dd["volume_sdud"]),
            })
    summary_ndc9 = pd.DataFrame(rows)

    per_ndc = (
        comp.groupby(["ndc9", "metric"])
        .apply(lambda g: pd.Series({"corr": _corr_safe(g["volume_iqvia"], g["volume_sdud"]), "n_months": len(g)}))
        .reset_index()
    )

    out_parquet = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_ndc9.parquet"
    out_csv     = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_ndc9.csv"
    out_perndc  = PROCESSED_DIR / f"{TAG}-iqvia_vs_sdud_ndc9_per_ndc_corr.csv"
    out_flags   = PROCESSED_DIR / f"{TAG}-iqvia_rows_with_multiple_ndc9_for_review.csv"

    comp.to_parquet(out_parquet, index=False)
    if WRITE_MERGED_CSV:
        comp.to_csv(out_csv, index=False)
    per_ndc.to_csv(out_perndc, index=False)

    if not flags_wide.empty:
        flags_wide.to_csv(out_flags, index=False)

    print("\n=== NDC9 (2015–2025) ===")
    print(f"Matched rows: {len(comp):,}")
    print(f"Saved: {out_parquet}")
    if WRITE_MERGED_CSV:
        print(f"Saved: {out_csv}")
    print(f"Saved: {out_perndc}")
    if not flags_wide.empty:
        print(f"Saved: {out_flags}")

    return comp, summary_ndc9, per_ndc

# -------------------- RUN --------------------
comp_ndc11, summary_ndc11, per_ndc11 = run_best_practice_ndc11_compare()
comp_ndc9,  summary_ndc9,  per_ndc9  = run_ndc9_compare_2015_2025()

# combined summary like your old file
summary_all = pd.concat(
    [
        summary_ndc9,
        summary_ndc11.assign(view="NDC11 best-practice (2019–2025)").rename(columns={"n_rows": "n", "corr_pooled": "corr"})[
            ["view", "metric", "n", "corr"]
        ],
    ],
    ignore_index=True
)

out_summary_all = PROCESSED_DIR / f"{TAG}-summary_iqvia_vs_sdud.csv"
summary_all.to_csv(out_summary_all, index=False)
print(f"\nSaved combined summary: {out_summary_all}")

# %%
