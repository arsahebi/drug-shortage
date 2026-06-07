# %%
# -*- coding: utf-8 -*-
"""
2026-02-24-compare_nadac_panel_versions.py

Compares the Dec-2025 and Feb-2026 cleaned IQVIA/SDUD/NADAC panel files to
verify whether the updated NADAC averaging method (direct annual mean across
all weekly records, instead of mean-of-monthly-means) produces different
nadac_price values at the monthly panel level.

Expected result: monthly nadac_price is IDENTICAL between versions because
the pipeline change only affects how downstream QA scripts compute the ANNUAL
price. The monthly panel itself still stores the mean of weekly prices within
each calendar month in both versions.

Usage:
    python 2026-02-24-compare_nadac_panel_versions.py
    # or override paths:
    OLD_PATH = "..." ; NEW_PATH = "..." at the top of the file.

Outputs printed to console + a CSV diff report saved alongside this script.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
# Script lives in processed/code/ — CSVs are one level up in processed/
_HERE = Path(__file__).resolve().parent
_PROCESSED = _HERE.parent

OLD_PATH = _PROCESSED / "2025-12-18-iqvia_with_sdud_nadac.cleaned.csv"
NEW_PATH = _PROCESSED / "2026-02-24-iqvia_with_sdud_nadac.cleaned.csv"

# Column to focus the NADAC comparison on
PRICE_COL = "nadac_price"
KEY_COLS   = ["ndc11", "date"]

# ── Load ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("NADAC Panel Version Comparison")
print(f"  OLD : {OLD_PATH}")
print(f"  NEW : {NEW_PATH}")
print("=" * 70)

old = pd.read_csv(OLD_PATH, dtype={"ndc11": str}, low_memory=False)
new = pd.read_csv(NEW_PATH, dtype={"ndc11": str}, low_memory=False)

print(f"\nOLD shape : {old.shape}")
print(f"NEW shape : {new.shape}")

# ── Schema diff ────────────────────────────────────────────────────────────
old_cols = set(old.columns)
new_cols = set(new.columns)
added   = new_cols - old_cols
removed = old_cols - new_cols
print(f"\nColumns added   : {sorted(added)  or 'none'}")
print(f"Columns removed : {sorted(removed) or 'none'}")

# ── Row-count diff ─────────────────────────────────────────────────────────
print(f"\nRow count OLD : {len(old):,}")
print(f"Row count NEW : {len(new):,}")
print(f"Difference    : {len(new) - len(old):+,}")

# ── Merge on keys ──────────────────────────────────────────────────────────
merged = old[KEY_COLS + [PRICE_COL]].merge(
    new[KEY_COLS + [PRICE_COL]],
    on=KEY_COLS,
    how="outer",
    suffixes=("_old", "_new"),
    indicator=True,
)

only_old = merged[merged["_merge"] == "left_only"]
only_new = merged[merged["_merge"] == "right_only"]
both     = merged[merged["_merge"] == "both"].copy()

print(f"\nRows only in OLD : {len(only_old):,}")
print(f"Rows only in NEW : {len(only_new):,}")
print(f"Rows in both     : {len(both):,}")

# ── NADAC price comparison on matched rows ─────────────────────────────────
both["abs_diff"] = (both[f"{PRICE_COL}_new"] - both[f"{PRICE_COL}_old"]).abs()
both["rel_diff"] = both["abs_diff"] / both[f"{PRICE_COL}_old"].replace(0, np.nan)

n_changed = (both["abs_diff"] > 1e-10).sum()
print(f"\n── {PRICE_COL} comparison (matched rows) ──")
print(f"  Rows with any difference (abs > 1e-10) : {n_changed:,}")

if n_changed == 0:
    print(
        "\n  ✓ nadac_price is IDENTICAL in both files at the monthly level.\n"
        "  This is expected: the pipeline change (direct annual averaging)\n"
        "  does NOT affect the monthly panel output. The difference will only\n"
        "  appear when QA/figure scripts compute the annual price downstream\n"
        "  by grouping on (ndc11, year) using individual weekly records.\n"
    )
else:
    print(f"\n  ✗ {n_changed:,} rows differ — summary of absolute differences:")
    print(both.loc[both["abs_diff"] > 1e-10, "abs_diff"].describe().to_string())
    print(f"\n  Relative difference summary:")
    print(both.loc[both["abs_diff"] > 1e-10, "rel_diff"].describe().to_string())

# ── Annual-level simulation ────────────────────────────────────────────────
# Even though monthly nadac_price is identical, simulate what would happen
# if someone averaged the monthly values to get annual (OLD method) vs if
# they had weekly counts (n_obs) to weight (NEW method intent).
print("\n── Annual nadac_price simulation (OLD method: mean of monthly means) ──")
for df, label in [(old, "OLD"), (new, "NEW")]:
    if PRICE_COL not in df.columns:
        continue
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["date"]).dt.year
    annual = (
        df2.dropna(subset=[PRICE_COL])
        .groupby(["ndc11", "year"], as_index=False)[PRICE_COL]
        .mean()
        .rename(columns={PRICE_COL: f"annual_{PRICE_COL}"})
    )
    print(f"  {label}: {len(annual):,} ndc11-year annual price records "
          f"| mean={annual[f'annual_{PRICE_COL}'].mean():.5f} "
          f"| std={annual[f'annual_{PRICE_COL}'].std():.5f}")

# ── n_obs check ────────────────────────────────────────────────────────────
if "n_obs" in old.columns and "n_obs" in new.columns:
    obs_merged = old[KEY_COLS + ["n_obs"]].merge(
        new[KEY_COLS + ["n_obs"]],
        on=KEY_COLS,
        how="inner",
        suffixes=("_old", "_new"),
    )
    obs_merged["n_obs_diff"] = obs_merged["n_obs_new"] - obs_merged["n_obs_old"]
    n_obs_changed = (obs_merged["n_obs_diff"].abs() > 0).sum()
    print(f"\n── n_obs (weekly record count per month) comparison ──")
    print(f"  Rows with changed n_obs : {n_obs_changed:,}")
    if n_obs_changed > 0:
        print(obs_merged[obs_merged["n_obs_diff"] != 0][["ndc11", "date", "n_obs_old", "n_obs_new"]].head(20).to_string(index=False))

# ── Other numeric columns ──────────────────────────────────────────────────
print("\n── All numeric columns: max absolute difference ──")
shared_num_cols = [
    c for c in old.columns
    if c in new.columns and c not in KEY_COLS
    and pd.api.types.is_numeric_dtype(old[c])
]
for col in shared_num_cols:
    m = old[KEY_COLS + [col]].merge(new[KEY_COLS + [col]], on=KEY_COLS, suffixes=("_o", "_n"))
    diff = (m[f"{col}_n"] - m[f"{col}_o"]).abs()
    print(f"  {col:<40s}  max_diff={diff.max():.6g}  n_diff={(diff > 1e-10).sum():,}")

# ── Save diff report ────────────────────────────────────────────────────────
diff_rows = both[both["abs_diff"] > 1e-10].copy()
report_path = _HERE / (Path(__file__).stem + "_diff_report.csv")
if len(diff_rows) > 0:
    diff_rows.to_csv(report_path, index=False)
    print(f"\nDiff report saved to: {report_path}")
else:
    print(f"\nNo differences found — no diff report written.")

print("\nDone.")
# %%
