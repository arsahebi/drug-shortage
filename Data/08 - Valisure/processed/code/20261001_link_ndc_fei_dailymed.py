"""
20261001_link_ndc_fei_dailymed.py
─────────────────────────────────────────────────────────────────────────────
Assign a manufacturing facility (FEI) to each NDC on Valisure's new drug list,
using Amir's DailyMed-based NDC-FEI linkage instead of ProPublica.

Only `opr_type == "manufacture"` rows are used. "api manufacture" is EXCLUDED:
an API supplier is not the plant that made the finished product, and mixing the
two is what muddied the earlier ProPublica linkage. Change KEEP_OPR_TYPES below
if you want them included.

Both sides are already in 5-4 format (Valisure writes 5-4-2, so its first two
segments are the labeler and product codes; the DailyMed file's `ndc` column is
5-4 throughout), so the join is a direct string match with no padding needed.

Run:
  python 20261001_link_ndc_fei_dailymed.py

Outputs (written to the parent processed/ folder):
  valisure_ndc_fei_dailymed.csv        one row per (ndc, fei) with API + labeler
  valisure_ndc_fei_dailymed_unmatched.csv   NDCs with no manufacturing facility
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../08 - Valisure/processed/code
PROCESSED = HERE.parent                         # .../08 - Valisure/processed
DATA = HERE.parents[2]                          # .../Data

VALISURE = DATA / "08 - Valisure" / "raw" / "DoD Testing Overview NEW_081026_NDCs.xlsx"
# folder name has a trailing space on disk, so glob rather than hard-code it
DAILYMED = next(DATA.glob("27 - Our NDC-FEI Linkage*/all_daily_med.csv"))

OUT = PROCESSED / "valisure_ndc_fei_dailymed.csv"
OUT_UNMATCHED = PROCESSED / "valisure_ndc_fei_dailymed_unmatched.csv"

KEEP_OPR_TYPES = ["manufacture"]


def ndc54(s: pd.Series) -> pd.Series:
    """Keep the first two NDC segments: 60687-0662-91 -> 60687-0662."""
    parts = s.astype(str).str.strip().str.split("-")
    return parts.str[0] + "-" + parts.str[1]


def main() -> None:
    # ── Valisure's new NDC list ──────────────────────────────────────────────
    v = pd.read_excel(VALISURE).rename(
        columns={"API": "api", "Labeler": "labeler", "NDC": "ndc_raw"})
    v = v[["api", "labeler", "ndc_raw"]].dropna(subset=["ndc_raw"])
    v["ndc"] = ndc54(v["ndc_raw"])
    v = v.drop_duplicates(subset=["api", "ndc"])
    print(f"Valisure list: {len(v):,} rows, {v['ndc'].nunique():,} NDCs, "
          f"{v['api'].nunique()} APIs")

    # ── DailyMed linkage, manufacturing sites only ───────────────────────────
    d = pd.read_csv(DAILYMED, low_memory=False)
    print(f"\nDailyMed linkage: {len(d):,} rows")
    d = d[d["opr_type"].isin(KEEP_OPR_TYPES)]
    print(f"  opr_type in {KEEP_OPR_TYPES}: {len(d):,} rows")
    d = d.dropna(subset=["FEI"])
    d["fei"] = d["FEI"].astype(int)
    print(f"  with an FEI: {len(d):,} rows, {d['fei'].nunique():,} facilities")

    link = d[["ndc", "fei", "name", "link_type"]].drop_duplicates(["ndc", "fei"])

    # ── join ────────────────────────────────────────────────────────────────
    merged = v.merge(link, on="ndc", how="left")
    matched = merged[merged["fei"].notna()].copy()
    matched["fei"] = matched["fei"].astype(int)

    n_all = v["ndc"].nunique()
    n_hit = matched["ndc"].nunique()
    print(f"\nMatched: {n_hit:,} of {n_all:,} NDCs ({100 * n_hit / n_all:.1f}%)")
    print(f"  facilities found : {matched['fei'].nunique():,}")
    print(f"  (ndc, fei) pairs : {len(matched):,}")
    print(f"  NDCs with >1 facility: "
          f"{int((matched.groupby('ndc')['fei'].nunique() > 1).sum()):,}")

    print("\nPer-API match rate:")
    per_api = (v.groupby("api")["ndc"].nunique().rename("ndcs")
               .to_frame()
               .join(matched.groupby("api")["ndc"].nunique().rename("matched"))
               .fillna({"matched": 0}))
    per_api["matched"] = per_api["matched"].astype(int)
    per_api["pct"] = (100 * per_api["matched"] / per_api["ndcs"]).round(1)
    print(per_api.sort_values("pct", ascending=False).to_string())

    matched.to_csv(OUT, index=False)
    merged[merged["fei"].isna()][["api", "labeler", "ndc_raw", "ndc"]] \
        .drop_duplicates().to_csv(OUT_UNMATCHED, index=False)
    print(f"\nSaved -> {OUT.name}")
    print(f"Saved -> {OUT_UNMATCHED.name}")


if __name__ == "__main__":
    main()
