"""
20260930_new_ndc_fei_coverage.py
─────────────────────────────────────────────────────────────────────────────
Valisure sent an expanded NDC list (DoD Testing Overview NEW_081026_NDCs.xlsx):
43 APIs, ~1,535 distinct NDCs, against the 14 APIs the current pipeline covers.

This script answers the coverage question before any modelling work starts:

  1. How many of the new NDCs can be linked to a manufacturing facility (FEI)
     using ProPublica's Rx Inspector NDC->FEI release?
  2. Of the FEIs that linkage produces, how many do we already hold Redica
     inspection history for? Redica is the ONLY source of 483 observation text,
     so any FEI outside it cannot contribute text features no matter what else
     we have.
  3. How many are covered by FDA's inspection dashboard, which we can use for
     non-text inspection features even where Redica has nothing?
  4. Per API: which are complete, which are partial, which are unusable.

NDC normalisation
─────────────────
Both sources carry PRODUCT-level NDCs (labeler + product, no package segment),
but in different widths. Valisure writes 5-4-2 throughout. ProPublica carries a
mix of 4-4, 5-3 and 5-4, because it preserves each product's native format.
Comparing the raw strings would silently drop the 4-4 and 5-3 rows, so both
sides are normalised to NDC-9: labeler zero-padded to 5 digits, product
zero-padded to 4, concatenated. This is the standard 11-digit NDC convention
(4-4-2 pads the labeler, 5-3-2 pads the product) applied to the first two
segments only.

Run:
  python 20260930_new_ndc_fei_coverage.py

Outputs (written next to this script):
  new_ndc_fei_crosswalk.csv          one row per (ndc9, fei) with API + labeler
  new_ndc_fei_coverage_by_api.csv    per-API coverage summary
  new_ndc_unmatched.csv              NDCs with no ProPublica facility
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]

VALISURE_NEW = DATA / "08 - Valisure" / "raw" / "DoD Testing Overview NEW_081026_NDCs.xlsx"
PROPUBLICA   = DATA / "26 - Propublica" / "raw" / "ndc_fei.csv"
REDICA       = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
FDA_INSP     = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"

OUT_XWALK    = HERE / "new_ndc_fei_crosswalk.csv"
OUT_BY_API   = HERE / "new_ndc_fei_coverage_by_api.csv"
OUT_UNMATCH  = HERE / "new_ndc_unmatched.csv"


def to_ndc9(s: pd.Series) -> pd.Series:
    """Product NDC (any segment width) -> 9-digit labeler(5) + product(4)."""
    parts = s.astype(str).str.strip().str.split("-")
    ok = parts.str.len() >= 2
    lab = parts.str[0].where(ok).str.zfill(5)
    prd = parts.str[1].where(ok).str.zfill(4)
    return (lab + prd).where(ok)


def main() -> None:
    # ── new Valisure NDC list ────────────────────────────────────────────────
    v = pd.read_excel(VALISURE_NEW)
    v = v.rename(columns={"API": "api", "Labeler": "labeler", "NDC": "ndc_raw"})
    v = v[["api", "labeler", "ndc_raw"]].dropna(subset=["ndc_raw"])
    v["ndc9"] = to_ndc9(v["ndc_raw"])
    bad = v["ndc9"].isna().sum()
    v = v.dropna(subset=["ndc9"]).drop_duplicates(subset=["api", "ndc9"])
    print("=" * 74)
    print("NEW VALISURE NDC LIST")
    print("=" * 74)
    print(f"  rows usable            : {len(v):,}  (dropped {bad} malformed NDCs)")
    print(f"  distinct APIs          : {v['api'].nunique()}")
    print(f"  distinct NDC-9         : {v['ndc9'].nunique():,}")

    # ── ProPublica NDC -> FEI ────────────────────────────────────────────────
    p = pd.read_csv(PROPUBLICA, low_memory=False)
    p["ndc9"] = to_ndc9(p["ndc"])
    p = p.dropna(subset=["ndc9"])
    print(f"\n  ProPublica rows        : {len(p):,}  "
          f"({p['ndc9'].nunique():,} NDC-9, {p['fei'].nunique():,} FEIs)")

    x = v.merge(
        p[["ndc9", "fei", "registrant", "country", "anda", "nda",
           "linkage_method", "api_mfr"]],
        on="ndc9", how="left",
    )
    matched = x[x["fei"].notna()].copy()
    matched["fei"] = matched["fei"].astype(int)

    n_ndc = v["ndc9"].nunique()
    n_ndc_matched = matched["ndc9"].nunique()
    print("\n" + "=" * 74)
    print("STEP 1 — NDC to FEI via ProPublica")
    print("=" * 74)
    print(f"  NDCs linked to >=1 FEI : {n_ndc_matched:,} / {n_ndc:,} "
          f"({100*n_ndc_matched/n_ndc:.1f}%)")
    print(f"  NDCs with NO facility  : {n_ndc - n_ndc_matched:,}")
    print(f"  distinct FEIs found    : {matched['fei'].nunique():,}")
    print(f"  (ndc9, fei) pairs      : {len(matched.drop_duplicates(['ndc9','fei'])):,}")
    print(f"\n  linkage method         : "
          f"{matched['linkage_method'].value_counts().to_dict()}")
    print(f"  API-only manufacturers : "
          f"{(matched['api_mfr'].astype(str).str.upper() == 'TRUE').sum():,} rows")
    print(f"  facility countries     : "
          f"{matched['country'].value_counts().head(6).to_dict()}")

    feis_new = set(matched["fei"].unique())

    # ── Redica coverage (the 483-text constraint) ────────────────────────────
    r = pd.read_csv(REDICA, low_memory=False)
    r_fei = set(pd.to_numeric(r["FEI"], errors="coerce").dropna().astype(int))
    print("\n" + "=" * 74)
    print("STEP 2 — which of those FEIs do we have Redica inspection history for?")
    print("=" * 74)
    print(f"  FEIs in Redica file    : {len(r_fei):,}")
    print(f"  new-list FEIs          : {len(feis_new):,}")
    print(f"    already in Redica    : {len(feis_new & r_fei):,} "
          f"({100*len(feis_new & r_fei)/len(feis_new):.1f}%)")
    print(f"    NOT in Redica        : {len(feis_new - r_fei):,}  "
          f"<- no 483 text possible until Redica extends the pull")
    print(f"  Redica FEIs not reached by the new list: {len(r_fei - feis_new):,}")

    # ── FDA dashboard coverage (non-text inspection features) ───────────────
    fda_fei: set[int] = set()
    try:
        f = pd.read_excel(FDA_INSP)
        fcol = next(c for c in f.columns if c.strip().upper().startswith("FEI"))
        fda_fei = set(pd.to_numeric(f[fcol], errors="coerce").dropna().astype(int))
        print("\n" + "=" * 74)
        print("STEP 3 — FDA inspection dashboard coverage (non-text features)")
        print("=" * 74)
        print(f"  FEIs in FDA file       : {len(fda_fei):,}")
        print(f"    new-list FEIs found  : {len(feis_new & fda_fei):,} "
              f"({100*len(feis_new & fda_fei)/len(feis_new):.1f}%)")
        print(f"  new-list FEIs with FDA but NOT Redica: "
              f"{len(feis_new & fda_fei - r_fei):,}  "
              f"<- usable for inspection features, not for text")
        print(f"  new-list FEIs in NEITHER source      : "
              f"{len(feis_new - r_fei - fda_fei):,}")
    except Exception as exc:
        print(f"\n  [FDA dashboard not read: {type(exc).__name__}: {exc}]")

    # ── per-API completeness ────────────────────────────────────────────────
    rows = []
    for api, g in v.groupby("api"):
        gm = matched[matched["api"] == api]
        n_all = g["ndc9"].nunique()
        n_lk = gm["ndc9"].nunique()
        f_all = set(gm["fei"].unique())
        f_red = f_all & r_fei
        n_red_ndc = gm[gm["fei"].isin(r_fei)]["ndc9"].nunique()
        rows.append({
            "api": api,
            "n_ndc": n_all,
            "n_ndc_linked": n_lk,
            "pct_ndc_linked": round(100 * n_lk / n_all, 1) if n_all else 0.0,
            "n_fei": len(f_all),
            "n_fei_in_redica": len(f_red),
            "pct_fei_in_redica": round(100 * len(f_red) / len(f_all), 1) if f_all else 0.0,
            "n_ndc_with_redica_fei": n_red_ndc,
            "pct_ndc_with_text": round(100 * n_red_ndc / n_all, 1) if n_all else 0.0,
        })
    by_api = pd.DataFrame(rows).sort_values("pct_ndc_with_text", ascending=False)

    print("\n" + "=" * 74)
    print("STEP 4 — per-API completeness  (pct_ndc_with_text is the binding one)")
    print("=" * 74)
    print(by_api.to_string(index=False))

    complete = by_api[by_api.pct_ndc_with_text >= 80]
    partial = by_api[(by_api.pct_ndc_with_text >= 20) & (by_api.pct_ndc_with_text < 80)]
    none_ = by_api[by_api.pct_ndc_with_text < 20]
    print(f"\n  APIs >=80% NDCs text-capable : {len(complete)}  "
          f"{sorted(complete.api.tolist())}")
    print(f"  APIs 20-80%                  : {len(partial)}  "
          f"{sorted(partial.api.tolist())}")
    print(f"  APIs <20%                    : {len(none_)}  "
          f"{sorted(none_.api.tolist())}")

    # ── save ────────────────────────────────────────────────────────────────
    out = matched.drop_duplicates(["api", "ndc9", "fei"]).copy()
    out["in_redica"] = out["fei"].isin(r_fei)
    if fda_fei:
        out["in_fda_dashboard"] = out["fei"].isin(fda_fei)
    out.to_csv(OUT_XWALK, index=False)
    by_api.to_csv(OUT_BY_API, index=False)
    x[x["fei"].isna()][["api", "labeler", "ndc_raw", "ndc9"]] \
        .drop_duplicates().to_csv(OUT_UNMATCH, index=False)
    print(f"\nSaved -> {OUT_XWALK.name}, {OUT_BY_API.name}, {OUT_UNMATCH.name}")


if __name__ == "__main__":
    main()
