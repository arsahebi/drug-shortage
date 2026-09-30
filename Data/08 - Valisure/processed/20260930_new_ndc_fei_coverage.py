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

Reading the two coverage numbers
────────────────────────────────
Per API we report BOTH:
  pct_ndc_with_history  share of NDCs whose facility appears in Redica's
                        inspection history (dates, classification, 483 counts)
  pct_ndc_with_text     share of NDCs whose facility has actual 483
                        observation text, which is what the LLM features need

The gap between them is NOT missing data. A facility can appear in the history
file with no 483 text because its inspections closed without a 483 being
issued. That is a clean regulatory record, and it is informative. Judge an API
on both numbers: low history coverage means we genuinely lack the facility,
whereas high history but low text means the facility exists and was inspected
without findings.

Caveat on the request scope: Redica's pull was built from the March 2026
NDC-FEI mapping, i.e. the original 14 APIs. Anything outside that request is
absent by construction, not because the facility has a clean record. Redica is
expected to extend the pull against the new NDC list.

Outputs (written next to this script):
  new_ndc_fei_crosswalk.csv          one row per (ndc9, fei) with API + labeler
  new_ndc_fei_coverage_by_api.csv    per-API coverage summary
  new_ndc_unmatched.csv              NDCs with no ProPublica facility

NOTE for consumers: ndc9 carries leading zeros. Read it with
pd.read_csv(..., dtype={"ndc9": str}) or pandas will parse it as an integer and
silently drop them, which makes every join against it fail.
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]

VALISURE_NEW = DATA / "08 - Valisure" / "raw" / "DoD Testing Overview NEW_081026_NDCs.xlsx"
PROPUBLICA   = DATA / "26 - Propublica" / "raw" / "ndc_fei.csv"
REDICA       = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
# The BINDING constraint. Redica gave us two different things: inspection history
# for 127 FEIs (dates, classification, 483 counts) and actual 483 observation TEXT
# for only 98 of them. Text is what the LLM features are built from, so coverage
# must be measured against this file, not the history file.
REDICA_TEXT  = DATA / "99 - Outputs - Text Analysis" / "step00_redica_483_observations.csv"
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
    hist_fei = set(pd.to_numeric(r["FEI"], errors="coerce").dropna().astype(int))
    t = pd.read_csv(REDICA_TEXT, low_memory=False)
    r_fei = set(pd.to_numeric(t["fei"], errors="coerce").dropna().astype(int))
    print("\n" + "=" * 74)
    print("STEP 2 — which of those FEIs do we have Redica inspection history for?")
    print("=" * 74)
    print(f"  Redica inspection HISTORY : {len(hist_fei):,} FEIs")
    print(f"  Redica 483 TEXT           : {len(r_fei):,} FEIs  "
          f"<- {len(hist_fei - r_fei)} have history but no text")
    print(f"  new-list FEIs             : {len(feis_new):,}")
    print(f"    with Redica history     : {len(feis_new & hist_fei):,} "
          f"({100*len(feis_new & hist_fei)/len(feis_new):.1f}%)")
    print(f"    with 483 TEXT           : {len(feis_new & r_fei):,} "
          f"({100*len(feis_new & r_fei)/len(feis_new):.1f}%)  <- the binding number")
    print(f"    NO text                 : {len(feis_new - r_fei):,}  "
          f"<- no text features possible until Redica extends the pull")
    print(f"  text FEIs not reached by the new list: {len(r_fei - feis_new):,}")

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
        f_hist = f_all & hist_fei
        f_text = f_all & r_fei
        n_hist_ndc = gm[gm["fei"].isin(hist_fei)]["ndc9"].nunique()
        n_text_ndc = gm[gm["fei"].isin(r_fei)]["ndc9"].nunique()
        rows.append({
            "api": api,
            "n_ndc": n_all,
            "n_ndc_linked": n_lk,
            "pct_ndc_linked": round(100 * n_lk / n_all, 1) if n_all else 0.0,
            "n_fei": len(f_all),
            "n_fei_with_history": len(f_hist),
            "n_fei_with_text": len(f_text),
            "n_ndc_with_history": n_hist_ndc,
            "pct_ndc_with_history": round(100 * n_hist_ndc / n_all, 1) if n_all else 0.0,
            "n_ndc_with_text": n_text_ndc,
            "pct_ndc_with_text": round(100 * n_text_ndc / n_all, 1) if n_all else 0.0,
            "history_minus_text_pp": round(100 * (n_hist_ndc - n_text_ndc) / n_all, 1) if n_all else 0.0,
        })
    by_api = pd.DataFrame(rows).sort_values(
        ["pct_ndc_with_history", "pct_ndc_with_text"], ascending=False)

    print("\n" + "=" * 74)
    print("STEP 4 — per-API completeness: history coverage AND text coverage")
    print("=" * 74)
    show = ["api", "n_ndc", "pct_ndc_linked", "n_fei",
            "pct_ndc_with_history", "pct_ndc_with_text", "history_minus_text_pp"]
    print(by_api[show].to_string(index=False))
    print("\n  history_minus_text_pp = facility is in Redica but issued no 483.")
    print("  A clean inspection record, not a data gap.")

    for lbl, col in [("HISTORY", "pct_ndc_with_history"), ("TEXT", "pct_ndc_with_text")]:
        hi = by_api[by_api[col] >= 80]
        mid = by_api[(by_api[col] >= 20) & (by_api[col] < 80)]
        lo = by_api[by_api[col] < 20]
        print(f"\n  by {lbl} coverage:  >=80%: {len(hi)}   20-80%: {len(mid)}   <20%: {len(lo)}")
        print(f"    >=80%: {sorted(hi.api.tolist())}")
        print(f"    <20% : {sorted(lo.api.tolist())}")

    # ── save ────────────────────────────────────────────────────────────────
    out = matched.drop_duplicates(["api", "ndc9", "fei"]).copy()
    out["in_redica_text"] = out["fei"].isin(r_fei)
    out["in_redica_history"] = out["fei"].isin(hist_fei)
    if fda_fei:
        out["in_fda_dashboard"] = out["fei"].isin(fda_fei)
    out.to_csv(OUT_XWALK, index=False)
    by_api.to_csv(OUT_BY_API, index=False)
    x[x["fei"].isna()][["api", "labeler", "ndc_raw", "ndc9"]] \
        .drop_duplicates().to_csv(OUT_UNMATCH, index=False)
    print(f"\nSaved -> {OUT_XWALK.name}, {OUT_BY_API.name}, {OUT_UNMATCH.name}")


if __name__ == "__main__":
    main()
