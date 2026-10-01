"""
20261001_link_ndc_fei.py
─────────────────────────────────────────────────────────────────────────────
Assign manufacturing facilities (FEIs) to every NDC on Valisure's new drug list,
using two independent linkages, and produce the combined FEI list to request
inspection history for from Redica.

An NDC having several manufacturing FEIs is expected, not an error: a product can
genuinely be made at more than one registered site. Those rows are kept as-is, one
row per (NDC, FEI) pair.

Output workbook: valisure_ndc_fei_linkage.xlsx
  Sheet "enhanced_rule_based"  our enhanced rule-based linkage (Data/27, Amir).
                      opr_type == "manufacture" only; "api manufacture" excluded,
                      since an API supplier is not the plant that made the
                      finished dose.
  Sheet "propublica"  ProPublica Rx Inspector linkage (Data/26), for comparison.
  Sheet "fei_union"   every distinct FEI either method found.

The Redica request is the WHOLE first column of fei_union, all 226 FEIs, not only
the ones we lack. Redica has sent history for some of them before, but their
holdings may have grown since, so re-requesting costs nothing and may return
richer data on facilities we already cover. The already_shared_by_redica flag is
there for group clarity about what we already hold, not to narrow the request.

NDC matching
────────────
Both linkages are joined on a padded 5-4 product NDC (5-digit labeler, 4-digit
product). Valisure writes 5-4-2 so its first two segments are used directly.
DailyMed is already 5-4. ProPublica preserves each product's native width, a mix
of 4-4, 5-3 and 5-4, so padding is required there or two thirds of the file would
silently fail to match.

Run:
  python 20261001_link_ndc_fei.py
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../08 - Valisure/processed/code
PROCESSED = HERE.parent                         # .../08 - Valisure/processed
DATA = HERE.parents[2]                          # .../Data

VALISURE = DATA / "08 - Valisure" / "raw" / "DoD Testing Overview NEW_081026_NDCs.xlsx"
# folder name has a trailing space on disk, so glob rather than hard-code it
DAILYMED = next(DATA.glob("27 - Our NDC-FEI Linkage*/all_daily_med.csv"))
PROPUBLICA = DATA / "26 - Propublica" / "raw" / "ndc_fei.csv"
REDICA_HISTORY = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
REDICA_TEXT = DATA / "99 - Outputs - Text Analysis" / "step00_redica_483_observations.csv"

OUT_XLSX = PROCESSED / "valisure_ndc_fei_linkage.xlsx"

KEEP_OPR_TYPES = ["manufacture"]


def ndc54(s: pd.Series) -> pd.Series:
    """Any product NDC -> padded 5-4. 0228-3090 -> 00228-3090."""
    parts = s.astype(str).str.strip().str.split("-")
    ok = parts.str.len() >= 2
    lab = parts.str[0].where(ok).str.zfill(5)
    prd = parts.str[1].where(ok).str.zfill(4)
    return (lab + "-" + prd).where(ok)


def load_valisure() -> pd.DataFrame:
    v = pd.read_excel(VALISURE).rename(
        columns={"API": "api", "Labeler": "labeler", "NDC": "ndc_raw"})
    v = v[["api", "labeler", "ndc_raw"]].dropna(subset=["ndc_raw"])
    v["ndc"] = ndc54(v["ndc_raw"])
    v = v.dropna(subset=["ndc"]).drop_duplicates(subset=["api", "ndc"])
    print(f"Valisure list: {len(v):,} rows, {v['ndc'].nunique():,} NDCs, "
          f"{v['api'].nunique()} APIs")
    return v


def link_enhanced_rule_based(v: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(DAILYMED, low_memory=False)
    print(f"\nEnhanced rule-based linkage: {len(d):,} rows")
    d = d[d["opr_type"].isin(KEEP_OPR_TYPES)].dropna(subset=["FEI"])
    d["fei"] = d["FEI"].astype(int)
    d["ndc"] = ndc54(d["ndc"])
    print(f"  manufacture rows with an FEI: {len(d):,}, "
          f"{d['fei'].nunique():,} facilities")

    link = (d[["ndc", "fei", "name", "link_type", "source_ndc"]]
            .drop_duplicates(["ndc", "fei"])
            .rename(columns={"name": "registrant"}))
    out = v.merge(link, on="ndc", how="inner")
    _report("enhanced_rule_based", v, out)
    return out


def link_propublica(v: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(PROPUBLICA, low_memory=False)
    p["ndc"] = ndc54(p["ndc"])
    p = p.dropna(subset=["ndc", "fei"])
    p["fei"] = p["fei"].astype(int)
    print(f"\nProPublica: {len(p):,} rows, {p['fei'].nunique():,} facilities")

    link = (p[["ndc", "fei", "registrant", "country", "anda", "nda",
               "linkage_method", "api_mfr"]]
            .drop_duplicates(["ndc", "fei"]))
    out = v.merge(link, on="ndc", how="inner")
    _report("propublica", v, out)
    return out


def _report(label: str, v: pd.DataFrame, out: pd.DataFrame) -> None:
    n_all, n_hit = v["ndc"].nunique(), out["ndc"].nunique()
    nf = out.groupby("ndc")["fei"].nunique()
    print(f"  [{label}] matched {n_hit:,} of {n_all:,} NDCs "
          f"({100 * n_hit / n_all:.1f}%), {out['fei'].nunique():,} facilities, "
          f"{len(out):,} (NDC, FEI) rows")
    print(f"  [{label}] facilities per NDC: {nf.value_counts().sort_index().to_dict()}")


def build_union(dm: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    """One row per FEI, flagged by source and by what Redica already sent us."""
    def per_fei(df, src):
        g = (df.groupby("fei")
               .agg(n_ndcs=("ndc", "nunique"),
                    n_apis=("api", "nunique"),
                    apis=("api", lambda s: ", ".join(sorted(set(s)))),
                    registrant=("registrant", "first"))
               .reset_index())
        g["source"] = src
        return g

    both = pd.concat([per_fei(dm, "enhanced_rule_based"), per_fei(pp, "propublica")])
    u = (both.groupby("fei")
             .agg(n_ndcs=("n_ndcs", "max"),
                  n_apis=("n_apis", "max"),
                  apis=("apis", lambda s: ", ".join(sorted({a.strip()
                        for x in s for a in x.split(",")}))),
                  registrant=("registrant", "first"))
             .reset_index())
    u["in_enhanced_rule_based"] = u["fei"].isin(set(dm["fei"]))
    u["in_propublica"] = u["fei"].isin(set(pp["fei"]))

    hist = set(pd.to_numeric(pd.read_csv(REDICA_HISTORY, low_memory=False)["FEI"],
                             errors="coerce").dropna().astype(int))
    text = set(pd.to_numeric(pd.read_csv(REDICA_TEXT, low_memory=False)["fei"],
                             errors="coerce").dropna().astype(int))
    # TRUE means Redica has already sent history for this FEI. Informational only:
    # we request the whole column regardless, since their holdings may have grown.
    u["already_shared_by_redica"] = u["fei"].isin(hist)
    _have_text = u["fei"].isin(text)   # reported below, not kept as a column

    u = u.sort_values(["already_shared_by_redica", "n_ndcs"], ascending=[True, False])
    u = u[["fei", "registrant", "n_ndcs", "n_apis", "apis",
           "in_enhanced_rule_based", "in_propublica", "already_shared_by_redica"]]

    print(f"\n=== FEI union: send this whole column to Redica ===")
    print(f"  distinct FEIs to request    : {len(u):,}")
    print(f"    found by both methods     : {int((u.in_enhanced_rule_based & u.in_propublica).sum()):,}")
    print(f"    enhanced rule-based only  : {int((u.in_enhanced_rule_based & ~u.in_propublica).sum()):,}")
    print(f"    ProPublica only           : {int((~u.in_enhanced_rule_based & u.in_propublica).sum()):,}")
    print(f"  for group clarity only:")
    print(f"    already shared by Redica  : {int(u.already_shared_by_redica.sum()):,}")
    print(f"    of those, with 483 text   : {int(_have_text.sum()):,}")
    print(f"    never shared              : {int((~u.already_shared_by_redica).sum()):,}")
    print(f"  Ask for all {len(u):,}. Redica may hold more on the covered ones than")
    print(f"  they sent the first time, and re-requesting costs nothing.")
    return u


def main() -> None:
    v = load_valisure()
    dm = link_enhanced_rule_based(v)
    pp = link_propublica(v)
    u = build_union(dm, pp)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        dm.to_excel(xw, sheet_name="enhanced_rule_based", index=False)
        pp.to_excel(xw, sheet_name="propublica", index=False)
        u.to_excel(xw, sheet_name="fei_union", index=False)
    print(f"\nSaved -> {OUT_XLSX.name}  "
          f"(sheets: enhanced_rule_based, propublica, fei_union)")


if __name__ == "__main__":
    main()
