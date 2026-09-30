"""
20260930_build_propublica_anda_faers.py
─────────────────────────────────────────────────────────────────────────────
Facility-level serious adverse-event counts for the EXPANDED Valisure drug
list, built on the ProPublica NDC->FEI->ANDA linkage instead of the hand-made
March 2026 mapping (which covered only the original 14 APIs and cannot be
extended by hand to 43).

Chain
─────
  Valisure NDC (new list)
    -> NDC-9
    -> ProPublica: facility (FEI) + application number (ANDA)   [new_ndc_fei_crosswalk.csv]
    -> strip prefix: "ANDA-209959" -> appl_no 209959
    -> FAERS: join on its own appl_no field
    -> serious events per (appl_no, quarter)
    -> attribute to facility

FAERS does NOT record which facility made the product a patient took. It
records the application number. So attributing events to a facility requires a
choice, and this script implements all three rather than burying one:

  full    every facility listed for an ANDA receives that ANDA's full count.
          Treats the number as EXPOSURE, not attribution. Facility counts then
          sum to more than the true national total. This is what the earlier
          pipeline did implicitly, without the choice ever being made.
  split   divide an ANDA's count equally across the facilities listed for it.
          Preserves the national total. Assumes equal production share, which
          is certainly wrong per facility but is unbiased in aggregate.
  single  keep only ANDAs mapped to exactly ONE facility. Attribution is then
          unambiguous. Smaller sample; the honest sensitivity check.

Report all three in the paper. If they agree, the attribution choice does not
drive the result, which is the thing a reviewer actually wants to know.

Caveats recorded for the write-up
─────────────────────────────────
  * 151 NDCs map to more than one facility (see REVIEW_multi_fei_ndcs.csv).
    Some of those facilities are API suppliers rather than finished-product
    manufacturers. ProPublica's api_mfr flag is set on very few rows, so it
    cannot separate them reliably. Pending manual DailyMed review, --drop-api-mfr
    removes the ones it does flag.
  * The FAERS file is already restricted to role_cod = PS (primary suspect) and
    is_anda = True, which is the right basis for attribution.
  * primaryid is de-duplicated within (appl_no, quarter). Counting rows instead
    inflated the old pipeline by about 1%.

Usage
─────
  python 20260930_build_propublica_anda_faers.py
  python 20260930_build_propublica_anda_faers.py --scheme split
  python 20260930_build_propublica_anda_faers.py --scheme single --drop-api-mfr

Outputs
───────
  propublica_anda_fei_map.csv                  ANDA <-> FEI with sharing counts
  propublica_fei_ae_quarterly_<scheme>.csv     FEI x quarter serious AE counts
"""

from pathlib import Path
import argparse
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]

XWALK = HERE / "new_ndc_fei_crosswalk.csv"
FAERS = DATA / "15 - FDA - Adverse Event" / "processed" / \
    "faers_all_drugs_anda_linked_2015Q1_2026Q1.csv"

OUT_MAP = HERE / "propublica_anda_fei_map.csv"

SERIOUS = {
    "Death", "Hospitalization", "Life-threatening", "Disability",
    "Congenital anomaly", "Required intervention", "Other serious",
}


def load_anda_fei(text_only: bool, drop_api_mfr: bool) -> pd.DataFrame:
    """ProPublica crosswalk -> one row per (appl_no, fei), with sharing counts."""
    # ndc9 carries leading zeros; pandas will silently make it an int otherwise
    x = pd.read_csv(XWALK, low_memory=False, dtype={"ndc9": str})
    print(f"crosswalk rows: {len(x):,}")

    if drop_api_mfr:
        before = len(x)
        x = x[x["api_mfr"].astype(str).str.upper() != "TRUE"]
        print(f"  dropped API-only manufacturer rows: {before - len(x)}")

    if text_only:
        x = x[x["in_redica_text"]]
        print(f"  restricted to facilities with 483 text: {len(x):,} rows, "
              f"{x['fei'].nunique()} FEIs")

    x["appl_no"] = pd.to_numeric(
        x["anda"].astype(str).str.replace(r"^[A-Za-z]+-?", "", regex=True),
        errors="coerce",
    ).astype("Int64")
    x = x.dropna(subset=["appl_no", "fei"])
    x["fei"] = x["fei"].astype(int)
    x["appl_no"] = x["appl_no"].astype(int)

    m = x[["appl_no", "fei", "api", "labeler", "registrant", "country",
           "in_redica_text", "in_redica_history"]].drop_duplicates(["appl_no", "fei"])

    # how many facilities share each ANDA, and how many ANDAs each facility holds
    m = m.merge(m.groupby("appl_no")["fei"].nunique().rename("n_fei_per_anda"),
                on="appl_no")
    m = m.merge(m.groupby("fei")["appl_no"].nunique().rename("n_anda_per_fei"),
                on="fei")

    print(f"\nANDA to FEI map: {len(m):,} pairs, "
          f"{m['appl_no'].nunique()} ANDAs, {m['fei'].nunique()} FEIs")
    print("  facilities per ANDA: "
          f"{m.drop_duplicates('appl_no')['n_fei_per_anda'].value_counts().sort_index().to_dict()}")
    shared = m[m["n_fei_per_anda"] > 1]["appl_no"].nunique()
    print(f"  ANDAs shared by >1 facility: {shared} of {m['appl_no'].nunique()} "
          f"<- the attribution problem")
    return m


def load_faers(andas: set[int]) -> pd.DataFrame:
    """Serious FAERS events per (appl_no, quarter), de-duplicated on primaryid."""
    d = pd.read_csv(FAERS, low_memory=False,
                    usecols=["primaryid", "appl_no", "severity", "period",
                             "role_cod", "is_anda"])
    print(f"\nFAERS rows: {len(d):,}")
    print(f"  role_cod values: {d['role_cod'].unique().tolist()}  "
          f"(PS only = primary suspect, correct for attribution)")

    d = d[d["severity"].isin(SERIOUS)]
    d["appl_no"] = pd.to_numeric(d["appl_no"], errors="coerce")
    d = d.dropna(subset=["appl_no", "period"])
    d["appl_no"] = d["appl_no"].astype(int)
    print(f"  serious rows: {len(d):,}")

    d = d[d["appl_no"].isin(andas)]
    print(f"  matched to our ANDAs: {len(d):,} rows, {d['appl_no'].nunique()} ANDAs")

    # count DISTINCT reports: one report naming an ANDA twice must not count twice
    ae = (d.groupby(["appl_no", "period"], as_index=False)
            .agg(n_ae_serious=("primaryid", "nunique"),
                 n_rows=("primaryid", "count")))
    infl = ae["n_rows"].sum() - ae["n_ae_serious"].sum()
    print(f"  duplicate reports removed: {infl:,} "
          f"({100 * infl / max(ae['n_rows'].sum(), 1):.1f}%)")
    return ae.drop(columns=["n_rows"])


def attribute(ae: pd.DataFrame, m: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Push ANDA-quarter counts down to facilities under the chosen scheme."""
    if scheme == "single":
        m = m[m["n_fei_per_anda"] == 1]
        print(f"\n[single] ANDAs with exactly one facility: "
              f"{m['appl_no'].nunique()}, {m['fei'].nunique()} FEIs")

    j = ae.merge(m[["appl_no", "fei", "n_fei_per_anda"]], on="appl_no", how="inner")

    if scheme == "split":
        j["n_ae_serious"] = j["n_ae_serious"] / j["n_fei_per_anda"]

    out = (j.groupby(["fei", "period"], as_index=False)
             .agg(n_ae_serious=("n_ae_serious", "sum"),
                  n_andas=("appl_no", "nunique")))
    out["ae_year"] = out["period"].str[:4].astype(int)
    out["ae_qtr"] = out["period"].str[-1].astype(int)
    out["ae_idx"] = out["ae_year"] * 4 + out["ae_qtr"]
    out["scheme"] = scheme

    national = ae["n_ae_serious"].sum()
    attributed = out["n_ae_serious"].sum()
    print(f"[{scheme}] {len(out):,} FEI-quarter rows, {out['fei'].nunique()} FEIs")
    print(f"[{scheme}] events: national {national:,.0f}, attributed "
          f"{attributed:,.0f}  (ratio {attributed / national:.2f}"
          f"{'  <- inflation from shared ANDAs' if attributed > national * 1.01 else ''})")
    print(f"[{scheme}] median events per FEI-quarter: "
          f"{out['n_ae_serious'].median():.1f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["full", "split", "single", "all"],
                    default="all", help="attribution scheme (default: write all three)")
    ap.add_argument("--text-only", action="store_true", default=True,
                    help="restrict to facilities whose 483 text we hold (default)")
    ap.add_argument("--all-facilities", dest="text_only", action="store_false",
                    help="keep every ProPublica facility, not just text-covered ones")
    ap.add_argument("--drop-api-mfr", action="store_true",
                    help="drop rows ProPublica flags as API-only manufacturers")
    args = ap.parse_args()

    m = load_anda_fei(args.text_only, args.drop_api_mfr)
    m.to_csv(OUT_MAP, index=False)
    print(f"Saved -> {OUT_MAP.name}")

    ae = load_faers(set(m["appl_no"].unique()))

    schemes = ["full", "split", "single"] if args.scheme == "all" else [args.scheme]
    for sc in schemes:
        out = attribute(ae, m, sc)
        p = HERE / f"propublica_fei_ae_quarterly_{sc}.csv"
        out.to_csv(p, index=False)
        print(f"Saved -> {p.name}\n")

    print("Compare the three before reporting anything. If they agree, the "
          "attribution choice does not drive the result.")


if __name__ == "__main__":
    main()
