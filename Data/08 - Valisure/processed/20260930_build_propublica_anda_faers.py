"""
20260930_build_propublica_anda_faers.py
─────────────────────────────────────────────────────────────────────────────
Facility-level serious adverse-event counts built on the ProPublica
NDC->FEI->ANDA linkage instead of the hand-made March 2026 mapping (which
covered only the original 14 APIs).

Scope (2026-10-19): every ANDA in ProPublica. Earlier versions kept only
drugs on Valisure's list; FAERS is its own outcome and is no longer tied to
that list. The Valisure and MarketScan outcomes apply their own scopes.

Chain
─────
  ProPublica ndc_fei.csv: NDC -> facility (FEI) + application number (ANDA)
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
          unambiguous. THIS IS THE DEFAULT.

Why single is the default
─────────────────────────
We want, per facility, the adverse events caused by THAT facility's production.
For a shared ANDA that quantity does not exist in FAERS: the reports are a
mixture across every site approved under the application, and nothing in the
data separates them. Splitting equally invents a number; full attribution
assigns one site's harm to the others. So the only construction that answers
the question is to keep applications with a single manufacturing site.

Diagnosed 2026-09-30: of 37 shared ANDAs, 27 are one firm running several
plants, 7 are contract manufacturing within a country, and 3 look like an API
supplier plus a finisher. So multi-site approval is genuine FDA practice, not a
linkage error, and no better matching would fix it.

Report full and split as sensitivity checks. If all three agree, the
attribution choice is not carrying the result.

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
  * Report versions (added 2026-10-15). FAERS issues a new primaryid each time
    a case is updated, and an update can arrive in a later quarterly file, so
    primaryid counts a followed-up case several times (Oct 6 review: 61,838
    primary IDs but 55,225 cases in the serious subset). --count-unit case
    (the default now) counts each (caseid, appl_no) once, in the first quarter
    it appears as a serious report. Using the first quarter also keeps later
    follow-ups, which were not available at that time, out of the count.
    Outputs carry a _cases suffix; --count-unit primaryid reproduces the
    earlier files unchanged.

Usage
─────
  python 20260930_build_propublica_anda_faers.py
  python 20260930_build_propublica_anda_faers.py --scheme split
  python 20260930_build_propublica_anda_faers.py --scheme single --drop-api-mfr
  python 20260930_build_propublica_anda_faers.py --count-unit primaryid   (old counts)

Outputs
───────
  propublica_anda_fei_map.csv                  ANDA <-> FEI with sharing counts
  propublica_fei_ae_quarterly_<scheme>_cases.csv   FEI x quarter serious cases (default)
  propublica_fei_ae_quarterly_<scheme>.csv     FEI x quarter serious reports (--count-unit primaryid)
  propublica_fei_anda_ae_quarterly_single_cases.csv  FEI x ANDA x quarter, single scheme
"""

from pathlib import Path
import argparse
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1]

# the full ProPublica NDC -> FEI file; FAERS scope is every ANDA in it
PROPUBLICA_ALL = DATA / "19 - ProPublica" / "raw" / "ndc_fei.csv"
REDICA_TEXT = DATA.parent / "Analysis" / "Text Analysis" / \
    "step00_redica_483_observations.csv"
REDICA_HIST = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"
FAERS = DATA / "15 - FDA - Adverse Event" / "processed" / \
    "faers_all_drugs_anda_linked_2015Q1_2026Q1.csv"

OUT_MAP = HERE / "propublica_anda_fei_map.csv"

SERIOUS = {
    "Death", "Hospitalization", "Life-threatening", "Disability",
    "Congenital anomaly", "Required intervention", "Other serious",
}


def count_ndcs(x: pd.DataFrame, label: str) -> None:
    """Print one line of the flow."""
    print(f"  {label:<55} {x['ndc'].nunique():>6} NDCs  "
          f"{x['appl_no'].nunique():>5} ANDAs  {x['fei'].nunique():>4} FEIs")


def load_anda_fei(text_only: bool, drop_api_mfr: bool) -> pd.DataFrame:
    """ProPublica NDC -> FEI file -> one row per (appl_no, fei), with sharing counts.

    Scope (2026-10-19): every ANDA in ProPublica, not only drugs on Valisure's
    list. FAERS is its own outcome, so it is not tied to the Valisure or
    MarketScan scope; those outcomes apply their own product lists.

    Attribution works at the ANDA-FEI level only. FAERS names the application
    (ANDA), not the NDC, so the one condition that matters is that the ANDA
    is made at a single facility. Facilities per ANDA are counted before we
    keep only our 483-text facilities, so a second plant outside our text
    corpus cannot hide.
    """
    # Step 0. Every ANDA in ProPublica with the facilities that make it.
    pp = pd.read_csv(PROPUBLICA_ALL, low_memory=False, dtype={"ndc": str})
    pp["appl_no"] = pd.to_numeric(
        pp["anda"].astype(str).str.replace(r"^[A-Za-z]+-?", "", regex=True),
        errors="coerce",
    )
    pp = pp.dropna(subset=["appl_no", "fei"])
    pp["fei"] = pp["fei"].astype(int)
    pp["appl_no"] = pp["appl_no"].astype(int)
    print("\nANDA flow")
    count_ndcs(pp, "0. all ANDAs in ProPublica")

    if drop_api_mfr:
        pp = pp[pp["api_mfr"].astype(str).str.upper() != "TRUE"]
        count_ndcs(pp, "   drop API-only manufacturer rows")

    # Data check only, nothing is dropped: an NDC is normally listed under
    # one application, but a few list two. Attribution never uses the NDC.
    andas_per_ndc = pp.groupby("ndc")["appl_no"].nunique()
    print(f"   (check) NDCs listed under more than one ANDA: "
          f"{(andas_per_ndc > 1).sum()}")

    # Step 1. How many facilities make each ANDA.
    sites_per_anda = pp.groupby("appl_no")["fei"].nunique()
    count_ndcs(pp[pp["appl_no"].map(sites_per_anda) == 1],
               "1. ANDA made at only one facility")

    # Step 2. Keep only facilities whose 483 text we hold.
    text_fei = set(pd.to_numeric(pd.read_csv(REDICA_TEXT, usecols=["fei"])["fei"],
                                 errors="coerce").dropna().astype(int))
    hist_fei = set(pd.to_numeric(pd.read_csv(REDICA_HIST, usecols=["FEI"])["FEI"],
                                 errors="coerce").dropna().astype(int))
    pp["in_redica_text"] = pp["fei"].isin(text_fei)
    pp["in_redica_history"] = pp["fei"].isin(hist_fei)
    if text_only:
        pp = pp[pp["in_redica_text"]]
        count_ndcs(pp, "2. facility has 483 text (all its ANDAs)")
        count_ndcs(pp[pp["appl_no"].map(sites_per_anda) == 1],
                   "3. ... and the ANDA is made only there (single scheme)")

    m = pp[["appl_no", "fei", "registrant", "country",
            "in_redica_text", "in_redica_history"]].drop_duplicates(["appl_no", "fei"])
    m["n_fei_per_anda"] = m["appl_no"].map(sites_per_anda)
    m["n_anda_per_fei"] = m["fei"].map(m.groupby("fei")["appl_no"].nunique())
    ndcs = pp.groupby(["appl_no", "fei"])["ndc"].apply(lambda s: ";".join(sorted(s.unique())))
    m = m.merge(ndcs.rename("ndc_list").reset_index(), on=["appl_no", "fei"])

    print(f"\nANDA to FEI map: {len(m):,} pairs, "
          f"{m['appl_no'].nunique()} ANDAs, {m['fei'].nunique()} FEIs")
    shared = m[m["n_fei_per_anda"] > 1]["appl_no"].nunique()
    print(f"  ANDAs also made at another facility: {shared} "
          f"<- the attribution problem")
    return m


def load_faers(andas: set[int], count_unit: str = "case") -> pd.DataFrame:
    """Serious FAERS events per (appl_no, quarter): distinct cases, each in the
    first quarter it appears (default), or distinct primaryids per quarter."""
    d = pd.read_csv(FAERS, low_memory=False,
                    usecols=["primaryid", "caseid", "appl_no", "severity", "period",
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

    if count_unit == "case":
        n_pid = d.groupby("appl_no")["primaryid"].nunique().sum()
        first = (d.sort_values("period")
                   .drop_duplicates(["caseid", "appl_no"], keep="first"))
        print(f"  report versions: {n_pid:,} distinct (primaryid, ANDA) vs "
              f"{len(first):,} distinct (case, ANDA); "
              f"{n_pid - len(first):,} later versions dropped")
        ae = (first.groupby(["appl_no", "period"], as_index=False)
                   .agg(n_ae_serious=("caseid", "nunique")))
        return ae

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
                    default="all",
                    help="attribution scheme; 'all' writes all three and is the "
                         "default so the sensitivity checks stay available. "
                         "single is the one to report.")
    ap.add_argument("--text-only", action="store_true", default=True,
                    help="restrict to facilities whose 483 text we hold (default)")
    ap.add_argument("--all-facilities", dest="text_only", action="store_false",
                    help="keep every ProPublica facility, not just text-covered ones")
    ap.add_argument("--drop-api-mfr", action="store_true",
                    help="drop rows ProPublica flags as API-only manufacturers")
    ap.add_argument("--count-unit", choices=["case", "primaryid"], default="case",
                    help="case = each FAERS case once, first serious quarter (default); "
                         "primaryid = every report version (the pre-2026-10-15 files)")
    args = ap.parse_args()

    m = load_anda_fei(args.text_only, args.drop_api_mfr)
    m.to_csv(OUT_MAP, index=False)
    print(f"Saved -> {OUT_MAP.name}")

    ae = load_faers(set(m["appl_no"].unique()), args.count_unit)
    single = m[m["n_fei_per_anda"] == 1]
    in_faers = single[single["appl_no"].isin(set(ae["appl_no"]))]
    n_ndc = len(set(";".join(in_faers["ndc_list"]).split(";")))
    print(f"\n  4. ANDA has a serious FAERS case (single scheme): {n_ndc} NDCs  "
          f"{in_faers['appl_no'].nunique()} ANDAs  {in_faers['fei'].nunique()} FEIs")
    tag = "_cases" if args.count_unit == "case" else ""

    # Per-ANDA counts for the single scheme (2026-10-19), so an outcome can
    # look at each product separately instead of the factory total.
    per_anda = ae.merge(single[["appl_no", "fei"]], on="appl_no", how="inner")
    per_anda["ae_idx"] = (per_anda["period"].str[:4].astype(int) * 4
                          + per_anda["period"].str[-1].astype(int))
    p = HERE / f"propublica_fei_anda_ae_quarterly_single{tag}.csv"
    per_anda.to_csv(p, index=False)
    print(f"Saved -> {p.name}  ({len(per_anda):,} FEI-ANDA-quarter rows)\n")

    schemes = ["full", "split", "single"] if args.scheme == "all" else [args.scheme]
    for sc in schemes:
        out = attribute(ae, m, sc)
        p = HERE / f"propublica_fei_ae_quarterly_{sc}{tag}.csv"
        out.to_csv(p, index=False)
        print(f"Saved -> {p.name}\n")

    print("Report 'single': every event traces to exactly one facility. Use "
          "'full' and 'split' as sensitivity checks only.")


if __name__ == "__main__":
    main()
