"""
01b_build_marketscan_outcomes.py
────────────────────────────────────────────────────────────────────────────
MarketScan product-switch outcomes for each 483 inspection event, built as a
parallel DV to the FAERS ae_rise_next4q outcome in 01_build_inspection_panel.py.
02_vai_signal_model.py merges this table onto the same inspection panel
(--outcome aband_excess | aband_raw | er_rise), so text features, OAI flags
and the FEI-grouped CV are identical across FAERS and MarketScan runs.

Source: Data/20 - Market Scan/raw/manufacturer_product_quarter_{CCAE,MDCR}.csv
(labeler x product x quarter). See raw/Description.docx and
docs/Qual_Score_Explained.pdf for how switches, stops and the 90-day ER
windows are defined.

Why the manufacturer file and not fei_product_quarter_*.csv
  The plant file reports far fewer switches than the manufacturer rows that
  the same file's own ProPublica link assigns to that plant (e.g. Solco
  lisinopril, plant 1038197: 605k vs 2k; Zydus atorvastatin: 233k vs 15k).
  The definition behind the plant file's switch count is unconfirmed (open
  question for the MarketScan team), so we attribute the manufacturer rows
  ourselves.

Plant bridge (labeler row -> one finished-dose FEI), first hit wins:
  1. plant_fei from the delivered file, where ProPublica gives a single plant
  2. Data/08 - Valisure/processed/valisure_fei_ndc_anda_crosswalk.csv
  3. Data/08 - Valisure/processed/new_ndc_fei_crosswalk.csv
  4. Data/17 - NDC-FEI Linkage/processed/all_daily_med.csv, opr_type
     "manufacture" / "fdf manufacture" only (API, testing, packing excluded)
  A row is attributed only when EVERY NDC9 in its ndc9_list resolves, in that
  source, to the same single FEI. Rows split across plants are left out, never
  divided. DailyMed agrees with ProPublica on 100% of rows where both assign
  a plant (checked 2026-10-06).

Rules
  - Five acute / anti-infective APIs dropped (ampicillin, ampicillin/
    sulbactam, magnesium sulfate, metronidazole, vancomycin): their
    abandonment is course completion and their dx windows reflect the reason
    for prescribing. Calcium gluconate has no fills.
  - Adjusted stops = n_stops - n_stops_dose_change - n_coverage_ended (the
    Qual Score's correction; the delivered abandonment_rate keeps coverage
    loss in).
  - Windows: 4 quarters before (t-4..t-1) and after (t+1..t+4) the inspection
    quarter, same as the FAERS design. Quarters after 2024Q3 are dropped
    (2024Q4 is partial).
  - MDCR: both windows must end by 2022Q4. MDCR 2023 is missing and the
    Medicare file is assembled from two vendor releases either side of it, so
    a window straddling the break is not comparable.
  - Minimum switches (300 per side per cohort) is applied in 02, not here, so
    the threshold can be varied.

Measures per inspection x cohort x side (pre/post)
  n_sw       switches attributed to the plant
  ab_raw     adjusted stops / switches
  ab_excess  (adjusted stops - expected stops) / switches, where expected =
             the plant row's switches x the abandonment rate of all OTHER
             labelers of the same exact product (ingredient, route, form,
             strength) in the same quarter. Removes calendar trends and
             product mix.
  er_net     (ER visits 90d after - 90d before switch) per 1,000 switches
  dx_net     same for the failure-mode diagnosis list (secondary only: its
             pre/post change does not agree between CCAE and MDCR)

Output
  outputs/marketscan_inspection_outcomes.parquet  (one row per FEI x inspection)
  outputs/tables/marketscan_bridge_coverage.csv   (share of switches bridged, by API)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DATA = ROOT / "Data"
OUT  = HERE / "outputs"

MS_DIR   = DATA / "20 - Market Scan" / "raw"
TEXT_TS  = ROOT / "Analysis" / "Text Analysis" / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
VAL_XW   = DATA / "08 - Valisure" / "processed" / "valisure_fei_ndc_anda_crosswalk.csv"
NEW_XW   = DATA / "08 - Valisure" / "processed" / "new_ndc_fei_crosswalk.csv"
DAILYMED = DATA / "17 - NDC-FEI Linkage" / "processed" / "all_daily_med.csv"

OUT_PARQ  = OUT / "marketscan_inspection_outcomes.parquet"
OUT_COVER = OUT / "tables" / "marketscan_bridge_coverage.csv"

COHORTS = ["CCAE", "MDCR"]
ACUTE_PREFIXES = ("AMPICILLIN", "MAGNESIUM", "METRONIDAZOLE", "VANCOMYCIN")
LAST_QI = 2024 * 4 + 3          # 2024Q3; 2024Q4 is partial
MDCR_LAST_QI = 2022 * 4 + 4     # 2022Q4; MDCR break at 2023
PRODUCT_KEY = ["ingredient", "route", "form", "strength"]
SUM_COLS = ["n_switches", "adj_stops", "expected_stops",
            "er_visits_after_90d", "er_visits_before_90d",
            "dx_visits_after_90d", "dx_visits_before_90d"]


def _qidx(q: str) -> int:
    """'2019-04' (quarter start month) -> year*4 + quarter, same index as 01."""
    y, m = q.split("-")
    return int(y) * 4 + (int(m) - 1) // 3 + 1


def _ndc9(x: str) -> str:
    """'49643-0128' -> '496430128'."""
    a, b = str(x).split("-")[:2]
    return a.zfill(5) + b.zfill(4)


def _load_crosswalks() -> list[tuple[str, pd.Series]]:
    vx = pd.read_csv(VAL_XW, dtype=str).dropna(subset=["fei", "ndc9"])
    vx_map = vx.groupby(vx["ndc9"].str.replace("-", "")).fei.apply(set)
    nx = pd.read_csv(NEW_XW, dtype=str).dropna(subset=["fei", "ndc9"])
    nx_map = nx.groupby(nx["ndc9"].str.zfill(9)).fei.apply(set)
    dm = pd.read_csv(DAILYMED, dtype=str, usecols=["ndc", "opr_type", "FEI"])
    dm = dm[dm["FEI"].notna() & dm["opr_type"].isin(["manufacture", "fdf manufacture"])]
    dm_map = dm.groupby(dm["ndc"].map(_ndc9)).FEI.apply(set)
    return [("valisure_xw", vx_map), ("new_xw", nx_map), ("dailymed", dm_map)]


def _single_fei(ndc9_list: str, mp: pd.Series) -> str | None:
    feis: set = set()
    for k in str(ndc9_list).split(";"):
        if k not in mp.index:
            return None
        feis |= mp[k]
    return next(iter(feis)) if len(feis) == 1 else None


def _load_cohort(cohort: str, xws: list[tuple[str, pd.Series]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(MS_DIR / f"manufacturer_product_quarter_{cohort}.csv",
                    dtype={"plant_fei": str, "manufacturer_labeler_code": str, "ndc9_list": str})
    d = d[~d["ingredient"].str.startswith(ACUTE_PREFIXES)].copy()
    d["qi"] = d["quarter"].map(_qidx)
    d = d[d["qi"] <= LAST_QI].copy()
    d["adj_stops"] = d["n_stops"] - d["n_stops_dose_change"] - d["n_coverage_ended"]

    # Peer expectation from ALL labelers of the product, linked or not.
    g = d.groupby(PRODUCT_KEY + ["qi"])[["adj_stops", "n_switches"]].transform("sum")
    peer_sw = (g["n_switches"] - d["n_switches"]).replace(0, np.nan)
    peer_rate = (g["adj_stops"] - d["adj_stops"]) / peer_sw
    d["expected_stops"] = d["n_switches"] * peer_rate

    single_pp = d["plant_fei"].notna() & ~d["plant_fei"].fillna("").str.contains(";")
    d["fei"] = d["plant_fei"].where(single_pp)
    d["bridge_source"] = np.where(single_pp, "propublica", None)
    for name, mp in xws:
        need = d["fei"].isna()
        hit = d.loc[need, "ndc9_list"].map(lambda l: _single_fei(l, mp))
        d.loc[hit.dropna().index, "fei"] = hit.dropna()
        d.loc[hit.dropna().index, "bridge_source"] = name

    cover = (d.assign(linked=d["n_switches"] * d["fei"].notna(),
                      linked_pp=d["n_switches"] * single_pp)
              .groupby("ingredient")[["n_switches", "linked_pp", "linked"]].sum())
    cover.loc["ALL (non-acute)"] = cover.sum()
    cover["pct_propublica"] = (100 * cover["linked_pp"] / cover["n_switches"]).round(1)
    cover["pct_bridged"] = (100 * cover["linked"] / cover["n_switches"]).round(1)
    cover.insert(0, "cohort", cohort)

    # Rows without a peer (sole labeler of a product-quarter) carry no
    # expected value; they still count for the raw and ER measures.
    d = d[d["fei"].notna()]
    d["has_peer"] = d["expected_stops"].notna()
    d["expected_stops"] = d["expected_stops"].fillna(0)
    d["n_switches_peer"] = d["n_switches"] * d["has_peer"]
    d["adj_stops_peer"] = d["adj_stops"] * d["has_peer"]
    agg = d.groupby(["fei", "qi"])[SUM_COLS + ["n_switches_peer", "adj_stops_peer"]].sum()
    print(f"  {cohort}: {d['fei'].nunique()} plants bridged, "
          f"{cover.loc['ALL (non-acute)', 'pct_bridged']}% of non-acute switches "
          f"(ProPublica alone {cover.loc['ALL (non-acute)', 'pct_propublica']}%)")
    return agg, cover.reset_index()


def _window(agg: pd.DataFrame, fei: str, qis: list[int]) -> pd.Series:
    sub = agg.reindex([(fei, q) for q in qis]).dropna(how="all")
    return sub.sum() if len(sub) else pd.Series(0.0, index=agg.columns)


def main() -> None:
    print("Loading crosswalks...")
    xws = _load_crosswalks()
    aggs, covers = {}, []
    for c in COHORTS:
        aggs[c], cov = _load_cohort(c, xws)
        covers.append(cov)

    ts = pd.read_csv(TEXT_TS, usecols=["fei", "snapshot_date"], parse_dates=["snapshot_date"])
    ts["fei"] = ts["fei"].astype("int64").astype(str)
    ts["qi"] = ts["snapshot_date"].dt.year * 4 + ts["snapshot_date"].dt.quarter
    print(f"  {len(ts)} inspection events, {ts['fei'].nunique()} FEIs")

    rows = []
    for _, r in ts.iterrows():
        row = {"fei": int(r["fei"]), "insp_date": r["snapshot_date"]}
        pre_q = [r["qi"] + k for k in (-4, -3, -2, -1)]
        post_q = [r["qi"] + k for k in (1, 2, 3, 4)]
        for c in COHORTS:
            last = MDCR_LAST_QI if c == "MDCR" else LAST_QI
            ok = max(post_q) <= last
            for side, qs in (("pre", pre_q), ("post", post_q)):
                v = _window(aggs[c], r["fei"], qs)
                n = v["n_switches"] if ok else 0.0
                npeer = v["n_switches_peer"] if ok else 0.0
                p = f"ms_{c.lower()}_{side}"
                row[f"{p}_n_sw"] = n
                row[f"{p}_ab_raw"] = v["adj_stops"] / n if n > 0 else np.nan
                row[f"{p}_ab_excess"] = ((v["adj_stops_peer"] - v["expected_stops"]) / npeer
                                         if npeer > 0 else np.nan)
                row[f"{p}_er_net"] = (1000 * (v["er_visits_after_90d"] - v["er_visits_before_90d"]) / n
                                      if n > 0 else np.nan)
                row[f"{p}_dx_net"] = (1000 * (v["dx_visits_after_90d"] - v["dx_visits_before_90d"]) / n
                                      if n > 0 else np.nan)
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQ, index=False)
    pd.concat(covers).to_csv(OUT_COVER, index=False)

    for c in COHORTS:
        p = f"ms_{c.lower()}"
        k = (out[f"{p}_pre_n_sw"] >= 300) & (out[f"{p}_post_n_sw"] >= 300)
        print(f"  {c}: {k.sum()} inspections ({out.loc[k, 'fei'].nunique()} FEIs) "
              f"with >=300 switches on both sides")
    print(f"\nSaved -> {OUT_PARQ}\nSaved -> {OUT_COVER}")


if __name__ == "__main__":
    main()
