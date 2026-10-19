"""
13_time_to_quality_report.py
─────────────────────────────────────────────────────────────────────────────
Li et al. (2026) case study 3, applied to inspections (2026-10-19).

They linked features of a regulatory document to the time until a device's
first adverse event report (Cox model, exploratory). Here:

  unit      one product (single-site ANDA) at an inspected factory, per inspection
  start     the inspection quarter
  event     the first later quarter with a quality-defect FAERS report for
            that ANDA (faers_product_quality_reactions.csv)
  censored  at the factory's next inspection, 8 quarters, or 2026Q1,
            whichever comes first
  products  ANDAs with at least one serious FAERS case of any kind in the
            year before the inspection (so the product is on the market and
            reported on)
  model     Cox, one text signal at a time (per SD), plus two baseline
            controls: the ANDA had a quality report in the year before (0/1)
            and log(1 + its serious cases in the year before). Standard
            errors clustered by factory. Holm over the 17 signals. FDA's OAI
            flag is run the same way as the comparator.

Outputs: outputs/tables/time_to_quality_report.csv
"""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
TEXT = HERE.parent
VAL = HERE.parents[2] / "Data" / "08 - Valisure" / "processed"
STEP02 = TEXT / "step02_483_fei_text_features_timeseries_redica_claudesonnet5_v2.csv"
INSP = TEXT / "step00_redica_inspections.csv"
SERIOUS = VAL / "propublica_fei_anda_ae_quarterly_single_cases.csv"
QUALITY = VAL / "propublica_fei_anda_ae_quarterly_single_cases_quality.csv"
OUT = HERE / "outputs" / "tables" / "time_to_quality_report.csv"

FIRST_IDX, LAST_IDX = 2015 * 4 + 1, 2026 * 4 + 1
MAX_FOLLOW = 8

spec = importlib.util.spec_from_file_location("m02", HERE / "02_vai_signal_model.py")
m02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m02)
FEATURES = list(m02.TEXT_FEATURES)


def build() -> pd.DataFrame:
    """One row per (inspection, ANDA) with follow-up time and event."""
    insp = pd.read_csv(INSP, parse_dates=["insp_date"]).query("keep").copy()
    insp["idx"] = insp["insp_date"].dt.year * 4 + insp["insp_date"].dt.quarter
    insp = insp.sort_values(["fei", "idx"])
    insp["next_idx"] = insp.groupby("fei")["idx"].shift(-1)
    insp = insp[(insp["idx"] - 4 >= FIRST_IDX) & (insp["idx"] < LAST_IDX)]

    ser = pd.read_csv(SERIOUS)
    qual = pd.read_csv(QUALITY)
    rows = []
    for _, r in insp.iterrows():
        i = r["idx"]
        end = min(i + MAX_FOLLOW, LAST_IDX,
                  r["next_idx"] if pd.notna(r["next_idx"]) else LAST_IDX)
        s = ser[(ser["fei"] == r["fei"]) & ser["ae_idx"].between(i - 4, i - 1)]
        pre_serious = s.groupby("appl_no")["n_ae_serious"].sum()
        q = qual[qual["fei"] == r["fei"]]
        for a, n_pre in pre_serious.items():
            qa = q[q["appl_no"] == a]
            pre_q = int(qa["ae_idx"].between(i - 4, i - 1).any())
            after = qa.loc[(qa["ae_idx"] > i) & (qa["ae_idx"] <= end), "ae_idx"]
            event = int(len(after) > 0)
            rows.append({"fei": r["fei"], "insp_date": r["insp_date"], "appl_no": a,
                         "dqa_grade": r["dqa_grade"] if isinstance(r["dqa_grade"], str) else "",
                         "duration": (after.min() - i) if event else (end - i),
                         "event": event, "pre_quality": pre_q,
                         "log_pre_serious": np.log1p(n_pre)})
    d = pd.DataFrame(rows)
    d = d[d["duration"] > 0]
    ts = pd.read_csv(STEP02, parse_dates=["snapshot_date"])
    d = d.merge(ts[["fei", "snapshot_date"] + FEATURES],
                left_on=["fei", "insp_date"], right_on=["fei", "snapshot_date"])
    d["any_oai"] = (d["dqa_grade"] == "OAI").astype(int)
    print(f"(inspection, product) rows: {len(d):,}  inspections: "
          f"{d[['fei', 'insp_date']].drop_duplicates().shape[0]}  factories: {d['fei'].nunique()}  "
          f"events: {d['event'].sum():,} ({100 * d['event'].mean():.1f}%)")
    return d


def fit(d: pd.DataFrame, x: str) -> dict:
    cols = ["duration", "event", "pre_quality", "log_pre_serious", x, "fei"]
    m = d[cols].dropna().copy()
    if m[x].std() == 0:
        return {"feature": x}
    if x != "any_oai":
        m[x] = (m[x] - m[x].mean()) / m[x].std()
    cph = CoxPHFitter()
    cph.fit(m, duration_col="duration", event_col="event", cluster_col="fei")
    s = cph.summary.loc[x]
    return {"feature": x, "n_rows": len(m), "hr": s["exp(coef)"],
            "ci_lo": s["exp(coef) lower 95%"], "ci_hi": s["exp(coef) upper 95%"], "p": s["p"]}


def main() -> None:
    d = build()
    for label, sub in [("all kept inspections", d),
                       ("grade known", d[d["dqa_grade"].isin(["NAI", "VAI", "OAI"])])]:
        t = pd.DataFrame([fit(sub, f) for f in FEATURES + ["any_oai"]])
        text = t["feature"] != "any_oai"
        t.loc[text, "p_holm17"] = multipletests(t.loc[text, "p"], method="holm")[1]
        t.insert(0, "sample", label)
        print(f"\n{label}: hazard ratio per SD (any_oai: OAI vs not)")
        print(t.sort_values("p").round(3).to_string(index=False))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(OUT, mode="w" if label.startswith("all") else "a",
                 header=label.startswith("all"), index=False)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
