"""
Module 19 — FEI x year shortage prediction model, and the VAI-only leading-
indicator hypothesis.

Panel: FEI x year, 2015-2024, over our full 129-FEI reference universe that
has FDA Inspection Details records (not restricted to 483-text coverage --
see "Universe fix" below). For each facility in year t, predict whether any
drug it manufactures enters a UUtah-tracked shortage in year t+1.

New in this module (2026-09-16):
  - OAI/VAI/NAI classification now comes directly from
    Data/14 - FDA - Inspection/raw/Inspections Details.xlsx, refreshed by the
    user this session (current through 2026-09-09), instead of the Redica-
    derived classification m14/m17 use (redica_all_drugs_combined.csv, last
    refreshed 2026-07-15). m14/m17 are left untouched; this module is the
    first to use the fresher FDA-direct source, since the specific question
    here (OAI vs. VAI shortage risk) depends on having the most current
    facility classification available.
  - Facility-level shortage outcome, bridged from UUtah's drug-level
    shortage list via the same Valisure FEI-API map m14/m17 use. Every
    facility that manufactures a drug is treated as "exposed" when that drug
    enters shortage -- this is a real limitation (shortage is a molecule-
    level event, not necessarily caused by any one facility), same reason
    the team moved off shortage-as-outcome in June 2026. Kept explicit in
    the panel summary rather than hidden.
  - Universe fix: the panel is no longer restricted to text-covered FEIs.
    A facility's inspection record is real signal on its own; restricting
    the whole panel to text coverage discarded it for no reason related to
    the model actually being run. A separate baseline model (inspection +
    structural only, full reference universe) is checked and run
    independently of the with-text model, same fix applied to m14/m17.
  - Two analyses beyond the standard prediction models:
      (a) Descriptive: does an OAI-ever facility show lower forward shortage
          risk than a VAI-only facility (testing the literature's
          "OAI reduces shortage risk" finding, Wang/Anand/Ball/Park)?
      (b) VAI-only subgroup: does the 483 text signal predict shortage
          within facilities that never received an OAI, i.e. the same
          "leading indicator FDA's classification misses" test already run
          for adverse events (vai_signal_validation/), now for shortage.

Outputs:
  outputs/models/metrics_shortage_baseline.csv (inspection+structural, full universe)
  outputs/models/metrics_shortage_fei.csv       (with-text, text-covered subset)
  outputs/models/metrics_shortage_vai_only.csv
  outputs/tables/shortage_fei_panel_summary.md
  outputs/tables/oai_vs_vai_shortage_rates.csv
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    _SKLEARN = True
except ModuleNotFoundError:
    _SKLEARN = False

from config import (
    REDICA_CSV, VALISURE_FEI, VALISURE_CSV, UUTAH_FILE, DATA,
    TEXT_TIMESERIES_REDICA_CSV,
    OUT_DATA, OUT_FIGS, OUT_TABS, OUT_MODELS, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR, SEED,
)
from utils import get_logger, write_table, ValisureDrugMatcher, load_valisure_api_names

log = get_logger("m19_shortage_fei", OUT_LOGS / "m19_shortage_fei.log")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FDA_INSP_XLSX = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
OB_PRODUCTS_CSV = DATA / "01 - Orange Book" / "output_data" / "products.csv"

_PARENTERAL_ROUTES = {
    "INJECTION", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS",
    "INJECTION, INTRAVENOUS", "INTRAVENOUS, SUBCUTANEOUS",
    "INTRAMUSCULAR, INTRAVENOUS", "INJECTABLE", "IRRIGATION",
    "INJECTION, SUBCUTANEOUS",
}

# ── Feature groups (identical to m14/m17) ────────────────────────────────────

INSP_FEATURES = ["n_oai_cumul", "n_vai_t", "n_inspections_t"]

# severity_majmod_share (Major+Moderate collapsed), not severity_critmajor_share
# (Critical+Major collapsed): human-eval accuracy (eval/results_and_notes/
# 20260916_LLM_Extraction_Validation_Report.docx, Section 2) is 90-94% for the
# Major/Moderate collapse vs. 66-68% for the raw 4-tier -- the extraction's real
# confusion is at the Major/Moderate boundary, not Critical/Major.
TEXT_FEATURES = [
    "severity_majmod_share",
    "scope_facilitywide_share",
    "scope_multipleproducts_share",
    "cultural_root_cause_share",
    "contamination_llm_share",
    "data_integrity_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "vc_laboratorycontrolssystem_share",
    "vc_qualitysystem_share",
    "remediation_none_share",
    "remediation_weak_share",
]

STRUCT_FEATURES = ["parenteral_ever", "n_feis_drug"]

ALL_FEATURES = INSP_FEATURES + TEXT_FEATURES + STRUCT_FEATURES


# ── OAI/VAI/NAI classification from FDA Inspection Details (primary here) ────

def _load_fda_inspection_outcomes() -> pd.DataFrame:
    """FEI x year OAI/VAI/NAI counts, direct from FDA Inspection Details.xlsx.

    Refreshed by the user 2026-09-16 (current through 2026-09-09), fresher
    than the Redica-derived classification m14/m17 use (2026-07-15).
    """
    fda = pd.read_excel(
        FDA_INSP_XLSX,
        usecols=["FEI Number", "Inspection End Date", "Classification", "Project Area"],
    )
    fda = fda[fda["Project Area"] == "Drug Quality Assurance"].copy()
    fda["fei"]  = pd.to_numeric(fda["FEI Number"], errors="coerce").astype("Int64")
    fda["year"] = pd.to_datetime(fda["Inspection End Date"], errors="coerce").dt.year.astype("Int64")
    fda = fda.dropna(subset=["fei", "year", "Classification"])
    cls = fda["Classification"].astype(str).str.upper()
    fda["is_oai"] = cls.str.contains("OFFICIAL ACTION").astype(int)
    fda["is_vai"] = cls.str.contains("VOLUNTARY ACTION").astype(int)

    agg = fda.groupby(["fei", "year"], as_index=False).agg(
        n_inspections_t=("Classification", "count"),
        n_oai_t=("is_oai", "sum"),
        n_vai_t=("is_vai", "sum"),
    )
    log.info(
        "FDA Inspection Details (folder 14, refreshed): %d FEI-year rows, %d FEIs, "
        "raw data through %s",
        len(agg), agg["fei"].nunique(),
        pd.to_datetime(fda["Inspection End Date"]).max().date(),
    )
    return agg


def _add_cumulative_oai(fei_year: pd.DataFrame) -> pd.DataFrame:
    fei_year = fei_year.sort_values(["fei", "year"])
    fei_year["n_oai_cumul"] = fei_year.groupby("fei")["n_oai_t"].cumsum()
    return fei_year


# ── Text features (identical to m14/m17) ─────────────────────────────────────

def _load_text_features() -> pd.DataFrame:
    if not TEXT_TIMESERIES_REDICA_CSV.exists():
        log.warning("Redica text timeseries not found at %s", TEXT_TIMESERIES_REDICA_CSV)
        return pd.DataFrame(columns=["fei", "snapshot_date"] + TEXT_FEATURES)
    df = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    df["fei"] = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["fei", "snapshot_date"])
    log.info("Text timeseries: %d rows, %d FEIs", len(df), df["fei"].nunique())
    return df


def _join_text_as_of_year(panel: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    if text.empty:
        for col in TEXT_FEATURES:
            panel[col] = np.nan
        return panel
    text = text.sort_values(["fei", "snapshot_date"])
    result_rows = []
    for (fei_val, year_val), grp in panel.groupby(["fei", "year"]):
        cutoff = pd.Timestamp(int(year_val), 12, 31)
        fei_snaps = text[text["fei"] == fei_val]
        valid = fei_snaps[fei_snaps["snapshot_date"] <= cutoff]
        if valid.empty:
            row_extras = {col: np.nan for col in TEXT_FEATURES}
        else:
            latest = valid.iloc[-1]
            row_extras = {col: latest.get(col, np.nan) for col in TEXT_FEATURES}
        for idx in grp.index:
            result_rows.append({**panel.loc[idx].to_dict(), **row_extras})
    return pd.DataFrame(result_rows)


# ── Structural features (identical to m14/m17) ───────────────────────────────

def _parenteral_apis_from_ob() -> set[str]:
    fallback = {
        "Ampicillin", "Ampicillin; Sulbactam", "Vancomycin",
        "Potassium Chloride", "Magnesium Sulfate", "Calcium Gluconate",
        "Pantoprazole", "Azithromycin",
    }
    if not OB_PRODUCTS_CSV.exists():
        return fallback
    ob = pd.read_csv(OB_PRODUCTS_CSV)
    ob["_route"] = ob["DF;Route"].str.split(";").str[-1].str.strip().str.upper()
    par_ingredients = set(
        ob.loc[ob["_route"].isin(_PARENTERAL_ROUTES), "Ingredient"].str.upper().dropna().unique()
    )
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    api_col = next((c for c in fei_map.columns if c.lower() == "api"), None)
    if api_col is None:
        return fallback
    all_apis = fei_map[api_col].dropna().astype(str).str.strip().unique()
    parenteral = set()
    for api in all_apis:
        parts = [p.strip().upper().split()[0] for p in api.split(";") if p.strip()]
        for ing in par_ingredients:
            ing_words = ing.split()
            if any(part in ing_words for part in parts):
                parenteral.add(api)
                break
    return parenteral


def _load_fei_drug_map() -> pd.DataFrame:
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    fei_col = next(c for c in fei_map.columns if "fei" in c.lower() and "unique" not in c.lower())
    api_col = next(c for c in fei_map.columns if c.lower() == "api")
    fm = fei_map[[fei_col, api_col]].dropna().rename(columns={fei_col: "fei", api_col: "api"})
    fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")
    return fm.dropna(subset=["fei"]).drop_duplicates()


def _load_structural_features(fei_drug_map: pd.DataFrame) -> pd.DataFrame:
    parenteral_apis = _parenteral_apis_from_ob()
    fm = fei_drug_map.copy()
    fm["parenteral_ever"] = fm["api"].isin(parenteral_apis).astype(int)
    api_fei_counts = fm.groupby("api")["fei"].nunique()
    fm["n_feis_drug"] = fm["api"].map(api_fei_counts)
    out = fm.groupby("fei", as_index=False).agg(
        parenteral_ever=("parenteral_ever", "max"),
        n_feis_drug=("n_feis_drug", "min"),
    )
    log.info("Structural features: %d FEIs, parenteral_ever=%d",
             len(out), int(out["parenteral_ever"].sum()))
    return out


# ── Shortage outcome (new) ────────────────────────────────────────────────────

def _load_shortage_drug_year() -> pd.DataFrame:
    """Drug (Valisure API) x year shortage-onset indicator from UUtah."""
    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher = ValisureDrugMatcher(api_names)

    raw = pd.read_excel(UUTAH_FILE, header=1)
    raw = raw.rename(columns={raw.columns[0]: "drug_name"})
    raw["drug_norm"] = raw["drug_name"].map(matcher.match)
    raw = raw.dropna(subset=["drug_norm"]).copy()

    raw["date_notified"] = pd.to_datetime(raw["Date Notified"], errors="coerce")
    yr_int = pd.to_numeric(raw["yr"], errors="coerce")
    raw["start_year"] = raw["date_notified"].dt.year.fillna(yr_int).astype("Int64")
    raw = raw.dropna(subset=["start_year"])

    starts = (raw.groupby(["drug_norm", "start_year"], as_index=False)
                 .size().rename(columns={"start_year": "year", "size": "n_starts"}))
    starts["shortage_started"] = 1
    log.info("UUtah shortage starts matched to our 14 APIs: %d drug-year rows, "
             "%d drugs, %d total onset events", len(starts), starts["drug_norm"].nunique(),
             int(raw.shape[0]))
    return starts[["drug_norm", "year", "shortage_started", "n_starts"]]


def _load_shortage_fei_year(fei_drug_map: pd.DataFrame) -> pd.DataFrame:
    """Bridge drug-year shortage onsets to FEI-year via the FEI-API map.

    Every FEI manufacturing a drug that enters shortage in year Y is marked
    exposed in year Y. This is a real limitation: shortage is a molecule-
    level event and not every manufacturer of a shortaged drug caused it.
    Kept explicit rather than implied.
    """
    fei_map = fei_drug_map.rename(columns={"api": "drug_norm"})
    shortage_dy = _load_shortage_drug_year()
    merged = fei_map.merge(shortage_dy, on="drug_norm", how="inner")
    agg = merged.groupby(["fei", "year"], as_index=False).agg(
        shortage_started=("shortage_started", "max"),
        n_starts=("n_starts", "sum"),
    )
    log.info("Shortage-exposed FEI-year rows (any drug in shortage that year): %d, %d FEIs",
             len(agg), agg["fei"].nunique())
    return agg


# ── Panel assembly ────────────────────────────────────────────────────────────

def build_panel() -> pd.DataFrame:
    redica_fy = _load_fda_inspection_outcomes()
    redica_fy = _add_cumulative_oai(redica_fy)
    text_ts   = _load_text_features()
    fei_drug_map = _load_fei_drug_map()
    struct    = _load_structural_features(fei_drug_map)
    shortage_fy = _load_shortage_fei_year(fei_drug_map)

    # Universe: our 129-FEI reference universe (Valisure FEI-API map, the same set
    # m14/m17 draw from) intersected with FEIs that actually have FDA Inspection
    # Details records -- NOT intersected with text coverage. fda_feis itself is
    # unfiltered (it's every US facility in the raw FDA file, ~9,952 FEIs), so it
    # must be restricted to our reference universe via fei_drug_map regardless;
    # the text-coverage restriction on top of that was throwing away real
    # inspection-based signal for facilities without 483 text yet, the same
    # problem fixed in m14/m17 on 2026-09-16.
    text_feis = set(text_ts["fei"].dropna().unique())
    fda_feis  = set(redica_fy["fei"].dropna().unique())
    reference_feis = set(fei_drug_map["fei"].dropna().unique())
    all_feis  = np.array(sorted(fda_feis & reference_feis))
    log.info(
        "Panel universe: %d FEIs in our reference universe with FDA Inspection Details "
        "records (%d also have 483 text coverage)",
        len(all_feis), len(set(all_feis) & text_feis),
    )
    years = range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
    panel = pd.DataFrame(
        index=pd.MultiIndex.from_product([all_feis, years], names=["fei", "year"])
    ).reset_index()
    panel["fei"]  = panel["fei"].astype("Int64")
    panel["year"] = panel["year"].astype("Int64")

    panel = panel.merge(
        redica_fy[["fei", "year", "n_oai_cumul", "n_vai_t", "n_inspections_t"]],
        on=["fei", "year"], how="left",
    )
    panel = panel.merge(struct, on="fei", how="left")
    panel = _join_text_as_of_year(panel, text_ts)

    # Outcome: shortage exposure in year t+1
    shortage_next = shortage_fy.rename(columns={"year": "year_next", "shortage_started": "shortage_next"})
    shortage_next["year"] = shortage_next["year_next"] - 1
    panel = panel.merge(shortage_next[["fei", "year", "shortage_next"]], on=["fei", "year"], how="left")
    panel["y_shortage_next"] = panel["shortage_next"].fillna(0).astype(int)

    for col in ["n_inspections_t", "n_vai_t"]:
        panel[col] = panel[col].fillna(0)
    panel["n_oai_cumul"] = panel["n_oai_cumul"].fillna(0)

    log.info(
        "Panel: %d rows, %d FEIs, %d years | shortage-exposed FEI-years: %d (%.1f%%)",
        len(panel), panel["fei"].nunique(), panel["year"].nunique(),
        int(panel["y_shortage_next"].sum()), 100 * panel["y_shortage_next"].mean(),
    )
    return panel


# ── Modeling ──────────────────────────────────────────────────────────────────

def _prep(panel: pd.DataFrame, features: list[str]):
    df = panel.dropna(subset=["y_shortage_next"]).copy()
    feats_in = [f for f in features if f in df.columns]
    text_in_features = [f for f in feats_in if f in TEXT_FEATURES]
    if text_in_features:
        before = len(df)
        df = df.dropna(subset=text_in_features)
        if before - len(df):
            log.info("Dropped %d FEI-year rows with no as-of-year text snapshot (%d remain)",
                      before - len(df), len(df))
    X = df[feats_in].fillna(0).astype(float)
    y = df["y_shortage_next"].astype(int)
    groups = df["fei"].astype(str)
    return X, y, groups, df


def _cv_metrics(X, y, groups, model_factory, n_splits=5):
    n_splits = min(n_splits, max(2, groups.nunique() - 1))
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros(len(y))
    fold_aucs = []
    for tr, te in gkf.split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            preds[te] = y.iloc[tr].mean()
            continue
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = m.predict_proba(X.iloc[te])[:, 1]
        y_te = y.iloc[te]
        if y_te.nunique() == 2:
            fold_aucs.append(roc_auc_score(y_te, preds[te]))
    auc = roc_auc_score(y, preds) if y.sum() > 0 else float("nan")
    ap  = average_precision_score(y, preds) if y.sum() > 0 else float("nan")
    # One-tailed t-test of fold-level AUCs vs. 0.5 -- same convention as
    # vai_signal_validation/02_vai_signal_model.py and the original INFORMS
    # slide's own stated test. Not previously computed for this model; added
    # so the baseline finding (see RESULTS.docx) can be read against a
    # significance threshold instead of just the pooled AUC.
    if len(fold_aucs) >= 2:
        t_stat, p_two = stats.ttest_1samp(fold_aucs, 0.5)
        p_vs_half = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:
        p_vs_half = float("nan")
    return preds, {
        "auc": auc, "ap": ap, "n": len(y), "events": int(y.sum()),
        "p_vs_0.5": p_vs_half, "n_folds": len(fold_aucs),
    }


def _oai_vs_vai_shortage_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Descriptive check: does a facility with an OAI on record as of year t
    show a different forward shortage risk than one with no OAI as of year t
    (VAI/NAI only so far)? Tests the literature's OAI-reduces-shortage-risk
    finding directly on our panel.

    Uses n_oai_cumul (cumulative OAI count up to and including year t),
    already computed as-of-year in the panel -- not a whole-history "ever
    OAI" flag, which would leak information from OAIs that happened *after*
    the year t+1 shortage outcome being measured.
    """
    df = panel.dropna(subset=["y_shortage_next"]).copy()
    df["oai_as_of_t"] = (df["n_oai_cumul"] > 0).astype(int)
    rows = []
    for label, sub in [("OAI on record as of year t", df[df["oai_as_of_t"] == 1]),
                        ("No OAI as of year t (VAI/NAI only so far)", df[df["oai_as_of_t"] == 0])]:
        rows.append({
            "group": label,
            "n_fei_years": len(sub),
            "n_feis": sub["fei"].nunique(),
            "shortage_next_rate": round(sub["y_shortage_next"].mean(), 4) if len(sub) else float("nan"),
            "shortage_next_events": int(sub["y_shortage_next"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not _SKLEARN:
        log.warning("scikit-learn not installed; skipping m19")
        return

    panel = build_panel()
    write_table(panel, OUT_DATA / "shortage_fei_panel.parquet", log)

    # ── Panel summary ──
    modeled = panel.dropna(subset=["y_shortage_next"])
    with_snapshot = modeled.dropna(subset=TEXT_FEATURES)
    summary_lines = [
        "# Shortage FEI Panel Summary",
        "",
        f"- **FEI x year rows (full panel, baseline model population):** {len(panel):,}",
        f"- **Unique FEIs:** {panel['fei'].nunique()}",
        f"- **Rows with a valid outcome (baseline model population):** {len(modeled):,} "
        f"({modeled['fei'].nunique()} FEIs)",
        f"- **Rows with an as-of-year text snapshot (with-text model population, no "
        f"zero-fill):** {len(with_snapshot):,} ({with_snapshot['fei'].nunique()} FEIs, "
        f"{int(with_snapshot['y_shortage_next'].sum())} shortage-exposed events)",
        f"- **Shortage-exposed FEI-years, full outcome set:** "
        f"{int(modeled['y_shortage_next'].sum())} ({100*modeled['y_shortage_next'].mean():.1f}%)",
        "",
        "Three models are attempted independently (see RESULTS.docx): a baseline using "
        "inspection + structural features on the full panel above, a with-text model "
        "restricted to the as-of-year-snapshot population, and a VAI-only text-signal "
        "subgroup model within that.",
        "",
        "Caveat: shortage exposure is bridged from drug-level UUtah events through the "
        "FEI-API map. Every facility manufacturing a shortaged drug is marked exposed; "
        "this does not mean that facility caused the shortage.",
    ]
    (OUT_TABS / "shortage_fei_panel_summary.md").write_text("\n".join(summary_lines))
    log.info("\n".join(summary_lines))

    # ── Descriptive: OAI vs VAI-only forward shortage risk ──
    rates = _oai_vs_vai_shortage_rates(panel)
    rates.to_csv(OUT_TABS / "oai_vs_vai_shortage_rates.csv", index=False)
    log.info("OAI vs VAI-only forward shortage risk:\n%s", rates.to_string(index=False))

    # ── Baseline: inspection + structural only, full reference universe, not
    # restricted to text coverage (same fix applied to m14/m17 on 2026-09-16).
    # Checked and modeled independently of the with-text models below. ──
    X_base, y_base, g_base, df_base = _prep(panel, INSP_FEATURES + STRUCT_FEATURES)
    log.info("Baseline (inspection+structural, full universe): rows=%d events=%d FEIs=%d",
              len(X_base), int(y_base.sum()), g_base.nunique())
    if y_base.sum() >= 3 and len(X_base) >= 20:
        Xz_base = pd.DataFrame(StandardScaler().fit_transform(X_base), columns=X_base.columns)
        _, met_base_l2 = _cv_metrics(Xz_base, y_base, g_base,
            lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500,
                                       class_weight="balanced", random_state=SEED))
        _, met_base_rf = _cv_metrics(X_base, y_base, g_base,
            lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1))
        pd.DataFrame([
            {"model": "L2_logit", **met_base_l2},
            {"model": "RandomForest", **met_base_rf},
        ]).to_csv(OUT_MODELS / "metrics_shortage_baseline.csv", index=False)
        log.info("Baseline L2 AUC=%.3f (p=%.3f) RF AUC=%.3f (p=%.3f)",
                  met_base_l2["auc"], met_base_l2["p_vs_0.5"],
                  met_base_rf["auc"], met_base_rf["p_vs_0.5"])
    else:
        log.warning("Baseline model has too few events (n=%d events=%d); skipping, not "
                     "reporting an unreliable number", len(X_base), int(y_base.sum()))

    # ── With-text prediction model (text-covered subset) ──
    X_all, y, groups, df_model = _prep(panel, ALL_FEATURES)
    log.info("With-text model: rows=%d events=%d FEIs=%d", len(X_all), int(y.sum()), groups.nunique())
    if y.sum() >= 3 and len(X_all) >= 20:
        Xz = pd.DataFrame(StandardScaler().fit_transform(X_all), columns=X_all.columns)
        _, met_l2 = _cv_metrics(Xz, y, groups,
            lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500,
                                       class_weight="balanced", random_state=SEED))
        _, met_rf = _cv_metrics(X_all, y, groups,
            lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1))
        pd.DataFrame([
            {"model": "L2_logit", **met_l2},
            {"model": "RandomForest", **met_rf},
        ]).to_csv(OUT_MODELS / "metrics_shortage_fei.csv", index=False)
        log.info("With-text L2 AUC=%.3f (p=%.3f) RF AUC=%.3f (p=%.3f)",
                  met_l2["auc"], met_l2["p_vs_0.5"], met_rf["auc"], met_rf["p_vs_0.5"])
    else:
        log.warning("Too few shortage events (n=%d events=%d) for the with-text model; "
                     "skipping, not reporting an unreliable number", len(X_all), int(y.sum()))

    # ── VAI-only subgroup: does text predict shortage where FDA's label can't? ──
    # As-of-year-t, not whole-history "ever OAI" -- a row from before a
    # facility's first OAI belongs in the VAI-only group for that year.
    df_vai = df_model[df_model["n_oai_cumul"] == 0].copy()
    log.info("VAI-only subgroup: %d FEIs, %d rows, %d shortage events",
             df_vai["fei"].nunique(), len(df_vai), int(df_vai["y_shortage_next"].sum()))
    if df_vai["y_shortage_next"].sum() >= 3 and len(df_vai) >= 20:
        X_vai = df_vai[[f for f in TEXT_FEATURES if f in df_vai.columns]].astype(float)
        y_vai = df_vai["y_shortage_next"].astype(int)
        g_vai = df_vai["fei"].astype(str)
        Xz_vai = pd.DataFrame(StandardScaler().fit_transform(X_vai), columns=X_vai.columns)
        _, met_vai_l2 = _cv_metrics(Xz_vai, y_vai, g_vai,
            lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500,
                                       class_weight="balanced", random_state=SEED))
        _, met_vai_rf = _cv_metrics(X_vai, y_vai, g_vai,
            lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1))
        pd.DataFrame([
            {"model": "L2_logit", **met_vai_l2},
            {"model": "RandomForest", **met_vai_rf},
        ]).to_csv(OUT_MODELS / "metrics_shortage_vai_only.csv", index=False)
        log.info("VAI-only text-signal shortage model: L2 AUC=%.3f (p=%.3f) RF AUC=%.3f (p=%.3f)",
                  met_vai_l2["auc"], met_vai_l2["p_vs_0.5"],
                  met_vai_rf["auc"], met_vai_rf["p_vs_0.5"])
    else:
        log.warning(
            "VAI-only subgroup has too few shortage events (n=%d events=%d) to model "
            "reliably; not reporting an unreliable number. This is the same sparsity "
            "problem that moved the team off shortage-as-outcome in June 2026.",
            len(df_vai), int(df_vai["y_shortage_next"].sum()),
        )

    log.info("m19 complete -- outputs in %s", OUT_MODELS)


if __name__ == "__main__":
    main()
