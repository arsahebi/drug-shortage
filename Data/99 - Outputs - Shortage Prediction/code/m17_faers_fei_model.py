"""
Module 17 — FEI × year adverse event prediction model.

Panel: FEI × year, 2015–2024.  For each facility in year t, predict whether
serious adverse event volume is above-median in year t+1
(y_ae_next = 1 if drug-year AE count > median for that drug across all years).

Adverse events are the primary dependent variable: they directly measure patient
harm from quality failures and are more interpretable than shortage (which has
confounding supply/demand factors) and more common than shortage events.

AE outcome construction:
  - Source: FAERS_ALL (14 Valisure drugs, pre-filtered serious AEs, 2015–2024)
  - Aggregate FAERS prod_ai × year → n_ae (count of serious AE reports)
  - Match prod_ai to Valisure API name (first-word fuzzy via ValisureDrugMatcher)
  - Join to FEI via Valisure API Only_FEI Mapping sheet
  - ae_high = 1 if n_ae > per-drug median, 0 otherwise (binary, per drug-year)
  - y_ae_next: take ae_high at FEI level (max across drugs) for year t+1

Feature groups (identical to m14):
  Inspection (Redica):    n_oai_cumul, n_vai_t, n_inspections_t, n_warning_letters_t
  Text/LLM (483 combined): severity_critmajor_share, scope_facilitywide_share,
                           scope_multipleproducts_share, cultural_root_cause_share,
                           contamination_llm_share, data_integrity_llm_share,
                           investigation_llm_share, repeat_cross_insp_share,
                           vc_labcontrols_share, vc_qualitysystem_share,
                           remediation_none_share, remediation_weak_share
  Structural (Valisure + OB): parenteral_ever, n_feis_drug

Time-aware join: for prediction year t, the text snapshot with the most recent
snapshot_date ≤ Dec 31 of year t is used (as-of feature).

Cross-validation: GroupKFold grouped by FEI to prevent data leakage.

Models: Logistic Regression (L2) and Random Forest.

Outputs:
  outputs/models/metrics_faers_fei.csv
  outputs/models/rf_importance_faers_fei.csv
  outputs/models/text_ablation_faers_fei.csv
  outputs/figures/roc_faers_fei.png
  outputs/figures/feature_importance_faers_fei.png
  outputs/figures/text_feature_lift_faers_fei.png
  outputs/tables/faers_fei_panel_summary.md
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        roc_curve, brier_score_loss,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ModuleNotFoundError:
    _SKLEARN = False

from config import (
    FAERS_ALL, REDICA_CSV, VALISURE_FEI, VALISURE_CSV,
    TEXT_TIMESERIES_REDICA_CSV, DATA,
    OUT_DATA, OUT_FIGS, OUT_TABS, OUT_MODELS, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR, SEED,
)
from utils import (
    get_logger, write_table, ValisureDrugMatcher, load_valisure_api_names,
)

OB_PRODUCTS_CSV = DATA / "01 - Orange Book" / "output_data" / "products.csv"

_PARENTERAL_ROUTES = {
    "INJECTION", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS",
    "INJECTION, INTRAVENOUS", "INTRAVENOUS, SUBCUTANEOUS",
    "INTRAMUSCULAR, INTRAVENOUS", "INJECTABLE", "IRRIGATION",
    "INJECTION, SUBCUTANEOUS",
}

log = get_logger("m17_faers_fei", OUT_LOGS / "m17_faers_fei.log")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Feature groups ──────────────────────────────────────────────────────────

INSP_FEATURES = [
    "n_oai_cumul",
    "n_vai_t",
    "n_inspections_t",
    "n_warning_letters_t",
]

TEXT_FEATURES = [
    "severity_critmajor_share",
    "scope_facilitywide_share",
    "scope_multipleproducts_share",
    "cultural_root_cause_share",
    "contamination_llm_share",
    "data_integrity_llm_share",
    "investigation_llm_share",
    "repeat_cross_insp_share",
    "vc_labcontrols_share",
    "vc_qualitysystem_share",
    "remediation_none_share",
    "remediation_weak_share",
]

STRUCT_FEATURES = [
    "parenteral_ever",
    "n_feis_drug",
]

ALL_FEATURES = INSP_FEATURES + TEXT_FEATURES + STRUCT_FEATURES

_GROUP_COLORS = {f: "steelblue" for f in INSP_FEATURES}
_GROUP_COLORS.update({f: "darkorange" for f in TEXT_FEATURES})
_GROUP_COLORS.update({f: "seagreen" for f in STRUCT_FEATURES})


# ── Data loading ────────────────────────────────────────────────────────────

def _load_fei_drug_map() -> pd.DataFrame:
    """Load FEI → Valisure API mapping (one row per FEI-API pair)."""
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    fei_col = next((c for c in fei_map.columns if "fei" in c.lower()), None)
    api_col = next((c for c in fei_map.columns if c.lower() == "api"), None)
    if not fei_col or not api_col:
        log.warning("Cannot find FEI/API columns in Valisure FEI map")
        return pd.DataFrame(columns=["fei", "api"])
    fm = fei_map[[fei_col, api_col]].dropna().rename(columns={fei_col: "fei", api_col: "api"})
    fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")
    return fm.dropna(subset=["fei"])


def _load_faers_fei_year() -> pd.DataFrame:
    """Build FEI × year adverse event outcome from FAERS.

    Steps:
      1. Read FAERS (prod_ai, year, severity) — already filtered to 14 drugs + serious AEs
      2. Aggregate to drug-year: n_ae = count of serious AE reports
      3. Map FAERS prod_ai → Valisure canonical API name via ValisureDrugMatcher
      4. Compute per-drug median n_ae → ae_high = 1 if n_ae > median
      5. Join to FEI via Valisure API Only_FEI Mapping
      6. Return FEI × drug × year with ae_high column
    """
    if not FAERS_ALL.exists():
        log.warning("FAERS file not found at %s; returning empty AE frame", FAERS_ALL)
        return pd.DataFrame(columns=["fei", "year", "drug", "n_ae", "ae_high"])

    df = pd.read_csv(FAERS_ALL, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    prod_col = next((c for c in df.columns if c.lower() == "prod_ai"), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if not prod_col or not year_col:
        log.warning("FAERS file missing prod_ai or year column; columns: %s", list(df.columns))
        return pd.DataFrame(columns=["fei", "year", "drug", "n_ae", "ae_high"])

    df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", prod_col])
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]

    # Aggregate to prod_ai × year
    drug_year = (
        df.groupby([prod_col, "year"], as_index=False)
          .size()
          .rename(columns={"size": "n_ae", prod_col: "prod_ai"})
    )

    # Map prod_ai → Valisure canonical API name
    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher = ValisureDrugMatcher(api_names)
    drug_year["api"] = drug_year["prod_ai"].astype(str).map(matcher.match)
    unmatched = drug_year["api"].isna().sum()
    if unmatched:
        log.warning("FAERS: %d drug-year rows unmatched to Valisure API (dropped)", unmatched)
    drug_year = drug_year.dropna(subset=["api"])

    # Aggregate multiple prod_ai variants to same API (e.g. "TACROLIMUS" + "TACROLIMUS.")
    api_year = drug_year.groupby(["api", "year"], as_index=False)["n_ae"].sum()

    # Binary outcome: ae_high = 1 if this drug-year is above median for that drug
    median_ae = api_year.groupby("api")["n_ae"].transform("median")
    api_year["ae_high"] = (api_year["n_ae"] > median_ae).astype(int)

    log.info(
        "FAERS drug-year AE rates: %d rows, %d APIs, ae_high=%.1f%%",
        len(api_year), api_year["api"].nunique(),
        100 * api_year["ae_high"].mean(),
    )

    # Join to FEI via Valisure mapping
    fei_drug = _load_fei_drug_map()
    merged = fei_drug.merge(api_year, on="api", how="inner")
    merged = merged.rename(columns={"api": "drug"})

    log.info(
        "FAERS FEI-year outcome: %d rows, %d FEIs, %d drugs",
        len(merged), merged["fei"].nunique(), merged["drug"].nunique(),
    )
    return merged[["fei", "year", "drug", "n_ae", "ae_high"]]


def _load_redica_fei_year() -> pd.DataFrame:
    """FEI × year inspection features from Redica."""
    df = pd.read_csv(REDICA_CSV, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    df["fei"]  = pd.to_numeric(df["FEI"], errors="coerce").astype("Int64")
    df["year"] = pd.to_datetime(df["Event Date"], errors="coerce").dt.year.astype("Int64")
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]

    cls = df["Classification"].astype(str).str.strip()
    df["is_oai"] = (cls == "OAI").astype(int)
    df["is_vai"] = (cls == "VAI").astype(int)
    df["n_wl"]   = pd.to_numeric(df.get("Warning Letter", 0), errors="coerce").fillna(0).astype(int)

    agg = df.groupby(["fei", "year"], as_index=False).agg(
        n_inspections_t=("Event Date", "count"),
        n_oai_t=("is_oai", "sum"),
        n_vai_t=("is_vai", "sum"),
        n_warning_letters_t=("n_wl", "sum"),
    )
    log.info("Redica FEI-year rows: %d, FEIs: %d", len(agg), agg["fei"].nunique())
    return agg


def _add_cumulative_oai(fei_year: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative OAI count up to and including year t."""
    fei_year = fei_year.sort_values(["fei", "year"])
    fei_year["n_oai_cumul"] = fei_year.groupby("fei")["n_oai_t"].cumsum()
    return fei_year


def _load_text_features() -> pd.DataFrame:
    """Load combined timeseries; if not present, fall back gracefully."""
    if not TEXT_TIMESERIES_REDICA_CSV.exists():
        log.warning(
            "Redica text timeseries not found at %s. "
            "Run: python 02_aggregate_fei_features.py --source redica",
            TEXT_TIMESERIES_REDICA_CSV,
        )
        return pd.DataFrame(columns=["fei", "snapshot_date"] + TEXT_FEATURES)

    df = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    df["fei"]           = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["fei", "snapshot_date"])
    log.info("Text timeseries: %d rows, %d FEIs", len(df), df["fei"].nunique())
    return df


def _parenteral_apis_from_ob() -> set[str]:
    """Return set of API names that have ≥1 parenteral ANDA in Orange Book."""
    fallback = {
        "Ampicillin", "Ampicillin; Sulbactam", "Vancomycin",
        "Potassium Chloride", "Magnesium Sulfate", "Calcium Gluconate",
        "Pantoprazole", "Azithromycin",
    }
    if not OB_PRODUCTS_CSV.exists():
        log.warning("Orange Book not found at %s; using hardcoded parenteral set", OB_PRODUCTS_CSV)
        return fallback

    ob = pd.read_csv(OB_PRODUCTS_CSV)
    ob["_route"] = ob["DF;Route"].str.split(";").str[-1].str.strip().str.upper()
    par_ingredients = set(
        ob.loc[ob["_route"].isin(_PARENTERAL_ROUTES), "Ingredient"]
        .str.upper()
        .dropna()
        .unique()
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

    log.info("Parenteral APIs from Orange Book: %s", sorted(parenteral))
    return parenteral


def _load_structural_features() -> pd.DataFrame:
    """Load parenteral_ever and n_feis_drug from Valisure FEI map + Orange Book."""
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]

    fei_col = next((c for c in fei_map.columns if "fei" in c.lower()), None)
    api_col = next((c for c in fei_map.columns if c.lower() == "api"), None)
    if not fei_col or not api_col:
        log.warning("Cannot load structural features: FEI or API column missing")
        return pd.DataFrame(columns=["fei", "parenteral_ever", "n_feis_drug"])

    fm = fei_map[[fei_col, api_col]].dropna().rename(columns={fei_col: "fei", api_col: "api"})
    fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")

    parenteral_apis = _parenteral_apis_from_ob()
    fm["parenteral_ever"] = fm["api"].isin(parenteral_apis).astype(int)

    api_fei_counts = fm.groupby("api")["fei"].nunique()
    fm["n_feis_drug"] = fm["api"].map(api_fei_counts)

    out = fm.groupby("fei", as_index=False).agg(
        parenteral_ever=("parenteral_ever", "max"),
        n_feis_drug=("n_feis_drug", "min"),
    )
    log.info(
        "Structural features: %d FEIs, parenteral_ever=%d, n_feis_drug range=%d–%d",
        len(out), int(out["parenteral_ever"].sum()),
        int(out["n_feis_drug"].min()), int(out["n_feis_drug"].max()),
    )
    return out


def _load_facility_names() -> pd.DataFrame:
    """Load facility legal names from Inspections Details if available."""
    insp_path = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
    if not insp_path.exists():
        return pd.DataFrame(columns=["fei", "facility_name"])
    try:
        df = pd.read_excel(insp_path, usecols=["FEI Number", "Legal Name"])
        df.columns = ["fei", "facility_name"]
        df["fei"] = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
        return df.dropna(subset=["fei"]).drop_duplicates("fei")
    except Exception as exc:
        log.warning("Could not load facility names: %s", exc)
        return pd.DataFrame(columns=["fei", "facility_name"])


# ── Panel assembly ──────────────────────────────────────────────────────────

def _join_text_as_of_year(panel: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """For each (fei, year) in panel, join the most recent text snapshot ≤ Dec 31 of year."""
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


def build_panel() -> pd.DataFrame:
    """Build the FEI × year modeling panel with FAERS AE outcome."""
    ae_fy     = _load_faers_fei_year()
    redica_fy = _load_redica_fei_year()
    redica_fy = _add_cumulative_oai(redica_fy)
    text_ts   = _load_text_features()
    struct    = _load_structural_features()

    # Universe: all FEIs present in Redica data, crossed with all panel years
    all_feis = redica_fy["fei"].dropna().unique()
    years    = range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
    panel = pd.DataFrame(
        index=pd.MultiIndex.from_product([all_feis, years], names=["fei", "year"])
    ).reset_index()
    panel["fei"]  = panel["fei"].astype("Int64")
    panel["year"] = panel["year"].astype("Int64")

    # Inspection features
    panel = panel.merge(
        redica_fy[["fei", "year", "n_oai_cumul", "n_vai_t", "n_inspections_t", "n_warning_letters_t"]],
        on=["fei", "year"], how="left",
    )

    # Structural features
    panel = panel.merge(struct, on="fei", how="left")

    # Text features (time-aware)
    panel = _join_text_as_of_year(panel, text_ts)

    # Outcome: ae_high in year t+1 (max across drugs per FEI-year)
    ae_next = (
        ae_fy.groupby(["fei", "year"], as_index=False)["ae_high"]
             .max()
             .rename(columns={"year": "year_next", "ae_high": "ae_high_next"})
    )
    ae_next["year"] = ae_next["year_next"] - 1
    panel = panel.merge(ae_next[["fei", "year", "ae_high_next"]], on=["fei", "year"], how="left")
    panel["y_ae_next"] = panel["ae_high_next"].fillna(np.nan)

    # Fill inspection zeros
    for col in ["n_inspections_t", "n_vai_t", "n_warning_letters_t"]:
        panel[col] = panel[col].fillna(0)
    panel["n_oai_cumul"] = panel["n_oai_cumul"].fillna(0)

    n_events = int(panel["y_ae_next"].sum()) if panel["y_ae_next"].notna().any() else 0
    n_mod    = int(panel["y_ae_next"].notna().sum())
    log.info(
        "Panel: %d rows, %d FEIs, %d years | AE-high events: %d / %d rows (%.1f%%)",
        len(panel), panel["fei"].nunique(), panel["year"].nunique(),
        n_events, n_mod, 100 * n_events / max(n_mod, 1),
    )
    return panel


# ── Modeling ────────────────────────────────────────────────────────────────

def _prep(panel: pd.DataFrame, features: list[str]):
    df = panel.dropna(subset=["y_ae_next"]).copy()
    feats_in = [f for f in features if f in df.columns]
    missing  = set(features) - set(feats_in)
    if missing:
        log.warning("Features missing (dropped): %s", sorted(missing))
    X      = df[feats_in].fillna(0).astype(float)
    y      = df["y_ae_next"].astype(int)
    groups = df["fei"].astype(str)
    return X, y, groups, df


def _cv_metrics(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_factory, n_splits: int = 5):
    n_splits = min(n_splits, max(2, groups.nunique() - 1))
    gkf   = GroupKFold(n_splits=n_splits)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            preds[te] = y.iloc[tr].mean()
            continue
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = m.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y, preds)         if y.sum() > 0 else float("nan")
    ap  = average_precision_score(y, preds) if y.sum() > 0 else float("nan")
    bs  = brier_score_loss(y, preds)
    return preds, {"auc": auc, "ap": ap, "brier": bs, "n": len(y), "events": int(y.sum())}


# ── Figures ─────────────────────────────────────────────────────────────────

_FIG_STYLE = {
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
}


def _fig_roc(preds_l2, preds_rf, met_l2, met_rf, y):
    with plt.rc_context(_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for preds, name, met in [
            (preds_l2, "Logistic Regression", met_l2),
            (preds_rf, "Random Forest",       met_rf),
        ]:
            if y.sum() > 0:
                fpr, tpr, _ = roc_curve(y, preds)
                ax.plot(fpr, tpr, label=f"{name} (AUC={met['auc']:.3f})", linewidth=1.8)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC — Predict FEI adverse event surge at year t+1")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "roc_faers_fei.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved roc_faers_fei.png")


def _fig_feature_importance(fi: pd.DataFrame, top_n: int = 12):
    fi = fi.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    colors = [_GROUP_COLORS.get(f, "gray") for f in fi["feature"]]

    with plt.rc_context(_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.45 * top_n + 1))
        ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="none")
        ax.set_xlabel("Random Forest Feature Importance")
        ax.set_title(f"Top {top_n} Features — Adverse Event Prediction (FEI level)")

        legend_patches = [
            mpatches.Patch(color="steelblue",  label="Inspection"),
            mpatches.Patch(color="darkorange", label="Text/LLM"),
            mpatches.Patch(color="seagreen",   label="Structural"),
        ]
        ax.legend(handles=legend_patches, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "feature_importance_faers_fei.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved feature_importance_faers_fei.png")


def _fig_text_lift(ablation_rows: list[dict]):
    labels = [r["label"] for r in ablation_rows]
    aucs   = [r["auc"]   for r in ablation_rows]
    colors = ["#7B9EC9", "#D4896A", "#D4896A"]

    with plt.rc_context(_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, aucs, color=colors, width=0.5, edgecolor="white")
        ax.set_ylim(max(0, min(aucs) - 0.1), min(1, max(aucs) + 0.1))
        ax.set_ylabel("AUC-ROC (GroupKFold CV)")
        ax.set_title("483 Text Feature Lift — Adverse Event Prediction")
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{auc:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "text_feature_lift_faers_fei.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved text_feature_lift_faers_fei.png")


# ── Panel summary table ─────────────────────────────────────────────────────

def _write_panel_summary(panel: pd.DataFrame) -> None:
    modeled = panel.dropna(subset=["y_ae_next"])
    n_events = int(modeled["y_ae_next"].sum())
    lines = [
        "# FAERS FEI Adverse Event Panel Summary",
        "",
        f"- **FEI × year rows (full panel):** {len(panel):,}",
        f"- **Unique FEIs:** {panel['fei'].nunique()}",
        f"- **Years:** {int(panel['year'].min())}–{int(panel['year'].max())}",
        f"- **Rows used in modeling:** {len(modeled):,}",
        f"- **AE-high events (y=1):** {n_events} "
          f"({100 * modeled['y_ae_next'].mean():.1f}%)",
        f"- **FEIs with ≥1 AE-high event in panel:** "
          f"{int((modeled.groupby('fei')['y_ae_next'].max() == 1).sum())}",
        "",
        "## Feature coverage",
        f"- Inspection features: {sum(modeled[f].notna().any() for f in INSP_FEATURES)}/{len(INSP_FEATURES)} present",
        f"- Text/LLM features:   {sum(modeled[f].notna().any() for f in TEXT_FEATURES)}/{len(TEXT_FEATURES)} present",
        f"- Structural features: {sum(modeled[f].notna().any() for f in STRUCT_FEATURES)}/{len(STRUCT_FEATURES)} present",
    ]
    out_path = OUT_TABS / "faers_fei_panel_summary.md"
    out_path.write_text("\n".join(lines))
    log.info("Saved faers_fei_panel_summary.md")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not _SKLEARN:
        log.warning("scikit-learn not installed; skipping m17. Install with: pip install scikit-learn")
        return

    panel = build_panel()
    write_table(panel, OUT_DATA / "faers_fei_panel.parquet", log)
    _write_panel_summary(panel)

    X_all, y, groups, df_model = _prep(panel, ALL_FEATURES)
    X_insp, y_i, g_i, _       = _prep(panel, INSP_FEATURES + STRUCT_FEATURES)

    if y.sum() < 3 or len(X_all) < 20:
        log.warning("Too few events (n=%d events=%d); skipping modeling", len(X_all), int(y.sum()))
        return

    log.info("Modeling: rows=%d events=%d FEIs=%d features=%d",
             len(X_all), int(y.sum()), groups.nunique(), X_all.shape[1])

    # L2 Logistic Regression
    Xz = pd.DataFrame(StandardScaler().fit_transform(X_all), columns=X_all.columns)
    preds_l2, met_l2 = _cv_metrics(
        Xz, y, groups,
        lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500,
                                   class_weight="balanced", random_state=SEED),
    )
    log.info("L2 Logit  AUC=%.3f AP=%.3f Brier=%.3f", met_l2["auc"], met_l2["ap"], met_l2["brier"])

    # Random Forest
    preds_rf, met_rf = _cv_metrics(
        X_all, y, groups,
        lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1),
    )
    log.info("RandomForest AUC=%.3f AP=%.3f Brier=%.3f", met_rf["auc"], met_rf["ap"], met_rf["brier"])

    pd.DataFrame([
        {"model": "L2_logit",     **met_l2},
        {"model": "RandomForest", **met_rf},
    ]).to_csv(OUT_MODELS / "metrics_faers_fei.csv", index=False)

    # Ablation: no text features
    Xz_i = pd.DataFrame(StandardScaler().fit_transform(X_insp), columns=X_insp.columns)
    _, met_no_text = _cv_metrics(
        Xz_i, y_i, g_i,
        lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=500,
                                   class_weight="balanced", random_state=SEED),
    )
    _, met_no_rf = _cv_metrics(
        X_insp, y_i, g_i,
        lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                       class_weight="balanced", random_state=SEED, n_jobs=-1),
    )
    log.info("Without text — L2 AUC=%.3f  RF AUC=%.3f", met_no_text["auc"], met_no_rf["auc"])

    ablation_rows = [
        {"label": "Without text",       "auc": met_no_text["auc"], "model": "L2"},
        {"label": "With text\n(all FEIs)", "auc": met_l2["auc"],   "model": "L2"},
    ]
    pd.DataFrame(ablation_rows).to_csv(OUT_MODELS / "text_ablation_faers_fei.csv", index=False)
    _fig_text_lift(ablation_rows)

    _fig_roc(preds_l2, preds_rf, met_l2, met_rf, y)

    rf_full = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                     class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_full.fit(X_all, y)

    fi = pd.DataFrame({"feature": X_all.columns, "importance": rf_full.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(OUT_MODELS / "rf_importance_faers_fei.csv", index=False)
    _fig_feature_importance(fi)

    log.info("m17 complete — outputs in %s", OUT_MODELS)


if __name__ == "__main__":
    main()
