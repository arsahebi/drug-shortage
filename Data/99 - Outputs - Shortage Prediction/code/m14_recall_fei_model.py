"""
Module 14 — FEI × year recall prediction model.

Panel: FEI × year, 2015–2024.  For each facility in year t, predict whether
a recall event occurs in year t+1 (y_recall_next = 1 if ≥1 recall linked to that FEI).

Recalls are a cleaner prediction target than shortages for this FEI-level panel:
  - Binary (0/1) at facility level, well-defined onset date
  - More events than shortages across the 99-FEI universe
  - Published literature has established baseline models for comparison

Feature groups:
  Inspection (Redica):    n_oai_cumul, n_vai_t, n_inspections_t, n_warning_letters_t
  Text/LLM (483 combined): severity_critmajor_share, scope_facilitywide_share,
                           scope_multipleproducts_share, cultural_root_cause_share,
                           contamination_llm_share, data_integrity_llm_share,
                           investigation_llm_share, repeat_cross_insp_share,
                           vc_labcontrols_share, vc_qualitysystem_share,
                           remediation_none_share, remediation_weak_share
  Structural (Valisure + OB): parenteral_ever (Orange Book dosage routes),
                              n_feis_drug (FEI count per drug — supply concentration)

Time-aware join: for prediction year t, the text snapshot with the most recent
snapshot_date ≤ Dec 31 of year t is used (as-of feature).

Cross-validation: GroupKFold grouped by FEI to prevent data leakage across
facility-years of the same facility.

Models: Logistic Regression (L2) and Random Forest.

Outputs:
  outputs/models/metrics_recall_fei.csv
  outputs/models/rf_importance_recall_fei.csv
  outputs/figures/roc_recall_fei.png
  outputs/figures/feature_importance_recall_fei.png
  outputs/figures/fei_risk_ranking.png
  outputs/tables/recall_fei_panel_summary.md
  outputs/tables/fei_risk_ranking.csv
  outputs/figures/text_feature_lift_recall_fei.png      (ablation Fig D)

TODO (SDUD volume weighting): Incorporate Medicaid volume shares as a feature or
sample weight.  The Valisure NDC_FEI Mapping sheet has full 11-digit NDCs; strip
dashes and cast to int to join against SDUD's ndc11 integer column (98% match rate).
The pre-joined SDUD+NADAC panel at 04_11/processed/2026-03-26-sdud_nadac_panel.csv
already has drug_name + ndc11 for 13 of 14 target drugs.  See config.py SDUD TODO.
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
    REDICA_CSV, VALISURE_FEI, VALISURE_CSV, RECALL_FILT,
    TEXT_TIMESERIES_REDICA_CSV, DATA,
    OUT_DATA, OUT_FIGS, OUT_TABS, OUT_MODELS, OUT_LOGS,
    PANEL_START_YEAR, PANEL_END_YEAR, SEED,
)

OB_PRODUCTS_CSV = DATA / "01 - Orange Book" / "output_data" / "products.csv"

_PARENTERAL_ROUTES = {
    "INJECTION", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS",
    "INJECTION, INTRAVENOUS", "INTRAVENOUS, SUBCUTANEOUS",
    "INTRAMUSCULAR, INTRAVENOUS", "INJECTABLE", "IRRIGATION",
    "INJECTION, SUBCUTANEOUS",
}
from utils import get_logger, write_table, ValisureDrugMatcher, load_valisure_api_names

log = get_logger("m14_recall_fei", OUT_LOGS / "m14_recall_fei.log")
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
    "parenteral_ever",   # derived from Orange Book dosage form/route
    "n_feis_drug",       # count of FEIs per drug (supply concentration)
]

ALL_FEATURES = INSP_FEATURES + TEXT_FEATURES + STRUCT_FEATURES

# Color mapping for feature groups (used in Fig A)
_GROUP_COLORS = {f: "steelblue" for f in INSP_FEATURES}
_GROUP_COLORS.update({f: "darkorange" for f in TEXT_FEATURES})
_GROUP_COLORS.update({f: "seagreen" for f in STRUCT_FEATURES})


# ── Data loading ────────────────────────────────────────────────────────────

def _load_recall_fei_year() -> pd.DataFrame:
    """Build FEI × year recall counts from the raw filtered recall events."""
    df = pd.read_csv(RECALL_FILT, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    fei_col  = next((c for c in df.columns if "fei" in c.lower()), None)
    date_col = "Recall_Date"
    cls_col  = next((c for c in df.columns if "classification" in c.lower() and "event" in c.lower()), None)
    if cls_col is None:
        cls_col = next((c for c in df.columns if "classification" in c.lower()), None)

    if fei_col is None or date_col not in df.columns:
        log.warning("Recall file missing FEI or date column; returning empty frame")
        return pd.DataFrame(columns=["fei", "year", "n_recalls", "n_class_i"])

    df["fei"]  = pd.to_numeric(df[fei_col], errors="coerce").astype("Int64")
    df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year.astype("Int64")
    df = df.dropna(subset=["fei", "year"])

    df["is_class_i"] = 0
    if cls_col:
        df["is_class_i"] = df[cls_col].astype(str).str.contains("Class I|Class-I|I$", regex=True).astype(int)

    agg = df.groupby(["fei", "year"], as_index=False).agg(
        n_recalls=("fei", "count"),
        n_class_i=("is_class_i", "sum"),
    )
    log.info("Recall events: %d rows, %d FEIs, years %s–%s",
             len(df), df["fei"].nunique(), int(df["year"].min()), int(df["year"].max()))
    return agg


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
    fei_year["n_oai_cumul"] = (
        fei_year.groupby("fei")["n_oai_t"].cumsum()
    )
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
    """Return set of API names (as in Valisure FEI map) that have ≥1 parenteral ANDA in OB.

    Matches by checking whether the first meaningful token of each Valisure API name
    (before any semicolon, e.g. "Ampicillin" from "Ampicillin; Sulbactam") appears as
    a word in any OB Ingredient with a parenteral route.
    Falls back to a known-correct hardcoded set if the OB file is unavailable.
    """
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

    # Load Valisure API list
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    api_col = next((c for c in fei_map.columns if c.lower() == "api"), None)
    if api_col is None:
        return fallback
    all_apis = fei_map[api_col].dropna().astype(str).str.strip().unique()

    parenteral = set()
    for api in all_apis:
        # First word of each component (handles "Ampicillin; Sulbactam" → ["AMPICILLIN","SULBACTAM"])
        parts = [p.strip().upper().split()[0] for p in api.split(";") if p.strip()]
        for ing in par_ingredients:
            ing_words = ing.split()
            if any(part in ing_words for part in parts):
                parenteral.add(api)
                break

    log.info("Parenteral APIs from Orange Book: %s", sorted(parenteral))
    return parenteral


def _load_structural_features() -> pd.DataFrame:
    """Load parenteral_ever and n_feis_drug from Valisure FEI map + Orange Book.

    parenteral_ever: 1 if this FEI produces any drug with a parenteral (injectable/IV)
        dosage form in the Orange Book.  Derived from OB DF;Route column — not hardcoded.
    n_feis_drug: number of FEIs in the Valisure mapping that produce the same drug.
        Proxy for supply concentration: n_feis_drug=1 means sole-source (highest risk).
    """
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

    # n_feis_drug: count of distinct FEIs per drug in Valisure mapping
    api_fei_counts = fm.groupby("api")["fei"].nunique()
    fm["n_feis_drug"] = fm["api"].map(api_fei_counts)

    out = fm.groupby("fei", as_index=False).agg(
        parenteral_ever=("parenteral_ever", "max"),
        n_feis_drug=("n_feis_drug",     "min"),  # most concentrated drug this FEI makes
    )
    log.info(
        "Structural features: %d FEIs, parenteral_ever=%d, n_feis_drug range=%d–%d",
        len(out), int(out["parenteral_ever"].sum()),
        int(out["n_feis_drug"].min()), int(out["n_feis_drug"].max()),
    )
    return out


def _load_facility_names() -> pd.DataFrame:
    """Load facility legal names from Inspections Details if available."""
    from config import DATA
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
    """Build the FEI × year modeling panel."""
    recall_fy = _load_recall_fei_year()
    redica_fy = _load_redica_fei_year()
    redica_fy = _add_cumulative_oai(redica_fy)
    text_ts   = _load_text_features()
    struct    = _load_structural_features()

    # Universe: all FEIs present in Redica data, crossed with all panel years
    all_feis  = redica_fy["fei"].dropna().unique()
    years     = range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
    panel = pd.MultiIndex.from_product([all_feis, years], names=["fei", "year"])
    panel = pd.DataFrame(index=panel).reset_index()
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

    # Outcome: ≥1 recall in year t+1
    recall_next = recall_fy.rename(columns={"year": "year_next", "n_recalls": "n_recalls_next",
                                            "n_class_i": "n_class_i_next"})
    recall_next["year"] = recall_next["year_next"] - 1
    panel = panel.merge(
        recall_next[["fei", "year", "n_recalls_next", "n_class_i_next"]],
        on=["fei", "year"], how="left",
    )
    panel["y_recall_next"] = (panel["n_recalls_next"].fillna(0) >= 1).astype(int)

    # Fill inspection zeros (FEI-years with no Redica event = 0 inspections)
    for col in ["n_inspections_t", "n_vai_t", "n_warning_letters_t"]:
        panel[col] = panel[col].fillna(0)
    panel["n_oai_cumul"] = panel["n_oai_cumul"].fillna(0)

    log.info(
        "Panel: %d rows, %d FEIs, %d years | recall events: %d (%.1f%%)",
        len(panel), panel["fei"].nunique(), panel["year"].nunique(),
        int(panel["y_recall_next"].sum()),
        100 * panel["y_recall_next"].mean(),
    )
    return panel


# ── Modeling ────────────────────────────────────────────────────────────────

def _prep(panel: pd.DataFrame, features: list[str]):
    df = panel.dropna(subset=["y_recall_next"]).copy()
    feats_in = [f for f in features if f in df.columns]
    missing  = set(features) - set(feats_in)
    if missing:
        log.warning("Features missing (dropped): %s", sorted(missing))
    X      = df[feats_in].fillna(0).astype(float)
    y      = df["y_recall_next"].astype(int)
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
    auc = roc_auc_score(y, preds)  if y.sum() > 0 else float("nan")
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
    """Fig B — ROC curves for both models."""
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
        ax.set_title("ROC — Predict FEI recall at year t+1")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "roc_recall_fei.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved roc_recall_fei.png")


def _fig_feature_importance(fi: pd.DataFrame, top_n: int = 12):
    """Fig A — Horizontal bar chart, color-coded by feature group."""
    fi = fi.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    colors = [_GROUP_COLORS.get(f, "gray") for f in fi["feature"]]

    with plt.rc_context(_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 0.45 * top_n + 1))
        ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="none")
        ax.set_xlabel("Random Forest Feature Importance")
        ax.set_title(f"Top {top_n} Features — Recall Prediction (FEI level)")

        legend_patches = [
            mpatches.Patch(color="steelblue",  label="Inspection"),
            mpatches.Patch(color="darkorange", label="Text/LLM"),
            mpatches.Patch(color="seagreen",   label="Structural"),
        ]
        ax.legend(handles=legend_patches, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "feature_importance_recall_fei.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved feature_importance_recall_fei.png")


def _fig_risk_ranking(panel: pd.DataFrame, rf_full, feature_cols: list[str],
                      facility_names: pd.DataFrame):
    """Fig C — Top-20 FEI risk ranking table."""
    latest = (
        panel.dropna(subset=["y_recall_next"])
             .sort_values("year")
             .groupby("fei")
             .last()
             .reset_index()
    )
    X_all = latest[feature_cols].fillna(0).astype(float)
    latest["p_recall"] = rf_full.predict_proba(X_all)[:, 1]

    top20 = latest.sort_values("p_recall", ascending=False).head(20).copy()
    top20 = top20.merge(facility_names, on="fei", how="left")

    rank_cols = ["fei", "facility_name", "p_recall", "n_oai_cumul",
                 "severity_critmajor_share", "contamination_llm_share"]
    rank_cols_present = [c for c in rank_cols if c in top20.columns]
    top20 = top20[rank_cols_present].reset_index(drop=True)
    top20.index = top20.index + 1

    top20.to_csv(OUT_TABS / "fei_risk_ranking.csv", index=True, index_label="rank")
    log.info("Saved fei_risk_ranking.csv (%d rows)", len(top20))

    # Table as figure
    display_cols = [c for c in rank_cols_present if c != "fei"]
    cell_text = []
    for _, row in top20.iterrows():
        cell_row = []
        for c in display_cols:
            v = row[c]
            if pd.isna(v):
                cell_row.append("—")
            elif isinstance(v, float):
                cell_row.append(f"{v:.3f}")
            else:
                cell_row.append(str(v))
        cell_text.append(cell_row)

    col_labels = [c.replace("_", " ").title() for c in display_cols]
    n_rows = len(cell_text)

    with plt.rc_context(_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(col_labels)), 0.35 * n_rows + 1.2))
        ax.axis("off")
        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            rowLabels=[str(i + 1) for i in range(n_rows)],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(True)
        tbl.scale(1, 1.3)
        ax.set_title("Top 20 Highest-Risk FEIs (RF predicted recall probability)", pad=12)
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "fei_risk_ranking.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved fei_risk_ranking.png")


def _fig_text_lift(ablation_rows: list[dict]):
    """Fig D — AUC with vs without 483 text features (ablation bar chart)."""
    labels = [r["label"] for r in ablation_rows]
    aucs   = [r["auc"]   for r in ablation_rows]
    colors = ["#7B9EC9", "#D4896A", "#D4896A"]  # first bar = no text; rest = with text

    with plt.rc_context(_FIG_STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, aucs, color=colors, width=0.5, edgecolor="white")
        ax.set_ylim(max(0, min(aucs) - 0.1), min(1, max(aucs) + 0.1))
        ax.set_ylabel("AUC-ROC (GroupKFold CV)")
        ax.set_title("483 Text Feature Lift — Recall Prediction")
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{auc:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_FIGS / "text_feature_lift_recall_fei.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved text_feature_lift_recall_fei.png")


# ── Panel summary table ─────────────────────────────────────────────────────

def _write_panel_summary(panel: pd.DataFrame) -> None:
    modeled = panel.dropna(subset=["y_recall_next"])
    lines = [
        "# Recall FEI Panel Summary",
        "",
        f"- **FEI × year rows (full panel):** {len(panel):,}",
        f"- **Unique FEIs:** {panel['fei'].nunique()}",
        f"- **Years:** {int(panel['year'].min())}–{int(panel['year'].max())}",
        f"- **Rows used in modeling:** {len(modeled):,}",
        f"- **Recall events (y=1):** {int(modeled['y_recall_next'].sum())} "
          f"({100*modeled['y_recall_next'].mean():.1f}%)",
        f"- **FEIs with ≥1 recall event in panel:** "
          f"{int((modeled.groupby('fei')['y_recall_next'].max() == 1).sum())}",
        "",
        "## Feature coverage",
        f"- Inspection features: {sum(modeled[f].notna().any() for f in INSP_FEATURES)}/{len(INSP_FEATURES)} present",
        f"- Text/LLM features:   {sum(modeled[f].notna().any() for f in TEXT_FEATURES)}/{len(TEXT_FEATURES)} present",
        f"- Structural features: {sum(modeled[f].notna().any() for f in STRUCT_FEATURES)}/{len(STRUCT_FEATURES)} present",
    ]
    out_path = OUT_TABS / "recall_fei_panel_summary.md"
    out_path.write_text("\n".join(lines))
    log.info("Saved recall_fei_panel_summary.md")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not _SKLEARN:
        log.warning("scikit-learn not installed; skipping m14. Install with: pip install scikit-learn")
        return

    panel = build_panel()
    write_table(panel, OUT_DATA / "recall_fei_panel.parquet", log)
    _write_panel_summary(panel)

    X_all, y, groups, df_model = _prep(panel, ALL_FEATURES)
    X_insp, y_i, g_i, _       = _prep(panel, INSP_FEATURES + STRUCT_FEATURES)
    X_38,   y_38, g_38, _     = _prep(
        panel[panel[TEXT_FEATURES[0]].notna()], ALL_FEATURES
    )

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

    # Save metrics
    pd.DataFrame([
        {"model": "L2_logit",     **met_l2},
        {"model": "RandomForest", **met_rf},
    ]).to_csv(OUT_MODELS / "metrics_recall_fei.csv", index=False)

    # ── Ablation: no text features ──────────────────────────────────────────
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

    # Use L2 logistic for ablation — it's the stronger model with this sample size
    ablation_rows = [
        {"label": "Without text", "auc": met_no_text["auc"], "model": "L2"},
        {"label": "With text\n(98 FEIs)", "auc": met_l2["auc"],  "model": "L2"},
    ]
    pd.DataFrame(ablation_rows).to_csv(OUT_MODELS / "text_ablation_recall_fei.csv", index=False)
    _fig_text_lift(ablation_rows)

    # ── Figures ─────────────────────────────────────────────────────────────
    _fig_roc(preds_l2, preds_rf, met_l2, met_rf, y)

    rf_full = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                     class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_full.fit(X_all, y)

    fi = pd.DataFrame({"feature": X_all.columns, "importance": rf_full.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(OUT_MODELS / "rf_importance_recall_fei.csv", index=False)
    _fig_feature_importance(fi)

    facility_names = _load_facility_names()
    _fig_risk_ranking(df_model, rf_full, list(X_all.columns), facility_names)

    log.info("m14 complete — outputs in %s", OUT_MODELS)


if __name__ == "__main__":
    main()
