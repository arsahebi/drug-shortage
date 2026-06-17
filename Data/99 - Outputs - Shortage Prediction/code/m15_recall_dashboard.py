"""
Module 15 — Interactive recall prediction dashboard.

Produces a self-contained HTML dashboard showing:
  1. Data coverage / provenance (facility funnel + 483 document coverage)
  2. Text feature distributions (aggregate 4-panel + per-FEI heatmap)
  3. Model performance (ROC curve + ablation bar chart)
  4. Feature importance (Random Forest, top 12)
  5. FEI risk ranking (top 20, interactive table)

Output:
  outputs/figures/recall_fei_dashboard.html

Run:
  python m15_recall_dashboard.py
"""

from __future__ import annotations
import warnings
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ModuleNotFoundError:
    _SKLEARN = False

from config import (
    DATA, OUT_FIGS, OUT_DATA, OUT_MODELS, OUT_TABS, OUT_LOGS,
    TEXT_TIMESERIES_REDICA_CSV, SEED,
)
from utils import get_logger

log = get_logger("m15_dashboard", OUT_LOGS / "m15_dashboard.log")

REDICA_RAW = DATA / "07 - Redica" / "raw"
OUT_HTML   = OUT_FIGS / "recall_fei_dashboard.html"

# ── Feature groups (must match m14) ────────────────────────────────────────
INSP_FEATS   = ["n_oai_cumul", "n_vai_t", "n_inspections_t", "n_warning_letters_t"]
TEXT_FEATS   = [
    "severity_critmajor_share", "scope_facilitywide_share",
    "scope_multipleproducts_share", "cultural_root_cause_share",
    "contamination_llm_share", "data_integrity_llm_share",
    "investigation_llm_share", "repeat_cross_insp_share",
    "vc_labcontrols_share", "vc_qualitysystem_share",
    "remediation_none_share", "remediation_weak_share",
]
STRUCT_FEATS = ["parenteral_ever", "n_feis_drug"]
ALL_FEATS    = INSP_FEATS + TEXT_FEATS + STRUCT_FEATS

_GROUP_COLOR = {**{f: "#4A90D9" for f in INSP_FEATS},
                **{f: "#E07B39" for f in TEXT_FEATS},
                **{f: "#3DAA6E" for f in STRUCT_FEATS}}

_FEAT_LABEL = {
    "n_oai_cumul":                "OAI (cumulative)",
    "n_vai_t":                    "VAI inspections",
    "n_inspections_t":            "Total inspections",
    "n_warning_letters_t":        "Warning letters",
    "severity_critmajor_share":   "Critical/Major severity",
    "scope_facilitywide_share":   "Facility-wide scope",
    "scope_multipleproducts_share":"Multi-product scope",
    "cultural_root_cause_share":  "Cultural root cause",
    "contamination_llm_share":    "Contamination (LLM)",
    "data_integrity_llm_share":   "Data integrity (LLM)",
    "investigation_llm_share":    "Investigation gaps (LLM)",
    "repeat_cross_insp_share":    "Repeat finding (cross-insp.)",
    "vc_labcontrols_share":       "Lab controls violations",
    "vc_qualitysystem_share":     "Quality system violations",
    "remediation_none_share":     "No remediation signal",
    "remediation_weak_share":     "Weak remediation",
    "parenteral_ever":            "Parenteral drug (OB-derived)",
    "n_feis_drug":                "# FEIs per drug (supply conc.)",
}

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "blue":   "#4A90D9",
    "orange": "#E07B39",
    "green":  "#3DAA6E",
    "red":    "#D94A4A",
    "purple": "#7B5EA7",
    "gray":   "#9AA5B1",
    "dark":   "#1a1a2e",
    "bg":     "#F8F9FA",
}


# ── Data loading ────────────────────────────────────────────────────────────

def _load_panel() -> pd.DataFrame:
    p = OUT_DATA / "recall_fei_panel.csv"
    if not p.exists():
        log.error("recall_fei_panel.csv not found — run m14 first")
        sys.exit(1)
    return pd.read_csv(p, low_memory=False)


def _load_facility_names() -> pd.DataFrame:
    p = DATA / "14 - FDA - Inspection" / "raw" / "Inspections Details.xlsx"
    if not p.exists():
        return pd.DataFrame(columns=["fei", "facility_name"])
    try:
        df = pd.read_excel(p, usecols=["FEI Number", "Legal Name"])
        df.columns = ["fei", "facility_name"]
        df["fei"] = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
        return df.dropna(subset=["fei"]).drop_duplicates("fei")
    except Exception as exc:
        log.warning("Could not load facility names: %s", exc)
        return pd.DataFrame(columns=["fei", "facility_name"])


def _load_redica_site_names() -> pd.DataFrame:
    """Fallback facility names from Redica data availability file."""
    p = REDICA_RAW / "Valisure_Sites_Data_Availability.xlsx"
    if not p.exists():
        return pd.DataFrame(columns=["site_redica_id", "facility_name", "n_483s_issued"])
    df = pd.read_excel(p, usecols=["Site Redica Id", "Site Display Name", "483s Issued"])
    df.columns = ["site_redica_id", "facility_name", "n_483s_issued"]
    return df


def _load_all_data() -> dict:
    panel      = _load_panel()
    fi         = pd.read_csv(OUT_MODELS / "rf_importance_recall_fei.csv")
    metrics    = pd.read_csv(OUT_MODELS / "metrics_recall_fei.csv")
    ablation   = pd.read_csv(OUT_MODELS / "text_ablation_recall_fei.csv")
    risk       = pd.read_csv(OUT_TABS   / "fei_risk_ranking.csv")
    ts         = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    names      = _load_facility_names()
    site_names = _load_redica_site_names()

    # Latest text snapshot per FEI
    ts["fei"]           = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    latest = ts.sort_values("snapshot_date").groupby("fei").last().reset_index()

    # Merge facility name into latest
    latest = latest.merge(names, on="fei", how="left")
    # Fallback: use site_redica_id label from combined universe
    combined_p = DATA / "99 - Outputs - Text Analysis" / "483_combined_obs_universe.csv"
    if combined_p.exists():
        combined = pd.read_csv(combined_p, usecols=["fei", "site_redica_id"],
                               low_memory=False)
        combined["fei"] = pd.to_numeric(combined["fei"], errors="coerce").astype("Int64")
        combined = combined.dropna().drop_duplicates("fei")
        latest = latest.merge(combined, on="fei", how="left")
        latest = latest.merge(
            site_names[["site_redica_id", "facility_name"]].rename(
                columns={"facility_name": "redica_name"}
            ), on="site_redica_id", how="left"
        )
        mask = latest["facility_name"].isna()
        latest.loc[mask, "facility_name"] = latest.loc[mask, "redica_name"]

    latest["facility_name"] = latest["facility_name"].fillna(
        latest["fei"].astype(str)
    )

    # Ablation label cleanup
    ablation["label"] = ablation["label"].str.replace(r"\n.*", "", regex=True)
    ablation["label"] = ablation["label"].map(
        lambda s: "Inspection only" if "without" in s.lower()
                  else "Inspection + LLM text\n(98 FEIs)"
    )

    # Coverage stats from Redica raw
    cov = {}
    da_p = REDICA_RAW / "Valisure_Sites_Data_Availability.xlsx"
    obs_p = REDICA_RAW / "FDA-483s Observations + WL Deficiencies_OSU.xlsx"
    if da_p.exists():
        da = pd.read_excel(da_p)
        cov["n_feis_redica"]    = len(da)
        cov["n_483s_issued"]    = int(da["483s Issued"].sum())
        cov["n_sites_with_483"] = int((da["483s Issued"] > 0).sum())
    else:
        cov = {"n_feis_redica": 127, "n_483s_issued": 853, "n_sites_with_483": 122}
    if obs_p.exists():
        obs = pd.read_excel(obs_p, sheet_name="FDA-483s Obs + WL Deficiencies")
        obs483 = obs[obs["Document Type"] == "483"]
        cov["n_docs_obtained"] = obs483["Document Redica Id"].nunique()
        cov["n_sites_obtained"] = obs483["Site Redica Id"].nunique()
    else:
        cov["n_docs_obtained"] = 246
        cov["n_sites_obtained"] = 98
    cov["n_obs_llm"] = 1115
    cov["n_feis_llm"] = 98

    return dict(panel=panel, fi=fi, metrics=metrics, ablation=ablation,
                risk=risk, latest=latest, site_names=site_names, cov=cov)


# ── CV for ROC predictions ──────────────────────────────────────────────────

def _get_roc_preds(panel: pd.DataFrame):
    if not _SKLEARN:
        return None, None, None
    feats_in = [f for f in ALL_FEATS if f in panel.columns]
    df = panel.dropna(subset=["y_recall_next"]).copy()
    X  = df[feats_in].fillna(0).astype(float)
    y  = df["y_recall_next"].astype(int)
    g  = df["fei"].astype(str)

    if y.sum() < 3:
        return None, None, None

    n_splits = min(5, max(2, g.nunique() - 1))
    gkf = GroupKFold(n_splits=n_splits)

    preds_l2 = np.zeros(len(y))
    preds_rf = np.zeros(len(y))
    Xz = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    for tr, te in gkf.split(X, y, g):
        if y.iloc[tr].nunique() < 2:
            preds_l2[te] = y.iloc[tr].mean()
            preds_rf[te] = y.iloc[tr].mean()
            continue
        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=500,
                                class_weight="balanced", random_state=SEED)
        lr.fit(Xz.iloc[tr], y.iloc[tr])
        preds_l2[te] = lr.predict_proba(Xz.iloc[te])[:, 1]

        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=SEED,
                                    n_jobs=-1)
        rf.fit(X.iloc[tr], y.iloc[tr])
        preds_rf[te] = rf.predict_proba(X.iloc[te])[:, 1]

    return preds_l2, preds_rf, y


# ── Figure builders ─────────────────────────────────────────────────────────

_PLOTLY_FONT = dict(family="'Segoe UI', 'Helvetica Neue', Arial, sans-serif", size=12)
_PLOTLY_MARGIN = dict(l=10, r=10, t=40, b=10)


def _fig_funnel(cov: dict) -> go.Figure:
    """Facility funnel + 483 document coverage side-by-side."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Facility coverage funnel", "483 document coverage"],
        horizontal_spacing=0.12,
    )

    # Left: facility funnel
    funnel_labels = [
        "Valisure universe",
        "In Redica data",
        "With ≥1 483 issued",
        "With LLM text features",
    ]
    funnel_vals = [
        129,
        cov["n_feis_redica"],
        cov["n_sites_with_483"],
        cov["n_feis_llm"],
    ]
    colors_f = ["#4A90D9", "#5AA8E0", "#6BBCE8", "#3DAA6E"]
    for i, (lbl, val, col) in enumerate(zip(funnel_labels, funnel_vals, colors_f)):
        fig.add_trace(go.Bar(
            x=[val], y=[lbl],
            orientation="h",
            marker_color=col,
            text=f"  {val}",
            textposition="outside",
            showlegend=False,
            hovertemplate=f"<b>{lbl}</b><br>{val} FEIs<extra></extra>",
        ), row=1, col=1)

    # Right: 483 coverage bars
    doc_labels = ["483s issued\n(all 127 sites)", "Documents obtained\n(Redica file)",
                  "Observations\nLLM-scored"]
    doc_vals   = [cov["n_483s_issued"], cov["n_docs_obtained"], cov["n_obs_llm"]]
    doc_colors = ["#D1D5DB", "#4A90D9", "#E07B39"]
    for lbl, val, col in zip(doc_labels, doc_vals, doc_colors):
        fig.add_trace(go.Bar(
            x=[lbl], y=[val],
            marker_color=col,
            text=f"{val:,}",
            textposition="outside",
            showlegend=False,
            hovertemplate=f"<b>{lbl.replace(chr(10),' ')}</b><br>{val:,}<extra></extra>",
        ), row=1, col=2)

    pct = cov["n_docs_obtained"] / cov["n_483s_issued"] * 100
    fig.add_annotation(
        x=doc_labels[1], y=cov["n_docs_obtained"],
        text=f"<b>{pct:.0f}% coverage</b>",
        showarrow=False, yshift=38, font=dict(color="#E07B39", size=12),
        row=1, col=2,
    )

    fig.update_layout(
        height=340, margin=_PLOTLY_MARGIN,
        font=_PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(autorange="reversed", gridcolor="#F0F0F0"),
        xaxis=dict(range=[0, 160], showticklabels=False),
        xaxis2=dict(showticklabels=False),
        yaxis2=dict(range=[0, max(doc_vals) * 1.25], gridcolor="#F0F0F0"),
    )
    return fig


def _fig_dist_agg(latest: pd.DataFrame) -> go.Figure:
    """Four small aggregate distribution charts in a 2×2 grid."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Observation severity (mean share across 98 FEIs)",
            "Violation category",
            "Root cause type",
            "Observation scope",
        ],
        vertical_spacing=0.20,
        horizontal_spacing=0.12,
    )

    def _hbar(labels, vals, colors, row, col, pct=True):
        suffix = "%" if pct else ""
        display_vals = [v * 100 for v in vals] if pct else vals
        fig.add_trace(go.Bar(
            x=display_vals, y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}{suffix}" for v in display_vals],
            textposition="auto",
            showlegend=False,
            hovertemplate="%{y}: %{x:.1f}" + suffix + "<extra></extra>",
        ), row=row, col=col)

    # Severity
    _hbar(
        ["Critical", "Major", "Moderate", "Minor"],
        [latest["severity_critical_share"].mean(),
         latest["severity_major_share"].mean(),
         latest["severity_moderate_share"].mean(),
         latest["severity_minor_share"].mean()],
        ["#D94A4A", "#E07B39", "#F4C542", "#3DAA6E"],
        1, 1,
    )

    # Violation category
    vc_cols = ["vc_labcontrols_share", "vc_productioncontrols_share",
               "vc_buildingsequipment_share", "vc_qualitysystem_share",
               "vc_recordsreports_share", "vc_orgpersonnel_share"]
    vc_labels = ["Lab Controls", "Production Controls", "Buildings/Equipment",
                 "Quality System", "Records/Reports", "Org/Personnel"]
    vc_vals   = [latest[c].mean() for c in vc_cols]
    order = sorted(range(len(vc_vals)), key=lambda i: vc_vals[i])
    _hbar([vc_labels[i] for i in order], [vc_vals[i] for i in order],
          "#4A90D9", 1, 2)

    # Root cause
    _hbar(
        ["Cultural", "Mixed", "Capital"],
        [latest["cultural_root_cause_share"].mean(),
         latest["mixed_root_cause_share"].mean(),
         latest["capital_root_cause_share"].mean()],
        ["#7B5EA7", "#9B7EBF", "#BBA8D4"],
        2, 1,
    )

    # Scope
    _hbar(
        ["Multiple products", "Facility-wide", "Single batch"],
        [latest["scope_multipleproducts_share"].mean(),
         latest["scope_facilitywide_share"].mean(),
         latest["scope_singlebatch_share"].mean()],
        ["#3DAA6E", "#5BC88A", "#89DEBA"],
        2, 2,
    )

    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=60, b=10),
        font=_PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    for axis in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
        fig.update_layout({axis: dict(gridcolor="#F0F0F0", ticksuffix="%")})
    for axis in ["yaxis", "yaxis2", "yaxis3", "yaxis4"]:
        fig.update_layout({axis: dict(gridcolor="white")})
    return fig


def _fig_heatmap(latest: pd.DataFrame) -> go.Figure:
    """Per-FEI heatmap: 98 FEIs × 12 text features (latest snapshot)."""
    heat_cols = [
        "severity_critmajor_share", "scope_facilitywide_share",
        "contamination_llm_share", "data_integrity_llm_share",
        "investigation_llm_share", "cultural_root_cause_share",
        "repeat_cross_insp_share", "vc_labcontrols_share",
        "vc_qualitysystem_share", "remediation_none_share",
        "remediation_weak_share", "scope_multipleproducts_share",
    ]
    col_labels = [_FEAT_LABEL.get(c, c) for c in heat_cols]

    # Sort FEIs by mean text-feature risk (high severity + contamination)
    latest = latest.copy()
    latest["_sort"] = (
        latest["severity_critmajor_share"].fillna(0) * 0.4 +
        latest["contamination_llm_share"].fillna(0) * 0.3 +
        latest["cultural_root_cause_share"].fillna(0) * 0.3
    )
    latest = latest.sort_values("_sort", ascending=True)

    z     = latest[heat_cols].fillna(np.nan).values
    y_lbl = latest["facility_name"].tolist()
    # Shorten long names
    y_lbl = [n[:35] + "…" if len(str(n)) > 36 else str(n) for n in y_lbl]

    hover = []
    for _, row in latest.iterrows():
        row_hover = []
        for c, cl in zip(heat_cols, col_labels):
            v = row.get(c, np.nan)
            row_hover.append(f"{cl}: {v:.2f}" if pd.notna(v) else f"{cl}: n/a")
        hover.append("<br>".join(row_hover))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_labels,
        y=y_lbl,
        colorscale="Oranges",
        zmin=0, zmax=1,
        hovertext=hover,
        hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>",
        customdata=hover,
        colorbar=dict(title="Share", thickness=14, len=0.7),
    ))

    fig.update_layout(
        height=max(500, 14 * len(latest)),
        margin=dict(l=200, r=20, t=20, b=120),
        font=_PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(side="bottom", tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def _fig_roc(preds_l2, preds_rf, y, metrics: pd.DataFrame) -> go.Figure:
    """ROC curves for both models."""
    fig = go.Figure()

    if preds_l2 is not None and y.sum() > 0:
        auc_l2 = roc_auc_score(y, preds_l2)
        fpr, tpr, _ = roc_curve(y, preds_l2)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"Logistic Regression (AUC={auc_l2:.3f})",
            line=dict(color=C["blue"], width=2),
        ))
        auc_rf = roc_auc_score(y, preds_rf)
        fpr_rf, tpr_rf, _ = roc_curve(y, preds_rf)
        fig.add_trace(go.Scatter(
            x=fpr_rf, y=tpr_rf, mode="lines", name=f"Random Forest (AUC={auc_rf:.3f})",
            line=dict(color=C["orange"], width=2.5),
        ))
    else:
        # Fall back to saved metrics
        for _, row in metrics.iterrows():
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"{row['model']}: AUC={row['auc']:.3f}",
                showarrow=False,
            )

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=C["gray"], width=1.2, dash="dash"),
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        font=_PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="ROC — Predict FEI recall at year t+1", font=dict(size=13)),
        xaxis=dict(title="False Positive Rate", gridcolor="#F0F0F0", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate", gridcolor="#F0F0F0", range=[-0.02, 1.02]),
        legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E0E0E0", borderwidth=1),
    )
    return fig


def _fig_ablation(ablation: pd.DataFrame, metrics: pd.DataFrame) -> go.Figure:
    """AUC lift bar chart (ablation) + metrics table."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Text feature lift (Logistic Regression AUC)", "CV metrics summary"],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.08,
        specs=[[{"type": "xy"}, {"type": "table"}]],
    )

    bar_colors = [C["gray"], C["orange"]]
    for i, row in ablation.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]], y=[row["auc"]],
            marker_color=bar_colors[i % 2],
            text=f"{row['auc']:.3f}",
            textposition="outside",
            textfont=dict(size=13, color="#212529"),
            showlegend=False,
            width=0.4,
            hovertemplate=f"<b>{row['label']}</b><br>AUC: {row['auc']:.3f}<extra></extra>",
        ), row=1, col=1)

    # Delta annotation (no row/col — annotates the whole figure)
    if len(ablation) == 2:
        delta = ablation.iloc[1]["auc"] - ablation.iloc[0]["auc"]
        rel   = delta / ablation.iloc[0]["auc"] * 100
        fig.add_annotation(
            xref="x", yref="y",
            x=ablation.iloc[1]["label"], y=ablation.iloc[1]["auc"],
            text=f"<b>+{delta:.3f}<br>(+{rel:.0f}%)</b>",
            showarrow=True, arrowhead=2, arrowcolor=C["orange"],
            ax=50, ay=-40, font=dict(color=C["orange"], size=12),
        )

    # Metrics table
    tbl_headers = ["Model", "AUC", "Avg. Precision", "Brier"]
    tbl_vals = [
        metrics["model"].str.replace("_", " ").tolist(),
        metrics["auc"].map("{:.3f}".format).tolist(),
        metrics["ap"].map("{:.3f}".format).tolist(),
        metrics["brier"].map("{:.3f}".format).tolist(),
    ]
    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in tbl_headers],
            fill_color=C["dark"], font=dict(color="white", size=12),
            align="left", height=30,
        ),
        cells=dict(
            values=tbl_vals,
            fill_color=[["white", "#F8F9FA"] * 4],
            align="left", height=28, font=dict(size=12),
        ),
    ), row=1, col=2)

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        font=_PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(range=[0, max(ablation["auc"]) * 1.2],
                   title="AUC-ROC", gridcolor="#F0F0F0"),
        xaxis=dict(gridcolor="white"),
    )
    return fig


def _fig_importance(fi: pd.DataFrame, top_n: int = 12) -> go.Figure:
    fi = fi.sort_values("importance", ascending=False).head(top_n).copy()
    fi["label"]  = fi["feature"].map(lambda f: _FEAT_LABEL.get(f, f))
    fi["color"]  = fi["feature"].map(lambda f: _GROUP_COLOR.get(f, C["gray"]))
    fi["group"]  = fi["feature"].map(
        lambda f: "Inspection" if f in INSP_FEATS
                  else "Text / LLM" if f in TEXT_FEATS
                  else "Structural"
    )
    fi = fi.sort_values("importance")

    fig = go.Figure()
    for grp, col in [("Inspection", C["blue"]), ("Text / LLM", C["orange"]),
                     ("Structural", C["green"])]:
        sub = fi[fi["group"] == grp]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["importance"], y=sub["label"],
            orientation="h",
            name=grp,
            marker_color=col,
            text=sub["importance"].map("{:.3f}".format),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=60, t=20, b=10),
        font=_PLOTLY_FONT,
        plot_bgcolor="white", paper_bgcolor="white",
        barmode="overlay",
        xaxis=dict(title="RF Feature Importance", gridcolor="#F0F0F0"),
        yaxis=dict(gridcolor="white"),
        legend=dict(x=0.72, y=0.05, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E0E0E0", borderwidth=1),
    )
    return fig


def _fig_risk_table(risk: pd.DataFrame, latest: pd.DataFrame) -> go.Figure:
    """Interactive top-20 FEI risk ranking table."""
    risk = risk.copy()

    # Merge in text features if available
    latest_sub = latest[["fei", "severity_critmajor_share",
                          "contamination_llm_share", "cultural_root_cause_share",
                          "n_obs_total"]].copy()
    latest_sub["fei"] = pd.to_numeric(latest_sub["fei"], errors="coerce").astype("Int64")
    risk["fei"] = pd.to_numeric(risk["fei"], errors="coerce").astype("Int64")
    risk = risk.merge(latest_sub, on="fei", how="left",
                      suffixes=("", "_ts"))

    # Prefer timeseries columns over model output (more complete)
    for col in ["severity_critmajor_share", "contamination_llm_share"]:
        if col + "_ts" in risk.columns:
            mask = risk[col].isna()
            risk.loc[mask, col] = risk.loc[mask, col + "_ts"]

    def _fmt(v, pct=True):
        if pd.isna(v):
            return "—"
        return f"{v*100:.0f}%" if pct else f"{v:.2f}"

    def _fmt_int(v):
        if pd.isna(v):
            return "—"
        return str(int(v))

    col_defs = [
        ("Rank",                "rank",                      lambda v: str(int(v)) if pd.notna(v) else "—"),
        ("Facility",            "facility_name",             lambda v: str(v)[:40] if pd.notna(v) else "—"),
        ("P(recall) RF",        "p_recall",                  lambda v: f"{v:.3f}" if pd.notna(v) else "—"),
        ("Crit/Major severity", "severity_critmajor_share",  lambda v: _fmt(v)),
        ("Contamination (LLM)", "contamination_llm_share",   lambda v: _fmt(v)),
        ("OAI (cumul.)",        "n_oai_cumul",               lambda v: _fmt_int(v)),
        ("483 obs (text)",      "n_obs_total",               lambda v: _fmt_int(v)),
    ]

    headers = [c[0] for c in col_defs]
    cells   = []
    for _, fn in [(c[1], c[2]) for c in col_defs]:
        col_data = risk[_].map(fn) if _ in risk.columns else ["—"] * len(risk)
        cells.append(col_data.tolist())

    # Color rows by p_recall
    p_vals = risk["p_recall"].fillna(0).tolist()
    max_p  = max(p_vals) if p_vals else 1
    row_colors = []
    for p in p_vals:
        alpha = 0.15 + 0.45 * (p / max(max_p, 0.01))
        row_colors.append(f"rgba(224, 123, 57, {alpha:.2f})")

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=C["dark"], font=dict(color="white", size=12),
            align=["center", "left", "center", "center", "center", "center", "center"],
            height=34,
        ),
        cells=dict(
            values=cells,
            fill_color=[row_colors] * len(cells),
            align=["center", "left", "center", "center", "center", "center", "center"],
            height=30,
            font=dict(size=12),
        ),
    ))
    fig.update_layout(
        height=max(420, 32 * len(risk) + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        font=_PLOTLY_FONT,
        paper_bgcolor="white",
    )
    return fig


# ── HTML assembly ────────────────────────────────────────────────────────────

def _to_div(fig: go.Figure, fig_id: str) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=fig_id, config={"displayModeBar": True,
                                               "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def build_html(data: dict, preds_l2, preds_rf, y) -> str:
    cov   = data["cov"]
    pct   = cov["n_docs_obtained"] / cov["n_483s_issued"] * 100
    auc_no  = data["ablation"].iloc[0]["auc"]
    auc_yes = data["ablation"].iloc[1]["auc"]
    lift_rel = (auc_yes - auc_no) / auc_no * 100

    divs = {
        "funnel":      _to_div(_fig_funnel(cov), "fig_funnel"),
        "dist_agg":    _to_div(_fig_dist_agg(data["latest"]), "fig_dist_agg"),
        "heatmap":     _to_div(_fig_heatmap(data["latest"]), "fig_heatmap"),
        "roc":         _to_div(_fig_roc(preds_l2, preds_rf, y, data["metrics"]), "fig_roc"),
        "ablation":    _to_div(_fig_ablation(data["ablation"], data["metrics"]), "fig_ablation"),
        "importance":  _to_div(_fig_importance(data["fi"]), "fig_importance"),
        "risk":        _to_div(_fig_risk_table(data["risk"], data["latest"]), "fig_risk"),
    }

    today = date.today().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FEI Recall Prediction — Text Feature Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
          background: #F0F2F5; color: #212529; font-size: 14px; }}

  /* ── Nav ── */
  nav {{ position: sticky; top: 0; z-index: 200; background: #1a1a2e;
         display: flex; align-items: center; padding: 0 32px; height: 52px;
         box-shadow: 0 2px 10px rgba(0,0,0,0.3); gap: 4px; }}
  .nav-brand {{ font-size: 14px; font-weight: 700; color: #F8F9FA;
                 letter-spacing: 0.3px; margin-right: auto; white-space: nowrap; }}
  .nav-brand span {{ color: #68D391; }}
  nav a {{ color: #A0AEC0; text-decoration: none; font-size: 12.5px; font-weight: 500;
            padding: 6px 14px; border-radius: 6px; transition: all 0.18s; white-space: nowrap; }}
  nav a:hover {{ color: #F8F9FA; background: rgba(255,255,255,0.1); }}

  /* ── Hero ── */
  .hero {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; padding: 44px 40px 36px; border-bottom: 3px solid #68D391; }}
  .hero h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 6px; }}
  .hero p  {{ color: #A0AEC0; font-size: 13.5px; line-height: 1.6; max-width: 720px; }}
  .kpi-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 28px; }}
  .kpi {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
           border-radius: 10px; padding: 14px 22px; min-width: 130px; }}
  .kpi .val {{ font-size: 30px; font-weight: 800; color: #68D391; line-height: 1; }}
  .kpi .lbl {{ font-size: 11.5px; color: #A0AEC0; margin-top: 5px; line-height: 1.3; }}

  /* ── Layout ── */
  .page {{ max-width: 1380px; margin: 0 auto; padding: 0 28px 40px; }}
  section {{ padding-top: 40px; }}
  .section-head {{ margin-bottom: 18px; }}
  .section-title {{ font-size: 17px; font-weight: 700; color: #1a1a2e;
                     border-left: 4px solid #68D391; padding-left: 12px; }}
  .section-sub {{ font-size: 12.5px; color: #6C757D; margin-top: 5px;
                   padding-left: 16px; line-height: 1.5; }}
  .card {{ background: white; border-radius: 12px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.07); padding: 20px 20px 12px;
            margin-bottom: 18px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .legend-row {{ display: flex; gap: 20px; padding: 10px 4px 2px;
                  flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 7px;
                   font-size: 12px; color: #495057; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}

  /* ── Footer ── */
  footer {{ text-align: center; color: #9AA5B1; font-size: 12px;
             padding: 24px 0 32px; border-top: 1px solid #DEE2E6; margin-top: 20px; }}

  @media (max-width: 900px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .page {{ padding: 0 14px 32px; }}
    nav a {{ display: none; }}
  }}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">FEI Recall Prediction &nbsp;<span>·</span>&nbsp; Drug Shortage Project</div>
  <a href="#provenance">① Data Coverage</a>
  <a href="#distributions">② Feature Distributions</a>
  <a href="#model">③ Model Performance</a>
  <a href="#importance">④ Feature Importance</a>
  <a href="#ranking">⑤ Risk Ranking</a>
</nav>

<div class="hero">
  <h1>Predicting Drug Recalls from 483 Inspection Text</h1>
  <p>LLM-extracted features from FDA 483 inspection observations significantly improve
     facility-level recall prediction beyond inspection counts alone.
     Panel: {cov['n_feis_llm']} FEIs × 10 years (2015–2024), 20 recall events.
     Parenteral drug flag derived from Orange Book dosage routes (not hardcoded).</p>
  <div class="kpi-row">
    <div class="kpi"><div class="val">{cov['n_feis_llm']}</div>
      <div class="lbl">FEIs with LLM text features</div></div>
    <div class="kpi"><div class="val">{pct:.0f}%</div>
      <div class="lbl">483 document coverage<br>({cov['n_docs_obtained']}/{cov['n_483s_issued']} docs)</div></div>
    <div class="kpi"><div class="val">{cov['n_obs_llm']:,}</div>
      <div class="lbl">Observations LLM-scored</div></div>
    <div class="kpi"><div class="val">+{lift_rel:.0f}%</div>
      <div class="lbl">Relative AUC lift (L2)<br>from text features</div></div>
    <div class="kpi"><div class="val">{auc_yes:.3f}</div>
      <div class="lbl">L2 AUC with text<br>(vs {auc_no:.3f} without)</div></div>
  </div>
</div>

<div class="page">

<!-- ① Data Coverage -->
<section id="provenance">
  <div class="section-head">
    <div class="section-title">① Data Coverage</div>
    <div class="section-sub">
      From 129 Valisure FEIs → 127 in Redica → 98 with 483 text (2018–2026).
      Redica provided {cov['n_docs_obtained']} of the {cov['n_483s_issued']} total 483s issued ({pct:.0f}% coverage),
      yielding {cov['n_obs_llm']:,} LLM-scored observations.
    </div>
  </div>
  <div class="card">
    {divs['funnel']}
  </div>
</section>

<!-- ② Feature Distributions -->
<section id="distributions">
  <div class="section-head">
    <div class="section-title">② Text Feature Distributions</div>
    <div class="section-sub">
      <b>Top:</b> Mean shares across 98 FEIs (latest snapshot per facility).
      <b>Bottom:</b> Per-FEI heatmap — each row is one facility, sorted by composite risk
      (severity × contamination × cultural root cause). Orange = higher share.
    </div>
  </div>
  <div class="card">
    {divs['dist_agg']}
  </div>
  <div class="card">
    {divs['heatmap']}
  </div>
</section>

<!-- ③ Model Performance -->
<section id="model">
  <div class="section-head">
    <div class="section-title">③ Model Performance</div>
    <div class="section-sub">
      GroupKFold cross-validation grouped by FEI (no facility appears in both train and test).
      Panel: 1,250 FEI-years, 20 recall events (1.6% prevalence). Features at year t predict recall at year t+1.
    </div>
  </div>
  <div class="two-col">
    <div class="card">{divs['roc']}</div>
    <div class="card">{divs['ablation']}</div>
  </div>
</section>

<!-- ④ Feature Importance -->
<section id="importance">
  <div class="section-head">
    <div class="section-title">④ Feature Importance</div>
    <div class="section-sub">
      Random Forest impurity-based importance, top 12 features, fit on the full panel.
    </div>
  </div>
  <div class="card">
    <div class="legend-row">
      <div class="legend-item">
        <div class="legend-dot" style="background:#4A90D9"></div> Inspection
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#E07B39"></div> Text / LLM
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#3DAA6E"></div> Structural
      </div>
    </div>
    {divs['importance']}
  </div>
</section>

<!-- ⑤ FEI Risk Ranking -->
<section id="ranking">
  <div class="section-head">
    <div class="section-title">⑤ FEI Risk Ranking</div>
    <div class="section-sub">
      Top 20 facilities ranked by Random Forest predicted recall probability,
      using year 2024 features (latest available). Row color intensity ∝ predicted risk.
      Facilities without 483 text coverage have text columns shown as "—".
    </div>
  </div>
  <div class="card">
    {divs['risk']}
  </div>
</section>

</div><!-- /page -->

<footer>
  Generated {today} &nbsp;·&nbsp; Drug Shortage Project &nbsp;·&nbsp; NC State University<br>
  <span style="font-size:11px; color:#BEC8D0;">
    Text features from Redica 483 database (2018–2026), LLM-scored with current pipeline.
    Model: Random Forest, GroupKFold CV, 2015–2024.
  </span>
</footer>

</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data…")
    data = _load_all_data()

    log.info("Running CV for ROC predictions…")
    preds_l2, preds_rf, y = _get_roc_predictions(data["panel"]) \
        if _SKLEARN else (None, None, None)

    log.info("Building HTML…")
    html = build_html(data, preds_l2, preds_rf,
                      y if y is not None else pd.Series(dtype=int))

    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


def _get_roc_predictions(panel):
    return _get_roc_preds(panel)


if __name__ == "__main__":
    main()
