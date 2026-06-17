"""
Module 15 — Interactive shortage story dashboard.

Tells the full causal chain: manufacturing quality failures → two paths → drug shortage.

  Path A: Quality failure → Recall → Supply disruption → Shortage
  Path B: Quality failure → Market exit (firm discontinues) → Shortage

Sections:
  1. Data coverage — facility funnel
  2. Supply landscape — FEIs per drug (concentration risk) + risk bubble matrix
  3. Events timeline — recalls & shortages by drug × year
  4. Causal paths — shortage reason breakdown + quality signal vs outcome scatter
  5. Model evidence — text feature AUC lift + feature importance

Output:
  outputs/figures/recall_fei_dashboard.html

Run:
  python m15_recall_dashboard.py
"""

from __future__ import annotations
import re
import warnings
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
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ModuleNotFoundError:
    _SKLEARN = False

from config import (
    DATA, OUT_FIGS, OUT_DATA, OUT_MODELS, OUT_TABS, OUT_LOGS,
    TEXT_TIMESERIES_REDICA_CSV, RECALL_FILT, VALISURE_FEI,
    PANEL_START_YEAR, PANEL_END_YEAR, SEED,
)
from utils import get_logger

log = get_logger("m15_dashboard", OUT_LOGS / "m15_dashboard.log")

REDICA_RAW    = DATA / "07 - Redica" / "raw"
UUTAH_FILE    = DATA / "24 - UUtah - Drug Shortage" / "raw" / "efox shortages small file through 2025 final.xlsx"
OB_PRODUCTS   = DATA / "01 - Orange Book" / "output_data" / "products.csv"
OUT_HTML      = OUT_FIGS / "recall_fei_dashboard.html"

# Feature groups — must match m14
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

_FEAT_LABEL = {
    "n_oai_cumul":                "OAI (cumulative)",
    "n_vai_t":                    "VAI inspections",
    "n_inspections_t":            "Total inspections",
    "n_warning_letters_t":        "Warning letters",
    "severity_critmajor_share":   "Critical/Major severity",
    "scope_facilitywide_share":   "Facility-wide scope",
    "scope_multipleproducts_share": "Multi-product scope",
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
    "n_feis_drug":                "# FEIs per drug",
}

_GROUP_COLOR = {
    **{f: "#4A90D9" for f in INSP_FEATS},
    **{f: "#E07B39" for f in TEXT_FEATS},
    **{f: "#3DAA6E" for f in STRUCT_FEATS},
}

C = {
    "blue":   "#4A90D9",
    "orange": "#E07B39",
    "green":  "#3DAA6E",
    "red":    "#D94A4A",
    "purple": "#7B5EA7",
    "gray":   "#9AA5B1",
    "dark":   "#1a1a2e",
    "teal":   "#2E8B8B",
}

_PLOTLY_FONT   = dict(family="'Segoe UI', Helvetica, Arial, sans-serif", size=12)
_PLOTLY_MARGIN = dict(l=10, r=10, t=40, b=10)

_PARENTERAL_ROUTES = {
    "INJECTION", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS",
    "INJECTION, INTRAVENOUS", "INTRAVENOUS, SUBCUTANEOUS",
    "INTRAMUSCULAR, INTRAVENOUS", "INJECTABLE", "IRRIGATION",
    "INJECTION, SUBCUTANEOUS",
}

# Shortage reason → causal path
_PATH_A = re.compile(
    r"manufactur|regulatory|cgmp|quality|recall|contamina|potency|ingredient",
    re.IGNORECASE,
)
_PATH_B = re.compile(
    r"discontinu|business decision|market exit|exit|withdrew",
    re.IGNORECASE,
)
_DEMAND = re.compile(r"demand|supply", re.IGNORECASE)
_RAW    = re.compile(r"raw material|ingredient shortage|api shortage", re.IGNORECASE)

TARGET_DRUGS = {
    "metformin": "Metformin",
    "atorvastatin": "Atorvastatin",
    "bupropion": "Bupropion",
    "pantoprazole": "Pantoprazole",
    "vancomycin": "Vancomycin",
    "ampicillin; sulbactam": "Ampicillin; Sulbactam",
    "ampicillin": "Ampicillin",
    "calcium gluconate": "Calcium Gluconate",
    "magnesium sulfate": "Magnesium sulfate",
    "potassium chloride": "Potassium chloride",
    "lisinopril": "Lisinopril",
    "metoprolol": "Metoprolol",
    "metronidazole": "Metronidazole",
    "tacrolimus": "Tacrolimus",
}


# ── Data loading ────────────────────────────────────────────────────────────

def _fei_drug_map() -> pd.DataFrame:
    """Valisure API Only_FEI Mapping → fei × drug, parenteral flag."""
    fm = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fm.columns = [c.strip() for c in fm.columns]
    fei_col = next(c for c in fm.columns if "fei" in c.lower())
    api_col = next(c for c in fm.columns if c.lower() == "api")
    df = fm[[fei_col, api_col]].dropna().rename(columns={fei_col: "fei", api_col: "drug"})
    df["fei"] = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")

    # Parenteral from OB
    parenteral_drugs: set[str] = set()
    if OB_PRODUCTS.exists():
        ob = pd.read_csv(OB_PRODUCTS)
        ob["_route"] = ob["DF;Route"].str.split(";").str[-1].str.strip().str.upper()
        par_ing = set(
            ob.loc[ob["_route"].isin(_PARENTERAL_ROUTES), "Ingredient"]
            .str.upper().dropna().unique()
        )
        for drug in df["drug"].unique():
            parts = [p.strip().upper().split()[0] for p in drug.split(";") if p.strip()]
            for ing in par_ing:
                if any(p in ing.split() for p in parts):
                    parenteral_drugs.add(drug)
                    break

    df["parenteral"] = df["drug"].isin(parenteral_drugs)
    return df


def _load_supply_concentration(fdmap: pd.DataFrame) -> pd.DataFrame:
    """Count FEIs per drug and recall rate."""
    supply = (
        fdmap.groupby("drug", as_index=False)
        .agg(n_feis=("fei", "nunique"), parenteral=("parenteral", "max"))
        .sort_values("n_feis")
    )
    return supply


def _load_recall_by_drug(fdmap: pd.DataFrame) -> pd.DataFrame:
    """Join recall events to drugs via Valisure FEI mapping."""
    recall = pd.read_csv(RECALL_FILT, low_memory=False)
    recall.columns = [c.strip() for c in recall.columns]
    recall["fei"]  = pd.to_numeric(recall["FEI Number"], errors="coerce").astype("Int64")
    recall["year"] = pd.to_datetime(recall["Recall_Date"], errors="coerce").dt.year.astype("Int64")
    cls_col = next((c for c in recall.columns if "event" in c.lower() and "class" in c.lower()), None)
    if cls_col is None:
        cls_col = next((c for c in recall.columns if "class" in c.lower()), "")
    if cls_col:
        recall["class_i"] = recall[cls_col].astype(str).str.contains("Class I|Class-I", regex=True).astype(int)
    else:
        recall["class_i"] = 0

    df = recall.merge(fdmap[["fei", "drug"]].drop_duplicates(), on="fei", how="inner")
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]
    agg = df.groupby(["drug", "year"], as_index=False).agg(
        n_recalls=("fei", "count"),
        n_class_i=("class_i", "sum"),
    )
    log.info("Recall-drug events: %d rows, %d drugs", len(agg), agg["drug"].nunique())
    return agg


def _load_shortage_by_drug() -> pd.DataFrame:
    """UUtah shortage data → drug × year × causal path category."""
    sh = pd.read_excel(UUTAH_FILE, header=1)
    sh.columns = [c.strip() for c in sh.columns]
    sh = sh.rename(columns={
        "Drug Shortages": "drug_raw",
        "yr": "year",
        "Reason": "reason",
    })
    sh["drug_raw"] = sh["drug_raw"].astype(str).str.lower()
    sh["reason"]   = sh["reason"].fillna("").astype(str)

    def _match(name: str) -> str | None:
        for k, v in sorted(TARGET_DRUGS.items(), key=lambda x: -len(x[0])):
            if k in name:
                return v
        return None

    def _path(reason: str) -> str:
        if _PATH_B.search(reason):
            return "Path B — Discontinuation/Exit"
        if _PATH_A.search(reason):
            return "Path A — Quality/Manufacturing"
        if _DEMAND.search(reason):
            return "Demand surge"
        if _RAW.search(reason):
            return "Raw material"
        return "Unknown"

    sh["drug"] = sh["drug_raw"].map(_match)
    sh["path"] = sh["reason"].map(_path)
    sh = sh.dropna(subset=["drug"])
    sh["year"] = pd.to_numeric(sh["year"], errors="coerce").astype("Int64")
    sh = sh.dropna(subset=["year"])
    sh = sh[(sh["year"] >= PANEL_START_YEAR) & (sh["year"] <= PANEL_END_YEAR)]

    agg = sh.groupby(["drug", "year", "path"], as_index=False).size().rename(columns={"size": "n"})
    log.info("Shortage-drug events: %d rows, %d drugs", len(agg), agg["drug"].nunique())
    return agg


def _load_drug_quality_profile(fdmap: pd.DataFrame) -> pd.DataFrame:
    """Average text risk signals per drug (latest FEI snapshot → join → drug mean)."""
    QUALITY_COLS = [
        "contamination_llm_share", "severity_critmajor_share",
        "data_integrity_llm_share", "cultural_root_cause_share",
        "investigation_llm_share",
    ]
    if not TEXT_TIMESERIES_REDICA_CSV.exists():
        return pd.DataFrame(columns=["drug"] + QUALITY_COLS)

    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"]           = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    latest = ts.sort_values("snapshot_date").groupby("fei").last().reset_index()

    drug_fei = fdmap[["fei", "drug"]].drop_duplicates()
    merged = latest.merge(drug_fei, on="fei", how="inner")
    cols_present = [c for c in QUALITY_COLS if c in merged.columns]
    profile = merged.groupby("drug", as_index=False)[cols_present].mean()

    # Composite quality risk (simple average)
    profile["quality_risk"] = profile[cols_present].mean(axis=1)
    log.info("Drug quality profiles: %d drugs", len(profile))
    return profile


def _load_coverage() -> dict:
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
    cov["n_obs_llm"]  = 1115
    cov["n_feis_llm"] = 98
    return cov


def _load_model_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    fi  = pd.read_csv(OUT_MODELS / "rf_importance_recall_fei.csv")
    abl = pd.read_csv(OUT_MODELS / "text_ablation_recall_fei.csv")
    abl["label"] = abl["label"].str.replace(r"\n.*", "", regex=True).map(
        lambda s: "Inspection only" if "without" in s.lower()
                  else "Inspection + LLM text (98 FEIs)"
    )
    return fi, abl


# ── Figure builders ─────────────────────────────────────────────────────────

def _fig_funnel(cov: dict) -> go.Figure:
    labels = ["Valisure universe", "In Redica data", "With ≥1 483 issued", "With LLM text features"]
    vals   = [129, cov["n_feis_redica"], cov["n_sites_with_483"], cov["n_feis_llm"]]
    colors = ["#4A90D9", "#5AA8E0", "#6BBCE8", "#3DAA6E"]

    fig = go.Figure()
    for lbl, val, col in zip(labels, vals, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[lbl], orientation="h",
            marker_color=col,
            text=f"  {val}", textposition="outside",
            showlegend=False,
            hovertemplate=f"<b>{lbl}</b>: {val} FEIs<extra></extra>",
        ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=60, t=10, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[0, 170], showticklabels=False),
        yaxis=dict(autorange="reversed"),
        title=dict(text="Facility coverage funnel", font=dict(size=13)),
    )
    return fig


def _fig_supply_concentration(supply: pd.DataFrame, recalls_by_drug: pd.DataFrame) -> go.Figure:
    """Grouped bars: FEIs per drug (left) + recall count (right), colored by parenteral flag."""
    sup = supply.copy().sort_values("n_feis")
    tot_recalls = recalls_by_drug.groupby("drug")["n_recalls"].sum().to_dict()
    sup["n_recalls_total"] = sup["drug"].map(tot_recalls).fillna(0).astype(int)

    fei_colors    = [C["orange"] if p else C["blue"] for p in sup["parenteral"]]
    recall_colors = [C["red"]] * len(sup)

    hover_fei = [
        f"<b>{row.drug}</b><br>FEIs: {row.n_feis}<br>"
        f"{'Has injectable formulation (OB)' if row.parenteral else 'Oral formulations only'}"
        for _, row in sup.iterrows()
    ]
    hover_recall = [
        f"<b>{row.drug}</b><br>Recalls linked to FEIs: {row.n_recalls_total} (2015–2024)"
        for _, row in sup.iterrows()
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FEIs producing each drug (supply concentration)",
                        "Recall events linked to FEIs (2015–2024)"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(
        x=sup["n_feis"], y=sup["drug"], orientation="h",
        marker_color=fei_colors,
        hovertext=hover_fei, hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=sup["n_recalls_total"], y=sup["drug"], orientation="h",
        marker_color=recall_colors,
        hovertext=hover_recall, hoverinfo="text",
        showlegend=False,
    ), row=1, col=2)

    # Legend traces
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=C["orange"],
                         name="Has injectable form (OB)", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=C["blue"],
                         name="Oral formulations only", showlegend=True))

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="# FEIs", gridcolor="#F0F0F0"),
        xaxis2=dict(title="# Recall events", gridcolor="#F0F0F0"),
        yaxis=dict(gridcolor="white"),
        yaxis2=dict(showticklabels=False, gridcolor="white"),
        legend=dict(x=0.72, y=0.02),
        barmode="overlay",
    )
    return fig




def _fig_timeline(recalls_by_drug: pd.DataFrame, shortages: pd.DataFrame) -> go.Figure:
    """Two time series: recalls (top) and shortages (bottom) by drug × year."""
    years = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))
    drugs = sorted(set(recalls_by_drug["drug"]) | set(shortages["drug"]))

    palette = [
        "#4A90D9", "#E07B39", "#3DAA6E", "#D94A4A", "#7B5EA7",
        "#2E8B8B", "#C6833A", "#5588BB", "#8E4B9E", "#357A38",
        "#C0392B", "#1A7A4A", "#845EC2", "#FF9671",
    ]
    drug_color = {d: palette[i % len(palette)] for i, d in enumerate(sorted(drugs))}

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Recall events by drug × year (Path A)",
                        "Shortage events by drug × year (Path A + B + other)"],
        vertical_spacing=0.14,
        shared_xaxes=True,
    )

    def _to_yearly(df_sub: pd.DataFrame, val_col: str) -> list[int]:
        s = df_sub.set_index("year")[val_col].reindex(years, fill_value=0)
        return s.tolist()

    # Recalls
    for drug in sorted(drugs):
        sub = recalls_by_drug[recalls_by_drug["drug"] == drug][["year", "n_recalls"]].copy()
        sub["year"] = sub["year"].astype(int)
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=years, y=_to_yearly(sub, "n_recalls"),
            name=drug, marker_color=drug_color[drug],
            showlegend=True, legendgroup=drug,
            hovertemplate=f"<b>{drug}</b><br>Year: %{{x}}<br>Recalls: %{{y}}<extra></extra>",
        ), row=1, col=1)

    # Shortages — stack by path
    path_colors = {
        "Path A — Quality/Manufacturing": C["red"],
        "Path B — Discontinuation/Exit":  C["purple"],
        "Demand surge":                    C["orange"],
        "Raw material":                    C["teal"],
        "Unknown":                         C["gray"],
    }
    path_seen: set[str] = set()
    for drug in sorted(drugs):
        sub_d = shortages[shortages["drug"] == drug]
        if sub_d.empty:
            continue
        for path, pcolor in path_colors.items():
            sub_p = sub_d[sub_d["path"] == path][["year", "n"]].copy()
            sub_p["year"] = sub_p["year"].astype(int)
            if sub_p.empty:
                continue
            fig.add_trace(go.Bar(
                x=years, y=_to_yearly(sub_p, "n"),
                name=path, marker_color=pcolor,
                showlegend=(path not in path_seen),
                legendgroup=f"path_{path}",
                hovertemplate=f"<b>{drug}</b> | {path}<br>Year: %{{x}}<br>Shortages: %{{y}}<extra></extra>",
            ), row=2, col=1)
            path_seen.add(path)

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=50, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        barmode="stack",
        legend=dict(x=1.01, y=1, font=dict(size=10)),
        xaxis2=dict(title="Year"),
        yaxis=dict(title="# Recall events", gridcolor="#F0F0F0"),
        yaxis2=dict(title="# Shortage events", gridcolor="#F0F0F0"),
    )
    return fig


def _fig_path_breakdown(shortages: pd.DataFrame) -> go.Figure:
    """Shortage reason breakdown and drug-level path composition."""
    path_totals = shortages.groupby("path")["n"].sum().sort_values(ascending=False)

    path_colors = {
        "Path A — Quality/Manufacturing": C["red"],
        "Path B — Discontinuation/Exit":  C["purple"],
        "Demand surge":                    C["orange"],
        "Raw material":                    C["teal"],
        "Unknown":                         C["gray"],
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Shortage events by causal path (2015–2024)",
                        "Path composition by drug"],
        horizontal_spacing=0.1,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
    )

    # Left: overall path breakdown (donut)
    fig.add_trace(go.Pie(
        labels=path_totals.index.tolist(),
        values=path_totals.values.tolist(),
        hole=0.45,
        marker_colors=[path_colors.get(p, C["gray"]) for p in path_totals.index],
        textinfo="label+percent",
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate="<b>%{label}</b><br>Events: %{value}<br>%{percent}<extra></extra>",
    ), row=1, col=1)

    # Right: stacked bar by drug (Path A vs B vs other)
    drugs = sorted(shortages["drug"].unique())
    path_seen: set[str] = set()
    for path, pcolor in path_colors.items():
        vals = []
        for drug in drugs:
            sub = shortages[(shortages["drug"] == drug) & (shortages["path"] == path)]
            vals.append(int(sub["n"].sum()))
        fig.add_trace(go.Bar(
            x=drugs, y=vals,
            name=path, marker_color=pcolor,
            showlegend=(path not in path_seen),
            legendgroup=f"path_{path}",
            hovertemplate="<b>%{x}</b><br>" + path + "<br>Events: %{y}<extra></extra>",
        ), row=1, col=2)
        path_seen.add(path)

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=50, b=80),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        barmode="stack",
        xaxis2=dict(tickangle=-35),
        yaxis2=dict(title="# Shortage events", gridcolor="#F0F0F0"),
        legend=dict(x=1.01, y=1, font=dict(size=10)),
    )
    return fig


def _fig_model_evidence(fi: pd.DataFrame, ablation: pd.DataFrame) -> go.Figure:
    """Left: L2 text feature lift (ablation bar). Right: feature importance (top 12)."""
    fi_top = fi.sort_values("importance", ascending=False).head(12).copy()
    fi_top["label"] = fi_top["feature"].map(lambda f: _FEAT_LABEL.get(f, f))
    fi_top["color"] = fi_top["feature"].map(lambda f: _GROUP_COLOR.get(f, C["gray"]))
    fi_top["group"] = fi_top["feature"].map(
        lambda f: "Inspection" if f in INSP_FEATS
                  else "Text / LLM" if f in TEXT_FEATS
                  else "Structural"
    )
    fi_top = fi_top.sort_values("importance")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["AUC lift from 483 text (L2 logistic regression, GroupKFold CV)",
                        "Feature importance (Random Forest, top 12)"],
        horizontal_spacing=0.1,
        column_widths=[0.35, 0.65],
    )

    # Ablation bars
    bar_colors = [C["gray"], C["orange"]]
    for i, row in ablation.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]], y=[row["auc"]],
            marker_color=bar_colors[i % 2],
            text=f"{row['auc']:.3f}", textposition="outside",
            textfont=dict(size=13),
            showlegend=False, width=0.4,
            hovertemplate=f"<b>{row['label']}</b><br>AUC: {row['auc']:.3f}<extra></extra>",
        ), row=1, col=1)

    if len(ablation) == 2:
        delta = ablation.iloc[1]["auc"] - ablation.iloc[0]["auc"]
        rel   = delta / max(ablation.iloc[0]["auc"], 0.001) * 100
        fig.add_annotation(
            xref="x", yref="y",
            x=ablation.iloc[1]["label"], y=ablation.iloc[1]["auc"],
            text=f"<b>+{delta:.3f}<br>(+{rel:.0f}%)</b>",
            showarrow=True, arrowhead=2, arrowcolor=C["orange"],
            ax=55, ay=-40, font=dict(color=C["orange"], size=12),
        )

    # Feature importance
    for grp, col in [("Inspection", C["blue"]), ("Text / LLM", C["orange"]),
                     ("Structural", C["green"])]:
        sub = fi_top[fi_top["group"] == grp]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["importance"], y=sub["label"],
            orientation="h", name=grp, marker_color=col,
            text=sub["importance"].map("{:.3f}".format),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        height=440, margin=dict(l=10, r=80, t=50, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        barmode="overlay",
        yaxis=dict(range=[0, max(ablation["auc"]) * 1.25], title="AUC-ROC",
                   gridcolor="#F0F0F0"),
        xaxis=dict(gridcolor="white"),
        xaxis2=dict(title="RF Feature Importance", gridcolor="#F0F0F0"),
        yaxis2=dict(gridcolor="white"),
        legend=dict(x=0.73, y=0.05, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E0E0E0", borderwidth=1),
    )
    return fig


def _load_case_study(drug: str = "Metformin") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load monthly quality signals, recalls, and yearly shortage events for one drug."""
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    api_col = next(c for c in fei_map.columns if c.lower() == "api")
    fei_col = next(c for c in fei_map.columns if "fei" in c.lower())
    drug_feis = set(
        pd.to_numeric(
            fei_map[fei_map[api_col] == drug][fei_col], errors="coerce"
        ).dropna().astype(int)
    )

    # Quality signals: average over drug FEIs by snapshot month
    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    drug_ts = ts[ts["fei"].isin(drug_feis)].copy()
    drug_ts["month_start"] = drug_ts["snapshot_date"].dt.to_period("M").dt.start_time

    SIGNAL_COLS = ["contamination_llm_share", "severity_critmajor_share",
                   "data_integrity_llm_share"]
    cols_present = [c for c in SIGNAL_COLS if c in drug_ts.columns]
    quality_monthly = drug_ts.groupby("month_start")[cols_present].mean().reset_index()

    # Recalls: aggregate to month level
    recall = pd.read_csv(RECALL_FILT, low_memory=False)
    recall.columns = [c.strip() for c in recall.columns]
    recall["fei"] = pd.to_numeric(recall["FEI Number"], errors="coerce").astype("Int64")
    recall["recall_dt"] = pd.to_datetime(recall["Recall_Date"], errors="coerce")
    r_drug = recall[recall["fei"].isin(drug_feis) & recall["recall_dt"].notna()].copy()
    r_drug["month_start"] = r_drug["recall_dt"].dt.to_period("M").dt.start_time
    recalls_monthly = (
        r_drug.groupby("month_start", as_index=False).size()
        .rename(columns={"size": "n_recalls"})
    )

    # Shortages: year-level (UUtah is annual)
    sh = pd.read_excel(UUTAH_FILE, header=1)
    sh.columns = [c.strip() for c in sh.columns]
    sh = sh.rename(columns={"Drug Shortages": "drug_raw", "yr": "year"})
    sh["year"] = pd.to_numeric(sh["year"], errors="coerce").astype("Int64")
    sh["drug_raw"] = sh["drug_raw"].astype(str).str.lower()
    sh_drug = sh[sh["drug_raw"].str.contains(drug.lower(), na=False)]
    shortages_yearly = (
        sh_drug[sh_drug["year"].between(PANEL_START_YEAR, PANEL_END_YEAR)]
        .groupby("year", as_index=False).size()
        .rename(columns={"size": "n_shortages"})
    )

    return quality_monthly, recalls_monthly, shortages_yearly


def _fig_case_study(
    quality: pd.DataFrame,
    recalls: pd.DataFrame,
    shortages: pd.DataFrame,
    drug: str = "Metformin",
) -> go.Figure:
    """Monthly-resolution story: quality signal → recall → shortage, showing the time lag."""
    date_start = pd.Timestamp(f"{PANEL_START_YEAR}-01-01")
    date_end   = pd.Timestamp(f"{PANEL_END_YEAR}-12-31")
    q = quality[quality["month_start"].between(date_start, date_end)].copy()
    r = recalls[recalls["month_start"].between(date_start, date_end)].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bars: monthly recall count (left Y)
    fig.add_trace(go.Bar(
        x=r["month_start"],
        y=r["n_recalls"],
        name="Recall events",
        marker_color=C["red"],
        opacity=0.75,
        hovertemplate="<b>%{x|%b %Y}</b><br>Recalls: %{y}<extra></extra>",
    ), secondary_y=False)

    # Lines: quality signals (right Y) — points only where snapshots exist
    signal_cfg = [
        ("severity_critmajor_share",  "Critical/Major severity share", C["orange"], "solid"),
        ("contamination_llm_share",   "Contamination flag share",      C["purple"], "dash"),
        ("data_integrity_llm_share",  "Data integrity flag share",     C["teal"],   "dot"),
    ]
    for col, label, col_color, dash in signal_cfg:
        if col in q.columns:
            fig.add_trace(go.Scatter(
                x=q["month_start"],
                y=q[col],
                name=label,
                mode="lines+markers",
                line=dict(color=col_color, width=2, dash=dash),
                marker=dict(size=7),
                connectgaps=False,
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.2f}}<extra></extra>",
            ), secondary_y=True)

    # Shortage vertical markers (year-level UUtah — mark Jan 1 of each shortage year)
    for _, row in shortages.iterrows():
        yr = int(row["year"])
        x_str = f"{yr}-01-01"
        fig.add_vline(x=x_str, line_dash="dot", line_color=C["green"], line_width=1.5)
        fig.add_annotation(
            x=x_str, y=1.08, yref="paper",
            text=f"⚠ Shortage ({yr})",
            showarrow=False,
            font=dict(color=C["green"], size=9), align="center",
        )

    if drug == "Metformin":
        fig.add_annotation(
            x="2020-06-01", y=14,
            text="<b>June 2020: 14 recalls</b><br>NDMA contamination<br>(CGMP deviations)",
            showarrow=True, arrowhead=2, arrowcolor=C["red"],
            ax=80, ay=-60,
            font=dict(color=C["red"], size=10),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=C["red"], borderwidth=1,
        )
        # Annotate the Jan 2020 quality spike if data exists
        jan2020 = q[q["month_start"] == pd.Timestamp("2020-01-01")]
        if not jan2020.empty and "severity_critmajor_share" in jan2020.columns:
            sev_val = float(jan2020["severity_critmajor_share"].iloc[0])
            fig.add_annotation(
                x="2020-01-01", y=sev_val, yref="y2",
                text="<b>Jan 2020: quality spike</b><br>Severity=1.0, DI=0.43<br>→ ~5 month early warning",
                showarrow=True, arrowhead=2, arrowcolor=C["orange"],
                ax=-110, ay=-45,
                font=dict(color=C["orange"], size=10),
                bgcolor="rgba(255,255,255,0.85)", bordercolor=C["orange"], borderwidth=1,
            )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            title="Month", gridcolor="#F0F0F0",
            tickformat="%b %Y", dtick="M6", tickangle=-30,
        ),
        yaxis=dict(title="# Recall events", gridcolor="#F0F0F0"),
        yaxis2=dict(title="Quality signal share (0–1)", range=[0, 1.2],
                    gridcolor="rgba(0,0,0,0)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E0E0E0", borderwidth=1, font=dict(size=11)),
        bargap=0.3,
    )
    return fig


# ── HTML assembly ────────────────────────────────────────────────────────────

def _div(fig: go.Figure, fig_id: str) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=fig_id,
                       config={"displayModeBar": True,
                               "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def build_html(
    cov: dict,
    supply: pd.DataFrame,
    recalls_by_drug: pd.DataFrame,
    shortages: pd.DataFrame,
    fi: pd.DataFrame,
    ablation: pd.DataFrame,
    cs_quality: pd.DataFrame | None = None,
    cs_recalls: pd.DataFrame | None = None,
    cs_shortages: pd.DataFrame | None = None,
) -> str:
    auc_no  = ablation.iloc[0]["auc"]
    auc_yes = ablation.iloc[1]["auc"]
    lift_abs = auc_yes - auc_no
    lift_rel = lift_abs / max(auc_no, 0.001) * 100

    total_recalls   = int(recalls_by_drug["n_recalls"].sum())
    total_shortages = int(shortages["n"].sum())
    path_b_share    = int(
        shortages[shortages["path"] == "Path B — Discontinuation/Exit"]["n"].sum()
        / max(total_shortages, 1) * 100
    )

    cs_div = ""
    if cs_quality is not None and cs_recalls is not None and cs_shortages is not None:
        cs_div = _div(_fig_case_study(cs_quality, cs_recalls, cs_shortages), "fig_case")

    divs = {
        "funnel":   _div(_fig_funnel(cov),                                "fig_funnel"),
        "supply":   _div(_fig_supply_concentration(supply, recalls_by_drug), "fig_supply"),
        "timeline": _div(_fig_timeline(recalls_by_drug, shortages),       "fig_timeline"),
        "paths":    _div(_fig_path_breakdown(shortages),                  "fig_paths"),
        "case":     cs_div,
        "model":    _div(_fig_model_evidence(fi, ablation),               "fig_model"),
    }

    today = date.today().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drug Shortage — Quality Risk & Causal Paths</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
          background: #F0F2F5; color: #212529; font-size: 14px; }}

  nav {{ position: sticky; top: 0; z-index: 200; background: #1a1a2e;
         display: flex; align-items: center; padding: 0 28px; height: 52px;
         box-shadow: 0 2px 10px rgba(0,0,0,0.3); gap: 2px; }}
  .nav-brand {{ font-size: 13.5px; font-weight: 700; color: #F8F9FA;
                 margin-right: auto; white-space: nowrap; }}
  .nav-brand span {{ color: #68D391; }}
  nav a {{ color: #A0AEC0; text-decoration: none; font-size: 12px; font-weight: 500;
            padding: 6px 12px; border-radius: 6px; transition: all 0.15s; white-space: nowrap; }}
  nav a:hover {{ color: #F8F9FA; background: rgba(255,255,255,0.1); }}

  .hero {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white; padding: 40px 40px 32px; border-bottom: 3px solid #68D391; }}
  .hero h1 {{ font-size: 23px; font-weight: 700; margin-bottom: 8px; }}
  .hero p  {{ color: #A0AEC0; font-size: 13.5px; line-height: 1.65; max-width: 800px; margin-bottom: 18px; }}

  .paths-row {{ display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 24px; }}
  .path-card {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
                 border-radius: 10px; padding: 14px 20px; flex: 1; min-width: 220px; }}
  .path-card .path-label {{ font-size: 11px; color: #A0AEC0; text-transform: uppercase;
                              letter-spacing: 0.5px; margin-bottom: 6px; }}
  .path-card .path-title {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; }}
  .path-card .path-desc  {{ font-size: 12px; color: #A0AEC0; line-height: 1.5; }}
  .path-a {{ border-left: 3px solid #D94A4A; }}
  .path-b {{ border-left: 3px solid #7B5EA7; }}

  .kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .kpi {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
           border-radius: 10px; padding: 12px 18px; min-width: 120px; }}
  .kpi .val {{ font-size: 28px; font-weight: 800; color: #68D391; line-height: 1; }}
  .kpi .lbl {{ font-size: 11px; color: #A0AEC0; margin-top: 4px; line-height: 1.3; }}

  .page {{ max-width: 1380px; margin: 0 auto; padding: 0 28px 40px; }}
  section {{ padding-top: 36px; }}
  .section-head {{ margin-bottom: 16px; }}
  .section-title {{ font-size: 16px; font-weight: 700; color: #1a1a2e;
                     border-left: 4px solid #68D391; padding-left: 12px; }}
  .section-sub {{ font-size: 12.5px; color: #6C757D; margin-top: 5px;
                   padding-left: 16px; line-height: 1.5; max-width: 900px; }}
  .card {{ background: white; border-radius: 12px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.07); padding: 18px 18px 10px;
            margin-bottom: 16px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}

  footer {{ text-align: center; color: #9AA5B1; font-size: 12px;
             padding: 22px 0 30px; border-top: 1px solid #DEE2E6; margin-top: 20px; }}

  @media (max-width: 900px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .page {{ padding: 0 12px 28px; }}
    nav a {{ display: none; }}
  }}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">Drug Shortage &nbsp;<span>·</span>&nbsp; Quality Risk & Causal Paths</div>
  <a href="#coverage">① Coverage</a>
  <a href="#supply">② Supply</a>
  <a href="#timeline">③ Timeline</a>
  <a href="#paths">④ Causal Paths</a>
  <a href="#casestudy">⑤ Case Study</a>
  <a href="#model">⑥ Model Evidence</a>
</nav>

<div class="hero">
  <h1>Manufacturing Quality Failures → Drug Shortages</h1>
  <p>Quality failures at generic drug manufacturing facilities (FEIs) can cause shortages through
     two distinct causal paths. This dashboard tracks both paths using FDA 483 inspection data,
     recall records, and UUtah shortage data across 14 Valisure-validated APIs (2015–2024).</p>

  <div class="paths-row">
    <div class="path-card path-a">
      <div class="path-label">Path A — Recall</div>
      <div class="path-title" style="color:#D94A4A;">Quality failure → Recall → Supply gap</div>
      <div class="path-desc">483 observations (contamination, lab controls, data integrity) trigger
        a recall event. Product is pulled from market. If too few alternative suppliers exist,
        a shortage follows.</div>
    </div>
    <div class="path-card path-b">
      <div class="path-label">Path B — Silent Exit</div>
      <div class="path-title" style="color:#7B5EA7;">Quality failure → Facility exit → Shortage</div>
      <div class="path-desc">A major 483 citing expensive remediation forces a business decision:
        the firm discontinues the product rather than invest in compliance. No recall is ever issued,
        but supply shrinks. Harder to detect — SDUD volume drop is the best signal.</div>
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="val">14</div><div class="lbl">APIs tracked<br>(Valisure-tested)</div></div>
    <div class="kpi"><div class="val">{cov['n_feis_llm']}</div><div class="lbl">FEIs with<br>LLM text features</div></div>
    <div class="kpi"><div class="val">{total_recalls}</div><div class="lbl">Recall events<br>linked to FEIs</div></div>
    <div class="kpi"><div class="val">{total_shortages}</div><div class="lbl">Shortage events<br>(UUtah, 2015–2024)</div></div>
    <div class="kpi"><div class="val">{path_b_share}%</div><div class="lbl">Shortages from<br>discontinuation (Path B)</div></div>
    <div class="kpi"><div class="val">+{lift_rel:.0f}%</div><div class="lbl">AUC lift from<br>483 text features</div></div>
  </div>
</div>

<div class="page">

<!-- ① Data Coverage -->
<section id="coverage">
  <div class="section-head">
    <div class="section-title">① Data Coverage</div>
    <div class="section-sub">
      Starting from 129 Valisure FEIs: 127 in Redica inspection database,
      {cov['n_sites_with_483']} with ≥1 483 issued, {cov['n_feis_llm']} with
      {cov['n_obs_llm']:,} observations scored by the current LLM pipeline.
    </div>
  </div>
  <div class="card">{divs['funnel']}</div>
</section>

<!-- ② Supply Landscape -->
<section id="supply">
  <div class="section-head">
    <div class="section-title">② Supply Concentration</div>
    <div class="section-sub">
      Left panel: how many FEIs (manufacturing facilities) produce each drug, based on
      the March 2026 Valisure mapping. Fewer producing FEIs = higher concentration risk.
      Orange bars = drug has at least one injectable formulation approved in the Orange Book —
      this does <i>not</i> mean the drug is exclusively injectable. For example, Vancomycin
      is given IV for bloodstream infections but orally for C. diff; Metoprolol is primarily
      oral but has an IV cardiac form.
      Right panel: total recall events linked to these FEIs (2015–2024). Hover for drug details.
    </div>
  </div>
  <div class="card">{divs['supply']}</div>
</section>

<!-- ③ Events Timeline -->
<section id="timeline">
  <div class="section-head">
    <div class="section-title">③ Recall & Shortage Events Over Time</div>
    <div class="section-sub">
      Top panel: recall events per drug per year (Path A signal — a facility had a quality failure
      serious enough to trigger a product recall).
      Bottom panel: shortage events by drug per year, color-coded by reason.
      Notable spikes: Metformin 2020 (NDMA contamination crisis), Potassium Chloride 2020 (COVID supply chain).
    </div>
  </div>
  <div class="card">{divs['timeline']}</div>
</section>

<!-- ④ Causal Paths -->
<section id="paths">
  <div class="section-head">
    <div class="section-title">④ Causal Path Analysis</div>
    <div class="section-sub">
      <b>How paths are classified:</b> The University of Utah drug shortage database
      includes a free-text <i>Reason</i> field for each shortage event. We classify
      events by keyword matching on that field — "manufacturing / CGMP / quality / recall"
      → Path A; "discontinued / business decision / market exit" → Path B.
      This is observational labeling from a self-reported field; it does not establish causation.
      <b>Key limitation: ~50% of events in our 13-drug subset are reported as "Unknown" reason</b>,
      so the true split between paths is uncertain. The breakdown below reflects only events
      where a reason was recorded.
      <br><br>
      Path B evidence in this panel: Lisinopril 2020 (Business Decision), Vancomycin 2002
      (Business Decision — outside panel window), Potassium chloride 2019 (combo product discontinuation).
      These are the clearest examples of potential silent exits in the data.
    </div>
  </div>
  <div class="card">{divs['paths']}</div>
</section>

<!-- ⑤ Case Study: Metformin NDMA 2020 -->
<section id="casestudy">
  <div class="section-head">
    <div class="section-title">⑤ Case Study: Metformin — Path A at Monthly Resolution</div>
    <div class="section-sub">
      Monthly-resolution view showing the real time lag between quality signals and recalls.
      <b>The chain:</b>
      (1) FDA 483 inspection snapshots for Metformin FEIs in <b>January 2020</b> show
      Critical/Major severity = 1.0 and data integrity = 0.43 — the highest values in the panel;
      (2) In <b>June 2020</b> — approximately 5 months later — 14 recalls are issued, all citing
      "CGMP Deviations: NDMA impurity above acceptable intake level";
      (3) UUtah records a Metformin shortage in <b>2021 and 2024</b> as downstream supply disruption follows.
      <br><br>
      <b>Note on quality signal lines:</b> LLM text features come from Redica inspection snapshots,
      recorded at irregular dates (not monthly). Points on the quality lines represent actual
      snapshot dates aggregated to their calendar month — gaps mean no snapshot in that month.
      Bars = recall events (left axis, monthly). Lines = LLM quality signals averaged across
      Metformin FEIs per snapshot month (right axis, 0–1 share).
      Green dotted lines = shortage events (UUtah, year-level — placed at Jan 1 of the shortage year).
    </div>
  </div>
  <div class="card">{divs['case']}</div>
</section>

<!-- ⑥ Model Evidence -->
<section id="model">
  <div class="section-head">
    <div class="section-title">⑥ Model Evidence: 483 Text Improves Recall Prediction</div>
    <div class="section-sub">
      <b>What this model predicts:</b> whether a FEI will have a <i>recall event</i> in year t+1,
      given features measured in year t. This is the intermediate step in Path A
      (quality failure → recall) — not a direct shortage prediction. Recall prediction is
      the tractable modeling target; the path from recall to shortage depends on supply
      concentration (Section ②) and market structure.
      <br><br>
      L2 logistic regression, GroupKFold CV by FEI (a facility never appears in both train and test).
      Adding LLM-extracted 483 text features improves AUC from {auc_no:.3f} to {auc_yes:.3f}
      (+{lift_abs:.3f}, +{lift_rel:.0f}% relative lift), showing that inspection text carries
      predictive signal beyond structured inspection counts alone.
      Right panel: feature importance from a full-panel Random Forest (top 12 features by group).
    </div>
  </div>
  <div class="card">{divs['model']}</div>
</section>

</div>

<footer>
  Generated {today} &nbsp;·&nbsp; Drug Shortage Project &nbsp;·&nbsp; NC State University<br>
  <span style="font-size:11px; color:#BEC8D0;">
    483 text features from Redica database (2018–2026), LLM pipeline.
    Recall data: FDA CDER. Shortage data: University of Utah. Supply concentration: Valisure (March 2026).
  </span>
</footer>

</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data…")
    fdmap    = _fei_drug_map()
    supply   = _load_supply_concentration(fdmap)
    recalls  = _load_recall_by_drug(fdmap)
    shortage = _load_shortage_by_drug()
    cov      = _load_coverage()
    fi, abl  = _load_model_outputs()
    cs_q, cs_r, cs_s = _load_case_study("Metformin")
    log.info("Case study: Metformin — %d quality months, %d recall months, %d shortage years",
             len(cs_q), len(cs_r), len(cs_s))

    log.info("Building HTML…")
    html = build_html(cov, supply, recalls, shortage, fi, abl,
                      cs_quality=cs_q, cs_recalls=cs_r, cs_shortages=cs_s)

    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


if __name__ == "__main__":
    main()
