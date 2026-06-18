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
SDUD_PANEL    = DATA / "04_11 - Build - Monthly Panel (SDUD+NADAC)" / "processed" / "2026-03-26-sdud_nadac_panel.csv"
NDC_PRODUCT   = DATA / "03 - FDA - NDC" / "product.csv"

# Per-drug story FEI: fei_id → (display_name, sdud_keyword)
_STORY_FEIS: dict[str, dict[int, tuple[str, str]]] = {
    "Metformin":  {3005263655: ("Amneal Pharmaceuticals", "amneal")},
    "Lisinopril": {3007549629: ("Lupin Pharmaceuticals", "lupin")},
}

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


def _load_sdud_manufacturers(
    drug: str,
    top_n: int = 5,
    force_include_kw: list[str] | None = None,
) -> pd.DataFrame:
    """Monthly SDUD volumes grouped by manufacturer (top N + forced story manufacturers).
    Returns DataFrame: date, mfr_name, units (millions)."""
    if not SDUD_PANEL.exists():
        return pd.DataFrame(columns=["date", "mfr_name", "units"])
    try:
        sdud = pd.read_csv(SDUD_PANEL, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=["date", "mfr_name", "units"])

    sdud["date"] = pd.to_datetime(sdud["date"], errors="coerce")
    first_word = drug.split(";")[0].strip().split()[0]
    sdud = sdud[sdud["drug_name"].str.contains(first_word, case=False, na=False)].copy()
    if sdud.empty:
        return pd.DataFrame(columns=["date", "mfr_name", "units"])

    sdud["labeler_code"] = sdud["ndc11"].astype(str).str[:5]

    lab_names: dict[str, str] = {}
    if NDC_PRODUCT.exists():
        try:
            prod = pd.read_csv(NDC_PRODUCT, low_memory=False, encoding="latin1")
            prod["_lab"] = prod["PRODUCTNDC"].str.split("-").str[0].str.zfill(5)
            lab_names = prod.drop_duplicates("_lab").set_index("_lab")["LABELERNAME"].to_dict()
        except Exception:
            pass

    _KW: dict[str, str] = {
        "amneal": "Amneal",    "aurobindo": "Aurobindo",
        "zydus": "Zydus",      "teva": "Teva",
        "mylan": "Mylan/Viatris", "lupin": "Lupin",
        "accord": "Accord",    "solco": "Solco",
        "sandoz": "Sandoz",    "ascend": "Ascend",
        "granules": "Granules", "avet": "Avet/Sun",
        "heritage": "Avet/Sun",
    }

    def _group(lab: str) -> str:
        raw = lab_names.get(lab, lab).lower()
        for kw, name in _KW.items():
            if kw in raw:
                return name
        return " ".join(lab_names.get(lab, f"Labeler {lab}").split()[:3])

    sdud["mfr_name"] = sdud["labeler_code"].map(_group)
    agg = (
        sdud.groupby(["date", "mfr_name"])["sdud_units_reimbursed"]
        .sum().reset_index().rename(columns={"sdud_units_reimbursed": "units"})
    )
    agg["units"] /= 1e6  # millions

    totals = agg.groupby("mfr_name")["units"].sum().sort_values(ascending=False)
    top_mfrs = totals.head(top_n).index.tolist()

    if force_include_kw:
        for kw in force_include_kw:
            extra = [m for m in totals.index if kw.lower() in m.lower()]
            for m in extra:
                if m not in top_mfrs:
                    top_mfrs.append(m)

    return agg[agg["mfr_name"].isin(top_mfrs)].sort_values(["mfr_name", "date"]).reset_index(drop=True)


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
    """Per-drug dropdown: recall events (top) and shortage events (bottom) by year."""
    years = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))
    drugs = sorted(set(recalls_by_drug["drug"]) | set(shortages["drug"]))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Recall events", "Shortage events"],
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    for i, drug in enumerate(drugs):
        visible = (i == 0)

        # Recalls
        sub_r = recalls_by_drug[recalls_by_drug["drug"] == drug][["year", "n_recalls"]].copy()
        sub_r["year"] = sub_r["year"].astype(int)
        y_r = sub_r.set_index("year")["n_recalls"].reindex(years, fill_value=0).tolist()
        fig.add_trace(go.Bar(
            x=years, y=y_r, name="Recalls",
            marker_color=C["red"], visible=visible, showlegend=False,
            hovertemplate="<b>Year %{x}</b><br>Recalls: %{y}<extra></extra>",
        ), row=1, col=1)

        # Shortages (total, no path split)
        tot_s = shortages[shortages["drug"] == drug].groupby("year")["n"].sum().reset_index()
        tot_s["year"] = tot_s["year"].astype(int)
        y_s = tot_s.set_index("year")["n"].reindex(years, fill_value=0).tolist()
        fig.add_trace(go.Bar(
            x=years, y=y_s, name="Shortages",
            marker_color=C["purple"], visible=visible, showlegend=False,
            hovertemplate="<b>Year %{x}</b><br>Shortage events: %{y}<extra></extra>",
        ), row=2, col=1)

    # Dropdown — 2 traces per drug
    n = 2
    buttons = []
    for i, drug in enumerate(drugs):
        vis = [False] * (len(drugs) * n)
        vis[i * n]     = True
        vis[i * n + 1] = True
        buttons.append(dict(label=drug, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=80, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.0, y=1.22,
            xanchor="left", yanchor="top",
            showactive=True,
            bgcolor="white", bordercolor="#DDD",
            font=dict(size=12),
        )],
        xaxis2=dict(title="Year", dtick=1, gridcolor="#F0F0F0"),
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


def _load_case_study(drug: str = "Metformin") -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load quality signals, recalls, shortages, SDUD, and per-FEI recalls for one drug."""
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    api_col = next(c for c in fei_map.columns if c.lower() == "api")
    fei_col = next(c for c in fei_map.columns if "fei" in c.lower())
    drug_feis = set(
        pd.to_numeric(
            fei_map[fei_map[api_col] == drug][fei_col], errors="coerce"
        ).dropna().astype(int)
    )

    # Quality signals: per-FEI snapshot rows
    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    drug_ts = ts[ts["fei"].isin(drug_feis)].copy()
    SIGNAL_COLS = ["contamination_llm_share", "severity_critmajor_share",
                   "data_integrity_llm_share"]
    cols_present = [c for c in SIGNAL_COLS if c in drug_ts.columns]
    keep_cols = ["fei", "snapshot_date", "n_obs_total"] + cols_present
    quality_raw = drug_ts[[c for c in keep_cols if c in drug_ts.columns]].copy()

    # Recalls: monthly total AND per-FEI breakdown
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
    if not r_drug.empty:
        recalls_by_fei = (
            r_drug.groupby(["fei", "month_start"], as_index=False).size()
            .rename(columns={"size": "n_recalls"})
        )
    else:
        recalls_by_fei = pd.DataFrame(columns=["fei", "month_start", "n_recalls"])

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

    # SDUD: top manufacturers, force-include story FEIs' manufacturers
    story_kws = [kw for _, (_, kw) in _STORY_FEIS.get(drug, {}).items()]
    sdud_mfr = _load_sdud_manufacturers(drug, top_n=5, force_include_kw=story_kws)

    return quality_raw, recalls_monthly, shortages_yearly, sdud_mfr, recalls_by_fei


def _build_combined_dropdown(
    fig: go.Figure,
    n_sig_total: int,
    n_sdud: int,
    story_feis: dict[int, tuple[str, str]],
    sig_labels_with_idx: list[tuple[int, str]],
) -> None:
    """Combined dropdown: signal-filter (aggregate) options + per-FEI spotlight options.

    Trace layout expected in fig.data:
      [0..2*n_sig_total-1] : (agg_line, raw_dots) × n_sig_total
      [2*n_sig_total]       : recall_bar_total
      [2*n_sig_total+1
        ..2*n_sig_total+n_sdud] : SDUD regular traces  (n_sdud)
      per story FEI k (n_sig_total+2 traces each):
        [base+0..base+n_sig_total-1] : signal spotlight dots
        [base+n_sig_total]            : per-FEI recall bar
        [base+n_sig_total+1]          : SDUD highlight
    """
    K         = len(story_feis)
    n_per_fei = n_sig_total + 2
    n_base    = 2 * n_sig_total + 1 + n_sdud
    n_total   = n_base + K * n_per_fei

    recall_idx = 2 * n_sig_total
    sdud_start = recall_idx + 1
    spot_start = sdud_start + n_sdud

    def _vis(agg_sigs, raw_sigs, recall_on, sdud_on, spot_idx):
        v = [False] * n_total
        for i in agg_sigs:
            v[i * 2] = True
        for i in raw_sigs:
            v[i * 2 + 1] = True
        if recall_on:
            v[recall_idx] = True
        for j in range(n_sdud):
            v[sdud_start + j] = sdud_on
        if spot_idx >= 0:
            base = spot_start + spot_idx * n_per_fei
            for s in range(n_per_fei):
                v[base + s] = True
        return v

    buttons = [dict(
        label="All signals",
        method="update",
        args=[{"visible": _vis(list(range(n_sig_total)), [], True, True, -1)}],
    )]
    for sig_i, lbl in sig_labels_with_idx:
        buttons.append(dict(
            label=lbl, method="update",
            args=[{"visible": _vis([sig_i], [sig_i], True, True, -1)}],
        ))
    for k, (fei_id, (fei_name, _)) in enumerate(story_feis.items()):
        short = fei_name.split()[0]
        buttons.append(dict(
            label=f"Spotlight: {short} (FEI {fei_id})",
            method="update",
            args=[{"visible": _vis([], [], False, False, k)}],
        ))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down",
        x=0.0, y=1.10, xanchor="left", yanchor="top",
        showactive=True, bgcolor="white", bordercolor="#DDD", font=dict(size=11),
    )])


def _semiann_quality(q: pd.DataFrame, sig_cols: list[str]) -> pd.DataFrame:
    """Aggregate raw FEI snapshots to semi-annual periods (H1=Jan, H2=Jul).
    Returns period_start, mean signals, n_feis, n_obs."""
    q = q.copy()
    q["period_start"] = q["snapshot_date"].apply(
        lambda d: pd.Timestamp(f"{d.year}-01-01") if d.month <= 6
                  else pd.Timestamp(f"{d.year}-07-01")
    )
    agg_dict: dict = {col: "mean" for col in sig_cols if col in q.columns}
    agg_dict["fei"] = "nunique"
    if "n_obs_total" in q.columns:
        agg_dict["n_obs_total"] = "sum"
    agg = q.groupby("period_start").agg(agg_dict).reset_index()
    agg = agg.rename(columns={"fei": "n_feis"})
    if "n_obs_total" in agg.columns:
        agg = agg.rename(columns={"n_obs_total": "n_obs"})
    else:
        agg["n_obs"] = agg["n_feis"]
    return agg.sort_values("period_start").reset_index(drop=True)


def _fig_case_study(
    quality: pd.DataFrame,
    recalls: pd.DataFrame,
    shortages: pd.DataFrame,
    sdud_mfr: pd.DataFrame,
    recalls_by_fei: pd.DataFrame,
    drug: str = "Metformin",
) -> go.Figure:
    """Three-row case study: quality signals / recall events / SDUD manufacturer volumes.

    Dropdown: signal-filter modes (aggregate + per-FEI dots) AND facility spotlight modes.
    Spotlight mode reveals all 3 signal dots for one FEI + its recall bar + its SDUD line.
    """
    SIGNAL_CFG = [
        ("severity_critmajor_share",  "Critical/Major severity", C["orange"], "solid"),
        ("contamination_llm_share",   "Contamination flag",      C["purple"], "dash"),
        ("data_integrity_llm_share",  "Data integrity flag",     C["teal"],   "dot"),
    ]
    _SDUD_PAL = [C["orange"], C["blue"], C["green"], C["purple"], C["teal"], C["red"], C["gray"]]
    _N_SIG = len(SIGNAL_CFG)  # always 3

    date_start = pd.Timestamp(f"{PANEL_START_YEAR}-01-01")
    date_end   = pd.Timestamp(f"{PANEL_END_YEAR}-12-31")
    q = quality[quality["snapshot_date"].between(date_start, date_end)].copy()
    r = recalls[recalls["month_start"].between(date_start, date_end)].copy()

    sig_cols_present = [col for col, _, _, _ in SIGNAL_CFG
                        if col in q.columns and not q[col].isna().all()]
    agg = _semiann_quality(q, sig_cols_present) if sig_cols_present else pd.DataFrame()

    story_feis = _STORY_FEIS.get(drug, {})
    sdud_mfrs  = sdud_mfr["mfr_name"].unique().tolist() if not sdud_mfr.empty else []

    # Ensure n_sdud >= 1 (need at least one trace in SDUD row for layout)
    if not sdud_mfrs:
        sdud_mfrs = ["(no data)"]

    n_sdud = len(sdud_mfrs)
    vl_color = C["green"] if drug == "Metformin" else C["purple"]

    subplot_titles = [
        f"{drug} FEIs — quality signals, semi-annual average "
        "(bubble = # facilities; use dropdown to drill into signal or spotlight a facility)",
        "Recall events linked to these FEIs (monthly count)",
        "Medicaid utilization by manufacturer — monthly, millions of units",
    ]
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True,
        row_heights=[0.46, 0.27, 0.27],
    )

    # ── AGGREGATE TRACES ── (indices 0..2*_N_SIG-1, always 6 traces) ────────
    for i, (col, label, col_color, dash) in enumerate(SIGNAL_CFG):
        is_present = col in q.columns and not q[col].isna().all()

        if is_present and not agg.empty and col in agg.columns:
            sizes = np.clip(agg["n_feis"] * 3 + 7, 8, 30).values
            cd    = np.column_stack([agg["n_feis"], agg["n_obs"]])
            fig.add_trace(go.Scatter(
                x=agg["period_start"], y=agg[col], name=label,
                mode="lines+markers",
                line=dict(color=col_color, width=2.5, dash=dash),
                marker=dict(size=sizes, sizemode="diameter"),
                connectgaps=False, customdata=cd,
                hovertemplate=(
                    f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.0%}}<br>"
                    "Facilities: %{customdata[0]:.0f} · 483 obs: %{customdata[1]:.0f}"
                    "<extra></extra>"
                ),
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name=label,
                                      visible=False, showlegend=False), row=1, col=1)

        if is_present:
            fig.add_trace(go.Scatter(
                x=q["snapshot_date"], y=q[col],
                name=f"{label} (per FEI)", mode="markers",
                marker=dict(color=col_color, size=9, opacity=0.40,
                            symbol="circle-open", line=dict(width=1.5)),
                visible=False, showlegend=False,
                customdata=q[["fei"]].values,
                hovertemplate=(
                    f"<b>%{{x|%b %Y}}</b> · FEI %{{customdata[0]}}<br>"
                    f"{label}: %{{y:.0%}}<br><i>Individual facility snapshot</i>"
                    "<extra></extra>"
                ),
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name=f"{label}_raw",
                                      visible=False, showlegend=False), row=1, col=1)

    # ── RECALL BAR ── (index 2*_N_SIG = 6) ─────────────────────────────────
    if not r.empty:
        fig.add_trace(go.Bar(
            x=r["month_start"], y=r["n_recalls"],
            name="Recalls (all FEIs)", marker_color=C["red"], opacity=0.85,
            showlegend=False,
            hovertemplate="<b>%{x|%b %Y}</b><br>Recalls: %{y}<extra></extra>",
        ), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=[], y=[], name="no_recalls", showlegend=False), row=2, col=1)

    # ── SDUD REGULAR TRACES ── (indices 7..6+n_sdud) ────────────────────────
    has_real_sdud = not sdud_mfr.empty
    for j, mfr in enumerate(sdud_mfrs):
        if not has_real_sdud:
            fig.add_trace(go.Scatter(x=[], y=[], name="no_sdud",
                                      showlegend=False), row=3, col=1)
            continue
        sub = sdud_mfr[sdud_mfr["mfr_name"] == mfr].sort_values("date")
        is_story = any(kw.lower() in mfr.lower()
                       for _, (_, kw) in story_feis.items())
        color = _SDUD_PAL[j % len(_SDUD_PAL)]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["units"], name=mfr, mode="lines",
            line=dict(color=color, width=2.5 if is_story else 1.5),
            opacity=1.0 if is_story else 0.65,
            hovertemplate=f"<b>{mfr}</b><br>%{{x|%b %Y}}: %{{y:.2f}}M units<extra></extra>",
        ), row=3, col=1)

    # ── SPOTLIGHT TRACES ── (_N_SIG+2 traces per story FEI) ─────────────────
    n_per_fei = _N_SIG + 2  # 3 signal dots + 1 recall bar + 1 SDUD highlight
    for k, (fei_id, (fei_name, sdud_kw)) in enumerate(story_feis.items()):
        q_fei = q[q["fei"] == fei_id].copy()

        # Signal dot traces
        for col, label, col_color, dash in SIGNAL_CFG:
            is_present = col in q_fei.columns and not q_fei[col].isna().all()
            if is_present and not q_fei.empty:
                fig.add_trace(go.Scatter(
                    x=q_fei["snapshot_date"], y=q_fei[col],
                    name=f"{fei_name.split()[0]}: {label}", mode="markers",
                    marker=dict(color=col_color, size=14, opacity=0.90,
                                symbol="diamond", line=dict(width=2, color="white")),
                    visible=False, showlegend=True,
                    hovertemplate=(
                        f"<b>%{{x|%b %Y}}</b><br><b>{fei_name}</b><br>"
                        f"{label}: %{{y:.0%}}<br>FEI {fei_id}"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=[], y=[], name=f"spot_{col}_{k}",
                                          visible=False, showlegend=False), row=1, col=1)

        # Per-FEI recall bar
        rfei_mask = (
            recalls_by_fei["fei"].astype(str) == str(fei_id)
            if not recalls_by_fei.empty else pd.Series([], dtype=bool)
        )
        r_fei = recalls_by_fei[rfei_mask].copy() if not recalls_by_fei.empty else pd.DataFrame()
        if not r_fei.empty:
            fig.add_trace(go.Bar(
                x=r_fei["month_start"], y=r_fei["n_recalls"],
                name=f"Recalls — {fei_name.split()[0]}",
                marker_color=C["red"], opacity=0.9,
                visible=False, showlegend=False,
                hovertemplate=(
                    f"<b>%{{x|%b %Y}}</b><br>{fei_name.split()[0]} recalls: %{{y}}<extra></extra>"
                ),
            ), row=2, col=1)
        else:
            fig.add_trace(go.Bar(x=[], y=[], name=f"no_recalls_{k}",
                                  visible=False, showlegend=False), row=2, col=1)

        # SDUD highlight trace
        if has_real_sdud:
            sdud_match = sdud_mfr[
                sdud_mfr["mfr_name"].str.lower().str.contains(sdud_kw.lower(), na=False)
            ]
            if not sdud_match.empty:
                mfr_agg = sdud_match.groupby("date")["units"].sum().reset_index().sort_values("date")
                mfr_idx = next(
                    (j for j, m in enumerate(sdud_mfrs) if sdud_kw.lower() in m.lower()), 0
                )
                hi_color = _SDUD_PAL[mfr_idx % len(_SDUD_PAL)]
                fig.add_trace(go.Scatter(
                    x=mfr_agg["date"], y=mfr_agg["units"],
                    name=f"{fei_name.split()[0]} (SDUD)",
                    mode="lines", line=dict(color=hi_color, width=4.0),
                    opacity=1.0, visible=False, showlegend=True,
                    hovertemplate=(
                        f"<b>{fei_name}</b><br>%{{x|%b %Y}}: %{{y:.2f}}M units<extra></extra>"
                    ),
                ), row=3, col=1)
            else:
                fig.add_trace(go.Scatter(x=[], y=[], name=f"no_sdud_{k}",
                                          visible=False, showlegend=False), row=3, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name=f"no_sdud_{k}",
                                      visible=False, showlegend=False), row=3, col=1)

    # ── VLINES & ANNOTATIONS ─────────────────────────────────────────────────
    for _, sh_row in shortages.iterrows():
        yr = int(sh_row["year"])
        for rn in [1, 2, 3]:
            fig.add_vline(x=f"{yr}-01-01", line_dash="dot",
                         line_color=vl_color, line_width=1.5, row=rn, col=1)
        fig.add_annotation(
            x=f"{yr}-01-01", y=1.08, yref="paper",
            text=f"⚠ Shortage {yr}",
            showarrow=False, font=dict(color=vl_color, size=9), align="center",
        )

    if drug == "Metformin":
        fig.add_annotation(
            x="2020-06-01", y=14, yref="y2",
            text="<b>June 2020</b><br>14 NDMA recalls",
            showarrow=True, arrowhead=2, arrowcolor=C["red"],
            ax=65, ay=-35, font=dict(color=C["red"], size=10),
            bgcolor="rgba(255,255,255,0.92)", bordercolor=C["red"], borderwidth=1,
        )
        if not agg.empty and "severity_critmajor_share" in agg.columns:
            h1_2020 = agg[agg["period_start"] == pd.Timestamp("2020-01-01")]
            if not h1_2020.empty:
                sev = float(h1_2020["severity_critmajor_share"].iloc[0])
                n_f = int(h1_2020["n_feis"].iloc[0])
                n_o = int(h1_2020["n_obs"].iloc[0])
                fig.add_annotation(
                    x="2020-01-01", y=sev, yref="y",
                    text=f"<b>H1 2020</b><br>{n_f} facilities · {n_o} obs<br>Quality spike",
                    showarrow=True, arrowhead=2, arrowcolor=C["orange"],
                    ax=-115, ay=-40, font=dict(color=C["orange"], size=10),
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=C["orange"], borderwidth=1,
                )
        if has_real_sdud:
            fig.add_annotation(
                x="2020-07-01", y=0.3, yref="y3",
                text="<b>Post-recall collapse</b><br>Amneal exits ~2022",
                showarrow=True, arrowhead=2, arrowcolor=C["orange"],
                ax=80, ay=-45, font=dict(color=C["orange"], size=9),
                bgcolor="rgba(255,255,255,0.88)", bordercolor=C["orange"], borderwidth=1,
            )
    elif drug == "Lisinopril":
        if has_real_sdud:
            fig.add_annotation(
                x="2022-04-01", y=0.03, yref="y3",
                text="<b>Mfr. 18506 exits</b><br>Apr 2022",
                showarrow=True, arrowhead=2, arrowcolor=C["red"],
                ax=60, ay=-45, font=dict(color=C["red"], size=9),
                bgcolor="rgba(255,255,255,0.88)", bordercolor=C["red"], borderwidth=1,
            )
        for _, sh_row in shortages.iterrows():
            yr = int(sh_row["year"])
            fig.add_annotation(
                x=f"{yr}-07-01", y=1.14, yref="paper",
                text="Business Decision",
                showarrow=False, font=dict(color=C["purple"], size=8), align="center",
            )

    if not sig_cols_present:
        fig.add_annotation(
            x=0.5, y=0.80, xref="paper", yref="paper",
            text=f"<i>No LLM text features found for {drug} FEIs<br>"
                 f"(483 pipeline covers 98/129 Valisure FEIs)</i>",
            showarrow=False, font=dict(color="#888", size=12), align="center",
        )

    # ── COMBINED DROPDOWN ────────────────────────────────────────────────────
    sig_labels_with_idx = [
        (i, label)
        for i, (col, label, _, _) in enumerate(SIGNAL_CFG)
        if col in q.columns and not q[col].isna().all()
    ]
    _build_combined_dropdown(fig, _N_SIG, n_sdud, story_feis, sig_labels_with_idx)

    # ── LAYOUT ───────────────────────────────────────────────────────────────
    fig.update_layout(
        height=840,
        margin=dict(l=10, r=10, t=70, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="% of 483 obs flagged", tickformat=".0%",
                   range=[-0.05, 1.15], gridcolor="#F0F0F0"),
        yaxis2=dict(title="# Recalls", gridcolor="#F0F0F0"),
        yaxis3=dict(title="Units (M/month)", gridcolor="#F0F0F0"),
        xaxis3=dict(
            title="Month / Half-year", gridcolor="#F0F0F0",
            tickformat="%b %Y", dtick="M6", tickangle=-30,
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#E0E0E0", borderwidth=1, font=dict(size=11)),
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
    cs_sdud: pd.DataFrame | None = None,
    cs_rfei: pd.DataFrame | None = None,
    cs_b_quality: pd.DataFrame | None = None,
    cs_b_recalls: pd.DataFrame | None = None,
    cs_b_shortages: pd.DataFrame | None = None,
    cs_b_sdud: pd.DataFrame | None = None,
    cs_b_rfei: pd.DataFrame | None = None,
) -> str:
    auc_no   = ablation.iloc[0]["auc"]
    auc_yes  = ablation.iloc[1]["auc"]
    lift_abs = auc_yes - auc_no
    lift_rel = lift_abs / max(auc_no, 0.001) * 100

    total_recalls   = int(recalls_by_drug["n_recalls"].sum())
    total_shortages = int(shortages["n"].sum())
    path_b_share    = int(
        shortages[shortages["path"] == "Path B — Discontinuation/Exit"]["n"].sum()
        / max(total_shortages, 1) * 100
    )

    _empty = pd.DataFrame()
    cs_div = ""
    if cs_quality is not None and cs_recalls is not None and cs_shortages is not None:
        cs_div = _div(
            _fig_case_study(
                cs_quality, cs_recalls, cs_shortages,
                cs_sdud if cs_sdud is not None else _empty,
                cs_rfei  if cs_rfei  is not None else _empty,
                "Metformin",
            ),
            "fig_case_a",
        )

    cs_b_div = ""
    if cs_b_quality is not None and cs_b_recalls is not None and cs_b_shortages is not None:
        cs_b_div = _div(
            _fig_case_study(
                cs_b_quality, cs_b_recalls, cs_b_shortages,
                cs_b_sdud if cs_b_sdud is not None else _empty,
                cs_b_rfei  if cs_b_rfei  is not None else _empty,
                "Lisinopril",
            ),
            "fig_case_b",
        )

    divs = {
        "funnel":   _div(_fig_funnel(cov),                                   "fig_funnel"),
        "supply":   _div(_fig_supply_concentration(supply, recalls_by_drug),  "fig_supply"),
        "paths":    _div(_fig_path_breakdown(shortages),                      "fig_paths"),
        "timeline": _div(_fig_timeline(recalls_by_drug, shortages),           "fig_timeline"),
        "case_a":   cs_div,
        "case_b":   cs_b_div,
        "model":    _div(_fig_model_evidence(fi, ablation),                   "fig_model"),
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
  .hero p   {{ color: #A0AEC0; font-size: 13.5px; line-height: 1.65; max-width: 800px; margin-bottom: 24px; }}

  /* ── Causal path flowchart ── */
  .flow-diagram {{ display: flex; flex-direction: column; align-items: center;
                   gap: 0; margin-bottom: 28px; }}
  .flow-row {{ display: flex; justify-content: center; align-items: center; width: 100%; }}
  .flow-split {{ gap: 60px; align-items: flex-start; margin: 0; }}
  .flow-side {{ display: flex; flex-direction: column; align-items: center; gap: 0; }}
  .flow-box {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.18);
                border-radius: 10px; padding: 12px 26px; text-align: center;
                min-width: 180px; max-width: 230px; }}
  .flow-box.flow-start {{ border-color: #4A90D9; border-width: 2px; }}
  .flow-box.flow-a     {{ border-color: #D94A4A; border-width: 2px; }}
  .flow-box.flow-b     {{ border-color: #7B5EA7; border-width: 2px; }}
  .flow-box.flow-end   {{ border-color: #3DAA6E; border-width: 2px;
                           background: rgba(61,170,110,0.10); }}
  .flow-title {{ color: #F0F0F0; font-size: 13px; font-weight: 600; display: block; }}
  .flow-sub   {{ color: #A0AEC0; font-size: 11px; display: block; margin-top: 3px; }}
  .flow-arrow-v {{ color: #A0AEC0; font-size: 20px; line-height: 1; padding: 3px 0;
                    text-align: center; }}
  .flow-path-badge {{ font-size: 11px; font-weight: 600; letter-spacing: 0.6px;
                       text-transform: uppercase; margin-bottom: 6px; padding: 3px 10px;
                       border-radius: 4px; }}
  .badge-a {{ color: #D94A4A; background: rgba(217,74,74,0.12); }}
  .badge-b {{ color: #7B5EA7; background: rgba(123,94,167,0.12); }}

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

  footer {{ text-align: center; color: #9AA5B1; font-size: 12px;
             padding: 22px 0 30px; border-top: 1px solid #DEE2E6; margin-top: 20px; }}

  @media (max-width: 860px) {{
    .flow-split {{ flex-direction: column; align-items: center; gap: 20px; }}
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
  <a href="#shortage-analysis">③ Shortage Analysis</a>
  <a href="#timeline">④ Timeline</a>
  <a href="#case-a">⑤ Path A Case Study</a>
  <a href="#case-b">⑥ Path B Case Study</a>
  <a href="#model">⑦ Model Evidence</a>
</nav>

<div class="hero">
  <h1>Manufacturing Quality Failures → Drug Shortages</h1>
  <p>Quality failures at generic drug manufacturing facilities (FEIs) can trigger shortages
     through two distinct causal paths. This dashboard follows both paths using FDA 483
     inspection text, recall records, and UUtah shortage data across 14 Valisure-validated
     APIs (2015–2024).</p>

  <!-- Causal path flowchart -->
  <div class="flow-diagram">
    <div class="flow-row">
      <div class="flow-box flow-start">
        <span class="flow-title">FDA 483 Inspection</span>
        <span class="flow-sub">Contamination · Lab controls · Data integrity</span>
      </div>
    </div>
    <div class="flow-row"><div class="flow-arrow-v">↓</div></div>
    <div class="flow-row"><div class="flow-box" style="border-color:rgba(255,255,255,0.3);">
      <span class="flow-title">Quality Failure Documented</span>
      <span class="flow-sub">LLM extracts severity, scope, root cause</span>
    </div></div>
    <div class="flow-row" style="gap:120px; margin:2px 0;">
      <span class="flow-arrow-v" style="font-size:16px; color:#D94A4A;">↙</span>
      <span class="flow-arrow-v" style="font-size:16px; color:#7B5EA7;">↘</span>
    </div>
    <div class="flow-row flow-split">
      <div class="flow-side">
        <div class="flow-path-badge badge-a">Path A — Recall</div>
        <div class="flow-box flow-a">
          <span class="flow-title">Recall Issued</span>
          <span class="flow-sub">FDA enforcement action</span>
        </div>
        <div class="flow-arrow-v">↓</div>
        <div class="flow-box flow-a">
          <span class="flow-title">Supply Gap</span>
          <span class="flow-sub">Product pulled from market</span>
        </div>
      </div>
      <div class="flow-side">
        <div class="flow-path-badge badge-b">Path B — Silent Exit</div>
        <div class="flow-box flow-b">
          <span class="flow-title">Business Decision</span>
          <span class="flow-sub">Remediation cost &gt; profit margin</span>
        </div>
        <div class="flow-arrow-v">↓</div>
        <div class="flow-box flow-b">
          <span class="flow-title">Market Exit</span>
          <span class="flow-sub">No recall — silent supply drop</span>
        </div>
      </div>
    </div>
    <div class="flow-row" style="gap:120px; margin:2px 0;">
      <span class="flow-arrow-v" style="font-size:16px; color:#3DAA6E;">↘</span>
      <span class="flow-arrow-v" style="font-size:16px; color:#3DAA6E;">↙</span>
    </div>
    <div class="flow-row">
      <div class="flow-box flow-end">
        <span class="flow-title">Drug Shortage</span>
        <span class="flow-sub">Patients cannot access medication</span>
      </div>
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="val">14</div><div class="lbl">APIs tracked<br>(Valisure-tested)</div></div>
    <div class="kpi"><div class="val">{cov['n_feis_llm']}</div><div class="lbl">FEIs with<br>LLM text features</div></div>
    <div class="kpi"><div class="val">{total_recalls}</div><div class="lbl">Recall events<br>linked to FEIs</div></div>
    <div class="kpi"><div class="val">{total_shortages}</div><div class="lbl">Shortage events<br>(UUtah, 2015–2024)</div></div>
    <div class="kpi"><div class="val">{path_b_share}%</div><div class="lbl">Shortages labeled<br>discontinuation (Path B)</div></div>
    <div class="kpi"><div class="val">+{lift_rel:.0f}%</div><div class="lbl">AUC lift from<br>483 text features</div></div>
  </div>
</div>

<div class="page">

<!-- ① Data Coverage -->
<section id="coverage">
  <div class="section-head">
    <div class="section-title">① Data Coverage</div>
    <div class="section-sub">
      Starting from 129 Valisure FEIs: 127 matched in Redica inspection database,
      {cov['n_sites_with_483']} with ≥1 483 issued, {cov['n_feis_llm']} with
      {cov['n_obs_llm']:,} observations scored by the LLM pipeline.
    </div>
  </div>
  <div class="card">{divs['funnel']}</div>
</section>

<!-- ② Supply Concentration -->
<section id="supply">
  <div class="section-head">
    <div class="section-title">② Supply Concentration</div>
    <div class="section-sub">
      Left: how many FEIs produce each drug (March 2026 Valisure mapping) — fewer producers = higher
      concentration risk. Orange = drug has at least one injectable formulation in the Orange Book;
      this is not exclusive (Vancomycin is given IV for bloodstream infections but orally for C. diff;
      Metoprolol is primarily oral but has an IV cardiac form).
      Right: total recall events linked to these FEIs (2015–2024).
    </div>
  </div>
  <div class="card">{divs['supply']}</div>
</section>

<!-- ③ Shortage Analysis -->
<section id="shortage-analysis">
  <div class="section-head">
    <div class="section-title">③ Shortage Analysis</div>
    <div class="section-sub">
      How many shortage events fall under each causal path?
      <b>Classification method:</b> The UUtah shortage database includes a free-text
      <i>Reason</i> field. We classify events by keyword matching — "manufacturing / CGMP /
      quality / recall" → Path A; "discontinued / business decision / market exit" → Path B.
      This is observational labeling from a self-reported field and does not establish causation.
      <b>~50% of events in our 13-drug subset are reported as "Unknown"</b> — the true
      path split is uncertain and the chart below reflects only events with a recorded reason.
      <br><br>
      Path B evidence in this panel: Lisinopril 2020 (Business Decision), Potassium chloride 2019
      (combo product discontinued), Vancomycin 2002 (Business Decision, outside panel window).
    </div>
  </div>
  <div class="card">{divs['paths']}</div>
</section>

<!-- ④ Recall & Shortage Timeline (per-drug) -->
<section id="timeline">
  <div class="section-head">
    <div class="section-title">④ Recall &amp; Shortage Timeline — Per Drug</div>
    <div class="section-sub">
      Use the dropdown to select a drug and see its recall events (top) and shortage events
      (bottom) by year. No path coloring here — just the raw counts to show when events occurred.
      Note whether recalls precede shortages (Path A pattern) or shortages appear with no
      preceding recall spike (Path B pattern).
    </div>
  </div>
  <div class="card">{divs['timeline']}</div>
</section>

<!-- ⑤ Case Study A: Metformin — Path A -->
<section id="case-a">
  <div class="section-head">
    <div class="section-title">⑤ Case Study — Path A: Metformin NDMA 2020</div>
    <div class="section-sub">
      Full causal chain for Metformin:
      (1) <b>H1 2020</b>: Redica snapshots show Critical/Major severity = 1.0 across 3 Metformin FEIs;
      (2) <b>June 2020</b> (~5 months later): 14 recalls citing NDMA impurity;
      (3) <b>Post-recall collapse</b>: Amneal Pharmaceuticals (largest Metformin manufacturer,
      peak 12M units/month) volumes plummet and exit the market by ~2022;
      (4) <b>2021 &amp; 2024</b>: UUtah records Metformin shortage events.
      <br><br>
      Use the <b>dropdown</b> to: (a) select a signal to reveal per-facility inspection dots,
      or (b) <i>Spotlight: Amneal</i> to isolate that facility's signals, its recalls, and its
      Medicaid volume collapse. Zero-share observations are visible below the axis in this view.
      <i>Bottom panel</i>: Medicaid units dispensed monthly (millions) by manufacturer.
      Green dashed lines = shortage years (UUtah).
    </div>
  </div>
  <div class="card">{divs['case_a']}</div>
</section>

<!-- ⑥ Case Study B: Lisinopril — Path B -->
<section id="case-b">
  <div class="section-head">
    <div class="section-title">⑥ Case Study — Path B: Lisinopril 2020 (Business Decision)</div>
    <div class="section-sub">
      Path B contrast: the 2020 Lisinopril shortage (UUtah) was attributed to a "Business Decision"
      — not a recall. Quality signals at available Lisinopril FEIs (including Lupin, which had
      3 OAI inspections) were elevated before 2020.
      <br><br>
      <b>Market exit evidence (bottom panel)</b>: an unnamed manufacturer (Labeler 18506)
      exited the Lisinopril market by April 2022, contributing to supply thinning.
      Lupin (the facility with quality data, peak 31M units/month) remained active —
      illustrating that quality risk does not always lead to a recall; sometimes the business
      decision to exit is the mechanism.
      <br><br>
      Use the dropdown to <i>Spotlight: Lupin (FEI 3007549629)</i> to see that facility's
      quality signals and SDUD volume in isolation. Purple dashed line = shortage year.
    </div>
  </div>
  <div class="card">{divs['case_b']}</div>
</section>

<!-- ⑦ Model Evidence -->
<section id="model">
  <div class="section-head">
    <div class="section-title">⑦ Model Evidence: 483 Text Improves Recall Prediction</div>
    <div class="section-sub">
      <b>What this model predicts:</b> whether a FEI will have a <i>recall event</i> in year t+1,
      given features measured in year t — the intermediate step in Path A. This is not a direct
      shortage prediction; the link from recall to shortage depends on supply concentration (Section ②).
      <br><br>
      L2 logistic regression, GroupKFold CV by FEI (a facility never appears in both train and test).
      Adding LLM-extracted 483 text features improves AUC from {auc_no:.3f} to {auc_yes:.3f}
      (+{lift_abs:.3f}, +{lift_rel:.0f}% relative lift). Right panel: RF feature importance (top 12).
    </div>
  </div>
  <div class="card">{divs['model']}</div>
</section>

</div>

<footer>
  Generated {today} &nbsp;·&nbsp; Drug Shortage Project &nbsp;·&nbsp; NC State University<br>
  <span style="font-size:11px; color:#BEC8D0;">
    483 text features: Redica database (2018–2026), LLM pipeline.
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

    cs_q, cs_r, cs_s, cs_sdud, cs_rfei = _load_case_study("Metformin")
    log.info("Case study A (Metformin): %d quality rows, %d recall months, %d shortage years, "
             "%d SDUD mfr-months, %d fei-recall rows",
             len(cs_q), len(cs_r), len(cs_s), len(cs_sdud), len(cs_rfei))

    cs_b_q, cs_b_r, cs_b_s, cs_b_sdud, cs_b_rfei = _load_case_study("Lisinopril")
    log.info("Case study B (Lisinopril): %d quality rows, %d recall months, %d shortage years, "
             "%d SDUD mfr-months, %d fei-recall rows",
             len(cs_b_q), len(cs_b_r), len(cs_b_s), len(cs_b_sdud), len(cs_b_rfei))

    log.info("Building HTML…")
    html = build_html(
        cov, supply, recalls, shortage, fi, abl,
        cs_quality=cs_q, cs_recalls=cs_r, cs_shortages=cs_s,
        cs_sdud=cs_sdud, cs_rfei=cs_rfei,
        cs_b_quality=cs_b_q, cs_b_recalls=cs_b_r, cs_b_shortages=cs_b_s,
        cs_b_sdud=cs_b_sdud, cs_b_rfei=cs_b_rfei,
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


if __name__ == "__main__":
    main()
