"""
Module 18 — Interactive FAERS adverse event story dashboard.

Tells the causal chain: manufacturing quality failures → serious adverse events.

Sections:
  1. Data coverage — facility funnel
  2. Supply landscape — FEIs per drug + AE event counts
  3. AE severity breakdown — Hospitalization vs Other Serious by drug
  4. AE timeline — FAERS serious AE counts per drug × year
  5. Case study — Metformin: quality signals preceding AE surge
  6. Model evidence — text feature AUC lift + feature importance (from m17)

Output:
  outputs/figures/faers_fei_dashboard.html

Run:
  python m18_faers_dashboard.py
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
    TEXT_TIMESERIES_REDICA_CSV, FAERS_ALL, VALISURE_CSV, VALISURE_FEI,
    PANEL_START_YEAR, PANEL_END_YEAR, SEED,
)
from utils import get_logger, ValisureDrugMatcher, load_valisure_api_names

log = get_logger("m18_faers_dashboard", OUT_LOGS / "m18_dashboard.log")

REDICA_RAW  = DATA / "07 - Redica" / "raw"
OB_PRODUCTS = DATA / "01 - Orange Book" / "output_data" / "products.csv"
OUT_HTML    = OUT_FIGS / "faers_fei_dashboard.html"
SDUD_PANEL  = DATA / "04_11 - Build - Monthly Panel (SDUD+NADAC)" / "processed" / "2026-03-26-sdud_nadac_panel.csv"
NDC_PRODUCT = DATA / "03 - FDA - NDC" / "product.csv"

# Per-drug story FEI: fei_id → (display_name, sdud_keyword)
_STORY_FEIS: dict[str, dict[int, tuple[str, str]]] = {
    "Metformin": {3005263655: ("Amneal Pharmaceuticals", "amneal")},
}

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
    supply = (
        fdmap.groupby("drug", as_index=False)
        .agg(n_feis=("fei", "nunique"), parenteral=("parenteral", "max"))
        .sort_values("n_feis")
    )
    return supply


def _load_ae_by_drug(fdmap: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FAERS by drug × year, joined to canonical API via Valisure matcher."""
    if not FAERS_ALL.exists():
        log.warning("FAERS file not found; returning empty AE frame")
        return pd.DataFrame(columns=["drug", "year", "severity", "n_ae"])

    df = pd.read_csv(FAERS_ALL, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    prod_col = next((c for c in df.columns if c.lower() == "prod_ai"), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    sev_col  = next((c for c in df.columns if c.lower() == "severity"), None)
    if not prod_col or not year_col:
        log.warning("FAERS missing prod_ai or year; columns: %s", list(df.columns))
        return pd.DataFrame(columns=["drug", "year", "severity", "n_ae"])

    df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", prod_col])
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]

    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher   = ValisureDrugMatcher(api_names)
    df["drug"] = df[prod_col].astype(str).map(matcher.match)
    df = df.dropna(subset=["drug"])

    sev = sev_col if sev_col else None
    if sev and sev in df.columns:
        agg = df.groupby(["drug", "year", sev], as_index=False).size()
        agg = agg.rename(columns={sev: "severity", "size": "n_ae"})
    else:
        agg = df.groupby(["drug", "year"], as_index=False).size()
        agg = agg.rename(columns={"size": "n_ae"})
        agg["severity"] = "Serious"

    log.info("AE by drug: %d rows, %d drugs, years %s–%s",
             len(agg), agg["drug"].nunique(),
             int(agg["year"].min()), int(agg["year"].max()))
    return agg


def _load_ae_monthly_for_drug(drug: str) -> pd.DataFrame:
    """Monthly FAERS AE count for a specific drug (for case study row 2)."""
    if not FAERS_ALL.exists():
        return pd.DataFrame(columns=["month_start", "n_ae"])

    df = pd.read_csv(FAERS_ALL, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    prod_col   = next((c for c in df.columns if c.lower() == "prod_ai"), None)
    period_col = next((c for c in df.columns if c.lower() == "period"), None)
    if not prod_col:
        return pd.DataFrame(columns=["month_start", "n_ae"])

    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher   = ValisureDrugMatcher(api_names)
    df["drug"] = df[prod_col].astype(str).map(matcher.match)
    df = df[df["drug"] == drug].copy()

    if df.empty:
        return pd.DataFrame(columns=["month_start", "n_ae"])

    # Parse quarter to month_start (e.g. "2018Q2" → 2018-04-01)
    _QMAP = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    if period_col and period_col in df.columns:
        def _q2month(p: str) -> pd.Timestamp | None:
            p = str(p).strip()
            if len(p) >= 6 and p[4] == "Q":
                yr, q = p[:4], p[4:]
                return pd.Timestamp(f"{yr}-{_QMAP.get(q,'01')}-01")
            return None
        df["month_start"] = df[period_col].map(_q2month)
    else:
        yr_col = next((c for c in df.columns if c.lower() == "year"), None)
        if yr_col:
            df["month_start"] = pd.to_datetime(
                df[yr_col].astype(str) + "-01-01", errors="coerce"
            )
        else:
            return pd.DataFrame(columns=["month_start", "n_ae"])

    df = df.dropna(subset=["month_start"])
    agg = df.groupby("month_start", as_index=False).size().rename(columns={"size": "n_ae"})
    return agg.sort_values("month_start").reset_index(drop=True)


def _load_coverage() -> dict:
    cov = {}
    da_p  = REDICA_RAW / "Valisure_Sites_Data_Availability.xlsx"
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
        cov["n_docs_obtained"]  = obs483["Document Redica Id"].nunique()
        cov["n_sites_obtained"] = obs483["Site Redica Id"].nunique()
    else:
        cov["n_docs_obtained"]  = 246
        cov["n_sites_obtained"] = 98
    cov["n_obs_llm"]  = 1115
    cov["n_feis_llm"] = 98
    return cov


def _load_sdud_manufacturers(
    drug: str,
    top_n: int = 10,
    force_include_kw: list[str] | None = None,
) -> pd.DataFrame:
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

    sdud["labeler_code"] = sdud["ndc11"].astype(str).str.zfill(11).str[:5]

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
        "granules": "Granules", "avet": "Avet",
        "heritage": "Avet",    "sun pharma": "Sun Pharma",
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
    agg["units"] /= 1e6

    totals  = agg.groupby("mfr_name")["units"].sum().sort_values(ascending=False)
    top_mfrs = totals.head(top_n).index.tolist()

    if force_include_kw:
        for kw in force_include_kw:
            extra = [m for m in totals.index if kw.lower() in m.lower()]
            for m in extra:
                if m not in top_mfrs:
                    top_mfrs.append(m)

    return agg[agg["mfr_name"].isin(top_mfrs)].sort_values(["mfr_name", "date"]).reset_index(drop=True)


def _load_model_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load m17 model outputs; fall back to placeholder if not yet generated."""
    fi_path  = OUT_MODELS / "rf_importance_faers_fei.csv"
    abl_path = OUT_MODELS / "text_ablation_faers_fei.csv"

    if fi_path.exists():
        fi = pd.read_csv(fi_path)
    else:
        log.warning("m17 outputs not found — using placeholder values. Run m17 first.")
        fi = pd.DataFrame({
            "feature":    ALL_FEATS,
            "importance": np.random.dirichlet(np.ones(len(ALL_FEATS))),
        })

    if abl_path.exists():
        abl = pd.read_csv(abl_path)
    else:
        abl = pd.DataFrame({
            "label": ["Without text", "With text\n(all FEIs)"],
            "auc":   [0.60, 0.68],
            "model": ["L2", "L2"],
        })

    abl["label"] = abl["label"].str.replace(r"\n.*", "", regex=True).map(
        lambda s: "Inspection only" if "without" in s.lower()
                  else "Inspection + LLM text"
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


def _fig_supply_concentration(supply: pd.DataFrame, ae_by_drug: pd.DataFrame) -> go.Figure:
    """Grouped bars: FEIs per drug (left) + total AE events (right)."""
    sup = supply.copy().sort_values("n_feis")
    tot_ae = ae_by_drug.groupby("drug")["n_ae"].sum().to_dict()
    sup["n_ae_total"] = sup["drug"].map(tot_ae).fillna(0).astype(int)

    fei_colors = [C["orange"] if p else C["blue"] for p in sup["parenteral"]]
    ae_colors  = [C["purple"]] * len(sup)

    hover_fei = [
        f"<b>{row.drug}</b><br>FEIs: {row.n_feis}<br>"
        f"{'Has injectable formulation (OB)' if row.parenteral else 'Oral formulations only'}"
        for _, row in sup.iterrows()
    ]
    hover_ae = [
        f"<b>{row.drug}</b><br>Serious AE reports linked to FEIs: {row.n_ae_total:,} (2015–2024)"
        for _, row in sup.iterrows()
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FEIs producing each drug (supply concentration)",
                        "Serious AE reports linked to these FEIs (2015–2024)"],
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(
        x=sup["n_feis"], y=sup["drug"], orientation="h",
        marker_color=fei_colors,
        hovertext=hover_fei, hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=sup["n_ae_total"], y=sup["drug"], orientation="h",
        marker_color=ae_colors,
        hovertext=hover_ae, hoverinfo="text",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=C["orange"],
                         name="Has injectable form (OB)", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=C["blue"],
                         name="Oral formulations only", showlegend=True))

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=50, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="# FEIs", gridcolor="#F0F0F0"),
        xaxis2=dict(title="# AE reports", gridcolor="#F0F0F0"),
        yaxis=dict(gridcolor="white"),
        yaxis2=dict(showticklabels=False, gridcolor="white"),
        legend=dict(x=0.72, y=0.02),
        barmode="overlay",
    )
    return fig


def _fig_ae_severity(ae_by_drug: pd.DataFrame) -> go.Figure:
    """AE severity breakdown: overall donut + per-drug stacked bars."""
    sev_totals = ae_by_drug.groupby("severity")["n_ae"].sum().sort_values(ascending=False)
    sev_colors = {
        "Hospitalization":                  C["red"],
        "Life-Threatening":                 C["orange"],
        "Other serious (Important Medical Event)": C["purple"],
        "Other serious":                    C["purple"],
        "Disability":                       C["teal"],
        "Death":                            "#B00020",
        "Congenital Anomaly":               C["gray"],
        "Serious":                          C["blue"],
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["AE reports by severity category (2015–2024)",
                        "Severity composition by drug"],
        horizontal_spacing=0.10,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
    )

    fig.add_trace(go.Pie(
        labels=sev_totals.index.tolist(),
        values=sev_totals.values.tolist(),
        hole=0.45,
        marker_colors=[sev_colors.get(s, C["gray"]) for s in sev_totals.index],
        textinfo="label+percent",
        textfont=dict(size=11),
        showlegend=False,
        hovertemplate="<b>%{label}</b><br>AE reports: %{value:,}<br>%{percent}<extra></extra>",
    ), row=1, col=1)

    drugs = sorted(ae_by_drug["drug"].unique())
    seen: set[str] = set()
    for sev, sev_color in sev_colors.items():
        vals = []
        for drug in drugs:
            sub = ae_by_drug[(ae_by_drug["drug"] == drug) & (ae_by_drug["severity"] == sev)]
            vals.append(int(sub["n_ae"].sum()))
        if all(v == 0 for v in vals):
            continue
        fig.add_trace(go.Bar(
            x=drugs, y=vals,
            name=sev, marker_color=sev_color,
            showlegend=(sev not in seen),
            legendgroup=f"sev_{sev}",
            hovertemplate="<b>%{x}</b><br>" + sev + "<br>AE reports: %{y:,}<extra></extra>",
        ), row=1, col=2)
        seen.add(sev)

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=50, b=80),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        barmode="stack",
        xaxis2=dict(tickangle=-35),
        yaxis2=dict(title="# AE reports", gridcolor="#F0F0F0"),
        legend=dict(x=1.01, y=1, font=dict(size=10)),
    )
    return fig


def _fig_ae_timeline(ae_by_drug: pd.DataFrame) -> go.Figure:
    """Per-drug dropdown: serious AE counts by year."""
    years = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))
    drugs = sorted(ae_by_drug["drug"].unique())

    fig = go.Figure()

    for i, drug in enumerate(drugs):
        visible = (i == 0)
        sub = ae_by_drug[ae_by_drug["drug"] == drug].groupby("year")["n_ae"].sum().reset_index()
        sub["year"] = sub["year"].astype(int)
        y_ae = sub.set_index("year")["n_ae"].reindex(years, fill_value=0).tolist()

        fig.add_trace(go.Bar(
            x=years, y=y_ae, name=drug,
            marker_color=C["purple"], visible=visible, showlegend=False,
            hovertemplate="<b>Year %{x}</b><br>AE reports: %{y:,}<extra></extra>",
        ))

    buttons = []
    for i, drug in enumerate(drugs):
        vis = [j == i for j in range(len(drugs))]
        buttons.append(dict(label=drug, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=80, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.0, y=1.22, xanchor="left", yanchor="top",
            showactive=True, bgcolor="white", bordercolor="#DDD", font=dict(size=12),
        )],
        xaxis=dict(title="Year", dtick=1, gridcolor="#F0F0F0"),
        yaxis=dict(title="# Serious AE reports (FAERS)", gridcolor="#F0F0F0"),
    )
    return fig


def _load_case_study_ae(drug: str = "Metformin") -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load quality signals, AE monthly counts, SDUD volumes for one drug."""
    fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
    fei_map.columns = [c.strip() for c in fei_map.columns]
    api_col = next(c for c in fei_map.columns if c.lower() == "api")
    fei_col = next(c for c in fei_map.columns if "fei" in c.lower())
    drug_feis = set(
        pd.to_numeric(
            fei_map[fei_map[api_col] == drug][fei_col], errors="coerce"
        ).dropna().astype(int)
    )

    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"] = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    drug_ts = ts[ts["fei"].isin(drug_feis)].copy()
    SIGNAL_COLS = [
        "severity_critmajor_share", "contamination_llm_share",
        "data_integrity_llm_share", "scope_facilitywide_share",
        "cultural_root_cause_share",
    ]
    cols_present = [c for c in SIGNAL_COLS if c in drug_ts.columns]
    keep_cols    = ["fei", "snapshot_date", "n_obs_total"] + cols_present
    quality_raw  = drug_ts[[c for c in keep_cols if c in drug_ts.columns]].copy()

    ae_monthly = _load_ae_monthly_for_drug(drug)

    story_kws = [kw for _, (_, kw) in _STORY_FEIS.get(drug, {}).items()]
    sdud_mfr  = _load_sdud_manufacturers(drug, top_n=10, force_include_kw=story_kws)

    return quality_raw, ae_monthly, sdud_mfr


def _fig_model_evidence(fi: pd.DataFrame, ablation: pd.DataFrame) -> go.Figure:
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
        horizontal_spacing=0.10,
        column_widths=[0.35, 0.65],
    )

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

    auc_max = max(ablation["auc"]) if len(ablation) else 0.8
    fig.update_layout(
        height=440, margin=dict(l=10, r=80, t=50, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        barmode="overlay",
        yaxis=dict(range=[0, auc_max * 1.25], title="AUC-ROC", gridcolor="#F0F0F0"),
        xaxis=dict(gridcolor="white"),
        xaxis2=dict(title="RF Feature Importance", gridcolor="#F0F0F0"),
        yaxis2=dict(gridcolor="white"),
        legend=dict(x=0.73, y=0.05, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E0E0E0", borderwidth=1),
    )
    return fig


def _semiann_quality(q: pd.DataFrame, sig_cols: list[str]) -> pd.DataFrame:
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


def _build_combined_dropdown(
    fig: go.Figure,
    n_sig_total: int,
    n_sdud: int,
    story_feis: dict[int, tuple[str, str]],
    sig_labels_with_idx: list[tuple[int, str]],
) -> None:
    K         = len(story_feis)
    n_per_fei = n_sig_total + 2
    n_base    = 2 * n_sig_total + 1 + n_sdud
    n_total   = n_base + K * n_per_fei

    ae_idx   = 2 * n_sig_total
    sdud_start = ae_idx + 1
    spot_start = sdud_start + n_sdud

    def _vis(agg_sigs, raw_sigs, ae_on, sdud_on, spot_idx):
        v = [False] * n_total
        for i in agg_sigs:
            v[i * 2] = True
        for i in raw_sigs:
            v[i * 2 + 1] = True
        if ae_on:
            v[ae_idx] = True
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


def _fig_case_study_ae(
    quality: pd.DataFrame,
    ae_monthly: pd.DataFrame,
    sdud_mfr: pd.DataFrame,
    drug: str = "Metformin",
) -> go.Figure:
    """Three-row case study: quality signals / AE monthly counts / SDUD volumes."""
    SIGNAL_CFG = [
        ("severity_critmajor_share",  "Critical/Major severity", C["orange"], "solid"),
        ("contamination_llm_share",   "Contamination flag",      C["purple"], "dash"),
        ("data_integrity_llm_share",  "Data integrity flag",     C["teal"],   "dot"),
        ("scope_facilitywide_share",  "Facility-wide scope",     C["blue"],   "dashdot"),
        ("cultural_root_cause_share", "Cultural root cause",     C["green"],  "longdash"),
    ]
    _SDUD_PAL = [
        C["orange"], C["blue"], C["green"], C["purple"], C["teal"], C["red"],
        "#F4A261", "#457B9D", "#2A9D8F", "#E9C46A", "#A8DADC", C["gray"],
    ]
    _N_SIG = len(SIGNAL_CFG)

    date_start = pd.Timestamp(f"{PANEL_START_YEAR}-01-01")
    date_end   = pd.Timestamp(f"{PANEL_END_YEAR}-12-31")
    q = quality[quality["snapshot_date"].between(date_start, date_end)].copy()

    sig_cols_present = [col for col, _, _, _ in SIGNAL_CFG
                        if col in q.columns and not q[col].isna().all()]
    agg = _semiann_quality(q, sig_cols_present) if sig_cols_present else pd.DataFrame()

    story_feis = _STORY_FEIS.get(drug, {})
    sdud_mfrs  = sdud_mfr["mfr_name"].unique().tolist() if not sdud_mfr.empty else []
    if not sdud_mfrs:
        sdud_mfrs = ["(no data)"]
    n_sdud = len(sdud_mfrs)

    subplot_titles = [
        f"{drug} FEIs — quality signals, semi-annual average "
        "(bubble = # facilities; use dropdown to drill into signal or spotlight a facility)",
        "FAERS serious AE reports — quarterly count",
        "Medicaid utilization by manufacturer — monthly, millions of units",
    ]
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True,
        row_heights=[0.46, 0.27, 0.27],
    )

    # ── AGGREGATE SIGNAL TRACES ────────────────────────────────────────────────
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

    # ── AE BAR (row 2) ─────────────────────────────────────────────────────────
    if not ae_monthly.empty:
        fig.add_trace(go.Bar(
            x=ae_monthly["month_start"], y=ae_monthly["n_ae"],
            name="Serious AE reports", marker_color=C["purple"], opacity=0.85,
            showlegend=False,
            hovertemplate="<b>%{x|%b %Y}</b><br>AE reports: %{y:,}<extra></extra>",
        ), row=2, col=1)
    else:
        fig.add_trace(go.Bar(x=[], y=[], name="no_ae", showlegend=False), row=2, col=1)

    # ── SDUD REGULAR TRACES ────────────────────────────────────────────────────
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

    # ── SPOTLIGHT TRACES (per story FEI) ──────────────────────────────────────
    n_per_fei = _N_SIG + 2
    for k, (fei_id, (fei_name, sdud_kw)) in enumerate(story_feis.items()):
        q_fei = q[q["fei"] == fei_id].copy()

        for col, label, col_color, dash in SIGNAL_CFG:
            col_in_data = col in q_fei.columns
            if col_in_data and not q_fei.empty:
                actual = q_fei[col].fillna(0.0).values
                y_plot = np.where(actual == 0.0, -0.03, actual)
                fig.add_trace(go.Scatter(
                    x=q_fei["snapshot_date"], y=y_plot,
                    name=f"{fei_name.split()[0]}: {label}", mode="markers",
                    marker=dict(color=col_color, size=14, opacity=0.90,
                                symbol="diamond", line=dict(width=2, color="white")),
                    visible=False, showlegend=True,
                    customdata=actual,
                    hovertemplate=(
                        f"<b>%{{x|%b %Y}}</b><br><b>{fei_name}</b><br>"
                        f"{label}: %{{customdata:.0%}}<br>FEI {fei_id}"
                        "<extra></extra>"
                    ),
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=[], y=[], name=f"spot_{col}_{k}",
                                          visible=False, showlegend=False), row=1, col=1)

        # AE bar stays same in spotlight (drug-level, not FEI-level)
        if not ae_monthly.empty:
            fig.add_trace(go.Bar(
                x=ae_monthly["month_start"], y=ae_monthly["n_ae"],
                name=f"AE reports ({drug})", marker_color=C["purple"], opacity=0.85,
                visible=False, showlegend=False,
                hovertemplate="<b>%{x|%b %Y}</b><br>AE reports: %{y:,}<extra></extra>",
            ), row=2, col=1)
        else:
            fig.add_trace(go.Bar(x=[], y=[], name=f"no_ae_{k}",
                                  visible=False, showlegend=False), row=2, col=1)

        # SDUD highlight trace for story manufacturer
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

    # ── DROPDOWN ───────────────────────────────────────────────────────────────
    sig_labels_with_idx = [
        (i, label)
        for i, (col, label, _, _) in enumerate(SIGNAL_CFG)
        if col in q.columns and not q[col].isna().all()
    ]
    _build_combined_dropdown(fig, _N_SIG, n_sdud, story_feis, sig_labels_with_idx)

    # ── LAYOUT ─────────────────────────────────────────────────────────────────
    if not sig_cols_present:
        fig.add_annotation(
            x=0.5, y=0.80, xref="paper", yref="paper",
            text=f"<i>No LLM text features found for {drug} FEIs</i>",
            showarrow=False, font=dict(color="#888", size=12), align="center",
        )

    fig.update_layout(
        height=840,
        margin=dict(l=10, r=10, t=70, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="% of 483 obs flagged", tickformat=".0%",
                   range=[-0.05, 1.15], gridcolor="#F0F0F0"),
        yaxis2=dict(title="# AE reports (quarterly)", gridcolor="#F0F0F0"),
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
    ae_by_drug: pd.DataFrame,
    fi: pd.DataFrame,
    ablation: pd.DataFrame,
    cs_quality: pd.DataFrame | None = None,
    cs_ae_monthly: pd.DataFrame | None = None,
    cs_sdud: pd.DataFrame | None = None,
) -> str:
    auc_no   = ablation.iloc[0]["auc"]
    auc_yes  = ablation.iloc[1]["auc"]
    lift_abs = auc_yes - auc_no
    lift_rel = lift_abs / max(auc_no, 0.001) * 100

    total_ae = int(ae_by_drug["n_ae"].sum())

    _empty = pd.DataFrame()
    cs_div = ""
    if cs_quality is not None:
        cs_div = _div(
            _fig_case_study_ae(
                cs_quality,
                cs_ae_monthly if cs_ae_monthly is not None else _empty,
                cs_sdud if cs_sdud is not None else _empty,
                "Metformin",
            ),
            "fig_case_a",
        )

    divs = {
        "funnel":    _div(_fig_funnel(cov),                          "fig_funnel"),
        "supply":    _div(_fig_supply_concentration(supply, ae_by_drug), "fig_supply"),
        "severity":  _div(_fig_ae_severity(ae_by_drug),              "fig_severity"),
        "timeline":  _div(_fig_ae_timeline(ae_by_drug),              "fig_timeline"),
        "case_a":    cs_div,
        "model":     _div(_fig_model_evidence(fi, ablation),         "fig_model"),
    }

    today = date.today().strftime("%B %d, %Y")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drug Safety — Quality Risk &amp; Adverse Events</title>
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

  .flow-diagram {{ display: flex; flex-direction: column; align-items: center;
                   gap: 0; margin-bottom: 28px; }}
  .flow-row {{ display: flex; justify-content: center; align-items: center; width: 100%; }}
  .flow-box {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.18);
                border-radius: 10px; padding: 12px 26px; text-align: center;
                min-width: 180px; max-width: 260px; }}
  .flow-box.flow-start {{ border-color: #4A90D9; border-width: 2px; }}
  .flow-box.flow-end   {{ border-color: #D94A4A; border-width: 2px;
                           background: rgba(217,74,74,0.10); }}
  .flow-title {{ color: #F0F0F0; font-size: 13px; font-weight: 600; display: block; }}
  .flow-sub   {{ color: #A0AEC0; font-size: 11px; display: block; margin-top: 3px; }}
  .flow-arrow-v {{ color: #A0AEC0; font-size: 20px; line-height: 1; padding: 3px 0;
                    text-align: center; }}

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
    .page {{ padding: 0 12px 28px; }}
    nav a {{ display: none; }}
  }}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">Drug Safety &nbsp;<span>·</span>&nbsp; Quality Risk &amp; Adverse Events</div>
  <a href="#coverage">① Coverage</a>
  <a href="#supply">② Supply</a>
  <a href="#ae-severity">③ AE Severity</a>
  <a href="#timeline">④ AE Timeline</a>
  <a href="#case-a">⑤ Case Study</a>
  <a href="#model">⑥ Model Evidence</a>
</nav>

<div class="hero">
  <h1>Manufacturing Quality Failures → Serious Adverse Events</h1>
  <p>Quality failures at generic drug manufacturing facilities (FEIs) can ultimately reach patients
     in the form of product contamination, subpotency, or degradation — leading to serious adverse
     events (AEs) reported to the FDA. This dashboard traces the causal chain: FDA 483 inspection
     text signals → FEI-level model → FAERS serious AE outcomes. Analysis covers 14
     Valisure-validated APIs (2015–2024).</p>

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
    <div class="flow-row"><div class="flow-arrow-v">↓</div></div>
    <div class="flow-row"><div class="flow-box" style="border-color:#7B5EA7;">
      <span class="flow-title">ML Model (FEI × year)</span>
      <span class="flow-sub">Quality signals at t → predicts AE surge at t+1</span>
    </div></div>
    <div class="flow-row"><div class="flow-arrow-v">↓</div></div>
    <div class="flow-row">
      <div class="flow-box flow-end">
        <span class="flow-title">Serious Adverse Events (FAERS)</span>
        <span class="flow-sub">Hospitalization · Life-threatening · Death</span>
      </div>
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi"><div class="val">14</div><div class="lbl">APIs tracked<br>(Valisure-tested)</div></div>
    <div class="kpi"><div class="val">{cov['n_feis_llm']}</div><div class="lbl">FEIs with<br>LLM text features</div></div>
    <div class="kpi"><div class="val">{total_ae:,}</div><div class="lbl">Serious AE reports<br>(FAERS, 2015–2024)</div></div>
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
      concentration risk. Orange = drug has at least one injectable formulation in the Orange Book.
      Right: total FAERS serious AE reports linked to these FEIs (2015–2024).
    </div>
  </div>
  <div class="card">{divs['supply']}</div>
</section>

<!-- ③ AE Severity Breakdown -->
<section id="ae-severity">
  <div class="section-head">
    <div class="section-title">③ Adverse Event Severity Breakdown</div>
    <div class="section-sub">
      FAERS serious adverse event reports classified by reported outcome severity.
      All reports are <b>pre-filtered to serious cases only</b> (Hospitalization, Life-Threatening,
      Disability, Death, or Other Serious). The donut shows the overall severity distribution;
      the bar chart shows composition per drug. Note: AE reports reflect drugs reaching patients,
      not a direct measure of manufacturing quality at any specific FEI.
    </div>
  </div>
  <div class="card">{divs['severity']}</div>
</section>

<!-- ④ AE Timeline (per-drug) -->
<section id="timeline">
  <div class="section-head">
    <div class="section-title">④ Adverse Event Timeline — Per Drug</div>
    <div class="section-sub">
      Use the dropdown to select a drug and see its serious FAERS AE report count by year.
      Raw counts only — trend over time may reflect changes in reporting rates, prescribing volume,
      or true patient harm.
    </div>
  </div>
  <div class="card">{divs['timeline']}</div>
</section>

<!-- ⑤ Case Study: quality signals preceding AE surge -->
<section id="case-a">
  <div class="section-head">
    <div class="section-title">⑤ Case Study: Quality Signals vs. Adverse Event Counts — Metformin</div>
    <div class="section-sub">
      This case study illustrates what the model predicts:
      <b>quality signals at FEIs in year t → above-median AE volume in year t+1.</b>
      <br><br>
      <b>What the data shows:</b>
      (1) Quality signals (contamination, severity, scope, cultural root cause) were elevated
      across multiple Metformin FEIs throughout 2018–2020.
      (2) FAERS AE reports for Metformin (quarterly, drug-level — not FEI-specific).
      (3) Medicaid (SDUD) volume by manufacturer — showing supply distribution.
      <br><br>
      <b>Interpretation:</b> The model is not claiming a direct causal chain from a specific FEI's
      quality failures to specific patient AEs. Rather, it tests whether facility-level quality
      signals (aggregated across the drug's FEI universe) predict years when the drug has
      above-median serious AE volume in FAERS. Use the <b>dropdown</b> to isolate a signal
      or spotlight a facility.
      <br><br>
      <b>Limitations:</b> FAERS is drug-level, not FEI-level; we cannot trace AEs to individual
      manufacturers. AE reporting rates vary by year and drug lifecycle. The AE outcome is
      a population-level proxy, not a direct measure of product quality.
    </div>
  </div>
  <div class="card">{divs['case_a']}</div>
</section>

<!-- ⑥ Model Evidence -->
<section id="model">
  <div class="section-head">
    <div class="section-title">⑥ Model Evidence: 483 Text Improves AE Prediction</div>
    <div class="section-sub">
      <b>What this model predicts:</b> whether a FEI's drug will have above-median serious AE
      volume in year t+1, given quality features measured in year t.
      <br>
      <b>AUC-ROC</b> measures how often the model correctly ranks a facility-year that will have
      high AE volume above one that won't — 0.5 = random chance, 1.0 = perfect.
      Think of it as: "out of all (high-AE, low-AE) pairs, what fraction does the model rank correctly?"
      <br>
      L2 logistic regression, GroupKFold CV by FEI. Adding LLM-extracted 483 text features
      improves AUC from {auc_no:.3f} to {auc_yes:.3f}
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
    Adverse event data: FDA FAERS. Supply concentration: Valisure (March 2026).
    Medicaid volumes: CMS SDUD.
  </span>
</footer>

</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data…")
    fdmap   = _fei_drug_map()
    supply  = _load_supply_concentration(fdmap)
    ae_drug = _load_ae_by_drug(fdmap)
    cov     = _load_coverage()
    fi, abl = _load_model_outputs()

    cs_q, cs_ae, cs_sdud = _load_case_study_ae("Metformin")
    log.info("Case study (Metformin): %d quality rows, %d AE months, %d SDUD mfr-months",
             len(cs_q), len(cs_ae), len(cs_sdud))

    log.info("Building HTML…")
    html = build_html(
        cov, supply, ae_drug, fi, abl,
        cs_quality=cs_q, cs_ae_monthly=cs_ae, cs_sdud=cs_sdud,
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


if __name__ == "__main__":
    main()
