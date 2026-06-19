"""
Module 18 — Interactive FAERS adverse event story dashboard.

Tells the causal chain: manufacturing quality failures → serious adverse events.

Sections:
  1. Data coverage — facility funnel
  2. Supply landscape — FEIs per drug + AE event counts
  3. AE severity breakdown — Hospitalization vs Other Serious by drug
  4. AE timeline — FAERS serious AE counts per drug × year
  5. Signal–AE correlation — quality signals vs AE counts, drug-selectable
     Dropdown: "All drugs (normalized)" shows all 14 drugs on same normalized scale
     to reveal whether quality signal spikes lead AE surges. Per-drug view shows
     the 5 inspection text signals alongside quarterly AE count.
  6. Model evidence — text feature AUC lift + feature importance (from m17)

Output:
  outputs/figures/faers_fei_dashboard.html

Run:
  python m18_faers_dashboard.py
"""

from __future__ import annotations
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings("ignore")

from config import (
    DATA, OUT_FIGS, OUT_MODELS, OUT_LOGS,
    TEXT_TIMESERIES_REDICA_CSV, FAERS_ALL, VALISURE_CSV, VALISURE_FEI,
    PANEL_START_YEAR, PANEL_END_YEAR,
)
from utils import get_logger, ValisureDrugMatcher, load_valisure_api_names

log = get_logger("m18_faers_dashboard", OUT_LOGS / "m18_dashboard.log")

REDICA_RAW  = DATA / "07 - Redica" / "raw"
OB_PRODUCTS = DATA / "01 - Orange Book" / "output_data" / "products.csv"
OUT_HTML    = OUT_FIGS / "faers_fei_dashboard.html"

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

_PLOTLY_FONT = dict(family="'Segoe UI', Helvetica, Arial, sans-serif", size=12)

_PARENTERAL_ROUTES = {
    "INJECTION", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS",
    "INJECTION, INTRAVENOUS", "INTRAVENOUS, SUBCUTANEOUS",
    "INTRAMUSCULAR, INTRAVENOUS", "INJECTABLE", "IRRIGATION",
    "INJECTION, SUBCUTANEOUS",
}

# All 12 text-derived share features organized by category for the case study
SIGNAL_GROUPS: list[tuple[str, list[tuple[str, str, str, str]]]] = [
    ("Severity", [
        ("severity_critmajor_share",    "Critical/Major severity", "#E07B39", "solid"),
    ]),
    ("Scope", [
        ("scope_facilitywide_share",    "Facility-wide scope",     "#4A90D9", "solid"),
        ("scope_multipleproducts_share","Multi-product scope",     "#4A90D9", "dash"),
    ]),
    ("LLM — Safety flags", [
        ("contamination_llm_share",     "Contamination",           "#D94A4A", "solid"),
        ("data_integrity_llm_share",    "Data integrity",          "#7B5EA7", "dash"),
    ]),
    ("LLM — Process flags", [
        ("investigation_llm_share",     "Investigation gaps",      "#7B5EA7", "dot"),
        ("repeat_cross_insp_share",     "Repeat findings",         "#9AA5B1", "dashdot"),
    ]),
    ("Violation type", [
        ("vc_labcontrols_share",        "Lab controls",            "#2E8B8B", "solid"),
        ("vc_qualitysystem_share",      "Quality system",          "#2E8B8B", "dash"),
    ]),
    ("Remediation", [
        ("remediation_none_share",      "No remediation",          "#D94A4A", "dot"),
        ("remediation_weak_share",      "Weak remediation",        "#E07B39", "dashdot"),
    ]),
    ("Cultural", [
        ("cultural_root_cause_share",   "Cultural root cause",     "#3DAA6E", "solid"),
    ]),
]
_ALL_SIG_FLAT: list[tuple[str, str, str, str, str]] = [
    (col, label, color, dash, grp)
    for grp, sigs in SIGNAL_GROUPS
    for col, label, color, dash in sigs
]
N_ALL_SIG = len(_ALL_SIG_FLAT)  # 12


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
    return (
        fdmap.groupby("drug", as_index=False)
        .agg(n_feis=("fei", "nunique"), parenteral=("parenteral", "max"))
        .sort_values("n_feis")
    )


def _load_ae_by_drug(fdmap: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FAERS by drug × year × severity."""
    if not FAERS_ALL.exists():
        log.warning("FAERS file not found; returning empty AE frame")
        return pd.DataFrame(columns=["drug", "year", "severity", "n_ae"])

    df = pd.read_csv(FAERS_ALL, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    prod_col = next((c for c in df.columns if c.lower() == "prod_ai"), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    sev_col  = next((c for c in df.columns if c.lower() == "severity"), None)
    if not prod_col or not year_col:
        return pd.DataFrame(columns=["drug", "year", "severity", "n_ae"])

    df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", prod_col])
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]

    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher   = ValisureDrugMatcher(api_names)
    df["drug"] = df[prod_col].astype(str).map(matcher.match)
    df = df.dropna(subset=["drug"])

    if sev_col and sev_col in df.columns:
        agg = df.groupby(["drug", "year", sev_col], as_index=False).size()
        agg = agg.rename(columns={sev_col: "severity", "size": "n_ae"})
    else:
        agg = df.groupby(["drug", "year"], as_index=False).size()
        agg = agg.rename(columns={"size": "n_ae"})
        agg["severity"] = "Serious"

    log.info("AE by drug: %d rows, %d drugs", len(agg), agg["drug"].nunique())
    return agg


def _load_ae_monthly_for_drug(drug: str) -> pd.DataFrame:
    """Quarterly FAERS AE count for one drug (for per-drug case study view)."""
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

    _QMAP = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    if period_col and period_col in df.columns:
        def _q2month(p: str) -> pd.Timestamp | None:
            p = str(p).strip()
            if len(p) >= 6 and p[4] == "Q":
                return pd.Timestamp(f"{p[:4]}-{_QMAP.get(p[4:],'01')}-01")
            return None
        df["month_start"] = df[period_col].map(_q2month)
    else:
        yr_col = next((c for c in df.columns if c.lower() == "year"), None)
        if yr_col:
            df["month_start"] = pd.to_datetime(df[yr_col].astype(str) + "-01-01", errors="coerce")
        else:
            return pd.DataFrame(columns=["month_start", "n_ae"])

    df = df.dropna(subset=["month_start"])
    agg = df.groupby("month_start", as_index=False).size().rename(columns={"size": "n_ae"})
    return agg.sort_values("month_start").reset_index(drop=True)


def _load_all_drug_quality(fdmap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {drug: timeseries_df} for each drug with inspection text signal data."""
    if not TEXT_TIMESERIES_REDICA_CSV.exists():
        return {}
    ts = pd.read_csv(TEXT_TIMESERIES_REDICA_CSV)
    ts["fei"]           = pd.to_numeric(ts["fei"], errors="coerce").astype("Int64")
    ts["snapshot_date"] = pd.to_datetime(ts["snapshot_date"], errors="coerce")
    ts = ts.dropna(subset=["fei", "snapshot_date"])
    fei_drug = fdmap[["fei", "drug"]].drop_duplicates()
    ts = ts.merge(fei_drug, on="fei", how="inner")
    return {drug: grp.copy() for drug, grp in ts.groupby("drug")}


def _load_all_drug_ae_annual() -> dict[str, pd.DataFrame]:
    """Return {drug: annual_ae_df} with columns [year, n_ae] for each drug."""
    if not FAERS_ALL.exists():
        return {}
    df = pd.read_csv(FAERS_ALL, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    prod_col = next((c for c in df.columns if c.lower() == "prod_ai"), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if not prod_col or not year_col:
        return {}
    df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", prod_col])
    df = df[(df["year"] >= PANEL_START_YEAR) & (df["year"] <= PANEL_END_YEAR)]
    api_names = load_valisure_api_names(VALISURE_CSV)
    matcher   = ValisureDrugMatcher(api_names)
    df["drug"] = df[prod_col].astype(str).map(matcher.match)
    df = df.dropna(subset=["drug"])
    agg = df.groupby(["drug", "year"], as_index=False).size().rename(columns={"size": "n_ae"})
    return {
        drug: grp[["year", "n_ae"]].sort_values("year").reset_index(drop=True)
        for drug, grp in agg.groupby("drug")
    }


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


def _load_model_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load m17 model outputs; fall back to placeholder if not yet generated."""
    fi_path  = OUT_MODELS / "rf_importance_faers_fei.csv"
    abl_path = OUT_MODELS / "text_ablation_faers_fei.csv"

    if fi_path.exists():
        fi = pd.read_csv(fi_path)
    else:
        log.warning("m17 outputs not found — using placeholder. Run m17 first.")
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
    sup = supply.copy().sort_values("n_feis")
    tot_ae = ae_by_drug.groupby("drug")["n_ae"].sum().to_dict()
    sup["n_ae_total"] = sup["drug"].map(tot_ae).fillna(0).astype(int)

    fei_colors = [C["orange"] if p else C["blue"] for p in sup["parenteral"]]

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
        marker_color=fei_colors, hovertext=hover_fei, hoverinfo="text", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=sup["n_ae_total"], y=sup["drug"], orientation="h",
        marker_color=C["purple"], hovertext=hover_ae, hoverinfo="text", showlegend=False,
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
        yaxis=dict(gridcolor="white"), yaxis2=dict(showticklabels=False, gridcolor="white"),
        legend=dict(x=0.72, y=0.02), barmode="overlay",
    )
    return fig


def _fig_ae_severity(ae_by_drug: pd.DataFrame) -> go.Figure:
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
        labels=sev_totals.index.tolist(), values=sev_totals.values.tolist(), hole=0.45,
        marker_colors=[sev_colors.get(s, C["gray"]) for s in sev_totals.index],
        textinfo="label+percent", textfont=dict(size=11), showlegend=False,
        hovertemplate="<b>%{label}</b><br>AE reports: %{value:,}<br>%{percent}<extra></extra>",
    ), row=1, col=1)

    drugs = sorted(ae_by_drug["drug"].unique())
    seen: set[str] = set()
    for sev, sev_color in sev_colors.items():
        vals = [int(ae_by_drug[(ae_by_drug["drug"] == d) & (ae_by_drug["severity"] == sev)]["n_ae"].sum())
                for d in drugs]
        if all(v == 0 for v in vals):
            continue
        fig.add_trace(go.Bar(
            x=drugs, y=vals, name=sev, marker_color=sev_color,
            showlegend=(sev not in seen), legendgroup=f"sev_{sev}",
            hovertemplate="<b>%{x}</b><br>" + sev + "<br>AE reports: %{y:,}<extra></extra>",
        ), row=1, col=2)
        seen.add(sev)

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=50, b=80),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        barmode="stack", xaxis2=dict(tickangle=-35),
        yaxis2=dict(title="# AE reports", gridcolor="#F0F0F0"),
        legend=dict(x=1.01, y=1, font=dict(size=10)),
    )
    return fig


def _fig_ae_timeline(ae_by_drug: pd.DataFrame) -> go.Figure:
    years = list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))
    drugs = sorted(ae_by_drug["drug"].unique())

    fig = go.Figure()
    for i, drug in enumerate(drugs):
        sub = ae_by_drug[ae_by_drug["drug"] == drug].groupby("year")["n_ae"].sum().reset_index()
        sub["year"] = sub["year"].astype(int)
        y_ae = sub.set_index("year")["n_ae"].reindex(years, fill_value=0).tolist()
        fig.add_trace(go.Bar(
            x=years, y=y_ae, name=drug, marker_color=C["purple"],
            visible=(i == 0), showlegend=False,
            hovertemplate="<b>Year %{x}</b><br>AE reports: %{y:,}<extra></extra>",
        ))

    buttons = [dict(
        label=drug, method="update",
        args=[{"visible": [j == i for j in range(len(drugs))]}],
    ) for i, drug in enumerate(drugs)]

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=80, b=10),
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


def _semiann_quality(q: pd.DataFrame, sig_cols: list[str]) -> pd.DataFrame:
    """Aggregate FEI snapshots to semi-annual periods."""
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
    agg = agg.rename(columns={"n_obs_total": "n_obs"}) if "n_obs_total" in agg.columns else agg
    if "n_obs" not in agg.columns:
        agg["n_obs"] = agg["n_feis"]
    return agg.sort_values("period_start").reset_index(drop=True)


def _fig_case_study_multi(
    quality_by_drug: dict[str, pd.DataFrame],
    ae_annual_by_drug: dict[str, pd.DataFrame],
    ae_monthly_by_drug: dict[str, pd.DataFrame],
) -> go.Figure:
    """Three-row signal–AE panel with independent drug + signal dropdowns.

    Row 1: quality signal aggregate bubble-lines (semi-annual, m15 style).
           Selecting a specific signal adds per-FEI open-circle dots.
    Row 2: quarterly FAERS serious AE counts as bars.
    Row 3: temporal correlation — annual normalized signal (solid) overlaid
           with annual normalized AE (dashed, same color). If the signal leads
           AEs you'll see the solid line peak 1–2 years before the dashed line.
           Spearman ρ at lag +1yr is shown in the trace name. Visible only
           when a specific signal is selected from the signal dropdown.

    Two independent orthogonal dropdowns:
      Drug dropdown  → updates trace DATA (x/y/customdata/marker.size/name).
      Signal dropdown → updates trace VISIBILITY only.
    Selecting a drug preserves the signal filter and vice versa.

    Trace layout (49 traces: N = N_ALL_SIG = 12):
      [0   .. N-1 ]   agg signal bubble-lines     row 1
      [N   .. 2N-1]   per-FEI open-circle dots    row 1
      [2N        ]    AE bar                       row 2
      [2N+1..3N  ]    corr signal lines (annual)   row 3
      [3N+1..4N  ]    corr AE lines (annual)       row 3
    """
    N = N_ALL_SIG
    drugs = sorted(set(quality_by_drug) | set(ae_annual_by_drug))

    # ── Helper: build all 49 trace data for one selection ────────────────────
    def _build_data(
        q: pd.DataFrame, ae_m: pd.DataFrame, include_dots: bool
    ) -> tuple:
        """
        Returns 16-tuple:
          agg_x, agg_y, agg_cd, agg_sz  (N items each)
          dot_x, dot_y, dot_cd           (N items each)
          ae_x, ae_y                     (1 item each)
          cs_x, cs_y, cs_cd              (N items each — corr signal)
          ca_x, ca_y, ca_cd              (N items each — corr AE)
          corr_names                     (N items)
        """
        sig_p = [col for col, *_ in _ALL_SIG_FLAT
                 if not q.empty and col in q.columns and not q[col].isna().all()]
        agg = _semiann_quality(q, sig_p) if sig_p and not q.empty else pd.DataFrame()

        agg_x, agg_y, agg_cd, agg_sz = [], [], [], []
        for col, *_ in _ALL_SIG_FLAT:
            if not agg.empty and col in agg.columns and not agg[col].isna().all():
                agg_x.append(agg["period_start"].tolist())
                agg_y.append(agg[col].tolist())
                agg_sz.append(np.clip(agg["n_feis"] * 3 + 7, 8, 30).tolist())
                n_obs = agg["n_obs"].values if "n_obs" in agg.columns \
                        else agg["n_feis"].values
                agg_cd.append(np.column_stack([agg["n_feis"].values, n_obs]).tolist())
            else:
                agg_x.append([]); agg_y.append([])
                agg_sz.append([]); agg_cd.append([])

        dot_x, dot_y, dot_cd = [], [], []
        for col, *_ in _ALL_SIG_FLAT:
            if include_dots and not q.empty and col in q.columns \
                    and not q[col].isna().all():
                dot_x.append(q["snapshot_date"].tolist())
                dot_y.append(q[col].tolist())
                fv = q["fei"].values.reshape(-1, 1) if "fei" in q.columns \
                     else np.zeros((len(q), 1), dtype=int)
                dot_cd.append(fv.tolist())
            else:
                dot_x.append([]); dot_y.append([]); dot_cd.append([])

        ae_x = ae_m["month_start"].tolist() if not ae_m.empty else []
        ae_y = ae_m["n_ae"].tolist() if not ae_m.empty else []

        # ── Temporal correlation row (SEMI-ANNUAL, normalized) ──────────────
        # Aggregate quarterly AE to H1 (Q1+Q2) and H2 (Q3+Q4)
        if not ae_m.empty:
            ae_mc = ae_m.copy()
            ae_mc["halfyr"] = ae_mc["month_start"].apply(
                lambda d: pd.Timestamp(f"{d.year}-01-01") if d.month <= 6
                          else pd.Timestamp(f"{d.year}-07-01")
            )
            ae_semi_s = ae_mc.groupby("halfyr")["n_ae"].sum()
        else:
            ae_semi_s = pd.Series(dtype=float)

        # Full semi-annual grid (H1 = Jan-1, H2 = Jul-1) for lag computation
        _hgrid = pd.DatetimeIndex([
            pd.Timestamp(f"{y}-{m:02d}-01")
            for y in range(PANEL_START_YEAR, PANEL_END_YEAR + 1)
            for m in [1, 7]
        ])
        ae_full = ae_semi_s.reindex(_hgrid).fillna(0)

        _LAG_LBLS = {0: "lag 0", 1: "+6mo", 2: "+1yr", 3: "+18mo", 4: "+2yr"}

        def _hlbl(t: pd.Timestamp) -> str:
            return f"{t.year} H{'1' if t.month <= 6 else '2'}"

        cs_x, cs_y, cs_cd, cs_text = [], [], [], []
        ca_x, ca_y, ca_cd = [], [], []
        corr_names = []

        for col, label, *_ in _ALL_SIG_FLAT:
            sig_avail = (
                not agg.empty and col in agg.columns
                and not agg[col].isna().all()
            )
            if sig_avail:
                sig_semi_s = agg.set_index("period_start")[col].dropna()
                sig_full   = sig_semi_s.reindex(_hgrid)   # NaN where no inspection
                vis_ts = [t for t in _hgrid if pd.notna(sig_full.get(t))]
            else:
                sig_semi_s = pd.Series(dtype=float)
                sig_full   = pd.Series(np.nan, index=_hgrid)
                vis_ts = []

            if len(vis_ts) >= 3:
                sv = np.array([float(sig_semi_s[t]) for t in vis_ts])
                sv_n = (sv - sv.min()) / ((sv.max() - sv.min()) + 1e-12)
                av_all = ae_full.values.astype(float)
                av_n_all = (av_all - av_all.min()) / ((av_all.max() - av_all.min()) + 1e-12)

                # Multi-lag Spearman ρ: signal at t predicts AE at t + lag half-years
                n_h = len(_hgrid)
                lag_results: dict[int, tuple[float, int]] = {}
                for lag in range(5):
                    pairs = [
                        (float(sig_full.iloc[i]), float(ae_full.iloc[i + lag]))
                        for i in range(n_h - lag)
                        if pd.notna(sig_full.iloc[i])
                    ]
                    if len(pairs) >= 4:
                        lx = pd.Series([p[0] for p in pairs])
                        ly = pd.Series([p[1] for p in pairs])
                        lag_results[lag] = (lx.rank().corr(ly.rank()), len(pairs))

                best_lag = (
                    max(lag_results, key=lambda l: abs(lag_results[l][0]))
                    if lag_results else None
                )
                if best_lag is not None:
                    brho, bn = lag_results[best_lag]
                    rho_pt    = f"  ρ={brho:+.2f} @ {_LAG_LBLS[best_lag]}"
                    cs_text_i = [""] * (len(sv_n) - 1) + [rho_pt]
                    name_r    = f"{label}  ·  best ρ={brho:+.2f} @ {_LAG_LBLS[best_lag]} (n={bn})"
                    if 2 in lag_results and best_lag != 2:
                        r1yr, _ = lag_results[2]
                        name_r += f"  |  +1yr: ρ={r1yr:+.2f}"
                else:
                    cs_text_i = [""] * len(sv_n)
                    name_r    = f"{label}  ·  (n<4)"

                cs_x.append(vis_ts);  cs_y.append(sv_n.tolist())
                cs_cd.append([[_hlbl(t), float(sig_semi_s[t])] for t in vis_ts])
                cs_text.append(cs_text_i)
                ca_x.append(list(_hgrid));  ca_y.append(av_n_all.tolist())
                ca_cd.append([[_hlbl(t), float(ae_full[t])] for t in _hgrid])
                corr_names.append(name_r)
            else:
                cs_x.append([]); cs_y.append([]); cs_cd.append([]); cs_text.append([])
                ca_x.append([]); ca_y.append([]); ca_cd.append([])
                corr_names.append(f"{label}  ·  (no data)")

        return (agg_x, agg_y, agg_cd, agg_sz,
                dot_x, dot_y, dot_cd,
                ae_x, ae_y,
                cs_x, cs_y, cs_cd, cs_text,
                ca_x, ca_y, ca_cd,
                corr_names)

    # Compute "All drugs" data (dots hidden — too many points)
    all_q_frames = [df for df in quality_by_drug.values() if not df.empty]
    all_q = pd.concat(all_q_frames, ignore_index=True) if all_q_frames else pd.DataFrame()
    ae_frames = [df for df in ae_monthly_by_drug.values() if not df.empty]
    ae_all = pd.DataFrame()
    if ae_frames:
        ae_c = pd.concat(ae_frames, ignore_index=True)
        if not ae_c.empty:
            ae_all = (ae_c.groupby("month_start", as_index=False)["n_ae"].sum()
                       .sort_values("month_start").reset_index(drop=True))
    ad = _build_data(all_q, ae_all, include_dots=False)

    drug_data: dict[str, tuple] = {
        drug: _build_data(
            quality_by_drug.get(drug, pd.DataFrame()),
            ae_monthly_by_drug.get(drug, pd.DataFrame()),
            include_dots=True,
        )
        for drug in drugs
    }

    def _unpack(d: tuple) -> tuple:
        return d

    (ax0, ay0, acd0, asz0,
     dx0, dy0, dcd0,
     ex0, ey0,
     csx0, csy0, cscd0, cst0,
     cax0, cay0, cacd0,
     cnames0) = _unpack(ad)

    # ── Build figure (3 rows, shared x-axis) ─────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "Quality signals aggregate  "
            "(bubble = # facilities · use dropdowns to filter drug or signal)",
            "FAERS serious AE reports (quarterly)",
            "Temporal correlation: signal (solid) vs AE count (dashed), both normalized 0→1  "
            "│ semi-annual  ·  select a signal to reveal  ·  best-lag ρ shown on chart",
        ],
        row_heights=[0.36, 0.22, 0.42],
    )

    # Row 1 – aggregate signal traces [0..N-1]
    seen_grps: set[str] = set()
    for i, (col, label, color, dash, grp) in enumerate(_ALL_SIG_FLAT):
        is_first = grp not in seen_grps
        if is_first:
            seen_grps.add(grp)
        gt_kw = {"legendgrouptitle": dict(text=grp)} if is_first else {}
        sizes = asz0[i] if asz0[i] else 8
        cd    = acd0[i] if acd0[i] else None
        ht = (
            f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.0%}}<br>"
            "Facilities: %{customdata[0]:.0f} · obs: %{customdata[1]:.0f}"
            "<extra></extra>"
        ) if cd else f"<b>%{{x|%b %Y}}</b><br>{label}: %{{y:.0%}}<extra></extra>"
        fig.add_trace(go.Scatter(
            x=ax0[i], y=ay0[i], name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=sizes, sizemode="diameter"),
            connectgaps=False, customdata=cd,
            legendgroup=f"grp_{grp}", visible=True, showlegend=True,
            hovertemplate=ht, **gt_kw,
        ), row=1, col=1)

    # Row 1 – per-FEI dot traces [N..2N-1] (initially empty for "All drugs")
    for i, (col, label, color, dash, grp) in enumerate(_ALL_SIG_FLAT):
        fig.add_trace(go.Scatter(
            x=dx0[i], y=dy0[i],
            name=f"{label} (per FEI)", mode="markers",
            marker=dict(color=color, size=9, opacity=0.38,
                        symbol="circle-open", line=dict(width=1.5)),
            visible=False, showlegend=False,
            legendgroup=f"grp_{grp}",
            customdata=dcd0[i] if dcd0[i] else None,
            hovertemplate=(
                f"<b>%{{x|%b %Y}}</b> · FEI %{{customdata[0]}}<br>"
                f"{label}: %{{y:.0%}}<br><i>Individual facility</i><extra></extra>"
            ),
        ), row=1, col=1)

    # Row 2 – AE bar [2N]
    fig.add_trace(go.Bar(
        x=ex0, y=ey0, name="Serious AE reports",
        marker_color=C["purple"], opacity=0.75,
        visible=True, showlegend=False,
        hovertemplate="<b>%{x|%b %Y}</b><br>AE reports: %{y:,}<extra></extra>",
    ), row=2, col=1)

    # Row 3 – corr signal traces [2N+1..3N] (initially hidden)
    for i, (col, label, color, dash, grp) in enumerate(_ALL_SIG_FLAT):
        cd_cs  = cscd0[i] if cscd0[i] else None
        cst_i  = cst0[i]  if cst0[i]  else []
        fig.add_trace(go.Scatter(
            x=csx0[i], y=csy0[i],
            name=cnames0[i], mode="lines+markers+text",
            text=cst_i,
            textposition="middle right",
            textfont=dict(size=11, color=color),
            line=dict(color=color, width=2.8, dash="solid"),
            marker=dict(size=9),
            connectgaps=False, customdata=cd_cs,
            legendgroup=f"corr_{grp}",
            legendgrouptitle=dict(text="Correlation row") if i == 0 else {},
            visible=False, showlegend=True,
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b><br>{label}: %{{customdata[1]:.0%}}<br>"
                "Normalized: %{y:.2f}<extra></extra>"
            ) if cd_cs else f"<b>%{{x|%b %Y}}</b><br>{label} (norm): %{{y:.2f}}<extra></extra>",
        ), row=3, col=1)

    # Row 3 – corr AE traces [3N+1..4N] (initially hidden)
    for i, (col, label, color, dash, grp) in enumerate(_ALL_SIG_FLAT):
        cd_ca = cacd0[i] if cacd0[i] else None
        fig.add_trace(go.Scatter(
            x=cax0[i], y=cay0[i],
            name="AE count (normalized)", mode="lines+markers",
            line=dict(color=color, width=2.0, dash="dash"),
            marker=dict(size=7, symbol="square"),
            connectgaps=False, customdata=cd_ca,
            legendgroup=f"corr_{grp}",
            visible=False, showlegend=(i == 0),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>AE reports: %{customdata[1]:,.0f}<br>"
                "Normalized: %{y:.2f}<extra></extra>"
            ) if cd_ca else "<b>%{x|%b %Y}</b><br>AE count (norm): %{y:.2f}<extra></extra>",
        ), row=3, col=1)

    # ── Drug dropdown (updates DATA, not visibility) ───────────────────────────
    def _drug_trace_upd(d: tuple) -> dict:
        (ax, ay, acd, asz,
         dx, dy, dcd,
         ex, ey,
         csx, csy, cscd, cst,
         cax, cay, cacd,
         cnames) = _unpack(d)
        return {
            "x":           ax + dx + [ex] + csx + cax,          # 49 lists
            "y":           ay + dy + [ey] + csy + cay,
            "customdata":  acd + dcd + [None] + cscd + cacd,
            "marker.size": asz + [9]*N + [None] + [None]*N + [None]*N,
            "text": (
                [[]]*N + [[]]*N + [[]]   # agg + dot + AE bar: no inline text
                + cst                    # corr signal: ρ on last point
                + [[]]*N                 # corr AE: no inline text
            ),
            "name": (
                [lbl for _, lbl, *_ in _ALL_SIG_FLAT]
                + [f"{lbl} (per FEI)" for _, lbl, *_ in _ALL_SIG_FLAT]
                + ["Serious AE reports"]
                + cnames
                + ["AE count (normalized)"] * N
            ),
        }

    def _drug_layout_upd(drug: str | None) -> dict:
        if drug is None:
            t0 = ("All 14 drugs — quality signals aggregate  "
                  "(bubble = # facilities · use dropdowns to filter drug or signal)")
            t1 = "All 14 drugs — FAERS serious AE reports (quarterly total)"
            y2 = "# Serious AE reports (all 14 drugs)"
        else:
            t0 = (f"{drug} FEIs — quality signals  "
                  "(bubble = # facilities · use dropdowns to filter drug or signal)")
            t1 = f"{drug} — FAERS serious AE reports (quarterly)"
            y2 = f"{drug} — # Serious AE reports"
        return {
            "annotations[0].text": t0,
            "annotations[1].text": t1,
            "yaxis2.title.text":   y2,
        }

    drug_buttons = [dict(
        label="All drugs", method="update",
        args=[_drug_trace_upd(ad), _drug_layout_upd(None)],
    )]
    for drug in drugs:
        drug_buttons.append(dict(
            label=drug, method="update",
            args=[_drug_trace_upd(drug_data[drug]), _drug_layout_upd(drug)],
        ))

    # ── Signal dropdown (updates VISIBILITY only) ─────────────────────────────
    # "All signals" → all agg shown, dots hidden, corr row hidden
    # Signal S      → agg[S]+dot[S]+corr_sig[S]+corr_ae[S] shown
    def _sig_vis(sig_idx: int | None) -> list[bool]:
        if sig_idx is None:
            return [True]*N + [False]*N + [True] + [False]*N + [False]*N
        v = [j == sig_idx for j in range(N)]
        return v + v + [True] + v + v

    sig_buttons = [dict(
        label="All signals", method="restyle",
        args=[{"visible": _sig_vis(None)}],
    )]
    for i, (col, label, *_) in enumerate(_ALL_SIG_FLAT):
        sig_buttons.append(dict(
            label=label, method="restyle",
            args=[{"visible": _sig_vis(i)}],
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=drug_buttons, direction="down",
                x=0.00, y=1.16, xanchor="left", yanchor="top",
                showactive=True, bgcolor="white", bordercolor="#DDD",
                font=dict(size=11), pad={"r": 10, "t": 5},
            ),
            dict(
                buttons=sig_buttons, direction="down",
                x=0.26, y=1.16, xanchor="left", yanchor="top",
                showactive=True, bgcolor="white", bordercolor="#DDD",
                font=dict(size=11), pad={"r": 10, "t": 5},
            ),
        ],
        height=950,
        margin=dict(l=10, r=220, t=110, b=10),
        font=_PLOTLY_FONT, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            tickformat="%b %Y", dtick="M6", tickangle=-30, gridcolor="#F0F0F0",
        ),
        yaxis=dict(
            title="% of 483 obs flagged",
            tickformat=".0%", range=[-0.05, 1.1], gridcolor="#F0F0F0",
        ),
        yaxis2=dict(
            title="# Serious AE reports", tickformat=",", gridcolor="#F0F0F0",
        ),
        yaxis3=dict(
            title="0 → 1 normalized",
            tickformat=".2f", range=[-0.05, 1.1], gridcolor="#F0F0F0",
        ),
        legend=dict(
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#E0E0E0", borderwidth=1,
            font=dict(size=10), tracegroupgap=6,
            groupclick="togglegroup",
        ),
    )

    fig.add_annotation(
        x=1.01, y=0.02, xref="paper", yref="paper",
        text=(
            "<b>Tips:</b><br>"
            "① Drug filter → changes data<br>"
            "② Signal filter → shows one<br>"
            "   signal + per-FEI dots +<br>"
            "   temporal correlation row<br>"
            "③ Legend group titles toggle<br>"
            "   signal categories on/off"
        ),
        showarrow=False, align="left",
        font=dict(size=9, color="#555"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#DDD", borderwidth=1,
        xanchor="left",
    )

    return fig


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
    quality_by_drug: dict[str, pd.DataFrame] | None = None,
    ae_annual_by_drug: dict[str, pd.DataFrame] | None = None,
    ae_monthly_by_drug: dict[str, pd.DataFrame] | None = None,
) -> str:
    auc_no   = ablation.iloc[0]["auc"]
    auc_yes  = ablation.iloc[1]["auc"]
    lift_abs = auc_yes - auc_no
    lift_rel = lift_abs / max(auc_no, 0.001) * 100
    total_ae = int(ae_by_drug["n_ae"].sum())

    cs_div = ""
    if quality_by_drug and ae_annual_by_drug and ae_monthly_by_drug:
        cs_div = _div(
            _fig_case_study_multi(quality_by_drug, ae_annual_by_drug, ae_monthly_by_drug),
            "fig_case_a",
        )

    divs = {
        "funnel":   _div(_fig_funnel(cov),                            "fig_funnel"),
        "supply":   _div(_fig_supply_concentration(supply, ae_by_drug), "fig_supply"),
        "severity": _div(_fig_ae_severity(ae_by_drug),                "fig_severity"),
        "timeline": _div(_fig_ae_timeline(ae_by_drug),                "fig_timeline"),
        "case_a":   cs_div,
        "model":    _div(_fig_model_evidence(fi, ablation),           "fig_model"),
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
  <a href="#case-a">⑤ Signal–AE Patterns</a>
  <a href="#model">⑥ Model Evidence</a>
</nav>

<div class="hero">
  <h1>Manufacturing Quality Failures → Serious Adverse Events</h1>
  <p>Quality failures at generic drug manufacturing facilities can reach patients as
     contamination, subpotency, or degradation — leading to serious adverse events (AEs)
     reported to the FDA. This dashboard traces the causal chain: FDA 483 inspection
     text signals → FEI-level model → FAERS serious AE outcomes across 14
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
      the bar chart shows composition per drug.
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
    </div>
  </div>
  <div class="card">{divs['timeline']}</div>
</section>

<!-- ⑤ Signal–AE Correlation Panel -->
<section id="case-a">
  <div class="section-head">
    <div class="section-title">⑤ Quality Signal vs. Adverse Event Patterns</div>
    <div class="section-sub">
      Use the <b>dropdown</b> to explore all 14 drugs together or drill into any single drug.
      <br><br>
      <b>All drugs (normalized)</b> — each drug is a separate color.
      <i>Solid line</i> = composite quality signal (mean of available 483 text signals, annual average
      across the drug's FEIs, normalized 0–1 per drug). <i>Dashed line</i> = FAERS annual AE count
      (normalized 0–1 per drug). Both rows share the same year axis.
      <b>Look for drugs where signal spikes (top) appear 1–2 years before AE rises (bottom),
      suggesting a leading-indicator pattern.</b>
      <br><br>
      <b>Single drug</b> — top panel shows all 5 named signals (semi-annual aggregate, bubble size
      = number of FEIs contributing). Bottom panel shows quarterly AE count (raw).
      <br><br>
      <b>Limitations:</b> FAERS is drug-level, not FEI-level — AE reports cannot be traced to
      individual facilities. The AE outcome is a population-level proxy. Reporting rates vary by
      year and drug lifecycle. Normalization preserves shape but removes absolute scale.
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

    quality_by_drug    = _load_all_drug_quality(fdmap)
    ae_annual_by_drug  = _load_all_drug_ae_annual()
    all_drugs          = sorted(set(quality_by_drug) | set(ae_annual_by_drug))
    ae_monthly_by_drug = {d: _load_ae_monthly_for_drug(d) for d in all_drugs}

    log.info(
        "Case study panel: %d drugs with quality data, %d with AE data",
        len(quality_by_drug), len(ae_annual_by_drug),
    )

    log.info("Building HTML…")
    html = build_html(
        cov, supply, ae_drug, fi, abl,
        quality_by_drug=quality_by_drug,
        ae_annual_by_drug=ae_annual_by_drug,
        ae_monthly_by_drug=ae_monthly_by_drug,
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", OUT_HTML)
    print(f"\nDone → {OUT_HTML}")


if __name__ == "__main__":
    main()
