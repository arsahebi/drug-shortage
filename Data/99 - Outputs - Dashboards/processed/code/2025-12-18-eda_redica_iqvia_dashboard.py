# %%
# # -*- coding: utf-8 -*-
"""
EDA dashboard: IQVIA metformin volumes vs Redica FDA inspection outcomes.

New module structure expectations:
<Module>/
  raw/                 (optional; not used here)
  processed/
    code/
      YYYY-MM-DD-eda_redica_iqvia_dashboard.py
  source.txt           (written by this script)

Outputs:
- processed/YYYY-MM-DD-Metformin_IQVIA_Inspection_Dashboard.html
- source.txt (at module root)
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

pd.set_option("display.max_columns", None)

# -------------------- CONFIG --------------------

# Edit this once if needed
DATA_ROOT = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/Data"
)

# Inputs (per your new structure)
REDICA_RAW_DIR = DATA_ROOT / "07 - Redica" / "raw"
IQVIA_XLSX     = DATA_ROOT / "06 - IQVIA"  / "raw" / "Metformin Jan 2015 - Mar 2025 No NDC.xlsx"

MOLECULE = "METFORMIN"

# Optional: ignore subfolders when scanning Redica
EXCLUDE_DIRS = {"derived", "Derived"}

# Output behavior
AUTO_OPEN_HTML = True


# -------------------- PATHS (module-local) --------------------

def _infer_run_tag() -> str:
    stem = Path(__file__).stem
    m = re.match(r"^\d{4}-\d{2}-\d{2}", stem)
    return m.group(0) if m else datetime.now().strftime("%Y-%m-%d")

RUN_TAG = "2025-12-18"

# script lives in: <module>/processed/code/script.py
MODULE_DIR    = DATA_ROOT / "99 - Outputs - Dashboards"
PROCESSED_DIR = MODULE_DIR / "processed"
OUT_HTML      = PROCESSED_DIR / f"{RUN_TAG}-Metformin_IQVIA_Inspection_Dashboard.html"
SOURCE_TXT    = MODULE_DIR / "source.txt"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_source_txt(redica_files: list[Path]) -> None:
    lines = []
    lines.append(f"created_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"module_dir: {MODULE_DIR}")
    lines.append(f"script: {Path(__file__).resolve()}")
    lines.append("")
    lines.append("inputs:")
    lines.append(f"- IQVIA_XLSX: {IQVIA_XLSX}")
    lines.append(f"- REDICA_RAW_DIR: {REDICA_RAW_DIR}")
    lines.append(f"- REDICA_FILES_COUNT: {len(redica_files)}")
    for fp in redica_files:
        lines.append(f"  - {fp}")
    lines.append("")
    lines.append("outputs:")
    lines.append(f"- DASHBOARD_HTML: {OUT_HTML}")
    SOURCE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------- HELPERS --------------------

NBSP = "\u00A0"

def list_redica_files(base_dir: Path) -> list[Path]:
    """
    Returns Excel file paths whose basename starts with 100 + digits,
    skipping temp files (~$). Recurses through subfolders.
    """
    out = []
    for fp in glob.glob(str(base_dir / "**" / "*.xlsx"), recursive=True):
        p = Path(fp)
        base = p.name
        if base.startswith("~$"):
            continue
        if not re.match(r"^100\d+.*\.xlsx$", base):
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


# Regex that allows ordinary or NB-spaces around the hyphen and inside the brackets
name_re = re.compile(
    rf"""^
        (?P<site_id>100\d+)         # numeric id beginning with 100
        [\s{NBSP}]*-\s*             # hyphen separator (liberal on whitespace)
        (?P<mfg>.*?)                # manufacturer name (lazy)
        [\s{NBSP}]*\[
        (?P<city>.*?)               # city
        [\s{NBSP}]*_[\s{NBSP}]*     # underscore separator
        (?P<country>.*?)            # country
        \]\.xlsx$
    """,
    re.X | re.I,
)

from typing import Optional, Dict
def parse_filename(base: str) -> Optional[Dict]:
    clean = base.replace(NBSP, " ").strip()
    m = name_re.match(clean)
    if m:
        return m.groupdict()

    # fallback: split by " - " then "[..._...]"
    try:
        left, right = clean.split(" - ", 1)
        site_id = left.strip()
        mfg, loc = right.split("[", 1)
        loc = loc.rstrip("].xlsx")
        city, country = [x.strip() for x in loc.split("_", 1)]
        return dict(site_id=site_id, mfg=mfg.strip(), city=city, country=country)
    except ValueError:
        return None


def _clean_name(s: str) -> str:
    return re.sub(r"\W+", " ", str(s)).upper().strip()


# -------------------- LOAD: REDICA INSPECTIONS --------------------

def load_redica_inspections(redica_files: list[Path]) -> pd.DataFrame:
    dfs = []
    for fp in redica_files:
        meta = parse_filename(fp.name)
        if meta is None:
            print(f"⚠️  Could not parse metadata from: {fp.name}")
            continue
        try:
            df = pd.read_excel(fp, engine="openpyxl")
            for k, v in meta.items():
                df[k] = v
            dfs.append(df)
        except Exception as e:
            print(f"❌ Could not read {fp.name}: {e}")

    if not dfs:
        raise ValueError(f"No readable Redica inspection files found under: {REDICA_RAW_DIR}")

    inspections = pd.concat(dfs, ignore_index=True)

    renamer = {
        "Date (Event End Date)": "event_date",
        "Red Flag Criticality":  "criticality",
        "Red Flag Type":         "flag_type",
        "Red Flag Value":        "flag_value",
        "Red Flag Agency":       "agency",
        "Site Score":            "site_score",
    }
    inspections = inspections.rename(columns=renamer)
    inspections["event_date"] = pd.to_datetime(inspections["event_date"], errors="coerce")
    inspections["site_id"]    = pd.to_numeric(inspections["site_id"], errors="coerce").astype("Int64")
    inspections["year"]       = inspections["event_date"].dt.year
    inspections["quarter"]    = inspections["event_date"].dt.to_period("Q")

    return inspections


# -------------------- EVENT CLASSIFICATION (Table A1 logic) --------------------

ENF_PAT = re.compile(r"WARNING\s*LETTER|ENFORCEMENT|SEIZURE|INJUNCTION|CONSENT\s*DECREE|IMPORT\s*ALERT", re.I)
OAI_PAT = re.compile(r"\bOAI\b|OFFICIAL\s*ACTION|CFR\s*CITATION", re.I)
VAI_PAT = re.compile(r"\bVAI\b|VOLUNTARY\s*ACTION", re.I)
NAI_PAT = re.compile(r"\bNAI\b|NO\s*ACTION|COMPLIANT", re.I)

QR_MAP = {
    (False, "N"): 0.0,
    (False, "E"): 0.5,
    (False, "A"): 3.0,
    ( True, "N"): 1.0,
    ( True, "E"): 1.5,
    ( True, "A"): 3.5,
}

def _event_codes(flag_list) -> dict:
    vs = [str(v).strip().replace(NBSP, " ") for v in flag_list if pd.notna(v)]
    vup = [v.upper() for v in vs]

    has_no_483 = any("NO 483" in v for v in vup)
    has_483 = False if has_no_483 else any(re.search(r"\b483\b", v) for v in vup)

    if any(OAI_PAT.search(v) for v in vs):
        district = "A"
    elif any(VAI_PAT.search(v) for v in vs):
        district = "E"
    elif any(NAI_PAT.search(v) for v in vs):
        district = "N"
    else:
        district = "N"

    enforcement = any(ENF_PAT.search(v) for v in vs)
    return {"district": district, "has_483": has_483, "enforcement": enforcement}

def build_event_table(inspections: pd.DataFrame) -> pd.DataFrame:
    fda = (
        inspections.loc[
            (inspections["agency"] == "US - FDA") &
            (inspections["flag_type"] == "2. Inspection Outcome"),
            ["site_id", "mfg", "event_date", "flag_value"]
        ]
        .copy()
    )

    events = (
        fda.groupby(["site_id", "event_date", "mfg"], as_index=False)
           .agg(flag_values=("flag_value", list))
    )

    codes_df = pd.DataFrame(events["flag_values"].apply(_event_codes).tolist(), index=events.index)
    events = pd.concat([events, codes_df], axis=1)

    def _compute_qr(row):
        if row.get("enforcement", False):
            return 10.0
        return QR_MAP[(bool(row["has_483"]), row["district"])]

    events["qr_score"] = events.apply(_compute_qr, axis=1)
    return events


# -------------------- LOAD: IQVIA (monthly by manufacturer) --------------------

MONTH_PAT = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$", re.I)

def _load_iqvia_sheet(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    try:
        wide = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
    except ValueError:
        # sheet missing
        return pd.DataFrame(columns=["Manufacturer", "date", "volume", "metric"])

    wide.columns = (
        wide.columns.astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.strip()
    )
    # Remove metric prefix from month headers ("TRx May 2015" -> "May 2015")
    wide.columns = [re.sub(r"^(TRx|Extended Units|EUTRx)\s*", "", c, flags=re.I) for c in wide.columns]

    if "Combined Molecule" in wide.columns:
        wide = wide.loc[wide["Combined Molecule"].astype(str).str.upper() == MOLECULE]
    if "Manufacturer" not in wide.columns:
        raise ValueError(f"IQVIA file missing 'Manufacturer' column on sheet '{sheet}'.")

    month_cols = [c for c in wide.columns if MONTH_PAT.match(str(c))]
    if not month_cols:
        raise ValueError(f"IQVIA sheet '{sheet}' has no month columns like 'Jan 2015' (after cleaning).")

    long = (
        wide.melt(id_vars=["Manufacturer"], value_vars=month_cols,
                  var_name="month", value_name="volume")
            .dropna(subset=["volume"])
    )
    long["date"] = pd.to_datetime(long["month"], format="%b %Y") + pd.offsets.MonthEnd(0)
    long["metric"] = sheet
    return long[["Manufacturer", "date", "volume", "metric"]]


def load_iqvia_monthly_by_mfg(iqvia_xlsx: Path) -> pd.DataFrame:
    frames = []
    for sheet in ["TRx", "Extended Units"]:
        df = _load_iqvia_sheet(iqvia_xlsx, sheet)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"No IQVIA data loaded from: {iqvia_xlsx}")

    iqvia_raw = pd.concat(frames, ignore_index=True)
    iqvia = (
        iqvia_raw.groupby(["Manufacturer", "date", "metric"], as_index=False)
                 .agg(volume=("volume", "sum"))
    )
    # rename metric for display consistency
    iqvia.loc[iqvia["metric"].str.lower().eq("extended units"), "metric"] = "ERx"
    return iqvia


# -------------------- MERGE + DASHBOARD --------------------

MANUAL_MAP = {
    "AUROBINDO PHARM":        "AUROBINDO PHARMA LIMITED",
    "GRANULES PHARMA":        "GRANULES INDIA LIMITED",
    "LAURUS LABS LTD":        "LAURUS LABS LIMITED",
    "SCIEGEN PHARMA":         "SCIEGEN PHARMACEUTICALS INC",
    "SUN PHARMACEUTICAL":     "SUN PHARMACEUTICAL INDUSTRIES LIMITED",
    "ZYDUS PHARM":            "ZYDUS LIFESCIENCES LIMITED",
}

ACTION_SHORT = {"N": "NAI", "E": "VAI", "A": "OAI"}

COMBO_STYLE = {
    "No483-NAI": dict(color="green",  dash="dash"),
    "No483-VAI": dict(color="orange", dash="dash"),
    "No483-OAI": dict(color="red",    dash="dash"),
    "483-NAI":   dict(color="green",  dash="solid"),
    "483-VAI":   dict(color="orange", dash="solid"),
    "483-OAI":   dict(color="red",    dash="solid"),
    "483-ENF":   dict(color="black",  dash="solid"),
}
COMBO_LABEL = {
    "No483-NAI": "No 483 – NAI",
    "No483-VAI": "No 483 – VAI",
    "No483-OAI": "No 483 – OAI",
    "483-NAI":   "483 – NAI",
    "483-VAI":   "483 – VAI",
    "483-OAI":   "483 – OAI",
    "483-ENF":   "483 – ENF",
}
COMBO_ORDER = ["No483-NAI","No483-VAI","No483-OAI","483-NAI","483-VAI","483-OAI","483-ENF"]


def _tableA(df_company: pd.DataFrame) -> pd.DataFrame:
    ev = (
        df_company.drop_duplicates(subset=["DisplayMfg","event_date","combo_key"])
                  .assign(has_483=lambda d: d["combo_key"].str.startswith("483"),
                          action=lambda d: d["combo_key"].str.split("-").str[1])
    )
    mat = (ev.pivot_table(index="has_483", columns="action", values="event_date",
                          aggfunc="count", fill_value=0)
             .reindex(index=[False, True], fill_value=0)
             .reindex(columns=["NAI","VAI","OAI","ENF"], fill_value=0))
    mat.index = ["No 483","483"]
    mat["Total"] = mat.sum(axis=1)
    total_row = pd.DataFrame([mat.sum(axis=0)], index=["Total"])
    return pd.concat([mat, total_row])


def build_dashboard(merged_common: pd.DataFrame) -> None:
    companies = sorted(merged_common["DisplayMfg"].dropna().unique())
    if not companies:
        raise ValueError("No companies available after merge; check MANUAL_MAP / manufacturer normalization.")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"type": "table"}]],
        column_widths=[0.82, 0.18],
        horizontal_spacing=0.06
    )

    # TRx/ERx traces per company (only one visible at a time via dropdown)
    for idx, mfg in enumerate(companies):
        sub = merged_common[merged_common["DisplayMfg"] == mfg]
        trx = sub[sub["metric"] == "TRx"].groupby("date", as_index=False)["volume"].sum()
        erx = sub[sub["metric"] == "ERx"].groupby("date", as_index=False)["volume"].sum()

        fig.add_trace(go.Scatter(
            x=trx["date"], y=trx["volume"], name="TRx",
            legendgroup="TRx", mode="lines+markers",
            showlegend=(idx == 0), visible=(idx == 0)
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=erx["date"], y=erx["volume"], name="ERx",
            legendgroup="ERx", mode="lines+markers",
            showlegend=(idx == 0), visible=(idx == 0)
        ), row=1, col=1, secondary_y=True)

    # legend dummies for combined classes
    first_date = merged_common["date"].min()
    for key in COMBO_ORDER:
        style = COMBO_STYLE[key]
        fig.add_trace(go.Scatter(
            x=[first_date], y=[0], mode="lines",
            line=dict(**style, width=2),
            name=COMBO_LABEL[key], showlegend=True, visible=True
        ), row=1, col=1, secondary_y=False)

    # one table per company
    table_traces_idx = []
    for idx, mfg in enumerate(companies):
        t = _tableA(merged_common[merged_common["DisplayMfg"] == mfg])
        header_vals = ["", "NAI", "VAI", "OAI", "ENF", "Total"]
        cell_vals = [
            t.index.tolist(),
            t["NAI"].tolist(), t["VAI"].tolist(),
            t["OAI"].tolist(), t["ENF"].tolist(),
            t["Total"].tolist()
        ]
        vis = (idx == 0)
        fig.add_trace(go.Table(
            columnwidth=[100, 58, 58, 58, 58, 68],
            header=dict(values=header_vals, align=["left","center","center","center","center","center"]),
            cells=dict(values=cell_vals, align=["left","center","center","center","center","center"]),
            visible=vis
        ), row=1, col=2)
        table_traces_idx.append(len(fig.data) - 1)

    total_traces = len(fig.data)
    dummy_count  = len(COMBO_ORDER)

    # dropdown toggles
    buttons = []
    for idx, mfg in enumerate(companies):
        vis = [False] * total_traces

        # enable this company's TRx/ERx traces
        vis[2*idx:2*idx+2] = [True, True]

        # keep all legend dummies visible
        start_dummy = 2 * len(companies)
        vis[start_dummy:start_dummy + dummy_count] = [True] * dummy_count

        # show the corresponding table
        vis[table_traces_idx[idx]] = True

        # vertical lines for events
        evs = (merged_common[merged_common["DisplayMfg"] == mfg]
               .drop_duplicates("event_date")
               .dropna(subset=["event_date"]))
        shapes = []
        for _, r in evs.iterrows():
            style = COMBO_STYLE.get(r["combo_key"], dict(color="grey", dash="dot"))
            shapes.append(dict(
                type="line", x0=r["event_date"], x1=r["event_date"],
                y0=0, y1=1, yref="paper",
                line=dict(color=style["color"], dash=style["dash"], width=1)
            ))

        buttons.append(dict(
            label=mfg,
            method="update",
            args=[{"visible": vis},
                  {"title": f"Metformin Dispensing vs FDA Inspections – {mfg}",
                   "shapes": shapes}]
        ))

    fig.update_layout(
        title=f"Metformin Dispensing vs FDA Inspections – {companies[0]}",
        xaxis_title="Month",
        yaxis_title="TRx",
        yaxis2_title="ERx",
        legend=dict(orientation="h", x=0, y=1.11),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0.55, y=1.20, showactive=True)],
        height=650,
        width=1400,
        margin=dict(l=60, r=40, t=80, b=60)
    )

    _ensure_dir(PROCESSED_DIR)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", auto_open=AUTO_OPEN_HTML)
    print("Dashboard written to:", OUT_HTML)


# -------------------- RUN (no main) --------------------

redica_files = list_redica_files(REDICA_RAW_DIR)
if not redica_files:
    raise ValueError(f"No Redica files found under: {REDICA_RAW_DIR}")

# write source.txt first (so failed runs still show intended inputs)
_ensure_dir(PROCESSED_DIR)
# _write_source_txt(redica_files)

inspections = load_redica_inspections(redica_files)
print(f"Loaded {len(inspections):,} rows from {inspections['site_id'].nunique()} sites")

events = build_event_table(inspections)

iqvia = load_iqvia_monthly_by_mfg(IQVIA_XLSX)
iqvia["mfg_clean"]  = iqvia["Manufacturer"].apply(_clean_name)
events["mfg_clean"] = events["mfg"].apply(_clean_name)

# map IQVIA -> Redica manufacturer naming
iqvia["event_mfg_clean"] = iqvia["mfg_clean"].map(MANUAL_MAP)
iqvia_common = iqvia.dropna(subset=["event_mfg_clean"]).copy()

merged = iqvia_common.merge(
    events[["mfg_clean", "event_date", "district", "has_483", "enforcement"]],
    left_on="event_mfg_clean",
    right_on="mfg_clean",
    how="left"
)

merged["DisplayMfg"] = merged["event_mfg_clean"].str.title()

merged["combo_key"] = np.where(
    merged["enforcement"].fillna(False),
    "483-ENF",
    np.where(
        merged["has_483"].fillna(False),
        "483-" + merged["district"].map(ACTION_SHORT),
        "No483-" + merged["district"].map(ACTION_SHORT),
    ),
)

print("Post-merge sample:")
print(merged[["Manufacturer","event_mfg_clean","event_date","district"]].drop_duplicates().head())

build_dashboard(merged)

# %%
