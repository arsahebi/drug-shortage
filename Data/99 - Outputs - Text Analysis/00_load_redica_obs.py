"""
00_load_redica_obs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loads and standardizes Redica 483 observation data into the same
observation schema used by the PDF+LLM pipeline (Step 1).

Inputs
  Data/07 - Redica/Raw/FDA-483s Observations + WL Deficiencies_OSU.xlsx
      sheet: FDA-483s Obs + WL Deficiencies
  Data/07 - Redica/Raw/Site List.xlsx
      columns: Site Redica Id, Site Display Name, FEI

Output
  redica_483_observations.csv  (one row per 483 observation)

  fei              : FDA Facility Establishment Identifier (int64)
  insp_date        : inspection date, YYYY-MM-DD
  obs_num          : observation number within the document (numeric string)
  obs_text         : full observation/deficiency text
  obs_summary      : Redica-generated one-paragraph summary
  redica_severity  : Redica's 3-tier severity (Critical / Major / Other / NaN)
  redica_qsl       : QSL Area label — Redica's regulatory domain label
  redica_vc        : QSL Area mapped to our 8-class violation_category schema
  redica_di_flag   : True when DI Labels list is non-empty
  redica_di_labels : semicolon-joined DI label strings
  source           : 'redica' (constant; distinguishes from 'pdf_llm' rows)
  document_id      : Redica document identifier (RDO...)
  site_redica_id   : Redica site identifier (RSI...)

QSL Area → violation_category mapping
  Laboratory               → LabControls
  Production               → ProductionControls
  Facilities and Equipment → BuildingsEquipment
  Quality Unit             → QualitySystem
  Packaging and Labeling   → PackagingLabeling
  Materials                → ProductionControls  (no exact match; closest domain)
  NaN / unlisted           → Other

Note: Redica severity uses 3 tiers (Critical / Major / Other).
  "Other" maps to our Moderate+Minor combined; the finer split requires
  LLM extraction (run 01_extract_observation_signals.py on obs_text).
  Warning letters in the source file are filtered out — only doc_type='483'.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent  # Data/

REDICA_OBS_XLSX = DATA / "07 - Redica" / "Raw" / "FDA-483s Observations + WL Deficiencies_OSU.xlsx"
SITE_LIST_XLSX  = DATA / "07 - Redica" / "Raw" / "Site List.xlsx"
SHEET_NAME      = "FDA-483s Obs + WL Deficiencies"
OUT_CSV         = HERE / "redica_483_observations.csv"

QSL_TO_VC: dict[str, str] = {
    "Laboratory":               "LabControls",
    "Production":               "ProductionControls",
    "Facilities and Equipment": "BuildingsEquipment",
    "Quality Unit":             "QualitySystem",
    "Packaging and Labeling":   "PackagingLabeling",
    "Materials":                "ProductionControls",
}


def _parse_di_labels(val) -> list[str]:
    """Parse Redica DI Labels JSON string → list of label strings."""
    if pd.isna(val) or str(val).strip() in ("", "[]"):
        return []
    try:
        parsed = json.loads(str(val))
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_obs_num(val) -> str:
    """'observation_3' → '3', 'deficiency_2' → '2', else return as-is."""
    if pd.isna(val):
        return ""
    parts = str(val).rsplit("_", 1)
    return parts[-1] if len(parts) == 2 and parts[-1].isdigit() else str(val)


def main() -> None:
    print("Loading Redica source files...")
    raw   = pd.read_excel(REDICA_OBS_XLSX, sheet_name=SHEET_NAME)
    sites = pd.read_excel(SITE_LIST_XLSX)

    print(f"  Raw rows (all doc types): {len(raw)}")
    print(f"  Document types: {raw['Document Type'].value_counts().to_dict()}")

    # ── filter to 483 only ──────────────────────────────────────────────────
    obs = raw[raw["Document Type"] == "483"].copy()
    print(f"  483 observations kept: {len(obs)}")

    # ── join FEI crosswalk ──────────────────────────────────────────────────
    sites_clean = sites[["Site Redica Id", "FEI"]].rename(columns={"FEI": "fei"})
    obs = obs.merge(sites_clean, on="Site Redica Id", how="left")
    n_no_fei = obs["fei"].isna().sum()
    if n_no_fei:
        print(f"  Warning: {n_no_fei} observations have no FEI match — dropped")

    obs = obs.dropna(subset=["fei"]).copy()
    obs["fei"] = obs["fei"].astype("int64")

    # ── DI Labels ───────────────────────────────────────────────────────────
    obs["_di_list"]        = obs["DI Labels"].apply(_parse_di_labels)
    obs["redica_di_flag"]  = obs["_di_list"].apply(lambda x: len(x) > 0)
    obs["redica_di_labels"] = obs["_di_list"].apply(lambda x: "; ".join(x))

    # ── QSL → violation_category ────────────────────────────────────────────
    obs["redica_vc"] = obs["QSL Area"].map(QSL_TO_VC).fillna("Other")

    # ── build output ────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "fei":              obs["fei"],
        "insp_date":        pd.to_datetime(obs["Date Issued"]).dt.strftime("%Y-%m-%d"),
        "obs_num":          obs["Observation/Deficiency Number"].apply(_extract_obs_num),
        "obs_text":         obs["Observation/Deficiency Text"],
        "obs_summary":      obs["Observation/Deficiency Summary"],
        "redica_severity":  obs["Observation/Deficiency Severity"],
        "redica_qsl":       obs["QSL Area"],
        "redica_vc":        obs["redica_vc"],
        "redica_di_flag":   obs["redica_di_flag"],
        "redica_di_labels": obs["redica_di_labels"],
        "source":           "redica",
        "document_id":      obs["Document Redica Id"],
        "site_redica_id":   obs["Site Redica Id"],
    }).reset_index(drop=True)

    n_before = len(out)
    out = out.drop_duplicates(subset=["fei", "insp_date", "obs_num"]).reset_index(drop=True)
    if len(out) < n_before:
        print(f"  Dropped {n_before - len(out)} duplicate (fei, insp_date, obs_num) rows")
    out.to_csv(OUT_CSV, index=False)

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\nOutput → {OUT_CSV}")
    print(f"  Rows : {len(out)}")
    print(f"  FEIs : {out['fei'].nunique()}")
    print(f"  Date range: {out['insp_date'].min()} → {out['insp_date'].max()}")
    print(f"\nSeverity distribution (Redica 3-tier):")
    print(out["redica_severity"].value_counts(dropna=False).to_string())
    print(f"\nViolation category (mapped from QSL Area):")
    print(out["redica_vc"].value_counts(dropna=False).to_string())
    print(f"\nDI flag rate: {out['redica_di_flag'].mean():.1%}  "
          f"({out['redica_di_flag'].sum()} flagged observations)")


if __name__ == "__main__":
    main()
