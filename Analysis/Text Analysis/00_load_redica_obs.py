"""
00_load_redica_obs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loads and standardizes Redica 483 observation data into the same
observation schema used by the PDF+LLM pipeline (Step 1).

Inputs
  Data/07 - Redica/Raw/Valisure14_FDA_483_Observations_WL_Deficiencies_OSU.xlsx
      sheet: FDA-483s Obs + WL Deficiencies
  Data/07 - Redica/Raw/Valisure14_Site_List.xlsx
      columns: Site Redica Id, Site Display Name, FEI

Output
  step00_redica_483_observations.csv  (one row per 483 observation)
  step00_redica_inspections.csv       (one row per inspection, added 2026-10-19)

Inspection table (2026-10-19)
  One row per (site, Date Issued). The FDA grade comes from Redica's
  Valisure14_Sites_Red_Flag_Events.xlsx, column F "Risk Event Attribute
  Value", taking ONLY the "<grade>: Drug Quality Assurance" label (same rule
  as the Metformin paper). Event Type must be "Inspection" and Event Date
  must equal the 483's Date Issued.
    keep = False when
      * no 483 observation of the inspection has "Human Drugs" in Industry List, or
      * the event carries grades only for other FDA programs (devices,
        bioresearch monitoring, pre-approval, blood, animal drugs).
    keep = True with a blank grade when the event has no grade posted yet,
    or no event matches; these are "unknown grade".
  Observations are NOT dropped from the observation file, because the LLM
  extraction was run on all of them. 02_aggregate_fei_features.py drops the
  observations of inspections with keep = False.

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
  Materials (storage/temp) → BuildingsEquipment  (keyword split on obs_summary)
  Materials (testing/samp) → LabControls         (keyword split on obs_summary)
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
DATA = HERE.parent.parent / "Data"

REDICA_OBS_XLSX = DATA / "07 - Redica" / "Raw" / "Valisure14_FDA_483_Observations_WL_Deficiencies_OSU.xlsx"
SITE_LIST_XLSX  = DATA / "07 - Redica" / "Raw" / "Valisure14_Site_List.xlsx"
EVENTS_XLSX     = DATA / "07 - Redica" / "Raw" / "Valisure14_Sites_Red_Flag_Events.xlsx"
OUT_INSP_CSV    = HERE / "step00_redica_inspections.csv"
SHEET_NAME      = "FDA-483s Obs + WL Deficiencies"
OUT_CSV         = HERE / "step00_redica_483_observations.csv"

QSL_TO_VC: dict[str, str] = {
    "Laboratory":               "LabControls",
    "Production":               "ProductionControls",
    "Facilities and Equipment": "BuildingsEquipment",
    "Quality Unit":             "QualitySystem",
    "Packaging and Labeling":   "PackagingLabeling",
    # "Materials" is handled separately in _map_materials_vc() below
}

# Keywords that indicate a storage/warehousing concern within Materials observations
_STORAGE_KEYWORDS = ("temperature", "humidity", "warehousing procedure", "temperature mapping", "temperature condition")


def _map_materials_vc(text: str) -> str:
    """
    Split Redica 'Materials' QSL into BuildingsEquipment (storage/temperature)
    or LabControls (incoming testing, sampling, supplier qualification).
    Manual review of all 33 Materials observations confirmed two themes:
      ~12 rows: storage conditions, temperature/humidity warehousing → BuildingsEquipment
      ~21 rows: component testing, sampling, supplier qualification → LabControls
    """
    if any(kw in str(text).lower() for kw in _STORAGE_KEYWORDS):
        return "BuildingsEquipment"
    return "LabControls"


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


def _grades(val) -> list[str]:
    """Column F of the red flag events file, e.g.
    '["483", "VAI: Drug Quality Assurance"]' -> ['VAI: Drug Quality Assurance']."""
    try:
        items = json.loads(str(val))
    except (json.JSONDecodeError, TypeError):
        return []
    return [x for x in items if str(x)[:4] in ("NAI:", "VAI:", "OAI:")]


def build_inspections(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per inspection with its Drug Quality Assurance grade and a keep flag.
    obs = 483 observations with columns Site Redica Id, fei, Date Issued, Industry List."""
    # Step 1. One row per inspection = (site, Date Issued) of a 483.
    obs = obs.copy()
    obs["insp_date"] = pd.to_datetime(obs["Date Issued"]).dt.normalize()
    obs["human_drugs"] = obs["Industry List"].astype(str).str.contains("Human Drugs")
    insp = (obs.groupby(["Site Redica Id", "fei", "insp_date"], as_index=False)
               .agg(has_human_drugs=("human_drugs", "any")))

    # Step 2. Inspection events from Redica's red flag file, column F grades.
    ev = pd.read_excel(EVENTS_XLSX)
    ev = ev[ev["Event Type"] == "Inspection"].copy()
    ev["insp_date"] = pd.to_datetime(ev["Event Date"]).dt.normalize()
    ev["grades"] = ev["Risk Event Attribute Value"].apply(_grades)
    ev = (ev.groupby(["Site Redica Id", "insp_date"])["grades"]
            .apply(lambda g: sorted({x for lst in g for x in lst})).reset_index())

    # Step 3. Match on site and exact date.
    insp = insp.merge(ev, on=["Site Redica Id", "insp_date"], how="left",
                      indicator=True)
    insp["has_event"] = insp["_merge"] == "both"
    insp["grades"] = insp["grades"].apply(lambda g: g if isinstance(g, list) else [])

    # Step 4. Drug Quality Assurance grade; the worse one if there are two.
    def dqa_grade(grades):
        g = [x[:3] for x in grades if x.endswith("Drug Quality Assurance")]
        for level in ("OAI", "VAI", "NAI"):
            if level in g:
                return level
        return ""
    insp["dqa_grade"] = insp["grades"].apply(dqa_grade)
    insp["other_programs"] = insp["grades"].apply(
        lambda g: "; ".join(x for x in g if not x.endswith("Drug Quality Assurance")))

    # Step 5. Keep rule.
    only_other = (insp["dqa_grade"] == "") & (insp["other_programs"] != "")
    insp["drop_reason"] = ""
    insp.loc[only_other, "drop_reason"] = "graded only under another FDA program"
    insp.loc[~insp["has_human_drugs"], "drop_reason"] = "not tagged Human Drugs"
    insp["keep"] = insp["drop_reason"] == ""

    insp["insp_date"] = insp["insp_date"].dt.strftime("%Y-%m-%d")
    insp = insp.rename(columns={"Site Redica Id": "site_redica_id"})
    insp["grades"] = insp["grades"].apply("; ".join)
    return insp[["fei", "site_redica_id", "insp_date", "has_human_drugs", "has_event",
                 "dqa_grade", "other_programs", "grades", "keep", "drop_reason"]]


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

    # ── inspection table: DQA grade and keep flag ──────────────────────────
    insp = build_inspections(obs)
    insp.to_csv(OUT_INSP_CSV, index=False)
    k = insp[insp["keep"]]
    print(f"\nInspections: {len(insp)}, kept {len(k)}")
    print(f"  dropped: {insp.loc[~insp['keep'], 'drop_reason'].value_counts().to_dict()}")
    print(f"  kept, DQA grade: {k['dqa_grade'].replace('', 'unknown').value_counts().to_dict()}")
    print(f"  kept, unknown grade: event with no grade {((k['dqa_grade'] == '') & k['has_event']).sum()}, "
          f"no matching event {(~k['has_event']).sum()}")
    print(f"Output → {OUT_INSP_CSV}\n")

    # ── DI Labels ───────────────────────────────────────────────────────────
    obs["_di_list"]        = obs["DI Labels"].apply(_parse_di_labels)
    obs["redica_di_flag"]  = obs["_di_list"].apply(lambda x: len(x) > 0)
    obs["redica_di_labels"] = obs["_di_list"].apply(lambda x: "; ".join(x))

    # ── QSL → violation_category ────────────────────────────────────────────
    obs["redica_vc"] = obs["QSL Area"].map(QSL_TO_VC)
    mat_mask = obs["QSL Area"] == "Materials"
    obs.loc[mat_mask, "redica_vc"] = obs.loc[mat_mask, "Observation/Deficiency Summary"].apply(_map_materials_vc)
    obs["redica_vc"] = obs["redica_vc"].fillna("Other")

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
