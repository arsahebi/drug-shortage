"""
04_build_combined_obs_universe.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Merges the PDF+LLM pipeline observations with Redica structured
observations into a single master table covering 98 FEIs.

Inputs
  fdapdf_483_obs_llm_signals_anthropic.csv   (Step 1, source='pdf_llm')
  redica_483_observations.csv           (Step 0, source='redica')

Design
  Both sources are kept regardless of overlap — rows are NOT
  deduplicated across sources. A document present in both will have
  N_pdf rows with source='pdf_llm' and N_redica rows with
  source='redica'.  The source column lets downstream consumers
  filter or weight by provenance.

  Two convenience columns are added:
    best_severity  : LLM severity_tier if available, else Redica 3-tier
    best_vc        : LLM violation_category if available, else redica_vc

  Redica rows carry NaN for all LLM-only columns (scope, root_cause_type,
  remediation_signal, contamination_flag_llm, etc.) until LLM extraction
  is run on Redica text via 01_extract_observation_signals.py.

Output
  483_combined_obs_universe.csv   (one row per observation, all sources)

Coverage summary printed on exit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

OUR_CSV         = HERE / "step01_fdapdf_483_obs_llm_signals_anthropic.csv"
REDICA_CSV      = HERE / "step00_redica_483_observations.csv"
REDICA_LLM_CSV  = HERE / "step01_redica_483_obs_llm_signals_anthropic.csv"   # created by 01 --source redica --provider anthropic
OUT_CSV         = HERE / "step04_483_combined_obs_universe.csv"

# Columns carried from the PDF+LLM pipeline
PDF_COLS = [
    "fei", "insp_date", "obs_num", "obs_text_clean",
    "extraction_status", "confidence", "model_name",
    # regex flags
    "has_repeat_regex", "has_data_integrity_regex", "has_contamination_regex",
    "has_oos_oot_regex", "has_patient_risk_regex", "has_investigation_regex",
    "has_wl_ref_regex", "has_systemic_regex", "has_documentation_regex",
    "has_laboratory_regex", "has_equipment_facility_regex", "has_process_control_regex",
    "has_quality_unit_regex",
    # LLM categorical
    "violation_category", "severity_tier", "severity_rationale",
    "scope", "root_cause_type", "root_cause_rationale", "remediation_signal",
    # LLM binary flags
    "repeat_flag_llm", "patient_risk_flag_llm", "data_integrity_flag_llm",
    "contamination_flag_llm", "investigation_flag_llm",
    "evidence_quote",
]

# Columns carried from Redica
REDICA_COLS = [
    "fei", "insp_date", "obs_num", "obs_text",
    "redica_severity", "redica_qsl", "redica_vc",
    "redica_di_flag", "redica_di_labels",
    "document_id", "site_redica_id",
]

# Redica severity → our 4-tier (best approximation for "Other" bucket)
REDICA_SEV_MAP = {
    "Critical": "Critical",
    "Major":    "Major",
    "Other":    "Moderate",   # Moderate is the default "Other" tier
}


def load_pdf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep columns that exist
    keep = [c for c in PDF_COLS if c in df.columns]
    df = df[keep].copy()
    df["source"] = "pdf_llm"
    df = df.rename(columns={"obs_text_clean": "obs_text"})
    df["insp_date"] = pd.to_datetime(df["insp_date"]).dt.strftime("%Y-%m-%d")
    return df


def load_redica(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in REDICA_COLS if c in df.columns]
    df = df[keep].copy()
    df["source"] = "redica"
    df["insp_date"] = pd.to_datetime(df["insp_date"]).dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    pdf    = load_pdf(OUR_CSV)
    redica = load_redica(REDICA_CSV)

    print(f"PDF+LLM pipeline:  {len(pdf)} obs, {pdf['fei'].nunique()} FEIs")
    print(f"Redica structured: {len(redica)} obs, {redica['fei'].nunique()} FEIs")

    # ── merge Redica LLM signals if available ────────────────────────────────
    if REDICA_LLM_CSV.exists():
        llm_red = pd.read_csv(REDICA_LLM_CSV)
        llm_red = llm_red[llm_red["extraction_status"].isin(["ok", "partial"])].copy()
        llm_red["insp_date"] = pd.to_datetime(llm_red["insp_date"]).dt.strftime("%Y-%m-%d")
        llm_red["fei"] = llm_red["fei"].astype("int64")
        llm_cols = [c for c in llm_red.columns
                    if c in ("violation_category", "severity_tier", "severity_rationale",
                             "scope", "root_cause_type", "root_cause_rationale",
                             "remediation_signal", "evidence_quote", "confidence",
                             "model_name", "extraction_status")
                    or c.endswith("_flag_llm")]
        llm_red = llm_red[["fei", "insp_date", "obs_num"] + llm_cols]
        redica = redica.merge(llm_red, on=["fei", "insp_date", "obs_num"], how="left")
        n_scored = redica["violation_category"].notna().sum()
        print(f"Redica LLM signals merged: {n_scored}/{len(redica)} obs scored")
    else:
        print("Redica LLM signals not found — run: "
              "python3 01_extract_observation_signals.py --source redica")

    # ── concatenate ──────────────────────────────────────────────────────────
    combined = pd.concat([pdf, redica], ignore_index=True, sort=False)
    combined["fei"] = combined["fei"].astype("int64")

    # ── best_severity ────────────────────────────────────────────────────────
    # pdf_llm rows: use LLM 4-tier (already in severity_tier)
    # redica rows:  map Redica 3-tier to our 4-tier as best approximation
    redica_sev_approx = combined["redica_severity"].map(REDICA_SEV_MAP)
    combined["best_severity"] = combined["severity_tier"].fillna(redica_sev_approx)

    # ── best_vc ──────────────────────────────────────────────────────────────
    # pdf_llm rows: use LLM violation_category
    # redica rows:  use redica_vc (QSL-mapped)
    combined["best_vc"] = combined["violation_category"].fillna(combined.get("redica_vc"))

    # ── column order ─────────────────────────────────────────────────────────
    id_cols    = ["fei", "insp_date", "obs_num", "source"]
    text_cols  = ["obs_text"]
    meta_cols  = ["best_severity", "best_vc"]
    llm_cols   = [c for c in combined.columns
                  if c.endswith("_flag_llm") or c.endswith("_regex")
                  or c in ("violation_category", "severity_tier", "severity_rationale",
                           "scope", "root_cause_type", "root_cause_rationale",
                           "remediation_signal", "evidence_quote",
                           "confidence", "model_name", "extraction_status")]
    redica_meta = [c for c in combined.columns if c.startswith("redica_")
                   or c in ("document_id", "site_redica_id")]
    other_cols  = [c for c in combined.columns
                   if c not in id_cols + text_cols + meta_cols + llm_cols + redica_meta]

    ordered = (id_cols + text_cols + meta_cols + sorted(llm_cols)
               + sorted(redica_meta) + [c for c in other_cols if c not in id_cols])
    ordered = [c for c in dict.fromkeys(ordered) if c in combined.columns]
    combined = combined[ordered]

    combined = combined.sort_values(["fei", "insp_date", "source", "obs_num"]).reset_index(drop=True)
    combined.to_csv(OUT_CSV, index=False)

    # ── coverage summary ─────────────────────────────────────────────────────
    print(f"\nOutput → {OUT_CSV}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Total FEIs: {combined['fei'].nunique()}")
    print(f"\nRows by source:")
    print(combined["source"].value_counts().to_string())

    pdf_feis    = set(pdf["fei"].unique())
    redica_feis = set(redica["fei"].unique())
    both_feis   = pdf_feis & redica_feis
    only_pdf    = pdf_feis - redica_feis
    only_redica = redica_feis - pdf_feis

    print(f"\nFEI coverage:")
    print(f"  Both sources:     {len(both_feis)} FEIs  ({len(both_feis)} documents validate-able)")
    print(f"  PDF+LLM only:     {len(only_pdf)} FEIs  (LLM features only)")
    print(f"  Redica only:      {len(only_redica)} FEIs  (structured fields only; "
          f"run LLM extraction to add contamination/root-cause/scope)")

    print(f"\nLLM feature availability (pdf_llm rows only):")
    llm_flag_cols = [c for c in combined.columns if c.endswith("_flag_llm")]
    pdf_rows = combined[combined["source"] == "pdf_llm"]
    for c in sorted(llm_flag_cols):
        if c in pdf_rows.columns:
            rate = pdf_rows[c].mean()
            print(f"  {c:35s}: {rate:.1%}")

    print(f"\nRedica structured feature availability (redica rows only):")
    red_rows = combined[combined["source"] == "redica"]
    for c in ["redica_severity", "redica_vc", "redica_di_flag"]:
        if c in red_rows.columns:
            n_ok = red_rows[c].notna().sum()
            print(f"  {c:35s}: {n_ok}/{len(red_rows)} non-null")

    print("""
Next steps
  1. Run 03_validate_vs_redica.py to compare LLM vs Redica on the
     overlapping documents before using combined features in the model.
  2. To get LLM features (contamination, root cause, scope, remediation)
     for the 98 Redica FEIs, run 01_extract_observation_signals.py with
     the Redica obs_text as input. Estimated cost: ~$1–2 at gpt-4o-mini.
  3. Feed this combined universe into 02_aggregate_fei_features.py
     (add source filter as needed) to expand FEI coverage from 38 → 98.
""")


if __name__ == "__main__":
    main()
