"""
05_aggregate_fei_features.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Aggregates observation-level signals from Step 4 up to FEI-level features
  in five interpretable layers, enabling separate evaluation of each signal source.

PIPELINE POSITION
  Step 4  (04_extract_observation_signals.py)
      Reads : Data/12 - FDA - 483/processed/483_observations.csv
      Writes: Data/99 - Outputs - Text Analysis/483_observation_context_signals.csv
              → one row per observation; regex flags + LLM semantic fields

  Step 5  (this script)
      Reads : 483_observation_context_signals.csv  (from Step 4)
      Writes: 483_fei_context_features.csv
              → one row per FEI; layered text-derived features

  These are STANDALONE text-analysis outputs in folder 99.
  They do NOT overwrite Data/12 - FDA - 483/processed/483_fei_features.csv,
  which is the original regex FEI summary from folder 12.
  Prediction / shortage modeling happens later in a separate folder.

WHY CFR DOMAIN FEATURES ARE EXCLUDED
  483 observation text does not contain explicit CFR citations — inspectors
  write narrative findings, not regulation references.  CFR domain features
  already exist in Data/14 - FDA - Inspection/processed/ as structured data
  and should be merged directly in the prediction model.

WHY WARNING LETTER TEXT IS EXCLUDED
  Data/21 - FDA - Warning Letter/processed/ already has a regex/rule-based
  pipeline that extracts CFR codes, domain counts, repeat flags, and a severity
  score.  Those features should also be merged in the prediction model directly.

STEP 4 DATA QUALITY NOTES (as of full run)
  - 347 total rows; 345 ok, 1 json_error, 1 empty_response
  - (fei, filename, obs_num) is NOT always unique — OCR/splitting artifacts
    produce duplicate row keys; do not treat it as a primary key
  - 13 rows have confidence < LOW_CONFIDENCE_THRESHOLD (OCR fragments)
  - 42 rows have blank evidence_quote; do not rely on it for aggregation
  - LLM flags are semantic screening signals, not ground truth:
      contamination_flag_llm = contamination/sterility-control risk (not confirmed)
      patient_risk_flag_llm  = potential patient-risk relevance (not confirmed harm)
  - 219 / 345 scored rows have no remediation_signal — this means "none mentioned",
    which is itself an important signal, not a data gap

DENOMINATOR RULES
  Source/quality counts  → all rows
  Regex shares           → all rows
  LLM shares             → scored rows only (extraction_status in ok | partial)
  Agreement features     → scored rows only (both regex + LLM values exist there)
  Failed rows            → excluded from LLM features; never become false/zero labels

FEATURE LAYERS
  Layer 1 — Source/extraction quality
  Layer 2 — Regex baseline (deterministic)
  Layer 3 — LLM semantic (model-derived, scored rows only)
  Layer 4 — Regex/LLM agreement and semantic lift
  Layer 5 — Composite risk indices (transparent weighted scores)

COMPOSITE INDEX FORMULAS  (weights defined as constants below)

  text_risk_index (TRI) [0, 100]:
    0.35 × severity_high_share
  + 0.20 × severity_moderate_share
  + 0.20 × (1 − remediation_strong_share)
  + 0.15 × repeat_llm_share
  + 0.10 × systemic_llm_share

  sterility_contamination_risk_index (SCRI) [0, 100]:
    0.50 × contamination_llm_share
  + 0.30 × contamination_regex_share
  + 0.20 × severity_high_share

  investigation_remediation_weakness_index (IRWI) [0, 100]:
    0.40 × remediation_none_share
  + 0.35 × investigation_llm_share
  + 0.25 × remediation_weak_share

  quality_culture_index (QCI) [0, 100]:
    0.40 × systemic_llm_share
  + 0.35 × repeat_llm_share
  + 0.25 × cultural_root_cause_share
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
SIGNALS_CSV = HERE / "483_observation_context_signals.csv"
FEI_CSV     = HERE / "483_fei_context_features.csv"

# Rows with confidence below this threshold are flagged as low-quality OCR fragments.
# Based on Step 4 data: min=0.12, mean=0.83; 13 rows fall below 0.70.
LOW_CONFIDENCE_THRESHOLD = 0.70

# ── Composite index weights ────────────────────────────────────────────────

# text_risk_index (TRI)
# severity_high (0.35):        direct patient-harm potential; top priority
# severity_moderate (0.20):    process deviations that accumulate into supply risk
# no strong remediation (0.20): absence of CAPA predicts recurrence
# repeat (0.15):               empirically linked to Warning Letters / escalation
# systemic (0.10):             facility-wide failure; lower weight, overlaps severity
_TRI_W = dict(
    severity_high_share     = 0.35,
    severity_moderate_share = 0.20,
    no_strong_remediation   = 0.20,
    repeat_llm_share        = 0.15,
    systemic_llm_share      = 0.10,
)
assert abs(sum(_TRI_W.values()) - 1.0) < 1e-9, "TRI weights must sum to 1.0"

# sterility_contamination_risk_index (SCRI)
# LLM signal weighted highest; regex adds deterministic confirmation; severity anchors
_SCRI_W = dict(
    contamination_llm_share   = 0.50,
    contamination_regex_share = 0.30,
    severity_high_share       = 0.20,
)

# investigation_remediation_weakness_index (IRWI)
# Missing remediation is the strongest predictor of recurrence;
# investigation discussion shows awareness even without action
_IRWI_W = dict(
    remediation_none_share  = 0.40,
    investigation_llm_share = 0.35,
    remediation_weak_share  = 0.25,
)

# quality_culture_index (QCI)
# Systemic + repeat + cultural root cause together signal culture breakdown
_QCI_W = dict(
    systemic_llm_share        = 0.40,
    repeat_llm_share          = 0.35,
    cultural_root_cause_share = 0.25,
)


# ── Generic helpers ────────────────────────────────────────────────────────

def _share(series: pd.Series) -> float:
    """Fraction of truthy values; 0.0 for empty."""
    if len(series) == 0:
        return 0.0
    return float(series.astype(bool).mean())


def _value_share(series: pd.Series, value) -> float:
    """Fraction of non-NaN rows equal to value."""
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid == value).sum()) / len(valid)


def _mode_or_default(series: pd.Series, default: str = "Other") -> str:
    counts = series.dropna().value_counts()
    return str(counts.index[0]) if len(counts) > 0 else default


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce True/False/1/0/string variants to bool."""
    return series.astype(str).str.lower().isin(["true", "1"])


def _is_blank(series: pd.Series) -> pd.Series:
    """True where a string column is NaN or empty / whitespace-only."""
    return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")


# ── Layer 1: Source / extraction quality ───────────────────────────────────

def _layer1_quality(grp: pd.DataFrame, ns: int) -> dict:
    """
    Counts and dates describing how much evidence exists for this FEI
    and how reliable the extraction was.
    """
    n      = len(grp)
    n_fail = n - ns

    scored = grp[grp["extraction_status"].isin(["ok", "partial"])]
    n_low_conf = int(
        (scored["confidence"] < LOW_CONFIDENCE_THRESHOLD).sum()
    ) if ns > 0 and "confidence" in scored.columns else 0

    n_blank_eq = int(_is_blank(grp["evidence_quote"]).sum()) if "evidence_quote" in grp.columns else 0

    mean_conf = round(float(scored["confidence"].mean()), 4) \
        if ns > 0 and "confidence" in scored.columns else float("nan")

    dates = pd.to_datetime(grp["insp_date"], errors="coerce")

    return {
        "n_obs_total":            n,
        "n_obs_scored":           ns,
        "n_obs_failed":           n_fail,
        "n_obs_low_confidence":   n_low_conf,
        "mean_confidence":        mean_conf,
        "n_blank_evidence_quotes": n_blank_eq,
        "n_files_483":            int(grp["filename"].nunique()),
        "earliest_483_date":      dates.min(),
        "latest_483_date":        dates.max(),
    }


# ── Layer 2: Regex baseline ────────────────────────────────────────────────

def _layer2_regex(grp: pd.DataFrame) -> dict:
    """
    Aggregate deterministic flag columns (has_*_regex) to per-FEI shares.
    Uses ALL rows — regex runs independently of LLM extraction, so failed
    rows still have valid regex flags.
    """
    pairs = [
        ("repeat_regex_share",             "has_repeat_regex"),
        ("systemic_regex_share",           "has_systemic_regex"),
        ("documentation_regex_share",      "has_documentation_regex"),
        ("investigation_regex_share",      "has_investigation_regex"),
        ("contamination_regex_share",      "has_contamination_regex"),
        ("data_integrity_regex_share",     "has_data_integrity_regex"),
        ("patient_risk_regex_share",       "has_patient_risk_regex"),
        ("quality_unit_regex_share",       "has_quality_unit_regex"),
        ("laboratory_regex_share",         "has_laboratory_regex"),
        ("equipment_facility_regex_share", "has_equipment_facility_regex"),
        ("process_control_regex_share",    "has_process_control_regex"),
        # extra flags present in Step 4 output — kept for completeness
        ("wl_ref_regex_share",             "has_wl_ref_regex"),
        ("oos_oot_regex_share",            "has_oos_oot_regex"),
    ]
    out = {}
    for feat, col in pairs:
        out[feat] = round(_share(_to_bool(grp[col])), 4) if col in grp.columns else float("nan")
    return out


# ── Layer 3: LLM semantic ──────────────────────────────────────────────────

def _layer3_llm(scored: pd.DataFrame, ns: int) -> dict:
    """
    Aggregate LLM-derived categorical fields and binary flags.
    Uses ONLY scored rows (ok / partial) so failed rows never become
    false/zero LLM labels.

    remediation_none_share: NaN and literal 'None' both mean no remediation
    mentioned — this is itself a meaningful signal, not a data gap.
    """
    sev = scored["severity_tier"]      if ns > 0 else pd.Series(dtype=str)
    rem = scored["remediation_signal"] if ns > 0 else pd.Series(dtype=str)
    rc  = scored["root_cause_type"]    if ns > 0 else pd.Series(dtype=str)
    vc  = scored["violation_category"] if ns > 0 else pd.Series(dtype=str)

    # remediation_none_share: fraction with no remediation information
    rem_none = round(
        float((rem.isna() | (rem.astype(str).str.strip().isin(["None", "nan", ""]))).sum())
        / max(ns, 1), 4
    ) if ns > 0 else float("nan")

    llm_flags = [
        ("repeat_llm_share",         "repeat_flag_llm"),
        ("systemic_llm_share",       "systemic_flag_llm"),
        ("patient_risk_llm_share",   "patient_risk_flag_llm"),
        ("data_integrity_llm_share", "data_integrity_flag_llm"),
        ("contamination_llm_share",  "contamination_flag_llm"),
        ("documentation_llm_share",  "documentation_flag_llm"),
        ("investigation_llm_share",  "investigation_flag_llm"),
    ]
    flag_shares = {}
    for feat, col in llm_flags:
        if col in scored.columns and ns > 0:
            flag_shares[feat] = round(_share(_to_bool(scored[col])), 4)
        else:
            flag_shares[feat] = float("nan")

    out = {
        "severity_high_share":         round(_value_share(sev, "High"),     4),
        "severity_moderate_share":     round(_value_share(sev, "Moderate"), 4),
        "severity_low_share":          round(_value_share(sev, "Low"),       4),
        "dominant_violation_category": _mode_or_default(vc,  "Other"),
        "dominant_root_cause":         _mode_or_default(rc,  "Unclear"),
        "capital_root_cause_share":    round(_value_share(rc, "Capital"),    4),
        "cultural_root_cause_share":   round(_value_share(rc, "Cultural"),   4),
        "mixed_root_cause_share":      round(_value_share(rc, "Mixed"),      4),
        "unclear_root_cause_share":    round(_value_share(rc, "Unclear"),    4),
        "remediation_strong_share":    round(_value_share(rem, "Strong"),    4),
        "remediation_partial_share":   round(_value_share(rem, "Partial"),   4),
        "remediation_weak_share":      round(_value_share(rem, "Weak"),      4),
        "remediation_none_share":      rem_none,
    }
    out.update(flag_shares)
    return out


# ── Layer 4: Agreement and semantic lift ───────────────────────────────────

def _layer4_agreement(scored: pd.DataFrame, ns: int) -> dict:
    """
    Compares regex and LLM signals at the observation level before aggregating.

    *_regex_llm_agreement : fraction of scored obs where both flags agree
                            (both True or both False)
    *_llm_only_share      : fraction where LLM=True AND regex=False
                            ("semantic lift" — risk the regex missed)

    Uses SCORED rows only: regex flags are always present, LLM flags only
    exist on successfully extracted rows, so this is the right denominator.
    """
    paired = [
        ("repeat",         "has_repeat_regex",        "repeat_flag_llm"),
        ("systemic",       "has_systemic_regex",       "systemic_flag_llm"),
        ("contamination",  "has_contamination_regex",  "contamination_flag_llm"),
        ("data_integrity", "has_data_integrity_regex", "data_integrity_flag_llm"),
        ("documentation",  "has_documentation_regex",  "documentation_flag_llm"),
        ("investigation",  "has_investigation_regex",  "investigation_flag_llm"),
        ("patient_risk",   "has_patient_risk_regex",   "patient_risk_flag_llm"),
    ]
    out = {}
    for label, rx_col, lm_col in paired:
        if rx_col in scored.columns and lm_col in scored.columns and ns > 0:
            rx = _to_bool(scored[rx_col])
            lm = _to_bool(scored[lm_col])
            out[f"{label}_regex_llm_agreement"] = round(float((rx == lm).mean()), 4)
            out[f"{label}_llm_only_share"]       = round(float((~rx & lm).mean()),  4)
        else:
            out[f"{label}_regex_llm_agreement"] = float("nan")
            out[f"{label}_llm_only_share"]       = float("nan")
    return out


# ── Layer 5: Composite indices ─────────────────────────────────────────────

def _layer5_indices(l2: dict, l3: dict) -> dict:
    """
    Transparent weighted scores built from named component columns.
    NaN components are treated as 0.0 so one missing feature doesn't
    propagate NaN into every index.  Component features are preserved in
    the output so models can use raw layers or composite scores independently.
    """
    def _g(d: dict, k: str) -> float:
        v = d.get(k)
        return 0.0 if (v is None or (isinstance(v, float) and v != v)) else float(v)

    tri = (
        _TRI_W["severity_high_share"]     * _g(l3, "severity_high_share")
      + _TRI_W["severity_moderate_share"] * _g(l3, "severity_moderate_share")
      + _TRI_W["no_strong_remediation"]   * (1.0 - _g(l3, "remediation_strong_share"))
      + _TRI_W["repeat_llm_share"]        * _g(l3, "repeat_llm_share")
      + _TRI_W["systemic_llm_share"]      * _g(l3, "systemic_llm_share")
    ) * 100

    scri = (
        _SCRI_W["contamination_llm_share"]   * _g(l3, "contamination_llm_share")
      + _SCRI_W["contamination_regex_share"] * _g(l2, "contamination_regex_share")
      + _SCRI_W["severity_high_share"]       * _g(l3, "severity_high_share")
    ) * 100

    irwi = (
        _IRWI_W["remediation_none_share"]  * _g(l3, "remediation_none_share")
      + _IRWI_W["investigation_llm_share"] * _g(l3, "investigation_llm_share")
      + _IRWI_W["remediation_weak_share"]  * _g(l3, "remediation_weak_share")
    ) * 100

    qci = (
        _QCI_W["systemic_llm_share"]        * _g(l3, "systemic_llm_share")
      + _QCI_W["repeat_llm_share"]          * _g(l3, "repeat_llm_share")
      + _QCI_W["cultural_root_cause_share"] * _g(l3, "cultural_root_cause_share")
    ) * 100

    return {
        "text_risk_index":                          round(tri,  2),
        "sterility_contamination_risk_index":       round(scri, 2),
        "investigation_remediation_weakness_index": round(irwi, 2),
        "quality_culture_index":                    round(qci,  2),
    }


# ── Top-level aggregation ──────────────────────────────────────────────────

def _aggregate_fei(grp: pd.DataFrame) -> dict:
    scored = grp[grp["extraction_status"].isin(["ok", "partial"])]
    ns     = len(scored)

    l1 = _layer1_quality(grp, ns)
    l2 = _layer2_regex(grp)
    l3 = _layer3_llm(scored, ns)
    l4 = _layer4_agreement(scored, ns)
    l5 = _layer5_indices(l2, l3)

    flat = {}
    for layer in (l1, l2, l3, l4, l5):
        flat.update(layer)
    return flat


# ── Column ordering ────────────────────────────────────────────────────────

_COL_ORDER = [
    # key
    "fei",
    # Layer 1 — source / extraction quality
    "n_obs_total", "n_obs_scored", "n_obs_failed", "n_obs_low_confidence",
    "mean_confidence", "n_blank_evidence_quotes",
    "n_files_483", "earliest_483_date", "latest_483_date",
    # Layer 2 — regex baseline
    "repeat_regex_share", "systemic_regex_share", "documentation_regex_share",
    "investigation_regex_share", "contamination_regex_share",
    "data_integrity_regex_share", "patient_risk_regex_share",
    "quality_unit_regex_share", "laboratory_regex_share",
    "equipment_facility_regex_share", "process_control_regex_share",
    "wl_ref_regex_share", "oos_oot_regex_share",
    # Layer 3 — LLM semantic (scored rows only)
    "severity_high_share", "severity_moderate_share", "severity_low_share",
    "dominant_violation_category", "dominant_root_cause",
    "capital_root_cause_share", "cultural_root_cause_share",
    "mixed_root_cause_share", "unclear_root_cause_share",
    "remediation_strong_share", "remediation_partial_share",
    "remediation_weak_share", "remediation_none_share",
    "repeat_llm_share", "systemic_llm_share", "patient_risk_llm_share",
    "data_integrity_llm_share", "contamination_llm_share",
    "documentation_llm_share", "investigation_llm_share",
    # Layer 4 — agreement / semantic lift (scored rows only)
    "repeat_regex_llm_agreement",         "repeat_llm_only_share",
    "systemic_regex_llm_agreement",       "systemic_llm_only_share",
    "contamination_regex_llm_agreement",  "contamination_llm_only_share",
    "data_integrity_regex_llm_agreement", "data_integrity_llm_only_share",
    "documentation_regex_llm_agreement",  "documentation_llm_only_share",
    "investigation_regex_llm_agreement",  "investigation_llm_only_share",
    "patient_risk_regex_llm_agreement",   "patient_risk_llm_only_share",
    # Layer 5 — composite indices
    "text_risk_index",
    "sterility_contamination_risk_index",
    "investigation_remediation_weakness_index",
    "quality_culture_index",
]


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("05_aggregate_fei_features.py — Layered FEI feature aggregation")
    print(sep)

    if not SIGNALS_CSV.exists():
        sys.exit(
            f"\n[ERROR] Observation signals file not found:\n  {SIGNALS_CSV}\n"
            "Run 04_extract_observation_signals.py first.\n"
        )

    df = pd.read_csv(SIGNALS_CSV)

    # ── Input QA ────────────────────────────────────────────────────────────
    print(f"\n── Input QA ──")
    print(f"  Rows loaded          : {len(df):,}")
    print(f"  Unique FEIs          : {df['fei'].nunique()}")
    print(f"  Unique filenames     : {df['filename'].nunique()}")
    dup_mask = df.duplicated(subset=["fei", "filename", "obs_num"], keep=False)
    print(f"  Duplicate row keys   : {dup_mask.sum()} rows share a (fei,filename,obs_num)"
          f"  [OCR/splitting artifact — not deduplicated]")
    status_counts = df["extraction_status"].value_counts(dropna=False)
    for status, cnt in status_counts.items():
        print(f"  extraction_status={status!s:<18}: {cnt}")

    # Coerce types before aggregation
    df["fei"]        = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
    df["n_cfrs"]     = pd.to_numeric(df["n_cfrs"],     errors="coerce").fillna(0)
    df["n_examples"] = pd.to_numeric(df["n_examples"], errors="coerce").fillna(0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df = df.dropna(subset=["fei"])
    print(f"  Rows with valid FEI  : {len(df):,}")

    scored_mask = df["extraction_status"].isin(["ok", "partial"])
    scored_all  = df[scored_mask]
    print(f"  Scored rows (ok/partial) : {scored_mask.sum()}")
    if "confidence" in scored_all.columns:
        low_conf = (scored_all["confidence"] < LOW_CONFIDENCE_THRESHOLD).sum()
        print(f"  Low-confidence rows (<{LOW_CONFIDENCE_THRESHOLD}) : {low_conf}")
    if "evidence_quote" in df.columns:
        blank_eq = _is_blank(df["evidence_quote"]).sum()
        print(f"  Blank evidence_quotes    : {blank_eq}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    rows = []
    for fei_val, grp in df.groupby("fei"):
        agg        = _aggregate_fei(grp.copy())
        agg["fei"] = int(fei_val)
        rows.append(agg)

    fei_df = pd.DataFrame(rows)

    ordered = [c for c in _COL_ORDER if c in fei_df.columns]
    extras  = [c for c in fei_df.columns if c not in set(_COL_ORDER)]
    fei_df  = fei_df[ordered + extras]
    fei_df  = fei_df.sort_values("text_risk_index", ascending=False).reset_index(drop=True)

    fei_df.to_csv(FEI_CSV, index=False)

    # ── Output summary ───────────────────────────────────────────────────────
    print()
    print(sep)
    print(f"DONE — {len(fei_df)} FEIs → {FEI_CSV.name}  ({len(fei_df.columns)} columns)")
    print(sep)

    print("\n── Layer 1: per-FEI extraction quality ──")
    q_cols = ["fei", "n_obs_total", "n_obs_scored", "n_obs_failed",
              "n_obs_low_confidence", "mean_confidence", "n_blank_evidence_quotes", "n_files_483"]
    q_cols = [c for c in q_cols if c in fei_df.columns]
    print(fei_df[q_cols].to_string(index=False))

    print("\n── Layer 2: regex baseline (FEI means) ──")
    rx_cols = [c for c in fei_df.columns if c.endswith("_regex_share")]
    if rx_cols:
        print(fei_df[rx_cols].describe().loc[["mean", "min", "max"]].round(3).to_string())

    print("\n── Layer 3: LLM semantic (FEI means) ──")
    print(f"  severity_high_share     : mean={fei_df['severity_high_share'].mean():.3f}")
    print(f"  severity_moderate_share : mean={fei_df['severity_moderate_share'].mean():.3f}")
    print(f"  remediation_none_share  : mean={fei_df['remediation_none_share'].mean():.3f}")
    print(f"  repeat_llm_share        : mean={fei_df['repeat_llm_share'].mean():.3f}")
    print(f"  systemic_llm_share      : mean={fei_df['systemic_llm_share'].mean():.3f}")
    print(f"  dominant_violation_category (mode): "
          f"{fei_df['dominant_violation_category'].value_counts().index[0]}")

    print("\n── Layer 4: agreement / semantic lift (FEI means) ──")
    agree_cols   = [c for c in fei_df.columns if c.endswith("_regex_llm_agreement")]
    llmonly_cols = [c for c in fei_df.columns if c.endswith("_llm_only_share")]
    if agree_cols:
        print("  Agreement rates:")
        for c in agree_cols:
            print(f"    {c:<40}: {fei_df[c].mean():.3f}")
    if llmonly_cols:
        print("  Semantic lift (LLM-only) shares:")
        for c in llmonly_cols:
            print(f"    {c:<40}: {fei_df[c].mean():.3f}")

    print("\n── Layer 5: composite index distributions ──")
    idx_cols = [
        "text_risk_index",
        "sterility_contamination_risk_index",
        "investigation_remediation_weakness_index",
        "quality_culture_index",
    ]
    for col in idx_cols:
        if col in fei_df.columns:
            s = fei_df[col]
            print(f"  {col}:")
            print(f"    mean={s.mean():.2f}  min={s.min():.2f}  max={s.max():.2f}")

    print("\n── Top 10 FEIs by text_risk_index ──")
    show = ["fei", "n_obs_scored", "text_risk_index", "severity_high_share",
            "dominant_violation_category", "dominant_root_cause"]
    show = [c for c in show if c in fei_df.columns]
    print(fei_df[show].head(10).to_string(index=False))

    print(f"\nOutput: {FEI_CSV}")
    print()
    print("Merge these features in the prediction model with:")
    print("  CFR domain features → Data/14 - FDA - Inspection/processed/facility_feature_matrix.csv")
    print("  WL features         → Data/21 - FDA - Warning Letter/processed/warning_letter_fei_features.csv")


if __name__ == "__main__":
    main()
