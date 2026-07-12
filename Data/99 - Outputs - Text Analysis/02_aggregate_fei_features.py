"""
02_aggregate_fei_features.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Aggregates observation-level signals from Step 1 into time-stamped FEI-level
  snapshots suitable for shortage prediction.

  Each output row is a cumulative feature profile of one facility (FEI) as of
  a specific 483 inspection date.  Features are computed from all observations
  of that FEI on or before snapshot_date, so the row captures everything the
  model would know about the facility up to that point in time.

  Downstream shortage model joins on fei and takes the most recent snapshot
  before the month being predicted.

PIPELINE POSITION (current — Redica/Anthropic mode)
  Step 0  (00_load_redica_obs.py)
      Reads : Data/07 - Redica/Raw/...
      Writes: redica_483_observations.csv

  Step 1  (01_extract_observation_signals.py --source redica --provider anthropic)
      Reads : redica_483_observations.csv
      Writes: redica_483_obs_llm_signals_anthropic_v2.csv
              one row per observation; LLM semantic fields

  Step 4  (04_build_combined_obs_universe.py)
      Reads : redica_483_observations.csv
              redica_483_obs_llm_signals_anthropic_v2.csv
      Writes: 483_combined_obs_universe.csv
              one row per observation, all sources combined

  Step 2  (this script --source redica)
      Reads : 483_combined_obs_universe.csv
      Writes: 483_fei_text_features_timeseries_redica.csv
              one row per (fei, snapshot_date)

OUTPUT SCHEMA
  fei             : facility establishment identifier
  snapshot_date   : 483 inspection date of the event that updated this snapshot
  year_month      : YYYY-MM (for joining against monthly shortage panels)
  n_obs_*         : cumulative observation counts up to snapshot_date
  *_regex_share   : cumulative regex flag rates (all rows, regardless of LLM)
  severity_*      : cumulative LLM 4-tier severity distribution (Critical/Major/
                    Moderate/Minor; scored rows only)
  scope_*         : cumulative LLM scope distribution (SingleBatch/
                    MultipleProducts/FacilityWide/Unclear)
  *_root_cause_*  : cumulative root cause distribution
  remediation_*   : cumulative remediation signal distribution
  *_llm_share     : cumulative LLM binary flag rates (scored rows only)
  *_llm_only_share: share of obs where LLM flagged but regex did not (semantic lift)
  repeat_cross_insp_* : repeat detection ACROSS inspections of the same FEI:
                    the standardized citation sentence (text before
                    "Specifically,") matches an earlier inspection's observation
                    (template token overlap >= 0.80). Complements
                    repeat_flag_llm, which only sees one observation at a time

NOTE ON COMPOSITE INDICES
  TRI/SCRI/IRWI/QCI are removed.  Raw feature shares are passed directly to
  the model, which avoids manual weighting and lets the model learn weights
  from data.

OUTPUTS
  483_fei_text_features_timeseries.csv  (primary, time-aware)
    One row per (FEI, snapshot_date).  Cumulative features up to that date.
    Use for shortage prediction: join on fei and take the most recent snapshot
    before each prediction month.

  483_fei_context_features.csv  (static, legacy)
    One row per FEI.  All observations pooled, no time dimension.
    Kept for MQRI pipeline and any analysis that does not require time-awareness.
    Do NOT use for shortage prediction -- it leaks future observations.

DENOMINATOR RULES
  Regex shares     : all cumulative rows
  LLM shares       : scored (ok/partial) cumulative rows only
  Agreement/lift   : scored rows only (both regex and LLM present)
  Failed rows      : excluded from LLM features; never become false/zero labels
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent

# PDF pipeline (--source pdf)
SIGNALS_CSV = HERE / "483_observation_context_signals.csv"
OUT_CSV     = HERE / "483_fei_text_features_timeseries.csv"
OUT_STATIC  = HERE / "483_fei_context_features.csv"

# Redica pipeline (--source redica) — reads directly from step 01 output, no step 04 needed
REDICA_SIGNALS_CSV = HERE / "redica_483_obs_llm_signals_anthropic_v2.csv"
OUT_REDICA_CSV     = HERE / "483_fei_text_features_timeseries_redica.csv"

# Combined pipeline (--source combined) — requires step 04 first
COMBINED_CSV     = HERE / "483_combined_obs_universe.csv"
OUT_COMBINED_CSV = HERE / "483_fei_text_features_timeseries_combined.csv"

LOW_CONFIDENCE_THRESHOLD = 0.70


# ── Generic helpers ────────────────────────────────────────────────────────

def _share(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(bool).mean())


def _value_share(series: pd.Series, value) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    return float((valid == value).sum()) / len(valid)


def _mode_or_default(series: pd.Series, default: str = "Other") -> str:
    counts = series.dropna().value_counts()
    return str(counts.index[0]) if len(counts) > 0 else default


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")


# ── Cross-inspection repeat detection ──────────────────────────────────────
# An observation is a cross-inspection repeat when its standardized TEMPLATE
# FIRST SENTENCE (the citation language before "Specifically,") matches an
# observation of the SAME FEI at a STRICTLY EARLIER inspection date. The
# template sentence is the CFR-derived citation text, so a match means the
# same deficiency was cited again — the regulatory notion of a repeat finding.
# Full-text similarity does not work here: shared GMP vocabulary and OCR
# header noise give unrelated observations a ~0.45 baseline overlap, while
# true repeats differ in their specific examples.
# Similarity: overlap coefficient |A∩B| / min(|A|,|B|) on template tokens.

CROSS_REPEAT_OVERLAP  = 0.80
MIN_TEMPLATE_TOKENS   = 6


def _template_tokens(text: str) -> frozenset:
    """Tokens of the standardized citation sentence (text before 'Specifically')."""
    t = re.sub(r"^\s*OBSERVATION\s+\d+\s*", "", str(text), flags=re.IGNORECASE)
    head = re.split(r"[Ss]pecifically", t, maxsplit=1)[0]
    return frozenset(re.findall(r"[a-z]{3,}", head.lower()))


def _flag_cross_inspection_repeats(df: pd.DataFrame) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    text_col = "obs_text_clean" if "obs_text_clean" in df.columns else "obs_text" if "obs_text" in df.columns else None
    if text_col is None:
        return flags
    for _, grp in df.groupby("fei"):
        if grp["insp_date"].nunique() < 2:
            continue
        grp = grp.sort_values("insp_date")
        toks = {idx: _template_tokens(t) for idx, t in grp[text_col].items()}
        idxs = list(grp.index)
        for i, idx in enumerate(idxs):
            a = toks[idx]
            if len(a) < MIN_TEMPLATE_TOKENS:
                continue
            cur_date = grp.loc[idx, "insp_date"]
            for jdx in idxs[:i]:
                if grp.loc[jdx, "insp_date"] >= cur_date:
                    continue
                b = toks[jdx]
                if len(b) < MIN_TEMPLATE_TOKENS:
                    continue
                if len(a & b) / min(len(a), len(b)) >= CROSS_REPEAT_OVERLAP:
                    flags.loc[idx] = True
                    break
    return flags


# ── Layer 1: extraction quality ────────────────────────────────────────────

def _layer1_quality(grp: pd.DataFrame, ns: int) -> dict:
    n      = len(grp)
    n_fail = n - ns
    scored = grp[grp["extraction_status"].isin(["ok", "partial"])]

    n_low_conf = int(
        (scored["confidence"] < LOW_CONFIDENCE_THRESHOLD).sum()
    ) if ns > 0 and "confidence" in scored.columns else 0

    n_blank_eq = int(_is_blank(grp["evidence_quote"]).sum()) if "evidence_quote" in grp.columns else 0

    mean_conf = round(float(scored["confidence"].mean()), 4) \
        if ns > 0 and "confidence" in scored.columns else float("nan")

    # Use filename if available; fall back to document_id for combined obs universe
    if "filename" in grp.columns:
        n_files = int(grp["filename"].nunique())
    elif "document_id" in grp.columns:
        n_files = int(grp["document_id"].dropna().nunique())
    else:
        n_files = int(grp["insp_date"].nunique())

    return {
        "n_obs_total":             n,
        "n_obs_scored":            ns,
        "n_obs_failed":            n_fail,
        "n_obs_low_confidence":    n_low_conf,
        "mean_confidence":         mean_conf,
        "n_blank_evidence_quotes": n_blank_eq,
        "n_files_483":             n_files,
    }


# ── Layer 2: Regex baseline ────────────────────────────────────────────────

def _layer2_regex(grp: pd.DataFrame) -> dict:
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
        ("wl_ref_regex_share",             "has_wl_ref_regex"),
        ("oos_oot_regex_share",            "has_oos_oot_regex"),
    ]
    out = {}
    for feat, col in pairs:
        out[feat] = round(_share(_to_bool(grp[col])), 4) if col in grp.columns else float("nan")
    return out


# ── Layer 3: LLM semantic ──────────────────────────────────────────────────

def _layer3_llm(scored: pd.DataFrame, ns: int) -> dict:
    sev = scored["severity_tier"]      if ns > 0 else pd.Series(dtype=str)
    rem = scored["remediation_signal"] if ns > 0 else pd.Series(dtype=str)
    rc  = scored["root_cause_type"]    if ns > 0 else pd.Series(dtype=str)
    vc  = scored["violation_category"] if ns > 0 else pd.Series(dtype=str)
    sc  = scored["scope"] if ns > 0 and "scope" in scored.columns else pd.Series(dtype=str)

    # remediation_none: NaN or literal "None" both mean no remediation mentioned
    rem_none = round(
        float((rem.isna() | (rem.astype(str).str.strip().isin(["None", "nan", ""]))).sum())
        / max(ns, 1), 4
    ) if ns > 0 else float("nan")

    llm_flags = [
        ("repeat_llm_share",         "repeat_flag_llm"),
        ("patient_risk_llm_share",   "patient_risk_flag_llm"),
        ("data_integrity_llm_share", "data_integrity_flag_llm"),
        ("contamination_llm_share",  "contamination_flag_llm"),
        ("investigation_llm_share",  "investigation_flag_llm"),
    ]
    flag_shares = {}
    for feat, col in llm_flags:
        flag_shares[feat] = round(_share(_to_bool(scored[col])), 4) \
            if col in scored.columns and ns > 0 else float("nan")

    # Per-category violation share (not just dominant)
    categories = ["LabControls", "ProductionControls", "BuildingsEquipment",
                  "OrgPersonnel", "PackagingLabeling", "RecordsReports", "QualitySystem", "Other"]
    vc_shares = {}
    for cat in categories:
        key = f"vc_{cat.lower()}_share"
        vc_shares[key] = round(_value_share(vc, cat), 4)

    out = {
        "severity_critical_share":     round(_value_share(sev, "Critical"), 4),
        "severity_major_share":        round(_value_share(sev, "Major"),    4),
        "severity_moderate_share":     round(_value_share(sev, "Moderate"), 4),
        "severity_minor_share":        round(_value_share(sev, "Minor"),    4),
        # 4-tier analog of the old severity_high_share; single column for models
        "severity_critmajor_share":    round(_value_share(sev, "Critical")
                                             + _value_share(sev, "Major"),  4),
        "scope_singlebatch_share":     round(_value_share(sc, "SingleBatch"),      4),
        "scope_multipleproducts_share": round(_value_share(sc, "MultipleProducts"), 4),
        "scope_facilitywide_share":    round(_value_share(sc, "FacilityWide"),     4),
        "scope_unclear_share":         round(_value_share(sc, "Unclear"),          4),
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
    out.update(vc_shares)
    return out


# ── Layer 4: Agreement and semantic lift ───────────────────────────────────

def _layer4_agreement(scored: pd.DataFrame, ns: int) -> dict:
    paired = [
        ("repeat",         "has_repeat_regex",        "repeat_flag_llm"),
        ("contamination",  "has_contamination_regex",  "contamination_flag_llm"),
        ("data_integrity", "has_data_integrity_regex", "data_integrity_flag_llm"),
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


# ── Snapshot aggregation ───────────────────────────────────────────────────

def _aggregate_snapshot(subset: pd.DataFrame) -> dict:
    """Aggregate all observations in subset (cumulative up to snapshot_date)."""
    scored = subset[subset["extraction_status"].isin(["ok", "partial"])]
    ns     = len(scored)

    l1 = _layer1_quality(subset, ns)
    l2 = _layer2_regex(subset)
    l3 = _layer3_llm(scored, ns)
    l4 = _layer4_agreement(scored, ns)

    flat = {}
    for layer in (l1, l2, l3, l4):
        flat.update(layer)

    if "repeat_cross_insp" in subset.columns:
        flat["n_repeat_cross_insp"]     = int(subset["repeat_cross_insp"].sum())
        flat["repeat_cross_insp_share"] = round(_share(subset["repeat_cross_insp"]), 4)
    return flat


# ── Static (non-time) aggregation — one row per FEI ───────────────────────

def _build_static(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all observations per FEI with no time dimension.
    Kept for MQRI pipeline and cross-sectional analyses.
    Do NOT use for shortage prediction (leaks future information).
    """
    rows = []
    for fei_val, grp in df.groupby("fei"):
        agg        = _aggregate_snapshot(grp.copy())
        agg["fei"] = int(fei_val)
        rows.append(agg)
    static_df = pd.DataFrame(rows)

    ordered = ["fei"] + [c for c in _COL_ORDER if c in static_df.columns and c not in ("fei", "snapshot_date", "year_month")]
    extras  = [c for c in static_df.columns if c not in set(ordered)]
    static_df = static_df[ordered + extras].reset_index(drop=True)
    return static_df


# ── Column ordering ────────────────────────────────────────────────────────

_COL_ORDER = [
    # keys
    "fei", "snapshot_date", "year_month",
    # earliest date in current snapshot and total files seen
    "n_obs_total", "n_obs_scored", "n_obs_failed", "n_obs_low_confidence",
    "mean_confidence", "n_blank_evidence_quotes", "n_files_483",
    # Layer 2 -- regex
    "repeat_regex_share", "systemic_regex_share", "documentation_regex_share",
    "investigation_regex_share", "contamination_regex_share",
    "data_integrity_regex_share", "patient_risk_regex_share",
    "quality_unit_regex_share", "laboratory_regex_share",
    "equipment_facility_regex_share", "process_control_regex_share",
    "wl_ref_regex_share", "oos_oot_regex_share",
    # Layer 3 -- LLM categorical
    "severity_critical_share", "severity_major_share",
    "severity_moderate_share", "severity_minor_share",
    "severity_critmajor_share",
    "scope_singlebatch_share", "scope_multipleproducts_share",
    "scope_facilitywide_share", "scope_unclear_share",
    "dominant_violation_category", "dominant_root_cause",
    "capital_root_cause_share", "cultural_root_cause_share",
    "mixed_root_cause_share", "unclear_root_cause_share",
    "remediation_strong_share", "remediation_partial_share",
    "remediation_weak_share", "remediation_none_share",
    # Layer 3 -- LLM binary flags
    "repeat_llm_share", "patient_risk_llm_share",
    "data_integrity_llm_share", "contamination_llm_share",
    "investigation_llm_share",
    # Layer 3 -- per-category violation shares
    "vc_labcontrols_share", "vc_productioncontrols_share",
    "vc_buildingsequipment_share", "vc_orgpersonnel_share",
    "vc_packaginglabeling_share", "vc_recordsreports_share",
    "vc_qualitysystem_share", "vc_other_share",
    # Layer 4 -- agreement / semantic lift
    "repeat_regex_llm_agreement",         "repeat_llm_only_share",
    "contamination_regex_llm_agreement",  "contamination_llm_only_share",
    "data_integrity_regex_llm_agreement", "data_integrity_llm_only_share",
    "investigation_regex_llm_agreement",  "investigation_llm_only_share",
    "patient_risk_regex_llm_agreement",   "patient_risk_llm_only_share",
    # Cross-inspection repeat (text similarity vs earlier inspections)
    "n_repeat_cross_insp", "repeat_cross_insp_share",
]


# ── Main ───────────────────────────────────────────────────────────────────

def _run_aggregation(df: pd.DataFrame, out_csv: Path, out_static: Path | None, label: str) -> pd.DataFrame:
    """Core aggregation logic shared by both PDF-only and combined modes."""
    sep = "=" * 70
    print(sep)
    print(f"02_aggregate_fei_features.py — {label}")
    print(sep)

    print(f"\n-- Input QA --")
    print(f"  Rows loaded      : {len(df):,}")
    print(f"  Unique FEIs      : {df['fei'].nunique()}")
    if "filename" in df.columns:
        print(f"  Unique filenames : {df['filename'].nunique()}")
    status_counts = df["extraction_status"].value_counts(dropna=False)
    for status, cnt in status_counts.items():
        print(f"  status={status!s:<20}: {cnt}")

    df["fei"]        = pd.to_numeric(df["fei"], errors="coerce").astype("Int64")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["insp_date"]  = pd.to_datetime(df["insp_date"], errors="coerce")
    df = df.dropna(subset=["fei", "insp_date"])
    print(f"  Rows with valid fei + date: {len(df):,}")

    df["repeat_cross_insp"] = _flag_cross_inspection_repeats(df)
    n_cross = int(df["repeat_cross_insp"].sum())
    print(f"  Cross-inspection repeats  : {n_cross} observations "
          f"(template overlap >= {CROSS_REPEAT_OVERLAP} vs an earlier inspection of same FEI)")

    snapshots_index = (
        df.groupby(["fei", "insp_date"])
        .size()
        .reset_index()[["fei", "insp_date"]]
        .sort_values(["fei", "insp_date"])
    )
    print(f"\n  Snapshots to build: {len(snapshots_index)}")
    print(f"  (one per unique FEI x inspection date)\n")

    rows = []
    for _, snap in snapshots_index.iterrows():
        fei  = snap["fei"]
        date = snap["insp_date"]
        subset = df[(df["fei"] == fei) & (df["insp_date"] <= date)].copy()
        agg = _aggregate_snapshot(subset)
        agg["fei"]           = int(fei)
        agg["snapshot_date"] = date.date().isoformat()
        agg["year_month"]    = date.strftime("%Y-%m")
        rows.append(agg)

    out_df = pd.DataFrame(rows)
    ordered = [c for c in _COL_ORDER if c in out_df.columns]
    extras  = [c for c in out_df.columns if c not in set(_COL_ORDER)]
    out_df  = out_df[ordered + extras]
    out_df  = out_df.sort_values(["fei", "snapshot_date"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    if out_static is not None:
        static_df = _build_static(df)
        static_df.to_csv(out_static, index=False)
        print(f"Static output: {out_static.name}  ({len(static_df)} FEIs, {len(static_df.columns)} columns)")

    print(sep)
    print(f"DONE -- {len(out_df)} snapshots across {out_df['fei'].nunique()} FEIs")
    print(f"Output: {out_csv.name}  ({len(out_df.columns)} columns)")
    print(sep)

    print("\n-- Snapshots per FEI --")
    snap_counts = out_df.groupby("fei").size().describe()
    print(f"  mean={snap_counts['mean']:.1f}  min={snap_counts['min']:.0f}  max={snap_counts['max']:.0f}")
    print("\n-- Date range --")
    print(f"  Earliest: {out_df['snapshot_date'].min()}")
    print(f"  Latest  : {out_df['snapshot_date'].max()}")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FEI text features into time-series snapshots.")
    parser.add_argument(
        "--source", choices=["pdf", "combined", "redica"], default="pdf",
        help=(
            "'pdf'      — reads 483_observation_context_signals.csv (38 FEIs, PDF pipeline).\n"
            "'redica'   — reads redica_483_obs_llm_signals_anthropic_v2.csv directly "
            "(98 FEIs, current pipeline; use this for modeling). No step 04 needed.\n"
            "'combined' — reads 483_combined_obs_universe.csv (all sources). "
            "Requires 04_build_combined_obs_universe.py to run first."
        ),
    )
    args = parser.parse_args()

    if args.source == "combined":
        if not COMBINED_CSV.exists():
            sys.exit(f"\n[ERROR] Combined obs universe not found:\n  {COMBINED_CSV}\n"
                     "Run 04_build_combined_obs_universe.py first.\n")
        df = pd.read_csv(COMBINED_CSV)
        if "extraction_status" not in df.columns:
            df["extraction_status"] = "ok"
        _run_aggregation(df, OUT_COMBINED_CSV, out_static=None,
                         label="Combined (PDF + Redica) — 99 FEIs")
        return

    if args.source == "redica":
        if not REDICA_SIGNALS_CSV.exists():
            sys.exit(f"\n[ERROR] Redica LLM signals not found:\n  {REDICA_SIGNALS_CSV}\n"
                     "Run 01_extract_observation_signals.py --source redica --provider anthropic first.\n")
        df = pd.read_csv(REDICA_SIGNALS_CSV)
        if "extraction_status" not in df.columns:
            df["extraction_status"] = "ok"
        _run_aggregation(df, OUT_REDICA_CSV, out_static=None,
                         label="Redica pipeline (claude-haiku, 98 FEIs)")
        return

    # Default: PDF-only pipeline
    if not SIGNALS_CSV.exists():
        sys.exit(
            f"\n[ERROR] Observation signals file not found:\n  {SIGNALS_CSV}\n"
            "Run 01_extract_observation_signals.py first.\n"
        )

    df = pd.read_csv(SIGNALS_CSV)
    out_df = _run_aggregation(df, OUT_CSV, OUT_STATIC, label="PDF pipeline — 38 FEIs")

    print("\n-- LLM lift at latest snapshot per FEI (mean across facilities) --")
    latest = out_df.sort_values("snapshot_date").groupby("fei").last().reset_index()
    for col in ["patient_risk_llm_only_share", "contamination_llm_only_share",
                "data_integrity_llm_only_share", "investigation_llm_only_share",
                "repeat_cross_insp_share"]:
        if col in latest.columns:
            print(f"  {col:<40}: {latest[col].mean():.3f}")

    print("\n-- Downstream join --")
    print("  Join: shortage_panel[fei, year_month]")
    print("        -> take most recent snapshot where snapshot_date <= prediction month")
    print("        -> use *_llm_share, *_regex_share, severity_*, remediation_* as features")
    print()


if __name__ == "__main__":
    main()
