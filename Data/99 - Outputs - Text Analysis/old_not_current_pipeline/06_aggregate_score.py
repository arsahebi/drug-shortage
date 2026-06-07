"""
06_aggregate_score.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Rolls observation-level LLM signals (from 05) up to per-FEI aggregates and
  computes a transparent, interpretable composite Text Risk Index (TRI).

  This is step 3 of the LLM pipeline (04 → 05 → 06 → 07).

WHEN TO RUN
  Run after 05_extract_signals_langgraph.py has produced fei_observation_signals.csv.
  Fast (<1 min). No API key needed.
  Re-run whenever 05 adds new observations.

REQUIRED FOR COMBINED DATASET?  NO — optional LLM enrichment.
  Run before 03_build_interactive_dashboard.py to get the Risk Signals tab.

INPUTS (in this folder)
  fei_observation_signals.csv  ← produced by 05

OUTPUTS (in this folder)
  fei_risk_signals.csv  ← one row per FEI, aggregate columns + TRI

TRI FORMULA (weights must sum to 1.00; bounded [0, 100])
  TRI = (
      0.35 × severity_high_share
    + 0.20 × severity_mod_share
    + 0.20 × (1 − remediation_strong_share)
    + 0.15 × repeat_flag_share
    + 0.10 × systemic_flag_share
  ) × 100

WEIGHT RATIONALE (for paper)
  severity_high_share (0.35)        — direct patient-harm potential; top priority
  severity_mod_share (0.20)         — process deviations that accumulate into supply risk
  (1 − remediation_strong) (0.20)   — absence of corrective action predicts recurrence
  repeat_flag_share (0.15)          — repeat violations predict Warning Letters (empirical)
  systemic_flag_share (0.10)        — systemic failure; lower weight (overlaps severity)
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
OUT          = Path(__file__).parent
SIGNALS_CSV  = OUT / "fei_observation_signals.csv"
RISK_CSV     = OUT / "fei_risk_signals.csv"

# ── TRI weights (must sum to 1.0) ──────────────────────────────────────────
W_SEV_HIGH   = 0.35
W_SEV_MOD    = 0.20
W_NO_STRONG_REM = 0.20  # (1 - remediation_strong_share)
W_REPEAT     = 0.15
W_SYSTEMIC   = 0.10

assert abs(W_SEV_HIGH + W_SEV_MOD + W_NO_STRONG_REM + W_REPEAT + W_SYSTEMIC - 1.0) < 1e-9, \
    "TRI weights must sum to 1.0"


def _safe_share(series_bool: pd.Series) -> float:
    """Return share of True values; returns 0.0 for empty series."""
    if len(series_bool) == 0:
        return 0.0
    return float(series_bool.astype(bool).mean())


def aggregate_fei(grp: pd.DataFrame) -> dict:
    n = len(grp)

    # ── Severity distribution ──────────────────────────────────────────────
    sev_counts   = grp["severity_tier"].value_counts()
    sev_high     = sev_counts.get("High",     0) / n
    sev_mod      = sev_counts.get("Moderate", 0) / n
    sev_low      = sev_counts.get("Low",      0) / n

    # ── Root cause distribution ────────────────────────────────────────────
    rc_counts    = grp["root_cause_type"].value_counts()
    capital_sh   = rc_counts.get("Capital", 0) / n
    cultural_sh  = rc_counts.get("Cultural",0) / n
    mixed_sh     = rc_counts.get("Mixed",   0) / n
    unclear_sh   = rc_counts.get("Unclear", 0) / n
    dominant_rc  = grp["root_cause_type"].value_counts().index[0] \
                   if n > 0 else "Unclear"

    # ── Remediation distribution ───────────────────────────────────────────
    rem_counts   = grp["remediation_signal"].value_counts()
    rem_strong   = rem_counts.get("Strong",  0) / n
    rem_partial  = rem_counts.get("Partial", 0) / n
    rem_weak     = rem_counts.get("Weak",    0) / n
    rem_none     = rem_counts.get("None",    0) / n

    # ── Binary signal shares ───────────────────────────────────────────────
    repeat_sh    = _safe_share(grp["repeat_flag"])
    systemic_sh  = _safe_share(grp["systemic_flag"])
    patient_sh   = _safe_share(grp["patient_risk_flag"])

    # ── Source split ───────────────────────────────────────────────────────
    src_counts   = grp["source_type"].value_counts() if "source_type" in grp.columns else {}
    n_483_obs    = int(src_counts.get("483", 0)) if hasattr(src_counts, "get") else 0
    n_wl_obs     = int(src_counts.get("WL",  0)) if hasattr(src_counts, "get") else 0

    # ── Mean confidence ────────────────────────────────────────────────────
    mean_conf    = float(grp["confidence"].mean()) if "confidence" in grp.columns else 0.0

    # ── Violation category plurality ───────────────────────────────────────
    dominant_cat = grp["violation_category"].value_counts().index[0] \
                   if "violation_category" in grp.columns and n > 0 else "Other"

    # ── Text Risk Index ────────────────────────────────────────────────────
    tri = (
        W_SEV_HIGH      * sev_high
      + W_SEV_MOD       * sev_mod
      + W_NO_STRONG_REM * (1.0 - rem_strong)
      + W_REPEAT        * repeat_sh
      + W_SYSTEMIC      * systemic_sh
    ) * 100

    return {
        "n_obs_scored":           n,
        "n_483_obs":              n_483_obs,
        "n_wl_obs":               n_wl_obs,
        # Severity
        "severity_high_share":    round(sev_high,  4),
        "severity_mod_share":     round(sev_mod,   4),
        "severity_low_share":     round(sev_low,   4),
        # Root cause
        "dominant_root_cause":    dominant_rc,
        "capital_share":          round(capital_sh,  4),
        "cultural_share":         round(cultural_sh, 4),
        "mixed_share":            round(mixed_sh,    4),
        "unclear_share":          round(unclear_sh,  4),
        # Remediation
        "remediation_strong_share": round(rem_strong,  4),
        "remediation_partial_share":round(rem_partial, 4),
        "remediation_weak_share":   round(rem_weak,    4),
        "remediation_none_share":   round(rem_none,    4),
        # Binary flags
        "repeat_flag_share":      round(repeat_sh,   4),
        "systemic_flag_share":    round(systemic_sh, 4),
        "patient_risk_share":     round(patient_sh,  4),
        # Dominant violation category
        "dominant_violation_category": dominant_cat,
        # Quality
        "mean_confidence":        round(mean_conf, 4),
        # Score
        "text_risk_index":        round(tri, 2),
    }


def main():
    print("=" * 65)
    print("06_aggregate_score.py — Aggregate signals → Text Risk Index")
    print("=" * 65)

    if not SIGNALS_CSV.exists():
        sys.exit(
            f"Observation signals file not found: {SIGNALS_CSV}\n"
            "Run 05_extract_signals_langgraph.py first."
        )

    signals = pd.read_csv(SIGNALS_CSV)
    print(f"Observations loaded: {len(signals)}")

    # Coerce types
    for bool_col in ["repeat_flag", "systemic_flag", "patient_risk_flag"]:
        if bool_col in signals.columns:
            signals[bool_col] = signals[bool_col].astype(str).str.lower() \
                                 .map({"true": True, "false": False, "1": True, "0": False}) \
                                 .fillna(False)

    signals["fei"]        = pd.to_numeric(signals["fei"], errors="coerce").astype("Int64")
    signals["confidence"] = pd.to_numeric(signals["confidence"], errors="coerce").fillna(0.0)

    # Drop rows with invalid FEI
    signals = signals.dropna(subset=["fei"])
    print(f"Observations with valid FEI: {len(signals)}")
    print(f"FEIs represented: {signals['fei'].nunique()}")

    # Aggregate per FEI
    rows = []
    for fei_val, grp in signals.groupby("fei"):
        agg = aggregate_fei(grp.copy())
        agg["fei"] = int(fei_val)
        rows.append(agg)

    risk_df = pd.DataFrame(rows)

    # Column order: fei first, then sorted groups
    col_order = [
        "fei",
        "n_obs_scored", "n_483_obs", "n_wl_obs",
        "severity_high_share", "severity_mod_share", "severity_low_share",
        "dominant_root_cause",
        "capital_share", "cultural_share", "mixed_share", "unclear_share",
        "remediation_strong_share", "remediation_partial_share",
        "remediation_weak_share", "remediation_none_share",
        "repeat_flag_share", "systemic_flag_share", "patient_risk_share",
        "dominant_violation_category",
        "mean_confidence",
        "text_risk_index",
    ]
    risk_df = risk_df[[c for c in col_order if c in risk_df.columns]]
    risk_df = risk_df.sort_values("text_risk_index", ascending=False).reset_index(drop=True)

    risk_df.to_csv(RISK_CSV, index=False)

    print(f"\n{'='*65}")
    print(f"DONE — {len(risk_df)} FEIs aggregated")
    print(f"{'='*65}")
    print(f"\nTRI distribution (text_risk_index):")
    print(risk_df["text_risk_index"].describe().round(2).to_string())
    print(f"\nTop 10 highest-risk FEIs:")
    print(risk_df[["fei", "text_risk_index", "dominant_root_cause",
                   "severity_high_share", "n_obs_scored"]]
          .head(10).to_string(index=False))
    print(f"\nTRI weight breakdown (for paper):")
    print(f"  severity_high_share          × {W_SEV_HIGH:.2f}  = {W_SEV_HIGH*100:.0f}%")
    print(f"  severity_mod_share           × {W_SEV_MOD:.2f}  = {W_SEV_MOD*100:.0f}%")
    print(f"  (1 − remediation_strong_sh)  × {W_NO_STRONG_REM:.2f}  = {W_NO_STRONG_REM*100:.0f}%")
    print(f"  repeat_flag_share            × {W_REPEAT:.2f}  = {W_REPEAT*100:.0f}%")
    print(f"  systemic_flag_share          × {W_SYSTEMIC:.2f}  = {W_SYSTEMIC*100:.0f}%")
    print(f"\nOutput: {RISK_CSV}")


if __name__ == "__main__":
    main()
