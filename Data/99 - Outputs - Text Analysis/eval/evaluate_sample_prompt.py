"""
evaluate_sample_prompt.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Validates the revised extraction prompt (4-tier severity, scope field,
tightened patient_risk) on a stratified sample before a full re-run.

Checks:
  (a) Severity distribution is graded — not top-heavy like the old
      3-tier prompt (70% High).
  (b) Critical+Major share separates OAI from VAI inspections better
      than the old severity_high baseline (joined via Redica).
  (c) patient_risk fire rate dropped from ~91% to an informative range.
  (d) Scope distribution is spread, not 94% FacilityWide.

Usage:
  python eval/evaluate_sample_prompt.py                # default sample50
  python eval/evaluate_sample_prompt.py --sample-file 483_observation_context_signals_sample50.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.parent          # .../99 - Outputs - Text Analysis/
DATA = HERE.parent                            # .../Data/

OLD_SIGNALS = HERE / "483_observation_context_signals.csv"
REDICA_CSV  = DATA / "07 - Redica" / "processed" / "redica_all_drugs_combined.csv"

# Max days between 483 inspection date and Redica event date to count as the
# same inspection. 483s are issued at inspection close; Redica event dates are
# inspection dates, so a tight window is correct.
REDICA_MATCH_DAYS = 30


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1"])


def _attach_redica_classification(df: pd.DataFrame, redica: pd.DataFrame) -> pd.DataFrame:
    """Join each observation to the Redica inspection classification by
    FEI + nearest event date within REDICA_MATCH_DAYS."""
    red = redica.dropna(subset=["Classification", "Event Date"]).copy()
    red["event_date"] = pd.to_datetime(red["Event Date"], errors="coerce")
    red = red.dropna(subset=["event_date"])

    df = df.copy()
    df["insp_date_dt"] = pd.to_datetime(df["insp_date"], errors="coerce")

    classifications = []
    for _, row in df.iterrows():
        if pd.isna(row["insp_date_dt"]):
            classifications.append(None)
            continue
        cand = red[red["FEI"] == row["fei"]]
        if cand.empty:
            classifications.append(None)
            continue
        deltas = (cand["event_date"] - row["insp_date_dt"]).abs()
        if deltas.min() <= pd.Timedelta(days=REDICA_MATCH_DAYS):
            classifications.append(cand.loc[deltas.idxmin(), "Classification"])
        else:
            classifications.append(None)
    df["redica_classification"] = classifications
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file", default="483_observation_context_signals_sample50.csv")
    args = parser.parse_args()

    sample_path = HERE / args.sample_file
    if not sample_path.exists():
        sys.exit(f"[ERROR] Sample file not found: {sample_path}\n"
                 "Run: python 01_extract_observation_signals.py --sample 50")

    new = pd.read_csv(sample_path)
    new = new[new["extraction_status"].isin(["ok", "partial"])]
    print("=" * 70)
    print(f"Revised-prompt sample: {len(new)} scored observations, "
          f"{new['fei'].nunique()} FEIs")
    print("=" * 70)

    # ── (a) Severity distribution ───────────────────────────────────────────
    print("\n(a) SEVERITY DISTRIBUTION — new 4-tier prompt")
    sev_new = new["severity_tier"].value_counts(normalize=True).mul(100).round(1)
    for tier in ["Critical", "Major", "Moderate", "Minor"]:
        print(f"    {tier:<10}: {sev_new.get(tier, 0.0):5.1f}%")
    top_share = sev_new.get("Critical", 0) + sev_new.get("Major", 0)
    print(f"    Critical+Major combined: {top_share:.1f}%")

    if OLD_SIGNALS.exists():
        old = pd.read_csv(OLD_SIGNALS)
        old = old[old["extraction_status"].isin(["ok", "partial"])]
        # Restrict old to the same (fei, filename, obs_num) keys for a fair comparison
        keys = set(zip(new["fei"], new["filename"], new["obs_num"]))
        old_matched = old[old.apply(
            lambda r: (r["fei"], r["filename"], r["obs_num"]) in keys, axis=1)]
        sev_old = old_matched["severity_tier"].value_counts(normalize=True).mul(100).round(1)
        print(f"\n    Old 3-tier prompt on the SAME {len(old_matched)} observations:")
        for tier in ["High", "Moderate", "Low"]:
            print(f"    {tier:<10}: {sev_old.get(tier, 0.0):5.1f}%")

    graded = (sev_new.get("Critical", 0) < 35) and (sev_new.max() < 65)
    print(f"\n    VERDICT (a): {'PASS — graded' if graded else 'FAIL — still top-heavy'} "
          f"(Critical < 35% and no tier > 65%)")

    # ── (b) OAI/VAI separation via Redica ──────────────────────────────────
    print("\n(b) OAI vs VAI SEPARATION — Critical+Major share by inspection outcome")
    if not REDICA_CSV.exists():
        print("    [SKIP] Redica file not found")
    else:
        redica = pd.read_csv(REDICA_CSV)
        new_r = _attach_redica_classification(new, redica)
        matched = new_r.dropna(subset=["redica_classification"])
        print(f"    Observations matched to a Redica inspection: {len(matched)} / {len(new_r)}")
        if len(matched) >= 10:
            matched = matched.copy()
            matched["crit_major"] = matched["severity_tier"].isin(["Critical", "Major"])
            by_class = matched.groupby("redica_classification")["crit_major"].agg(["mean", "count"])
            print(by_class.round(3).to_string())
            if {"OAI", "VAI"}.issubset(by_class.index):
                sep = by_class.loc["OAI", "mean"] - by_class.loc["VAI", "mean"]
                print(f"    OAI - VAI separation: {sep:+.3f}")
                print(f"    VERDICT (b): {'PASS' if sep > 0.05 else 'WEAK — review per-tier counts'}")
            else:
                print("    VERDICT (b): insufficient OAI or VAI matches in sample — "
                      "consider --sample 80 or evaluate after full run")
        else:
            print("    VERDICT (b): too few Redica matches — evaluate after full run")

    # ── (c) patient_risk fire rate ──────────────────────────────────────────
    print("\n(c) PATIENT RISK FIRE RATE")
    pr_rate = _to_bool(new["patient_risk_flag_llm"]).mean() * 100
    print(f"    New prompt: {pr_rate:.1f}%   (old prompt: 90.8% — uninformative)")
    print(f"    VERDICT (c): {'PASS' if 10 <= pr_rate <= 60 else 'REVIEW'} "
          f"(target range 10-60%)")

    # ── (d) Scope distribution ──────────────────────────────────────────────
    print("\n(d) SCOPE DISTRIBUTION (replaces binary systemic flag, old: 94.2% True)")
    scope = new["scope"].value_counts(normalize=True).mul(100).round(1)
    for s in ["SingleBatch", "MultipleProducts", "FacilityWide", "Unclear"]:
        print(f"    {s:<17}: {scope.get(s, 0.0):5.1f}%")
    print(f"    VERDICT (d): {'PASS' if scope.max() < 75 else 'REVIEW — one class dominates'}")

    # ── Spot-check table ────────────────────────────────────────────────────
    print("\n── Spot-check: 10 random rows (verify rationale references evidence) ──")
    cols = ["fei", "obs_num", "severity_tier", "scope", "patient_risk_flag_llm",
            "severity_rationale"]
    sample_rows = new.sample(min(10, len(new)), random_state=7)[cols]
    for _, r in sample_rows.iterrows():
        print(f"\n  FEI {r['fei']} obs {r['obs_num']}: {r['severity_tier']} / {r['scope']} "
              f"/ patient_risk={r['patient_risk_flag_llm']}")
        print(f"    rationale: {str(r['severity_rationale'])[:160]}")

    print("\nIf all four verdicts PASS: archive the old signals CSV, then run")
    print("  python 01_extract_observation_signals.py --force")


if __name__ == "__main__":
    main()
