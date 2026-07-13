"""
eval/generate_labeling_template.py

Samples ~40 observations from fei_observation_signals.csv and writes
a labeling template CSV pre-populated with the model's predictions.
Fill in the human_* columns, then run evaluate_extraction.py.

Sampling strategy: stratified by severity_tier to ensure all tiers
are represented; also ensures multiple FEIs are sampled.

Run from the project root or from this folder:
  python eval/generate_labeling_template.py
"""

import sys
from pathlib import Path

import pandas as pd

OUT          = Path(__file__).parent.parent
SIGNALS_CSV  = OUT / "fei_observation_signals.csv"
TEMPLATE_CSV = Path(__file__).parent / "labeling_template.csv"

N_SAMPLE = 40     # target sample size
SEED     = 42


def main():
    if not SIGNALS_CSV.exists():
        sys.exit(
            f"Observation signals not found: {SIGNALS_CSV}\n"
            "Run 05_extract_signals_langgraph.py first."
        )

    df = pd.read_csv(SIGNALS_CSV)
    print(f"Total observations available: {len(df)}")
    print(f"FEIs: {df['fei'].nunique()}")

    if len(df) == 0:
        sys.exit("No observations to sample.")

    # Stratified sample by severity_tier
    tiers   = ["High", "Moderate", "Low"]
    n_each  = max(1, N_SAMPLE // len(tiers))
    samples = []
    for tier in tiers:
        sub = df[df["severity_tier"] == tier]
        n   = min(n_each, len(sub))
        if n > 0:
            # Within each tier, sample across multiple FEIs
            samples.append(sub.sample(n=n, random_state=SEED))

    # Fill remaining quota from any tier if one tier was small
    sampled = pd.concat(samples).drop_duplicates(subset=["observation_id"])
    remaining = N_SAMPLE - len(sampled)
    if remaining > 0:
        leftover = df[~df["observation_id"].isin(sampled["observation_id"])]
        extra    = leftover.sample(n=min(remaining, len(leftover)), random_state=SEED+1)
        sampled  = pd.concat([sampled, extra]).drop_duplicates(subset=["observation_id"])

    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(sampled)} observations")
    print(sampled["severity_tier"].value_counts().to_string())

    # Build template rows
    rows = []
    for _, r in sampled.iterrows():
        # Text preview: first 300 chars, escape commas for CSV safety
        preview = str(r.get("evidence_quote", ""))[:300].replace("\n", " ")
        rows.append({
            "fei":                        r.get("fei", ""),
            "doc_id":                     r.get("doc_id", ""),
            "observation_id":             r.get("observation_id", ""),
            "source_type":                r.get("source_type", ""),
            "observation_text_preview":   preview,
            # Model predictions (pre-filled for reference)
            "model_violation_category":   r.get("violation_category", ""),
            "model_severity_tier":        r.get("severity_tier", ""),
            "model_severity_tier_baseline": r.get("severity_tier_baseline", ""),
            "model_root_cause_type":      r.get("root_cause_type", ""),
            "model_remediation_signal":   r.get("remediation_signal", ""),
            "model_repeat_flag":          r.get("repeat_flag", ""),
            "model_systemic_flag":        r.get("systemic_flag", ""),
            "model_patient_risk_flag":    r.get("patient_risk_flag", ""),
            "model_evidence_quote":       str(r.get("evidence_quote", ""))[:200],
            "model_confidence":           round(float(r.get("confidence", 0)), 3),
            # Human labels — FILL THESE IN
            "human_violation_category":   "",
            "human_severity_tier":        "",
            "human_root_cause_type":      "",
            "human_remediation_signal":   "",
            "human_repeat_flag":          "",
            "human_systemic_flag":        "",
            "human_patient_risk_flag":    "",
            "notes":                      "",
        })

    template_df = pd.DataFrame(rows)
    template_df.to_csv(TEMPLATE_CSV, index=False)
    print(f"\nTemplate written to: {TEMPLATE_CSV}")
    print("\nInstructions:")
    print("  Fill in the human_* columns using these valid values:")
    print("  human_violation_category : LabControls | ProductionControls |")
    print("                             BuildingsEquipment | OrgPersonnel |")
    print("                             PackagingLabeling | RecordsReports |")
    print("                             QualitySystem | Other")
    print("  human_severity_tier      : Low | Moderate | High")
    print("  human_root_cause_type    : Capital | Cultural | Mixed | Unclear")
    print("  human_remediation_signal : Strong | Partial | Weak | None")
    print("  human_repeat/systemic/patient_risk_flag : True | False")
    print("\nThen run: python eval/evaluate_extraction.py")


if __name__ == "__main__":
    main()
