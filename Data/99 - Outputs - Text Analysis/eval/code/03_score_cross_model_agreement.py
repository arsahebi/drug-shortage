"""
eval/code/03_score_cross_model_agreement.py

Scores Claude Sonnet 5 vs. GPT-5-mini extraction agreement on the full
redica v2 observation set (1,067 observations, both providers, same prompt).

This is a much larger comparison than the 50-row human-eval sample in
eval/sent_to_abdul/ and eval/validation_data/ -- a free, at-scale robustness
check. Fields where the two models agree closely are lower risk to use
as-is; fields where they diverge are candidates for a smaller, targeted
human-review pass instead of a large blind one (see
eval/results_and_notes/20260909_session_handoff.md, "Cross-model agreement").

Run from this folder:
  python 03_score_cross_model_agreement.py

Output:
  eval/results_and_notes/claude_vs_gpt_agreement.csv
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE      = Path(__file__).parent
EVAL_ROOT = HERE.parent
DATA      = EVAL_ROOT.parent

CLAUDE_CSV = DATA / "step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv"
GPT_CSV    = DATA / "step01_redica_483_obs_llm_signals_openai_v2.csv"
OUT_CSV    = EVAL_ROOT / "results_and_notes" / "claude_vs_gpt_agreement.csv"

JOIN_KEY = ["fei", "insp_date", "obs_num"]

CATEGORICAL_FIELDS = [
    "violation_category", "severity_tier", "scope", "root_cause_type", "remediation_signal",
]
BINARY_FIELDS = [
    "repeat_flag_llm", "patient_risk_flag_llm", "contamination_flag_llm",
    "contamination_risk_flag_llm", "investigation_flag_llm", "data_integrity_flag_llm",
]


def main() -> None:
    claude = pd.read_csv(CLAUDE_CSV)
    gpt    = pd.read_csv(GPT_CSV)
    print(f"Claude: {len(claude)} rows, {claude['fei'].nunique()} FEIs")
    print(f"GPT:    {len(gpt)} rows, {gpt['fei'].nunique()} FEIs")

    merged = claude.merge(gpt, on=JOIN_KEY, suffixes=("_c", "_g"))
    print(f"Merged (matched on {JOIN_KEY}): {len(merged)} rows")
    if len(merged) != len(claude) or len(merged) != len(gpt):
        print("  WARNING: merged row count does not match both inputs -- "
              "check the join key still uniquely identifies observations.")

    rows = []
    for field in CATEGORICAL_FIELDS + BINARY_FIELDS:
        a, b = merged[f"{field}_c"], merged[f"{field}_g"]
        valid = a.notna() & b.notna()
        n = int(valid.sum())
        if n == 0:
            rows.append({"field": field, "n": 0, "agreement": float("nan"), "kappa": float("nan")})
            continue
        agreement = float((a[valid] == b[valid]).mean())
        try:
            kappa = float(cohen_kappa_score(a[valid].astype(str), b[valid].astype(str)))
        except Exception:
            kappa = float("nan")
        rows.append({"field": field, "n": n, "agreement": round(agreement, 4), "kappa": round(kappa, 4)})

    result = pd.DataFrame(rows).sort_values("agreement", ascending=False)
    print("\n" + result.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
