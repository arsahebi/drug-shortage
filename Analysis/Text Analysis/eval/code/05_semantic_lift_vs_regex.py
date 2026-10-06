"""
eval/code/05_semantic_lift_vs_regex.py

Semantic lift: what does the LLM catch that keyword rules cannot?

Why this script exists
──────────────────────
The manuscript's "Semantic Lift: LLM vs. Regex" analysis is the argument for
why an LLM is needed here at all rather than a keyword baseline. Its numbers
could not be reproduced on the current corpus: every has_*_regex column in
step01_redica_*.csv is False for all 1,067 observations.

Cause (found 2026-09-28): 01_extract_observation_signals.py copies regex flags
from its input via REGEX_FLAG_MAP, i.e. obs_row.get("has_contamination", False).
The PDF-sourced input (Data/12 - FDA - 483/processed/483_observations.csv) does
carry those columns, but the Redica-sourced input
(step00_redica_483_observations.csv) never had them, so every lookup silently
returned the False default. The regex baseline was never actually run on the
Redica corpus.

This script recomputes the baseline directly on the Redica observation text,
using the SAME patterns that produced the original PDF-corpus flags, copied
verbatim from
Data/12 - FDA - 483/processed/code/20260316_483_comprehensive_extraction.py
(SIGNAL_PATTERNS), so the comparison is like-for-like with the manuscript's
original design.

Note on contamination: prompt v2 split the single contamination concept into
contamination_flag_llm (confirmed event) and contamination_risk_flag_llm
(control gap, no confirmed event). The regex pattern does not make that
distinction, so contamination is reported twice: against the confirmed flag
alone, and against either flag.

Run from this folder:
  python 05_semantic_lift_vs_regex.py

Output:
  eval/results_and_notes/semantic_lift_vs_regex.csv
"""

from pathlib import Path
import re
import pandas as pd

HERE      = Path(__file__).parent
EVAL_ROOT = HERE.parent
DATA      = EVAL_ROOT.parent

REDICA_OBS_CSV = DATA / "step00_redica_483_observations.csv"
LLM_CSV        = DATA / "step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv"
OUT_CSV        = EVAL_ROOT / "results_and_notes" / "semantic_lift_vs_regex.csv"

JOIN_KEY = ["fei", "insp_date", "obs_num"]

# Verbatim from 20260316_483_comprehensive_extraction.py -- do not "improve"
# these patterns. The point of the comparison is that this is what a
# reasonable keyword baseline looks like, and it is the exact baseline the
# original analysis used.
SIGNAL_PATTERNS = {
    "repeat": (
        r"\b(?:repeat(?:ed)?|r\s+e\s*p\s*e\s*a\s*t)\s+"
        r"(?:observation|observations|violation|violations|finding|findings|cite|citation|citations)\b"
    ),
    "systemic": r"\b(frequently|routinely|consistently|numerous|systemic|widespread|multiple)\b",
    "wl_ref": r"warning\s+letter|consent\s+decree|import\s+alert",
    "data_integrity": r"data\s+integrit|ALCOA|audit\s+trail|original\s+record|metadata",
    "contamination": r"contamina|microbial|particulate|endotoxin|bioburden|sterility\s+failure",
    "oos_oot": r"\bOOS\b|\bOOT\b|out[-\s]of[-\s]specification|out[-\s]of[-\s]trend",
    "patient_risk": r"patient\s+(risk|harm|impact|safety)|adverse\s+event|recall|released.*market|US market",
    "quality_unit": r"quality\s+(unit|control|assurance)|\bQU\b|\bQA\b",
    "investigation": r"investigat(e|ed|ion|ions)|root\s+cause|corrective\s+action|\bCAPA\b",
    "documentation": r"document|record|logbook|SOP|procedure|written\s+procedure",
    "laboratory": r"laborator|method|assay|chromatograph|specification|sample",
    "equipment_facility": r"equipment|facility|building|HEPA|HVAC|maintenance|cleaning",
    "process_control": r"in[-\s]?process|process\s+validation|batch\s+(record|production)|manufacturing",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in SIGNAL_PATTERNS.items()}

# regex signal -> LLM flag column(s). Only the five concepts the LLM also
# extracts as binary flags are comparable; the rest of the regex set has no
# LLM counterpart and is not part of this table.
COMPARISONS = [
    ("Contamination (confirmed)", "contamination", ["contamination_flag_llm"]),
    ("Contamination (confirmed or risk)", "contamination",
     ["contamination_flag_llm", "contamination_risk_flag_llm"]),
    ("Patient risk",   "patient_risk",   ["patient_risk_flag_llm"]),
    ("Data integrity", "data_integrity", ["data_integrity_flag_llm"]),
    ("Investigation",  "investigation",  ["investigation_flag_llm"]),
    ("Repeat finding", "repeat",         ["repeat_flag_llm"]),
]


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def main() -> None:
    obs = pd.read_csv(REDICA_OBS_CSV, low_memory=False)
    llm = pd.read_csv(LLM_CSV, low_memory=False)
    print(f"Redica observations: {len(obs)} rows")
    print(f"LLM signals:         {len(llm)} rows")

    text = obs["obs_text"].fillna("").astype(str)
    for name, rx in COMPILED.items():
        obs[f"regex_{name}"] = text.str.contains(rx)

    for k in JOIN_KEY:
        obs[k] = obs[k].astype(str).str.strip()
        llm[k] = llm[k].astype(str).str.strip()

    regex_cols = [f"regex_{k}" for k in COMPILED]
    merged = llm.merge(obs[JOIN_KEY + regex_cols], on=JOIN_KEY, how="inner")
    print(f"Merged on {JOIN_KEY}: {len(merged)} rows")
    if len(merged) != len(llm):
        print("  WARNING: merged row count differs from the LLM file -- check the join key.")

    rows = []
    n = len(merged)
    for label, rx_name, llm_flags in COMPARISONS:
        missing = [c for c in llm_flags if c not in merged.columns]
        if missing:
            print(f"  [SKIP] {label}: missing {missing}")
            continue
        r = merged[f"regex_{rx_name}"].fillna(False)
        l = _to_bool(merged[llm_flags[0]])
        for extra in llm_flags[1:]:
            l = l | _to_bool(merged[extra])
        rows.append({
            "signal":          label,
            "regex_share":     round(100 * r.mean(), 1),
            "llm_share":       round(100 * l.mean(), 1),
            "llm_only_lift":   round(100 * (~r & l).mean(), 1),
            "regex_only":      round(100 * (r & ~l).mean(), 1),
            "both":            round(100 * (r & l).mean(), 1),
            "agreement":       round(100 * (r == l).mean(), 1),
            "n":               n,
        })

    result = pd.DataFrame(rows)
    print("\nSemantic lift (all values are % of observations):")
    print(result.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
