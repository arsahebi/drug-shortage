"""
eval/code/06_build_round2_labeling_set.py

Builds the round-2 human labeling file.

Why round 2 exists
──────────────────
Round 1 (50 observations, labeled 2026-09-02) drove the prompt fixes and the
labeling-rule amendments that followed it. Scoring the final pipeline against
those same labels is circular: the rules and the prompt were changed because
of those specific disagreements. Round 1 is therefore the development set, and
round 2 -- labeled under rules frozen beforehand -- is the held-out validation
set that produces the reported accuracy.

What this file contains
───────────────────────
Two kinds of rows, shuffled together and given fresh row_ids so the annotator
cannot tell them apart:

  1. REPEAT rows: a random subset of the round-1 observations, re-issued.
     Scoring these against the annotator's own round-1 answers gives
     test-retest (intra-rater) reliability. With a single annotator this is
     the available substitute for interrater agreement, which the comparable
     published work (Li et al., npj Digital Medicine 2026;9:221) reports
     alongside every accuracy figure as the benchmark. It measures
     consistency, not correctness -- a consistently wrong annotator would
     still look reliable here, and that limitation belongs in the paper.

  2. NEW rows, in two groups:
     a. A random stratum drawn at the corpus's natural distribution. This
        anchors an unbiased accuracy estimate.
     b. Rare-category strata, deliberately oversampled, because round 1 had
        1 Minor severity observation, 3 non-blank remediation values, and
        1-3 observations in the smaller violation categories -- too few to
        say anything per-category. The same paper solved the same problem
        the same way, stratifying on the LLM's own labels to guarantee
        coverage of rare positives.

Stratification uses the model's labels (the only labels available for
unlabeled observations). That is a sampling device only; it does not tell
the annotator anything and does not enter the scoring. Because the rare
strata are oversampled, overall accuracy MUST be weighted back to the corpus
distribution before reporting -- the per-row stratum and its weight are
recorded in the private key for exactly this reason.

Outputs
───────
  eval/sent_to_abdul/labeling_template_round2.xlsx   <- send this one
  eval/validation_data/DO_NOT_SHARE_round2_key.csv   <- never send; maps
      row_id to source (new/repeat), original round-1 row_id, stratum,
      and sampling weight

Run from this folder:
  python 06_build_round2_labeling_set.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

HERE      = Path(__file__).parent
EVAL_ROOT = HERE.parent
DATA      = EVAL_ROOT.parent

FULL_CSV   = DATA / "step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv"
ROUND1_XLS = EVAL_ROOT / "sent_to_abdul" / "labeling_template_v2.xlsx"
OUT_XLS    = EVAL_ROOT / "sent_to_abdul" / "labeling_template_round2.xlsx"
OUT_KEY    = EVAL_ROOT / "validation_data" / "DO_NOT_SHARE_round2_key.csv"

SEED = 20260929
N_REPEAT = 20

# stratum name -> (row filter, target n). Applied in order; rows already
# chosen are not reused, so earlier strata get first claim on overlaps.
def _strata(pool: pd.DataFrame):
    return [
        ("severity_Minor",        pool["severity_tier"] == "Minor",                    8),
        ("remediation_nonblank",  pool["remediation_signal"].notna(),                  8),
        ("violation_rare",        pool["violation_category"].isin(
                                      ["MaterialsSystem", "PackagingLabelingSystem", "Other"]), 8),
        ("scope_Unclear",         pool["scope"] == "Unclear",                          3),
        ("rootcause_Unclear",     pool["root_cause_type"] == "Unclear",                3),
        ("random_natural",        pd.Series(True, index=pool.index),                  20),
    ]

HUMAN_COLS = [
    "human_violation_category", "human_severity_tier", "human_scope",
    "human_root_cause_type", "human_remediation_signal", "human_repeat_flag",
    "human_patient_risk_flag", "human_patient_risk_why", "human_contamination_flag",
    "human_contamination_risk_flag", "human_contamination_why",
    "human_investigation_flag", "human_data_integrity_flag",
    "human_confidence_1to5", "notes",
]


def _key(df):
    return (df["fei"].astype(str).str.strip() + "|"
            + df["insp_date"].astype(str).str.strip() + "|"
            + df["obs_num"].astype(str).str.strip())


def main() -> None:
    rng = np.random.default_rng(SEED)

    full = pd.read_csv(FULL_CSV, low_memory=False)
    r1   = pd.read_excel(ROUND1_XLS, sheet_name="Labeling")
    full["key"] = _key(full)
    r1["key"]   = _key(r1)

    # Round 1 showed the annotator obs_text_clean under the header
    # "observation_text"; verified byte-identical to step00's obs_text. Keep
    # the same column name and the same text so round 2 looks identical.
    full = full.rename(columns={"obs_text_clean": "observation_text"})
    if "cfr_codes" not in full.columns:
        full["cfr_codes"] = ""   # blank in round 1 too; 483s carry no CFR codes

    pool = full[~full["key"].isin(set(r1["key"]))].copy()
    print(f"Corpus {len(full)}, round 1 {len(r1)}, unlabeled pool {len(pool)}")

    # ── repeats ──────────────────────────────────────────────────────────
    rep_idx = rng.choice(r1.index.values, size=min(N_REPEAT, len(r1)), replace=False)
    repeats = r1.loc[rep_idx, ["key", "row_id", "fei", "insp_date", "obs_num",
                                "cfr_codes", "observation_text"]].copy()
    repeats = repeats.rename(columns={"row_id": "round1_row_id"})
    repeats["source"]  = "repeat"
    repeats["stratum"] = "repeat"
    print(f"Repeats drawn from round 1: {len(repeats)}")

    # ── new rows ─────────────────────────────────────────────────────────
    chosen, taken = [], set()
    for name, mask, target in _strata(pool):
        avail = pool[mask & ~pool["key"].isin(taken)]
        n = min(target, len(avail))
        if n == 0:
            print(f"  [{name}] none available, skipped")
            continue
        pick = avail.loc[rng.choice(avail.index.values, size=n, replace=False)].copy()
        pick["stratum"] = name
        # sampling weight: how much this stratum was over/under-sampled
        # relative to its share of the corpus, for weighting results back
        stratum_share_corpus = float(mask.sum()) / len(pool)
        pick["stratum_share_corpus"] = round(stratum_share_corpus, 6)
        chosen.append(pick)
        taken.update(pick["key"])
        print(f"  [{name}] requested {target}, available {len(avail)}, took {n}")

    new = pd.concat(chosen, ignore_index=True)
    new["source"] = "new"
    new["round1_row_id"] = pd.NA
    new = new[["key", "round1_row_id", "fei", "insp_date", "obs_num", "cfr_codes",
               "observation_text", "source", "stratum", "stratum_share_corpus"]]
    repeats["stratum_share_corpus"] = pd.NA
    repeats = repeats[new.columns]

    combined = pd.concat([new, repeats], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    combined.insert(0, "row_id", range(1, len(combined) + 1))
    print(f"\nTotal round-2 rows: {len(combined)} "
          f"({(combined.source=='new').sum()} new, {(combined.source=='repeat').sum()} repeat)")

    # ── annotator file: no source, no stratum, no model labels ───────────
    out = combined[["row_id", "fei", "insp_date", "obs_num", "cfr_codes",
                    "observation_text"]].copy()
    for c in HUMAN_COLS:
        out[c] = ""

    instructions = pd.read_excel(ROUND1_XLS, sheet_name="Instructions", header=None)
    note = pd.DataFrame({0: [
        "ROUND 2",
        "",
        "Same task and same fields as round 1. Two things to know:",
        "",
        "1. The labeling rules have been updated since round 1 (data integrity now "
        "includes testing/retesting into compliance; a confirmed EM excursion inside a "
        "classified Grade A/B area counts as confirmed contamination for patient risk; a "
        "market complaint naming a batch counts as evidence that batch was distributed). "
        "Please work from the current 483_Labeling_Rules_v2.docx, not from memory of round 1.",
        "",
        "2. Please label every row independently, using only the observation text in front "
        "of you. Do not look back at your round-1 answers.",
        "",
        "-" * 60,
        "",
    ]})
    instructions = pd.concat([note, instructions], ignore_index=True)

    OUT_XLS.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLS, engine="openpyxl") as xw:
        instructions.to_excel(xw, sheet_name="Instructions", index=False, header=False)
        out.to_excel(xw, sheet_name="Labeling", index=False)
    print(f"Saved annotator file -> {OUT_XLS}")

    OUT_KEY.parent.mkdir(parents=True, exist_ok=True)
    combined[["row_id", "key", "source", "round1_row_id", "stratum",
              "stratum_share_corpus", "fei", "insp_date", "obs_num"]].to_csv(OUT_KEY, index=False)
    print(f"Saved private key   -> {OUT_KEY}")
    print("\nDo not send the key file. It identifies which rows are repeats.")


if __name__ == "__main__":
    main()
