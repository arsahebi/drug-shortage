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


# ── workbook styling: copied from round 1's labeling_template_v2.xlsx ──────
# Round 1 was usable because every judgment column was a dropdown and the
# input columns were visibly shaded. Rebuilding that here by hand, since
# pandas.to_excel writes a bare grid. The spec below was read back off the
# round-1 file with openpyxl, so round 2 looks identical to the annotator.

COL_WIDTHS = {
    "A": 12, "B": 12, "C": 12, "D":  9, "E": 14, "F": 70, "G": 26,
    "H": 16, "I": 18, "J": 16, "K": 18, "L": 14, "M": 16, "N": 40,
    "O": 18, "P": 22, "Q": 40, "R": 16, "S": 18, "T": 14, "U": 40,
}

# column letter -> allowed values. Columns N, Q, U are free text.
DROPDOWNS = {
    "G": "QualitySystem,ProductionSystem,MaterialsSystem,FacilitiesEquipmentSystem,"
         "LaboratoryControlsSystem,PackagingLabelingSystem,Other",
    "H": "Critical,Major,Moderate,Minor",
    "I": "SingleBatch,MultipleProducts,FacilityWide,Unclear",
    "J": "Capital,Cultural,Mixed,Unclear",
    "K": "Strong,Partial,Weak,None",
    "L": "TRUE,FALSE",
    "M": "TRUE,FALSE",
    "O": "TRUE,FALSE",
    "P": "TRUE,FALSE",
    "R": "TRUE,FALSE",
    "S": "TRUE,FALSE",
    "T": "1,2,3,4,5",
}

FIRST_INPUT_COL = "G"      # A-F are read-only context
HEADER_FILL     = "00D9D9D9"
INPUT_FILL      = "00FFF9E6"

INSTRUCTIONS = [
    "483 Observation Labeling -- Instructions (ROUND 2)",
    "",
    "What this is: real FDA Form 483 observations from our facility inspection "
    "dataset. Assign the same 11 labels a domain expert would, using ONLY the "
    "observation text -- do not look up the facility, drug, or any outside "
    "information.",
    "",
    "Two things that are different from round 1:",
    "",
    "  1. The rules have been amended since you labeled round 1. Data integrity now "
    "includes testing or retesting into compliance. A confirmed environmental "
    "monitoring excursion inside a classified Grade A/B area counts as confirmed "
    "contamination for patient risk. A market complaint naming a batch counts as "
    "evidence that batch was distributed. Please work from the attached "
    "483_Labeling_Rules_v2.docx, which is the current version.",
    "",
    "  2. Please label every row from scratch, using only the text in front of you. "
    "Do not open or consult your round-1 file while you work. Some rows may look "
    "familiar; label them as you read them now rather than trying to match what you "
    "said before. This is deliberate and it is how we measure the consistency of the "
    "labeling itself.",
    "",
    "Full field definitions, valid values, and worked examples are in "
    "483_Labeling_Rules_v2.docx. Read that first -- these are the exact same "
    "definitions our LLM pipeline uses, so your labels are directly comparable to "
    "its output.",
    "",
    "How to fill this in:",
    "  1. Work in the 'Labeling' tab. Columns A-F are read-only context (do not edit).",
    "  2. Columns with a pale yellow fill are yours to complete. Most are dropdown "
    "lists -- click the cell and choose from the arrow.",
    "  3. human_patient_risk_why: a short sentence on WHY you set the patient-risk "
    "flag the way you did. Required whenever the flag is TRUE.",
    "  4. human_contamination_why: a short sentence on WHY you set "
    "human_contamination_flag and human_contamination_risk_flag the way you did -- "
    "this is the newest, subtlest distinction in the rules (confirmed event vs. "
    "control-risk gap), so we want to see your reasoning even when you're confident.",
    "  5. human_confidence_1to5: how confident you are in your own labels for that "
    "row (1 = guessing, 5 = certain).",
    "  6. notes: anything ambiguous, anything you'd flag for discussion, or where the "
    "text itself seems incomplete/redacted in a way that affects your answer.",
    "",
    "Please work independently and do not discuss individual rows with anyone until "
    "you're done -- this keeps the comparison clean. If a row is genuinely ambiguous "
    "even after reading the rules, label it your best judgment and flag it in 'notes' "
    "rather than skipping it.",
    "",
    "When you're done, save the file and send it back -- do not change the file name "
    "or sheet names, and do not add/remove/reorder columns.",
]


def _write_styled_workbook(out: pd.DataFrame) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import column_index_from_string
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()

    ws = wb.active
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 100
    for i, line in enumerate(INSTRUCTIONS, start=1):
        c = ws.cell(row=i, column=1, value=line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        c.font = Font(size=11, bold=(i == 1))
    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 8

    lab = wb.create_sheet("Labeling")
    lab.append(list(out.columns))
    for row in out.itertuples(index=False):
        lab.append(list(row))

    n_rows = len(out)
    last_row = n_rows + 1
    first_input = column_index_from_string(FIRST_INPUT_COL)

    for letter, width in COL_WIDTHS.items():
        lab.column_dimensions[letter].width = width

    header_fill = PatternFill("solid", fgColor=HEADER_FILL)
    for c in lab[1]:
        c.font = Font(bold=True, size=10)
        c.fill = header_fill
        c.alignment = Alignment(wrap_text=True, vertical="center")
    lab.row_dimensions[1].height = 28

    input_fill = PatternFill("solid", fgColor=INPUT_FILL)
    for r in range(2, last_row + 1):
        lab.row_dimensions[r].height = 90
        for ci in range(1, len(out.columns) + 1):
            c = lab.cell(row=r, column=ci)
            c.font = Font(size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if ci >= first_input:
                c.fill = input_fill

    for letter, allowed in DROPDOWNS.items():
        dv = DataValidation(type="list", formula1=f'"{allowed}"', allow_blank=True)
        lab.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{last_row}")

    lab.freeze_panes = "G2"

    wb.save(OUT_XLS)
    print(f"  styled: {len(COL_WIDTHS)} columns, {n_rows} data rows, "
          f"{len(DROPDOWNS)} dropdowns, frozen at G2")


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

    OUT_XLS.parent.mkdir(parents=True, exist_ok=True)
    _write_styled_workbook(out)
    print(f"Saved annotator file -> {OUT_XLS}")

    OUT_KEY.parent.mkdir(parents=True, exist_ok=True)
    combined[["row_id", "key", "source", "round1_row_id", "stratum",
              "stratum_share_corpus", "fei", "insp_date", "obs_num"]].to_csv(OUT_KEY, index=False)
    print(f"Saved private key   -> {OUT_KEY}")
    print("\nDo not send the key file. It identifies which rows are repeats.")


if __name__ == "__main__":
    main()
