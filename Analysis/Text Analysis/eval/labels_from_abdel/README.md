# Completed labels from Abdel Kharij

Blinded human annotation of FDA Form 483 observations, used to validate the
Claude v2 extraction. The outgoing package he works from (rules, reference
guide, worked example, blank templates) is in `../sent_to_abdel/`; this folder
holds only what came back.

Data files stay in Google Drive and are not tracked in git.

| File | What it is | n | Use |
|---|---|---|---|
| `20260902_round1_labels_abdel.xlsx` | Round 1, completed | 50 | Scored in `results_and_notes/human_eval_metrics_v2.md`. Also the source of the 20 hidden repeats in round 2, so it is the only basis for test-retest. |
| `20261014_round2_labels_abdel.xlsx` | Round 2, his own labels | 70 | **Primary.** Held-out validation set. |
| `20261014_round2_labels_abdel_ai_assisted.xlsx` | Round 2, after his Claude self-check | 70 | Sensitivity only. Carries the `AI_Revision_Log` sheet. |

## Which round-2 file to use, and why

Use `20261014_round2_labels_abdel.xlsx`. Abdel labeled the 70 observations
himself, then ran them through Claude as a self-check, and sent both versions
so the difference was visible rather than hidden. The assisted file is a
legitimate sensitivity analysis, but it cannot serve as independent human
validation: of its 24 edits to categorical or flag fields, 18 moved his label
to exactly the value the Claude v2 extraction had produced. Pooled agreement
rises from 79.9% on his own labels to about 82% on the assisted ones. An edit
made by the model family under evaluation does not count as independent
judgment even where the rule reading is correct.

Verified 2026-10-14, all three checks passed:

- every one of the 33 cells in `AI_Revision_Log` holds the logged
  `original_value` in the primary file, 33 of 33;
- a full cell-by-cell diff of the two Labeling tabs shows exactly 33 differing
  cells, precisely the 33 in the log, with nothing changed that is not logged;
- `fei`, `obs_num` and `observation_text` are identical across both files on
  all 70 rows.

## Conventions

- Round 2 sampling is recorded in `../validation_data/DO_NOT_SHARE_round2_key.csv`:
  `source` is 50 new and 20 repeat, `round1_row_id` identifies the repeats, and
  `stratum` gives the sampling cell.
- A blank `human_remediation_signal` means he observed nothing about
  remediation, which he confirmed by email. Treat blank as a value, not as
  missing. The extraction does the same, leaving that field blank on 912 of
  1,067 observations.
- Round 1 cannot be the final validation, because the v2 prompt rules were
  revised in response to what it found. Round 2 is the held-out test.

## Naming

Files are named `<date received>_<round>_labels_abdel[_ai_assisted].xlsx`.
Earlier names (`labeling_template_v2.xlsx`,
`labeling_template_round2_done.xlsx`, `labeling_template_round2_done_v02.xlsx`)
were retired on 2026-10-15: they described completed labels as templates, and
the `_v02` suffix read as a later revision when it was in fact the original
unassisted work.
