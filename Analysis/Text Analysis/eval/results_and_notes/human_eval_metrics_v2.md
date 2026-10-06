# Human Labeling vs. LLM (v2) -- Agreement Metrics

> Generated: 2026-09-02  |  Source: `eval/human_eval_02_score.py`

## Summary Table

| Field | Accuracy | Macro F1 | N |
|---|---|---|---|
| violation_category | 0.860 | 0.892 | 50 |
| severity_tier | 0.820 | 0.850 | 50 |
| scope | 0.720 | 0.574 | 50 |
| root_cause_type | 0.840 | 0.786 | 50 |
| remediation_signal | 0.960 | 0.596 | 50 |
| repeat_flag | 1.000 | 1.000 | 50 |
| patient_risk_flag | 0.880 | 0.777 | 50 |
| contamination_flag | 0.840 | 0.729 | 50 |
| contamination_risk_flag | 0.680 | 0.672 | 50 |
| investigation_flag | 0.960 | 0.952 | 50 |
| data_integrity_flag | 0.960 | 0.905 | 50 |

## Per-Class Breakdown

### violation_category

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| facilitiesequipmentsystem | 0.917 | 1.000 | 0.957 | 11 | 1 | 0 |
| laboratorycontrolssystem | 0.769 | 1.000 | 0.870 | 10 | 3 | 0 |
| materialssystem | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| other | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| productionsystem | 0.857 | 0.545 | 0.667 | 6 | 1 | 5 |
| qualitysystem | 0.857 | 0.857 | 0.857 | 12 | 2 | 2 |

### severity_tier

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| critical | 1.000 | 0.625 | 0.769 | 5 | 0 | 3 |
| major | 0.667 | 0.875 | 0.757 | 14 | 7 | 2 |
| minor | 1.000 | 1.000 | 1.000 | 1 | 0 | 0 |
| moderate | 0.913 | 0.840 | 0.875 | 21 | 2 | 4 |

### scope

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| facilitywide | 0.750 | 0.818 | 0.783 | 18 | 6 | 4 |
| multipleproducts | 0.765 | 0.722 | 0.743 | 13 | 4 | 5 |
| singlebatch | 1.000 | 0.625 | 0.769 | 5 | 0 | 3 |
| unclear | 0.000 | 0.000 | 0.000 | 0 | 4 | 2 |

### root_cause_type

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| capital | 0.952 | 0.909 | 0.930 | 20 | 1 | 2 |
| cultural | 0.923 | 0.750 | 0.828 | 12 | 1 | 4 |
| mixed | 0.600 | 0.900 | 0.720 | 9 | 6 | 1 |
| unclear | 1.000 | 0.500 | 0.667 | 1 | 0 | 1 |

### remediation_signal

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| none | 1.000 | 0.979 | 0.989 | 46 | 0 | 1 |
| partial | 1.000 | 0.667 | 0.800 | 2 | 0 | 1 |
| weak | 0.000 | 0.000 | 0.000 | 0 | 2 | 0 |

### repeat_flag

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| true | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |
| false | 1.000 | 1.000 | 1.000 | 47 | 0 | 0 |

### patient_risk_flag

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| true | 1.000 | 0.455 | 0.625 | 5 | 0 | 6 |
| false | 0.867 | 1.000 | 0.929 | 39 | 6 | 0 |

### contamination_flag

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| true | 0.833 | 0.417 | 0.556 | 5 | 1 | 7 |
| false | 0.841 | 0.974 | 0.902 | 37 | 7 | 1 |

### contamination_risk_flag

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| true | 0.448 | 1.000 | 0.619 | 13 | 16 | 0 |
| false | 1.000 | 0.568 | 0.724 | 21 | 0 | 16 |

### investigation_flag

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| true | 0.875 | 1.000 | 0.933 | 14 | 2 | 0 |
| false | 1.000 | 0.944 | 0.971 | 34 | 0 | 2 |

### data_integrity_flag

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| true | 0.833 | 0.833 | 0.833 | 5 | 1 | 1 |
| false | 0.977 | 0.977 | 0.977 | 43 | 1 | 1 |

## patient_risk_flag Rationale Comparison (qualitative)

Side-by-side text for every row where the RA marked patient_risk = TRUE (regardless of what the model said) -- read through for agreement in reasoning, not just the boolean flag.

**FEI 1450022, obs 7**
- RA: Scenario (a). Injectable (aseptically filled vials) with foreign material actually found - significant debris on filled vials including ones that had already passed inspection, plus apparent rust on the stopper bowl platform, which stoppers sit on. (a) doesn't need release.
- Model (False): No confirmed sterility breach or contamination in released product is stated, no high-risk drug class is named, and no statement affirms the affected lot was distributed or released without QA disposition.

**FEI 3002808385, obs 1**
- RA: Scenario (b). Confirmed defect (brown spots found on the filling bottles) in product the text says reached the market - about 50% of units of the first batch were 'released and distributed to the market prior to the discovery', and the second batch on the same bottle lot 'was released, distributed'. That's a real release statement, not the boilerplate.
- Model (True): Scenario (b) applies: the text affirmatively states "Approximately 50% ... were released and distributed to the market" and a second batch "was released, distributed" with the same defective bottle lot.

**FEI 3002949099, obs 2**
- RA: Scenario (b). Confirmed OOS results (listed as a qualifying defect) in product the text says reached the market - part C says results were recalculated for 'previously tested, approved and shipped batches' and that no fewer than the stated number of batches of that strength are 'currently on the U.S. market', tested by the unverified in-house method.
- Model (True): (b) text-affirmed release/distribution: "there are currently no less than (b)(4) batches of the (b)(4) mg strength of (b)(4) Capsules USP on the U.S. market" using an unverified/OOS-linked test method, and separately "previously tested, approved and shipped batches" were recalculated using a non-scientifically-justified method change.

**FEI 3002984011, obs 5**
- RA: Scenario (a). Sterile and aseptically filled product with confirmed contamination - EM trend data shows recurring Bacillus and Pseudomonas recoveries in Grade B and Grade A, specific individuals identified as a source, and a Grade B corridor identified as a route into Grade A. (a) doesn't need release.
- Model (False): Although recurring microbial contamination is documented on personnel/environment, the text does not state that any specific drug product batch was released or distributed, nor does it confirm contamination in the drug product itself — only in the personnel/environmental monitoring program — so none of the four patient risk scenarios (a, a2, b, c) are met.

**FEI 3003342394, obs 3**
- RA: Scenario (a). Injectable (prefilled syringes) with confirmed contamination - particles and fibers found in the PPQ batches during visual inspection. (a) needs no release language, and PPQ batches are pre-commercial anyway.
- Model (False): No statement that any affected PPQ batch was released or distributed to patients; these are PPQ (process performance qualification) batches with confirmed defects but no stated release, and no text-named high-risk drug class or sterile contamination event confirmed.

**FEI 3003952717, obs 2**
- RA: Scenario (b). Confirmed OOS results (a listed qualifying defect) in batches the text says were 'shipped to the U.S.A'. That's a distribution statement on its own, separate from the boilerplate.
- Model (True): (b) text-affirmed release/distribution: "USP (b)(4) batch shipped to the U.S.A (b)(4) and (b)(4) USP batch shipped to the U.S.A (b)(4)" confirms distribution of batches with unresolved OOS/root cause issues.

**FEI 3003999190, obs 4**
- RA: Scenario (b). Item 2 has a confirmed defect in distributed product - a customer got an OOS for dissolution on a named EU market batch and the OOS was confirmed on retain testing. A customer complaint on a market batch says the product was distributed.
- Model (False): No scenario applies: the opening phrase "whether not the batch has been already distributed" is boilerplate CFR language, not an affirmative statement of distribution, and no sterile/injectable contamination, text-named high-risk drug class, or QA-disposition bypass is described.

**FEI 3004021229, obs 6**
- RA: Scenario (b). Confirmed quality defect (empty blisters, missing units, empty cartons, with root causes the firm assigned at the punching tool and at reject repacking) in product that reached the US market. Roughly 100 field complaints from 2016 to the inspection have to come from distributed product.
- Model (False): No sterile/injectable contamination, no text-named NTI/oncology/nitrosamine drug class, and no statement that the specific defective units were actually distributed to patients/market beyond being the subject of complaints; complaints alone do not affirmatively state release of a specific defective batch under the (b) scenario definition.

**FEI 3004097901, obs 4**
- RA: Scenario (b). Market complaint MC-GGP-23-0066 has black particles confirmed in named batches, with root cause established as discoloured API stuck to the chamber wall that the visual inspection missed. 'Market complaint' says the affected product was distributed.
- Model (False): No scenario applies: the complaint describes a confirmed defect found via market complaint but the text does not affirmatively state which batches were distributed/released as defective beyond the complaint being received (implying some distribution occurred but not explicitly stated as released), the product is an oral tablet with no text-named NTI/oncology/nitrosamine class, and there is no statement of release without QA disposition.

**FEI 3004537783, obs 1**
- RA: Scenario (b). Confirmed quality defect - impurity results above the regulatory spec, verified by retain and stability testing back in September 2023 - in product the text says was released to the US market between 11/23/2023 and 03/21/2024, plus lots already identified in the US market.
- Model (True): (b) text-affirmed release/distribution: "your firm released [redacted] lots of multiple drug products to the US market with [impurity] results" exceeding specification.

**FEI 3004819820, obs 1**
- RA: Scenario (b). Confirmed defects in product the text says was released - 'These batches ... were released to US market based on (b)(4) confirmatory test' after unresolved assay OOS results, and item 2's batch with 224,957 defective spotted tablets and an out of spec yield 'was released to US market'.
- Model (True): (b) text-affirmed release/distribution: "These batches...were released to US market" and "The batch was released to US market based on new...sample preparation" and "The batch was released to US market" for the defective spotted tablets.

## contamination_flag / contamination_risk_flag Disagreements (qualitative)

The model has no dedicated rationale field for this split (unlike patient_risk), so this is a disagreement audit trail, not a side-by-side comparison: every row where the RA's contamination_flag or contamination_risk_flag differs from the model's, with the RA's stated reasoning. Use this to tell a genuine misread of the confirmed-event-vs-control-risk split apart from a defensible close call.

**FEI 1036352, obs 2**
- RA: contamination_flag=False, contamination_risk_flag=False -- Neither. Samples sitting unprotected on an open bench is a sample integrity problem, not one of the contamination control categories, and nothing was actually found.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 1450022, obs 7**
- RA: contamination_flag=True, contamination_risk_flag=False -- The debris was actually observed on filled vials, not just risked, so I put it under the event flag and risk goes FALSE by mutual exclusivity. The skipped cleanings would otherwise read as a straight control gap.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 1525864, obs 1**
- RA: contamination_flag=True, contamination_risk_flag=False -- Metallic particles were actually observed on the separation and dosing stations during production of a named batch. Material found, not a hypothetical, so event flag.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 2111358, obs 1**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed cross-contamination. Swab analysis found Propafenone HCl in the dryer's hot air inlet against a spec, plus unidentified peaks. That's material actually found and measured.
- Model: contamination_flag=True, contamination_risk_flag=True

**FEI 2434153, obs 1**
- RA: contamination_flag=False, contamination_risk_flag=False -- Neither. Humidity control for empty capsules is about moisture and degradation, not one of the contamination control categories, and nothing was found.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 3002608066, obs 2**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. The lot failed the visible particles spec at 12 months and an actual plastic particle was recovered. Particulate confirmed in product.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 3002808385, obs 1**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Brown spots were actually observed on the bottles used to fill product on 02/13/2019. Foreign material found on product containers.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 3002984011, obs 5**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Recurring microbial growth was actually recovered in Grade A and B and from personnel, and 'microbial growth found' is one of the flag's own anchors. That beats the monitoring design gaps in the same observation.
- Model: contamination_flag=True, contamination_risk_flag=True

**FEI 3003342394, obs 3**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Particles and fibers were actually found in product during visual inspection, which is one of the flag's own anchors, so risk goes FALSE by mutual exclusivity. The incomplete follow-up belongs to investigation_flag, not here.
- Model: contamination_flag=True, contamination_risk_flag=True

**FEI 3003395329, obs 1**
- RA: contamination_flag=False, contamination_risk_flag=False -- Neither. Some complaints sound like contamination ('look moldy', 'look contaminated and brown') but nobody verified them because the retains were never examined, so nothing is confirmed. And no contamination control gap is described, the deficiency is complaint investigation.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 3003999190, obs 4**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Foreign matter was actually observed in a batch (DF-23012) and similar foreign matter was found in the equipment duct. Material found, not a gap.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 3004097901, obs 4**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Black particles were actually found in the product batches and traced to material adhered to the chamber wall. Foreign matter confirmed in product.
- Model: contamination_flag=True, contamination_risk_flag=True

**FEI 3004106442, obs 1**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Potassium chloride from Klor-Con Batch 412238 was still on the shared blender AFTER a Minor Clean, and there was significant white particulate build-up on the pre-weigh booth filters in a room used for many APIs. Residue actually found on shared product contact equipment, not just a procedural risk.
- Model: contamination_flag=True, contamination_risk_flag=True

**FEI 3004453700, obs 2**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Video review showed compressed air blowing Oxycodone/APAP product onto already-cleaned equipment, walls and ceilings, and those weren't cleaned before the next product was packaged. That's actual cross-contamination of cleaned product contact equipment, not just a validation gap.
- Model: contamination_flag=False, contamination_risk_flag=True

**FEI 3004537783, obs 1**
- RA: contamination_flag=False, contamination_risk_flag=False -- Neither, but this is a close one. A chemical impurity leaching out of packaging material reads as an impurity and spec failure rather than a microbial, particulate or cross-contamination event, and no contamination control gap is described. That said, the impurity did literally get into the product from the container, so I can see the other call.
- Model: contamination_flag=True, contamination_risk_flag=True

**FEI 3004819820, obs 1**
- RA: contamination_flag=True, contamination_risk_flag=False -- Confirmed. Spots were actually observed on finished tablets, 224,957 defective tablets were separated out, and no qualitative test was run to rule out contaminants other than the assumed source. Foreign material confirmed on product.
- Model: contamination_flag=False, contamination_risk_flag=True

## Notes for Paper

- Field definitions match `483_Labeling_Rules_v2.docx` (rules) and `483_Background_Reference_Guide.docx` (onboarding/background, not shown to the model) -- the rules document is a plain-language rewrite of the v2 LLM prompt in `01_extract_observation_signals.py`.
- Labeling was blind: the RA never saw model predictions while labeling (see `human_eval_01_generate_template.py`).
- `patient_risk_flag` carries a rationale comparison above -- treat the flag's F1 alongside a read of the reasoning, not the F1 alone, given how narrowly it is scoped.
- This is a complementary check to the existing LLM-vs-Redica convergent-validity comparison (`eval/validate_llm_vs_redica.py`); Redica is itself human-coded, so this file is the actual independent ground-truth check.
