# %%
"""
01_extract_observation_signals.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Sends each 483 observation (obs_text_clean) to OpenAI and extracts a set of
  context-aware structured risk signals: violation category, severity tier,
  root-cause type, remediation signal, seven binary flags, and an evidence
  quote verbatim from the observation text.

  This is step 1 of the optional LLM pipeline:
      01 → 02 → (03 optional merge)

  The core combined dataset (01 → 03) does NOT need this script.

INPUT (current — --source redica --provider anthropic)
  redica_483_observations.csv
    One row per observation. Created by 00_load_redica_obs.py.

OUTPUT (current — --source redica --provider anthropic)
  redica_483_obs_llm_signals_anthropic_v2.csv
    One row per observation. Stable join key: (fei, insp_date, obs_num).
    Carries all source metadata + LLM fields.
    Fed into 04_build_combined_obs_universe.py as the Redica LLM signal file.

IDEMPOTENCY
  On re-run, already-scored rows (matched by fei + filename + obs_num) are
  skipped automatically. Use --force to re-score everything.

PARTIAL SAVES
  Results are written to disk every SAVE_EVERY observations so a crash
  does not lose progress. Existing rows are always preserved.

CLI OPTIONS
  --dry-run    Show observation counts and cost estimate; no API calls.
  --limit N    Process only the first N pending observations (for testing).
  --fei N      Process only observations for a single FEI (for testing).
  --force      Re-score every observation even if already in the output file.
  --sample N   Stratified sample of N observations (round-robin across FEIs).
               Writes to 483_observation_context_signals_sampleN.csv and does
               NOT touch the main output file. Used to validate prompt changes
               before a full re-run.

INTERACTIVE USE
  This file is organized as notebook-style cells. To run line by line, edit the
  INTERACTIVE CONFIG values below and execute cells from top to bottom.

DEPENDENCIES
  pip install openai anthropic pandas

ENVIRONMENT
  export OPENAI_API_KEY="sk-..."      # for --provider openai (default)
  export ANTHROPIC_API_KEY="sk-ant-"  # for --provider anthropic
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
DATA       = HERE.parent                             # .../Data/
OBS_CSV    = DATA / "12 - FDA - 483" / "processed" / "483_observations.csv"
SIGNALS_CSV = HERE / "483_observation_context_signals.csv"

# ── Source mode — set via --source argument (overrides below after argparse) ──
# "pdf"    : read from 483_observations.csv, text col = obs_text_clean (default)
# "redica" : read from redica_483_observations.csv, text col = obs_text
SOURCE = "pdf"
_REDICA_OBS_CSV     = HERE / "redica_483_observations.csv"
_REDICA_SIGNALS_CSV_OPENAI    = HERE / "redica_483_obs_llm_signals.csv"
_REDICA_SIGNALS_CSV_ANTHROPIC = HERE / "redica_483_obs_llm_signals_anthropic_v2.csv"
_REDICA_SIGNALS_CSV = _REDICA_SIGNALS_CSV_OPENAI  # resolved after argparse

# ── Provider / Model ───────────────────────────────────────────────────────
PROVIDER         = "anthropic"                 # "openai" or "anthropic"
MODEL_NAME       = "gpt-5-mini"               # OpenAI model
ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001" # Anthropic model (cheapest/fastest)
MAX_TOKENS       = 4000
RATE_LIMIT_RETRIES = 4    # retries per request on RateLimitError
RATE_LIMIT_SLEEP   = 65   # seconds; grows linearly per attempt
SAVE_EVERY = 50      # write partial results every N observations

# ── Regex flag columns → renamed output columns ────────────────────────────
REGEX_FLAG_MAP = {
    "has_repeat":             "has_repeat_regex",
    "has_systemic":           "has_systemic_regex",
    "has_wl_ref":             "has_wl_ref_regex",
    "has_data_integrity":     "has_data_integrity_regex",
    "has_contamination":      "has_contamination_regex",
    "has_oos_oot":            "has_oos_oot_regex",
    "has_patient_risk":       "has_patient_risk_regex",
    "has_quality_unit":       "has_quality_unit_regex",
    "has_investigation":      "has_investigation_regex",
    "has_documentation":      "has_documentation_regex",
    "has_laboratory":         "has_laboratory_regex",
    "has_equipment_facility": "has_equipment_facility_regex",
    "has_process_control":    "has_process_control_regex",
}

# ── Valid categorical values ───────────────────────────────────────────────
VALID_VIOLATION_CATEGORY = {
    "LabControls", "ProductionControls", "BuildingsEquipment",
    "OrgPersonnel", "PackagingLabeling", "RecordsReports",
    "QualitySystem", "Other",
}
VALID_SEVERITY_TIER    = {"Critical", "Major", "Moderate", "Minor"}
VALID_SCOPE            = {"SingleBatch", "MultipleProducts", "FacilityWide", "Unclear"}
VALID_ROOT_CAUSE_TYPE  = {"Capital", "Cultural", "Mixed", "Unclear"}
VALID_REMEDIATION           = {"Strong", "Partial", "Weak", "None"}
VALID_DATA_INTEGRITY_TYPE   = {
    "Falsification", "AuditTrail", "RawData", "ContemporaneousRecording", "NoIssue"
}  # kept for backward compat; not used in schema — DI is now binary flag only

LLM_FLAG_FIELDS = [
    "repeat_flag_llm", "patient_risk_flag_llm",
    "contamination_flag_llm",
    "investigation_flag_llm",
]

# ── Patient risk rules (provider-specific) ─────────────────────────────────
# OpenAI version: same as before — no changes.
# Anthropic version: stricter negative examples to fix the 79% over-firing.
_PATIENT_RISK_RULE_OPENAI = (
    "mark true ONLY when an EXPLICIT harm pathway to patients "
    "exists in the text: (a) sterile or injectable product with a contamination or sterility "
    "assurance failure, OR (b) a confirmed quality defect (OOS, mix-up, wrong potency, "
    "mislabeling) in product that was released or distributed, OR (c) the text states product "
    "was released without required QA disposition or testing. Do NOT mark true for generic "
    'quality deviations where harm would require a chain of hypotheticals. "Could affect '
    'quality" is NOT a harm pathway.'
)

_PATIENT_RISK_RULE_ANTHROPIC = (
    "mark true ONLY for these three scenarios — nothing else qualifies:\n"
    "  (a) Sterile or injectable product with CONFIRMED contamination or sterility breach "
    "documented in the observation.\n"
    "  (b) A quality defect (OOS result, mix-up, wrong potency, mislabeling) confirmed in "
    "product that was ALREADY released or distributed to patients.\n"
    "  (c) The text explicitly states product was released without required QA testing or "
    "disposition.\n"
    "  ALWAYS mark false for: oral solid dose forms (tablets, capsules, powders); missing "
    "SOPs or documentation gaps; environmental monitoring gaps without confirmed contamination; "
    "equipment validation gaps without confirmed product impact; data integrity issues without "
    "confirmed release of affected product; any general quality system failures; training or "
    "personnel qualification deficiencies; stability testing gaps; specification issues without "
    "a released OOS result.\n"
    "  Rule of thumb: if a patient is not already at risk RIGHT NOW from something the "
    "facility already released, mark false."
)

# ── Prompt template (OpenAI — observation before rules, JSON schema inline) ──
_PROMPT_TEMPLATE = """\
You are analyzing FDA Form 483 Inspectional Observation text from a pharmaceutical \
manufacturing inspection.

FDA Form 483 observations are written by FDA investigators to document specific \
violations or deficiencies found at a drug manufacturing facility. Each observation \
typically cites one or more sections of Title 21 CFR (Code of Federal Regulations).

CFR codes cited in this observation: {cfr_codes}

Observation text:
---
{obs_text_clean}
---

Return a single JSON object with EXACTLY these fields \
(no extra text, no markdown fences, just the JSON):

{{
  "violation_category": "<LabControls | ProductionControls | BuildingsEquipment | \
OrgPersonnel | PackagingLabeling | RecordsReports | QualitySystem | Other>",
  "severity_tier": "<Critical | Major | Moderate | Minor>",
  "severity_rationale": "<1–2 sentences. MUST reference the evidence_quote text to \
justify the tier assignment>",
  "scope": "<SingleBatch | MultipleProducts | FacilityWide | Unclear>",
  "root_cause_type": "<Capital | Cultural | Mixed | Unclear>",
  "root_cause_rationale": "<1–2 sentences. Capital = equipment/facility/SOP design gap; \
Cultural = training/management/data-integrity failure; Mixed = clear evidence of both; \
Unclear = text insufficient to decide>",
  "remediation_signal": "<Strong | Partial | Weak | None>",
  "repeat_flag_llm": <true or false — explicit evidence this is a repeat finding>,
  "patient_risk_flag_llm": <true or false — explicit harm pathway to patients exists>,
  "data_integrity_flag_llm": <true or false — explicit data integrity failure is documented>,
  "contamination_flag_llm": <true or false — contamination or sterility-control risk is described>,
  "investigation_flag_llm": <true or false — explicit failure to investigate or inadequate investigation is described>,
  "evidence_quote": "<verbatim substring from the observation text (6–30 words) that most \
directly supports your severity and root-cause classification>",
  "confidence": <float 0.0–1.0 reflecting overall confidence in the above classifications>
}}

Field rules:
- violation_category: choose the single best fit for the PRIMARY violation domain. \
Definitions:
  * LabControls: laboratory and testing deficiencies — test methods, specifications, \
OOS/OOT investigation procedures, stability testing, calibration or qualification of \
laboratory instruments, sampling plans, reserve samples (21 CFR 211.160–211.194).
  * ProductionControls: manufacturing process deficiencies — batch production and control \
records, manufacturing instructions, in-process testing and controls, yield calculations, \
component handling, charge-in of components, process validation (21 CFR 211.100–211.132).
  * BuildingsEquipment: facility and equipment deficiencies — facility design and \
maintenance, HVAC, utilities, equipment cleaning and sanitization, preventive maintenance, \
calibration of non-laboratory equipment, equipment qualification (21 CFR 211.42–211.68).
  * OrgPersonnel: people and organization deficiencies — training and qualification of \
personnel, responsibilities and independence of the quality control unit, consultant use, \
personnel hygiene (21 CFR 211.22, 211.25–211.34).
  * PackagingLabeling: packaging and labeling deficiencies — label issuance and \
reconciliation, label accuracy, cut label controls, packaging specifications, examination \
of labeled and packaged products (21 CFR 211.122–211.137).
  * RecordsReports: documentation and recordkeeping deficiencies — batch records, \
laboratory records, distribution records, complaint files, annual product review, \
record retention (21 CFR 211.180–211.198).
  * QualitySystem: overall quality management system deficiencies — quality unit \
authority and independence, change control, CAPA system, deviation management, \
supplier/vendor qualification, internal audits. Use this when the failure is in the \
quality management framework itself rather than a specific operational domain.
  * Other: does not clearly fit any of the above domains.

- severity_tier: graded like EU GMP deficiency classification. The tier is decided by \
ONE question: what level of ACTUAL product impact does the text DOCUMENT? \
A deficiency that merely COULD affect product quality is Moderate, no matter how \
serious the system failure sounds. Most 483 observations are Moderate. \
Assign the LOWEST tier that fits.
  * Critical: the text documents that affected product was RELEASED or DISTRIBUTED: \
affected lots were distributed; confirmed OOS product was released; contamination was \
found in released/finished product; sterility failure in released sterile product. \
Anchor examples: "contaminated lots were distributed before the investigation was closed"; \
"batch failing assay specification was released without an investigation".
  * Major: the text documents an ACTUAL defect, failure, or unreliable result found at \
the facility (but no evidence of release); OR a significant systemic failure where the \
risk of an actual product defect is near-certain without immediate correction. \
Confirmed examples: an actual OOS/failing result, contamination or particulates observed \
in product, a failed batch, a product mix-up, falsified or invalidated test data, a \
failed media fill. \
Significant systemic examples: environmental controls have been persistently failing; \
cleaning validation was never performed for a product-contact surface; a sterility-critical \
parameter was not monitored across multiple production runs. \
Anchor examples: "particulate matter was observed in several lots"; \
"test results were invalidated without quality unit approval"; \
"no cleaning validation study has been performed for [active product-contact equipment]".
  * Moderate: the text documents a deficient procedure, system, or practice but NO \
actual product defect or unreliable result: missing or failed validation, inadequate \
or unfollowed procedures, incomplete investigations, environmental monitoring gaps, \
aseptic practice deficiencies without observed contamination, equipment qualification \
gaps, systems that ALLOW data deletion without evidence it occurred. This is the \
DEFAULT tier for most observations. \
Anchor examples: "media fill runs do not include the same number of manual interventions \
as routine production"; "logbook data can be overwritten and original data erased"; \
"cleaning procedures do not specify rinse times or volumes".
  * Minor: documentation or administrative gap with no plausible product impact: \
missing signature, outdated SOP formatting, late record filing. \
Anchor examples: "the SOP index was not updated to reflect the current revision".
  Decision test: released product affected -> Critical; actual defect/failure found \
on site -> Major; deficient system or procedure only -> Moderate; paperwork only -> Minor.

- scope: the breadth of the failure described in THIS observation.
  * SingleBatch = confined to one batch, lot, line event, or single occurrence
  * MultipleProducts = affects several batches, products, or production lines
  * FacilityWide = quality-system-level failure affecting all production (e.g., "there \
are no written procedures for production and process controls" — nothing batch-specific)
  * Unclear = text insufficient to judge breadth

- remediation_signal: Strong = specific corrective actions clearly stated; \
Partial = some corrective intent mentioned; Weak = vague; None = not mentioned

- repeat_flag_llm: mark true ONLY when the observation explicitly states this is \
a repeat observation/finding, previously observed, previously cited, recurring from \
a prior inspection, or equivalent. Do NOT mark true merely because multiple examples \
within the same current observation recur or affect multiple products/lines.

- patient_risk_flag_llm: {patient_risk_rule}

- data_integrity_flag_llm: mark true ONLY for explicit data trustworthiness failures: \
falsification, backdating, deleted or altered records, missing raw data, audit-trail \
problems (disabled/bypassed audit trail, unauthorized system access), unreported OOS \
results, or records reconstructed after the fact. \
Do NOT mark true for ordinary missing SOPs, incomplete documentation, weak \
recordkeeping, or inventory/storage control unless data reliability is directly at issue.

- contamination_flag_llm: mark true for actual contamination OR clear contamination-control \
risk, including sterility assurance failures, aseptic processing deficiencies, environmental \
monitoring failures, microbial/particulate contamination, inadequate cleaning/sterilization, \
or cross-contamination controls. This flag means contamination/sterility-control risk; it \
does NOT necessarily mean confirmed contaminated product.

- investigation_flag_llm: mark true ONLY for an explicit failed, missing, delayed, \
or inadequate investigation of a concrete event, such as a deviation, complaint, \
batch failure, OOS/OOT result, positive unit, contamination event, or particulate \
event. Examples include missing root cause, missing CAPA, or failure to assess \
product impact. Do NOT mark true for general missing evaluation/assessment/rationale \
or because a procedure says an investigation would be required. Do NOT mark true for \
validation/remediation acceptance-criteria weaknesses unless a specific event investigation failed.

- evidence_quote: copy-paste a short exact phrase from the observation text — do NOT \
paraphrase. Prefer 6–30 words and avoid OCR-damaged text when a cleaner exact quote exists.
- confidence: lower if the text is very short, illegible, or ambiguous
"""

OPENAI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "violation_category": {
            "type": "string",
            "enum": sorted(VALID_VIOLATION_CATEGORY),
        },
        "severity_tier": {
            "type": "string",
            "enum": sorted(VALID_SEVERITY_TIER),
            "description": (
                "Tier = documented product impact. Critical: affected product was "
                "released/distributed. Major: an actual defect/failure found on site "
                "(no release) OR a significant systemic failure where actual product "
                "defect is near-certain without correction (e.g., persistent environmental "
                "control failures, no cleaning validation on product-contact surface). "
                "Moderate: deficient procedure/system only, no actual defect documented "
                "(the default for most observations). Minor: paperwork/administrative "
                "gaps. Assign the LOWEST tier that fits."
            ),
        },
        "severity_rationale": {
            "type": "string",
            "description": "Must reference the evidence_quote to justify the tier.",
        },
        "scope": {
            "type": "string",
            "enum": sorted(VALID_SCOPE),
            "description": (
                "Breadth of the failure: SingleBatch (one batch/lot/event), "
                "MultipleProducts (several batches/products/lines), FacilityWide "
                "(quality-system level affecting all production), Unclear."
            ),
        },
        "root_cause_type": {
            "type": "string",
            "enum": sorted(VALID_ROOT_CAUSE_TYPE),
        },
        "root_cause_rationale": {"type": "string"},
        "remediation_signal": {
            "type": "string",
            "enum": sorted(VALID_REMEDIATION),
        },
        "repeat_flag_llm": {
            "type": "boolean",
            "description": (
                "True only when the observation explicitly states this is a repeat "
                "observation or finding, previously observed, previously cited, or "
                "recurring from a prior inspection. False for repeated examples within "
                "the same current observation."
            ),
        },
        "patient_risk_flag_llm": {
            "type": "boolean",
            "description": (
                "True ONLY when an explicit harm pathway exists: sterile/injectable "
                "product with contamination or sterility assurance failure; confirmed "
                "quality defect in released/distributed product; or product released "
                "without required QA disposition. False for generic quality deviations "
                "where harm requires a chain of hypotheticals."
            ),
        },
        "data_integrity_flag_llm": {
            "type": "boolean",
            "description": (
                "True only for explicit data trustworthiness failures: falsification, "
                "backdating, deleted/altered records, missing raw data, disabled/bypassed "
                "audit trail, unauthorized system access, unreported OOS results, or "
                "records reconstructed after the fact. False for missing SOPs, incomplete "
                "documentation, or weak recordkeeping where data reliability is not directly at issue."
            ),
        },
        "contamination_flag_llm": {
            "type": "boolean",
            "description": (
                "True for actual contamination or clear contamination-control risk: "
                "sterility assurance failures, aseptic processing deficiencies, "
                "environmental monitoring failures, microbial or particulate contamination, "
                "inadequate cleaning/sterilization, or cross-contamination controls. "
                "This does not necessarily mean confirmed contaminated product."
            ),
        },
        "investigation_flag_llm": {
            "type": "boolean",
            "description": (
                "True only for an explicit failed, missing, delayed, or inadequate "
                "investigation of a concrete event such as a deviation, complaint, "
                "batch failure, OOS/OOT result, positive unit, contamination event, "
                "or particulate event. Includes missing root cause, missing CAPA, or "
                "failure to assess product impact. False for general missing evaluation "
                "or when investigation is only a procedure requirement. False for "
                "validation/remediation acceptance-criteria weaknesses unless a specific "
                "event investigation failed."
            ),
        },
        "evidence_quote": {
            "type": "string",
            "description": (
                "A short exact quote copied from the observation text, preferably 6-30 "
                "words. Do not paraphrase."
            ),
        },
        "confidence": {"type": "number"},
    },
    "required": [
        "violation_category", "severity_tier", "severity_rationale", "scope",
        "root_cause_type", "root_cause_rationale", "remediation_signal",
        "repeat_flag_llm", "patient_risk_flag_llm",
        "data_integrity_flag_llm", "contamination_flag_llm",
        "investigation_flag_llm",
        "evidence_quote", "confidence",
    ],
    "additionalProperties": False,
}


# ── Anthropic tool definition ───────────────────────────────────────────────
# Tool use with tool_choice={"type":"tool"} is the Anthropic equivalent of
# OpenAI structured outputs — the model MUST call the tool and the input is
# validated against input_schema server-side.
# Uses a copy of OPENAI_JSON_SCHEMA with a stricter patient_risk description,
# and adds cache_control so the tool definition is cached across calls.
_ANTHROPIC_SCHEMA = copy.deepcopy(OPENAI_JSON_SCHEMA)
_ANTHROPIC_SCHEMA["properties"]["patient_risk_flag_llm"]["description"] = (
    "True ONLY for: (a) sterile/injectable with CONFIRMED contamination or sterility breach; "
    "(b) confirmed quality defect in already-released/distributed product; "
    "(c) product explicitly released without required QA disposition. "
    "ALWAYS false for oral solid dose forms, missing SOPs, monitoring gaps without confirmed "
    "contamination, equipment gaps without product impact, DI issues without release, "
    "general QS failures, training deficiencies, stability gaps. "
    "If no patient is at risk RIGHT NOW from a released product, return false."
)

ANTHROPIC_TOOL = {
    "name": "extract_483_signals",
    "description": (
        "Extract structured risk signals from an FDA Form 483 pharmaceutical "
        "manufacturing observation. Return all required fields."
    ),
    "input_schema": _ANTHROPIC_SCHEMA,
    "cache_control": {"type": "ephemeral"},  # cache tool definition across calls
}

# ── Anthropic prompt: rules-first layout for prompt caching ─────────────────
# Observation text is placed at the END so everything before it (all field rules)
# is eligible for caching. cache_control is applied to this fixed block;
# only the variable part (CFR codes + obs text) is billed at full input price.
_ANTHROPIC_PROMPT_FIXED = """\
You are analyzing FDA Form 483 Inspectional Observation text from a pharmaceutical \
manufacturing inspection.

FDA Form 483 observations are written by FDA investigators to document specific \
violations or deficiencies found at a drug manufacturing facility. Each observation \
typically cites one or more sections of Title 21 CFR (Code of Federal Regulations).

Call the extract_483_signals tool with your analysis. Apply each rule exactly:

- violation_category: choose the single best fit for the PRIMARY violation domain.
  * LabControls: laboratory and testing deficiencies — test methods, specifications, \
OOS/OOT investigation procedures, stability testing, calibration or qualification of \
laboratory instruments, sampling plans, reserve samples (21 CFR 211.160–211.194).
  * ProductionControls: manufacturing process deficiencies — batch production and control \
records, manufacturing instructions, in-process testing and controls, yield calculations, \
component handling, charge-in of components, process validation (21 CFR 211.100–211.132).
  * BuildingsEquipment: facility and equipment deficiencies — facility design and \
maintenance, HVAC, utilities, equipment cleaning and sanitization, preventive maintenance, \
calibration of non-laboratory equipment, equipment qualification (21 CFR 211.42–211.68).
  * OrgPersonnel: people and organization deficiencies — training and qualification of \
personnel, responsibilities and independence of the quality control unit, consultant use, \
personnel hygiene (21 CFR 211.22, 211.25–211.34).
  * PackagingLabeling: packaging and labeling deficiencies — label issuance and \
reconciliation, label accuracy, cut label controls, packaging specifications, examination \
of labeled and packaged products (21 CFR 211.122–211.137).
  * RecordsReports: documentation and recordkeeping deficiencies — batch records, \
laboratory records, distribution records, complaint files, annual product review, \
record retention (21 CFR 211.180–211.198).
  * QualitySystem: overall quality management system deficiencies — quality unit \
authority and independence, change control, CAPA system, deviation management, \
supplier/vendor qualification, internal audits. Use this when the failure is in the \
quality management framework itself rather than a specific operational domain.
  * Other: does not clearly fit any of the above domains.

- severity_tier: graded like EU GMP deficiency classification. The tier is decided by \
ONE question: what level of ACTUAL product impact does the text DOCUMENT? \
A deficiency that merely COULD affect product quality is Moderate, no matter how \
serious the system failure sounds. Most 483 observations are Moderate. \
Assign the LOWEST tier that fits.
  * Critical: the text documents that affected product was RELEASED or DISTRIBUTED.
  * Major: the text documents an ACTUAL defect, failure, or unreliable result found at \
the facility (but no evidence of release); OR a significant systemic failure where the \
risk of an actual product defect is near-certain without immediate correction. \
Confirmed examples: an actual OOS/failing result, contamination or particulates observed \
in product, a failed batch, a product mix-up, falsified or invalidated test data, a \
failed media fill. \
Significant systemic examples: environmental controls have been persistently failing; \
cleaning validation was never performed for a product-contact surface; a sterility-critical \
parameter was not monitored across multiple production runs.
  * Moderate: the text documents a deficient procedure, system, or practice but NO \
actual product defect or unreliable result. This is the DEFAULT tier for most observations.
  * Minor: documentation or administrative gap with no plausible product impact.
  Decision test: released product affected → Critical; actual defect/failure found \
on site → Major; deficient system or procedure only → Moderate; paperwork only → Minor.

- scope:
  * SingleBatch = confined to one batch, lot, line event, or single occurrence
  * MultipleProducts = affects several batches, products, or production lines
  * FacilityWide = quality-system-level failure affecting all production
  * Unclear = text insufficient to judge breadth

- root_cause_type: why did this failure occur? Choose based on the mechanism described, \
not on severity. Capital and Cultural failures can both be Minor or Critical.
  * Capital = deficiency caused by missing, inadequate, or broken equipment, facilities, \
or designed procedures (SOPs). Examples: no procedure exists; equipment was never qualified \
or validated; the SOP was inadequate; the facility design did not support contamination \
control; insufficient resources were allocated to build the system.
  * Cultural = deficiency caused by people not following or enforcing existing procedures — \
a behavioral, management, or organizational failure. Examples: trained personnel did not \
follow the SOP; management accepted non-compliance; data was altered or not recorded \
contemporaneously; deviations were not reported; quality unit did not exercise oversight.
  * Mixed = the text documents clear evidence of BOTH a design/resource gap AND a \
behavioral/management failure contributing to the same observation.
  * Unclear = the text is insufficient to distinguish whether the root cause is a gap \
in the system design or a gap in execution.
  Decision tip: if the SOP exists but was ignored → Cultural; if the SOP does not exist \
or is inadequate → Capital; if both are stated → Mixed.

- root_cause_rationale: 1–2 sentence justification for root_cause_type. Cite the specific \
text that drove the assignment. If Unclear, state what information is missing.

- remediation_signal: Strong = specific corrective actions clearly stated; \
Partial = some corrective intent mentioned; Weak = vague; None = not mentioned

- repeat_flag_llm: mark true ONLY when the observation explicitly states this is \
a repeat observation/finding from a prior inspection. False if examples merely recur \
within the same current observation.

- patient_risk_flag_llm: mark true ONLY for these three scenarios — nothing else qualifies:
  (a) Sterile or injectable product with CONFIRMED contamination or sterility breach \
documented in the observation.
  (b) A quality defect (OOS result, mix-up, wrong potency, mislabeling) confirmed in \
product that was ALREADY released or distributed to patients.
  (c) The text explicitly states product was released without required QA testing or \
disposition.
  ALWAYS mark false for: oral solid dose forms (tablets, capsules, powders); missing \
SOPs or documentation gaps; environmental monitoring gaps without confirmed contamination; \
equipment validation gaps without confirmed product impact; data integrity issues without \
confirmed release of affected product; any general quality system failures; training or \
personnel qualification deficiencies; stability testing gaps; specification issues without \
a released OOS result.
  Rule of thumb: if a patient is not already at risk RIGHT NOW from something the \
facility already released, mark false.
  IMPORTANT — ignore CFR boilerplate: 483 observations often open with standard \
regulatory language such as "whether or not the batch has already been distributed" \
or "that would alter the safety, identity, strength, quality or purity of the drug \
product." This is boilerplate CFR citation text, NOT evidence of actual patient risk. \
Do not trigger patient_risk based on this preamble language alone.

- data_integrity_flag_llm: mark true ONLY for explicit data trustworthiness failures: \
falsification, backdating, deleted or altered records, missing raw data, audit-trail \
problems (disabled/bypassed audit trail, unauthorized system access), unreported OOS \
results, or records reconstructed after the fact. \
  Do NOT mark true for ordinary missing SOPs, incomplete documentation, weak \
recordkeeping, or inventory/storage control unless data reliability is directly at issue.

- contamination_flag_llm: mark true for actual contamination OR clear contamination-control \
risk: sterility assurance failures, aseptic processing deficiencies, environmental \
monitoring failures, microbial/particulate contamination, inadequate cleaning/sterilization.

- investigation_flag_llm: mark true ONLY for an explicit failed, missing, delayed, \
or inadequate investigation of a concrete event (deviation, OOS/OOT, contamination event, \
batch failure). False for general missing evaluation or weak procedure requirements.

- evidence_quote: copy-paste a short exact phrase from the observation text (6–30 words). \
Do NOT paraphrase.
- confidence: lower if the text is very short, illegible, or ambiguous.

EDGE CASE GUIDANCE:

Severity calibration — common errors to avoid:
  (1) A procedure that "lacks" or "does not include" required testing = Moderate, not Major. \
Severity requires the text to document an actual failure event, not just a missing control.
  (2) "Multiple batches" with inadequate controls = still Moderate unless a specific batch \
was found defective. The NUMBER of affected batches does not elevate severity on its own.
  (3) Cleaning validation "was not performed" for a sterile product-contact surface is \
Major because the risk of an actual sterility failure is near-certain — this is the systemic \
exception where severity elevates without a confirmed defect.
  (4) Environmental monitoring trending out of alert levels = Major if confirmed contamination \
found; Moderate if only the TREND is adverse but no confirmed contamination.

Root cause calibration — common errors to avoid:
  (1) If an SOP EXISTED but personnel bypassed or failed to follow it → Cultural. Do not \
assign Capital just because the observation describes a gap in practice.
  (2) If the SOP itself is described as inadequate (e.g., "the SOP does not specify \
acceptance criteria") → Capital, even if personnel were trying to follow it.
  (3) A data integrity failure where management was aware and did not act → Cultural, not \
Mixed (management inaction is a cultural failure, not a capital gap).

Scope calibration:
  (1) An observation describing failures across all production batches reviewed = FacilityWide \
only if the failure is in the quality management system itself (e.g., CAPA program, deviation \
management). If it is an operational failure affecting many batches = MultipleProducts.
  (2) SingleBatch is correct even when multiple examples within the same batch are cited.

Analyze the following observation:\
"""

_ANTHROPIC_CACHE_STATS: dict = {"calls": 0, "write": 0, "read": 0}

_ANTHROPIC_PROMPT_VARIABLE_TEMPLATE = """\

CFR codes cited in this observation: {cfr_codes}

Observation text:
---
{obs_text_clean}
---\
"""


def _build_prompt(obs_text_clean: str, cfr_codes, provider: str = "openai") -> str:
    """Build the full prompt string (used by OpenAI; Anthropic uses split parts)."""
    cfr_str = str(cfr_codes).strip() if pd.notna(cfr_codes) and str(cfr_codes).strip() else "not specified"
    rule = _PATIENT_RISK_RULE_ANTHROPIC if provider == "anthropic" else _PATIENT_RISK_RULE_OPENAI
    return _PROMPT_TEMPLATE.format(
        obs_text_clean=obs_text_clean.strip(),
        cfr_codes=cfr_str,
        patient_risk_rule=rule,
    )


def _build_anthropic_prompt_parts(obs_text_clean: str, cfr_codes) -> tuple[str, str]:
    """Return (fixed_cacheable, variable) prompt parts for Anthropic caching."""
    cfr_str = str(cfr_codes).strip() if pd.notna(cfr_codes) and str(cfr_codes).strip() else "not specified"
    variable = _ANTHROPIC_PROMPT_VARIABLE_TEMPLATE.format(
        obs_text_clean=obs_text_clean.strip(),
        cfr_codes=cfr_str,
    )
    return _ANTHROPIC_PROMPT_FIXED, variable


def _parse_response(text: str) -> dict:
    """Extract JSON dict from raw API response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop opening fence (and optional language tag) and closing fence
        inner_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner_lines.append(line)
        text = "\n".join(inner_lines)
    return json.loads(text)


def _coerce_categorical(value, valid_set: set, fallback: str) -> str:
    if value in valid_set:
        return value
    # Try case-insensitive match
    fixed = next((v for v in valid_set if v.lower() == str(value).lower()), None)
    return fixed if fixed else fallback


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")


def _validate(result: dict, obs_text_clean: str) -> dict:
    """Coerce types, validate categoricals, apply evidence guard."""
    result["violation_category"] = _coerce_categorical(
        result.get("violation_category", ""), VALID_VIOLATION_CATEGORY, "Other"
    )
    result["severity_tier"] = _coerce_categorical(
        result.get("severity_tier", ""), VALID_SEVERITY_TIER, "Minor"
    )
    result["scope"] = _coerce_categorical(
        result.get("scope", ""), VALID_SCOPE, "Unclear"
    )
    result["root_cause_type"] = _coerce_categorical(
        result.get("root_cause_type", ""), VALID_ROOT_CAUSE_TYPE, "Unclear"
    )
    result["remediation_signal"] = _coerce_categorical(
        result.get("remediation_signal", ""), VALID_REMEDIATION, "None"
    )
    result["data_integrity_flag_llm"] = _coerce_bool(result.get("data_integrity_flag_llm", False))

    for flag in LLM_FLAG_FIELDS:
        result[flag] = _coerce_bool(result.get(flag, False))

    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    except (ValueError, TypeError):
        result["confidence"] = 0.5

    # Evidence guard: quote must appear verbatim in observation text
    quote = str(result.get("evidence_quote", "")).strip()
    if quote and quote not in obs_text_clean:
        # Normalise whitespace and retry
        q_norm = " ".join(quote.split())
        t_norm = " ".join(obs_text_clean.split())
        if q_norm not in t_norm:
            result["evidence_quote"] = ""   # failed guard — clear it

    return result


def _get_response_text(response) -> str:
    """Extract text from an OpenAI Responses API object."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _response_debug(response) -> str:
    """Small response summary for failed rows."""
    parts = []
    for attr in ["id", "status", "incomplete_details"]:
        value = getattr(response, attr, None)
        if value:
            parts.append(f"{attr}={value}")
    text = _get_response_text(response)
    if text:
        parts.append(f"text={text[:160]}")
    return " | ".join(parts)[:300]


def _call_openai(client, obs_row: pd.Series) -> tuple[dict, str, str]:
    """
    Call the OpenAI API for one observation row.
    Returns (llm_result_dict, extraction_status, extraction_error).
    """
    obs_text = str(obs_row.get("obs_text_clean") or obs_row.get("obs_text") or "").strip()
    if len(obs_text) < 30:
        return {}, "skipped_short", "obs_text_clean too short (<30 chars)"

    prompt = _build_prompt(obs_text, obs_row.get("cfr_codes", ""), provider="openai")

    try:
        response = None
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You extract structured risk signals from FDA Form 483 "
                                "observations. Return only schema-valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=MAX_TOKENS,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "form_483_observation_signal",
                            "strict": True,
                            "schema": OPENAI_JSON_SCHEMA,
                        }
                    },
                )
                break
            except Exception as exc:
                if "RateLimit" in type(exc).__name__ and attempt < RATE_LIMIT_RETRIES:
                    time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
                    continue
                raise
        response_text = _get_response_text(response).strip()
        if not response_text:
            return {}, "empty_response", _response_debug(response)

        result = _parse_response(response_text)
        result = _validate(result, obs_text)

        missing = {
            "violation_category", "severity_tier", "root_cause_type",
            "remediation_signal", "evidence_quote", "confidence",
        } - set(result.keys())

        status = "partial" if missing else "ok"
        error  = f"missing fields: {missing}" if missing else ""
        return result, status, error

    except json.JSONDecodeError as exc:
        preview = response_text[:160] if "response_text" in locals() else ""
        return {}, "json_error", f"{str(exc)[:120]} | text={preview}"
    except Exception as exc:
        exc_type = type(exc).__name__
        if "RateLimit" in exc_type:
            return {}, "rate_limit", "rate limited after retries — rerun to rescore"
        return {}, "api_error", f"{exc_type}: {str(exc)[:200]}"


def _call_anthropic(client, obs_row: pd.Series) -> tuple[dict, str, str]:
    """
    Call the Anthropic API for one observation row using forced tool use + prompt caching.

    Caching strategy (cuts input cost ~90% after first call):
      - System message: cached (same every call)
      - Tool definition (ANTHROPIC_TOOL): cached via cache_control on the tool
      - Fixed prompt block (_ANTHROPIC_PROMPT_FIXED): cached (all field rules)
      - Variable block (CFR codes + obs text): NOT cached (unique per observation)

    With tool_choice forced, block.input is already a validated dict — no JSON parsing.
    """
    obs_text = str(obs_row.get("obs_text_clean") or obs_row.get("obs_text") or "").strip()
    if len(obs_text) < 30:
        return {}, "skipped_short", "obs_text too short (<30 chars)"

    fixed_prompt, variable_prompt = _build_anthropic_prompt_parts(
        obs_text, obs_row.get("cfr_codes", "")
    )

    try:
        response = None
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=MAX_TOKENS,
                    system=[{
                        "type": "text",
                        "text": (
                            "You extract structured risk signals from FDA Form 483 "
                            "observations. Use the extract_483_signals tool to return "
                            "your analysis."
                        ),
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": fixed_prompt,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": variable_prompt,
                            },
                        ],
                    }],
                    tools=[ANTHROPIC_TOOL],
                    tool_choice={"type": "tool", "name": "extract_483_signals"},
                )
                break
            except Exception as exc:
                if "RateLimit" in type(exc).__name__ and attempt < RATE_LIMIT_RETRIES:
                    time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
                    continue
                raise

        usage = response.usage
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read  = getattr(usage, "cache_read_input_tokens", 0) or 0
        _ANTHROPIC_CACHE_STATS["write"] += cache_write
        _ANTHROPIC_CACHE_STATS["read"]  += cache_read
        _ANTHROPIC_CACHE_STATS["calls"] += 1

        result = None
        for block in response.content:
            if block.type == "tool_use":
                result = block.input  # already a validated dict — no JSON parsing needed
                break

        if result is None:
            return {}, "empty_response", "no tool_use block in Anthropic response"

        result = _validate(result, obs_text)

        missing = {
            "violation_category", "severity_tier", "root_cause_type",
            "remediation_signal", "evidence_quote", "confidence",
        } - set(result.keys())

        status = "partial" if missing else "ok"
        error  = f"missing fields: {missing}" if missing else ""
        return result, status, error

    except Exception as exc:
        exc_type = type(exc).__name__
        if "RateLimit" in exc_type:
            return {}, "rate_limit", "rate limited after retries — rerun to rescore"
        return {}, "api_error", f"{exc_type}: {str(exc)[:200]}"


def _build_row(obs_row: pd.Series, llm: dict, status: str, error: str) -> dict:
    """Assemble one output row from obs metadata + renamed regex flags + LLM fields."""
    row: dict = {
        # Stable join keys
        "fei":            obs_row["fei"],
        "filename":       obs_row.get("filename", ""),
        "insp_date":      obs_row.get("insp_date", ""),
        "obs_num":        obs_row["obs_num"],
        # Source text / metadata
        "obs_text_clean": obs_row.get("obs_text_clean") or obs_row.get("obs_text", ""),
        "cfr_codes":      obs_row.get("cfr_codes", ""),
        "n_cfrs":         obs_row.get("n_cfrs", 0),
        "n_examples":     obs_row.get("n_examples", 0),
    }

    # Renamed regex flags (baseline comparison)
    for src_col, dst_col in REGEX_FLAG_MAP.items():
        row[dst_col] = bool(obs_row.get(src_col, False))

    # LLM fields (None when extraction failed)
    for field in [
        "violation_category", "severity_tier", "severity_rationale", "scope",
        "root_cause_type", "root_cause_rationale", "remediation_signal",
        "data_integrity_flag_llm",
        *LLM_FLAG_FIELDS,
        "evidence_quote", "confidence",
    ]:
        row[field] = llm.get(field, None)

    row["model_name"]        = ANTHROPIC_MODEL if PROVIDER == "anthropic" else MODEL_NAME
    row["extraction_status"] = status
    row["extraction_error"]  = error
    return row


def _save(rows: list[dict], path: Path) -> None:
    if rows:
        df = pd.DataFrame(rows)
        n_before = len(df)
        df = df.drop_duplicates(subset=["fei", "insp_date", "obs_num"]).reset_index(drop=True)
        if len(df) < n_before:
            print(f"  Dropped {n_before - len(df)} duplicate (fei, insp_date, obs_num) rows before saving")
        df.to_csv(path, index=False)


# %%
# ── Interactive / CLI configuration ────────────────────────────────────────
# For line-by-line execution in an IDE, edit these values before running cells.
# CLI arguments override these values only when the file itself is run directly.
DRY_RUN = True       # Safe default for interactive work: no API calls
LIMIT   = None       # e.g., 5
FEI     = None       # e.g., 3002808406
FORCE   = False      # Re-score rows already in SIGNALS_CSV
SAMPLE  = None       # e.g., 50 — stratified sample to separate output file

try:
    _THIS_FILE = Path(__file__).resolve()
except NameError:
    _THIS_FILE = None

_RUNNING_AS_SCRIPT = (
    _THIS_FILE is not None
    and len(sys.argv) > 0
    and Path(sys.argv[0]).resolve() == _THIS_FILE
)

if _RUNNING_AS_SCRIPT:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LLM context signals from 483 observations.\n"
            "Reads 483_observations.csv; writes 483_observation_context_signals.csv."
        )
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts and cost estimate; no API calls.")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only N pending observations (testing).")
    parser.add_argument("--fei",     type=int, default=None,
                        help="Process only observations for a single FEI (testing).")
    parser.add_argument("--force",   action="store_true",
                        help="Re-score observations already in the output file.")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Stratified sample of N observations to a separate "
                             "output file (prompt validation).")
    parser.add_argument("--source",   choices=["pdf", "redica"], default="pdf",
                        help="Input source: 'pdf' (default, 483_observations.csv) or "
                             "'redica' (redica_483_observations.csv).")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                        help="LLM provider: 'openai' (default, gpt-5-mini) or "
                             "'anthropic' (claude-haiku-4-5-20251001).")
    parser.add_argument("--output",   type=str, default=None,
                        help="Override output CSV path (default: source-dependent).")
    args = parser.parse_args()
    DRY_RUN  = args.dry_run
    LIMIT    = args.limit
    FEI      = args.fei
    FORCE    = args.force
    SAMPLE   = args.sample
    SOURCE   = args.source
    PROVIDER = args.provider
    _output_override = Path(args.output) if args.output else None
else:
    _output_override = None

# ── Apply source-dependent paths ──────────────────────────────────────────
if SOURCE == "redica":
    OBS_CSV = _REDICA_OBS_CSV
    SIGNALS_CSV = (
        _REDICA_SIGNALS_CSV_ANTHROPIC if PROVIDER == "anthropic"
        else _REDICA_SIGNALS_CSV_OPENAI
    )
# pdf source keeps the defaults set above

if SAMPLE:
    SIGNALS_CSV = HERE / f"483_observation_context_signals_sample{SAMPLE}.csv"

# --output overrides everything above
if _output_override:
    SIGNALS_CSV = _output_override


# %%
# ── Load observations ──────────────────────────────────────────────────────
print("=" * 70)
print("01_extract_observation_signals.py")
print("FDA Form 483 -> LLM Context Signal Extraction")
print("=" * 70)
print(f"Mode                 : {'CLI' if _RUNNING_AS_SCRIPT else 'interactive'}")
print(f"Dry run              : {DRY_RUN}")
print(f"Limit                : {LIMIT}")
print(f"FEI filter           : {FEI}")
print(f"Force re-score       : {FORCE}")

if not OBS_CSV.exists():
    raise FileNotFoundError(f"Observations CSV not found:\n  {OBS_CSV}")

obs_df = pd.read_csv(OBS_CSV)
print(f"Observations loaded   : {len(obs_df):,}  ({obs_df['fei'].nunique()} FEIs)")

if FEI is not None:
    obs_df = obs_df[obs_df["fei"] == FEI]
    print(f"Filtered to FEI {FEI} : {len(obs_df)} observations")
    if obs_df.empty:
        raise ValueError(f"No observations found for FEI {FEI}")


# %%
# ── Idempotency: load already-scored rows ──────────────────────────────────
already_scored: set[tuple] = set()
existing_rows:  list[dict] = []

if SIGNALS_CSV.exists() and not FORCE and not SAMPLE:
    existing_df = pd.read_csv(SIGNALS_CSV)
    # Keep only successfully scored rows; failed rows are dropped so they
    # get rescored on this run.
    ok_df  = existing_df[existing_df["extraction_status"].isin(["ok", "partial"])]
    n_fail = len(existing_df) - len(ok_df)
    existing_rows = ok_df.to_dict("records")
    for _, r in ok_df.iterrows():
        key = (r["fei"], r.get("insp_date", r.get("filename", "")), r["obs_num"]) \
              if SOURCE == "redica" else (r["fei"], r.get("filename", ""), r["obs_num"])
        already_scored.add(key)
    print(f"Already scored        : {len(already_scored):,}  "
          f"(set FORCE=True or use --force to re-score)")
    if n_fail:
        print(f"Failed rows to rescore: {n_fail}")


# %%
# ── Determine pending observations ────────────────────────────────────────
if SAMPLE:
    # Stratified sample: round-robin one observation per FEI (shuffled within
    # FEI, seed=7) until N reached. Caps single-facility dominance and covers
    # the maximum number of FEIs. Fresh run every time; never resumes.
    rng_seed = 7
    shuffled = obs_df.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    by_fei = {fei_val: grp.reset_index(drop=True)
              for fei_val, grp in shuffled.groupby("fei")}
    picked_idx: list[pd.Series] = []
    depth = 0
    while len(picked_idx) < SAMPLE and depth < max(len(g) for g in by_fei.values()):
        for fei_val in sorted(by_fei.keys()):
            grp = by_fei[fei_val]
            if depth < len(grp) and len(picked_idx) < SAMPLE:
                picked_idx.append(grp.iloc[depth])
        depth += 1
    to_process = pd.DataFrame(picked_idx).reset_index(drop=True)
    existing_rows = []
    print(f"Stratified sample     : {len(to_process)} observations "
          f"from {to_process['fei'].nunique()} FEIs (round-robin, seed={rng_seed})")
elif FORCE:
    to_process = obs_df.copy()
    existing_rows = []
else:
    mask = obs_df.apply(
        lambda r: (
            (r["fei"], r.get("insp_date", ""), r["obs_num"])
            if SOURCE == "redica"
            else (r["fei"], r.get("filename", ""), r["obs_num"])
        ) not in already_scored,
        axis=1,
    )
    to_process = obs_df[mask].copy()

if LIMIT and not SAMPLE:
    to_process = to_process.head(LIMIT)

print(f"Pending to process    : {len(to_process):,}")


# %%
# ── Dry run ────────────────────────────────────────────────────────────────
_active_model = ANTHROPIC_MODEL if PROVIDER == "anthropic" else MODEL_NAME
# Cost per 1M tokens (input+output blended rough estimate)
_cost_per_m = 2.0 if PROVIDER == "anthropic" else 3.0  # Haiku ~$2/M blended

if DRY_RUN:
    avg_tokens   = 850    # rough estimate per observation (prompt + response)
    total_tokens = len(to_process) * avg_tokens
    cost_usd     = total_tokens / 1_000_000 * _cost_per_m
    print("\n[DRY RUN] No API calls made.")
    print(f"  Provider           : {PROVIDER}")
    print(f"  Model              : {_active_model}")
    print(f"  Observations       : {len(to_process)}")
    print(f"  Estimated tokens   : ~{total_tokens:,}")
    print(f"  Estimated cost     : ~${cost_usd:.2f} USD")
    print(f"  Output             : {SIGNALS_CSV}")


# %%
# ── Initialize client (OpenAI or Anthropic) ────────────────────────────────
client   = None
_call_fn = _call_openai  # dispatch function — set based on provider below

if not DRY_RUN and len(to_process) > 0:
    if PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "[ERROR] ANTHROPIC_API_KEY is not set.\n"
                "  export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        client   = Anthropic(api_key=api_key)
        _call_fn = _call_anthropic
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "[ERROR] OPENAI_API_KEY is not set.\n"
                "  export OPENAI_API_KEY='sk-...'"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        client   = OpenAI(api_key=api_key)
        _call_fn = _call_openai
elif not DRY_RUN:
    print("Nothing new to process. Set FORCE=True or use --force to re-score all.")


# %%
# ── Main extraction loop ───────────────────────────────────────────────────
new_rows:  list[dict] = []
n_ok = n_partial = n_error = n_skipped = 0
total = len(to_process)

if not DRY_RUN and client is not None:
    for i, (_, obs_row) in enumerate(to_process.iterrows(), 1):
        fei_id   = obs_row["fei"]
        obs_num  = obs_row["obs_num"]
        filename = str(obs_row.get("filename", ""))

        print(f"[{i:4d}/{total}] FEI {fei_id}  obs {obs_num:3d}  ", end="", flush=True)

        llm_result, status, error = _call_fn(client, obs_row)
        out_row = _build_row(obs_row, llm_result, status, error)
        new_rows.append(out_row)

        if status == "ok":
            n_ok += 1
            print(
                f"ok   conf={out_row.get('confidence') or 0:.2f}"
                f"  sev={out_row.get('severity_tier') or '?'}"
                f"  rc={out_row.get('root_cause_type') or '?'}"
            )
        elif status == "partial":
            n_partial += 1
            print(f"PARTIAL  {error[:60]}")
        elif status.startswith("skipped"):
            n_skipped += 1
            print(f"SKIP  {error}")
        else:
            n_error += 1
            print(f"ERROR  {status}: {error[:70]}")

        # Periodic checkpoint save
        if i % SAVE_EVERY == 0:
            combined = existing_rows + new_rows
            _save(combined, SIGNALS_CSV)
            print(f"  [checkpoint] {len(combined)} rows saved to {SIGNALS_CSV.name}")


# %%
# ── Final save ─────────────────────────────────────────────────────────────
if not DRY_RUN and client is not None:
    combined = existing_rows + new_rows
    _save(combined, SIGNALS_CSV)

    print()
    print("=" * 70)
    print(f"DONE  —  {total} processed  "
          f"(ok: {n_ok}  partial: {n_partial}  "
          f"errors: {n_error}  skipped: {n_skipped})")
    print(f"Total rows in output  : {len(combined)}")
    print(f"Output                : {SIGNALS_CSV}")
    if PROVIDER == "anthropic" and _ANTHROPIC_CACHE_STATS["calls"] > 0:
        s = _ANTHROPIC_CACHE_STATS
        pct = 100 * s["read"] / max(s["read"] + s["write"], 1)
        print()
        print(f"Anthropic cache stats : {s['calls']} calls  "
              f"write={s['write']:,} tok  read={s['read']:,} tok  "
              f"({pct:.0f}% cache hit rate)")
        if s["read"] == 0 and s["calls"] > 1:
            print("  ⚠ No cache hits — prompt prefix may be below the 4,096-token minimum.")
    print()
    print("Next step: python 02_aggregate_fei_features.py")
