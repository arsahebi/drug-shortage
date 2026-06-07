# LLM and Regex Flag Definitions

This note defines the observation-level flags used by
`04_extract_observation_signals.py`.

The flags are **research features**, not adjudicated legal findings. They should
be interpreted as text-derived signals from FDA Form 483 observations.

## Evidence Layers

### Regex flags

Columns ending in `_regex` come from deterministic keyword/regex rules in
`Data/12 - FDA - 483/processed/483_observations.csv`.

Examples:

- `has_repeat_regex`
- `has_investigation_regex`
- `has_contamination_regex`

These are baseline text flags. They are transparent and reproducible, but they
can miss context or over-count keywords.

### LLM flags

Columns ending in `_llm` come from the OpenAI structured extraction in
`04_extract_observation_signals.py`.

Examples:

- `repeat_flag_llm`
- `investigation_flag_llm`
- `contamination_flag_llm`

These are semantic/context-aware flags. They can use context, but they should be
audited and compared against regex and CFR/domain features.

## LLM Flag Definitions

### `repeat_flag_llm`

True only when the observation explicitly states this is a repeat observation or
repeat finding, was previously observed/cited, recurred from a prior inspection,
or uses equivalent language.

Not true merely because multiple examples appear within the same current
observation or because the same issue affects multiple products, lots, lines, or
rooms.

### `systemic_flag_llm`

True when the observation describes facility-wide, multi-process, multi-product,
multi-line, multi-department, or quality-system-level failure.

This can include repeated failures within a current inspection if the text
supports a broader system breakdown.

### `patient_risk_flag_llm`

True when the violation could directly affect patient safety, such as risk of
non-sterile product, contaminated product, sub/super-potent drug, defective
injectable product, or release of product without adequate quality assurance.

This does not require confirmed patient harm.

### `data_integrity_flag_llm`

True only for explicit data trustworthiness failures, such as unreliable data,
falsification, backdating, deleted/altered records, missing raw data, audit-trail
problems, unreported results, invalidated results without justification, or
records that cannot be trusted.

Not true for ordinary missing SOPs, incomplete documentation, weak recordkeeping,
inventory location/mix-up control, or storage control unless the observation
directly raises data reliability or trustworthiness.

### `contamination_flag_llm`

True for actual contamination or clear contamination-control risk.

Includes sterility assurance failures, aseptic processing deficiencies,
environmental monitoring failures, microbial contamination, particulate
contamination, inadequate cleaning/sterilization, or cross-contamination control
failures.

This flag means **contamination or sterility-control risk**. It does not
necessarily mean FDA confirmed contaminated product.

### `documentation_flag_llm`

True when inadequate documentation is a central finding: missing/inadequate SOPs,
missing required records, incomplete records, records not approved/reviewed, or
procedures that do not reflect actual practice.

Not true when documentation is only incidental to a different primary problem.

### `investigation_flag_llm`

True only for an explicit failed, missing, delayed, or inadequate investigation
of a concrete event, such as a deviation, complaint, batch failure, OOS/OOT
result, positive unit, contamination event, or particulate event.

Examples include missing root cause, missing CAPA, failure to assess product
impact, failure to investigate OOS/OOT results, or inadequate complaint/deviation
investigations.

Not true for general missing evaluation, missing assessment, missing rationale,
or because a procedure says an investigation would be required. Not true for
validation/remediation acceptance-criteria weaknesses unless a specific event
investigation failed.

## Research Use

Recommended analysis structure:

- Layer 1: CFR/domain features
- Layer 2: regex baseline flags
- Layer 3: LLM semantic flags
- Layer 4: agreement/disagreement between regex and LLM flags
- Layer 5: FEI-level aggregate indices

The LLM layer should be evaluated as incremental information beyond CFR/domain
and regex features, not as unquestioned ground truth.
