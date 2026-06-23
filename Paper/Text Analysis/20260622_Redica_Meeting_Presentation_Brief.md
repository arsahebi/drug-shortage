# Presentation Brief — Redica Collaboration Meeting
**Audience:** Redica technical team (feature engineers + domain experts)
**Duration:** 60 minutes
**Tone:** Collaborative — we explain our rationale, they explain theirs, we find alignment
**Handouts:** `20260611_483_LLM_Prompts_Expert_Review.docx` (verbatim prompt rules) · `20260616_Redica_Classification_Comparison.docx` (field-by-field validation numbers)

---

## Slide 1 — Title
**Title:** Understanding Drug Manufacturing Quality from 483 Observations: A Collaborative Review
**Subtitle:** Drug Shortage Research Team × Redica Systems — June 2026
**Visual:** Clean title slide, no bullet points

---

## Slide 2 — Why we care about 483 observations
**Title:** Our research question: can manufacturing quality signals predict drug shortages?

**Context (3 bullets):**
- Drug shortages disproportionately affect generic drugs — often linked to manufacturing quality failures
- FDA Form 483 observations are the most granular public signal of quality breakdowns at a specific facility
- We are building a facility-level (FEI) risk model: can 483 signals predict recalls, adverse events, and shortages before they happen?

**The challenge:**
483 observations are free text — to use them as model features, we need to convert them into structured dimensions

**Speaker note:** We are exploring multiple prediction targets: Class I/II drug recalls, FDA adverse event signals (FAERS), and confirmed drug shortages. Each represents a different severity level of manufacturing failure. This is the motivation for why we built a classification system — we needed specific, structured signals for a predictive model.

---

## Slide 3 — Our data universe
**Title:** 127 manufacturing facilities, ~853 known 483s — and a data gap we want to close

**What Redica's event log tells us (Red Flag Events file):**
- **~829–853** total inspections where a 483 was issued across our 127 facilities (lifetime, 1998–2026)
- **~549** of those are pre-2018 → no documents shared
- **~280** are post-2018 → **246 Form 483 documents** actually shared (88% of post-2018 events)
- ~34 post-2018 483 inspections exist in Redica's event log but documents were not included

**Also shared:** 17 Warning Letter deficiency records (same file, excluded from LLM extraction)

**PDF pipeline (FDA Dashboard):**
- We also downloaded 82 PDFs from the FDA Dashboard for 38 FEIs
- 30 are pre-2018 (unique historical data); 51 are post-2018 (some overlap with Redica's 246)
- OCR quality is variable — Redica's text is cleaner and more complete

**FEI coverage breakdown (29 with no observation data):**
- 5 FEIs: never received a 483 at all (inspected, always NAI/clean)
- 12 FEIs: only have pre-2018 483s — nothing post-2018 to share
- **12 FEIs: post-2018 483s confirmed in Redica's event log, but no documents were shared with us**

**PDF pipeline (FDA Dashboard):**
- We also downloaded 82 PDFs from the FDA Dashboard for 38 FEIs
- 30 are pre-2018 (unique historical data); 51 are post-2018 (some overlap with Redica's 246)
- OCR quality is variable — Redica's text is cleaner and more complete

**Speaker note:** The most actionable ask is the 12 FEIs where Redica's own event log shows post-2018 483 inspections but no documents were included in the data share. Those exist — we can see them in the audit trail. Ask: why weren't they included, and can we get them?

---

## Slide 4 — How we extract features: the LLM pipeline
**Title:** Each observation text → GPT prompt → JSON with 10 structured dimensions

**Visual: simple flow diagram**
`[Observation text (verbatim)]` → `[GPT prompt + JSON schema]` → `[10-field structured output]`

**Key design choices:**
- Strict JSON schema (OpenAI structured outputs mode — no hallucinated fields)
- Each field independently defined with anchor examples in the prompt
- Binary flags derived from categorical fields (e.g., DI flag = type ≠ NoIssue)
- Applied to 1,083 Redica observations + 622 PDF observations (separate runs)

**Example — observation text to classification (output + the rule that drove it):**

> *"Batch production records were not completed at the time of manufacture. Entries were reconstructed from memory 2 days after processing."*

| Dimension | Value | Prompt rule (excerpt) |
|---|---|---|
| Severity | **Major** | *"Major: text documents an ACTUAL defect or confirmed failure found at the facility… confirmed examples: test results invalidated, batch records reconstructed"* |
| Data integrity type | **ContemporaneousRecording** | *"ContemporaneousRecording: entries not made at the time of activity, or records reconstructed after the fact"* |
| Scope | **SingleBatch** | *"SingleBatch: failure explicitly tied to one batch or one product run"* |
| Root cause | **Cultural** | *"Cultural: failure reflects a behavioral, procedural, or management gap — not lack of equipment or infrastructure"* |
| Remediation signal | **None** | *"None: no corrective action, commitment, or investigation language present in the text"* |
| Investigation flag | **False** | *"True only if text states an investigation was not performed, delayed, or inadequate"* |

**Speaker note:** This slide just shows what the system does. The full classification rules are in the leave-behind document — we are asking Redica to review those after the meeting and share any concerns.

---

## Slide 5 — Our classification system in full detail
**Title:** 10 structured dimensions — what we extract from each observation

**Table: all dimensions with all possible values**

| Dimension | Type | Values | Purpose |
|---|---|---|---|
| **Violation category** | 8-class | LabControls · ProductionControls · BuildingsEquipment · QualitySystem · PackagingLabeling · OrgPersonnel · RecordsReports · Other | Which CFR Part 211 domain |
| **Severity tier** | 4-class | Critical · Major · Moderate · Minor | Risk level of the violation |
| **Scope of failure** | 4-class | SingleBatch · MultipleProducts · FacilityWide · Unclear | How broadly the failure extends |
| **Root cause type** | 4-class | Capital · Cultural · Mixed · Unclear | Why the failure occurred |
| **Remediation signal** | 4-level ordinal | Strong · Partial · Weak · None | Is facility taking corrective action? |
| **Data integrity type** | 5-class | Falsification · AuditTrail · RawData · ContemporaneousRecording · NoIssue | Specific DI sub-type |
| **Repeat flag** | binary | True / False | Cross-inspection recurrence |
| **Patient risk flag** | binary | True / False | Direct patient safety implication |
| **Contamination flag** | binary | True / False | Physical/microbial contamination |
| **Investigation flag** | binary | True / False | Failure to investigate deviation |
| **OOS/OOT flag** | regex | True / False | Out-of-specification or out-of-trend result mentioned |
| **Warning letter reference** | regex | True / False | Prior warning letter referenced |

**Speaker note:** Ask Redica: which of these dimensions make sense from your experience? Which have you seen attempts to classify before? Which are you skeptical about — and why?

---

## Slide 6 — Redica's classification system in full detail
**Title:** Redica's AI + expert-validated system — severity, domain, and data integrity

**Severity (PIC/S GMP anchored):**

| Level | Definition |
|---|---|
| Critical | Deficiency likely causing direct risk to patient health |
| Major | Significant non-compliance; may cause product defect (not necessarily confirmed) |
| Other | Departure from GMP but not Major/Critical |
| Document rollup | Critical ≥ 1 Crit or > 5 Maj · Major = 1–5 Maj · Minor = all Other |

**Domain: 6 QSL Areas + Level 1 sub-labels (from 1,083 observations):**

| QSL Area | QSL L1 Sub-labels |
|---|---|
| Quality Unit (36%) | Reviews and Approvals · Inadequate · Qualified Personnel · Documentation · Corporate Processes · Agency Notification · Audit · Returned and Salvaged Drug Products |
| Production (19%) | Sterile Products · Process Control · Sterile Environment · Batch Records · Contamination Control · Training · Cleaning Validation · Personnel Responsibilities · Process Monitoring |
| Laboratory (18%) | Lab Controls · System Controls · Analytical Testing · Stability · Sample Management · Deviation Investigations · Routine Testing |
| Facilities and Equipment (18%) | Equipment · Facilities |
| Materials (3%) | Material Sampling and Testing · Material Control · Material Storage · Deviation Investigations |
| Packaging and Labeling (1%) | Labeling and Packaging Controls · Drug Product Containers and Closures |

**Data Integrity: ALCOA-based taxonomy (13 sub-labels):**

| DI Label | Closest match in our system |
|---|---|
| Data Manipulation | Falsification |
| Testing into Compliance | (no equivalent — unique) |
| System Controls | AuditTrail |
| Backup and Archival | AuditTrail |
| Paper Record Controls | AuditTrail |
| Original Data | RawData |
| Data Destruction | RawData |
| Contemporaneous | ContemporaneousRecording |
| Attributable (Batch / Lab / General) | ContemporaneousRecording (partial) |
| Complete | (no equivalent — gap) |
| Accurate | (no equivalent — gap) |

**Speaker note:** This is a genuine learning moment — walk us through the QSL annotation process. For Quality Unit specifically (36% of observations) — what triggers a QU assignment vs the specific operational domain?

---

## Slide 7 — How the two systems compare
**Title:** Three shared dimensions, four unique to us, two unique to Redica

**Shared (comparable):**

| Dimension | Us | Redica | Comparability |
|---|---|---|---|
| Severity | 4-tier (Critical/Major/Moderate/Minor) | 3-tier (Critical/Major/Other) | ✅ Yes — collapse Moderate+Minor → Other |
| Domain | 8-class (CFR Part 211) | 6 QSL Areas + L1 sub-labels | ✅ Mostly — 5 of 6 areas map directly |
| Data integrity | 5-class type | 13 ALCOA-based sub-labels | ⚠ Partial — 3 of our 5 types match; 2 Redica gaps (Complete, Accurate) |

**Unique to us (no Redica equivalent — expert validation requested):**
- Scope: SingleBatch / MultipleProducts / FacilityWide / Unclear
- Root cause type: Capital / Cultural / Mixed / Unclear
- Remediation signal: Strong / Partial / Weak / None
- 4 binary LLM flags: repeat, patient risk, contamination, investigation failure
- 2 regex flags: OOS/OOT, warning letter reference

**Unique to Redica (we could benefit from):**
- AI-generated observation summaries (cleaner text than our OCR PDFs)
- Full 483 document summaries
- QSL L1 sub-labels (more granular than our 8-class violation category)

**Speaker note:** Ask Redica: where do OrgPersonnel (training/qualification) violations and RecordsReports (documentation) violations fall in your QSL taxonomy? Both appear under Quality Unit in Redica — but we assign them to their own categories.

---

## Slide 8 — Where we disagree and why
**Title:** Three systematic gaps — all explainable, none random

**Gap 1 — Severity (the biggest gap)**
- Redica: 59% of observations = Major
- Us: 29% = Major (after updating our prompt)
- Root cause: Redica's PIC/S Major = "significant non-compliance even without confirmed failure." Our prompt requires a confirmed defect or near-certain risk. Neither is wrong — they answer different questions.

**Gap 2 — Domain assignment**
- Top pattern: Redica = QualitySystem, Us = ProductionControls (83 cases) or LabControls (79 cases)
- Root cause: We assign the specific operational domain where the failure occurred. Redica assigns QualitySystem when the quality unit failed to oversee that domain.

**Gap 3 — Data integrity**
- We flag ~30% more observations as DI issues (F1 = 0.51)
- Root cause: Our prompt fires on soft data-reliability language; Redica reserves DI labels for confirmed specific sub-types (Data Manipulation, Contemporaneous, etc.)

**Speaker note:** For each gap, the question is: which definition is more useful for predicting manufacturing risk?

---

## Slide 9 — What we are asking from Redica
**Title:** Three things we need your help with

**1. More data**
Do you have 483 coverage for our remaining 29 FEIs?
Are pre-2018 documents available — even a partial set?
Your observation summaries (AI-generated) would also help us — cleaner text than our OCR PDFs.

**2. Guidance on definitions**
For severity: what is the practical annotator boundary between Major and Other?
For QSL: where do training violations and documentation violations sit — Quality Unit or the specific domain?
For DI: help us map our 5 types to your 13 ALCOA sub-labels — confirm or suggest revisions.

**3. Expert validation of our unique dimensions**
Scope, root cause type, remediation signal, and our binary flags have no Redica equivalent.
We are leaving the full prompt rules document with you — please read through the prompts and tell us whether the definitions and classification rules are reasonable from a regulatory standpoint.
This validation step is required before we can publish these dimensions as research features.

---

## Slide 10 — Leave-behinds and next steps
**Title:** What we are leaving with you

**Documents:**
- This presentation
- Full prompt rules (all dimensions, verbatim definitions + examples) → `20260611_483_LLM_Prompts_Expert_Review.docx`
- Full field-by-field comparison with validation numbers → `20260616_Redica_Classification_Comparison.docx`

**Proposed next steps:**
1. Redica shares annotator rubric for severity (Major/Other boundary)
2. We update our Major prompt based on PIC/S guidance and re-run
3. Together map our 5 DI types to Redica's 13 ALCOA labels — confirm or revise
4. Agree on a ~50-observation sample: Redica team reviews our prompt rules and validates whether scope / root cause / remediation definitions make regulatory sense
5. Schedule follow-up in 4–6 weeks

---

## Design notes for Claude design
**If any slide has too much content to fit cleanly, split it into 2 or more slides. Prefer clarity over compression — a clean two-slide spread is better than a crowded single slide.**
- **Style:** Clean, professional, minimal — not corporate template heavy
- **Color:** Navy (#1a2744) headers, white backgrounds, amber (#d97706) for Redica references, blue (#2563eb) for our system
- **Font:** Sans-serif, large enough to read in a conference room
- **Tables:** Use consistently — this audience is comfortable with structured data
- **No stock photos** — this is a technical/academic meeting
- **Slide 3:** Two-column visual — left = 853/246/607 numbers; right = PDF pipeline note
- **Slide 4:** Show prompt example in a code/quote box — highlight field names in color
- **Slide 5:** Full table — consider alternating row shading by category type (categorical vs binary)
- **Slide 6:** Three stacked sections (Severity / Domain / DI) with clear visual separation; DI section shows the mapping column
- **Slide 7:** Three-part layout: shared (top), unique to us (middle), unique to Redica (bottom)
- **Slide 8:** Three-card layout, numbers called out large

---

## ⚠ Document checklist before sharing
- [ ] `20260611_483_LLM_Prompts_Expert_Review.docx` — **needs update**: still describes `data_integrity_flag_llm` as a binary flag. Must be updated to describe the 5-class `data_integrity_type` (Falsification / AuditTrail / RawData / ContemporaneousRecording / NoIssue). Binary flag is derived from this field.
- [ ] `20260616_Redica_Classification_Comparison.docx` — current (v2 validation numbers, Us/Our/We language)
