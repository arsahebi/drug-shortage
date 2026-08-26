# Presentation Brief — Redica Collaboration Meeting
**Audience:** Redica technical team (feature engineers + domain experts)
**Duration:** 60 minutes
**Tone:** Collaborative — we explain our rationale, they explain theirs, we find alignment
**Handouts:** *(none — prompt rules document will be shared after the meeting, not before)*

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
**Title:** Each observation text → Anthropic Haiku 4.5 prompt → JSON with 10 structured dimensions

**Visual: simple flow diagram**
`[Observation text (verbatim)]` → `[Anthropic Haiku 4.5 prompt + JSON schema]` → `[10-field structured output]`

**Key design choices:**
- Strict JSON schema (OpenAI structured outputs mode — no hallucinated fields)
- Each field independently defined with anchor examples in the prompt
- Binary flags output directly from the model (e.g., data integrity flag = True/False)
- Applied to 1,083 Redica observations + 622 PDF observations (separate runs)

**Example — observation text to classification (output + the rule that drove it):**

> *"Batch production records were not completed at the time of manufacture. Entries were reconstructed from memory 2 days after processing."*

| Dimension | Value | Prompt rule (excerpt) |
|---|---|---|
| Severity | **Major** | *"Major: text documents an ACTUAL defect or confirmed failure found at the facility… confirmed examples: test results invalidated, batch records reconstructed"* |
| Data integrity flag | **True** | *"True only for explicit data trustworthiness failures: falsification, backdating, deleted records, disabled audit trail, unreported OOS, reconstructed entries"* |
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
| **Data integrity flag** | binary | True / False | Is an explicit DI failure documented? |
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

**Data Integrity: ALCOA-based taxonomy (observed sub-labels from 143 DI-flagged obs, 12.8% of total):**

| Redica DI sub-label | % of DI obs | ALCOA+ principle | Question for meeting |
|---|---|---|---|
| System Controls | 26% | — (21 CFR 11 / system governance) | Is this ALCOA+ or a separate GMP requirement? |
| **Contemporaneous** | **21%** | **C — Contemporaneous** | ✓ confirmed ALCOA+ |
| Complete | 15% | Complete (+) | ✓ confirmed ALCOA+ |
| Attributable | 12% | A — Attributable | ✓ confirmed ALCOA+ |
| Data Manipulation | 9% | Accurate + Original | Which ALCOA+ dimension does this map to? |
| Testing into Compliance | 4% | Accurate | Unique to Redica? Not a named ALCOA+ dimension |
| Data Destruction | 3% | Enduring | ✓ confirmed ALCOA+ |
| Backup and Archival | 3% | Enduring / Available | ✓ confirmed ALCOA+ |
| Original Data | 2% | O — Original | ✓ confirmed ALCOA+ |
| Accurate / Paper Record Controls | ~2% | Accurate / Legible | |

**Our approach (binary flag):** We use a single True/False flag. True = explicit data trustworthiness failure (falsification, backdating, deleted records, disabled audit trail, unreported OOS, reconstructed entries). The sub-classification is deferred pending Redica feedback.

**Speaker note:** The key question is what drove Redica to build a 13-label ALCOA-anchored taxonomy vs a binary flag — does the sub-type granularity feed a specific downstream model or risk score? This is the question for the meeting that will inform whether we adopt a similar sub-classification or stay binary.

---

## Slide 7 — How the two systems compare
**Title:** Two shared dimensions, five unique to us, two unique to Redica

**Shared (comparable):**

| Dimension | Us | Redica | Comparability |
|---|---|---|---|
| Severity | 4-tier (Critical/Major/Moderate/Minor) | 3-tier (Critical/Major/Other) | ✅ Yes — collapse Moderate+Minor → Other |
| Domain | 8-class (CFR Part 211) | 6 QSL Areas + L1 sub-labels | ✅ Mostly — 5 of 6 areas map directly |

**Data integrity — different approaches:**

| | Us | Redica |
|---|---|---|
| **Approach** | Binary flag (True/False) | 13-label ALCOA-based taxonomy |
| **Rate** | ~17% of observations | 12.8% (143 of 1,083) |
| **Threshold** | Explicit data trustworthiness failure only | Confirmed ALCOA+ sub-type required |
| **Top labels** | — | System Controls 26% · Contemporaneous 21% · Complete 15% · Attributable 12% |
| **Origin** | Conservative binary rule | ALCOA+ (Attributable, Legible, Contemporaneous, Original, Accurate + Complete, Enduring, Available) |

**Question for Redica:** What drove the decision to build a 13-label ALCOA taxonomy rather than a binary flag? Does the sub-type granularity feed a specific downstream risk score?

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

**Speaker note:** Ask Redica: where do OrgPersonnel (training/qualification) violations and RecordsReports (documentation) violations fall in your QSL taxonomy? Both appear under Quality Unit in Redica — but we assign them to their own categories. On DI: we want to understand whether the 13-label taxonomy was built for a specific predictive model or driven by regulatory/audit use cases — the answer will determine whether we adopt a similar taxonomy after the meeting.

---

## Slide 8 — Where we agree and where we still differ
**Title:** Severity is calibrated — domain assignment is the key open question

**Resolved — Severity**
- Redica: 70% of observations = Major or Critical
- Us: 67% = Major or Critical — 3pp gap (was 25pp with prior prompt)
- Our updated prompt now aligns with Redica's PIC/S standard for confirmed defects
- Agreement rate: **79%** (same-text comparison, 1,066 matched observations)

**Persistent — Domain assignment**
- Agreement: **65%** (up from 62% with prior prompt)
- Top mismatch pattern: Redica = QualitySystem, Us = ProductionControls (83 cases) or LabControls (79 cases)
- Root cause: We assign the specific operational domain where the failure occurred. Redica assigns QualitySystem when the quality unit failed to oversee that domain.
- **Question for meeting:** Is domain assigned by where the failure happened, or who was responsible for preventing it?

**Speaker note:** Severity agreement is nearly perfect — the productive discussion topic is domain philosophy. The disagreement is systematic and interpretable: we assign to the technical domain (lab, production); Redica assigns to Quality Unit as the oversight body. Both are defensible. Understanding Redica's choice will tell us whether to realign our prompt or keep the split as a feature.

---

## Slide 9 — What we are asking from Redica
**Title:** Three things we need your help with

**1. More data**
Do you have 483 coverage for our remaining 29 FEIs?
Are pre-2018 documents available — even a partial set?
Your observation summaries (AI-generated) would also help us — cleaner text than our OCR PDFs.

**2. Guidance on domain classification**
For severity: what is the practical annotator boundary between Major and Other?
For domain (QSL): when a lab or production failure is documented, do you assign it to the specific technical domain, or to Quality Unit as the responsible oversight body?
Where do training violations (OrgPersonnel) and documentation violations (RecordsReports) fall in your QSL taxonomy?

**3. Expert validation of our unique dimensions**
Scope, root cause type, remediation signal, and our binary flags have no Redica equivalent.
We will share the full prompt rules document with you after this meeting — we are asking you to review whether the definitions and classification rules are reasonable from a regulatory standpoint.
This validation step is required before we can publish these dimensions as research features.

---

## Slide 10 — Open questions for today

**Title:** What we most want to understand from Redica

**On severity:**
- What is the practical annotator boundary between Major and Other — do annotators see a confirmed defect before calling it Major, or does significant non-compliance alone qualify?

**On domain:**
- When a lab failure is documented, do you assign to Laboratory or to Quality Unit?
- Where do training deficiencies and documentation gaps (OrgPersonnel, RecordsReports in our taxonomy) fall in your QSL?

**On data integrity:**
- What drove the decision to build a 13-label ALCOA-based taxonomy rather than a binary flag?
- Does each sub-label feed a specific downstream risk score, or is the taxonomy for regulatory audit use?
- We want to adopt a compatible approach — should we follow ALCOA+ sub-classification after this meeting?

**What happens after this meeting:**
- We will share our full prompt rules document for your review: definitions, anchor examples, and the distributions we observe
- We will realign our DI classification to match your ALCOA+ approach based on your feedback
- Goal: a shared classification framework we can both stand behind for publication

**Speaker note:** Keep this slide conversational — these are genuine open questions, not rhetorical. We want Redica's perspective on each before we commit to final prompt definitions.

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

**Shared at meeting (presentation only):**
- [x] This brief → presentation slides

**Shared after meeting:**
- [ ] `20260611_483_LLM_Prompts_Expert_Review.docx` — needs update before sharing:
  - Section 1.3: revert to binary `data_integrity_flag_llm` with `[PLACEHOLDER — DI prompt to be updated after Redica meeting feedback]`
  - Section 2.4: already updated (root_cause_type + rationale field, observed distribution)
- [ ] `20260616_Redica_Classification_Comparison.docx` — update with full-run numbers: severity 79%, domain 65%; remove DI comparison section (placeholder pending ALCOA+ alignment)
