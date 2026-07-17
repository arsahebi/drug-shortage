# Weekly Meeting Summary — July 16, 2026

**Attendees:** Amirreza Sahebi, Amir Hossein Sadeghi, Shailesh Divey, John Gray  
**Note:** Rob Handfield on vacation.

---

## Topic 1: INFORMS 2026 Presentation (Amirreza)

### Current Status
- Presentation draft is mostly complete; AE correlation, prediction model, gap analysis, and trajectory clustering are all in.
- Last section (reverse clustering / silent problem) still being refined.
- INFORMS presentation is around July 28–29, 2026.
- Team will do a full rehearsal next week (~July 23) with 5 days remaining before the conference.

### Redica Expert Validation
- No response from Redica yet; not enough time to incorporate even if they reply now.
- Decision: move Redica expert review to "future work" in the limitations slide.
- John and Shailesh have broad pharma manufacturing and inspection expertise — counts as informal domain validation. No systematic check yet.

### LLM Validation
- Amirreza found recent papers showing GPT-mini and Claude Haiku are reasonably accurate for text labeling tasks.
- Will cite those papers to support the extraction methodology, acknowledge no manual benchmark yet.
- Abdul (newly hired) will do manual labeling once prompts are finalized — future work.

---

## Topic 2: Key Analytical Insights from Discussion

### FDA Risk-Based Site Selection Model (SSM)
John confirmed FDA uses a formal, risk-based Site Selection Model (CDER MAPP 5014.1) to prioritize facilities for CGMP surveillance inspections. The model generates a composite site score from six factors:

| Factor | Our equivalent |
|--------|---------------|
| Compliance history (OAI/VAI/NAI) | `n_oai_cumul`, inspection outcome variables |
| Time since last inspection | Inspection gap variable |
| Inherent product risk (NTI drugs, sterility class) | Drug class features |
| Hazard signals (FAR, BPDR, **recalls**, **AEs**) | `n_recall_class_I`, FAERS AE counts |
| Patient exposure / volume | IQVIA market share |
| Facility type | Site type flag |

**Critical implication:** The SSM already incorporates adverse events and recalls as hazard signals. Rising AEs before an inspection are partly _why_ FDA selects that facility to visit. The gap analysis finding ("AEs rise 27% in the year before FDA arrives") is therefore not surprising — it reflects the SSM working as designed. The contribution of 483 text is that it predicts AE trajectories using information that the SSM does _not_ include: the content of prior inspection observations.

John's inspections paper (SSRN abstract_id=5252874, Wright, Gray et al., under second review at POM) has an explicit mapping of SSM factors to observable variables; cite this for the selection bias discussion.

### OAI→VAI Downgrades — the Core Mechanism for the "Silent Problem"
John shared findings from an unpublished analysis his team conducted using FDA's internal inspection data (2019–2021):
- FDA frequently downgrades OAI findings to VAI, primarily when worried about drug shortages or when political pressure is present.
- Facilities that received an OAI→VAI downgrade had the **worst patient outcomes** of any group, measured by adverse events.
- Mechanism: by reclassifying to VAI, FDA avoids issuing a warning letter, so the facility escapes mandatory corrective action and remediation timelines — yet the underlying quality problems are as bad as facilities that received warning letters.
- This is the most plausible explanation for the Hi-sig VAI pattern in our data: high Lab Controls + DI text signals, never formally OAI, but AEs not declining.

**Implication for slides and paper:**
- The "Three explanations" list on the strategic leniency slide should lead with OAI→VAI downgrade as the most documented mechanism.
- The silent problem framing is well-supported: we observe a downstream consequence of FDA's own internal downgrade practice.
- John's suggestion: create a mismatch indicator (high 483 text risk + VAI outcome) as a novel quality risk signal.

### 483 Text vs. Formal Classification
- John's view: 483 raw text is likely a better quality signal than OAI/VAI/NAI precisely because formal classifications are subject to shortage-related political distortion.
- Our result (text AUC 0.528 vs. OAI flag AUC 0.463) is exactly this story. John noted this will have audiences "nodding their heads."
- Sample is still small (2 facilities in the flat/rising cluster) but the direction is credible and matches their internal findings.

---

## Topic 3: Metformin Paper Revision (Health Affairs Scholar)

### NADAC — Confirmed Issue
- Reviewer was right: NADAC reports an average price across all NDCs for a molecule-size combination, not NDC-specific prices. Different manufacturers effectively get the same NADAC price.
- Plan: drop NADAC from the model if we cannot obtain NDC-specific price data from David Light. If dropped, cite prior papers that already established the price-quality relationship.

### Price Data
- David Light has price data from 2022 DoD/Valisure sweeps. Waiting for delivery.
- If not received in time, subset to 2022 sweep only (risks replicating an existing paper) or drop entirely.

### Inspection Timing Window
- Reviewer asked for a principled inspection window: using any prior inspection regardless of how long ago is indefensible (an inspection 6 years old says little).
- Plan: test windows such as 6 months to 2.5 years before the Valisure test date.
- John and Shailesh to coordinate. This may recover the inspection result that reviewers called "the most headline-grabbing finding."

### IR/ER Split Sample
- Reviewer wants separate analysis for immediate-release (IR) and extended-release (ER) formulations.
- Shailesh working on this. Need updated analysis after corrected FEI mapping is in place.

### ProPublica NDC→FEI Comparison
- Shailesh's charts show ProPublica's NDC→FEI mapping is very similar to Amir + Amirreza's manual work.
- John raised idea of partnering with ProPublica to update their data monthly.
- Trade-off: lose data-exclusivity advantage, gain public impact. Tabled for now.
- The team's mapping covers some NDCs ProPublica doesn't have; ProPublica has some the team doesn't. Few actual disagreements.

---

## Topic 4: Team Expansion

### NEMA (New Collaborator)
- Iranian-origin researcher with strong technical background; new baby is no longer newborn, teaching starts January, two revisions in.
- John is formally inviting him to work on **MarketScan** data independently.
- Shailesh compressed the raw data from 775 GB to 80 GB.
- MarketScan: actual hospital utilization, NDC-level, not voluntary reporting — best available quality outcome measure.
- First-mover advantage: very few papers use MarketScan for pharmaceutical quality research.
- He'll work as a separable workstream; team will integrate results once pipeline is established.

### Abdul (New Hire)
- Formally hired. John will send onboarding materials this week.
- Initial task TBD: potentially manual labeling of 483 observations for Amirreza's text analysis, or assisting with Health Affairs Scholar revision.

### FAERS vs. MarketScan Discussion
- FAERS weaknesses: voluntary reporting, recall-event spikes (media-driven), molecule-level attribution, causality challenge for reviewers.
- MarketScan strengths: observational (not voluntary), NDC-level, exogenous to manufacturer actions.
- Shailesh's test idea: use LexisNexis or media coverage timestamps to identify FAERS reporting lags induced by public events (e.g., NDMA recall → spike in Metformin AE reports).
- Plan: use FAERS for current work, MarketScan as primary outcome in future paper once NEMA's pipeline is ready.

---

## Topic 5: Broader Notes

### Funding
- NIH program manager flagged that the team is not spending grant funds quickly enough.
- John will invite NEMA and potentially Abdul using available grant funds.

### ProPublica Partnership (Tabled)
- John floated the idea of partnering with ProPublica: team provides scripts or manual labor to keep their inspection database updated monthly.
- Concerns: loss of data advantage, significant effort at scale (129 FEIs = 14 drugs, ~500 FEIs eventually).
- Tabled pending other priorities.

### Publication Target
- This text analysis paper does not fit Health Affairs Scholar (wrong audience).
- INFORMS OM journals: increasingly technical/ML-heavy (per John's report from MSOM). Benchmark requirements are rising.
- Target TBD: could be an OM journal if linked to a quality or operational outcome, or a pharmacovigilance/methods journal. Decision deferred until after INFORMS feedback.

---

## Action Items

| # | Owner | Action | Priority / Deadline |
|---|-------|--------|---------------------|
| 1 | Amirreza | Rehearsal presentation to team | ~July 23, 2026 |
| 2 | Amirreza | Add OAI→VAI downgrade context to slide 18 and 20 explanations | Before INFORMS |
| 3 | Amirreza | Reframe gap slide using SSM context (SSM uses AEs/recalls; 483 text is the missing piece) | Before INFORMS |
| 4 | Amirreza | Add mismatch indicator (high text risk + VAI) to future work or paper methods | Paper draft |
| 5 | John + Shailesh | Define and run inspection timing window analysis for HA Scholar | This week |
| 6 | John + Shailesh | IR/ER split analysis for HA Scholar | This week |
| 7 | John | Formally invite NEMA; assign MarketScan pipeline task | Immediately |
| 8 | John | Send Abdul onboarding materials (papers, websites, data) | Same day |
| 9 | John | Follow up with David Light on price data | Ongoing |
| 10 | John | Re-read Shailesh's email (NADAC / selection bias) before in-person meeting | Same day |

---

## References from Meeting

- **CDER Site Selection Model**: FDA MAPP 5014.1; presented by John Wan, Division of Quality Data Science. Six SSM factors: compliance history, inherent product risk, facility type, time since last inspection, hazard signals (FAR, BPDR, recalls, AEs), patient exposure.
- **Gray et al. inspections paper**: SSRN abstract_id=5252874. Maps SSM factors to observable proxies; under second review at POM (major revisions). Use to justify variable selection and frame endogeneity discussion.
- **OAI→VAI downgrade analysis**: Unpublished. Gray team analysis of FDA internal data (2019–2021). Finding: downgraded facilities had worst adverse event outcomes because they escaped mandatory remediation.
