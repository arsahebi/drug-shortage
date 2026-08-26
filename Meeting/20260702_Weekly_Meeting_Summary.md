# Weekly Meeting Summary — July 2, 2026

**Attendees:** Amirreza Sahebi, Amir Hossein Sadeghi, Robert Handfield, John Gray, Shailesh Divey  
**Deadline:** Metformin paper revision due July 15, 2026 (2 weeks from meeting date)

---

## Main Topic: Metformin Paper Revision

### NDC→FEI Coverage Update
- Original submission: ~75% coverage (82 NDCs matched out of ~112 total)
- After new review (ProPublica access + manual audit): **86% coverage** (96/112 NDCs matched; 16 unmatched)
- Improvements came from: ProPublica data (not available at original submission), manual re-check by Amir & Amirreza
- Some original FEI assignments were wrong ("Manufactured For" facilities mistakenly used as "Manufactured By")
- New Redica July 2026 data provides more complete inspection records than the Sep 2025 export

### Key Data Issues Identified
1. **Wrong FEI assignments**: Some NDCs previously linked to "Manufactured For" facilities — not the true manufacturer. ~5–6 identifiable errors.
2. **Multi-FEI NDCs**: Some NDCs have two legitimate manufacturing FEIs (two DUNS numbers, sometimes same campus, sometimes different). Previously only one was assigned.
3. **ProPublica errors**: At least two errors found — one address mismatch (different city/zip vs DailyMed), one case where ProPublica grabbed a "Manufactured For" facility as "Manufactured By." ProPublica is not used directly; used as a cross-check only.
4. **FEIs missing from Redica**: ~6–10 new FEIs need to be requested from Redica urgently.

### Decisions Made

| Decision | Detail |
|----------|--------|
| **Rerun analysis with corrected data** | Given known errors in FEI assignments, team will not keep old data; corrections required |
| **Multi-FEI NDCs: include both** | For NDCs with two manufacturing FEIs, include both in analysis and use the most recent inspection from either facility (whichever is most recent prior to the Valisure test date) |
| **ProPublica: not used as primary source** | Team manually verified and found errors; used only as a cross-check. Future larger studies may reconsider using ProPublica directly. |
| **Bangladesh and Canada: keep dropping** | Maintain existing decision to exclude from country-level analysis (insufficient data) |
| **Sensitivity analysis: not needed** | Reviewers did not request; not adding it |
| **IQVIA: add as limitation** | 5–10% of market hidden from IQVIA (Kaiser, VA, closed systems); volume data reliable, pricing less so |
| **NADAC: John believes reviewer is wrong** | NADAC does have NDC-level data; John to get confirmation from expert contact |
| **ER vs IR: Shailesh to evaluate** | Create table of NDC characteristics (dosage form, strength, ER/IR, manufacturer location); potentially split sample by ER/IR if warranted |
| **Wayback Machine check** | For discrepant FEIs, check Internet Archive for 2025 DailyMed labels to see if changes are label updates vs. original errors |

### Risk Assessment
- Rerunning analysis risks changing conclusions → risk of losing paper or needing rewrite
- John: "50-50" chance conclusions remain the same
- If conclusions unchanged → submit by July 15
- If conclusions change → request extension (John has not traveled until July 19; Rob out week of July 11)

---

## Action Items

| # | Owner | Action | Priority/Deadline |
|---|-------|--------|-------------------|
| 1 | Amirreza | Finalize complete FEI list (including secondary FEIs for multi-FEI NDCs) | **Urgent — same day** |
| 2 | Amirreza | Send updated FEI list to Redica; flag as urgent | **Urgent** |
| 3 | John | Email Redica on top of team request to emphasize urgency | Immediately after FEI list received |
| 4 | Amir + Amirreza | Go through each discrepant NDC row one by one; check Wayback Machine for 2025 labels where needed | Afternoon of July 2 |
| 5 | Amirreza | Restructure data with corrected NDC→FEI mapping (incl. both FEIs for multi-FEI NDCs); rerun full analysis | **By mid-next week (July 8–9)** |
| 6 | Shailesh | Create table of NDC characteristics (dosage form, strength, ER/IR, location); re-read reviewer comment; consider split-sample analysis | This week |
| 7 | Shailesh | Check if dosage forms are all oral/solid or if there are injectables | This week |
| 8 | John | Draft response document: editor intro, middlemen emphasis (Reviewer 1), reshoring mention (Reviewer 2), IQVIA limitation, NADAC defense | This week |
| 9 | John | Get confirmation from expert (Ben) on NADAC NDC-level data accuracy | Follow-up needed |
| 10 | John | Consider contacting ProPublica lead about errors found in their data | After revision |
| 11 | All | Assess whether conclusions change after rerun; if yes, request extension from journal | After analysis rerun |

---

## Other Reviewer Comments (Lower Priority)
- **Middlemen emphasis**: Reviewers want more; John to handle; aligns with Nikhil's dissertation work
- **Reshoring**: Add mention; team not opposed but notes trade-offs
- **IQVIA accuracy**: Add as limitation; 5–10% of market value hidden in closed systems (Kaiser, VA)
- **NADAC**: Reviewer claims it's at active ingredient level, not NDC — team believes reviewer is wrong; John to confirm
- **ER vs IR mixing**: Reviewer concerned that combining extended-release and immediate-release may confound results; Shailesh's subgroup analysis is a candidate response

---

## Notes
- Two-week deadline is tight; John is fully available through July 15; Rob out week of July 11
- Extension possible if needed — John will request if rerun results require rewriting
- Future work: as sample scales to 40+ drugs / 400–1,000 NDCs, manual FEI verification becomes impractical; team will evaluate whether ProPublica can be used as primary source with spot-check auditing
