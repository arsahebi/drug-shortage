# %%
"""
Build the September 2026 changelog document for the Metformin analysis.

Explains the switch from the manual NDC-FEI map to the rule-based one, the two
sample exclusions now enforced in step 2, the Chartwell Congers data gap, and the
Figure 4 statistics bug found while auditing the prior-inspection logic.

Style comes from doc_style.py, shared with build_figures_doc.py.

Output: outputs/20260910_metformin_pipeline_changes.docx
"""

from pathlib import Path

import doc_style as ds

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
OUT  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs/20260910_metformin_pipeline_changes.docx"

doc = ds.new_document()

def h(text, level=1): doc.add_heading(text, level=level)
def p(text, bold=False, italic=False, size=11): return ds.p(doc, text, bold=bold, italic=italic, size=size)
def bullet(text): return ds.bullet(doc, text)
def table(headers, rows, **kw): return ds.table(doc, headers, rows, **kw)
def box(lines): return ds.box(doc, lines)

# ── title ─────────────────────────────────────────────────────────────────────
doc.add_heading("Metformin Analysis: Pipeline Changes", level=0)
p("September 10, 2026", italic=True, size=10)
ds.rule(doc)

p("Three things changed in the Metformin pipeline. The NDC to FEI map is now built "
  "by a rule instead of by manual search. Canada and Bangladesh facilities are out of "
  "the analysis. Facilities with no Redica inspection history are out of the analysis. "
  "All six steps have been re-run and every figure has been regenerated.")

p("Two further items came out of the work. The prior-inspection logic was audited directly, "
  "because an earlier version of this code had once attributed the wrong inspection history "
  "to facilities. That logic is correct, but the audit turned up a separate bug in how "
  "Figure 4's pairwise significance tests were computed. Section 4 covers both, and confirms "
  "that the Figure 4 conclusion already circulated to the team still holds. Section 5 raises "
  "a facility the new method found that we have no Redica data for.")

box([("If you only read one section.",
      "Section 4. It confirms the inspection-history assignment is correct, documents a "
      "testing bug in the pairwise country comparisons, and shows that the result previously "
      "shared with the team survives a properly calibrated test.")])

# ── 1. rule-based map ─────────────────────────────────────────────────────────
h("1. The NDC to FEI map is now rule-based", 1)

p("The old map came from a manual search. Amir and Amirreza looked up each NDC across "
  "DailyMed, ProPublica and the Redica site list and recorded a facility, resolving "
  "conflicts by judgment. That map is kept as step1_ndc_fei_map_manual.csv for "
  "comparison but is no longer used.")

p("The new map is built by one rule. Take the NDC universe from tab 1 of "
  "step1_ndc_fei_map_rulebased.xlsx, join it to the DailyMed establishment-operation "
  "table in tab 2 on ndc_9, and keep a facility only where the operation type is "
  "\"manufacture\". Analysis, packaging, labeling, repackaging and relabeling sites are "
  "dropped, because those facilities are not where the drug is made and their "
  "inspection record should not be attributed to the product.")

p("Why change: the manual map is not reproducible. Someone else repeating the search "
  "would not necessarily reach the same answer, and we cannot state the selection rule "
  "in the methods section. The rule-based map can be stated in one sentence and re-run.")

p("Column D of tab 1, \"manufacture found\", is a flag for exactly this condition. We "
  "verified it agrees with the tab 2 filter for all 112 NDCs, so the two are consistent. "
  "Tab 3, \"Manufacture Only\", is tab 2 pre-filtered the same way and is not read.")

p("What the two maps produce:", bold=True)
table(
    ["", "Manual (old)", "Rule-based (new)"],
    [
        ["NDC11s in universe", "112", "112 (same)"],
        ["NDC11s with a facility", "89", "77"],
        ["NDC11s with no facility", "23", "35"],
        ["Unique FEIs", "28", "26"],
        ["NDC x FEI pairs", "114", "88"],
    ],
)

p("At the pair level: 83 pairs are shared, 31 were in the manual map only, 5 are new. "
  "Per NDC, 81 are identical, 17 lost their facility, 9 dropped one of two facilities, "
  "and 5 gained a facility they did not have before.")

p("Why NDCs lost a facility:", bold=True)
bullet("28 NDCs have no DailyMed establishment linkage at all. The label exists but "
       "carries no establishment records. The manual search had filled several of these "
       "from ProPublica. All four facilities that disappear from the map entirely "
       "(3005263655, 3006370524, 3010254278, 3030495702) are absent from the DailyMed "
       "table, meaning they were never DailyMed-sourced.")
bullet("7 NDCs are dropped by the manufacture rule itself. These are the cases the rule "
       "is meant to catch: 60687-0143, 60687-0155, 60687-0162, 60687-0640 and 53746-0178 "
       "are linked only to repackaging or packaging sites, and 71205-0884 only to "
       "relabeling and repackaging. Under the old map a repackager was being credited "
       "with a manufacturer's inspection record.")

p("What the new method found that the manual search missed:", bold=True)
table(
    ["FEI", "Facility", "NDCs", "In Redica?"],
    [
        ["3002809586", "Sun Pharmaceutical Industries, Halol (India)",
         "62756-0142-01, 62756-0142-02, 62756-0143-01", "Yes"],
        ["3008897678", "Chartwell Pharmaceuticals Congers (USA)",
         "62135-0680-18, 62135-0683-18", "No, see section 5"],
    ],
)

p("Sun Halol matters. It carries 12 inspection events including 5 OAI. Because OAI is "
  "scarce in this sample, adding one site with 5 of them is the single largest "
  "substantive change in this whole update, and it moves in the opposite direction from "
  "the shrinking sample. Total OAI observations in the analysis panel rise from 11 to 16 "
  "even though the panel is smaller.")

# ── 2. exclusions ─────────────────────────────────────────────────────────────
h("2. Two sample exclusions now apply to the entire analysis", 1)

p("Both are enforced once, in step2_build_panel_july26.py, so steps 3 through 6 inherit "
  "a single filtered panel. They are controlled by three flags at the top of that script:")

table(
    ["Flag", "Effect"],
    [
        ['EXCLUDE_COUNTRIES = {"Canada", "Bangladesh"}',
         "Drops Bausch Health Steinbach (3002806613, Canada, 6 events) and Beximco "
         "Kaliakair (3008763868, Bangladesh, 2 events)."],
        ["REQUIRE_REDICA_HISTORY = True",
         "Drops any FEI with no Redica inspection event. Catches Chartwell Congers "
         "(3008897678), plus the two country-excluded sites."],
        ["DROP_NDCS_WITHOUT_FEI = True",
         "Drops the 35 NDC11s that have no facility at all. Same logic: no facility "
         "means no inspection history either."],
    ],
)

p("Why enforce this in step 2 rather than per figure:", bold=True)
p("Previously neither rule was written down. What existed instead was a plotting "
  "constant in step 6, COUNTRY_ORDER = [\"IND\", \"CHN\", \"USA\"], and a row-level filter "
  "prior_outcome.notna() used in some figures but not others. The result was that each "
  "figure in the paper described a different sample:")

table(
    ["Figure", "n (old)", "Canada/Bangladesh rows kept", "No-history rows kept"],
    [
        ["Fig 1, market by outcome", "107", "0", "0"],
        ["Fig 2, volume vs quality", "126", "0", "20"],
        ["Fig 3, price vs quality", "117", "0", "19"],
        ["Fig 4, quality by country", "127", "0", "20"],
        ["Fig S1, months since inspection", "113", "6", "0"],
    ],
)

p("Figures 2, 3 and 4 are not about inspection history, so nobody filtered on it, and "
  "each carried around 20 products with no facility history. Figure S1 has no country "
  "filter at all and kept 6 Canada and Bangladesh rows. Sample sizes ranged from 107 to "
  "127 with the composition differing too, not just the count. That is not defensible in "
  "review, which is why both rules now apply everywhere.")

p("A further inconsistency this fixes:", bold=True)
p("Country was assigned by a fallback chain: the facility's country parsed from the "
  "Redica site name, falling back to the CountryCode column typed into the Q&A Sheet1 "
  "spreadsheet. For an NDC with no facility, the first lookup returned nothing. So among "
  "the 32 rows with no facility, 18 happened to have a country typed into the "
  "spreadsheet and passed the country whitelist into Figures 2, 3 and 4, while 14 had "
  "nothing and were silently discarded. Same kind of row, opposite treatment, decided by "
  "whether a spreadsheet cell was filled in. Those 14 rows were not empty: all 14 carried "
  "a DMF value and IQVIA volume, and 12 carried a price. Now every surviving row has a "
  "facility, country is always facility-derived, and the spreadsheet fallback never fires.")

p("Two assertions at the end of step 2 fail the build if an excluded country or a row "
  "without inspection history ever reaches the output, so this cannot quietly regress "
  "when the Redica data is refreshed. Every dropped facility and NDC is printed at run "
  "time.")

# ── 3. results ────────────────────────────────────────────────────────────────
h("3. What the re-run produced", 1)

p("Panel sizes at each step:", bold=True)
table(
    ["Step", "Before", "After"],
    [
        ["step2 inspection panel", "744 rows, 112 NDC11s, 28 FEIs", "554 rows, 70 NDC11s, 23 FEIs"],
        ["step2 unique inspection events", "169", "148"],
        ["step3 with Valisure", "n/a", "1,662 rows"],
        ["step4 with volume and price", "n/a", "1,662 rows"],
        ["step5 analysis panel", "148 rows, 112 NDC11s", "96 rows, 70 NDC11s"],
    ],
)

p("Inspection outcomes, unique events:", bold=True)
table(
    ["", "NAI", "VAI", "OAI"],
    [["Before", "70", "89", "10"], ["After", "58", "75", "15"]],
)

p("Prior-inspection outcome in the analysis panel:", bold=True)
table(
    ["", "NAI", "VAI", "OAI", "None"],
    [["Before", "22", "80", "11", "35"], ["After", "15", "62", "16", "3"]],
)

p("The 3 remaining rows with no prior outcome are Harman Finochem's 23155 NDCs. That "
  "facility does have Redica history, but its only inspection is in 2024, so there is no "
  "inspection prior to the 2024 test date. It correctly stays in the sample.")

p("Country composition, analysis panel rows:", bold=True)
table(
    ["", "India", "China", "USA", "Canada", "Bangladesh", "Missing"],
    [["Before", "87", "18", "22", "4", "3", "14"],
     ["After", "67", "17", "12", "0", "0", "0"]],
)

p("Effective sample size per figure:", bold=True)
table(
    ["Figure", "n before", "n after"],
    [["Fig 1", "107", "93"], ["Fig 2", "126", "96"], ["Fig 3", "117", "93"],
     ["Fig 4", "127", "96"], ["Fig S1", "113", "93"]],
)

p("The figures now sit at 93 or 96 rather than spanning 107 to 127. The remaining "
  "difference is legitimate: Figures 2 and 3 additionally require non-null volume or "
  "price, and Figures 1 and S1 require a prior inspection outcome, which excludes the "
  "3 Harman rows. Every figure now draws from the same 96-row, 70-NDC, 23-facility base.")

p("Multi-facility NDCs fall from 25 to 11, since the manufacture rule resolves many "
  "cases where the manual search had recorded a manufacturer alongside a testing or "
  "packaging site.")

p("All 18 figures were regenerated. The full step 6 run log, including every statistical "
  "test, is saved alongside them as step6_run_log_rulebased.txt.")

p("The statistical results need a fresh read before they go in the paper.", bold=True)
p("Figure 1 remains non-significant, Figure 3 remains a clean null, and Figure 2 keeps "
  "both of its correlations. Figure 4 is the one that moves, and section 4 explains why: "
  "part of the movement is a smaller sample and part of it is a testing bug that the audit "
  "uncovered. The figure-by-figure results and readings are in the companion document, "
  "20260910_metformin_figures_rulebased.docx.")

# ── 6. figure 4 statistics bug ────────────────────────────────────────────────
h("4. Audit of the prior-inspection logic, and a statistics bug it turned up", 1)

p("Two separate questions are answered here. First, is the right inspection history being "
  "attached to each product? Yes. Second, are the significance tests behind Figure 4 sound? "
  "The pairwise ones were not, and are now fixed. The regression results, which are what the "
  "team was shown, are sound and are confirmed below.")

p("The prior-inspection logic checks out.", bold=True)
p("Every (NDC, test year) row was re-derived independently from the step 2 panel and the "
  "step 1 map, then compared against what step 5 produced. All 96 rows agree on facility, "
  "outcome and inspection year, with no mismatches. Six further checks passed: the Redica "
  "ID to FEI mapping is strictly one to one across all 29 facilities, no facility carries "
  "more than one site name, no inspection event was dropped for an unmapped identifier, "
  "every assigned facility is one the NDC actually maps to, no prior inspection falls on "
  "or after the test year, and months since inspection recomputes to within 0.05 months.")

p("The Figure 4 significance tests were wrong.", bold=True)
p("Pairwise country comparisons used a centred bootstrap of the Spearman correlation "
  "between the measured value and a group dummy. That approximation assumes the bootstrap "
  "spread stands in for the null distribution. It does not when the data are dominated by "
  "ties. For NDMA, 44 of 55 observations are exactly zero, every US observation is zero, "
  "and 12 of 13 Chinese observations are zero. Two things then went wrong:")

bullet("Resamples in which every value came out identical produced an undefined correlation "
       "and were silently discarded. Those are precisely the no-difference resamples. For "
       "China versus USA, 36 percent of the 2,000 resamples were thrown away, which narrowed "
       "the null distribution and inflated significance.")
bullet("Even where nothing was discarded, the centred-bootstrap p understates the null "
       "spread under heavy ties, so marginal differences were reported as strongly "
       "significant.")

p("What the reported p-values should have been:", bold=True)
table(
    ["Contrast", "Reported before", "Corrected", "Mann-Whitney", "Verdict"],
    [
        ["NDMA, India vs USA", "0.0005", "**0.175**", "0.119", "Was overstated"],
        ["NDMA, India vs China", "0.0245", "**0.121**", "0.101", "Was overstated"],
        ["NDMA, China vs USA", "0.0329", "**1.000**", "0.529", "Was a false positive"],
        ["DMF, India vs USA", "0.0380", "**0.108**", "0.062", "Was overstated"],
        ["DMF, India vs China", "0.0810", "0.192", "0.089", "Unchanged, not significant"],
        ["DMF, China vs USA", "0.7080", "0.714", "0.702", "Unchanged, not significant"],
    ],
    widths=[1.6, 1.2, 1.0, 1.1, 1.6],
)

p("The China versus USA comparison is the clearest illustration. One of 13 Chinese "
  "observations and none of 7 US observations are above zero. Fisher's exact test on that "
  "table gives p = 1.0. The old code reported p = 0.033.")

p("How badly it misbehaves was measured directly.", bold=True)
p("Data were simulated under the null, with group labels assigned at random so that no real "
  "difference exists, and the old test was run 400 times per scenario. A valid test rejects "
  "at 5 percent. The old test rejects at:")
table(
    ["Data structure", "False-positive rate", "Verdict"],
    [
        ["Continuous values (Figure 1 volume)", "6 to 7 percent", "Acceptable"],
        ["19 percent zeros (DMF by country)", "7.8 percent", "Mildly liberal"],
        ["70 percent zeros, n=45 vs 12 (manual panel NDMA)", "**11 percent**", "Twice too liberal"],
        ["80 percent zeros, n=35 vs 7 (current panel NDMA)", "**20 percent**", "Four times too liberal"],
        ["80 percent zeros, n=13 vs 7 (China vs USA)", "**26 percent**", "Five times too liberal"],
        ["90 percent zeros", "**49 percent**", "Unusable"],
    ],
    widths=[2.8, 1.5, 1.5],
)
p("The cluster permutation test and Mann-Whitney sit at 1 to 7 percent in every one of these "
  "scenarios. The failure is specific to zero-inflated data and it worsens as the groups get "
  "smaller, which is why it bites harder in the current sample than it did in the manual one.")

box([("Scope: this affects the pairwise bootstrap only, not the regression models.",
      "The starred Model B tables, which are what the team was shown for Figure 4, come from "
      "a different code path and are unaffected. Figures 2 and 3 use the correlation path, "
      "which was also measured and is properly calibrated at 4.5 to 7.8 percent even with 80 "
      "percent zeros. Figure 1's comparisons involve volume, which has almost no ties, and "
      "were null under every method.")])

p("The fix.", bold=True)
p("Group comparisons now use a cluster permutation test, which builds the null directly by "
  "shuffling the group label across whole clusters and so handles ties correctly. Undefined "
  "resamples in the bootstrap are counted as zero rather than discarded. The permutation "
  "test requires clusters to sit entirely within one group, which holds for country (a "
  "product is made in one country) but not for inspection outcome (a product can be NAI in "
  "one test year and VAI in another). Where clusters straddle groups the function now "
  "returns nothing and the Mann-Whitney p is reported instead, rather than producing a "
  "number that looks authoritative but is not. The old bootstrap p is still printed in "
  "brackets in the run log for continuity.")

p("Figure 1 is unaffected. Its volume comparisons were null under every method before and "
  "after, and the bootstrap and Mann-Whitney p-values agree closely there because volume "
  "has few ties. Figures 2 and 3 use the correlation path rather than the group-comparison "
  "path and are also unaffected.")

box([("Scope note.",
      "Changing the test is an analytical decision, not a mechanical fix. It is applied here "
      "because the old test produced results that a reviewer could refute with a two-by-two "
      "table, but the choice of replacement is worth confirming before publication.")])

doc.add_heading("The result previously shared with the team still stands", 2)

p("Because the pairwise bootstrap was miscalibrated, the regression result was checked "
  "independently rather than assumed. Country labels were permuted across NDCs and the "
  "mixed model refitted 1,000 times, which tests the coefficient directly without relying "
  "on any standard error formula.")

table(
    ["Panel", "NDMA India coefficient", "Reported p", "Permutation p", "Verdict"],
    [
        ["Manual map (shared with the team)", "+1.709", "0.014", "**0.007**", "Supported, in fact stronger"],
        ["Rule-based map (current)", "+1.175", "0.014", "**0.076**", "Marginal, no longer significant"],
    ],
    widths=[2.0, 1.3, 0.8, 1.0, 1.6],
)

p("So the India versus USA NDMA finding that was circulated is not an artifact. A valid test "
  "supports it more strongly than the number that was reported. Nothing needs to be "
  "retracted.")

p("What changed is the sample, not the truth of the earlier analysis. The same coefficient "
  "falls to a permutation p of 0.076 in the current data because the sample is smaller: 55 "
  "observations across 47 NDCs, against 71 across 62. NDMA is also more sparse now, 80 "
  "percent zeros against 73 percent. The finding weakened through loss of power, not because "
  "it was wrong before.")

box([("Bottom line for the team.",
      "The previously shared Figure 4 conclusion holds. In the tighter sample it becomes "
      "marginal rather than significant, which is a sample-size argument for the broader "
      "14-drug analysis rather than a correction to anything already sent.")])


# ── 4. chartwell ──────────────────────────────────────────────────────────────
h("5. Open item: Chartwell Congers, a facility we have no Redica data for", 1)

p("The Redica pull covers 29 FEIs. All 28 facilities from the manual search are in it, "
  "so nothing we asked for is missing. The gap runs the other way.")

p("The rule-based map identified Chartwell Pharmaceuticals Congers, LLC, FEI 3008897678, "
  "as the manufacturer for two NDCs. The manual search never found this facility, so it "
  "was never included in the request we sent Redica, and we therefore have no inspection "
  "history for it.")

table(
    ["Item", "Detail"],
    [
        ["FEI", "3008897678"],
        ["Facility", "Chartwell Pharmaceuticals Congers, LLC"],
        ["NDCs affected", "62135-0680-18 and 62135-0683-18 (2 NDCs)"],
        ["Role", "Sole manufacturer for both, so both NDCs drop with the facility"],
        ["Rows lost", "2 rows in the analysis panel, both TestYear 2024"],
        ["Current status", "Excluded by REQUIRE_REDICA_HISTORY"],
    ],
)

p("Practical impact right now is zero. These same 2 NDCs were already being dropped from "
  "every figure under the old code, because they were among the 14 no-country rows the "
  "COUNTRY_ORDER whitelist silently discarded. The difference is that the exclusion is "
  "now explicit and logged rather than accidental.")

p("If we want them back, we need a Redica top-up request for this single FEI. Whether it "
  "is worth asking depends on whether two 2024 NDCs are worth a data request. Noting it "
  "here so the decision is on the record either way.")

p("Sun Pharmaceutical Halol, the other facility the manual search missed, happened to be "
  "in the Redica pull already, which is why it contributes to the analysis and Chartwell "
  "does not.")

# ── 7. figure labels ──────────────────────────────────────────────────────────
h("6. Figure axis labels", 1)
p("The Figure 1 axis labels no longer show the numeric severity scores. The categories now "
  "read NAI, VAI and OAI rather than NAI (0), VAI (1.5) and OAI (3.5), and the axis title is "
  "Prior Inspection Outcome rather than Prior Inspection Outcome (prior_score). The scores "
  "remain in the underlying data as prior_score; they are simply no longer displayed.")


# %%

# ── 5. files ──────────────────────────────────────────────────────────────────
h("7. Files", 1)
table(
    ["File", "Status"],
    [
        ["step1_build_ndc_fei_map_rulebased.py", "New. Builds the rule-based map."],
        ["step1_ndc_fei_map_rulebased.csv", "New. The map now in use."],
        ["step1_build_ndc_fei_map_manual.py", "Renamed from step1_build_ndc_fei_map.py. Kept for comparison, not in the pipeline."],
        ["step1_ndc_fei_map_manual.csv", "Renamed from step1_ndc_fei_map.csv."],
        ["step2_build_panel_july26.py", "Reads the rule-based map. Three exclusion flags and two guard assertions added."],
        ["step3 / step4 / step5 / step6", "Unchanged code, re-run on the new panel."],
        ["outputs/step6_run_log_rulebased.txt", "New. Full run log with all statistical output."],
        ["DESIGN_NOTES.md", "Updated for the rule, the exclusions, and the new counts."],
    ],
)

p("Verification note: running step 2 with the manual map and all exclusion flags off "
  "reproduces the previous panel exactly. The differences reported here come from the map "
  "change and the exclusions, not from incidental code drift.", italic=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"Saved: {OUT}")
# %%
