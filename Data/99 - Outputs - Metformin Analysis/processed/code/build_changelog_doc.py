# %%
"""
Build the September 2026 changelog document for the Metformin analysis.

Explains the switch from the manual NDC-FEI map to the rule-based one, the two
sample exclusions now enforced in step 2, and the Chartwell Congers data gap we
may want to raise with Redica.

Output: outputs/20260910_metformin_pipeline_changes.docx
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
OUT  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs/20260910_metformin_pipeline_changes.docx"

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def h(text, level=1):
    doc.add_heading(text, level=level)


def p(text, bold=False, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold, run.italic = bold, italic
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, htxt in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(htxt)
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(10)
    doc.add_paragraph()
    return t


# ── title ─────────────────────────────────────────────────────────────────────
title = doc.add_heading("Metformin Analysis: Pipeline Changes", level=0)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = sub.add_run("September 10, 2026")
r.italic = True
r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

p("Three things changed in the Metformin pipeline. The NDC to FEI map is now built "
  "by a rule instead of by manual search. Canada and Bangladesh facilities are out of "
  "the analysis. Facilities with no Redica inspection history are out of the analysis. "
  "All six steps have been re-run and every figure has been regenerated.")

p("There is also one open item at the end: a facility the new method found that we "
  "have no Redica data for, which we may want to request.")

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
         "62135-0680-18, 62135-0683-18", "No, see section 4"],
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
p("The Figure 1 volume-by-outcome results remain non-significant. The Figure 4 "
  "country comparison shifts: India vs USA on DMF is now significant under the "
  "NDC-clustered bootstrap (p=0.038), and all three NDMA country contrasts are now "
  "significant (India vs China p=0.025, India vs USA p=0.0005, China vs USA p=0.033). "
  "These moved because the sample changed, not because the method changed, so they "
  "should be interpreted afresh rather than compared to the old numbers.")

# ── 4. chartwell ──────────────────────────────────────────────────────────────
h("4. Open item: Chartwell Congers, a facility we have no Redica data for", 1)

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

# ── 5. files ──────────────────────────────────────────────────────────────────
h("5. Files", 1)
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
