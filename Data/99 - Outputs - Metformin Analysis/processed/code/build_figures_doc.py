# %%
"""
Build the September 2026 figures report for the Metformin analysis.

Every figure regenerated on the rule-based NDC-FEI map and the two sample
exclusions, each with its statistics and a short reading of what it shows.

Figure 4 statistics use the corrected cluster permutation test; see section 6 of
20260910_metformin_pipeline_changes.docx for why the earlier bootstrap p-values
were wrong.

Style comes from doc_style.py, shared with build_changelog_doc.py.

Output: outputs/20260910_metformin_figures_rulebased.docx
"""

from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH

import doc_style as ds

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
FIG  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs"
OUT  = FIG / "20260910_metformin_figures_rulebased.docx"

doc = ds.new_document()

def p(text="", bold=False, italic=False, size=11, align=None):
    return ds.p(doc, text, bold=bold, italic=italic, size=size, align=align)
def bullet(text, size=11): return ds.bullet(doc, text, size=size)
def table(headers, rows, **kw): return ds.table(doc, headers, rows, **kw)
def box(lines, size=10): return ds.box(doc, lines, size=size)
def figure(png, caption, width=6.0): return ds.figure(doc, FIG / png, caption, width=width)
def rule(): return ds.rule(doc)

doc.add_heading("Metformin Analysis: Regenerated Figures and Statistics", 0)
p("September 10, 2026", italic=True, size=10)
rule()

p("Every figure here was regenerated after two changes to the pipeline: the NDC to FEI map "
  "is now built by a rule rather than by manual search, and two sample exclusions now apply "
  "to the whole analysis. The changes themselves, and an audit of the inspection-history "
  "logic, are documented in the companion file 20260910_metformin_pipeline_changes.docx. "
  "This document shows what the figures look like now, the statistics behind each one, and "
  "a short reading of what each shows.")

p("If your reference point is comparison_prerevision_vs_july2026.docx, the last version "
  "circulated, here is how the NDC count reconciles. Valisure tested 112 NDCs, and that total "
  "is the same in both versions. Neither method matched all of them to a manufacturing "
  "facility.")

table(
    ["", "Manual search", "Rule-based"],
    [
        ["NDCs tested by Valisure", "112", "112"],
        ["Matched to a facility", "89", "77"],
        ["Not matched to any facility", "23", "35"],
        ["Of those matched, in Canada or Bangladesh", "5", "5"],
        ["Of those matched, facility has no Redica history", "0", "2"],
        ["**NDCs in the analysis**", "**84 if the exclusions had been applied**", "**70**"],
    ],
    widths=[2.9, 1.7, 1.4],
)

p("So the drop from 89 to 77 matched NDCs is the rule-based method being stricter, mostly "
  "because 28 NDCs have no DailyMed establishment record at all and the manual search had "
  "filled several of those from ProPublica. The further drop to 70 is the two exclusions: 5 "
  "NDCs made in Canada or Bangladesh, and 2 made by Chartwell Congers, which has no Redica "
  "inspection history. The same 5 Canada and Bangladesh NDCs were present in the manual map "
  "too; they were simply never excluded.")

p("The Figure 4 conclusion in that earlier document has been re-checked with a properly "
  "calibrated test and still holds. It is weaker here only because this sample is smaller. "
  "Section 4 of the companion file has the detail.")

p("One correction to that earlier document while we are here. Notes under its Figures 1 and 2 "
  "quote 145 rows in total and 110 rows across 85 NDCs for the single-FEI panel. Those three "
  "figures were hardcoded rather than computed, and the panel they describe actually held 148 "
  "rows, and 113 rows across 87 NDCs for single-FEI. The multi-FEI count of 25 is correct. "
  "Nothing downstream used the quoted numbers, so no result in that document is affected, but "
  "the counts themselves should not be carried forward.", italic=True, size=9)

doc.add_heading("Analysis sample", 1)

p("All figures draw on the same base: 96 observations, 70 NDCs, 23 facilities, across "
  "test years 2020, 2022 and 2024. Canada and Bangladesh facilities are excluded, as are "
  "facilities with no Redica inspection history and NDCs with no identified facility.")

table(
    ["Subset", "Rows", "NDCs", "Definition"],
    [
        ["Full sample", "96", "70", "All observations passing the two exclusions"],
        ["Single-FEI", "79", "59", "Drops the 11 NDCs with two registered manufacturing plants"],
        ["Gap 36mo", "70", "56", "Prior inspection within 36 months of the Valisure test"],
        ["Single-FEI + Gap 36mo", "53", "45", "Both restrictions applied"],
    ],
    widths=[1.6, 0.6, 0.6, 3.4],
)

p("Why the two robustness subsets exist:", bold=True)
bullet("Single-FEI. Eleven NDCs list two manufacturing plants on the same DailyMed label "
       "(Aurobindo Hyderabad and Jadcherla, Lupin Mormugao and Nagpur, Zydus Sanand twice). "
       "We cannot tell which plant made the lot Valisure tested, so the inspection outcome "
       "assigned to those NDCs depends on which plant was inspected most recently. Five of "
       "them flip between OAI and VAI across test years for that reason alone.")
bullet("Gap 36mo. An inspection several years before a test says less about the state of the "
       "plant at the time of the test. This subset keeps only observations where the prior "
       "inspection falls within three years.")

box([("Note on reading these results.",
      "The sample changed, the methods did not. Where a result differs from the July 2026 "
      "version it is because different products and facilities are in the sample, not "
      "because anything was estimated differently. These should be read fresh rather than "
      "compared line by line against the earlier numbers.")])

p("Methods. Group comparisons use a cluster permutation test on NDC, with Mann-Whitney "
  "reported alongside. Correlations use a Spearman coefficient with an NDC-cluster "
  "bootstrap, 2,000 resamples. Regression models use a random NDC intercept with two-way "
  "clustered standard errors on NDC and facility. Significance markers: * p < 0.05, "
  "** p < 0.01, *** p < 0.001.", italic=True, size=9)

doc.add_page_break()

# ── FIGURE 1 ──────────────────────────────────────────────────────────────────
doc.add_heading("Figure 1. Market outcomes by prior inspection outcome", 1)

figure("Figure1_Market_by_Outcome.png",
       "Figure 1. Market volume and Medicaid price by the FDA classification of the most "
       "recent inspection preceding the Valisure test. Full sample, n = 93.")

doc.add_heading("Descriptive volume, IQVIA extended units", 2)
table(
    ["Prior outcome", "n", "Mean", "Median", "P25", "P75"],
    [
        ["NAI", "15", "57,315,582", "7,478,303", "4,659,759", "29,341,387"],
        ["VAI", "62", "38,338,010", "2,651,394", "772,523", "27,701,244"],
        ["OAI", "16", "84,875,745", "18,425,376", "1,249,006", "70,743,883"],
    ],
    widths=[1.1, 0.5, 1.3, 1.2, 1.1, 1.2], align_right_from=1,
)

doc.add_heading("Tests", 2)
table(
    ["Test", "Contrast", "n obs", "Clusters", "p"],
    [
        ["Kruskal-Wallis", "All three groups", "93", "n/a", "0.169"],
        ["Bootstrap, NDC-clustered", "NAI vs VAI", "77", "60", "0.124"],
        ["Bootstrap, NDC-clustered", "NAI vs OAI", "31", "23", "0.966"],
        ["Bootstrap, NDC-clustered", "VAI vs OAI", "78", "60", "0.217"],
        ["Bootstrap, FEI-clustered", "NAI vs VAI", "77", "20", "0.363"],
        ["Bootstrap, FEI-clustered", "VAI vs OAI", "78", "20", "0.540"],
    ],
    widths=[1.9, 1.5, 0.7, 0.8, 0.7], align_right_from=2,
)

p("Regression, log volume on prior outcome, reference NAI:", bold=True)
table(
    ["Specification", "n obs", "NDCs", "FEIs", "VAI beta (SE)", "p", "OAI beta (SE)", "p"],
    [
        ["Full sample", "93", "67", "22", "-0.506 (1.100)", "0.647", "+0.177 (1.471)", "0.905"],
        ["Single-FEI", "76", "56", "18", "-0.447 (1.096)", "0.685", "+0.028 (1.266)", "0.982"],
        ["Gap 36mo", "67", "53", "18", "-0.491 (0.972)", "0.615", "+0.103 (1.339)", "0.939"],
        ["Single-FEI + Gap", "50", "42", "14", "-0.489 (1.051)", "0.644", "-0.343 (1.233)", "0.782"],
    ],
    size=9, widths=[1.3, 0.55, 0.5, 0.5, 1.15, 0.55, 1.15, 0.55], align_right_from=1,
)

doc.add_heading("What it shows", 2)
p("No relationship between a facility's prior inspection outcome and the market volume of "
  "the drug. Nothing approaches significance under any test or any subset, and the "
  "coefficients are small relative to their standard errors. The result is stable: the "
  "four specifications give VAI coefficients between -0.45 and -0.51, so the finding does "
  "not depend on how the sample is restricted.")

p("The descriptive means look odd at first glance, with OAI products having the highest "
  "mean and median volume. This is driven by a handful of very large products and is not "
  "statistically distinguishable from the other groups. The intraclass correlation is high "
  "throughout (0.70 to 0.85), meaning most of the variance in volume is between products "
  "rather than within them, which is what limits the power of this test.")

box([("Read this as a null result, not a weak one.",
      "With 16 OAI observations the test is underpowered for small effects, but the "
      "estimates are near zero rather than large and imprecise. There is no evidence here "
      "that inspection outcomes track market size.")])

doc.add_page_break()

# ── FIGURE 2 ──────────────────────────────────────────────────────────────────
doc.add_heading("Figure 2. Market volume versus tested drug quality", 1)

figure("Figure2_Volume_vs_Quality.png",
       "Figure 2. Valisure-measured impurity levels against IQVIA extended units, pooled "
       "across test years. Full sample.")

doc.add_heading("Spearman correlations, NDC-cluster bootstrap, 2,000 resamples", 2)
table(
    ["Metric", "Subset", "rho", "n", "NDCs", "p", "95% CI"],
    [
        ["DMF", "Full sample", "+0.232", "96", "70", "**0.034 ***", "[+0.011, +0.430]"],
        ["DMF", "Single-FEI", "+0.127", "79", "59", "0.297", "[-0.121, +0.363]"],
        ["DMF", "Gap 36mo", "+0.232", "70", "56", "0.080", "[-0.039, +0.473]"],
        ["DMF", "Single-FEI + Gap", "+0.107", "53", "45", "0.472", "[-0.186, +0.392]"],
        ["NDMA", "Full sample", "-0.308", "55", "47", "**0.009 ****", "[-0.517, -0.060]"],
        ["NDMA", "Single-FEI", "-0.263", "46", "39", "0.064", "[-0.508, +0.056]"],
        ["NDMA", "Gap 36mo", "-0.280", "46", "42", "**0.028 ***", "[-0.501, -0.019]"],
        ["NDMA", "Single-FEI + Gap", "-0.223", "37", "34", "0.145", "[-0.494, +0.121]"],
        ["Diff. Factor", "Full sample", "-0.330", "19", "19", "0.182", "[-0.757, +0.193]"],
        ["Diff. Factor", "Single-FEI", "-0.385", "17", "17", "0.125", "[-0.805, +0.201]"],
    ],
    size=9, widths=[0.9, 1.3, 0.6, 0.45, 0.5, 0.8, 1.3], align_right_from=2,
)

doc.add_heading("What it shows", 2)
p("Two correlations reach significance in the full sample and they point in opposite "
  "directions. Higher-volume products carry more DMF (rho = +0.23) and less NDMA "
  "(rho = -0.31). Both are modest in size.")

p("Neither survives the single-FEI restriction. DMF falls to +0.13 with p = 0.30, and NDMA "
  "to -0.26 with p = 0.064. The NDMA result does survive the recency restriction "
  "(p = 0.028), so it is the multi-plant NDCs specifically, not the older inspections, that "
  "the DMF result depends on. That is worth flagging: the eleven NDCs we cannot attribute "
  "to a single plant are doing a meaningful share of the work in the one positive finding.")

p("The Difference Factor panel has only 19 observations, all from 2024, and is not "
  "informative either way.")

box([("Recommendation.",
      "If Figure 2 appears in the paper, report the single-FEI column alongside the full "
      "sample rather than in a supplement. A reviewer who notices that the headline "
      "correlation disappears under a defensible robustness restriction will treat its "
      "absence from the main text as a problem.")])

doc.add_page_break()

# ── FIGURE 3 ──────────────────────────────────────────────────────────────────
doc.add_heading("Figure 3. Price versus tested drug quality", 1)

figure("Figure3_Price_vs_Quality.png",
       "Figure 3. Valisure-measured impurity levels against Medicaid price per unit, "
       "outliers above $50 per unit excluded. Full sample.")

doc.add_heading("Spearman correlations, NDC-cluster bootstrap, 2,000 resamples", 2)
table(
    ["Metric", "Subset", "rho", "n", "NDCs", "p", "95% CI"],
    [
        ["DMF", "Full sample", "-0.014", "93", "67", "0.911", "[-0.247, +0.237]"],
        ["DMF", "Single-FEI", "-0.009", "76", "56", "0.953", "[-0.290, +0.276]"],
        ["DMF", "Gap 36mo", "-0.014", "67", "53", "0.907", "[-0.254, +0.258]"],
        ["NDMA", "Full sample", "-0.091", "52", "44", "0.598", "[-0.406, +0.226]"],
        ["NDMA", "Single-FEI", "-0.073", "43", "36", "0.710", "[-0.434, +0.320]"],
        ["NDMA", "Gap 36mo", "-0.067", "43", "39", "0.678", "[-0.359, +0.266]"],
        ["Diff. Factor", "Full sample", "+0.135", "19", "19", "0.589", "[-0.312, +0.597]"],
        ["Diff. Factor", "Single-FEI", "+0.157", "17", "17", "0.547", "[-0.378, +0.633]"],
    ],
    size=9, widths=[0.9, 1.3, 0.6, 0.45, 0.5, 0.8, 1.3], align_right_from=2,
)

doc.add_heading("What it shows", 2)
p("Nothing. Every correlation is near zero, no p-value is below 0.48, and every confidence "
  "interval straddles zero comfortably. This holds across all three quality metrics and all "
  "subsets. Price and measured quality are unrelated in these data.")

p("This is the cleanest null in the set and the most stable across specifications, which "
  "makes it a usable finding rather than an absence of one. Buyers paying more are not "
  "getting a product with fewer impurities.")

box([("Caveat that belongs in the text.",
      "Price here is Medicaid price per unit, which reflects a reimbursement formula more "
      "than a market-clearing price. The absence of a quality signal in this measure does "
      "not establish that no price series would show one.")])

doc.add_page_break()

# ── FIGURE 4 ──────────────────────────────────────────────────────────────────
doc.add_heading("Figure 4. Drug quality by country of manufacture", 1)

figure("Figure4_Quality_by_Country.png",
       "Figure 4. Mean Valisure-measured impurity levels by the country of the "
       "manufacturing facility. India, China and the United States. Full sample.")

doc.add_heading("Group means", 2)
table(
    ["Metric", "Years", "Statistic", "India", "China", "United States"],
    [
        ["DMF", "2020, 2022, 2024", "Mean", "17,600", "3,470", "2,620"],
        ["DMF", "2020, 2022, 2024", "Median", "5,850", "1,910", "386"],
        ["DMF", "2020, 2022, 2024", "n", "67", "17", "12"],
        ["NDMA", "2020, 2022", "Mean", "29.4", "2.04", "0"],
        ["NDMA", "2020, 2022", "Median", "0", "0", "0"],
        ["NDMA", "2020, 2022", "n", "35", "13", "7"],
        ["Difference Factor", "2024", "Mean", "0.251", "0.220", "0.150"],
        ["Difference Factor", "2024", "n", "13", "4", "2"],
    ],
    size=9, widths=[1.2, 1.3, 0.8, 0.9, 0.8, 1.1], align_right_from=3,
)

doc.add_heading("Tests, cluster permutation on NDC", 2)
table(
    ["Metric", "Contrast", "n obs", "Clusters", "Permutation p", "Mann-Whitney p"],
    [
        ["DMF", "India vs China", "84", "61", "0.192", "0.089"],
        ["DMF", "India vs USA", "79", "59", "0.108", "0.062"],
        ["DMF", "China vs USA", "29", "20", "0.714", "0.702"],
        ["NDMA", "India vs China", "48", "40", "0.121", "0.101"],
        ["NDMA", "India vs USA", "42", "37", "0.175", "0.119"],
        ["NDMA", "China vs USA", "20", "17", "1.000", "0.529"],
        ["Diff. Factor", "India vs China", "17", "17", "0.760", "0.777"],
    ],
    size=9, widths=[0.9, 1.4, 0.6, 0.7, 1.2, 1.1], align_right_from=2,
)

p("Kruskal-Wallis across all three countries: DMF p = 0.064, NDMA p = 0.086, "
  "Difference Factor p = 0.769. None significant.")

box([("These pairwise p-values are corrected.",
      "The first run of this pipeline reported NDMA India versus USA at p = 0.0005, India versus China at "
      "0.025 and China versus USA at 0.033, and DMF India versus USA at 0.038. Those came "
      "from a centred-bootstrap approximation that breaks down when the data are dominated "
      "by ties, which NDMA is: 44 of 55 observations are exactly zero. Measured against "
      "simulated null data, that test rejects at 20 to 26 percent rather than 5. Section 6 "
      "of the pipeline changes document has the calibration table. The figures themselves "
      "are unchanged; only the pairwise significance testing was wrong.")])

p("Regression, log1p(metric) on country, reference United States:", bold=True)
table(
    ["Metric", "n obs", "NDCs", "FEIs", "India beta (SE)", "p", "China beta (SE)", "p"],
    [
        ["DMF", "93", "67", "22", "+1.845 (1.818)", "0.313", "-0.152 (1.669)", "0.928"],
        ["NDMA", "55", "47", "14", "+1.174 (0.464)", "**0.014 ***", "+0.309 (0.287)", "0.286"],
        ["Diff. Factor", "19", "19", "n/a", "+0.079 (0.034)", "**0.036 ***", "+0.058 (0.059)", "0.341"],
    ],
    size=9, widths=[0.95, 0.55, 0.5, 0.5, 1.25, 0.6, 1.25, 0.55], align_right_from=1,
)

doc.add_heading("What it shows", 2)
p("The pairwise rank tests show no significant difference in measured impurity between "
  "India, China and the United States. Every permutation test is above 0.10 and the omnibus "
  "Kruskal-Wallis tests are not significant either.")

p("The regression tells a slightly different story and is the more informative of the two. "
  "The India coefficient on NDMA is +1.174 with p = 0.014. That model was validated "
  "independently by permuting country labels across NDCs and refitting, which gives "
  "p = 0.076. So the effect is real in direction and marginal in strength: suggestive, not "
  "established, in this sample.")

box([("This finding was stronger in the earlier sample, and that earlier result stands.",
      "On the manual-map panel the same coefficient was +1.709 with a permutation p of 0.007. "
      "The conclusion previously shared with the team is supported by a valid test and does "
      "not need revisiting. It weakens here because the sample is smaller, 55 observations "
      "across 47 NDCs against 71 across 62, not because the earlier analysis was wrong.")])

p("The descriptive gap is nonetheless large and worth describing. Indian product has a mean "
  "DMF of 17,600 against 2,620 for US product, and a mean NDMA of 29.4 against 0. What the "
  "tests say is that with 12 US and 17 Chinese DMF observations, and 7 US and 13 Chinese "
  "NDMA observations, this sample cannot establish that the gap is more than sampling "
  "variation.")

p("The NDMA comparison deserves particular care. All three country medians are zero. The "
  "entire difference sits in how many products register any detectable NDMA at all: 10 of "
  "35 Indian observations, 1 of 13 Chinese, 0 of 7 US. Framed as a detection rate that is a "
  "clear descriptive contrast, but Fisher's exact test on 10 of 35 against 0 of 7 gives "
  "p = 0.168. Seven US observations cannot carry a claim about US manufacturing.")

p("The Difference Factor coefficient, India at p = 0.036, should not be leaned on at all: "
  "19 observations, of which 2 are American.")

box([("Suggested framing.",
      "Report the descriptive country gap, note that the India NDMA effect is directionally "
      "consistent and marginal here after being significant in the larger earlier sample, and "
      "let that carry the sample-size argument for the 14-drug analysis. Do not claim a "
      "significant country effect from the pairwise rank tests in this sample; a reviewer "
      "asking for the two-by-two table would not be satisfied.")])

doc.add_page_break()

# ── FIGURE S1 ─────────────────────────────────────────────────────────────────
doc.add_heading("Figure S1. Months since last inspection", 1)

figure("FigureS1_Months_Since_Inspection.png",
       "Figure S1. Distribution of the interval between the most recent prior FDA inspection "
       "and the Valisure test, by test year. n = 93.")

table(
    ["Test year", "n", "Mean", "Median", "IQR", "Range", "Over 36 months"],
    [
        ["2020", "20", "13.2", "6.8", "3 to 23", "1 to 45", "1"],
        ["2022", "35", "29.2", "24.6", "23 to 32", "0 to 57", "8"],
        ["2024", "38", "31.4", "29.9", "10 to 47", "5 to 81", "17"],
    ],
    widths=[0.9, 0.5, 0.7, 0.8, 1.0, 1.0, 1.2], align_right_from=1,
)

doc.add_heading("What it shows", 2)
p("The gap between inspection and test widens sharply over the study period. Median 6.8 "
  "months in 2020, 24.6 in 2022, 29.9 in 2024. By 2024, 17 of 38 observations rest on an "
  "inspection more than three years old, against 1 of 20 in 2020.")

p("This is the COVID inspection backlog showing up in the data. It is the reason the Gap "
  "36-month subset exists and the reason it costs so much sample: restricting to recent "
  "inspections removes 26 of 96 observations, and most of the loss is in 2024.")

p("It also means the inspection signal is weakest exactly where the Valisure coverage is "
  "richest. 2024 has the most tested products and the stalest inspection data. Any analysis "
  "pooling across years is implicitly averaging a well-measured 2020 against a "
  "poorly-measured 2024.")

doc.add_page_break()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
doc.add_heading("Summary across figures", 1)

table(
    ["Figure", "Finding", "Under single-FEI", "Under Gap 36mo"],
    [
        ["Fig 1, volume by outcome", "Null, stable", "Yes, still null", "Yes, still null"],
        ["Fig 2, DMF vs volume", "rho = +0.23, p = 0.034", "**No, p = 0.30**", "No, p = 0.080"],
        ["Fig 2, NDMA vs volume", "rho = -0.31, p = 0.009", "**No, p = 0.064**", "Yes, p = 0.028"],
        ["Fig 3, price vs quality", "Null, stable", "Yes, still null", "Yes, still null"],
        ["Fig 4, NDMA by country", "Marginal, regression p = 0.076", "Not applicable", "Not applicable"],
        ["Fig 4, DMF by country", "Not significant, p = 0.108", "Not applicable", "Not applicable"],
    ],
    size=9, widths=[1.7, 1.6, 1.5, 1.5],
)

p("Three things to decide before this goes in the paper:", bold=True)
bullet("Whether Figure 2 leads with the full sample or the single-FEI sample. Both "
       "significant correlations lose significance under the single-FEI restriction, so the "
       "choice materially changes what the paper claims.")
bullet("How to present Figure 4, where the pairwise rank tests are null but the India NDMA "
       "regression coefficient is marginal at a permutation p of 0.076, having been "
       "significant in the larger earlier sample. The descriptive gap is large and the "
       "sample is small, which is an argument for the 14-drug analysis rather than a null "
       "to bury. Single-FEI variants are not needed here: every multi-plant NDC has both "
       "plants in the same country, so the country label is unambiguous either way.")
bullet("Whether the cluster permutation test is the right replacement for the group "
       "comparisons. It is applied because the old bootstrap produced results refutable "
       "with a two-by-two table, but the choice is worth confirming.")

p("Files: all figures are in processed/outputs/ as both PNG and PDF, in four variants each "
  "for Figures 1 to 3. The complete statistical output, including tests not reproduced here, "
  "is in step6_run_log_rulebased.txt in the same folder.", italic=True, size=9)


OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"Saved: {OUT}")
# %%
