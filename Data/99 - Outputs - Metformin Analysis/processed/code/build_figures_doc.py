# %%
"""
Build the September 2026 figures report for the Metformin analysis.

Every figure regenerated on the rule-based NDC-FEI map and the two sample
exclusions, each with its statistics and a short reading of what it shows.

Style follows 483_Worked_Example_FEI3003342394_obs3.docx: Times New Roman,
black and white, thin ruled tables, grey header rows, no accent colours.

Output: outputs/20260910_metformin_figures_rulebased.docx
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
FIG  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs"
OUT  = FIG / "20260910_metformin_figures_rulebased.docx"

BLACK   = RGBColor(0x00, 0x00, 0x00)
HDR_FILL = "D9D9D9"
BOX_FILL = "F2F2F2"

doc = Document()

# ── base style: Times New Roman, black, 1in margins ──────────────────────────
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(11)
normal.font.color.rgb = BLACK
normal.paragraph_format.space_after = Pt(8)
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

for name, size in [("Title", 20), ("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 11)]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.italic = False
    st.font.color.rgb = BLACK
    if st.element.rPr is not None and st.element.rPr.rFonts is not None:
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

for s in doc.sections:
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)


# ── helpers ───────────────────────────────────────────────────────────────────
def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def set_borders(tbl, sz=4):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tbl._tbl.tblPr.append(borders)


def p(text="", bold=False, italic=False, size=11, align=None, space_after=None):
    par = doc.add_paragraph()
    if align is not None:
        par.alignment = align
    if space_after is not None:
        par.paragraph_format.space_after = Pt(space_after)
    if text:
        r = par.add_run(text)
        r.bold, r.italic = bold, italic
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
        r.font.color.rgb = BLACK
    return par


def bullet(text, size=11):
    par = doc.add_paragraph(style="List Bullet")
    r = par.add_run(text)
    r.font.size = Pt(size)
    r.font.name = "Times New Roman"
    r.font.color.rgb = BLACK
    return par


def table(headers, rows, size=9.5, widths=None, align_right_from=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t)
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        shade(c, HDR_FILL)
        c.text = ""
        par = c.paragraphs[0]
        par.paragraph_format.space_after = Pt(2)
        r = par.add_run(htxt)
        r.bold = True
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            if align_right_from is not None and i >= align_right_from:
                par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            txt = str(val)
            bold = txt.startswith("**") and txt.endswith("**")
            if bold:
                txt = txt[2:-2]
            r = par.add_run(txt)
            r.bold = bold
            r.font.size = Pt(size)
            r.font.name = "Times New Roman"
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def box(lines, size=10):
    """Single-cell shaded callout, as used in the 483 worked example."""
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t, sz=6)
    c = t.rows[0].cells[0]
    shade(c, BOX_FILL)
    c.text = ""
    par = c.paragraphs[0]
    for i, (label, txt) in enumerate(lines):
        if i:
            par = c.add_paragraph()
        par.paragraph_format.space_after = Pt(3)
        if label:
            r = par.add_run(f"{label}  ")
            r.bold = True
            r.font.size = Pt(size)
            r.font.name = "Times New Roman"
        r = par.add_run(txt)
        r.font.size = Pt(size)
        r.font.name = "Times New Roman"
    doc.add_paragraph()
    return t


def figure(png, caption, width=6.0):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(4)
    path = FIG / png
    if path.exists():
        par.add_run().add_picture(str(path), width=Inches(width))
    else:
        par.add_run("(figure not found)").italic = True
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.name = "Times New Roman"
    r.font.color.rgb = BLACK


def rule():
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(6)
    pPr = par._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"), "single")
    bot.set(qn("w:sz"), "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "000000")
    b.append(bot)
    pPr.append(b)


# ══════════════════════════════════════════════════════════════════════════════
doc.add_heading("Metformin Analysis: Regenerated Figures and Statistics", 0)
p("September 10, 2026", italic=True, size=10)
rule()

p("Every figure in this document was regenerated after two changes to the pipeline: "
  "the NDC to FEI map is now built by a rule rather than by manual search, and two "
  "sample exclusions now apply to the whole analysis. The changes themselves are "
  "documented separately in 20260910_metformin_pipeline_changes.docx. This document "
  "shows what the figures look like now, the statistics behind each one, and a short "
  "reading of what each shows.")

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

p("Significance markers: * p < 0.05, ** p < 0.01, *** p < 0.001. "
  "Primary specification throughout is the NDC-clustered bootstrap or, for the regression "
  "models, a random NDC intercept with two-way clustered standard errors on NDC and facility.",
  italic=True, size=9)

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
    ["Metric", "Years", "India", "China", "United States"],
    [
        ["DMF, mean", "2020, 2022, 2024", "17,600 (n=67)", "3,470 (n=17)", "2,620 (n=12)"],
        ["DMF, median", "", "5,850", "1,910", "386"],
        ["NDMA, mean", "2020, 2022", "29.4 (n=35)", "2.04 (n=13)", "0 (n=7)"],
        ["NDMA, median", "", "0", "0", "0"],
        ["Diff. Factor, mean", "2024", "0.251 (n=13)", "0.220 (n=4)", "0.150 (n=2)"],
    ],
    size=9, widths=[1.3, 1.4, 1.3, 1.1, 1.3], align_right_from=2,
)

doc.add_heading("Tests, bootstrap clustered on NDC", 2)
table(
    ["Metric", "Contrast", "n obs", "Clusters", "Kruskal-Wallis p", "Bootstrap p"],
    [
        ["DMF", "India vs China", "84", "61", "0.064", "0.081"],
        ["DMF", "India vs USA", "79", "59", "0.064", "**0.038 ***"],
        ["DMF", "China vs USA", "29", "20", "0.064", "0.708"],
        ["NDMA", "India vs China", "48", "40", "0.086", "**0.025 ***"],
        ["NDMA", "India vs USA", "42", "37", "0.086", "**0.0005 *****"],
        ["NDMA", "China vs USA", "20", "17", "0.086", "**0.033 ***"],
        ["Diff. Factor", "India vs China", "17", "17", "0.769", "0.736"],
    ],
    size=9, widths=[0.9, 1.4, 0.6, 0.7, 1.2, 0.9], align_right_from=2,
)

p("Regression, log1p(metric) on country, reference United States:", bold=True)
table(
    ["Metric", "n obs", "NDCs", "FEIs", "India beta (SE)", "p", "China beta (SE)", "p"],
    [
        ["DMF", "93", "67", "22", "+1.845 (1.818)", "0.313", "-0.152 (1.669)", "0.928"],
        ["NDMA", "55", "47", "14", "**+1.174 (0.464)**", "**0.014 ***", "+0.309 (0.287)", "0.286"],
        ["Diff. Factor", "19", "19", "n/a", "**+0.079 (0.034)**", "**0.036 ***", "+0.058 (0.059)", "0.341"],
    ],
    size=9, widths=[0.95, 0.55, 0.5, 0.5, 1.25, 0.6, 1.25, 0.55], align_right_from=1,
)

doc.add_heading("What it shows", 2)
p("This is where the new sample matters most. Indian-manufactured product carries higher "
  "measured impurities than US-manufactured product, and for NDMA the result is strong: "
  "India versus USA at p = 0.0005 in the clustered bootstrap, and a regression coefficient "
  "of +1.17 on the log scale with p = 0.014. All three NDMA country contrasts are "
  "significant. For DMF, India versus USA reaches p = 0.038 in the bootstrap, though the "
  "regression coefficient is not significant once the NDC random effect absorbs "
  "between-product variance.")

p("Two things temper this. The omnibus Kruskal-Wallis tests are not significant (p = 0.064 "
  "for DMF, p = 0.086 for NDMA), so the pairwise bootstrap results are doing the work. And "
  "the US and China cells are small: 12 and 17 observations for DMF, 7 and 13 for NDMA, "
  "and only 2 US observations for Difference Factor, which is too few to interpret.")

p("The NDMA medians are all zero in every country. The country difference is in the upper "
  "tail, not the typical product, and the mean of 29.4 for India against 0 for the US "
  "reflects a subset of Indian products with detectable NDMA rather than a shift in the "
  "whole distribution. That is a meaningful finding but it should be described as such "
  "rather than as Indian product being generally more contaminated.")

box([("This result moved with the new sample.",
      "Under the previous sample these country contrasts were weaker. The change comes from "
      "the composition of the sample, not from any change in method, so the finding needs to "
      "be presented on its own terms rather than as a strengthening of an earlier result.")])

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
    ["Figure", "Finding", "Holds under single-FEI?", "Holds under Gap 36mo?"],
    [
        ["Fig 1, volume by outcome", "Null, stable", "Yes, still null", "Yes, still null"],
        ["Fig 2, DMF vs volume", "rho = +0.23, p = 0.034", "**No, p = 0.30**", "No, p = 0.080"],
        ["Fig 2, NDMA vs volume", "rho = -0.31, p = 0.009", "**No, p = 0.064**", "Yes, p = 0.028"],
        ["Fig 3, price vs quality", "Null, stable", "Yes, still null", "Yes, still null"],
        ["Fig 4, NDMA by country", "India > USA, p = 0.0005", "Not tested by subset", "Not tested by subset"],
        ["Fig 4, DMF by country", "India > USA, p = 0.038", "Not tested by subset", "Not tested by subset"],
    ],
    size=9, widths=[1.7, 1.6, 1.5, 1.5],
)

p("Three things to decide before this goes in the paper:", bold=True)
bullet("Whether Figure 2 leads with the full sample or the single-FEI sample. Both "
       "significant correlations lose significance under the single-FEI restriction, so the "
       "choice materially changes what the paper claims.")
bullet("Whether the country results in Figure 4 warrant the robustness subsets too. Step 6 "
       "does not currently produce single-FEI or Gap 36mo variants of Figure 4, and given "
       "that these are the strongest results in the set, they probably should be checked "
       "the same way.")
bullet("How to frame the NDMA country difference given that all three country medians are "
       "zero. The difference is in the tail and the text should say so.")

p("Files: all figures are in processed/outputs/ as both PNG and PDF, in four variants each "
  "for Figures 1 to 3. The complete statistical output, including tests not reproduced here, "
  "is in step6_run_log_rulebased.txt in the same folder.", italic=True, size=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"Saved: {OUT}")
# %%
