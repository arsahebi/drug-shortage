# %%
"""
Build comparison Word document: Pre-revision vs July 2026
"""
from pathlib import Path
from pdf2image import convert_from_path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import tempfile

BASE    = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
OLD_FIG = BASE / "Paper/Metformin"
NEW_FIG = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs"
OUT_DOC = NEW_FIG / "comparison_prerevision_vs_july2026.docx"

# ── helpers ───────────────────────────────────────────────────────────────────
def pdf_to_png(pdf_path: Path, tmp_dir: str, dpi: int = 180) -> Path:
    imgs = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    out = Path(tmp_dir) / (pdf_path.stem + ".png")
    imgs[0].save(str(out), "PNG")
    return out

def shade_cell(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    return doc.add_heading(text, level=level)

def add_note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    return p

def add_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    # header row
    hdr = t.rows[0]
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        shade_cell(c, "2F5496")
        c.text = h
        run = c.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # data rows
    for ri, row_data in enumerate(rows):
        tr = t.rows[ri + 1]
        fill = "EEF2FA" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row_data):
            c = tr.cells[ci]
            shade_cell(c, fill)
            c.text = str(val)
            c.paragraphs[0].runs[0].font.size = Pt(9)
    if col_widths:
        for row in t.rows:
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Inches(w)
    return t

def add_fig_pair(doc, label_old, old_img, label_new, new_img, width=3.0):
    doc.add_paragraph()
    tbl = doc.add_table(rows=2, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ci, lbl in enumerate([label_old, label_new]):
        c = tbl.rows[0].cells[ci]
        p = c.paragraphs[0]
        run = p.add_run(lbl)
        run.bold = True
        run.font.size = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ci, img in enumerate([old_img, new_img]):
        c = tbl.rows[1].cells[ci]
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if img and Path(img).exists():
            run = p.add_run()
            run.add_picture(str(img), width=Inches(width))
        else:
            p.add_run("(not available)")
    doc.add_paragraph()

# ── convert old PDFs ──────────────────────────────────────────────────────────
print("Converting old figures...")
with tempfile.TemporaryDirectory() as tmpdir:
    old = {
        1: pdf_to_png(OLD_FIG / "HealthAffairsScholars_Fig1_Price_Volume_by_Inspection.pdf", tmpdir),
        2: pdf_to_png(OLD_FIG / "HealthAffairsScholars_Fig2_Quality_vs_Volume.pdf", tmpdir),
        3: pdf_to_png(OLD_FIG / "HealthAffairsScholars_Fig3_Quality_vs_Price.pdf", tmpdir),
        4: pdf_to_png(OLD_FIG / "HealthAffairsScholars_Fig4_Quality_by_Country.pdf", tmpdir),
    }
    new = {
        1: NEW_FIG / "Figure1_Market_by_Outcome.png",
        2: NEW_FIG / "Figure2_Volume_vs_Quality.png",
        3: NEW_FIG / "Figure3_Price_vs_Quality.png",
        4: NEW_FIG / "Figure4_Quality_by_Country.png",
    }

    # ── build document ────────────────────────────────────────────────────────
    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(1.8)
        section.bottom_margin = Cm(1.8)
        section.left_margin   = Cm(2.0)
        section.right_margin  = Cm(2.0)

    # Title
    t = doc.add_heading("Metformin Analysis: Pre-Revision vs July 2026 Comparison", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_note(doc, "Prior inspection rule: EventYear strictly < TestYear (same-year inspections excluded).")
    add_note(doc, "Pre-revision: Health Affairs Scholars 2026-05-29  |  New pipeline: Steps 1–6, Redica July 2026 refresh")
    doc.add_paragraph()

    # ── Universe counts ───────────────────────────────────────────────────────
    add_heading(doc, "Universe Counts", 1)
    add_table(doc,
        headers=["", "Pre-revision", "July 2026"],
        rows=[
            ["Total NDC11s (Valisure file)", "112", "112"],
            ["NDCs with FEI found", "88", "89 (23 without FEI)"],
            ["NDCs excluded (CAN / BGD)", "6 (CAN=4, BGD=2)", "5 (CAN=3, BGD=2)"],
            ["NDCs in analysis (IND/CHN/USA)", "82", "84"],
            ["Unique FEIs (IND/CHN/USA)", "15", "28"],
            ["Redica classified FDA inspections", "82  (18 FEIs, through 2025)", "195  (29 FEIs, through May 2026)"],
        ],
        col_widths=[2.8, 1.9, 2.0]
    )
    add_note(doc, "Pre-revision: IND=54 · USA=16 · CHN=12 · CAN=4 · BGD=2   |   July 2026: IND=55 · USA=17 · CHN=12 · CAN=3 · BGD=2")

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "Figure 1 — Market Outcomes by Prior FDA Inspection Outcome", 1)
    add_note(doc, "Left panel: NADAC price (blank in July 2026 — not yet in pipeline). Right panel: IQVIA extended units by inspection outcome.")
    add_fig_pair(doc, "Pre-revision (Health Affairs Scholars)", old[1],
                      "July 2026 (strict rule)", new[1], width=2.95)

    add_heading(doc, "NDC-Year Counts", 2)
    add_table(doc,
        headers=["", "Pre-revision", "July 2026 (strict rule)"],
        rows=[
            ["Total NDC-year obs (volume panel)", "110", "221"],
            ["Unique NDC11s", "82 (IND/CHN/USA)", "81"],
            ["NAI obs  (unique NDC11s)", "64  (—)", "26  (25)"],
            ["VAI obs  (unique NDC11s)", "33  (—)", "155  (77)"],
            ["OAI obs  (unique NDC11s)", "13  (—)", "40  (29)"],
        ],
        col_widths=[2.8, 1.9, 2.0]
    )
    doc.add_paragraph()
    add_heading(doc, "Distribution by Year (July 2026)", 2)
    add_table(doc,
        headers=["Outcome", "2020", "2022", "2024"],
        rows=[
            ["NAI", "20", "6", "4"],
            ["VAI", "45", "63", "64"],
            ["OAI", "16", "12", "13"],
        ],
        col_widths=[1.5, 1.5, 1.5, 1.5]
    )
    doc.add_paragraph()
    add_heading(doc, "OAI Facilities — July 2026 (4 FEIs)", 2)
    add_table(doc,
        headers=["FEI", "Country", "Prior Insp Year", "n_obs", "n_ndc", "Test Years", "Facility"],
        rows=[
            ["3002984011", "IND", "2019", "16", "8", "2020, 2022", "Zydus Lifesciences (Sanand)"],
            ["3007373532", "IND", "2019", "8",  "4", "2020, 2022", "Aurobindo Pharma (Jadcherla)"],
            ["3004819820", "IND", "2019", "4",  "4", "2020",       "Lupin Limited (Mormugao)"],
            ["3008298016", "USA", "2023", "13", "13","2024",       "ScieGen Pharmaceuticals (Hauppauge)"],
        ],
        col_widths=[0.9, 0.7, 0.9, 0.5, 0.5, 0.9, 2.2]
    )
    doc.add_paragraph()
    add_heading(doc, "NAI Facilities — July 2026 (6 FEIs)", 2)
    add_table(doc,
        headers=["FEI", "Country", "n_obs", "n_ndc", "Facility"],
        rows=[
            ["3004097901", "IND", "17", "17", "Granules India (Qutubullapur) — prior NAI 2018"],
            ["3006346108", "CHN", "4",  "2",  "Novast Laboratories (Nantong)"],
            ["3008565058", "IND", "3",  "1",  "Glenmark Pharmaceuticals (Dhar)"],
            ["3005263655", "USA", "2",  "2",  "Amneal Pharmaceuticals of NY (Centereach)"],
            ["3008223599", "IND", "2",  "1",  "Amneal Pharmaceuticals (Bavla)"],
            ["3010254278", "IND", "2",  "2",  "Amneal Pharmaceuticals (Sanand)"],
        ],
        col_widths=[1.0, 0.7, 0.5, 0.5, 3.9]
    )
    doc.add_paragraph()
    add_heading(doc, "Primary Model B — Volume by Outcome (RE + CGM Two-Way Clustered SE)", 2)
    add_table(doc,
        headers=["Coefficient", "Pre-revision", "July 2026 (strict rule)"],
        rows=[
            ["VAI vs NAI", "β=−1.820, SE=0.801, 95%CI [−3.389,−0.250], p=0.025 *",
                           "β=+2.311, SE=1.421, 95%CI [−0.474,+5.096], p=0.105"],
            ["OAI vs NAI", "β=+1.747, SE=0.952, 95%CI [−0.120,+3.613], p=0.069",
                           "β=+2.182, SE=1.895, 95%CI [−1.533,+5.897], p=0.251"],
            ["OAI vs VAI", "β=+3.566, SE=0.782, 95%CI [+2.033,+5.100], p<0.001 **",
                           "(implied ≈−0.129, ns)"],
        ],
        col_widths=[1.4, 2.7, 2.6]
    )
    doc.add_paragraph()
    add_heading(doc, "Descriptive Volume (July 2026)", 2)
    add_table(doc,
        headers=["Outcome", "n", "Mean", "Median", "P25", "P75"],
        rows=[
            ["NAI", "26",  "24,745,069", "862,990",   "47,187",  "6,535,536"],
            ["VAI", "155", "38,744,119", "2,483,318", "363,740", "16,434,334"],
            ["OAI", "40",  "46,534,770", "1,453,286", "212,008", "8,824,912"],
        ],
        col_widths=[0.8, 0.5, 1.4, 1.3, 1.2, 1.4]
    )
    add_note(doc, "Conclusion: Old Observation 1 (volume) NOT SUPPORTED. VAI p=0.105, OAI p=0.251 — both non-significant. Direction reversed (new: NAI<VAI<OAI; old claimed NAI>VAI). Paper text needs updating.")

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "Figure 2 — Market Volume vs Tested Drug Quality", 1)
    add_note(doc, "All years pooled within each metric. Spearman ρ with NDC-cluster block bootstrap (2000 resamples).")
    add_fig_pair(doc, "Pre-revision (Health Affairs Scholars)", old[2],
                      "July 2026 (new pipeline)", new[2], width=2.95)

    add_heading(doc, "NDC-Year Counts", 2)
    add_table(doc,
        headers=["Panel", "Pre-revision", "July 2026"],
        rows=[
            ["DMF  (2020+2022+2024)", "n=111  (82 NDC11s)", "n=126  (94 NDC11s)"],
            ["NDMA  (2020+2022)",     "n=63   (54 NDC11s)", "n=71   (62 NDC11s)"],
            ["Difference Factor (2024)", "n=48 (48 NDC11s)", "n=25 (25 NDC11s)"],
        ],
        col_widths=[2.3, 2.2, 2.2]
    )
    doc.add_paragraph()
    add_heading(doc, "Spearman ρ — Volume vs Quality", 2)
    add_table(doc,
        headers=["Association", "Pre-revision", "July 2026"],
        rows=[
            ["DMF vs Volume",        "ρ=+0.279, p=0.004, 95%CI [+0.064,+0.454]",
                                     "ρ=+0.302, p_boot=0.002, 95%CI [+0.112,+0.467], n=126 **"],
            ["NDMA vs Volume",       "not significant (p>0.10)",
                                     "ρ=−0.064, p_boot=0.635, n=71"],
            ["Diff Factor vs Volume","not significant (p>0.10)",
                                     "ρ=−0.162, p_boot=0.454, n=25"],
        ],
        col_widths=[1.8, 2.4, 2.5]
    )
    add_note(doc, "Conclusion: Old Observation 2 (DMF vs volume) STILL SUPPORTED — strengthened (ρ=+0.30, p=0.002). NDMA and DiffFactor vs volume remain non-significant.")

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "Figure 3 — Price vs Tested Drug Quality", 1)
    add_note(doc, "July 2026 figure is blank — NADAC price data not yet integrated into the pipeline.")
    add_fig_pair(doc, "Pre-revision (Health Affairs Scholars)", old[3],
                      "July 2026 (NADAC pending)", new[3], width=2.95)

    add_heading(doc, "Spearman ρ — Price vs Quality", 2)
    add_table(doc,
        headers=["Association", "Pre-revision", "July 2026"],
        rows=[
            ["DMF vs Price",        "not significant (p>0.10)", "pending NADAC integration"],
            ["NDMA vs Price",       "ρ=+0.282, p=0.013, 95%CI [+0.056,+0.490] *",
                                    "pending NADAC integration"],
            ["Diff Factor vs Price","not significant (p>0.10)", "pending NADAC integration"],
        ],
        col_widths=[1.8, 2.7, 2.2]
    )
    add_note(doc, "NADAC coverage in Q&A file: 107/111 pre-revision NDC-years (2020: 25/27, 2022: 38/38, 2024: 44/46 for IND/CHN/USA).")
    add_note(doc, "Conclusion: Cannot evaluate. Old Observation 2 price finding (NDMA vs price ρ=+0.282, p=0.013) pending NADAC integration.")

    # ── Figure 4 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "Figure 4 — Drug Quality by Country of Manufacture", 1)
    add_note(doc, "Primary model: MixedLM random NDC intercept + CGM two-way clustered SE, reference = USA.")
    add_fig_pair(doc, "Pre-revision (Health Affairs Scholars)", old[4],
                      "July 2026 (new pipeline)", new[4], width=2.95)

    add_heading(doc, "NDC-Year Counts", 2)
    add_table(doc,
        headers=["Panel", "Pre-revision", "July 2026"],
        rows=[
            ["DMF  (all years)",       "n=111 (IND=79, CHN=18, USA=14)", "n=127 (IND=87, CHN=18, USA=22)"],
            ["NDMA  (2020+2022)",       "n=63  (IND=44, CHN=13, USA=6)",  "n=71  (IND=46, CHN=13, USA=12)"],
            ["Difference Factor (2024)","n=48  (IND=30, CHN=10, USA=8)",  "n=25  (IND=16, CHN=5,  USA=4)"],
        ],
        col_widths=[2.3, 2.5, 2.5]
    )
    doc.add_paragraph()
    add_heading(doc, "Descriptive Means (July 2026)", 2)
    add_table(doc,
        headers=["Metric", "IND", "CHN", "USA"],
        rows=[
            ["DMF mean (ng/day)",   "28,607", "3,355", "4,696"],
            ["NDMA mean (ng/day)",  "65.2",   "2.0",   "0.0"],
            ["Diff Factor mean",    "0.261",  "0.226", "0.153"],
        ],
        col_widths=[2.0, 1.5, 1.5, 1.5]
    )
    doc.add_paragraph()
    add_heading(doc, "Model B Results — Quality by Country", 2)
    add_table(doc,
        headers=["Metric / Comparison", "Pre-revision", "July 2026"],
        rows=[
            ["DMF: IND vs USA",        "not significant",          "β=+2.650, p=0.136 (ns)"],
            ["DMF: CHN vs USA",        "—",                        "β=+0.277, p=0.867 (ns)"],
            ["DMF: CHN vs IND",        "—",                        "β=−2.374, p=0.092 (marginal)"],
            ["NDMA: IND vs USA",       "β=+1.345, p<0.001 **",     "β=+1.603, p=0.014 *"],
            ["NDMA: CHN vs USA",       "not significant",           "β=+0.310, p=0.284 (ns)"],
            ["NDMA: CHN vs IND",       "β=−1.090, p=0.022 *",      "β=−1.293, p=0.067 (marginal)"],
            ["DiffFactor: IND vs USA", "β=+0.117, p=0.011 *",      "β=+0.074, p=0.061 (marginal)"],
            ["DiffFactor: CHN vs USA", "not significant",           "β=+0.060, p=0.185 (ns)"],
            ["DiffFactor: CHN vs IND", "not significant",           "β=−0.015, p=0.802 (ns)"],
        ],
        col_widths=[2.4, 2.1, 2.2]
    )
    add_note(doc, "Conclusion: NDMA India>USA STILL SUPPORTED (p=0.014). NDMA China<India and DiffFactor India>USA WEAKENED to marginal (p=0.067, p=0.061). DMF country differences CONSISTENT (ns in both). Marginal results should be softened in paper text.")

    doc.save(str(OUT_DOC))
    print(f"Saved: {OUT_DOC}")
# %%
