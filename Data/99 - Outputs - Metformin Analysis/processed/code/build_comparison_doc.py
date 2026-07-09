# %%
"""
Build comparison Word document: Pre-revision vs July 2026
Clean academic style — no colored headers.
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
def pdf_to_png(pdf_path, tmp_dir, dpi=200):
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

def add_note(doc, text, size=9):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0x50, 0x50, 0x50)
    return p

def write_cell_content(cell, segments, font_size=9):
    """Write formatted runs into a cell. segments: list of {text, bold, red}."""
    cell.text = ''
    p = cell.paragraphs[0]
    for seg in segments:
        text = seg.get('text', '')
        bold = seg.get('bold', False)
        red  = seg.get('red',  False)
        parts = text.split('\n')
        for i, part in enumerate(parts):
            if i > 0:
                run = p.add_run()
                run.add_break()
            if part:
                run = p.add_run(part)
                run.font.size = Pt(font_size)
                if bold:
                    run.bold = True
                if red:
                    run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)

def simple_table(doc, headers, rows, col_widths=None, zebra=True):
    """Clean academic table: thin borders, E8E8E8 header, no colored data cells.
    Cell values may be str or list of {text, bold, red} dicts for rich formatting."""
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'

    hdr = t.rows[0]
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        shade_cell(c, 'E8E8E8')
        c.text = h
        run = c.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(9)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for ri, row_data in enumerate(rows):
        tr = t.rows[ri + 1]
        fill = 'F5F5F5' if (zebra and ri % 2 == 1) else 'FFFFFF'
        for ci, val in enumerate(row_data):
            c = tr.cells[ci]
            shade_cell(c, fill)
            if isinstance(val, list):
                write_cell_content(c, val)
            else:
                c.text = str(val)
                if c.paragraphs[0].runs:
                    c.paragraphs[0].runs[0].font.size = Pt(9)

    if col_widths:
        for row in t.rows:
            for ci, w in enumerate(col_widths):
                row.cells[ci].width = Inches(w)
    return t

def add_fig_pair(doc, label_old, old_img, label_new, new_img, width=6.2):
    """Old figure above, new figure below — each full-width for readability."""
    for label, img in [(label_old, old_img), (label_new, new_img)]:
        p_lbl = doc.add_paragraph()
        r = p_lbl.add_run(label)
        r.bold = True
        r.font.size = Pt(9)
        p_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER

        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if img and Path(img).exists():
            p_img.add_run().add_picture(str(img), width=Inches(width))
        else:
            p_img.add_run('(not available)')
        doc.add_paragraph()

SIG_NOTE = '(*) p < 0.05;  (**) p < 0.01;  (***) p < 0.001.  Significant p-values shown in bold red.'

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

    # ── build document ─────────────────────────────────────────────────────────
    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.0)
        section.right_margin  = Cm(2.0)

    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10)

    # ── Title ──────────────────────────────────────────────────────────────────
    title = doc.add_heading('Metformin Analysis: Pre-Revision vs July 2026 Comparison', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_note(doc, 'Prior inspection rule: EventYear strictly < TestYear (same-year inspections excluded).')
    add_note(doc, 'Pre-revision: Health Affairs Scholars, submitted 2026-05-29  |  New pipeline: Steps 1–6, Redica July 2026 refresh')
    doc.add_paragraph()

    # ── 1. Universe Counts ─────────────────────────────────────────────────────
    doc.add_heading('1. Universe Counts', 1)
    simple_table(doc,
        headers=['', 'Pre-revision', 'July 2026'],
        rows=[
            ['Total NDC11s (Valisure file)', '112', '112'],
            ['NDC11s with FEI found', '88', '89'],
            ['NDC11s excluded (CAN / BGD)', '6  (CAN=4, BGD=2)', '5  (CAN=3, BGD=2)'],
            ['NDC11s in analysis (IND/CHN/USA)', '82', '84'],
            ['Unique FEIs in analysis', '15', '28'],
            ['Redica: classified FDA inspections', '82  (18 FEIs, through 2025)', '195  (29 FEIs, through May 2026)'],
        ],
        col_widths=[2.8, 2.1, 2.1]
    )
    p = doc.add_paragraph()
    p.add_run('Country breakdown of FEI-mapped NDC11s — ').font.size = Pt(9)
    r = p.add_run('Pre-revision: '); r.bold = True; r.font.size = Pt(9)
    p.add_run('IND=54 · USA=16 · CHN=12 · CAN=4 · BGD=2   ').font.size = Pt(9)
    r3 = p.add_run('July 2026: '); r3.bold = True; r3.font.size = Pt(9)
    p.add_run('IND=55 · USA=17 · CHN=12 · CAN=3 · BGD=2').font.size = Pt(9)

    add_note(doc, (
        'Why counts differ: The NDC→FEI linking was done manually in both versions, but the new version uses '
        'DailyMed drug labels and ProPublica facility data for higher accuracy and explicitly handles NDCs '
        'linked to multiple manufacturing sites (multi-FEI NDCs). '
        'The Redica July 2026 refresh covers inspections through May 2026 and adds 11 FEIs not previously '
        'captured, more than doubling the classified inspection count (82 → 195).'
    ))

    # ── 2. Panel Details ───────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('2. Panel Details and Sample Composition', 1)
    add_note(doc, (
        'All panels restricted to IND, CHN, USA. '
        'The 84 NDC11s in analysis reduce to 81 unique NDC11s in the Figure 1 volume panel '
        'because 3 NDCs have no prior classified inspection before any test year in the Redica data '
        'and are therefore excluded from the inspection-outcome analysis. '
        'Pre-revision data sourced from Q&A file (Q&As1234_v8_v02.xlsx, Sheet1); '
        'prior inspection assigned using the same strict EventYear < TestYear rule applied to July 2026 data.'
    ))
    doc.add_paragraph()

    # 2.1
    doc.add_heading('2.1  NDC-Year Panel Counts by Figure', 2)
    simple_table(doc,
        headers=['Panel', 'Pre-revision', 'July 2026'],
        rows=[
            ['Fig 1 — Volume by inspection outcome', '111 NDC-years, 82 NDC11s', '221 NDC-years, 81 NDC11s'],
            ['Fig 2 — DMF vs volume  (2020+2022+2024)', '111 NDC-years, 82 NDC11s', '126 NDC-years, 94 NDC11s'],
            ['Fig 2 — NDMA vs volume  (2020+2022)', '63 NDC-years, 54 NDC11s', '71 NDC-years, 62 NDC11s'],
            ['Fig 2 — Diff Factor vs volume  (2024)', '48 NDC-years, 48 NDC11s', '25 NDC-years, 25 NDC11s'],
            ['Fig 3 — Quality vs price', 'same as Fig 2  (NADAC available)', 'NADAC not yet in pipeline'],
            ['Fig 4 — DMF by country  (2020+2022+2024)', '111  (IND=79, CHN=18, USA=14)', '127  (IND=87, CHN=18, USA=22)'],
            ['Fig 4 — NDMA by country  (2020+2022)', '63  (IND=44, CHN=13, USA=6)', '71  (IND=46, CHN=13, USA=12)'],
            ['Fig 4 — Diff Factor by country  (2024)', '48  (IND=30, CHN=10, USA=8)', '25  (IND=16, CHN=5, USA=4)'],
        ],
        col_widths=[2.9, 2.1, 2.1]
    )
    add_note(doc, (
        'Why NDC-years are more in July 2026: In the new data, all 84 NDC11s × 3 Valisure test years = 252 '
        'NDC-year combinations are populated (the July 2026 Valisure file covers all three test years for '
        'every NDC in the dataset). Of these, 243 have a prior classified Redica inspection (96%). '
        'In pre-revision, the same 82 NDC11s × 3 years = 246 combinations were possible, but the old '
        'Redica file (82 inspections, 18 FEIs) left most NDC-years without an eligible prior classified '
        'inspection — only 111 of 246 (45%) had the required inspection coverage. '
        'The Fig 1 regression panel uses 221 NDC-years (of 243) after requiring non-null IQVIA volume.'
    ))
    add_note(doc, (
        'Pre-revision paper reported 110 NDC-years (NAI=64, VAI=33, OAI=13) — a minor discrepancy '
        'from the 111 derived here, arising from differences in panel assembly at submission time.'
    ))

    # 2.2
    doc.add_paragraph()
    doc.add_heading('2.2  Figure 1 Panel — Outcome Distribution by Year', 2)
    simple_table(doc,
        headers=['Outcome', 'Pre-rev 2020', 'Pre-rev 2022', 'Pre-rev 2024', 'Pre-rev Total',
                 'New 2020', 'New 2022', 'New 2024', 'New Total'],
        rows=[
            ['NAI', '7',  '16', '16', '39  (25 NDC11s)',  '20', '6',  '4',  '30  (25 NDC11s)'],
            ['VAI', '13', '16', '27', '56  (40 NDC11s)',  '45', '63', '64', '172  (80 NDC11s)'],
            ['OAI', '6',  '5',  '5',  '16  (12 NDC11s)',  '16', '12', '13', '41  (29 NDC11s)'],
            ['Total (w/ prior)', '26', '37', '48', '111  (82 NDC11s)', '81', '81', '81', '243  (84 NDC11s)'],
        ],
        col_widths=[1.05, 0.72, 0.72, 0.72, 1.25, 0.62, 0.62, 0.62, 1.25]
    )
    add_note(doc, (
        'The NDC11 counts in parentheses (e.g., 25+80+29 = 134 for July 2026) sum to more than the universe '
        'of 84 NDC11s because the same NDC11 can carry different inspection outcomes in different test years. '
        'For example, an NDC whose manufacturing site received a VAI inspection prior to 2020 and a NAI '
        'inspection prior to 2024 will be counted in both the VAI group (for TestYear=2020) and the NAI '
        'group (for TestYear=2024). The unique NDC11 count in the Total row (82 / 84) is the unduplicated '
        'universe; the per-outcome NDC11 counts overlap across rows.'
    ))

    # 2.3
    doc.add_paragraph()
    doc.add_heading('2.3  Descriptive Volume by Outcome', 2)
    simple_table(doc,
        headers=['Outcome', 'Pre-rev n', 'Pre-rev Mean', 'Pre-rev Median', 'New n', 'New Mean', 'New Median'],
        rows=[
            ['NAI', '39', '125,591,916', '15,944,388', '26', '24,745,069', '862,990'],
            ['VAI', '56', '30,308,175',  '2,392,105',  '155', '38,744,119', '2,483,318'],
            ['OAI', '16', '85,051,570',  '18,425,376', '40', '46,534,770', '1,453,286'],
        ],
        col_widths=[0.9, 0.8, 1.35, 1.35, 0.7, 1.35, 1.25]
    )
    add_note(doc, (
        'IQVIA extended units. Pre-revision: Granules India (FEI 3004097901) dominates NAI with 30 '
        'NDC-year observations (22 NDC11s), driving the high NAI mean.'
    ))

    # 2.4 — OAI (July 2026 only)
    doc.add_paragraph()
    doc.add_heading('2.4  OAI Facilities — July 2026 (strict rule, 4 FEIs)', 2)
    simple_table(doc,
        headers=['FEI', 'Country', 'Prior Insp Year', 'n_obs', 'n_ndc', 'Test Years', 'Facility'],
        rows=[
            ['3002984011', 'IND', '2019', '16', '8',  '2020, 2022', 'Zydus Lifesciences (Sanand)'],
            ['3007373532', 'IND', '2019', '8',  '4',  '2020, 2022', 'Aurobindo Pharma (Jadcherla)'],
            ['3004819820', 'IND', '2019', '4',  '4',  '2020',       'Lupin Limited (Mormugao)'],
            ['3008298016', 'USA', '2023', '13', '13', '2024',       'ScieGen Pharmaceuticals (Hauppauge, NY)'],
        ],
        col_widths=[0.9, 0.7, 1.1, 0.5, 0.5, 0.9, 2.6]
    )
    add_note(doc, (
        'Zydus/Aurobindo/Lupin: OAI in 2019 is the most recent prior inspection for TestYears 2020 and 2022. '
        'ScieGen: OAI in 2023 becomes prior for TestYear 2024 under the strict rule (same-year 2024 NAI excluded).'
    ))

    # 2.5 — NAI (July 2026 only)
    doc.add_paragraph()
    doc.add_heading('2.5  NAI Facilities — July 2026 (strict rule, 6 FEIs)', 2)
    simple_table(doc,
        headers=['FEI', 'Country', 'n_obs', 'n_ndc', 'Facility'],
        rows=[
            ['3004097901', 'IND', '17', '17', 'Granules India (Qutubullapur) — prior NAI 2018; VAI for 2022 & 2024'],
            ['3006346108', 'CHN', '4',  '2',  'Novast Laboratories (Nantong)'],
            ['3008565058', 'IND', '3',  '1',  'Glenmark Pharmaceuticals (Dhar)'],
            ['3005263655', 'USA', '2',  '2',  'Amneal Pharmaceuticals of New York (Centereach)'],
            ['3008223599', 'IND', '2',  '1',  'Amneal Pharmaceuticals (Bavla)'],
            ['3010254278', 'IND', '2',  '2',  'Amneal Pharmaceuticals (Sanand)'],
        ],
        col_widths=[1.0, 0.75, 0.55, 0.55, 4.25]
    )

    # 2.6 — VAI (July 2026 only)
    doc.add_paragraph()
    doc.add_heading('2.6  VAI Top Facilities — July 2026 (21 FEIs total)', 2)
    simple_table(doc,
        headers=['FEI', 'Country', 'n_obs', 'n_ndc', 'Facility'],
        rows=[
            ['3004097901', 'IND', '34', '17', 'Granules India (Qutubullapur) — 2022 & 2024 only'],
            ['3008298016', 'USA', '26', '13', 'ScieGen Pharmaceuticals (Hauppauge) — 2020 & 2022'],
            ['2000021110', 'CHN', '21', '7',  'CSPC Ouyi Pharmaceutical (Shijiazhuang)'],
            ['3011922870', 'CHN', '9',  '3',  'Qingdao BAHEAL Pharmaceutical (Jimo)'],
            ['3011538548', 'IND', '9',  '3',  'Laurus Labs (Rambilli)'],
            ['3006370533', 'IND', '9',  '3',  'Alkem Laboratories (Baddi)'],
            ['3006230648', 'IND', '9',  '3',  'Marksans Pharma (Mormugao)'],
            ['3007938603', 'IND', '7',  '7',  'Zydus Lifesciences (Sanand — separate unit)'],
            ['3004819820', 'IND', '6',  '4',  'Lupin Limited (Mormugao) — 2022 & 2024'],
            ['3008232264', 'IND', '6',  '2',  'Inventia Healthcare (Ambernath)'],
            ['1930436',    'USA', '6',  '2',  'MPP Pharma (Kansas City)'],
            ['3006785788', 'IND', '6',  '2',  'Ajanta Pharma (Paithan)'],
            ['9 more FEIs', '', '1–4', '', ''],
        ],
        col_widths=[1.0, 0.75, 0.55, 0.55, 4.25]
    )

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 1 — Market Outcomes by Prior FDA Inspection Outcome', 1)
    add_note(doc, 'Left panel: NADAC price per unit (blank in July 2026 — not yet in pipeline). Right panel: Annual IQVIA extended units by inspection outcome (IND/CHN/USA, log scale).')
    add_fig_pair(doc, 'Pre-revision (Health Affairs Scholars)', old[1],
                      'July 2026 (strict rule: EventYear < TestYear)', new[1])

    doc.add_heading('Statistics Comparison — Model B (MixedLM + CGM two-way clustered SE)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Coefficient', 'Pre-revision', 'July 2026'],
        rows=[
            ['VAI vs NAI',
             [
                 {'text': 'β = −1.820, SE = 0.801\n95% CI [−3.389, −0.250],  '},
                 {'text': 'p = 0.025 (*)', 'bold': True, 'red': True},
             ],
             'β = +2.311, SE = 1.421\n95% CI [−0.474, +5.096],  p = 0.105'],
            ['OAI vs NAI',
             'β = +1.747, SE = 0.952\n95% CI [−0.120, +3.613],  p = 0.069',
             'β = +2.182, SE = 1.895\n95% CI [−1.533, +5.897],  p = 0.251'],
            ['OAI vs VAI',
             [
                 {'text': 'β = +3.566, SE = 0.782\n95% CI [+2.033, +5.100],  '},
                 {'text': 'p < 0.001 (***)', 'bold': True, 'red': True},
             ],
             'implied ≈ −0.129,  ns'],
        ],
        col_widths=[1.5, 3.0, 2.6]
    )
    add_note(doc, 'Reference = NAI. n_obs = 110 (pre-revision) / 221 (July 2026), n_NDC = 81, n_FEI = 23.')
    doc.add_paragraph()

    doc.add_heading('Descriptive Volume (July 2026 panel)', 2)
    simple_table(doc,
        headers=['Outcome', 'n', 'Mean', 'Median', 'P25', 'P75'],
        rows=[
            ['NAI', '26', '24,745,069', '862,990', '47,187', '6,535,536'],
            ['VAI', '155', '38,744,119', '2,483,318', '363,740', '16,434,334'],
            ['OAI', '40', '46,534,770', '1,453,286', '212,008', '8,824,912'],
        ],
        col_widths=[0.85, 0.5, 1.35, 1.25, 1.2, 1.45]
    )
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('Conclusion: '); r.bold = True; r.font.size = Pt(10)
    p.add_run(
        'Old Observation 1 (volume part) is not supported in the updated data. '
        'The primary model shows no significant relationship between inspection outcome and market volume '
        '(VAI p = 0.105, OAI p = 0.251). The direction is reversed from the old paper: the new data show '
        'NAI < VAI < OAI, whereas the old paper claimed NAI > VAI (β = −1.820, p = 0.025). '
        'The price part cannot be evaluated until NADAC is added to the pipeline.'
    ).font.size = Pt(10)

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 2 — Market Volume vs Tested Drug Quality', 1)
    add_note(doc, 'Each panel pools all available years for that metric. Spearman ρ with NDC-cluster block bootstrap (2,000 resamples).')
    add_fig_pair(doc, 'Pre-revision (Health Affairs Scholars)', old[2],
                      'July 2026 (new pipeline)', new[2])

    doc.add_heading('Statistics Comparison — Spearman ρ (NDC-cluster block bootstrap)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Association', 'Pre-revision', 'July 2026'],
        rows=[
            ['DMF vs Volume',
             [
                 {'text': 'ρ = +0.279,  '},
                 {'text': 'p = 0.004 (**)', 'bold': True, 'red': True},
                 {'text': '\n95% CI [+0.064, +0.454]'},
             ],
             [
                 {'text': 'ρ = +0.302,  '},
                 {'text': 'p_boot = 0.002 (**)', 'bold': True, 'red': True},
                 {'text': '\n95% CI [+0.112, +0.467],  n = 126'},
             ]],
            ['NDMA vs Volume',
             'not significant  (p > 0.10)',
             'ρ = −0.064,  p_boot = 0.635\nn = 71'],
            ['Difference Factor vs Volume',
             'not significant  (p > 0.10)',
             'ρ = −0.162,  p_boot = 0.454\nn = 25'],
        ],
        col_widths=[2.0, 2.5, 2.6]
    )
    add_note(doc, 'NDC-cluster block bootstrap resamples whole NDC clusters to obtain cluster-robust p-values and 95% CI.')
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('Conclusion: '); r.bold = True; r.font.size = Pt(10)
    p.add_run(
        'Old Observation 2 (DMF vs market volume) is still supported and strengthened '
        '(ρ = +0.30, p = 0.002, n = 126 vs n ≈ 86 previously). '
        'The positive association between higher DMF contamination and higher market volume is robust. '
        'NDMA and Difference Factor versus volume remain non-significant in both datasets.'
    ).font.size = Pt(10)

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 3 — Price vs Tested Drug Quality', 1)
    add_note(doc, 'July 2026 figure is blank — NADAC price data not yet integrated into the pipeline. NADAC is available in the pre-revision Q&A file (107 of 111 NDC-years have NADAC).')
    add_fig_pair(doc, 'Pre-revision (Health Affairs Scholars)', old[3],
                      'July 2026 (NADAC pending)', new[3])

    doc.add_heading('Statistics Comparison — Spearman ρ (NDC-cluster block bootstrap)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Association', 'Pre-revision', 'July 2026'],
        rows=[
            ['DMF vs Price',
             'not significant  (p > 0.10)',
             'pending NADAC integration'],
            ['NDMA vs Price',
             [
                 {'text': 'ρ = +0.282,  '},
                 {'text': 'p = 0.013 (*)', 'bold': True, 'red': True},
                 {'text': '\n95% CI [+0.056, +0.490]'},
             ],
             'pending NADAC integration'],
            ['Difference Factor vs Price',
             'not significant  (p > 0.10)',
             'pending NADAC integration'],
        ],
        col_widths=[2.0, 2.5, 2.6]
    )
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('Conclusion: '); r.bold = True; r.font.size = Pt(10)
    p.add_run(
        'Cannot evaluate. Old Observation 2 (NDMA vs price, ρ = +0.282, p = 0.013) '
        'cannot be reproduced until NADAC pricing is added to Step 5 of the pipeline. '
        'This remains an open item.'
    ).font.size = Pt(10)

    # ── Figure 4 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 4 — Drug Quality by Country of Manufacture', 1)
    add_note(doc, 'Primary model: MixedLM random NDC intercept + CGM two-way clustered SE (NDC × FEI), reference = USA.')
    add_fig_pair(doc, 'Pre-revision (Health Affairs Scholars)', old[4],
                      'July 2026 (new pipeline)', new[4])

    doc.add_heading('Statistics Comparison — Model B (MixedLM + CGM two-way clustered SE)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Metric / Comparison', 'Pre-revision', 'July 2026'],
        rows=[
            ['DMF:  IND vs USA',
             'not significant',
             'β = +2.650,  p = 0.136'],
            ['DMF:  CHN vs USA',
             '—',
             'β = +0.277,  p = 0.867'],
            ['DMF:  CHN vs IND',
             '—',
             'β = −2.374,  p = 0.092  (marginal)'],
            ['NDMA:  IND vs USA',
             [
                 {'text': 'β = +1.345,  '},
                 {'text': 'p < 0.001 (***)', 'bold': True, 'red': True},
             ],
             [
                 {'text': 'β = +1.603,  '},
                 {'text': 'p = 0.014 (*)', 'bold': True, 'red': True},
             ]],
            ['NDMA:  CHN vs USA',
             'not significant',
             'β = +0.310,  p = 0.284'],
            ['NDMA:  CHN vs IND',
             [
                 {'text': 'β = −1.090,  '},
                 {'text': 'p = 0.022 (*)', 'bold': True, 'red': True},
             ],
             'β = −1.293,  p = 0.067  (marginal)'],
            ['Diff Factor:  IND vs USA',
             [
                 {'text': 'β = +0.117,  '},
                 {'text': 'p = 0.011 (*)', 'bold': True, 'red': True},
             ],
             'β = +0.074,  p = 0.061  (marginal)'],
            ['Diff Factor:  CHN vs USA',
             'not significant',
             'β = +0.060,  p = 0.185'],
            ['Diff Factor:  CHN vs IND',
             'not significant',
             'β = −0.015,  p = 0.802'],
        ],
        col_widths=[2.0, 2.5, 2.6]
    )
    add_note(doc, (
        'Descriptive means (July 2026): '
        'DMF — IND 28,607 ng/day, CHN 3,355, USA 4,696.  '
        'NDMA — IND 65.2 ng/day, CHN 2.0, USA 0.0.  '
        'Difference Factor — IND 0.261, CHN 0.226, USA 0.153.'
    ))
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('Conclusion: '); r.bold = True; r.font.size = Pt(10)
    p.add_run(
        'Old Observation 3 (NDMA, India vs USA) remains statistically supported (p = 0.014, vs p < 0.001 previously). '
        'The claims that NDMA is lower in China than India (old p = 0.022, new p = 0.067) and that '
        'dissolution failure is more common in India than the US (old p = 0.011, new p = 0.061) '
        'are weakened to marginal significance and should be softened in the paper. '
        'DMF country differences remain non-significant in both datasets.'
    ).font.size = Pt(10)

    doc.save(str(OUT_DOC))
    print(f'Saved: {OUT_DOC}')

print('Done.')
# %%
