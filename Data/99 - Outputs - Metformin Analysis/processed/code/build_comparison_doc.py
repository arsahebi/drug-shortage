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

def _add_labeled_figure(doc, label, img, width=6.2):
    p_lbl = doc.add_paragraph()
    r = p_lbl.add_run(label); r.bold = True; r.font.size = Pt(9)
    p_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if img and Path(img).exists():
        p_img.add_run().add_picture(str(img), width=Inches(width))
    else:
        p_img.add_run('(not available)')
    doc.add_paragraph()

def add_fig_pair(doc, label_old, old_img, label_new, new_img, width=6.2):
    for label, img in [(label_old, old_img), (label_new, new_img)]:
        _add_labeled_figure(doc, label, img, width)

def add_fig_triple(doc, label_old, old_img, label_new, new_img, label_sfei, sfei_img, width=6.2):
    for label, img in [(label_old, old_img), (label_new, new_img), (label_sfei, sfei_img)]:
        _add_labeled_figure(doc, label, img, width)

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
    new_singlefei = {
        1: NEW_FIG / "Figure1_Market_by_Outcome_SingleFEI.png",
        2: NEW_FIG / "Figure2_Volume_vs_Quality_SingleFEI.png",
        3: NEW_FIG / "Figure3_Price_vs_Quality_SingleFEI.png",
    }
    figS1 = NEW_FIG / "FigureS1_Months_Since_Inspection.png"

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

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 1 — Market Outcomes by Prior FDA Inspection Outcome', 1)
    add_note(doc, 'Left panel: NADAC price per unit (blank in July 2026 — not yet in pipeline). Right panel: Annual IQVIA extended units by inspection outcome (IND/CHN/USA, log scale).')
    add_note(doc, 'All NDCs = full July 2026 panel including 25 multi-FEI NDC11s.  Single-FEI = those 25 excluded (87 NDC11s, 261 rows remain).')
    add_fig_triple(doc,
        'Pre-revision (Health Affairs Scholars)', old[1],
        'July 2026 — All NDCs (strict rule: EventYear < TestYear)', new[1],
        'July 2026 — Single-FEI NDCs only', new_singlefei[1])

    doc.add_heading('Statistics — Model B (MixedLM + CGM two-way clustered SE)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Coefficient', 'Pre-revision', 'July 2026 — All NDCs', 'July 2026 — Single-FEI'],
        rows=[
            ['VAI vs NAI',
             [{'text': 'β = −1.820, SE = 0.801\n95% CI [−3.389, −0.250],  '},
              {'text': 'p = 0.025 (*)', 'bold': True, 'red': True}],
             'β = +2.311, SE = 1.421\n95% CI [−0.474, +5.096],  p = 0.105',
             'β = +1.054, SE = 1.681\n95% CI [−2.241, +4.350],  p = 0.532'],
            ['OAI vs NAI',
             'β = +1.747, SE = 0.952\n95% CI [−0.120, +3.613],  p = 0.069',
             'β = +2.182, SE = 1.895\n95% CI [−1.533, +5.897],  p = 0.251',
             'β = −0.100, SE = 1.899\n95% CI [−3.821, +3.621],  p = 0.958'],
            ['OAI vs VAI',
             [{'text': 'β = +3.566, SE = 0.782\n95% CI [+2.033, +5.100],  '},
              {'text': 'p < 0.001 (***)', 'bold': True, 'red': True}],
             'implied ≈ −0.129,  ns',
             'implied ≈ −1.154,  ns'],
        ],
        col_widths=[1.3, 2.3, 2.2, 2.2]
    )
    add_note(doc, (
        'Reference = NAI.  '
        'All NDCs: n_obs=221, n_NDC=80, n_FEI=23.  '
        'Single-FEI: n_obs=149, n_NDC=55, n_FEI=18.'
    ))
    doc.add_paragraph()

    doc.add_heading('Descriptive Volume (IQVIA extended units)', 2)
    simple_table(doc,
        headers=['Outcome', 'Pre-rev n', 'Pre-rev Median',
                 'All-NDC n', 'All-NDC Median',
                 'Single-FEI n', 'Single-FEI Median'],
        rows=[
            ['NAI', '39', '15,944,388', '26', '862,990',   '20', '1,364,730'],
            ['VAI', '56', '2,392,105',  '155','2,483,318',  '113','2,168,090'],
            ['OAI', '16', '18,425,376', '40', '1,453,286',  '16', '515,978'],
        ],
        col_widths=[0.75, 0.65, 1.25, 0.65, 1.25, 0.75, 1.25]
    )
    add_note(doc, 'Pre-revision: Granules India (FEI 3004097901) dominates NAI with 30 NDC-year observations, driving the high NAI mean/median.')
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('Conclusion: '); r.bold = True; r.font.size = Pt(10)
    p.add_run(
        'Neither version supports the original observation that NAI facilities have higher volume. '
        'The primary model shows no significant relationship between inspection outcome and volume in either '
        'the all-NDC (VAI p = 0.105, OAI p = 0.251) or single-FEI (VAI p = 0.532, OAI p = 0.958) panel. '
        'In the single-FEI panel OAI median volume drops sharply (516K vs 1.5M in all-NDC), suggesting '
        'that multi-FEI NDCs inflate OAI volume in the full panel. '
        'The price side cannot be evaluated until NADAC is added to the pipeline.'
    ).font.size = Pt(10)

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 2 — Market Volume vs Tested Drug Quality', 1)
    add_note(doc, 'Each panel pools all available years for that metric. Spearman ρ with NDC-cluster block bootstrap (2,000 resamples).')
    add_note(doc, 'All NDCs = full July 2026 panel.  Single-FEI = 25 multi-FEI NDC11s excluded.')
    add_fig_triple(doc,
        'Pre-revision (Health Affairs Scholars)', old[2],
        'July 2026 — All NDCs', new[2],
        'July 2026 — Single-FEI NDCs only', new_singlefei[2])

    doc.add_heading('Statistics — Spearman ρ (NDC-cluster block bootstrap, 2,000 resamples)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Association', 'Pre-revision', 'July 2026 — All NDCs', 'July 2026 — Single-FEI'],
        rows=[
            ['DMF vs Volume',
             [{'text': 'ρ = +0.279,  '},
              {'text': 'p = 0.004 (**)', 'bold': True, 'red': True},
              {'text': '\n95% CI [+0.064, +0.454]'}],
             [{'text': 'ρ = +0.302,  '},
              {'text': 'p_boot = 0.002 (**)', 'bold': True, 'red': True},
              {'text': '\n95% CI [+0.112, +0.467],  n = 126'}],
             [{'text': 'ρ = +0.307,  '},
              {'text': 'p_boot = 0.003 (**)', 'bold': True, 'red': True},
              {'text': '\n95% CI [+0.090, +0.500],  n = 91'}]],
            ['NDMA vs Volume',
             'not significant  (p > 0.10)',
             'ρ = −0.064,  p_boot = 0.635\nn = 71',
             'ρ = +0.018,  p_boot = 0.898\nn = 54'],
            ['Diff Factor vs Volume',
             'not significant  (p > 0.10)',
             'ρ = −0.162,  p_boot = 0.454\nn = 25',
             'ρ = −0.118,  p_boot = 0.613\nn = 21'],
        ],
        col_widths=[1.5, 1.9, 2.1, 2.1]
    )
    add_note(doc, 'NDC-cluster block bootstrap resamples whole NDC clusters to obtain cluster-robust p-values and 95% CI.')
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('Conclusion: '); r.bold = True; r.font.size = Pt(10)
    p.add_run(
        'Old Observation 2 (DMF vs market volume) is supported and robust in both the all-NDC '
        '(ρ = +0.30, p = 0.002, n = 126) and single-FEI (ρ = +0.31, p = 0.003, n = 91) panels. '
        'Excluding multi-FEI NDCs slightly increases the effect size. '
        'NDMA and Difference Factor versus volume remain non-significant in both versions.'
    ).font.size = Pt(10)

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure 3 — Price vs Tested Drug Quality', 1)
    add_note(doc, 'July 2026 figures are blank — NADAC price data not yet integrated into the pipeline. NADAC is available in the pre-revision Q&A file (107 of 111 NDC-years have NADAC).')
    add_note(doc, 'All NDCs = full July 2026 panel.  Single-FEI = 25 multi-FEI NDC11s excluded.')
    add_fig_triple(doc,
        'Pre-revision (Health Affairs Scholars)', old[3],
        'July 2026 — All NDCs (NADAC pending)', new[3],
        'July 2026 — Single-FEI NDCs only (NADAC pending)', new_singlefei[3])

    doc.add_heading('Statistics Comparison — Spearman ρ (NDC-cluster block bootstrap)', 2)
    add_note(doc, SIG_NOTE)
    simple_table(doc,
        headers=['Association', 'Pre-revision', 'July 2026 — All NDCs', 'July 2026 — Single-FEI'],
        rows=[
            ['DMF vs Price',
             'not significant  (p > 0.10)',
             'pending NADAC integration',
             'pending NADAC integration'],
            ['NDMA vs Price',
             [
                 {'text': 'ρ = +0.282,  '},
                 {'text': 'p = 0.013 (*)', 'bold': True, 'red': True},
                 {'text': '\n95% CI [+0.056, +0.490]'},
             ],
             'pending NADAC integration',
             'pending NADAC integration'],
            ['Difference Factor vs Price',
             'not significant  (p > 0.10)',
             'pending NADAC integration',
             'pending NADAC integration'],
        ],
        col_widths=[1.5, 1.9, 1.7, 2.0]
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

    # ── Figure S1 — Months Since Last Inspection ──────────────────────────────
    doc.add_page_break()
    doc.add_heading('Figure S1 — Distribution of Months Since Last Inspection', 1)
    add_note(doc, (
        'For each (NDC11 × TestYear) row with a prior classified inspection, '
        '"months since inspection" = (TestYear − prior_event_year) × 12. '
        'Year-level resolution only (Redica does not record exact inspection dates consistently). '
        'This shows how stale the prior inspection signal is relative to the Valisure test year.'
    ))
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if figS1.exists():
        p_img.add_run().add_picture(str(figS1), width=Inches(6.2))
    else:
        p_img.add_run('(FigureS1_Months_Since_Inspection.png not found)')
    doc.add_paragraph()

    doc.save(str(OUT_DOC))
    print(f'Saved: {OUT_DOC}')

print('Done.')
# %%
