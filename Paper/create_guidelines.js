const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        HeadingLevel, AlignmentType, WidthType, ShadingType, BorderStyle, PageBreak } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22 }  // 11pt default
      }
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // Title
      new Paragraph({
        children: [
          new TextRun({
            text: "Health Affairs Scholar Submission Guidelines",
            bold: true,
            size: 32,
            font: "Arial"
          })
        ],
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Complete Reference for Authors",
            italic: true,
            size: 24,
            font: "Arial"
          })
        ],
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 }
      }),

      // Intro
      new Paragraph({
        children: [
          new TextRun("Health Affairs Scholar is a peer-reviewed, fully open access journal publishing 12 issues per year. It is the international open access journal of health policy and emerging health services research, published by Oxford University Press. This document consolidates submission requirements and formatting standards.")
        ],
        spacing: { after: 240 }
      }),

      // Section 1
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("1. Article Types and Word Limits")]
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Research Articles",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Primary original research reporting novel findings")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Required structure: Introduction, Methods, Results, Discussion (with focus on policy implications)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Abstract required: 150\u2013200 words (shorter is acceptable if all requirements met)")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Commentaries and Perspectives",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Optional abstract (varies by article type)")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "General Content Scope",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Core topics: health care costs, access, quality, equity")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Emerging areas: health care technology, population health, global health, health/social services intersections")
        ],
        spacing: { after: 200 }
      }),

      // Section 2
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("2. Abstract Requirements")]
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Length",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("150\u2013200 words (shorter abstracts acceptable if all requirements are met)")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Structure and Content",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Present the paper in miniature")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include: topic, context, what the paper contributes, specific findings/main point, implications/conclusions")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("NO jargon or high-level phrases known only to specialists")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("NO acronyms (spell out all abbreviations)")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Language to Avoid",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Avoid first-person phrasing such as &#x201C;we found&#x201D;")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Use active voice focused on the findings, not the researchers")
        ],
        spacing: { after: 200 }
      }),

      // Section 3
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("3. Formatting Requirements")]
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Font and Spacing",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Font: 12-point Courier New or equivalent sans-serif")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Line spacing: Double-spaced throughout (including abstract, methods, references)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Margins: 1 inch on all sides (top, bottom, left, right)")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Page Numbers and Headers/Footers",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include page numbers (typically in footer, right-aligned or centered)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Optional: running header with article title or shortened title")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Text Justification",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Left-aligned (not justified) preferred for readability")
        ],
        spacing: { after: 200 }
      }),

      // Section 4
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("4. Citation and Reference Style")]
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Citation Format",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Use consecutively numbered endnotes (NOT footnotes)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Number citations sequentially in the order they appear in text using superscript Arabic numerals")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Repeat reference numbers if the same source is cited multiple times")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Endnote Content",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("All notations and reference information go in endnotes (no footnotes)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include full bibliographic details: author(s), title, publication, year, and where applicable, URL or DOI")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Reference Limits",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("No specific maximum number mandated; however, keep references focused and relevant")
        ],
        spacing: { after: 200 }
      }),

      // Section 5
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("5. Figures and Tables")]
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "General Requirements",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include captions for all figures and tables")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Number sequentially: Figure 1, Figure 2, etc. and Table 1, Table 2, etc.")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Place figure/table label above the visual element")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Figure Specifications",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Resolution: minimum 300 dpi for publication quality")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Format: PNG, JPG, TIFF, or PDF (vector graphics preferred for charts/graphs)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Captions should be descriptive and concise (legend included as part of caption)")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Table Specifications",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include table title/caption above the table")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include table notes below the table (e.g., abbreviations, data sources, statistical significance indicators)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Use simple formatting: borders/gridlines for clarity, no shading unless necessary")
        ],
        spacing: { after: 160 }
      }),

      new Paragraph({
        children: [
          new TextRun({
            text: "Limits",
            bold: true
          })
        ],
        spacing: { after: 80 }
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("No specific maximum stated; however, authors should limit to those that directly support the narrative")
        ],
        spacing: { after: 200 }
      }),

      // Section 6
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("6. Keywords")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Provide 4\u20136 keywords (check OUP submission site for exact count)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Use MeSH (Medical Subject Headings) terms when applicable")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("List on a separate line after the abstract")
        ],
        spacing: { after: 200 }
      }),

      // Section 7
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("7. Author Information")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Author names: First name, middle initial(s), last name")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Affiliations: List all authors&#x2019; institutional affiliations in order")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Corresponding author: Designate one; include email and phone")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("ORCIDs: Include if available (optional but encouraged)")
        ],
        spacing: { after: 200 }
      }),

      // Section 8
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("8. File Submission Format")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Main manuscript: Word document (.docx or .doc)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Figures: Separate high-resolution image files")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Tables: Can be embedded in the main document or submitted separately")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Supplementary materials: Separate files if applicable")
        ],
        spacing: { after: 200 }
      }),

      // Section 9
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("9. Conflict of Interest and Ethics")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Declare any financial, professional, or personal relationships that could bias the work")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("For research involving human subjects: include IRB approval statement")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Include informed consent statement if applicable")
        ],
        spacing: { after: 200 }
      }),

      // Section 10
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("10. Open Access and Article Processing Charges")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Health Affairs Scholar is fully open access")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Article Processing Charges (APC) apply (check OUP or journal website for current fee structure)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Waivers may be available for authors from low-income countries or upon request")
        ],
        spacing: { after: 200 }
      }),

      // Section 11
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("11. Peer Review and Publication Timeline")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Single-anonymized peer review: authors&#x2019; identities known to editors/reviewers; reviewers&#x2019; identities hidden from authors")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Median time to first decision (without revision): 6 days")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Typical peer review + revision timeframe: 2 months")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Post-acceptance publication (Advance Access): within 1 week of signed license agreement")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Fast-track option available: target 4\u20136 weeks from submission with pre-submission inquiry (expedited consideration must be justified)")
        ],
        spacing: { after: 200 }
      }),

      // Section 12
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("12. Pre-Submission Checklist")]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Abstract: 150\u2013200 words, no jargon, no acronyms, includes context/findings/implications")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Formatting: 12pt font, double-spaced, 1-inch margins on all sides")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Citations: Consecutively numbered endnotes (not footnotes)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Figures/Tables: High-resolution images with captions and notes")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Keywords: 4\u20136 terms provided")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Author info: Names, affiliations, corresponding author contact details")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Conflict of Interest: Statement provided (or declaration of no conflicts)")
        ]
      }),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [
          new TextRun("Ethics: IRB approval statement (if applicable)")
        ]
      }),
    ]
  }],
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: "bullet",
            text: "●",
            alignment: AlignmentType.LEFT,
            style: {
              paragraph: {
                indent: { left: 720, hanging: 360 }
              }
            }
          }
        ]
      }
    ]
  }
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Health Affairs Scholar Submission Guidelines.docx", buffer);
  console.log("Guidelines document created successfully!");
});
