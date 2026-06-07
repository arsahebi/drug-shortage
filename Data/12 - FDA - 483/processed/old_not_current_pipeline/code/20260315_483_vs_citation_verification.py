# %%
"""
20260315_483_vs_citation_verification.py

Verifies the relationship between FDA 483 PDF observations and
the structured Citation Details from the FDA Inspection database.

Key question: Does the 'Long Description' in citation data match
the first sentence of each numbered 483 observation?

Uses FEI 3002809586 (Sun Pharmaceutical) as the test case.
"""

import pandas as pd
import pdfplumber
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]  # Data/
RAW_483 = BASE / "12 - FDA - 483" / "raw"

FEI = "3002809586"

# ── Load data ──────────────────────────────────────────────────────────────
cit = pd.read_excel(RAW_483 / f"{FEI} - Citation Data.xlsx")
insp = pd.read_excel(RAW_483 / f"{FEI} - Inspection Data.xlsx")
f483 = pd.read_excel(RAW_483 / f"{FEI} - 483 Data.xlsx")

# ── Match 483 PDFs to citations by inspection date ─────────────────────────
pdf_files = sorted(RAW_483.glob(f"{FEI} - *.pdf*"))
print(f"Found {len(pdf_files)} 483 PDFs for FEI {FEI}")
print(f"Citation records: {len(cit)}, spanning {cit['Inspection End Date'].nunique()} inspections")
print()

# ── For each inspection with both a PDF and citation data, compare ─────────
cit_dates = cit["Inspection End Date"].dt.strftime("%Y-%m-%d").unique()
f483_dates = f483["Record Date"].dt.strftime("%Y-%m-%d").tolist()

print("483 PDF dates:", sorted(f483_dates))
print("Citation dates:", sorted(cit_dates))
print()

# Focus on inspections that have both a PDF and citation data
common_dates = set(f483_dates) & set(cit_dates)
print(f"Dates with both PDF and citations: {sorted(common_dates)}")
print()


def extract_obs_headers(pdf_path: Path) -> list[str]:
    """Extract the first-sentence header of each numbered observation."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(
            p.extract_text() for p in pdf.pages if p.extract_text()
        )
    blocks = re.split(r"OBSERVATION \d+", full_text)[1:]
    headers = []
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        header_lines = []
        for line in lines:
            # Stop at numbered examples, employee sig lines, or 'Examples'
            if re.match(r"^\d+\.", line) or line.startswith("EMPLOYEE") or line.startswith("Examples"):
                break
            header_lines.append(line)
        if header_lines:
            # Clean OCR artifacts (common: 'diug' → 'drug', 'mu g' → 'drug')
            raw = " ".join(header_lines)
            raw = re.sub(r"\s+", " ", raw).strip()
            headers.append(raw)
    return headers


# ── Compare for one well-documented inspection: 2025-06-13 ────────────────
target_date = "2025-06-13"
pdf_match = [f for f in pdf_files if "2025" in f.name]

if pdf_match:
    pdf_path = pdf_match[0]
    obs_headers = extract_obs_headers(pdf_path)
    cit_subset = cit[cit["Inspection End Date"].dt.strftime("%Y-%m-%d") == target_date]

    print(f"=== Inspection {target_date} ===")
    print(f"  483 observations in PDF : {len(obs_headers)}")
    print(f"  Citation records in DB  : {len(cit_subset)}")
    print()

    # Build comparison table
    rows = []
    for i, h in enumerate(obs_headers):
        # Normalize for fuzzy match (lower, collapse spaces)
        h_norm = re.sub(r"\s+", " ", h).lower().strip()
        best_match = None
        best_score = 0
        for _, row in cit_subset.iterrows():
            ld_norm = re.sub(r"\s+", " ", str(row["Long Description"])).lower().strip()
            # Simple overlap: how many words match at start
            words_h = h_norm.split()
            words_ld = ld_norm.split()
            overlap = sum(1 for w1, w2 in zip(words_h, words_ld) if w1 == w2)
            score = overlap / max(len(words_ld), 1)
            if score > best_score:
                best_score = score
                best_match = row
        rows.append({
            "Obs #": i + 1,
            "483 Header (first 80 chars)": h[:80],
            "Best Match CFR": best_match["Act/CFR Number"] if best_match is not None else "—",
            "Best Match Short": best_match["Short Description"] if best_match is not None else "—",
            "Match Score": round(best_score, 2),
        })

    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))
    print()

# ── Summary: what the data layers mean ────────────────────────────────────
print("=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("""
1. Long Description  = standardized first-sentence TEMPLATE of each 483 observation.
   It matches the regulatory language tied to the CFR code.
   → Useful for categorization; NOT unique/specific to a particular facility.

2. Short Description = brief label for the CFR category (e.g. 'Data Integrity').
   → Best for aggregation, frequency counts, co-occurrence analysis.

3. Act/CFR Number    = most abstract regulatory code (e.g. 21 CFR 211.192).
   → Use for taxonomy building and cross-facility comparison.

4. Actual 483 body text (after the header sentence) contains specific details,
   facility-specific examples, measurements, dates, and product names.
   → This is the HIGH-VALUE text NOT captured in the citation database.
   → For richer NLP/quality prediction, we NEED the 483 PDFs.

5. Implication for modeling:
   - Citation DB (CFR + Short Description) → structured features, co-occurrence
   - 483 PDFs                              → semantic features, LLM embeddings
   - Both are complementary, not redundant
""")
