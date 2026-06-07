# %%
"""
20260316_warning_letter_features.py  (v2 — fixed)

Fixes applied vs. v1:
  1. Primary FEI extracted from WL text (first-paragraph inspection sentence),
     NOT from filename. Filename FEI stored as `search_fei` (the FEI the user
     searched that surfaced this WL — maps back to our 129-FEI universe).
  2. CFR codes counted only within numbered violation blocks, not boilerplate
     (e.g. "consultant qualified as set forth in 21 CFR 211.34" excluded).
  3. has_211_192 removed — covered by domain-level counts.
  4. CFR domain classifier rebuilt on official FDA 21 CFR Part 211 subpart
     numeric ranges instead of fragile string prefix matching.
  5. Repeat-section collector stops when it hits the NEXT repeat header,
     so both "Repeat Observation at Facility" AND "Repeat Violations at
     Multiple Sites" are captured as separate sections.
  6. Prior WL reference count requires the phrase "warning letter" to appear
     together with a specific date ("warning letter dated Month DD, YYYY"),
     eliminating generic/boilerplate mentions.
  7. has_drug_shortage_mention removed — it appears as FDA boilerplate in
     virtually every WL and carries no facility-specific signal.

Key columns:
  search_fei   — FEI from filename (user's search FEI; maps to our 129 FEIs)
  primary_fei  — FEI cited in the WL opening sentence (actual inspected facility)
  fei_mismatch — True when search_fei ≠ primary_fei (cross-site discovery)

Outputs (in processed/ folder):
  warning_letter_records.csv      — one row per PDF
  warning_letter_fei_features.csv — aggregated per search_fei
  wl_richness_comparison.csv      — text richness vs 483 vs citation DB
  wl_fei_network.csv              — cross-site FEI link network (all edge types)
"""

import pdfplumber
import re
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parents[3]
WL_RAW       = BASE / "Data/21 - FDA - Warning Letter/raw"
WL_OUT       = Path(__file__).parent
CITATION_CSV = BASE / "Data/14 - FDA - Inspection/processed/citations_with_classification.csv"

# ── Regex patterns ─────────────────────────────────────────────────────────
CFR_PATTERN = re.compile(
    r"21\s+CFR\s+(\d{3}[\.\(][\w\.\(\)]+)",
    re.IGNORECASE
)
FEI_IN_TEXT_PATTERN = re.compile(r"\bFEI[:\s#]*(\d{7,10})\b", re.IGNORECASE)
DATE_IN_TEXT_PATTERN = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}",
    re.IGNORECASE
)

# Fix #6 — genuine prior WL reference: "warning letter dated Month DD, YYYY"
# Optionally preceded by "issued a", "received a", "sent a", etc.
PRIOR_WL_DATE_PATTERN = re.compile(
    r"warning\s+letter\s+(?:[^.]{0,30}?\s+)?dated\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}",
    re.IGNORECASE
)

# ── Repeat-section header patterns ─────────────────────────────────────────
REPEAT_FACILITY_PATS = [
    re.compile(r"repeat\s+(?:observation|violation)s?\s+at\s+(?:this\s+)?(?:facility|site|firm)", re.IGNORECASE),
    re.compile(r"repeat\s+(?:observation|violation)s?\s+at\s+(?:your|the)\s+(?:facility|site|firm)", re.IGNORECASE),
]
REPEAT_MULTI_PATS = [
    re.compile(r"repeat\s+(?:observation|violation)s?\s+at\s+multiple\s+sites?", re.IGNORECASE),
    re.compile(r"repeat\s+(?:observation|violation)s?\s+(?:across|at)\s+(?:multiple|other)\s+(?:sites?|facilit)", re.IGNORECASE),
    re.compile(r"violations?\s+at\s+multiple\s+sites?", re.IGNORECASE),
]
ALL_REPEAT_PATS = REPEAT_FACILITY_PATS + REPEAT_MULTI_PATS

# ── Management / corporate language phrases ────────────────────────────────
# (has_drug_shortage_mention removed — boilerplate in every WL)
MGMT_OVERSIGHT_PHRASES = [
    "management oversight",
    "corporate oversight",
    "inadequate oversight",
    "quality oversight",
    "senior management",
    "executive management",
]
CORP_FAILURE_PHRASES = [
    "repeated failures",
    "repeated violations",
    "demonstrate that",
    "corporate commitment",
    "inadequate quality system",
    "inadequate quality management",
    "ineffective quality system",
    "systemic",
]


# ══════════════════════════════════════════════════════════════════════════
# FIX #4 — CFR domain classifier (official FDA 21 CFR Part 211 subparts)
# ══════════════════════════════════════════════════════════════════════════
def cfr_domain(cfr_str):
    """
    Maps a CFR string (e.g. '211.67(a)') to its FDA Part 211 subpart domain
    using the official numeric ranges.

    Official 21 CFR Part 211 subparts:
      Subpart B (22–34):  Organization & Personnel
      Subpart C+D (42–72): Buildings, Facilities & Equipment
      Subpart E+F (80–115): Components, Production & Process Controls
      Subpart G (122–137): Packaging & Labeling
      Subpart H (142–150): Holding & Distribution
      Subpart I (160–176): Laboratory Controls
      Subpart J (180–198): Records & Reports
      Subpart K (204–208): Returned & Salvaged Drug Products
    """
    s = str(cfr_str)

    # Non-211 regulatory areas
    if re.search(r"\b312\b|\b314\b|\b320\b", s):
        return "bioresearch"
    if re.search(r"\b210\b", s):
        return "general_cgmp"

    # Extract the numeric portion after "211."
    m = re.search(r"211\.(\d+)", s)
    if not m:
        return "other"

    num = int(m.group(1))

    if 22 <= num <= 34:    return "org_personnel"         # Subpart B
    if 42 <= num <= 72:    return "buildings_equipment"   # Subpart C + D
    if 80 <= num <= 115:   return "production"            # Subpart E + F
    if 122 <= num <= 137:  return "packaging_labeling"    # Subpart G
    if 142 <= num <= 150:  return "holding_distribution"  # Subpart H
    if 160 <= num <= 176:  return "lab_controls"          # Subpart I
    if 180 <= num <= 198:  return "records_reports"       # Subpart J
    if 204 <= num <= 208:  return "returned_salvaged"     # Subpart K

    return "other_211"


# ══════════════════════════════════════════════════════════════════════════
# FIX #1 — Extract primary FEI from WL text (inspection sentence)
# ══════════════════════════════════════════════════════════════════════════
def extract_primary_fei_from_text(text, search_fei):
    """
    Find the FEI of the actually-inspected facility from the WL opening paragraph.
    Typical pattern:
      'FDA inspected your drug manufacturing facility, [Firm], FEI XXXXXXX, at [address]'

    Returns (primary_fei: int, source: str).
    Falls back to search_fei with source='filename_fallback' if not found.
    """
    opening = text[:4000]  # first ~4000 chars covers the opening paragraph

    # Pattern 1: "inspected your ... facility ... FEI XXXXXXX"
    m = re.search(
        r"inspected\s+your[^.]{0,250}?FEI[:\s#]*(\d{7,10})",
        opening, re.IGNORECASE | re.DOTALL
    )
    if m:
        return int(m.group(1)), "text_inspection_sentence"

    # Pattern 2: "facility, [Firm Name], FEI XXXXXXX"
    m = re.search(
        r"facility[^.]{0,150},\s*FEI[:\s#]*(\d{7,10})",
        opening, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), "text_facility_comma_pattern"

    # Pattern 3: first FEI mention anywhere in the opening
    m = re.search(r"FEI[:\s#]*(\d{7,10})", opening, re.IGNORECASE)
    if m:
        return int(m.group(1)), "text_first_fei"

    # Fallback
    return search_fei, "filename_fallback"


# ══════════════════════════════════════════════════════════════════════════
# FIX #2 — Extract CFRs only from numbered violation blocks
# ══════════════════════════════════════════════════════════════════════════
VIOLATION_START = re.compile(
    r"^\s*[1-9]\d?\.\s+(?:21\s+CFR|Failure|During|Your\s+firm|There\s+is|The\s+firm)",
    re.IGNORECASE
)
SECTION_BREAK = re.compile(
    r"^(?:Conclusion|Recommendation|Data Integrity|CORRECTIVE|The violations?|"
    r"If you|We note|Please respond|Sincerely|Repeat\s+(?:Observation|Violation)|"
    r"FDA|U\.S\.\s+Food|Additional|Background)",
    re.IGNORECASE
)

def extract_violation_section_text(text):
    """
    Returns (violation_block_text, n_violations).
    Violation blocks: numbered items starting with '1. 21 CFR...' or '1. During...'
    Stops each block at the next numbered item or a section-break header.
    If no numbered blocks found, returns (None, 0).
    """
    lines = text.split('\n')
    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if VIOLATION_START.match(line):
            block = [line]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if VIOLATION_START.match(nxt):
                    break   # next numbered violation
                if SECTION_BREAK.match(nxt.strip()):
                    break   # section boundary
                block.append(nxt)
                j += 1
                if j - i > 120:   # safety cap
                    break
            blocks.append('\n'.join(block))
            i = j
        else:
            i += 1

    if blocks:
        return '\n'.join(blocks), len(blocks)
    return None, 0


# ══════════════════════════════════════════════════════════════════════════
# FIX #5 — Repeat-section extractor (stop at next repeat header)
# ══════════════════════════════════════════════════════════════════════════
def extract_repeat_sections(text):
    """
    Finds both 'Repeat Observation at Facility' AND 'Repeat Violations at
    Multiple Sites' sections as separate blocks by stopping block collection
    as soon as the next repeat-section header is encountered.

    Returns dict:
      repeat_at_facility (bool), repeat_multi_site (bool),
      n_repeat_sections (int),
      repeat_section_texts (list[str]),   — one entry per section found
      n_repeat_section_chars (int)
    """
    lines = text.split("\n")
    repeat_facility = False
    repeat_multi    = False
    section_texts   = []

    i = 0
    while i < len(lines):
        line = lines[i]
        is_facility = any(p.search(line) for p in REPEAT_FACILITY_PATS)
        is_multi    = any(p.search(line) for p in REPEAT_MULTI_PATS)

        if is_facility or is_multi:
            if is_facility: repeat_facility = True
            if is_multi:    repeat_multi    = True

            block = [line]
            j = i + 1
            blank_streak = 0

            while j < len(lines):
                l = lines[j]

                # FIX: stop if we hit ANOTHER repeat-section header
                if any(p.search(l) for p in ALL_REPEAT_PATS):
                    break

                if l.strip() == "":
                    blank_streak += 1
                    if blank_streak >= 3:
                        break
                else:
                    blank_streak = 0

                block.append(l)
                j += 1
                if j - i > 60:    # safety cap per section
                    break

            section_texts.append("\n".join(block))
            i = j   # continue from next line (possibly another repeat header)
        else:
            i += 1

    return {
        "repeat_at_facility":      repeat_facility,
        "repeat_multi_site":       repeat_multi,
        "n_repeat_sections":       len(section_texts),
        "repeat_section_texts":    section_texts,
        "n_repeat_section_chars":  sum(len(t) for t in section_texts),
    }


# ══════════════════════════════════════════════════════════════════════════
# FIX #6 — Count genuine prior WL references (date-anchored)
# ══════════════════════════════════════════════════════════════════════════
def count_prior_wl_references(text):
    """
    Count prior WL references by requiring 'warning letter' to co-occur
    with a specific calendar date within the same sentence/phrase.
    Generic mentions ('the purpose of this warning letter is...') are excluded.

    Also returns all distinct dates mentioned in the text (for temporal context).
    """
    prior_wl_hits = PRIOR_WL_DATE_PATTERN.findall(text)
    all_dates     = DATE_IN_TEXT_PATTERN.findall(text)
    return len(prior_wl_hits), len(set(all_dates))


# ══════════════════════════════════════════════════════════════════════════
# Text extractor
# ══════════════════════════════════════════════════════════════════════════
def extract_text(pdf_path):
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print(f"  WARNING: pdfplumber failed for {pdf_path.name}: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════
# Parse one PDF — incorporates all fixes
# ══════════════════════════════════════════════════════════════════════════
def parse_wl_pdf(pdf_path):
    name  = pdf_path.stem
    parts = name.split(" - ")

    # search_fei = FEI from filename (what user searched; maps to our 129 FEIs)
    try:
        search_fei = int(parts[0].replace("FEI", "").strip())
    except Exception:
        search_fei = None

    wl_number = parts[2].strip() if len(parts) > 2 else "unknown"
    date_raw  = parts[3].replace("_ FDA", "").strip() if len(parts) > 3 else ""
    try:
        wl_date     = datetime.strptime(date_raw.replace("_", "/"), "%m/%d/%Y")
        wl_date_str = wl_date.strftime("%Y-%m-%d")
    except Exception:
        wl_date     = None
        wl_date_str = date_raw

    text   = extract_text(pdf_path)
    n_chars = len(text)

    if n_chars < 500:
        print(f"  SKIP (scanned or empty): {pdf_path.name}")
        return None

    # ── Fix #1: Primary FEI from text ─────────────────────────────────────
    primary_fei, primary_fei_source = extract_primary_fei_from_text(text, search_fei)
    fei_mismatch = (primary_fei != search_fei)

    # ── Fix #2: CFRs only from violation blocks ────────────────────────────
    viol_text, n_violations = extract_violation_section_text(text)
    if viol_text:
        cfr_hits = CFR_PATTERN.findall(viol_text)
    else:
        # Fallback: full text but exclude obvious boilerplate CFR contexts
        boilerplate_excl = re.compile(
            r"(?:qualified as set forth in|pursuant to|as defined in|"
            r"as required by|recommend[^.]*?consultant[^.]*?)\s*21\s+CFR",
            re.IGNORECASE
        )
        clean_text = boilerplate_excl.sub("EXCLUDED_CFR_CONTEXT", text)
        cfr_hits = CFR_PATTERN.findall(clean_text)

    cfr_unique_list = sorted(set(f"21 CFR {c}" for c in cfr_hits))
    n_cfr_unique    = len(cfr_unique_list)
    n_cfr_hits      = len(cfr_hits)

    # n_violations from block count; fallback to ≥ n_cfr_unique if blocks not found
    if n_violations == 0 and n_cfr_unique > 0:
        n_violations = max(1, n_cfr_unique // 2)

    # ── Fix #4: Domain counts (official subpart ranges) ────────────────────
    domains = Counter(cfr_domain(c) for c in cfr_unique_list)

    # ── Fix #5: Repeat sections ────────────────────────────────────────────
    repeat = extract_repeat_sections(text)
    # Join all section texts for storage
    repeat_section_text_joined = " ||| ".join(repeat["repeat_section_texts"])

    # ── All FEIs in text (for network) ────────────────────────────────────
    all_feis_in_text = [int(f) for f in FEI_IN_TEXT_PATTERN.findall(text)]
    # Indirect = any FEI mentioned that is neither the primary nor the search FEI
    indirect_feis = sorted(set(
        f for f in all_feis_in_text
        if f != primary_fei and f != search_fei
    ))

    # ── Fix #6: Prior WL references (date-anchored) ───────────────────────
    n_prior_wl_refs, n_dates_mentioned = count_prior_wl_references(text)

    # ── Management / corporate language ───────────────────────────────────
    text_lower = text.lower()
    has_mgmt_oversight = any(p in text_lower for p in MGMT_OVERSIGHT_PHRASES)
    has_corp_failure   = any(p in text_lower for p in CORP_FAILURE_PHRASES)
    # Fix #7: has_drug_shortage_mention removed

    return {
        # Identity
        "search_fei":             search_fei,        # filename FEI → maps to our 129 FEIs
        "primary_fei":            primary_fei,        # FEI from WL text (actual inspected facility)
        "primary_fei_source":     primary_fei_source,
        "fei_mismatch":           fei_mismatch,       # True = cross-site discovery
        "wl_number":              wl_number,
        "wl_date":                wl_date_str,
        "wl_year":                wl_date.year if wl_date else None,
        "pdf_file":               pdf_path.name,

        # Text size
        "n_chars_total":          n_chars,

        # Fix #2: CFRs from violation blocks only
        "n_violations":           n_violations,
        "n_cfr_unique":           n_cfr_unique,
        "n_cfr_hits_total":       n_cfr_hits,
        "cfr_list":               "; ".join(cfr_unique_list),
        "violation_text_found":   viol_text is not None,

        # Fix #4: Domain counts (official subpart ranges)
        "n_domain_org_personnel":      domains.get("org_personnel", 0),
        "n_domain_buildings_equip":    domains.get("buildings_equipment", 0),
        "n_domain_production":         domains.get("production", 0),
        "n_domain_packaging_labeling": domains.get("packaging_labeling", 0),
        "n_domain_lab_controls":       domains.get("lab_controls", 0),
        "n_domain_records_reports":    domains.get("records_reports", 0),
        "n_domain_bioresearch":        domains.get("bioresearch", 0),

        # Fix #5: Repeat sections (both facility + multi-site captured separately)
        "has_repeat_at_facility":      repeat["repeat_at_facility"],
        "has_repeat_multi_site":       repeat["repeat_multi_site"],
        "n_repeat_sections":           repeat["n_repeat_sections"],
        "n_repeat_section_chars":      repeat["n_repeat_section_chars"],
        "repeat_section_text":         repeat_section_text_joined[:2000],

        # Fix #6: Genuine prior WL references (date-anchored)
        "n_prior_wl_refs":             n_prior_wl_refs,
        "n_dates_mentioned":           n_dates_mentioned,

        # Semantic flags (kept — facility-specific)
        "has_management_oversight":    has_mgmt_oversight,
        "has_corporate_failure_lang":  has_corp_failure,

        # Cross-site FEI network
        "indirect_feis_mentioned":     "; ".join(str(f) for f in indirect_feis),
        "n_indirect_feis":             len(indirect_feis),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("WARNING LETTER FEATURE EXTRACTION  (v2 — all fixes applied)")
print("="*70)

pdf_files = sorted([f for f in WL_RAW.iterdir() if f.suffix == ".pdf"])
print(f"\nFound {len(pdf_files)} PDFs\n")

records = []
for pdf_path in pdf_files:
    print(f"Processing: {pdf_path.name}")
    rec = parse_wl_pdf(pdf_path)
    if rec:
        records.append(rec)
        mismatch_note = f"  ⚠ FEI MISMATCH: search={rec['search_fei']} → primary={rec['primary_fei']} [{rec['primary_fei_source']}]" \
                        if rec["fei_mismatch"] else f"  FEI {rec['primary_fei']} (matches search)"
        print(f"  {mismatch_note}")
        print(f"    {rec['wl_date']} | {rec['n_chars_total']:,} chars | "
              f"{rec['n_cfr_unique']} violation CFRs | {rec['n_violations']} violations | "
              f"repeat_fac={rec['has_repeat_at_facility']} "
              f"repeat_multi={rec['has_repeat_multi_site']} | "
              f"n_repeat_sections={rec['n_repeat_sections']} | "
              f"prior_wl_refs={rec['n_prior_wl_refs']}")

records_df = pd.DataFrame(records)
records_df.to_csv(WL_OUT / "warning_letter_records.csv", index=False)
print(f"\nSaved {len(records_df)} records → warning_letter_records.csv")

# ── FEI mismatch report ───────────────────────────────────────────────────
mismatches = records_df[records_df["fei_mismatch"] == True]
if not mismatches.empty:
    print(f"\n{'='*70}")
    print(f"FEI MISMATCH REPORT: {len(mismatches)} WLs where search FEI ≠ primary FEI")
    print(f"{'='*70}")
    print(mismatches[["pdf_file", "search_fei", "primary_fei",
                       "primary_fei_source", "wl_date"]].to_string(index=False))
    print("\nNote: These WLs were found by searching 'search_fei' but the letter")
    print("      was formally issued to 'primary_fei' — a cross-site link.")


# ══════════════════════════════════════════════════════════════════════════
# AGGREGATE TO FEI LEVEL  (group by search_fei — maps to our 129-FEI list)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("AGGREGATING TO FEI LEVEL (by search_fei)")
print("="*70)

bool_cols  = ["has_repeat_at_facility", "has_repeat_multi_site",
              "has_management_oversight", "has_corporate_failure_lang",
              "fei_mismatch"]
count_cols = ["n_violations", "n_cfr_unique", "n_repeat_sections",
              "n_repeat_section_chars", "n_prior_wl_refs", "n_indirect_feis",
              "n_chars_total",
              "n_domain_org_personnel", "n_domain_buildings_equip",
              "n_domain_production", "n_domain_packaging_labeling",
              "n_domain_lab_controls", "n_domain_records_reports",
              "n_domain_bioresearch"]

agg_dict = {c: "max" for c in bool_cols}
agg_dict.update({c: "sum" for c in count_cols})
agg_dict["wl_date"]   = "max"    # most recent WL date
agg_dict["wl_number"] = "count"  # repurposed: number of WLs per FEI
agg_dict["primary_fei"] = lambda x: "; ".join(str(v) for v in sorted(set(x)))
agg_dict["cfr_list"]  = lambda x: "; ".join(sorted(set(
    item for sublist in x for item in sublist.split("; ") if item
)))

fei_agg = records_df.groupby("search_fei").agg(agg_dict).reset_index()
fei_agg.rename(columns={
    "wl_number":    "n_warning_letters",
    "wl_date":      "most_recent_wl_date",
    "primary_fei":  "primary_feis_in_wls",   # may differ from search_fei
}, inplace=True)

# Rename bool max → ever_*
for c in bool_cols:
    fei_agg.rename(columns={c: "ever_" + c.replace("has_", "")}, inplace=True)

# Composite severity score (updated: uses n_prior_wl_refs instead of raw WL count)
fei_agg["wl_severity_score"] = (
    fei_agg["n_warning_letters"] * 2
    + fei_agg["n_violations"]
    + fei_agg["ever_repeat_at_facility"].astype(int) * 3
    + fei_agg["ever_repeat_multi_site"].astype(int) * 5
    + fei_agg["ever_management_oversight"].astype(int) * 2
    + fei_agg["ever_corporate_failure_lang"].astype(int) * 3
    + fei_agg["n_prior_wl_refs"] * 2   # genuine prior WL references
)

fei_agg.to_csv(WL_OUT / "warning_letter_fei_features.csv", index=False)
print(fei_agg[["search_fei", "primary_feis_in_wls", "n_warning_letters",
               "n_violations", "n_cfr_unique",
               "n_repeat_sections", "ever_repeat_at_facility", "ever_repeat_multi_site",
               "ever_management_oversight", "ever_corporate_failure_lang",
               "n_prior_wl_refs", "wl_severity_score",
               "most_recent_wl_date"]].to_string())


# ══════════════════════════════════════════════════════════════════════════
# CROSS-SITE FEI NETWORK  (three edge types)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CROSS-SITE FEI NETWORK")
print("="*70)

network_rows = []
for _, row in records_df.iterrows():
    # Edge type 1: search_fei ↔ primary_fei  (cross-site discovery)
    if row["fei_mismatch"]:
        network_rows.append({
            "fei_a":       row["search_fei"],
            "fei_b":       row["primary_fei"],
            "edge_type":   "cross_site_discovery",
            "wl_date":     row["wl_date"],
            "wl_number":   row["wl_number"],
            "description": f"Searching FEI {row['search_fei']} surfaced WL for FEI {row['primary_fei']}",
        })

    # Edge type 2: primary_fei ↔ each indirect FEI mentioned in text
    if row["indirect_feis_mentioned"]:
        for fei_b in str(row["indirect_feis_mentioned"]).split("; "):
            if fei_b.strip():
                network_rows.append({
                    "fei_a":       row["primary_fei"],
                    "fei_b":       int(fei_b.strip()),
                    "edge_type":   "wl_text_mention" + (
                                       "_repeat_multi" if row["has_repeat_multi_site"] else ""),
                    "wl_date":     row["wl_date"],
                    "wl_number":   row["wl_number"],
                    "description": f"FEI {row['primary_fei']} WL mentions FEI {fei_b.strip()}",
                })

network_df = pd.DataFrame(network_rows)
if not network_df.empty:
    network_df.to_csv(WL_OUT / "wl_fei_network.csv", index=False)
    print(f"\nNetwork edges: {len(network_df)}")
    print(network_df.to_string(index=False))
else:
    print("No cross-site edges found.")


# ══════════════════════════════════════════════════════════════════════════
# TEXT RICHNESS COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEXT RICHNESS COMPARISON")
print("="*70)

if CITATION_CSV.exists():
    cit = pd.read_csv(CITATION_CSV)
    avg_long_desc  = cit["Long Description"].fillna("").str.len().mean()
    avg_short_desc = cit["Short Description"].fillna("").str.len().mean()
else:
    avg_long_desc  = 158
    avg_short_desc = 45

avg_wl_total    = records_df["n_chars_total"].mean()
avg_wl_per_viol = (records_df["n_chars_total"] /
                   records_df["n_violations"].replace(0, 1)).mean()

richness = pd.DataFrame([
    {"Source": "Citation DB — CFR Code",         "Avg_Chars": 14,
     "Facility_Specific": "No",  "Repeat_History": "No",  "Cross_Site_Info": "No"},
    {"Source": "Citation DB — Short Description", "Avg_Chars": int(avg_short_desc),
     "Facility_Specific": "No",  "Repeat_History": "No",  "Cross_Site_Info": "No"},
    {"Source": "Citation DB — Long Description",  "Avg_Chars": int(avg_long_desc),
     "Facility_Specific": "No",  "Repeat_History": "No",  "Cross_Site_Info": "No"},
    {"Source": "483 PDF — Observation Body",      "Avg_Chars": 2258,
     "Facility_Specific": "Yes", "Repeat_History": "Sometimes", "Cross_Site_Info": "No"},
    {"Source": "Warning Letter — Full Text",      "Avg_Chars": int(avg_wl_total),
     "Facility_Specific": "Yes", "Repeat_History": "Yes (formal section)",
     "Cross_Site_Info": "Yes"},
    {"Source": "Warning Letter — Per Violation",  "Avg_Chars": int(avg_wl_per_viol),
     "Facility_Specific": "Yes", "Repeat_History": "Yes", "Cross_Site_Info": "Sometimes"},
])
richness.to_csv(WL_OUT / "wl_richness_comparison.csv", index=False)
print(richness.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total WLs processed:            {len(records_df)}")
print(f"Unique search FEIs:             {records_df['search_fei'].nunique()}")
print(f"WLs with FEI mismatch:          {records_df['fei_mismatch'].sum()}"
      f"  (search FEI ≠ actual inspected FEI)")
print(f"FEIs with repeat-at-facility:   {fei_agg['ever_repeat_at_facility'].sum()}")
print(f"FEIs with repeat-multi-site:    {fei_agg['ever_repeat_multi_site'].sum()}")
print(f"FEIs with mgmt oversight lang:  {fei_agg['ever_management_oversight'].sum()}")
print(f"FEIs with corporate failure:    {fei_agg['ever_corporate_failure_lang'].sum()}")
print(f"Total genuine prior WL refs:    {fei_agg['n_prior_wl_refs'].sum()}")
print(f"\nAvg WL length:           {avg_wl_total:,.0f} chars")
print(f"Avg WL per-violation:    {avg_wl_per_viol:,.0f} chars")
print(f"Avg Long Description:    {avg_long_desc:.0f} chars")
print(f"Ratio WL / citation:     {avg_wl_total / avg_long_desc:.0f}x richer")
print(f"\nAll outputs saved to: {WL_OUT}")
