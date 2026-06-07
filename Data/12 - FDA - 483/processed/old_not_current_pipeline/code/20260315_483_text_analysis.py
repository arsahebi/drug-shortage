# %%
"""
20260315_483_text_analysis.py

Demonstrates how the body text of FDA 483 observations adds predictive value
beyond what the Citation database (CFR code + descriptions) provides.

Key question:
  If the same CFR code can appear in both OAI and VAI inspections,
  what does the body text tell us that the header alone cannot?

PDFs used (all have extractable text):
  OAI: 3002809586 - 06132025  (Sun Pharma, Halol — 8 observations, 19 pages)
  OAI: 3002984011 - 05032019  (Cadila Healthcare  — 14 observations, 23 pages)
  VAI: 3006370533 - 03272024  (Alkem Laboratories —  1 observation,  ~4 pages)

NOTE on 3002809586 - 08312018:
  That PDF is a scanned image (no extractable text). Also, the inspection DB
  records that date as NAI (Drug QA), not VAI. The VAI for that facility
  (02/23/2018) is also scanned. We therefore use Alkem 2024 as the VAI
  comparator since it shares CFR 211.192 with the Sun Pharma OAI.
"""

import re
import pandas as pd
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
RAW    = BASE / "12 - FDA - 483" / "raw"
OUT    = Path(__file__).parent

try:
    import pdfplumber
except ImportError:
    raise SystemExit("pip install pdfplumber")

# ── Utility: extract observations from a 483 PDF ───────────────────────────
def extract_observations(pdf_path: Path) -> list[dict]:
    """Return list of {label, header, body, full, n_chars} per observation."""
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
    full_text = "\n".join(pages)

    parts = re.split(r"(OBSERVATION\s+\d+)", full_text)
    observations = []
    for i in range(1, len(parts) - 1, 2):
        raw = parts[i + 1]
        # Trim at form footer / employee signature block
        raw = re.split(r"EMPLOYEE.{0,10}SIGNATURE|FORM FDA 483|PAGE \d+ of", raw)[0]
        raw = re.sub(r"\s+", " ", raw).strip()

        # Split header (first sentence) from body (everything after)
        first_period = raw.find(".")
        if first_period > 0:
            header = raw[: first_period + 1].strip()
            body   = raw[first_period + 1 :].strip()
        else:
            header = raw
            body   = ""

        observations.append({
            "label":  parts[i].strip(),
            "header": header,
            "body":   body,
            "full":   raw,
            "n_chars": len(raw),
        })
    return observations


# ── Severity signal extractor ──────────────────────────────────────────────
SEVERITY_KEYWORDS = {
    "repeat_violation":  [r"\brepeat\b", r"repeated", r"recurrent", r"again"],
    "warning_letter":    [r"warning letter", r"consent decree"],
    "systemic_failure":  [r"frequently", r"consistently", r"routinely", r"multiple", r"numerous"],
    "patient_impact":    [r"US market", r"distributed", r"released.*market", r"recall"],
    "data_integrity":    [r"data integrity", r"falsif", r"fabricat", r"alter.*record"],
    "contamination":     [r"contaminat", r"sterile", r"microb", r"HEPA"],
}

def extract_signals(text: str) -> dict:
    text_lower = text.lower()
    signals = {}
    for signal, patterns in SEVERITY_KEYWORDS.items():
        signals[signal] = any(re.search(p, text_lower) for p in patterns)

    # Count sub-examples (numbered or lettered list items)
    signals["n_examples"] = len(re.findall(r"(?<!\w)[1-9A-F]\.", text))
    signals["n_chars"]    = len(text)
    return signals


# ── Load PDFs ──────────────────────────────────────────────────────────────
pdfs = {
    "OAI · Sun Pharma 2025":  RAW / "3002809586 - 06132025 - Sun Pharmaceutical.pdf.pdf",
    "OAI · Cadila 2019":      RAW / "3002984011 - 05032019 - Cadila Healthcare Ltd, Ahmedabad, Gujarat, 382210, India .pdf",
    "VAI · Alkem 2024":       RAW / "3006370533 - 03272024 - Alkem Laboratories Limited, Baddi, India.pdf",
}

all_obs = {}
for label, path in pdfs.items():
    obs = extract_observations(path)
    all_obs[label] = obs
    print(f"{label}: {len(obs)} observations, {sum(o['n_chars'] for o in obs):,} chars total")

print()


# ── TABLE 1: What the Citation DB tells you vs what the PDF adds ──────────
# Focus on CFR 211.192 which appears in BOTH OAI (Sun Pharma) and VAI (Alkem)
print("=" * 80)
print("TABLE 1 — Same CFR, Same Header, Completely Different Body Text")
print("         CFR 21 CFR 211.192 : 'Investigations of discrepancies, failures'")
print("=" * 80)

# OAI: Observation 3 is 211.192 (Sun Pharma 2025)
oai_211192 = all_obs["OAI · Sun Pharma 2025"][2]
# VAI: Observation 1 is 211.192 (Alkem 2024)
vai_211192 = all_obs["VAI · Alkem 2024"][0]

for name, obs in [("OAI (Sun Pharma 2025)", oai_211192), ("VAI (Alkem 2024)", vai_211192)]:
    sigs = extract_signals(obs["full"])
    print(f"\n── {name} ──")
    print(f"  CFR Code    : 21 CFR 211.192")
    print(f"  Short Desc  : Investigations of discrepancies, failures")
    print(f"  Long Desc   : {obs['header'][:100]}...")
    print(f"  [Citation DB stops here — everything below is ONLY in the 483 PDF]")
    print(f"  Body text   : {obs['body'][:350]}...")
    print(f"  Chars (body): {len(obs['body'])}")
    print(f"  Signals:")
    for k, v in sigs.items():
        if k not in ("n_chars",):
            print(f"    {k:25s}: {v}")

print()


# ── TABLE 2: Observation-level text metrics, OAI vs VAI ───────────────────
print("=" * 80)
print("TABLE 2 — Text Characteristics by Inspection Outcome")
print("=" * 80)

rows = []
for insp_label, obs_list in all_obs.items():
    outcome = "OAI" if "OAI" in insp_label else "VAI"
    for obs in obs_list:
        sigs = extract_signals(obs["full"])
        rows.append({
            "Inspection": insp_label,
            "Outcome": outcome,
            "Observation": obs["label"],
            **sigs,
        })

df = pd.DataFrame(rows)

summary = (
    df.groupby("Outcome")
    .agg(
        avg_body_chars  =("n_chars", "mean"),
        avg_examples    =("n_examples", "mean"),
        pct_repeat      =("repeat_violation", "mean"),
        pct_warning_ltr =("warning_letter", "mean"),
        pct_systemic    =("systemic_failure", "mean"),
        pct_patient_imp =("patient_impact", "mean"),
        pct_data_integ  =("data_integrity", "mean"),
        pct_contaminat  =("contamination", "mean"),
    )
    .round(2)
)
print(summary.T.to_string())
print()

df.to_csv(OUT / "483_observation_signals.csv", index=False)
print(f"Saved: {OUT / '483_observation_signals.csv'}")


# ── TABLE 3: The 3-layer hierarchy of information ─────────────────────────
print()
print("=" * 80)
print("TABLE 3 — Information Hierarchy: CFR → Short → Long → 483 Body")
print("=" * 80)

hierarchy = [
    {
        "Layer":        "CFR Code",
        "Source":       "Citation DB",
        "Example":      "21 CFR 211.192",
        "What you know": "Regulatory category (investigations of failures)",
        "Facility-specific?": "No — same code for all facilities",
        "ML utility":   "Low (categorical feature only)",
    },
    {
        "Layer":        "Short Description",
        "Source":       "Citation DB",
        "Example":      "Investigations of discrepancies, failures",
        "What you know": "Brief human-readable label for the CFR",
        "Facility-specific?": "No — same label for all facilities",
        "ML utility":   "Low-medium (good for grouping / co-occurrence)",
    },
    {
        "Layer":        "Long Description",
        "Source":       "Citation DB",
        "Example":      "There is a failure to thoroughly review any unexplained discrepancy...",
        "What you know": "Standardized first sentence of the observation (CFR template)",
        "Facility-specific?": "No — identical wording for all facilities citing this CFR",
        "ML utility":   "Low (no new info beyond CFR code)",
    },
    {
        "Layer":        "483 Body Text",
        "Source":       "483 PDF only",
        "Example":      '"Investigations frequently fail... repeat from Dec 2022 Warning Letter"',
        "What you know": "Specific products, dates, scope, severity, repeat flag",
        "Facility-specific?": "YES — unique to this facility and inspection",
        "ML utility":   "High (severity signals, systemic vs. isolated, repeat flag)",
    },
]

hier_df = pd.DataFrame(hierarchy)
print(hier_df.to_string(index=False))
hier_df.to_csv(OUT / "483_information_hierarchy.csv", index=False)


# ── TABLE 4: Per-observation summary for OAI inspection ───────────────────
print()
print("=" * 80)
print("TABLE 4 — OAI Inspection (Sun Pharma 2025): Per-Observation Signal Profile")
print("=" * 80)

oai_rows = []
for obs in all_obs["OAI · Sun Pharma 2025"]:
    sigs = extract_signals(obs["full"])
    oai_rows.append({
        "Obs":           obs["label"].replace("OBSERVATION ", ""),
        "Header (short)": obs["header"][:55] + "...",
        "Body chars":    len(obs["body"]),
        "Examples (#)":  sigs["n_examples"],
        "Repeat":        "✓" if sigs["repeat_violation"] else "",
        "Warn. Letter":  "✓" if sigs["warning_letter"] else "",
        "Systemic":      "✓" if sigs["systemic_failure"] else "",
        "US market":     "✓" if sigs["patient_impact"] else "",
        "Contam.":       "✓" if sigs["contamination"] else "",
    })

oai_df = pd.DataFrame(oai_rows)
print(oai_df.to_string(index=False))
oai_df.to_csv(OUT / "483_oai_observation_profile.csv", index=False)


# ── Key takeaway ───────────────────────────────────────────────────────────
print()
print("=" * 80)
print("KEY TAKEAWAY FOR QUALITY PREDICTION")
print("=" * 80)
print("""
The Citation Database tells you WHAT was cited (the CFR code).
The 483 body text tells you HOW SERIOUS it was.

Both the OAI (Sun Pharma) and VAI (Alkem) received CFR 211.192 citations.
But the 483 body reveals:

  OAI · Sun Pharma 2025:
    → "investigations FREQUENTLY fail" (systemic, not isolated)
    → "repeat observations from the December 2022 Warning Letter"
    → environmental monitoring failures across entire sterile production

  VAI · Alkem 2024:
    → One specific instrument (UV spectrophotometer) malfunctioned
    → No impact assessment performed for those batches
    → Isolated, traceable, correctable issue

Text features that differentiate OAI from VAI/NAI:
  1. Repeat-violation flag    → historically highest OAI predictor
  2. "Frequently" / systemic  → indicates organization-wide problem
  3. Warning Letter reference → escalated prior enforcement history
  4. Observation length       → longer = more specific = more serious
  5. Number of examples       → more examples = broader scope

→ These can be extracted as BINARY FLAGS or COUNTS for each 483,
  added to the CFR-based feature matrix, and used in quality prediction.
""")

# %%
