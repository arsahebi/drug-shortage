# %%
"""
NDC → FEI Verification
=======================
Systematically verifies NDC→FEI assignments for all 112 Valisure-tested metformin NDCs.

Sources:
  Primary  : DailyMed REST API  — SPL XML contains manufacturer FEI in a standardized field
  Fallback : Claude API (haiku) — fetches DailyMed/ProPublica page and extracts FEI

Cross-reference:
  Redica Site List (Data/07 - Redica/raw/Site List.xlsx) — ground-truth FEI↔firm mapping

Output columns:
  ndc11, ndc_display, firm_panel
  fei_sheet1   — FEI from original Sheet1 (NaN if NDC was not in Sheet1)
  fei_amir     — FEI Amir found via DailyMed/ProPublica (NaN if NDC was already in Sheet1)
  fei_verified — FEI found by this script
  firm_verified — firm name for fei_verified (from Redica Site List)
  source        — how fei_verified was found
  all_feis_in_spl — all FEIs extracted from SPL (for reference)
  mismatch      — 1 if fei_verified disagrees with sheet1 or amir assignment
  mismatch_detail — which assignment it disagrees with
"""

import re
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import anthropic
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE      = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
QA_FILE   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST = BASE / "Data/07 - Redica/raw/Site List.xlsx"
PANEL_V1  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/metformin_panel_v1.csv"
OUT_FILE  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_verification.csv"

DAILYMED_NDC_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?ndc={ndc}"
DAILYMED_SPL_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
HEADERS          = {"User-Agent": "Mozilla/5.0 academic-research/drug-shortage-study"}
SLEEP_SEC        = 0.6   # polite rate limiting for DailyMed


# =============================================================================
# HELPERS
# =============================================================================
def clean_fei(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.search(r"[a-zA-Z]", s):
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return None


def to_ndc11(x):
    if pd.isna(x) or str(x).strip() in ("", "nan"):
        return None
    s = str(x).strip()
    parts = [p for p in s.replace(" ", "").split("-") if p]
    if len(parts) == 3:
        lab, prod, pkg = parts
        return lab.zfill(5) + prod.zfill(4) + pkg.zfill(2)[-2:]
    raw = s.replace("-", "").replace(" ", "")
    if len(raw) == 10:
        return raw[:5] + "0" + raw[5:]
    if len(raw) == 11:
        return raw
    return None


def ndc11_to_formats(ndc11):
    """Return list of NDC format strings to try in the DailyMed API."""
    lab, prod4, pkg = ndc11[:5], ndc11[5:9], ndc11[9:]
    prod3 = prod4.lstrip("0").zfill(3) if prod4.lstrip("0") else "000"
    return [
        f"{lab}-{prod3}-{pkg}",   # 5-3-2 hyphenated (most common on DailyMed)
        f"{lab}-{prod4}-{pkg}",   # 5-4-2 hyphenated
        ndc11,                     # 11-digit bare
        lab + prod3 + pkg,         # 10-digit bare
    ]


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading source data...")

# Redica Site List: ground-truth FEI ↔ firm
sl = pd.read_excel(SITE_LIST, dtype=str)
sl["FEI"] = sl["FEI"].astype(str).str.strip()
sl["redica_firm"] = sl["Site Display Name"].str.split("[").str[0].str.strip()
REDICA_FEIS   = set(sl["FEI"].unique())
FEI_TO_FIRM   = sl.set_index("FEI")["redica_firm"].to_dict()

# Panel: all 112 NDCs with their current FEI assignments
panel = pd.read_csv(PANEL_V1, dtype=str)
panel_ndc = (
    panel[["NDC11", "NDC", "Firm", "FEI"]]
    .drop_duplicates("NDC11")
    .reset_index(drop=True)
)
panel_ndc["fei_panel"] = panel_ndc["FEI"].apply(clean_fei)

# Sheet1 FEI assignments (original 88 NDCs)
xls    = pd.ExcelFile(QA_FILE)
sheet1 = xls.parse("Sheet1", dtype=str)
sheet1["ndc11"] = sheet1["NDC11"].apply(to_ndc11)
sheet1["fei"]   = sheet1["FEI"].apply(clean_fei)
S1_FEI = (
    sheet1.dropna(subset=["ndc11", "fei"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["fei"]
    .to_dict()
)

# Amir's FEI assignments (col H: new NDCs not in Sheet1)
amir_raw = xls.parse("Amir-Unique NDC from Valisure (", header=None, dtype=str)
amir_raw.columns = ["NDC", "NDC11", "c", "d", "e", "NDC11_F",
                    "FEI_G", "Found_FEI_H", "Notes1", "Notes2", "extra"]
amir_raw = amir_raw.iloc[1:].reset_index(drop=True)
amir_raw["ndc11"] = (
    amir_raw["NDC11_F"].apply(to_ndc11)
    .fillna(amir_raw["NDC11"].apply(to_ndc11))
)
amir_raw["fei"] = amir_raw["Found_FEI_H"].apply(clean_fei)
AMIR_FEI = (
    amir_raw.dropna(subset=["ndc11", "fei"])
    .drop_duplicates("ndc11")
    .set_index("ndc11")["fei"]
    .to_dict()
)

print(f"  Panel NDCs: {len(panel_ndc)}")
print(f"  Sheet1 FEI assignments: {len(S1_FEI)}")
print(f"  Amir FEI assignments:   {len(AMIR_FEI)}")
print(f"  Redica FEIs (ground truth): {len(REDICA_FEIS)}")


# =============================================================================
# 2. DailyMed: get organisation names from SPL XML
# =============================================================================
from lxml import etree as lxml_etree

# Ingredient words to exclude when identifying company names from SPL XML
_INGREDIENT_TOKENS = {
    "metformin", "hydrochloride", "tablet", "capsule", "extended", "release",
    "immediate", "solution", "suspension", "injection", "oral", "modified",
    "SILICON", "DIOXIDE", "MAGNESIUM", "STEARATE", "CELLULOSE", "POVIDONE",
    "CROSPOVIDONE", "HYPROMELLOSES", "HYPROMELLOSE", "POLYETHYLENE", "GLYCOL",
    "DIBUTYL", "SEBACATE", "AMMONIO", "METHACRYLATE", "COPOLYMER", "TALC",
    "LACTOSE", "MONOHYDRATE", "STEARIC", "ACID", "OLEIC", "ETHYLCELLULOSES",
    "TRIACETIN", "TITANIUM", "XANTHAN", "TRIGLYCERIDES", "MEDIUM", "CHAIN",
}


def _is_company_name(name: str) -> bool:
    """Heuristic: exclude drug/ingredient names, keep company names."""
    if len(name) < 5 or len(name) > 100:
        return False
    upper = name.upper()
    if any(tok in upper for tok in _INGREDIENT_TOKENS):
        return False
    # Company indicators
    if any(ind in upper for ind in ["INC", "LLC", "LTD", "LIMITED", "PHARMA",
                                     "LABORATORIES", "LABS", "COMPANY", "CORP",
                                     "INDUSTRIES", "HEALTHCARE", "PHARMACEUTICAL"]):
        return True
    # Short names that are clearly companies (2+ words, mostly uppercase letters)
    if re.match(r"^[A-Z][A-Z\s\.\,\-]+$", name) and len(name.split()) >= 2:
        return True
    return False


def get_spl_org_names(ndc_str: str) -> tuple[list[str], str | None]:
    """
    Query DailyMed for ndc_str, fetch the SPL XML, return (org_names, setid).
    org_names: deduplicated company names extracted from the XML.
    """
    url = DAILYMED_NDC_URL.format(ndc=ndc_str)
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return [], None
        items = r.json().get("data", [])
        if not items:
            return [], None
        setid = items[0]["setid"]
        time.sleep(SLEEP_SEC)
        spl_r = requests.get(DAILYMED_SPL_URL.format(setid=setid), headers=HEADERS, timeout=15)
        if spl_r.status_code != 200:
            return [], setid
        xml_root = lxml_etree.fromstring(spl_r.content)
        names = []
        for elem in xml_root.iter():
            if elem.tag.endswith("}name") or elem.tag == "name":
                txt = (elem.text or "").strip()
                if _is_company_name(txt):
                    names.append(txt)
        return list(dict.fromkeys(names)), setid
    except Exception as e:
        print(f"    DailyMed error: {e}")
        return [], None


def lookup_via_dailymed(ndc11: str) -> tuple[list[str], str | None]:
    """Try all NDC formats. Returns (org_names, setid) from first successful hit."""
    for fmt in ndc11_to_formats(ndc11):
        time.sleep(SLEEP_SEC)
        names, setid = get_spl_org_names(fmt)
        if names:
            return names, setid
    return [], None


# =============================================================================
# 3. Claude haiku: match org names → Redica FEI
# =============================================================================
claude_client = anthropic.Anthropic()

# Provide Claude with the full Redica site list as context
REDICA_LIST_TEXT = "\n".join(
    f"  FEI {fei}: {firm}"
    for fei, firm in sorted(FEI_TO_FIRM.items())
)


def claude_match_to_redica(ndc_display: str, org_names: list[str]) -> tuple[str | None, str]:
    """
    Ask Claude haiku to match org_names (from DailyMed SPL) to a Redica facility.
    Returns (fei_verified, source_label).
    """
    names_str = "\n".join(f"  - {n}" for n in org_names)
    prompt = (
        f"I have a drug label for NDC {ndc_display}. The label lists these organisations:\n"
        f"{names_str}\n\n"
        f"Below is a list of known manufacturing facilities with their FEI numbers.\n"
        f"Identify which facility MANUFACTURED this drug (not the US distributor).\n"
        f"Return ONLY the FEI number of the best match, or NOT_FOUND if none match.\n\n"
        f"Known facilities:\n{REDICA_LIST_TEXT}"
    )
    try:
        resp = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        feis = re.findall(r"\b\d{7,10}\b", text)
        in_redica = [f for f in feis if f in REDICA_FEIS]
        if in_redica:
            return in_redica[0], "dailymed+claude_haiku"
        return None, "claude_no_match"
    except Exception as e:
        print(f"    Claude API error: {e}")
        return None, "claude_error"


# =============================================================================
# 4. MAIN VERIFICATION LOOP
# =============================================================================
print("\nVerifying NDC→FEI assignments...")
results = []

for i, row in panel_ndc.iterrows():
    ndc11       = to_ndc11(str(row["NDC11"]))   # bare 11-digit, e.g. "00378060091"
    ndc_display = str(row["NDC"])
    firm_panel  = str(row["Firm"]) if pd.notna(row["Firm"]) else ""

    fei_sheet1 = S1_FEI.get(ndc11)
    fei_amir   = AMIR_FEI.get(ndc11)
    # Canonical "existing" assignment: Sheet1 takes priority over Amir
    fei_existing = fei_sheet1 or fei_amir

    print(f"[{i+1:3d}/112] {ndc_display}  existing={fei_existing}", end="  →  ")

    # --- Step 1: DailyMed → org names ---
    org_names, setid = lookup_via_dailymed(ndc11)

    fei_verified  = None
    firm_verified = None
    source        = None

    # --- Step 2: Claude haiku matches org names → Redica FEI ---
    if org_names:
        fei_verified, source = claude_match_to_redica(ndc_display, org_names)
        if fei_verified:
            firm_verified = FEI_TO_FIRM.get(fei_verified, "")
    else:
        source = "not_found"

    print(f"verified={fei_verified}  ({source})")

    # --- Mismatch logic ---
    # Compare fei_verified against both Sheet1 and Amir assignments
    mismatch_sheet1 = None
    mismatch_amir   = None
    if fei_verified and fei_sheet1:
        mismatch_sheet1 = int(fei_verified != fei_sheet1)
    if fei_verified and fei_amir:
        mismatch_amir = int(fei_verified != fei_amir)

    # Overall mismatch flag: 1 if any disagreement with an existing assignment
    mismatch = None
    detail_parts = []
    if mismatch_sheet1 == 1:
        detail_parts.append(f"sheet1={fei_sheet1}")
    if mismatch_amir == 1:
        detail_parts.append(f"amir={fei_amir}")
    if detail_parts:
        mismatch = 1
        mismatch_detail = "MISMATCH vs " + ", ".join(detail_parts)
    elif fei_verified and fei_existing:
        mismatch = 0
        mismatch_detail = "ok"
    else:
        mismatch_detail = "no_existing_assignment" if not fei_existing else "not_verified"

    results.append({
        "ndc11":            ndc11,
        "ndc_display":      ndc_display,
        "firm_panel":       firm_panel,
        "fei_sheet1":       fei_sheet1,
        "fei_amir":         fei_amir,
        "fei_verified":     fei_verified,
        "firm_verified":    firm_verified,
        "source":           source,
        "org_names_in_spl": " | ".join(org_names) if org_names else "",
        "mismatch_sheet1":  mismatch_sheet1,
        "mismatch_amir":    mismatch_amir,
        "mismatch":         mismatch,
        "mismatch_detail":  mismatch_detail,
    })

# =============================================================================
# 5. SAVE AND SUMMARISE
# =============================================================================
df = pd.DataFrame(results)
df.to_csv(OUT_FILE, index=False)

print(f"\n{'='*60}")
print(f"Saved: {OUT_FILE}")
print(f"\nSource breakdown:")
print(df["source"].value_counts().to_string())
print(f"\nMismatch summary:")
print(f"  vs Sheet1 assignments : {df['mismatch_sheet1'].sum()} mismatches")
print(f"  vs Amir assignments   : {df['mismatch_amir'].sum()} mismatches")
print(f"  Not verified          : {(df['source'] == 'not_found').sum()} NDCs")
print(f"\nNDCs requiring review (mismatch=1):")
flags = df[df["mismatch"] == 1][["ndc_display", "firm_panel", "fei_sheet1", "fei_amir", "fei_verified", "firm_verified", "mismatch_detail"]]
if len(flags):
    print(flags.to_string(index=False))
else:
    print("  None — all verified assignments match!")
# %%
