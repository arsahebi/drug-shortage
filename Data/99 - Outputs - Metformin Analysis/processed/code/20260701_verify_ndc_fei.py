# %%
"""
NDC → FEI Verification  (v2)
==============================
Strategy:
  1. Query DailyMed with the "DailyMed Friendly" NDC format (5-3, no package suffix).
  2. Fetch SPL XML; extract DUNS numbers (OID 1.3.6.1.4.1.519.1).
  3. Look up each DUNS in FDA DRLS registration file → get FEI.
     - If multiple FEI matches, prefer Redica FEIs (confirmed manufacturers); prefer
       non-US among those (actual foreign manufacturer over US distributor).
  4. If no DUNS hit → extract postal codes from XML address elements; try zip-code
     substring match against DRLS ADDRESS field.
  5. Record result with a short note on method used.

Output: NDC, FEI, note  (simple — to compare against Amir/Amirreza manual review)
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path
from lxml import etree

# =============================================================================
# PATHS
# =============================================================================
BASE      = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
QA_FILE   = BASE / "Data/06 - Metformin Data/Derived/Q&As1234_v8_v02.xlsx"
SITE_LIST = BASE / "Data/07 - Redica/raw/Site List.xlsx"
DRLS_FILE = BASE / "Data/05 - Firm Level/drls_reg.xlsx"
OUT_FILE  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/20260701_ndc_fei_verification.csv"

DAILYMED_NDC_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?ndc={ndc}"
DAILYMED_SPL_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
HEADERS   = {"User-Agent": "Mozilla/5.0 academic-research/drug-shortage-study"}
SLEEP_SEC = 0.7

DUNS_OID  = "1.3.6.1.4.1.519.1"
HL7_NS    = "urn:hl7-org:v3"


# =============================================================================
# 1. LOAD REFERENCE DATA
# =============================================================================
print("Loading reference data...")

# Redica Site List — ground-truth FEI set (127 known manufacturers)
sl = pd.read_excel(SITE_LIST, dtype=str)
sl["FEI"] = sl["FEI"].str.strip()
sl["firm"] = sl["Site Display Name"].str.split("[").str[0].str.strip()
REDICA_FEIS  = set(sl["FEI"])
REDICA_FIRM  = sl.set_index("FEI")["firm"].to_dict()
REDICA_SITE  = sl.set_index("FEI")["Site Display Name"].to_dict()

# DRLS — DUNS → FEI (and ADDRESS for zip fallback)
drls = pd.read_excel(DRLS_FILE, dtype=str)
drls["FEI_NUMBER"]  = drls["FEI_NUMBER"].str.strip()
drls["DUNS_NUMBER"] = drls["DUNS_NUMBER"].str.strip()
drls["ADDRESS"]     = drls["ADDRESS"].fillna("")
DUNS_TO_ROWS = {}
for _, r in drls.iterrows():
    d = r["DUNS_NUMBER"]
    if d:
        DUNS_TO_ROWS.setdefault(d, []).append(r)

# NDC panel — "Amir and Amirreza Review all ND" tab
tab = pd.read_excel(QA_FILE, sheet_name="Amir and Amirreza Review all ND", dtype=str)
# Columns: NDC11, DailyMed Friendly, ...
ndc_df = (
    tab[["NDC11", "DailyMed Friendly"]]
    .dropna(subset=["NDC11"])
    .drop_duplicates("NDC11")
    .reset_index(drop=True)
)
ndc_df.columns = ["ndc11", "dm_friendly"]

print(f"  NDCs to verify: {len(ndc_df)}")
print(f"  DRLS rows: {len(drls)} | unique DUNS: {len(DUNS_TO_ROWS)}")
print(f"  Redica FEIs: {len(REDICA_FEIS)}")


# =============================================================================
# 2. HELPERS
# =============================================================================
def _us_address(address: str) -> bool:
    """Heuristic: is this a US address?"""
    addr_upper = address.upper()
    return any(tok in addr_upper for tok in ("UNITED STATES", "(USA)", "(US)", ", US ", "U.S.A"))


def _select_fei(rows: list, redica_feis: set) -> tuple[str | None, str | None, str | None, bool]:
    """
    From a list of DRLS rows for a DUNS, pick the best FEI.
    Priority: (1) Redica + non-US, (2) Redica + US, (3) non-US, (4) first.
    Returns (fei, firm_name, address, in_redica).
    """
    if not rows:
        return None, None, None, False

    in_redica   = [r for r in rows if r["FEI_NUMBER"] in redica_feis]
    non_us      = [r for r in rows if not _us_address(r["ADDRESS"])]
    r_non_us    = [r for r in in_redica if not _us_address(r["ADDRESS"])]

    chosen = (r_non_us or in_redica or non_us or rows)[0]
    fei    = chosen["FEI_NUMBER"]
    return fei, chosen["FIRM_NAME"], chosen["ADDRESS"], fei in redica_feis


def get_spl_info(ndc_friendly: str) -> tuple[list[str], list[dict], str | None]:
    """
    Query DailyMed with a 'friendly' NDC string (e.g. '68180-336').
    Returns (duns_list, addr_list, setid).
    addr_list: list of dicts with keys postalCode, country (non-empty only).
    """
    url = DAILYMED_NDC_URL.format(ndc=ndc_friendly)
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return [], [], None
        items = r.json().get("data", [])
        if not items:
            return [], [], None
        setid = items[0]["setid"]
        time.sleep(SLEEP_SEC)
        spl_r = requests.get(DAILYMED_SPL_URL.format(setid=setid), headers=HEADERS, timeout=15)
        if spl_r.status_code != 200:
            return [], [], setid

        xml_root = etree.fromstring(spl_r.content)
        TAG = lambda t: f"{{{HL7_NS}}}{t}"

        # Extract all DUNS numbers
        duns_list = []
        for elem in xml_root.iter(TAG("id")):
            if elem.get("root") == DUNS_OID:
                d = (elem.get("extension") or "").strip()
                if d and d not in duns_list:
                    duns_list.append(d)

        # Extract addresses (postal code + country)
        addr_list = []
        for addr_elem in xml_root.iter(TAG("addr")):
            parts = {}
            for child in addr_elem:
                tag = child.tag.split("}")[-1]
                if tag in ("postalCode", "country") and (child.text or "").strip():
                    parts[tag] = child.text.strip()
            if parts.get("postalCode") and parts not in addr_list:
                addr_list.append(parts)

        return duns_list, addr_list, setid

    except Exception as e:
        print(f"    DailyMed error: {e}")
        return [], [], None


def zip_lookup(addr_list: list[dict]) -> tuple[str | None, str | None, str | None, bool]:
    """
    Try to match non-US postal codes from addr_list against DRLS ADDRESS strings.
    Returns (fei, firm_name, address, in_redica) for best match, or (None,...).
    """
    for entry in addr_list:
        zip_code = entry.get("postalCode", "").strip()
        country  = entry.get("country", "").strip().upper()
        if not zip_code or country in ("US", "USA", "840"):
            continue
        # Search DRLS ADDRESS for this zip code
        hits = drls[drls["ADDRESS"].str.contains(re.escape(zip_code), na=False)]
        if len(hits) == 1:
            row = hits.iloc[0]
            fei = row["FEI_NUMBER"]
            return fei, row["FIRM_NAME"], row["ADDRESS"], fei in REDICA_FEIS
        elif len(hits) > 1:
            # Multiple — prefer Redica
            r_hits = hits[hits["FEI_NUMBER"].isin(REDICA_FEIS)]
            pick = (r_hits if len(r_hits) else hits).iloc[0]
            fei = pick["FEI_NUMBER"]
            return fei, pick["FIRM_NAME"], pick["ADDRESS"], fei in REDICA_FEIS
    return None, None, None, False


# =============================================================================
# 3. MAIN LOOP
# =============================================================================
print(f"\nVerifying {len(ndc_df)} NDCs...\n")
results = []

for i, row in ndc_df.iterrows():
    ndc11      = str(row["ndc11"]).strip()
    dm_frndly  = str(row["dm_friendly"]).strip() if pd.notna(row["dm_friendly"]) else ""

    # Build display NDC from ndc11 (e.g. 68180033607 → 68180-0336-07)
    ndc_display = (
        f"{ndc11[:5]}-{ndc11[5:9]}-{ndc11[9:]}" if len(ndc11) == 11 else ndc11
    )

    print(f"[{i+1:3d}/112] {ndc_display}  (DM: {dm_frndly})", end="  →  ")

    fei_verified = None
    firm_name    = None
    address_note = None
    in_redica    = False
    method       = "not_found"
    all_duns     = []

    if dm_frndly:
        duns_list, addr_list, setid = get_spl_info(dm_frndly)
        all_duns = duns_list

        # --- Strategy A: DUNS → DRLS ---
        if duns_list:
            # Collect all DRLS rows for all DUNS found in this SPL
            all_rows = []
            for d in duns_list:
                all_rows.extend(DUNS_TO_ROWS.get(d, []))

            if all_rows:
                fei_verified, firm_name, address_note, in_redica = _select_fei(all_rows, REDICA_FEIS)
                method = "duns_drls"

        # --- Strategy B: zip code → DRLS ---
        if not fei_verified and addr_list:
            fei_verified, firm_name, address_note, in_redica = zip_lookup(addr_list)
            if fei_verified:
                method = "zip_drls"

        if not fei_verified and duns_list:
            method = "duns_no_drls_match"
        elif not fei_verified and not duns_list:
            method = "not_found"
    else:
        method = "no_dm_friendly"

    # Build Redica site label if available
    redica_label = REDICA_SITE.get(fei_verified, "") if fei_verified else ""

    note = f"{method}"
    if all_duns:
        note += f" | DUNS={','.join(all_duns)}"
    if fei_verified and not in_redica:
        note += " | FEI not in Redica"
    if fei_verified and in_redica:
        note += f" | Redica: {redica_label}"

    print(f"FEI={fei_verified}  [{method}]")

    results.append({
        "ndc11":         ndc11,
        "ndc_display":   ndc_display,
        "dm_friendly":   dm_frndly,
        "fei_verified":  fei_verified,
        "firm_name":     firm_name,
        "in_redica":     int(in_redica) if fei_verified else None,
        "note":          note,
    })

# =============================================================================
# 4. SAVE AND SUMMARISE
# =============================================================================
df = pd.DataFrame(results)
df.to_csv(OUT_FILE, index=False)

print(f"\n{'='*60}")
print(f"Saved: {OUT_FILE}")
print(f"\nMethod breakdown:")
print(df["method_clean"].value_counts().to_string() if "method_clean" in df.columns
      else df["note"].str.split(" | ").str[0].value_counts().to_string())
print(f"\nIn Redica: {df['in_redica'].sum()} / {len(df[df['fei_verified'].notna()])}")
print(f"Not found: {df['fei_verified'].isna().sum()} NDCs")
print(f"\nAll verified FEIs:")
print(df[df["fei_verified"].notna()][["ndc_display","fei_verified","firm_name","note"]].to_string(index=False))
# %%
