# %%
"""
Build the FDA Form 483 processed tables used by the text-analysis pipeline.

Run this after adding PDF files to:
    Data/12 - FDA - 483/raw/

It writes three files one level above this code folder:
    483_pdf_inventory.csv   one row per PDF
    483_observations.csv    one row per extracted observation
    483_fei_features.csv    one row per FEI, aggregated from the observations

The feature logic is intentionally concentrated near the top of the file so it
is easy to change after reading the PDFs.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import tempfile

import pandas as pd
import pytesseract
from pdf2image import convert_from_path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_483 = Path(__file__).resolve().parents[2]
RAW_DIR = DATA_483 / "raw"
OUT_DIR = DATA_483 / "processed"


# ---------------------------------------------------------------------------
# Editable feature definitions
# ---------------------------------------------------------------------------
MIN_EXTRACTABLE_CHARS = 800

SIGNAL_PATTERNS = {
    "repeat": (
        r"\b(?:repeat(?:ed)?|r\s+e\s*p\s*e\s*a\s*t)\s+"
        r"(?:observation|observations|violation|violations|finding|findings|cite|citation|citations)\b"
    ),
    "systemic": r"\b(frequently|routinely|consistently|numerous|systemic|widespread|multiple)\b",
    "wl_ref": r"warning\s+letter|consent\s+decree|import\s+alert",
    "data_integrity": r"data\s+integrit|ALCOA|audit\s+trail|original\s+record|metadata",
    "contamination": r"contamina|microbial|particulate|endotoxin|bioburden|sterility\s+failure",
    "oos_oot": r"\bOOS\b|\bOOT\b|out[-\s]of[-\s]specification|out[-\s]of[-\s]trend",
    "patient_risk": r"patient\s+(risk|harm|impact|safety)|adverse\s+event|recall|released.*market|US market",
    "quality_unit": r"quality\s+(unit|control|assurance)|\bQU\b|\bQA\b",
    "investigation": r"investigat(e|ed|ion|ions)|root\s+cause|corrective\s+action|\bCAPA\b",
    "documentation": r"document|record|logbook|SOP|procedure|written\s+procedure",
    "laboratory": r"laborator|method|assay|chromatograph|specification|sample",
    "equipment_facility": r"equipment|facility|building|HEPA|HVAC|maintenance|cleaning",
    "process_control": r"in[-\s]?process|process\s+validation|batch\s+(record|production)|manufacturing",
}

COMPILED_SIGNALS = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in SIGNAL_PATTERNS.items()
}

CFR_RE = re.compile(
    r"21\s*C\.?\s*F\.?\s*R\.?\s*(?:§|Part)?\s*(\d{3}(?:\.\d+)?(?:\([^)]+\))*)",
    re.IGNORECASE,
)
OBSERVATION_RE = re.compile(
    r"(?im)^\s*(?:(?:OBSERVAT|OB[\xb7··]SERVAT)\s*(?:ION|lON|I0N|0N|\(ON)|SERVAT(?:ION|lON|I0N|0N))\s+(\d{1,2})\b.*$"
)
NUMBERED_OBSERVATION_RE = re.compile(r"\n\s*(\d{1,2})\.\s+(?=[A-Z(])")
LETTERED_EXAMPLE_RE = re.compile(r"(?m)^\s*[A-Z][.)]\s+")

PAGE_FURNITURE_PATTERNS = [
    r"DEPARTMENT\s+OF\s+HEALTH\s+AND\s+HUMAN\s+SERVICES",
    r"FOOD\s+AND\s+DRUG\s+ADMIN",
    r"FORM\s+F[D0]A\s*483",
    r"INSPECTIONAL\s+OBSERVATIONS",
    r"PREVIOUS\s+EDITION\s+OBSOLETE",
    r"SEE\s+REVERSE",
    r"OF\s+THIS\s+PAGE",
    r"EMPLOYEE\(?S?\)?.{0,30}SIGNATURE",
    r"DATE\s+ISSUED|OATE\s+ISSUEO|DATE\s+lSSUEO",
    r"DISTRICT\s+ADDRESS",
    r"NAME\s+AND\s+TITLE\s+OF",
    r"FIRM\s+NAME\s+STREET",
    r"CITY.*STATE.*ZIP.*TYPE\s+ESTABLISHMENT",
    r"PAGE\s+\d+\s+OF\s+\d+",
    r"www\.fda\.gov|fda\.\s*gov",
    r"\bFax\s*:",
    r"\bFEI\s*NUMBER\b|\bFEJ\s*NUMBER\b|\bFEINUMBER\b",
]

COMPILED_PAGE_FURNITURE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in PAGE_FURNITURE_PATTERNS
]


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def make_date(year: int, month: int, day: int) -> datetime | None:
    """Return a valid date, interpreting two-digit years as 2000s."""
    try:
        year = 2000 + year if year < 100 else year
        return datetime(year, month, day)
    except ValueError:
        return None


def parse_mmddyyyy(value: str) -> datetime | None:
    """Parse MM/DD/YYYY, MM-DD-YY, MM.DD.YY, and similar strings."""
    m = re.match(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$", value)
    if not m:
        return None
    return make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))


def parse_date_from_filename(name: str) -> datetime | None:
    """Pull the best inspection/end date from the PDF filename."""
    # Example: --November-7-16--2016. If it is a range, use the ending day.
    m = re.search(r"--([A-Za-z]+)-(\d{1,2})(?:-(\d{1,2}))?--(\d{4})", name)
    if m:
        month = MONTHS.get(m.group(1)[:3].lower())
        day = int(m.group(3) or m.group(2))
        if month:
            date = make_date(int(m.group(4)), month, day)
            if date:
                return date

    # Example: 10-19 through 27-2023 or 5-1 thru 12-23.
    m = re.search(
        r"(\d{1,2})-\d{1,2}\s+(?:thru|through)\s+(\d{1,2})-(\d{2,4})",
        name,
        re.IGNORECASE,
    )
    if m:
        date = make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if date:
            return date

    # Example: 4-26 thru 5-9-22. Use the end date.
    m = re.search(r"\d{1,2}-\d{1,2}\s+(?:thru|through)\s+(\d{1,2})-(\d{1,2})-(\d{2,4})", name, re.IGNORECASE)
    if m:
        date = make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if date:
            return date

    # Example: 6.2-6.13.25. Use the second date.
    m = re.search(r"\d{1,2}\.\d{1,2}-(\d{1,2})\.(\d{1,2})\.(\d{2,4})", name)
    if m:
        date = make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if date:
            return date

    # Example: Dated-06-08-16.
    m = re.search(r"dated[-_](\d{1,2})[-_](\d{1,2})[-_](\d{2,4})", name, re.IGNORECASE)
    if m:
        date = make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if date:
            return date

    # Example: -02082019- or -012519-.
    for pattern, year_slice in [
        (r"[-_](\d{8})(?:[-_]|$)", slice(4, 8)),
        (r"[-_](\d{6})(?:[-_]|$)", slice(4, 6)),
    ]:
        m = re.search(pattern, name)
        if m:
            s = m.group(1)
            date = make_date(int(s[year_slice]), int(s[0:2]), int(s[2:4]))
            if date:
                return date

    # Single date anywhere in the name: 8.12.22, 06-17-2011, 2022-08-12.
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        return make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(r"(\d{1,2})[.\-](\d{1,2})[.\-](\d{2,4})", name)
    if m:
        return make_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))

    return None


def parse_date_from_pdf_header(text: str) -> datetime | None:
    """Use DATE(S) OF INSPECTION from the first page when it is readable."""
    header = text[:1500]
    m = re.search(
        r"DATE\s*\(?S?\)?\s*OF\s+INSPECTION.{0,200}?(\d{1,2}/\d{1,2}/\d{2,4})"
        r"(?:\s*[-\u2013]\s*(\d{1,2}/\d{1,2}/\d{2,4}))?",
        header,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    # If a range is present, use the end date. That is usually the 483 issue date.
    return parse_mmddyyyy(m.group(2) or m.group(1))


# ---------------------------------------------------------------------------
# PDF and observation parsing
# ---------------------------------------------------------------------------
def parse_fei_from_filename(name: str) -> int | None:
    m = re.match(r"^(\d{6,12})", name)
    return int(m.group(1)) if m else None


def read_pdf_text(pdf_path: Path) -> str:
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as exc:
        print(f"  Could not render {pdf_path.name}: {exc}")
        return ""
    pages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            img_path = Path(tmpdir) / f"page_{i}.png"
            img.save(img_path, "PNG")
            try:
                pages.append(pytesseract.image_to_string(str(img_path)))
            except Exception as exc:
                print(f"  OCR failed on page {i+1} of {pdf_path.name}: {exc}")
                pages.append("")
    return "\n".join(pages)


def parse_firm_from_text(text: str) -> str:
    m = re.search(r"FIRM\s+NAME[^\n]*\n([^\n]+)", text[:1800], re.IGNORECASE)
    return m.group(1).strip()[:120] if m else ""


def clean_observation_text(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_page_furniture_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip()
    if not normalized:
        return False
    if any(pattern.search(normalized) for pattern in COMPILED_PAGE_FURNITURE):
        return True
    if "investigator" in normalized.lower() and len(normalized) < 140:
        return True
    return False


def remove_page_furniture(obs_text: str) -> str:
    """Drop FDA 483 page headers/footers while preserving observation text."""
    kept_lines = [
        line
        for line in obs_text.splitlines()
        if not is_page_furniture_line(line)
    ]
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_on_observation_headers(text: str) -> list[tuple[int, str]]:
    matches = list(OBSERVATION_RE.finditer(text))
    observations: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        obs_num = int(match.group(1))
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        obs_text = clean_observation_text(text[start:end])
        if 1 <= obs_num <= 30 and len(obs_text) > 50:
            observations.append((obs_num, obs_text))

    return observations


def split_on_numbered_lines(text: str) -> list[tuple[int, str]]:
    """Fallback for older PDFs that do not contain clear OBSERVATION headings."""
    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*1\.\s+[A-Z(]", line):
            body_start = i
            break

    body = "\n" + "\n".join(lines[body_start:])
    parts = NUMBERED_OBSERVATION_RE.split(body)
    observations: list[tuple[int, str]] = []

    i = 1
    while i < len(parts) - 1:
        obs_num = int(parts[i])
        obs_text = clean_observation_text(parts[i + 1])
        if 1 <= obs_num <= 30 and len(obs_text) > 50:
            observations.append((obs_num, obs_text))
        i += 2

    return observations


def split_observations(text: str) -> tuple[list[tuple[int, str]], str]:
    observations = split_on_observation_headers(text)
    if observations:
        return observations, "observation_header"
    return split_on_numbered_lines(text), "numbered_line_fallback"


def split_header_body(obs_text: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", obs_text).strip()
    cleaned = re.sub(r"^OBSERVAT\S*\s+\d+\s*", "", cleaned, flags=re.IGNORECASE)
    first_period = cleaned.find(".")
    if first_period == -1:
        return cleaned, ""
    return cleaned[: first_period + 1], cleaned[first_period + 1 :].strip()


def normalize_cfr(raw_cfr: str) -> str:
    return re.sub(r"\s+", "", raw_cfr)


def extract_cfrs(text: str) -> list[str]:
    cfrs = {f"21 CFR {normalize_cfr(match)}" for match in CFR_RE.findall(text)}
    return sorted(cfrs)


def count_lettered_examples(obs_text: str) -> int:
    """Count A./B./C. style subparts; use 1 when no subparts are present."""
    n_lettered_parts = len(LETTERED_EXAMPLE_RE.findall(obs_text))
    return max(n_lettered_parts, 1)


def analyze_observation(obs_text_raw: str) -> dict:
    obs_text_clean = remove_page_furniture(obs_text_raw)
    cfrs = extract_cfrs(obs_text_clean)
    header, body = split_header_body(obs_text_clean)

    row = {
        "obs_header": header,
        "obs_body": body,
        "obs_text": obs_text_clean,
        "obs_text_raw": obs_text_raw,
        "obs_text_clean": obs_text_clean,
        "cfr_codes": "; ".join(cfrs),
        "n_cfrs": len(cfrs),
        "obs_total_chars": len(obs_text_clean),
        "obs_raw_chars": len(obs_text_raw),
        "obs_clean_chars": len(obs_text_clean),
        "obs_body_chars": len(body),
        "n_examples": count_lettered_examples(obs_text_clean),
    }

    for signal, pattern in COMPILED_SIGNALS.items():
        row[f"has_{signal}"] = bool(pattern.search(obs_text_clean))

    return row


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def build_pdf_inventory_and_observations(pdf_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_rows = []
    observation_rows = []

    for pdf_path in pdf_files:
        name = pdf_path.stem
        fei = parse_fei_from_filename(name)
        text = read_pdf_text(pdf_path)
        n_chars = len(text)
        is_extractable = n_chars > MIN_EXTRACTABLE_CHARS

        header_date = parse_date_from_pdf_header(text) if is_extractable else None
        filename_date = parse_date_from_filename(name)
        inspection_date = header_date or filename_date
        firm_name = parse_firm_from_text(text) if is_extractable else ""

        observations: list[tuple[int, str]] = []
        split_method = ""
        if is_extractable:
            observations, split_method = split_observations(text)

        inventory_rows.append(
            {
                "fei": fei,
                "filename": name,
                "n_chars": n_chars,
                "is_extractable": is_extractable,
                "insp_date": inspection_date.strftime("%Y-%m-%d") if inspection_date else None,
                "firm_name": firm_name,
                "n_observations": len(observations),
                "observation_split_method": split_method,
            }
        )

        if is_extractable:
            print(
                f"  FEI {fei} | {inspection_date:%Y-%m-%d} | {n_chars:,} chars | "
                f"{len(observations)} obs | {split_method}"
                if inspection_date
                else f"  FEI {fei} | no date | {n_chars:,} chars | {len(observations)} obs | {split_method}"
            )
        else:
            print(f"  FEI {fei} | {n_chars} chars | scanned/unreadable | {name[:70]}")

        for obs_num, obs_text in observations:
            observation_rows.append(
                {
                    "fei": fei,
                    "filename": name,
                    "insp_date": inspection_date.strftime("%Y-%m-%d") if inspection_date else None,
                    "obs_num": obs_num,
                    **analyze_observation(obs_text),
                }
            )

    return pd.DataFrame(inventory_rows), pd.DataFrame(observation_rows)


def aggregate_fei_features(inventory: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    signal_cols = [c for c in observations.columns if c.startswith("has_")]
    rows = []

    for fei, inv_group in inventory.groupby("fei", dropna=False):
        obs_group = observations[observations["fei"] == fei]
        n_extractable = int(inv_group["is_extractable"].sum())
        n_obs = len(obs_group)

        dates = pd.to_datetime(inv_group["insp_date"], errors="coerce").dropna()
        row = {
            "fei": fei,
            "n_483s_total": len(inv_group),
            "n_483s_extractable": n_extractable,
            "n_observations_total": n_obs,
            "avg_obs_per_483": round(n_obs / max(n_extractable, 1), 2),
            "latest_483_date": dates.max().strftime("%Y-%m-%d") if len(dates) else None,
            "earliest_483_date": dates.min().strftime("%Y-%m-%d") if len(dates) else None,
            "avg_obs_total_chars": round(obs_group["obs_total_chars"].mean(), 0) if n_obs else 0,
            "avg_obs_body_chars": round(obs_group["obs_body_chars"].mean(), 0) if n_obs else 0,
            "avg_n_examples": round(obs_group["n_examples"].mean(), 2) if n_obs else 0,
        }

        for col in signal_cols:
            suffix = col.removeprefix("has_")
            row[f"n_obs_{suffix}_total"] = int(obs_group[col].sum()) if n_obs else 0
            row[f"ever_{suffix}"] = bool(obs_group[col].any()) if n_obs else False

        all_cfrs: set[str] = set()
        for cfr_string in obs_group["cfr_codes"].dropna():
            all_cfrs.update(c.strip() for c in str(cfr_string).split(";") if c.strip())
        row["n_unique_cfrs_in_483"] = len(all_cfrs)
        row["cfr_codes_in_483"] = "; ".join(sorted(all_cfrs))

        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(inventory: pd.DataFrame, observations: pd.DataFrame, fei_features: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT_DIR / "483_pdf_inventory.csv", index=False)
    observations.to_csv(OUT_DIR / "483_observations.csv", index=False)
    fei_features.to_csv(OUT_DIR / "483_fei_features.csv", index=False)


def print_summary(inventory: pd.DataFrame, observations: pd.DataFrame, fei_features: pd.DataFrame) -> None:
    print("\nSummary")
    print(f"  PDFs scanned:          {len(inventory)}")
    print(f"  Extractable PDFs:      {int(inventory['is_extractable'].sum())}")
    print(f"  Observations extracted:{len(observations):>7}")
    print(f"  FEIs with PDFs:        {inventory['fei'].nunique()}")

    if observations.empty:
        return

    signal_cols = [c for c in observations.columns if c.startswith("has_")]
    print("\nObservation signal prevalence")
    for col in signal_cols:
        n = int(observations[col].sum())
        pct = n / len(observations) * 100
        print(f"  {col:<28} {n:>4} obs ({pct:>5.1f}%)")

    preview_cols = [
        "fei",
        "n_483s_total",
        "n_483s_extractable",
        "n_observations_total",
        "ever_repeat",
        "ever_systemic",
        "ever_data_integrity",
        "ever_contamination",
    ]
    existing_preview_cols = [c for c in preview_cols if c in fei_features.columns]
    print("\nFEI feature preview")
    print(fei_features[existing_preview_cols].head(12).to_string(index=False))

# %%
def main() -> None:
    print("=" * 70)
    print("FDA Form 483 extraction")
    print("=" * 70)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Raw PDFs: {len(pdf_files)}\n")

    inventory, observations = build_pdf_inventory_and_observations(pdf_files)
    fei_features = aggregate_fei_features(inventory, observations)
    save_outputs(inventory, observations, fei_features)
    print_summary(inventory, observations, fei_features)

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
