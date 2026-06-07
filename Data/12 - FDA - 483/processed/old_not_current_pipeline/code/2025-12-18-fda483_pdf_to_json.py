# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FDA 483 PDF → OCR → text extraction → JSON parsing pipeline.

Inputs are read from:
  <data_dir>/raw/*.pdf

Outputs are written to:
  <data_dir>/processed/
    ocr/        (OCR PDFs)
    text/       (pdftotext dumps)
    json/       (one JSON per PDF)
    summary.csv (one row per PDF)
    debug/*     (optional, when DEBUG=True)
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_DATA_DIR = (
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/12 - FDA - 483"
)

# When False, OCRmyPDF will skip pages that already contain text (faster).
FORCE_OCR = False


# =============================================================================
# UTILS
# =============================================================================

def sh(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sanitize_name(p: Path) -> Path:
    safe = re.sub(r"[^\w\s\-\.,@&()]+", "", p.name)
    safe = re.sub(r"\s+", " ", safe).strip()
    return p.with_name(safe)


class DebugSink:
    def __init__(
        self,
        base_dir: Path,
        stem: str,
        enabled: bool = False,
        trace: bool = False,
        token: Optional[str] = None,
    ):
        self.enabled = enabled
        self.trace = trace
        self.token = token
        # CHANGED: debug now lives under processed/
        self.root = base_dir / "processed" / "debug" / stem
        if self.enabled:
            ensure_dir(self.root)
            (self.root / "debug.log").write_text("", encoding="utf-8")

    def log(self, msg: str) -> None:
        if self.trace:
            print(f"[DEBUG] {msg}")
        if self.enabled:
            with (self.root / "debug.log").open("a", encoding="utf-8") as f:
                f.write(msg.rstrip() + "\n")

    def save_text(self, relpath: str, text: str) -> None:
        if not self.enabled:
            return
        out = self.root / relpath
        ensure_dir(out.parent)
        out.write_text(text, encoding="utf-8", errors="ignore")

    def mark_token(self, where: str, text: str) -> None:
        if not (self.enabled and self.token):
            return
        if self.token in text:
            self.log(f"[token-hit] {where}: token='{self.token}'")


# =============================================================================
# STEP 1: PREPARE
# =============================================================================

def step_prepare(data_dir: Path) -> None:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    for sub in (raw_dir, processed_dir / "ocr", processed_dir / "text", processed_dir / "json"):
        ensure_dir(sub)

    # If PDFs were mistakenly dropped in the module root, copy them into raw/
    for src in sorted(data_dir.glob("*.pdf")):
        dst = sanitize_name(src)
        dst = raw_dir / dst.name
        if not dst.exists():
            shutil.copy2(src, dst)

    print("Step 1 (prepare): OK")


# =============================================================================
# STEP 2: OCR + TEXT DUMP
# =============================================================================

def ocr_one(src_pdf: Path, dst_pdf: Path, force_ocr: bool = False) -> None:
    """Run OCRmyPDF to create a searchable PDF.

    If force_ocr=False, prefer '--skip-text' (faster + preserves existing text).
    """
    args = [
        "ocrmypdf",
        "--rotate-pages",
        "--deskew",
        "--optimize", "3",
        "--output-type", "pdf",
    ]
    args.insert(1, "--force-ocr" if force_ocr else "--skip-text")
    sh(args + [str(src_pdf), str(dst_pdf)])

def pdftotext_dump(pdf: Path, txt: Path) -> None:
    sh(["pdftotext", "-layout", "-nopgbrk", str(pdf), str(txt)])

def step_ocr_text(data_dir: Path) -> None:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    ocr_dir = processed_dir / "ocr"
    text_dir = processed_dir / "text"
    ensure_dir(processed_dir)
    ensure_dir(ocr_dir)
    ensure_dir(text_dir)

    for pdf in sorted(raw_dir.glob("*.pdf")):
        ocr_pdf = ocr_dir / pdf.name
        txt_out = text_dir / (pdf.stem + ".txt")

        if not ocr_pdf.exists():
            print(f"OCR: {pdf.name}")
            ocr_one(pdf, ocr_pdf, force_ocr=FORCE_OCR)
        else:
            print(f"OCR: {pdf.name} (skip; exists)")

        if not txt_out.exists():
            print(f"TXT: {ocr_pdf.name}")
            pdftotext_dump(ocr_pdf, txt_out)
        else:
            print(f"TXT: {ocr_pdf.name} (skip; exists)")

    print("Step 2 (ocr_text): OK")


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

REDACTION_RE = re.compile(
    r"""
    \(\s*b\s*\)\s*\(\s*4\s*\)      # (b)(4) variants
    |b\)\s*\(\s*4\s*\)
    |\(\s*b\s*\)\s*4
    |\[\s*b\s*\]\s*\(?\s*4\s*\)?
    """,
    re.I | re.VERBOSE,
)

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = REDACTION_RE.sub("[REDACTED]", s)
    s = re.sub(r"\u200b|\u200e|\ufb01|\ufb02", "", s)
    s = re.sub(r"[ \t]{2,}", "  ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def pdftotext_pages(pdf: Path, dbg: Optional[DebugSink] = None) -> List[str]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "dump.txt"
        sh(["pdftotext", "-layout", str(pdf), str(tmp)])
        s = tmp.read_text(encoding="utf-8", errors="ignore")

    pages = s.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    pages = [normalize_text(p) for p in pages]

    if dbg:
        for i, pg in enumerate(pages, 1):
            dbg.save_text(f"00_pages_raw/page-{i:03d}.txt", pg)
            dbg.mark_token(f"raw page {i}", pg)

    return pages

def read_pdf_page1_text(pdf: Path) -> str:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "p1.txt"
        sh(["pdftotext", "-f", "1", "-l", "1", "-layout", str(pdf), str(tmp)])
        return tmp.read_text(encoding="utf-8", errors="ignore")

def parse_header_from_page1(page1_text: str) -> Dict[str, Any]:
    header: Dict[str, Any] = {}
    t = normalize_text(page1_text)

    m = re.search(r"\bFEI\b[^\d]*(\d{7,10})", t, re.I)
    if m:
        header["fei_number"] = m.group(1)

    m = re.search(r"\bFirm Name\b[:\s]*(.+)", t, re.I)
    if m:
        header["firm_name"] = m.group(1).strip()

    m = re.search(r"\bDate of Inspection\b[:\s]*(.+)", t, re.I)
    if m:
        header["date_of_inspection"] = m.group(1).strip()

    return header


# =============================================================================
# HEADER/FOOTER EDGE DROP + OBSERVATION SPLIT LOGIC
# (copied from your pipeline; unchanged except file output paths handled elsewhere)
# =============================================================================

EDGE_RE = re.compile(r"^\s*(?:Page\s+\d+|\d+)\s*$", re.I)
OBS_HDR_RE = re.compile(r"^\s*OBSERVATION\s+(\d+)\s*$", re.I | re.M)

def _line_is_edgey(line: str) -> bool:
    if not line.strip():
        return True
    if EDGE_RE.match(line.strip()):
        return True
    return False

def drop_repeating_edges_per_page(
    pages: List[str],
    dbg: Optional[DebugSink] = None,
    auto: bool = True,
) -> List[str]:
    """
    Attempts to remove repeating header/footer furniture.
    This is intentionally conservative; it only drops lines that look like edges.
    """
    if not auto:
        return pages

    stripped_pages: List[str] = []
    for i, pg in enumerate(pages, 1):
        lines = pg.splitlines()

        # Drop leading edge-like lines
        top = 0
        while top < len(lines) and _line_is_edgey(lines[top]):
            top += 1

        # Drop trailing edge-like lines
        bot = len(lines)
        while bot > top and _line_is_edgey(lines[bot - 1]):
            bot -= 1

        new_pg = "\n".join(lines[top:bot]).strip() + "\n"
        stripped_pages.append(new_pg)

        if dbg:
            dbg.save_text(f"01_pages_edge_dropped/page-{i:03d}.txt", new_pg)
            dbg.mark_token(f"edge-dropped page {i}", new_pg)

    return stripped_pages

def split_observations(stitched: str) -> List[Tuple[int, str]]:
    """
    Split the stitched text into (observation_number, observation_text).
    """
    matches = list(OBS_HDR_RE.finditer(stitched))
    if not matches:
        return []

    chunks: List[Tuple[int, str]] = []
    for idx, m in enumerate(matches):
        obs_num = int(m.group(1))
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(stitched)
        chunk = stitched[start:end].strip()
        chunks.append((obs_num, chunk))
    return chunks

TABULAR_BLOCK_RE = re.compile(r"^\s*\|?.+\|.+\|", re.M)

def drop_tabular_blocks(text: str) -> str:
    # Basic drop of obvious pipe tables
    return TABULAR_BLOCK_RE.sub("", text)

LEFTOVER_RE = re.compile(r"\b(?:continued|cont\.)\b", re.I)

def scrub_leftover_artifacts(text: str) -> str:
    # Very light scrub; keep conservative
    return LEFTOVER_RE.sub("", text)


# =============================================================================
# CORE PARSER
# =============================================================================

def parse_pdf_to_json_dict(
    pdf: Path,
    ocr_dir: Path,
    dbg: Optional[DebugSink] = None,
    do_scrub_leftovers: bool = True,
    do_drop_tables: bool = True,
    do_auto_edge: bool = True,
) -> Dict[str, Any]:
    """
    Parse one PDF to a JSON-able dict with:
      - header (FEI, firm name, date)
      - observations: list of {observation_number, text}
    """
    # Use OCR PDF if it exists
    ocr_pdf = ocr_dir / pdf.name
    use_pdf = ocr_pdf if ocr_pdf.exists() else pdf
    if dbg:
        dbg.log(f"[use_pdf] {use_pdf.name}")

    # Header from page 1
    page1 = read_pdf_page1_text(use_pdf)
    header = parse_header_from_page1(page1)
    if dbg:
        dbg.save_text("02_header/page1.txt", normalize_text(page1))
        dbg.save_text("02_header/header.json", json.dumps(header, ensure_ascii=False, indent=2))

    # Full pages
    pages = pdftotext_pages(use_pdf, dbg=dbg)

    # Drop repeating edges (conservative)
    pages2 = drop_repeating_edges_per_page(pages, dbg=dbg, auto=do_auto_edge)

    # Stitch pages
    stitched = "\n".join(p.strip() for p in pages2 if p.strip()).strip() + "\n"
    if dbg:
        dbg.save_text("03_stitched/stitched_raw.txt", stitched)
        dbg.mark_token("stitched raw", stitched)

    # Optional scrubs
    if do_drop_tables:
        stitched = drop_tabular_blocks(stitched)
        if dbg:
            dbg.save_text("04_clean/stitched_no_tables.txt", stitched)

    if do_scrub_leftovers:
        stitched = scrub_leftover_artifacts(stitched)
        if dbg:
            dbg.save_text("04_clean/stitched_scrubbed.txt", stitched)

    # Split observations
    obs_chunks = split_observations(stitched)
    observations = []
    for num, txt in obs_chunks:
        observations.append({"observation_number": num, "text": txt})

    if dbg:
        for o in observations:
            dbg.save_text(f"05_observations/obs-{o['observation_number']:02d}.txt", o["text"])

    return {"file": pdf.name, "header": header, "observations": observations}


# =============================================================================
# STEP 3: PARSE JSON + SUMMARY
# =============================================================================

def step_parse_json(
    data_dir: Path,
    debug: bool = False,
    trace: bool = False,
    token: Optional[str] = None,
    no_scrub_leftovers: bool = False,
    no_drop_tables: bool = False,
    no_auto_edge: bool = False,
) -> None:
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    ocr_dir = processed_dir / "ocr"
    out_json = processed_dir / "json"
    ensure_dir(processed_dir)
    ensure_dir(out_json)

    summary_rows = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        stem = pdf.stem
        dbg = DebugSink(base_dir=data_dir, stem=stem, enabled=debug, trace=trace, token=token)
        dbg.log(f"[file] {pdf.name}")

        data = parse_pdf_to_json_dict(
            pdf,
            ocr_dir,
            dbg=dbg,
            do_scrub_leftovers=not no_scrub_leftovers,
            do_drop_tables=not no_drop_tables,
            do_auto_edge=not no_auto_edge,
        )

        (out_json / (pdf.stem + ".json")).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        header = data.get("header", {})
        summary_rows.append(
            {
                "file": pdf.name,
                "fei_number": header.get("fei_number"),
                "firm_name": header.get("firm_name"),
                "date_of_inspection": header.get("date_of_inspection"),
                "num_observations": len(data.get("observations", [])),
            }
        )

    sum_path = data_dir / "processed" / "summary.csv"
    ensure_dir(sum_path.parent)
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["file", "fei_number", "firm_name", "date_of_inspection", "num_observations"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Step 3 (parse_json): OK -> {sum_path}")


# =============================================================================
# RUN (no main)
# =============================================================================
# Expectation:
#   - Inputs:  <DATA_DIR>/raw/*.pdf
#   - Outputs: <DATA_DIR>/processed/{ocr,text,json,summary.csv,debug/*}
#
# Notes:
#   - Set STEPS to any combo of "1", "2", "3".
#   - Set FORCE_OCR=True if you want OCR even when the PDF already contains text.

DATA_DIR = Path(DEFAULT_DATA_DIR)

STEPS = "123"          # "1" prepare, "2" ocr+text, "3" parse json
DEBUG = False          # True -> writes processed/debug/<pdf-stem>/*
TRACE = False
TOKEN = None           # e.g. "188B" or None

NO_SCRUB_LEFTOVERS = False
NO_DROP_TABLES = False
NO_AUTO_EDGE = False

if "1" in STEPS:
    step_prepare(DATA_DIR)
if "2" in STEPS:
    step_ocr_text(DATA_DIR)
if "3" in STEPS:
    step_parse_json(
        DATA_DIR,
        debug=DEBUG,
        trace=TRACE,
        token=TOKEN,
        no_scrub_leftovers=NO_SCRUB_LEFTOVERS,
        no_drop_tables=NO_DROP_TABLES,
        no_auto_edge=NO_AUTO_EDGE,
    )

# %%
