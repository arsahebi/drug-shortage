# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# -------------------- CONFIG --------------------
DATA_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/Data/12 - FDA - 483"
)

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Expect parser outputs here:
JSON_DIR = PROCESSED_DIR / "json"          # <-- produced by pdf->json step
OUT_DIR  = PROCESSED_DIR / "llm_clean"     # <-- cleaned .txt will go here

PATTERN = "*.json"
OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"

MAX_CHARS = 12000
OVERWRITE = False
DRY_RUN = False


PROMPT_PREFIX = """\
You are a meticulous cleaner for FDA Form 483 texts.

CLEANING GOAL:
Return ONLY the observation narrative (plain paragraphs), removing any page furniture, headers/footers,
tables, and duplicated boilerplate.

AGGRESSIVELY REMOVE LINES THAT LOOK LIKE:
- Repeating page furniture (e.g., "DEPARTMENT OF HEALTH AND HUMAN SERVICES", "FOOD AND DRUG ADMINISTRATION",
  "FORM FDA 483", "INSPECTIONAL OBSERVATIONS", "SEE REVERSE", "EMPLOYEE(S) SIGNATURE", "DATE ISSUED",
  "PAGE X OF Y PAGES").
- Table-like content: rows with many columns/separators, ASCII/box drawing characters, or heavy multi-space
  alignment (e.g., columns, timestamps, IDs, serials, “|”, “──”, “│”, “╬”).
- Pagination junk, investigator bars, “PREVIOUS EDITION OBSOLETE”, etc.

KEEP:
- Natural-language observation text (complete sentences/paragraphs).
- Bullets that are real sentences.
- CFR citations if inline with real sentences.

OUTPUT FORMAT:
- Plain cleaned text only. No preface, no explanation, no markdown, no quotes.
"""


# -------------------- HELPERS --------------------
def chunk_text(s: str, max_chars: int = 12000) -> List[str]:
    """Split into chunks under max_chars, preferring paragraph boundaries."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    paras = s.split("\n\n")
    chunks, cur = [], []
    cur_len = 0

    for p in paras:
        p2 = p.strip("\n")
        add_len = len(p2) + (2 if cur else 0)

        if cur_len + add_len > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0

        if len(p2) > max_chars:  # hard split a mega-paragraph
            for i in range(0, len(p2), max_chars):
                part = p2[i : i + max_chars]
                if cur:
                    chunks.append("\n\n".join(cur))
                    cur, cur_len = [], 0
                chunks.append(part)
            continue

        cur.append(p2)
        cur_len += add_len

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def post_filter_lines(text: str) -> str:
    """Extra safety pass to drop obvious furniture/table lines that might slip through."""
    out_lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out_lines.append(ln)
            continue

        U = s.upper()
        if any(k in U for k in [
            "DEPARTMENT OF HEALTH AND HUMAN SERVICES",
            "FOOD AND DRUG ADMINISTRATION",
            "FORM FDA 483",
            "INSPECTIONAL OBSERVATIONS",
            "SEE REVERSE",
            "EMPLOYEE(S) SIGNATURE",
            "DATE ISSUED",
            "PAGE ",
            "PREVIOUS EDITION OBSOLETE",
        ]):
            continue

        if s.count("|") >= 2 or s.count("│") >= 2 or "──" in s or "╬" in s:
            continue
        if s.count("   ") >= 3:  # lots of big gaps = columnar
            continue

        out_lines.append(ln)

    return "\n".join(out_lines).strip()


def source_from_json(json_path: Path, include_titles: bool = True) -> str:
    """
    Build a single source string from FDA-483 JSON:
      {"file":..., "header":..., "observations":[{"observation_number":1,"title":...,"text":...}, ...]}
    """
    data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    obs_list = data.get("observations") or []

    parts: List[str] = []
    for obs in obs_list:
        title = (obs.get("title") or f"OBSERVATION {obs.get('observation_number','')}").strip()
        body = (obs.get("text") or "").strip()
        if not body:
            continue
        if include_titles:
            parts.append(title)
        parts.append(body)

    # fallback
    if not parts:
        header = data.get("header") or {}
        p1 = (header.get("page1_raw") or "").strip()
        if p1:
            parts.append(p1)

    return "\n\n".join(parts).strip()


# -------------------- OLLAMA BACKEND ONLY --------------------
class OllamaBackend:
    def __init__(self, host: str = "http://localhost:11434"):
        try:
            import requests  # noqa: F401
        except Exception:
            sys.exit("Missing dependency: requests. Install with: pip install requests")
        self.host = host

    def generate(self, model: str, prompt: str, max_retries: int = 5) -> str:
        import requests

        url = f"{self.host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}

        delay = 1.5
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=300)
                r.raise_for_status()
                data = r.json()
                return (data.get("response") or "").strip()
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(delay)
                delay *= 1.6


def clean_text_with_llm(backend: OllamaBackend, model: str, raw_text: str, max_chars: int) -> str:
    chunks = chunk_text(raw_text, max_chars=max_chars)
    cleaned_parts: List[str] = []

    for ch in chunks:
        prompt = PROMPT_PREFIX + "\n\n---\nSOURCE CHUNK:\n" + ch
        cleaned = backend.generate(model, prompt)
        cleaned = post_filter_lines(cleaned)
        if cleaned:
            cleaned_parts.append(cleaned)

    return "\n\n".join(cleaned_parts).strip()


def clean_one_json_file(backend: OllamaBackend, model: str, in_path: Path, out_path: Path) -> None:
    raw = source_from_json(in_path, include_titles=True)
    if not raw:
        print(f"• Skipping (empty observations): {in_path.name}")
        return

    final = clean_text_with_llm(backend, model, raw, max_chars=MAX_CHARS)
    if DRY_RUN:
        print(f"[DRY RUN] {in_path.name} -> would write {len(final)} chars")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final + "\n", encoding="utf-8")
    print(f"✓ Cleaned: {in_path.name} -> {out_path}")


# -------------------- RUN (no main) --------------------
if not JSON_DIR.exists():
    sys.exit(f"JSON_DIR not found (run pdf->json first): {JSON_DIR}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

backend = OllamaBackend(host=OLLAMA_HOST)
files = sorted(JSON_DIR.rglob(PATTERN))
if not files:
    sys.exit(f"No JSON files found under {JSON_DIR} matching {PATTERN}")

print(f"Backend: ollama | Host: {OLLAMA_HOST} | Model: {MODEL}")
print(f"Input JSON:  {JSON_DIR}")
print(f"Output TXT:  {OUT_DIR}")
print(f"Files: {len(files)} | Overwrite={OVERWRITE} | DryRun={DRY_RUN}")

for p in files:
    out_path = OUT_DIR / (p.stem + ".clean.txt")
    if out_path.exists() and not OVERWRITE:
        print(f"• Skip existing: {out_path.name}")
        continue
    try:
        clean_one_json_file(backend, MODEL, p, out_path)
    except Exception as e:
        print(f"✗ Failed: {p.name} -> {e}")

# %%
