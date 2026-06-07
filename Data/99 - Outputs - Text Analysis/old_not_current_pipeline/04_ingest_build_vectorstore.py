"""
04_ingest_build_vectorstore.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  LangChain ingestion layer: loads 483 PDF text + Warning Letter text for the
  129 reference FEIs, splits into individual observation/violation chunks,
  embeds them with a local sentence-transformer model, and stores them in a
  persistent ChromaDB vector store on disk.

  This is step 1 of the LLM pipeline (04 → 05 → 06 → 07).
  Must run before 05_extract_signals_langgraph.py.

WHEN TO RUN
  Run once before 05. Re-run when new 483 PDFs are added to Data/12 - FDA - 483/raw/.
  Safe to re-run — already-ingested chunks are skipped (idempotent).
  Takes ~5–15 minutes depending on PDF count. Does NOT require an API key.

REQUIRED FOR COMBINED DATASET?
  NO — optional LLM pipeline. Outputs feed 05→06→07, which optionally enrich
  the dashboard. The core combined dataset (01→03) does not need this.

INPUTS
  Data/12 - FDA - 483/raw/                              ← raw 483 PDF files
  Data/12 - FDA - 483/processed/483_pdf_inventory.csv   ← which PDFs are extractable
  Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv
  Data/08 - Valisure/raw/FEIs_March 2026.xlsx           ← 129 reference FEIs

OUTPUTS (in this folder)
  chroma_text_store/   ← persisted ChromaDB vector store (fda_observations collection)
  ingest_manifest.csv  ← one row per chunk: fei, doc_id, observation_id, status

IDEMPOTENCY
  Chunks are keyed by (doc_id + observation_id). Already-stored chunks are skipped
  on re-run. Crash-safe — partial progress is preserved in ChromaDB.

IMPORTANT — RECOMMENDED FUTURE UPGRADE (see README.md for full proposal)
  This script currently re-parses raw PDFs to extract observation text, duplicating
  work already done by Data/12 - FDA - 483/processed/code/20260316_483_comprehensive_extraction.py.

  Data/12 - FDA - 483/processed/483_observations.csv already contains:
    - 347 pre-parsed, cleaned observations (obs_text_clean column)
    - one row per observation with fei, filename, obs_num as stable IDs
    - all 13 regex flag columns (has_repeat, has_systemic, etc.)

  Switching 04 to read from 483_observations.csv instead of raw PDFs would:
    1. Eliminate duplicate PDF parsing (faster, more consistent text)
    2. Guarantee the same observation_id is used in both regex and LLM layers
    3. Allow direct join between regex flags and LLM signals on the same row
    4. Simplify this script substantially (~100 lines shorter)
  The Warning Letter ingestion section can remain unchanged.

DEPENDENCIES
  pip install chromadb>=0.5 sentence-transformers>=3.0 pdfplumber pandas openpyxl
  Optional: pytesseract + pillow (OCR fallback for scanned PDFs)
"""

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
OUT  = Path(__file__).parent

PDF_DIR     = BASE / "Data/12 - FDA - 483/raw"
INV_483     = BASE / "Data/12 - FDA - 483/processed/483_pdf_inventory.csv"
WL_REC_CSV  = BASE / "Data/21 - FDA - Warning Letter/processed/warning_letter_records.csv"
VALISURE    = BASE / "Data/08 - Valisure/raw/FEIs_March 2026.xlsx"
CHROMA_DIR  = OUT  / "chroma_text_store"
MANIFEST_CSV = OUT / "ingest_manifest.csv"

EMBED_MODEL    = "all-MiniLM-L6-v2"
COLLECTION_NAME = "fda_observations"
MIN_OBS_CHARS  = 80   # discard chunks shorter than this (headers, page numbers, etc.)
BATCH_SIZE     = 64   # ChromaDB add batch size


# ══════════════════════════════════════════════════════════════════════════
# OBSERVATION PARSERS
# ══════════════════════════════════════════════════════════════════════════

def parse_483_observations(text: str) -> list[str]:
    """
    Split a Form 483 PDF text into individual observation chunks.

    Strategy (in order of preference):
    1. Split on 'OBSERVATION N' or 'Observation N' header lines.
    2. Split on standalone numbered lines like '  1.' or '1.\n'.
    3. Fall back: return the whole document as a single chunk.
    """
    # Pattern 1: OBSERVATION N (case-insensitive, on its own line, optional colon)
    parts = re.split(r'\n\s*OBSERVATION\s+\d+\s*:?\s*\n', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        obs = [p.strip() for p in parts[1:] if len(p.strip()) >= MIN_OBS_CHARS]
        if obs:
            return obs

    # Pattern 2: standalone "  N." at line start (blank line before optional)
    parts = re.split(r'(?:^|\n)\s{0,4}(\d{1,2})\.\s*\n', text)
    # re.split with groups returns ['prefix', 'N', 'text', 'N', 'text', ...]
    if len(parts) > 2:
        obs = []
        # parts[0] is pre-first, then alternating (number, body)
        for i in range(2, len(parts), 2):
            body = parts[i].strip()
            if len(body) >= MIN_OBS_CHARS:
                obs.append(body)
        if obs:
            return obs

    # Fall back: whole document
    stripped = text.strip()
    return [stripped] if len(stripped) >= MIN_OBS_CHARS else []


def parse_wl_violations(text: str) -> list[str]:
    """
    Split Warning Letter repeat-section text into individual violation chunks.

    Strategy (in order):
    1. Split on numbered CFR citations '  1. 21 CFR ...' or '1. Observations ...'
    2. Split on raw '21 CFR' occurrences (each section starts with a CFR citation).
    3. Fall back: return the whole text as a single chunk.
    """
    if not text or len(text.strip()) < MIN_OBS_CHARS:
        return []

    # Pattern 1: numbered violations starting with "21 CFR" or "Observation"
    parts = re.split(
        r'(?:^|\n)\s{0,4}\d+\.\s+(?=21 CFR|Observation)',
        text,
        flags=re.IGNORECASE
    )
    if len(parts) > 1:
        chunks = [p.strip() for p in parts if len(p.strip()) >= MIN_OBS_CHARS]
        if chunks:
            return chunks

    # Pattern 2: split on any "21 CFR" occurrence
    parts = re.split(r'\n(?=21 CFR\s)', text)
    if len(parts) > 1:
        chunks = [p.strip() for p in parts if len(p.strip()) >= MIN_OBS_CHARS]
        if chunks:
            return chunks

    # Fall back: whole text
    stripped = text.strip()
    return [stripped] if len(stripped) >= MIN_OBS_CHARS else []


# ══════════════════════════════════════════════════════════════════════════
# PDF LOADER
# ══════════════════════════════════════════════════════════════════════════

def load_pdf_text(pdf_path: Path) -> str | None:
    """
    Extract text from a PDF using pdfplumber.
    Returns None if pdfplumber is unavailable or the PDF is unreadable.
    OCR fallback (pytesseract) is attempted if the extracted text is empty.
    """
    try:
        import pdfplumber
    except ImportError:
        print("  [WARN] pdfplumber not installed — skipping PDF text extraction.")
        return None

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
            text = "\n".join(pages).strip()

        if text:
            return text

        # No text layer — attempt OCR (optional)
        try:
            import pytesseract
            from PIL import Image
            import pdfplumber
            ocr_pages = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    img = page.to_image(resolution=200).original
                    ocr_pages.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_pages).strip()
            if text:
                return text
        except ImportError:
            pass  # tesseract not installed — that's fine, just skip

        return None
    except Exception as exc:
        print(f"  [WARN] Could not read {pdf_path.name}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Import heavy dependencies ──────────────────────────────────────────
    try:
        import chromadb
    except ImportError:
        sys.exit("chromadb not installed. Run: pip install chromadb>=0.5")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed. Run: pip install sentence-transformers>=3.0")

    # ── Load reference FEIs ───────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1 — Loading reference FEIs")
    print("=" * 65)
    valisure = pd.read_excel(VALISURE, sheet_name="API Only_FEI Mapping")
    ref_feis = set(valisure["FEI_NUMBER"].dropna().astype(int))
    print(f"Reference FEIs: {len(ref_feis)}")

    # ── Initialize ChromaDB ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — Initializing ChromaDB vector store")
    print("=" * 65)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Build set of already-ingested chunk IDs for idempotency
    existing_result = collection.get(include=[])
    existing_ids: set[str] = set(existing_result["ids"])
    print(f"Chunks already in store: {len(existing_ids)}")

    # ── Initialize embedder ───────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedding model ready.")

    # ── Collect all chunks to ingest ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Collecting chunks from 483 PDFs")
    print("=" * 65)

    all_chunks: list[dict] = []   # {chunk_id, fei, doc_id, observation_id, source_type, text, insp_date}
    manifest_rows: list[dict] = []

    # ── A: 483 PDFs ────────────────────────────────────────────────────────
    if not INV_483.exists():
        print(f"[WARN] 483 inventory not found: {INV_483}")
    else:
        inv = pd.read_csv(INV_483)
        inv["fei"] = pd.to_numeric(inv["fei"], errors="coerce").astype("Int64")
        inv = inv[inv["fei"].isin(ref_feis)].copy()
        inv = inv[inv["is_extractable"].astype(str).str.lower() == "true"].copy()
        print(f"Extractable 483 PDFs in inventory: {len(inv)}")

        n_pdfs_loaded = 0
        n_obs_found   = 0

        for _, row in inv.iterrows():
            fei      = int(row["fei"])
            filename = str(row["filename"])  # no .pdf extension in inventory
            insp_date = str(row.get("insp_date", ""))[:10]

            # Locate PDF file
            pdf_path = PDF_DIR / (filename + ".pdf")
            if not pdf_path.exists():
                # Try glob for partial filename match (some names are truncated)
                candidates = list(PDF_DIR.glob(f"{filename[:40]}*.pdf"))
                pdf_path = candidates[0] if candidates else None

            if pdf_path is None or not pdf_path.exists():
                manifest_rows.append({
                    "fei": fei, "doc_id": filename,
                    "observation_id": "N/A", "chunk_id": "N/A",
                    "source_type": "483", "n_chars": 0,
                    "status": "pdf_not_found", "insp_date": insp_date,
                })
                continue

            text = load_pdf_text(pdf_path)
            if not text:
                manifest_rows.append({
                    "fei": fei, "doc_id": filename,
                    "observation_id": "N/A", "chunk_id": "N/A",
                    "source_type": "483", "n_chars": 0,
                    "status": "no_text_extracted", "insp_date": insp_date,
                })
                continue

            n_pdfs_loaded += 1
            observations = parse_483_observations(text)

            for i, obs_text in enumerate(observations, start=1):
                obs_id   = f"{filename}_obs_{i:03d}"
                chunk_id = obs_id  # unique per (doc, observation)
                n_obs_found += 1

                status = "already_in_store" if chunk_id in existing_ids else "pending"
                manifest_rows.append({
                    "fei": fei, "doc_id": filename,
                    "observation_id": obs_id, "chunk_id": chunk_id,
                    "source_type": "483", "n_chars": len(obs_text),
                    "status": status, "insp_date": insp_date,
                })

                if chunk_id not in existing_ids:
                    all_chunks.append({
                        "chunk_id":       chunk_id,
                        "fei":            fei,
                        "doc_id":         filename,
                        "observation_id": obs_id,
                        "source_type":    "483",
                        "text":           obs_text[:4000],  # cap for embedding
                        "insp_date":      insp_date,
                    })

        print(f"  PDFs loaded: {n_pdfs_loaded}  |  Observations found: {n_obs_found}")

    # ── B: Warning Letter text ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Collecting chunks from Warning Letters")
    print("=" * 65)

    if not WL_REC_CSV.exists():
        print(f"[WARN] WL records not found: {WL_REC_CSV}")
    else:
        wl_df = pd.read_csv(WL_REC_CSV)
        fei_col = "search_fei" if "search_fei" in wl_df.columns else "primary_fei"
        wl_df["_fei"] = pd.to_numeric(wl_df[fei_col], errors="coerce").astype("Int64")
        wl_df = wl_df[wl_df["_fei"].isin(ref_feis)].copy()

        n_wl_loaded = 0
        n_viol_found = 0

        for _, row in wl_df.iterrows():
            fei        = int(row["_fei"])
            wl_number  = str(row.get("wl_number", "unknown"))
            wl_date    = str(row.get("wl_date", ""))[:10]
            rep_text   = str(row.get("repeat_section_text", "") or "")

            doc_id = f"WL_{wl_number}"

            violations = parse_wl_violations(rep_text)

            if not violations:
                manifest_rows.append({
                    "fei": fei, "doc_id": doc_id,
                    "observation_id": "N/A", "chunk_id": "N/A",
                    "source_type": "WL", "n_chars": 0,
                    "status": "no_violation_text", "insp_date": wl_date,
                })
                continue

            n_wl_loaded += 1
            for i, viol_text in enumerate(violations, start=1):
                obs_id   = f"{doc_id}_viol_{i:03d}"
                chunk_id = obs_id
                n_viol_found += 1

                status = "already_in_store" if chunk_id in existing_ids else "pending"
                manifest_rows.append({
                    "fei": fei, "doc_id": doc_id,
                    "observation_id": obs_id, "chunk_id": chunk_id,
                    "source_type": "WL", "n_chars": len(viol_text),
                    "status": status, "insp_date": wl_date,
                })

                if chunk_id not in existing_ids:
                    all_chunks.append({
                        "chunk_id":       chunk_id,
                        "fei":            fei,
                        "doc_id":         doc_id,
                        "observation_id": obs_id,
                        "source_type":    "WL",
                        "text":           viol_text[:4000],
                        "insp_date":      wl_date,
                    })

        print(f"  WL records: {n_wl_loaded}  |  Violation chunks found: {n_viol_found}")

    # ── Embed and store ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5 — Embedding and storing new chunks")
    print("=" * 65)
    print(f"New chunks to embed: {len(all_chunks)}")

    if all_chunks:
        ids        = [c["chunk_id"]    for c in all_chunks]
        texts      = [c["text"]        for c in all_chunks]
        metadatas  = [
            {
                "fei":            str(c["fei"]),
                "doc_id":         c["doc_id"],
                "observation_id": c["observation_id"],
                "source_type":    c["source_type"],
                "insp_date":      c["insp_date"],
            }
            for c in all_chunks
        ]

        # Embed in batches
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start: start + BATCH_SIZE]
            batch_ids   = ids[start: start + BATCH_SIZE]
            batch_meta  = metadatas[start: start + BATCH_SIZE]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = embedder.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            print(f"  Stored batch {start // BATCH_SIZE + 1} "
                  f"({start + len(batch_ids)} / {len(all_chunks)})")

        # Update manifest status for newly stored chunks
        stored_ids = set(ids)
        for row in manifest_rows:
            if row["chunk_id"] in stored_ids:
                row["status"] = "stored"
    else:
        print("  Nothing new to embed — all chunks already in store.")

    # ── Save manifest ─────────────────────────────────────────────────────
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    total_stored = collection.count()
    print(f"\n{'='*65}")
    print(f"DONE")
    print(f"{'='*65}")
    print(f"  Total chunks in store : {total_stored}")
    print(f"  Manifest rows         : {len(manifest_df)}")
    by_status = manifest_df.groupby("status").size().sort_values(ascending=False)
    print(by_status.to_string())
    by_source = manifest_df[manifest_df["status"].isin(["stored", "already_in_store"])].groupby("source_type").size()
    if not by_source.empty:
        print(f"\nChunks by source:")
        print(by_source.to_string())
    feis_covered = manifest_df[manifest_df["status"].isin(["stored", "already_in_store"])]["fei"].nunique()
    print(f"\nFEIs with at least one chunk: {feis_covered} / {len(ref_feis)}")
    print(f"\nOutputs:")
    print(f"  {CHROMA_DIR}")
    print(f"  {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
