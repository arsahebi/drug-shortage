"""
05_extract_signals_langgraph.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  LangGraph extraction graph: reads observation chunks from the ChromaDB store
  (built by 04), calls Claude once per unscored observation, and writes
  structured per-observation risk signals to CSV and JSON.

  This is step 2 of the LLM pipeline (04 → 05 → 06 → 07).
  Produces the core LLM signal table used by 06 and the dashboard.

  LangGraph nodes (run in order per observation):
    retrieve → pull chunk text + top-k context from the same FEI
    extract  → Claude call → structured JSON against the ObservationSignal schema
    classify → focused Claude call → refines root_cause_type; keyword baseline runs
               in parallel as an auditable eval comparator
    validate → schema + hallucination guard (evidence_quote must be a verbatim
               substring of source text); loops back to retrieve if low confidence
               (up to MAX_RETRIES attempts with broader context window)

WHEN TO RUN
  Run after 04_ingest_build_vectorstore.py.
  Requires ANTHROPIC_API_KEY to be set as an environment variable.
  Use --dry-run to count observations and estimate cost without making API calls.
  Use --fei <FEI_NUMBER> to test on a single facility.
  Use --limit N to process only N observations (useful for spot-checking).
  Idempotent: already-scored observations are skipped on re-run.
  Takes ~30–120 minutes for the full corpus (cost: ~600K tokens at claude-sonnet pricing).

REQUIRED FOR COMBINED DATASET?  NO — optional LLM enrichment layer.

INPUTS
  chroma_text_store/   ← ChromaDB vector store (built by 04)
  ingest_manifest.csv  ← chunk index (built by 04)

OUTPUTS (in this folder)
  fei_observation_signals.csv      ← one row per observation (primary LLM output)
  fei_observation_signals_raw.json ← same data in JSON

OUTPUT SCHEMA (per row in fei_observation_signals.csv)
  fei, doc_id, observation_id, source_type, insp_date
  violation_category   : LabControls | ProductionControls | BuildingsEquipment |
                         OrgPersonnel | PackagingLabeling | RecordsReports |
                         QualitySystem | Other
  severity_tier        : Low | Moderate | High
  severity_rationale   : 1-2 sentence explanation from Claude
  severity_tier_baseline: deterministic keyword result (audit comparator)
  root_cause_type      : Capital | Cultural | Mixed | Unclear  ← paper's key variable
  root_cause_rationale : 1-2 sentence explanation from Claude
  remediation_signal   : Strong | Partial | Weak | None
  repeat_flag          : bool — prior similar finding mentioned
  systemic_flag        : bool — facility-wide / programme-level failure indicated
  patient_risk_flag    : bool — direct patient safety risk mentioned
  evidence_quote       : verbatim substring from source text (hallucination guard)
  confidence           : float 0–1
  retry_count          : number of LangGraph retry loops used

ENVIRONMENT VARIABLES
  ANTHROPIC_API_KEY        (required)
  LANGCHAIN_TRACING_V2     (optional, set to "true" to enable LangSmith tracing)
  LANGCHAIN_API_KEY        (optional, needed if LangSmith tracing is enabled)
  LANGCHAIN_PROJECT        (optional, default: "fda-extraction")

DEPENDENCIES
  pip install langchain-anthropic>=0.3 langgraph>=0.2 chromadb>=0.5
              sentence-transformers>=3.0 pandas pydantic
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, field_validator

# ── Config constants (change model here) ──────────────────────────────────
CLAUDE_MODEL    = "claude-sonnet-4-6"
CONF_THRESHOLD  = 0.60    # below this → retry with more context
MAX_RETRIES     = 2
INITIAL_K       = 3       # context chunks retrieved on first attempt
FLUSH_EVERY     = 10      # write CSV after this many new observations

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
OUT  = Path(__file__).parent

CHROMA_DIR    = OUT / "chroma_text_store"
MANIFEST_CSV  = OUT / "ingest_manifest.csv"
SIGNALS_CSV   = OUT / "fei_observation_signals.csv"
SIGNALS_JSON  = OUT / "fei_observation_signals_raw.json"

COLLECTION_NAME = "fda_observations"
EMBED_MODEL     = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════
# PYDANTIC OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════

class ObservationSignal(BaseModel):
    fei:                  int
    doc_id:               str
    observation_id:       str
    source_type:          str  # "483" or "WL"
    violation_category:   Literal[
        "LabControls", "ProductionControls", "BuildingsEquipment",
        "OrgPersonnel", "PackagingLabeling", "RecordsReports",
        "QualitySystem", "Other"
    ]
    severity_tier:        Literal["Low", "Moderate", "High"]
    severity_rationale:   str
    severity_tier_baseline: str  # deterministic keyword baseline (eval comparator)
    root_cause_type:      Literal["Capital", "Cultural", "Mixed", "Unclear"]
    root_cause_rationale: str
    remediation_signal:   Literal["Strong", "Partial", "Weak", "None"]
    repeat_flag:          bool
    systemic_flag:        bool
    patient_risk_flag:    bool
    evidence_quote:       str
    confidence:           float
    retry_count:          int

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# ══════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════

class ExtractionState(dict):
    """
    TypedDict-compatible state for the LangGraph graph.
    Using a plain dict subclass for max langgraph-version compatibility.
    Fields:
      fei, doc_id, observation_id, source_type  — chunk identity
      chunk_text       — the primary observation text
      k                — number of context chunks to retrieve
      retrieved_context — list of neighbouring chunk texts (same FEI)
      extraction       — dict matching ObservationSignal (after extract node)
      retry_count      — how many re-retrieve loops have happened
      done             — True once validate decides to emit
      error            — error message string if something went wrong
    """


# ══════════════════════════════════════════════════════════════════════════
# KEYWORD BASELINE FOR SEVERITY TIER  (deterministic, used in eval)
# ══════════════════════════════════════════════════════════════════════════

_HIGH_KEYWORDS = [
    "contamination", "contaminated", "patient", "safety", "adverse",
    "out of specification", "oos", "out-of-specification", "sterility",
    "aseptic", "recall", "critical", "failed test", "data integrity",
    "falsif", "fabricat",
]
_MOD_KEYWORDS = [
    "repeat", "repeated", "recurring", "systemic", "documentation",
    "corrective action", "capa", "procedure not followed",
    "training", "management", "oversight", "deviation", "investigation",
]

def keyword_severity_baseline(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _HIGH_KEYWORDS):
        return "High"
    if any(k in t for k in _MOD_KEYWORDS):
        return "Moderate"
    return "Low"


# ══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════

EXTRACT_SYSTEM = """\
You are a pharmaceutical regulatory compliance expert analysing FDA Form 483
observations and Warning Letter violations. Your task is to extract structured
risk signals from regulatory text.

Return ONLY a single valid JSON object — no markdown, no explanation, no wrapper.
Every string field must be non-empty. evidence_quote must be a verbatim substring
of the OBSERVATION TEXT (copy it exactly, including punctuation)."""

EXTRACT_HUMAN_TMPL = """\
OBSERVATION TEXT:
{chunk_text}

CONTEXT (other observations from the same facility — for background only):
{context}

Extract the following fields and return a JSON object:
{{
  "violation_category": "<one of: LabControls | ProductionControls | BuildingsEquipment | OrgPersonnel | PackagingLabeling | RecordsReports | QualitySystem | Other>",
  "severity_tier": "<one of: Low | Moderate | High>",
  "severity_rationale": "<1-2 sentences>",
  "root_cause_type": "<one of: Capital | Cultural | Mixed | Unclear>",
  "root_cause_rationale": "<1-2 sentences>",
  "remediation_signal": "<one of: Strong | Partial | Weak | None>",
  "repeat_flag": <true|false>,
  "systemic_flag": <true|false>,
  "patient_risk_flag": <true|false>,
  "evidence_quote": "<verbatim span from OBSERVATION TEXT, max 220 chars>",
  "confidence": <float 0.0-1.0>
}}

FIELD DEFINITIONS:

violation_category — which GMP domain is violated:
  LabControls         = 21 CFR 211.160-176 (lab testing, OOS, specs)
  ProductionControls  = 21 CFR 211.80-115  (batch records, in-process controls)
  BuildingsEquipment  = 21 CFR 211.42-72   (equipment qualification, cleaning)
  OrgPersonnel        = 21 CFR 211.22-34   (responsibilities, training)
  PackagingLabeling   = 21 CFR 211.122-137
  RecordsReports      = 21 CFR 211.180-198 (document control, batch record review)
  QualitySystem       = overall QA/QC programme failure
  Other               = does not fit the above

severity_tier:
  High     = direct patient risk, contamination, OOS/OOT results, critical process failure
  Moderate = process deviations, documentation failures, repeat violations
  Low      = minor procedural gaps, administrative / labelling issues

root_cause_type (THEORETICAL CORE — assign carefully):
  Capital  = equipment, facility, or SOP / process-design gap
  Cultural = training, management oversight, or data-integrity failure
  Mixed    = clear textual evidence of BOTH Capital and Cultural
  Unclear  = observation text is insufficient to determine cause

remediation_signal (is there evidence the firm is addressing the issue?):
  Strong  = specific corrective actions described with timelines
  Partial = some acknowledgement or partial fix mentioned
  Weak    = vague commitment only ("will correct", "being addressed")
  None    = no remediation language at all

repeat_flag     = true if observation mentions prior similar findings or "repeat"
systemic_flag   = true if observation indicates facility-wide / programme-level failure
patient_risk_flag = true if observation mentions patient safety, adverse events, or
                    direct product quality risk to patients

confidence = your certainty in all fields combined (0.0 = no idea, 1.0 = certain)"""

CLASSIFY_SYSTEM = """\
You are a pharmaceutical regulatory expert. You will be given one
FDA regulatory observation and asked to determine the root cause type.
Return ONLY a JSON object with two fields."""

CLASSIFY_HUMAN_TMPL = """\
OBSERVATION EXCERPT (evidence quote):
{evidence_quote}

FULL OBSERVATION TEXT:
{chunk_text}

Determine the root cause type using these definitions:
  Capital  = equipment, facility, or SOP / process-design gap
  Cultural = training, management oversight, or data-integrity failure
  Mixed    = clear evidence of BOTH Capital and Cultural in this text
  Unclear  = text is insufficient to determine cause

Return ONLY:
{{
  "root_cause_type": "<Capital | Cultural | Mixed | Unclear>",
  "root_cause_rationale": "<1-2 concise sentences citing specific text>"
}}"""


# ══════════════════════════════════════════════════════════════════════════
# GRAPH NODE FUNCTIONS
# Each takes the full state dict and returns a dict of updated fields only.
# ══════════════════════════════════════════════════════════════════════════

def make_retrieve_node(collection, embedder):
    """Factory: returns the retrieve node with access to ChromaDB + embedder."""
    def retrieve(state: dict) -> dict:
        fei        = state["fei"]
        chunk_id   = state["observation_id"]
        chunk_text = state["chunk_text"]
        k          = state.get("k", INITIAL_K)

        context: list[str] = []

        # Only query for context if there are other chunks for this FEI
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q_emb = embedder.encode(
                    [chunk_text], normalize_embeddings=True, show_progress_bar=False
                ).tolist()

            results = collection.query(
                query_embeddings=q_emb,
                n_results=k + 1,  # +1 in case self is returned
                where={"fei": str(fei)},
                include=["documents", "ids"],
            )
            docs = results.get("documents", [[]])[0]
            ids  = results.get("ids",       [[]])[0]
            context = [
                d for d, i in zip(docs, ids)
                if i != chunk_id and len(d.strip()) > 50
            ][:k]
        except Exception:
            context = []

        return {"retrieved_context": context}
    return retrieve


def make_extract_node(llm, token_counter: dict):
    """Factory: returns the extract node with access to the LLM."""
    def extract(state: dict) -> dict:
        chunk_text = state["chunk_text"]
        context    = state.get("retrieved_context", [])

        context_str = "\n---\n".join(context) if context else "(no additional context)"

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=EXTRACT_SYSTEM),
            HumanMessage(content=EXTRACT_HUMAN_TMPL.format(
                chunk_text=chunk_text[:3500],
                context=context_str[:1500],
            )),
        ]

        try:
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            # Track token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_counter["input"]  += response.usage_metadata.get("input_tokens",  0)
                token_counter["output"] += response.usage_metadata.get("output_tokens", 0)
                token_counter["calls"]  += 1

            # Strip markdown code fences if Claude wrapped JSON
            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
            raw_text = re.sub(r'\s*```$',          '', raw_text, flags=re.MULTILINE)

            parsed = json.loads(raw_text)
            return {"extraction": parsed, "error": None}

        except json.JSONDecodeError as e:
            return {"extraction": None, "error": f"JSON parse error: {e}"}
        except Exception as e:
            return {"extraction": None, "error": f"LLM call failed: {e}"}
    return extract


def make_classify_node(llm, token_counter: dict):
    """Factory: returns the classify node with access to the LLM."""
    def classify(state: dict) -> dict:
        extraction = state.get("extraction")
        chunk_text = state["chunk_text"]

        if extraction is None:
            return {}  # no-op if extract failed

        # Deterministic keyword baseline (runs regardless of LLM result)
        baseline = keyword_severity_baseline(chunk_text)

        # Focused Claude call for root_cause_type
        evidence = extraction.get("evidence_quote", chunk_text[:300])

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=CLASSIFY_HUMAN_TMPL.format(
                evidence_quote=evidence[:500],
                chunk_text=chunk_text[:2500],
            )),
        ]

        try:
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_counter["input"]  += response.usage_metadata.get("input_tokens",  0)
                token_counter["output"] += response.usage_metadata.get("output_tokens", 0)
                token_counter["calls"]  += 1

            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
            raw_text = re.sub(r'\s*```$',          '', raw_text, flags=re.MULTILINE)

            classify_result = json.loads(raw_text)

            # Merge classify result into extraction, overriding root_cause fields
            updated = dict(extraction)
            updated["root_cause_type"]      = classify_result.get("root_cause_type",      extraction.get("root_cause_type", "Unclear"))
            updated["root_cause_rationale"] = classify_result.get("root_cause_rationale", extraction.get("root_cause_rationale", ""))
            updated["severity_tier_baseline"] = baseline
            return {"extraction": updated, "error": None}

        except Exception as e:
            # Fallback: keep LLM's root_cause from extract, add baseline
            updated = dict(extraction)
            updated["severity_tier_baseline"] = baseline
            return {"extraction": updated, "error": f"classify fallback ({e})"}
    return classify


def validate(state: dict) -> dict:
    """
    Schema integrity + hallucination guard.
    Sets done=True to emit, or increments retry_count to loop back.
    """
    extraction  = state.get("extraction")
    chunk_text  = state["chunk_text"]
    retry_count = state.get("retry_count", 0)

    REQUIRED = [
        "violation_category", "severity_tier", "severity_rationale",
        "root_cause_type", "root_cause_rationale", "remediation_signal",
        "repeat_flag", "systemic_flag", "patient_risk_flag",
        "evidence_quote", "confidence",
    ]
    VALID_CATS = {"LabControls","ProductionControls","BuildingsEquipment",
                  "OrgPersonnel","PackagingLabeling","RecordsReports",
                  "QualitySystem","Other"}
    VALID_SEV  = {"Low","Moderate","High"}
    VALID_RC   = {"Capital","Cultural","Mixed","Unclear"}
    VALID_REM  = {"Strong","Partial","Weak","None"}

    def _fail(reason: str) -> dict:
        if retry_count < MAX_RETRIES:
            return {
                "extraction":  None,
                "retry_count": retry_count + 1,
                "k":           state.get("k", INITIAL_K) + 3,
                "done":        False,
                "error":       reason,
            }
        # Max retries reached — emit None so the caller can log and skip
        return {"done": True, "error": reason}

    if extraction is None:
        return _fail("extraction is None")

    # Check required fields present and non-empty
    for field in REQUIRED:
        if field not in extraction or extraction[field] is None:
            return _fail(f"missing field: {field}")
        if isinstance(extraction[field], str) and not extraction[field].strip():
            return _fail(f"empty field: {field}")

    # Check enum values
    if extraction["violation_category"] not in VALID_CATS:
        return _fail(f"invalid violation_category: {extraction['violation_category']}")
    if extraction["severity_tier"] not in VALID_SEV:
        return _fail(f"invalid severity_tier: {extraction['severity_tier']}")
    if extraction["root_cause_type"] not in VALID_RC:
        return _fail(f"invalid root_cause_type: {extraction['root_cause_type']}")
    if extraction["remediation_signal"] not in VALID_REM:
        return _fail(f"invalid remediation_signal: {extraction['remediation_signal']}")

    # Hallucination guard: evidence_quote must be a substring of chunk_text
    eq = str(extraction.get("evidence_quote", "")).strip()
    # Use first 80 chars of quote for matching (Claude may truncate/elide)
    match_fragment = eq[:80].strip()
    if match_fragment and match_fragment.lower() not in chunk_text.lower():
        return _fail(f"evidence_quote not found in source text: '{match_fragment[:50]}…'")

    # Confidence threshold check
    conf = float(extraction.get("confidence", 0.0))
    if conf < CONF_THRESHOLD and retry_count < MAX_RETRIES:
        return {
            "extraction":  extraction,
            "retry_count": retry_count + 1,
            "k":           state.get("k", INITIAL_K) + 3,
            "done":        False,
            "error":       f"low confidence {conf:.2f} < {CONF_THRESHOLD}",
        }

    return {"done": True, "error": None}


def should_retry(state: dict) -> str:
    """Conditional edge: route back to retrieve or to END."""
    from langgraph.graph import END
    if state.get("done", False):
        return END
    return "retrieve"


# ══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_graph(collection, embedder, llm, token_counter: dict):
    from langgraph.graph import StateGraph, END

    graph = StateGraph(dict)

    graph.add_node("retrieve", make_retrieve_node(collection, embedder))
    graph.add_node("extract",  make_extract_node(llm, token_counter))
    graph.add_node("classify", make_classify_node(llm, token_counter))
    graph.add_node("validate", validate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract",  "classify")
    graph.add_edge("classify", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"retrieve": "retrieve", END: END},
    )

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LangGraph 483/WL signal extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count observations to score without calling the API")
    parser.add_argument("--fei", type=int, default=None,
                        help="Process only this FEI (for testing)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many new observations (for testing)")
    args = parser.parse_args()

    # ── API key check ──────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit(
            "\nERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "Or run with --dry-run to count observations without calling the API."
        )

    # ── LangSmith tracing (set env vars if provided) ───────────────────────
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith tracing: ENABLED (project: "
              f"{os.environ.get('LANGCHAIN_PROJECT', 'fda-extraction')})")
    else:
        print("LangSmith tracing: OFF (set LANGCHAIN_TRACING_V2=true to enable)")

    # ── Validate dependencies ──────────────────────────────────────────────
    try:
        import chromadb
    except ImportError:
        sys.exit("chromadb not installed. Run: pip install chromadb>=0.5")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers not installed.")
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        sys.exit("langchain-anthropic not installed. Run: pip install langchain-anthropic>=0.3")
    try:
        from langgraph.graph import StateGraph  # noqa: F401
    except ImportError:
        sys.exit("langgraph not installed. Run: pip install langgraph>=0.2")

    # ── Load ChromaDB ──────────────────────────────────────────────────────
    if not CHROMA_DIR.exists():
        sys.exit(
            f"ChromaDB store not found at {CHROMA_DIR}\n"
            "Run 04_ingest_build_vectorstore.py first."
        )
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"ChromaDB: {collection.count()} chunks loaded from {CHROMA_DIR.name}")

    # ── Load embedding model ───────────────────────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = SentenceTransformer(EMBED_MODEL)

    # ── Load manifest ──────────────────────────────────────────────────────
    if not MANIFEST_CSV.exists():
        sys.exit(
            f"Ingest manifest not found at {MANIFEST_CSV}\n"
            "Run 04_ingest_build_vectorstore.py first."
        )
    manifest = pd.read_csv(MANIFEST_CSV)
    scoreable = manifest[
        manifest["status"].isin(["stored", "already_in_store"])
        & (manifest["observation_id"] != "N/A")
    ].copy()

    if args.fei:
        scoreable = scoreable[scoreable["fei"] == args.fei].copy()

    # ── Load existing signals (idempotency) ────────────────────────────────
    existing_keys: set[tuple] = set()
    existing_results: list[dict] = []
    if SIGNALS_CSV.exists():
        existing_df = pd.read_csv(SIGNALS_CSV)
        existing_keys = set(zip(existing_df["doc_id"], existing_df["observation_id"]))
        existing_results = existing_df.to_dict("records")
        print(f"Already scored: {len(existing_keys)} observations (will skip)")

    # Filter to unscored
    unscored = scoreable[
        ~scoreable.apply(
            lambda r: (str(r["doc_id"]), str(r["observation_id"])) in existing_keys,
            axis=1
        )
    ].copy()

    if args.limit:
        unscored = unscored.head(args.limit)

    print(f"\nObservations to score: {len(unscored)}  (total scoreable: {len(scoreable)})")

    if args.dry_run:
        print("\nDry-run mode — no API calls made.")
        by_src = scoreable.groupby("source_type").size()
        print("By source type:")
        print(by_src.to_string())
        return

    if len(unscored) == 0:
        print("Nothing to score — all observations already processed.")
        return

    # ── Set up LLM ─────────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,
        max_tokens=1024,
        anthropic_api_key=api_key,
    )

    token_counter = {"input": 0, "output": 0, "calls": 0}

    # ── Build LangGraph ────────────────────────────────────────────────────
    app = build_graph(collection, embedder, llm, token_counter)

    # ── Process observations ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RUNNING EXTRACTION GRAPH")
    print("=" * 65)

    new_results: list[dict] = []
    n_success = 0
    n_failed  = 0

    # Fetch chunk texts from ChromaDB in bulk for efficiency
    all_chunk_ids = unscored["chunk_id"].dropna().tolist()
    chunk_text_map: dict[str, str] = {}
    if all_chunk_ids:
        BATCH = 500
        for i in range(0, len(all_chunk_ids), BATCH):
            batch_ids = all_chunk_ids[i: i + BATCH]
            result = collection.get(ids=batch_ids, include=["documents"])
            for cid, doc in zip(result["ids"], result["documents"]):
                chunk_text_map[cid] = doc

    for idx, (_, row) in enumerate(unscored.iterrows(), start=1):
        fei            = int(row["fei"])
        doc_id         = str(row["doc_id"])
        observation_id = str(row["observation_id"])
        chunk_id       = str(row.get("chunk_id", observation_id))
        source_type    = str(row.get("source_type", "483"))
        insp_date      = str(row.get("insp_date", ""))

        chunk_text = chunk_text_map.get(chunk_id, "")
        if not chunk_text.strip():
            print(f"  [{idx}/{len(unscored)}] SKIP {observation_id} — no text in store")
            n_failed += 1
            continue

        initial_state = {
            "fei":            fei,
            "doc_id":         doc_id,
            "observation_id": observation_id,
            "source_type":    source_type,
            "chunk_text":     chunk_text,
            "k":              INITIAL_K,
            "retrieved_context": [],
            "extraction":     None,
            "retry_count":    0,
            "done":           False,
            "error":          None,
        }

        try:
            final_state = app.invoke(initial_state)
        except Exception as exc:
            print(f"  [{idx}/{len(unscored)}] ERROR {observation_id}: {exc}")
            n_failed += 1
            continue

        extraction = final_state.get("extraction")
        if extraction is None:
            err = final_state.get("error", "unknown")
            print(f"  [{idx}/{len(unscored)}] FAILED {observation_id}: {err}")
            n_failed += 1
            continue

        # Build output record
        record = {
            "fei":                   fei,
            "doc_id":                doc_id,
            "observation_id":        observation_id,
            "source_type":           source_type,
            "insp_date":             insp_date,
            "violation_category":    extraction.get("violation_category",    "Other"),
            "severity_tier":         extraction.get("severity_tier",         "Low"),
            "severity_rationale":    extraction.get("severity_rationale",    ""),
            "severity_tier_baseline":extraction.get("severity_tier_baseline", keyword_severity_baseline(chunk_text)),
            "root_cause_type":       extraction.get("root_cause_type",       "Unclear"),
            "root_cause_rationale":  extraction.get("root_cause_rationale",  ""),
            "remediation_signal":    extraction.get("remediation_signal",    "None"),
            "repeat_flag":           bool(extraction.get("repeat_flag",      False)),
            "systemic_flag":         bool(extraction.get("systemic_flag",    False)),
            "patient_risk_flag":     bool(extraction.get("patient_risk_flag",False)),
            "evidence_quote":        str(extraction.get("evidence_quote",    ""))[:300],
            "confidence":            float(extraction.get("confidence",      0.0)),
            "retry_count":           int(final_state.get("retry_count",      0)),
        }

        new_results.append(record)
        n_success += 1

        # Running token report
        conf_str = f"{record['confidence']:.2f}"
        print(f"  [{idx}/{len(unscored)}] OK  fei={fei} {observation_id[-30:]:>30} "
              f"sev={record['severity_tier']:8} rc={record['root_cause_type']:8} "
              f"conf={conf_str}  "
              f"[tokens in={token_counter['input']:,} out={token_counter['output']:,}]")

        # Flush to disk periodically
        if n_success % FLUSH_EVERY == 0:
            all_records = existing_results + new_results
            pd.DataFrame(all_records).to_csv(SIGNALS_CSV, index=False)

    # ── Final write ────────────────────────────────────────────────────────
    all_records = existing_results + new_results
    results_df  = pd.DataFrame(all_records)
    results_df.to_csv(SIGNALS_CSV, index=False)

    with open(SIGNALS_JSON, "w") as f:
        json.dump(all_records, f, indent=2, default=str)

    print(f"\n{'='*65}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*65}")
    print(f"  New observations scored : {n_success}")
    print(f"  Failed / skipped        : {n_failed}")
    print(f"  Total in output file    : {len(all_records)}")
    print(f"  Token usage — input: {token_counter['input']:,}  "
          f"output: {token_counter['output']:,}  "
          f"calls: {token_counter['calls']:,}")
    print(f"\nOutputs:")
    print(f"  {SIGNALS_CSV}")
    print(f"  {SIGNALS_JSON}")

    if not results_df.empty:
        print(f"\nSignal distribution:")
        print(results_df["severity_tier"].value_counts().to_string())
        print(results_df["root_cause_type"].value_counts().to_string())
        feis_scored = results_df["fei"].nunique()
        print(f"\nFEIs with at least one scored observation: {feis_scored}")


if __name__ == "__main__":
    main()
