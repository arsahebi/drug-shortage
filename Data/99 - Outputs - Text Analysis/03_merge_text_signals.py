"""
03_merge_text_signals.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Left-joins the LLM FEI-level features (from 02) onto the FEI node summary
  (from analytics/01) so that every FEI carries both its structured regulatory
  event counts and its LLM-derived Text Risk Index in a single analysis-ready table.

  This is the optional final step of the LLM pipeline (01 → 02 → 03).
  The original fei_node_summary.csv is NOT modified.

WHEN TO RUN
  Run after 02_aggregate_fei_features.py.
  Fast (<1 min). No API key needed.

REQUIRED FOR COMBINED DATASET?
  NO — optional. The enriched CSV is useful for the downstream prediction model
  but is not required to run the analytics dashboard.

INPUTS
  analytics/fei_node_summary.csv           ← produced by analytics/01 (all 129 FEIs)
  483_fei_text_features_static_fdapdf.csv ← produced by 02 --source pdf (LLM-scored FEIs only)

OUTPUTS (in this folder)
  fei_node_summary_enriched.csv   ← left join: all 129 FEIs + LLM columns (NaN if unscored)
"""

import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
NODES_CSV    = HERE / "analytics" / "fei_node_summary.csv"
RISK_CSV     = HERE / "483_fei_text_features_static_fdapdf.csv"   # produced by 02 --source pdf
ENRICHED_CSV = HERE / "fei_node_summary_enriched.csv"


def main():
    print("=" * 65)
    print("03_merge_text_signals.py — Enrich node summary with TRI signals")
    print("=" * 65)

    if not NODES_CSV.exists():
        sys.exit(
            f"Node summary not found: {NODES_CSV}\n"
            "Run 01_build_combined_dataset.py first."
        )

    nodes = pd.read_csv(NODES_CSV)
    nodes["fei"] = pd.to_numeric(nodes["fei"], errors="coerce").astype("Int64")
    print(f"Node summary rows  : {len(nodes)}")

    if not RISK_CSV.exists():
        print(
            f"[WARN] FEI context features not found: {RISK_CSV}\n"
            "Run 05_aggregate_fei_features.py first.\n"
            "Writing enriched CSV with NaN text-signal columns."
        )
        risk = pd.DataFrame(columns=["fei"])
    else:
        risk = pd.read_csv(RISK_CSV)
        risk["fei"] = pd.to_numeric(risk["fei"], errors="coerce").astype("Int64")
        print(f"Risk signal rows   : {len(risk)}")
        print(f"FEIs with TRI score: {risk['fei'].nunique()}")

    # Left join — every FEI in node summary is preserved
    enriched = nodes.merge(risk, on="fei", how="left")

    # Coverage report
    n_with_tri  = enriched["text_risk_index"].notna().sum()
    n_without   = len(enriched) - n_with_tri
    print(f"\nCoverage after join:")
    print(f"  FEIs with TRI  : {n_with_tri}  ({100*n_with_tri/len(enriched):.1f}%)")
    print(f"  FEIs without   : {n_without}  (no scored observations — TRI = NaN)")

    enriched.to_csv(ENRICHED_CSV, index=False)
    print(f"\nOutput: {ENRICHED_CSV}")
    print(f"  Columns: {len(enriched.columns)}")
    print(f"  Rows   : {len(enriched)}")


if __name__ == "__main__":
    main()
