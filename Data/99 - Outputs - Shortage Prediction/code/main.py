"""
Orchestrator — runs the full pipeline end-to-end.

Usage:
    python main.py            # full pipeline
    python main.py --step 03  # run only one step
"""

from __future__ import annotations
import argparse
import importlib
import time
from pathlib import Path

from utils import get_logger
from config import OUT_LOGS

log = get_logger("main", OUT_LOGS / "main.log")

STEPS = [
    ("01", "m01_drug_universe"),
    ("02", "m02_uutah_panel"),
    ("03", "m03_faers_features"),
    ("04", "m04_recall_features"),
    ("05", "m05_valisure_scores"),
    ("06", "m06_redica_features"),
    ("07", "m07_panel_assembly"),
    ("08", "m08_eda"),
    ("09", "m09_model"),
    ("10", "m10_lead_time"),
]


def run(step_filter: str | None = None):
    for tag, mod_name in STEPS:
        if step_filter and tag != step_filter:
            continue
        log.info("--- STEP %s: %s ---", tag, mod_name)
        t0 = time.time()
        mod = importlib.import_module(mod_name)
        mod.main()
        log.info("--- STEP %s done in %.1fs ---", tag, time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default=None, help="Step tag to run alone (e.g. '03')")
    args = ap.parse_args()
    run(args.step)
