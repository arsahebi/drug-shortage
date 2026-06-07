"""
Monthly pipeline orchestrator.

Runs the 6 monthly modules in sequence, producing:
    data/master_panel_monthly.csv (.parquet)
    outputs/tables/lead_lag_monthly.csv
    outputs/tables/recall_circularity.csv
    outputs/tables/monthly_analysis_summary.md
    outputs/figures/lead_lag_monthly_redica.png
    outputs/figures/lead_lag_monthly_faers.png
    outputs/figures/lead_lag_monthly_recalls.png
    outputs/figures/recall_circularity_analysis.png

Does NOT touch the annual pipeline (main.py / modules m01–m10).
Run this script from the code/ directory, or add it to an existing Jupyter
session by importing and calling run_monthly_pipeline().
"""

from __future__ import annotations
import time
import logging

from config import OUT_LOGS
from utils import get_logger

log = get_logger("main_monthly", OUT_LOGS / "main_monthly.log")


def run_monthly_pipeline(skip_if_exists: bool = False) -> None:
    """Run all 6 monthly pipeline steps in sequence."""
    from config import OUT_DATA

    steps = [
        ("MM01", "uutah_monthly_panel",   "mm01_uutah_monthly",   "main"),
        ("MM02", "recall_monthly",         "mm02_recall_monthly",  "main"),
        ("MM03", "faers_monthly",          "mm03_faers_monthly",   "main"),
        ("MM04", "redica_monthly",         "mm04_redica_monthly",  "main"),
        ("MM05", "master_panel_monthly",   "mm05_panel_monthly",   "main"),
        ("MM06", None,                     "mm06_lead_lag_monthly", "main"),
        ("MM07", None,                     "mm07_dashboard",        "main"),
    ]

    for step_id, output_stem, module_name, fn_name in steps:
        # Optionally skip if output already exists
        if skip_if_exists and output_stem is not None:
            csv = OUT_DATA / f"{output_stem}.csv"
            if csv.exists():
                log.info("SKIP %s — %s already exists", step_id, csv.name)
                continue

        log.info("=" * 60)
        log.info("Starting %s (%s.%s)", step_id, module_name, fn_name)
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            fn  = getattr(mod, fn_name)
            fn()
        except Exception:
            log.exception("FAILED at %s — aborting pipeline", step_id)
            raise
        elapsed = time.time() - t0
        log.info("Finished %s in %.1fs", step_id, elapsed)

    log.info("=" * 60)
    log.info("Monthly pipeline complete.")


if __name__ == "__main__":
    run_monthly_pipeline()
