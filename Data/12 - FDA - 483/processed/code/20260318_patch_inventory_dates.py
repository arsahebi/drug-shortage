"""
Optional helper: patch missing insp_date values in 483_pdf_inventory.csv.

Most of the time you should rerun:
    20260316_483_comprehensive_extraction.py

That main script already applies the current date parser while rebuilding all
three 483 outputs. This helper is only for the narrow case where the inventory
CSV already exists and you want to backfill filename-derived dates without
re-reading every PDF.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = CODE_DIR.parent
INVENTORY_CSV = PROCESSED_DIR / "483_pdf_inventory.csv"
MAIN_EXTRACTOR = CODE_DIR / "20260316_483_comprehensive_extraction.py"


def load_main_extractor():
    spec = importlib.util.spec_from_file_location("fda483_extractor", MAIN_EXTRACTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MAIN_EXTRACTOR}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    extractor = load_main_extractor()
    inventory = pd.read_csv(INVENTORY_CSV)

    missing_before = int(inventory["insp_date"].isna().sum())
    fixed = 0

    for idx, row in inventory[inventory["insp_date"].isna()].iterrows():
        parsed = extractor.parse_date_from_filename(str(row["filename"]))
        if parsed is None:
            continue
        inventory.at[idx, "insp_date"] = parsed.strftime("%Y-%m-%d")
        fixed += 1
        print(f"fixed {row['fei']} -> {parsed:%Y-%m-%d} | {row['filename']}")

    inventory.to_csv(INVENTORY_CSV, index=False)

    missing_after = int(inventory["insp_date"].isna().sum())
    print(f"Loaded: {INVENTORY_CSV}")
    print(f"Missing before: {missing_before}")
    print(f"Fixed: {fixed}")
    print(f"Missing after: {missing_after}")


if __name__ == "__main__":
    main()
