# %%
"""
Module 1 — Drug universe definition.

This script reads the processed Valisure combined CSV (`VALISURE_CSV`),
computes a pilot API table (one row per API) with the count of unique FEIs
per API (stored in `n_companies`), and writes the pilot + UUtah unique
drug lists to `OUT_DATA` as parquet files.

The code is split into small functions so it works both in the pipeline
(`main.py`) and in an interactive window where each section can be run/debugged.
"""

from __future__ import annotations
import pandas as pd

from config import VALISURE_CSV, OUT_DATA, OUT_LOGS, UUTAH_FILE
from utils import ValisureDrugMatcher, get_logger, load_valisure_api_names, write_table

log = get_logger("m01_drugs", OUT_LOGS / "m01_drugs.log")


# %%
def build_pilot_universe() -> pd.DataFrame:
    """Build the small Valisure-scored API universe."""
    df_val = pd.read_csv(VALISURE_CSV)
    df_val.columns = [c.strip().lower() for c in df_val.columns]
    if "api" not in df_val.columns or "fei" not in df_val.columns:
        raise ValueError(
            f"Expected 'API' and 'FEI' columns in {VALISURE_CSV}; "
            f"found: {df_val.columns.tolist()}"
        )

    df_val["api"] = df_val["api"].astype(str).str.strip()
    df_val["fei"] = df_val["fei"].astype(str).str.strip()

    n_fei = df_val.groupby("api")["fei"].nunique().reset_index(name="n_companies")
    pilot = n_fei.copy()
    pilot["drug_norm"] = pilot["api"]
    pilot["api_norm"] = pilot["api"]
    pilot["source"] = "valisure_combined"
    pilot = pilot.drop_duplicates("drug_norm").reset_index(drop=True)

    log.info("Pilot Valisure universe: %d APIs", len(pilot))
    log.info("Pilot APIs: %s", ", ".join(pilot["api"].tolist()))
    return pilot


# %%
def build_broader_universe() -> pd.DataFrame:
    """Build the UUtah names that match the Valisure API universe."""
    matcher = ValisureDrugMatcher(load_valisure_api_names(VALISURE_CSV))
    df_u = pd.read_excel(UUTAH_FILE, header=1)
    drug_col = df_u.columns[0]
    names = df_u[drug_col].dropna().astype(str).drop_duplicates()

    broader = pd.DataFrame({"drug_name": names}).reset_index(drop=True)
    broader["drug_norm"] = broader["drug_name"].map(matcher.match)
    broader = broader.dropna(subset=["drug_norm"]).reset_index(drop=True)

    log.info("UUtah names matched to Valisure APIs: %d rows, %d APIs",
             len(broader), broader["drug_norm"].nunique())
    return broader


# %%
def write_outputs(pilot: pd.DataFrame, broader: pd.DataFrame) -> None:
    """Write drug-universe outputs consumed by downstream modules."""
    write_table(pilot, OUT_DATA / "pilot_drugs.parquet", log)
    write_table(broader, OUT_DATA / "uutah_unique_drugs.parquet", log)


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run module 1 and return outputs for interactive inspection."""
    pilot = build_pilot_universe()
    broader = build_broader_universe()
    write_outputs(pilot, broader)
    return pilot, broader


# %%
if __name__ == "__main__":
    pilot, broader = main()

# %%
