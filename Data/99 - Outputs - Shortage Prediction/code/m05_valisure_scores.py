# %%
"""
Module 5 — Valisure / DoD quality scores at drug-level.

Per drug (API), we summarize the processed Valisure combined file into:
    valisure_mean_score     : mean DoD Drug Score across tested manufacturers
    valisure_min_score      : worst single manufacturer
    valisure_n_companies    : number of distinct manufacturers tested
    valisure_n_failing      : number scoring < 70 (rough cutoff)
"""

from __future__ import annotations
import pandas as pd

from config import VALISURE_CSV, OUT_DATA, OUT_LOGS
from utils import get_logger, write_table

log = get_logger("m05_valisure", OUT_LOGS / "m05_valisure.log")


def build_valisure_features() -> pd.DataFrame:
    df = pd.read_csv(VALISURE_CSV)
    df.columns = [str(c).strip() for c in df.columns]
    score_col = "DoD Drug Score"
    if score_col not in df.columns:
        raise ValueError(f"Expected '{score_col}' in {VALISURE_CSV}; found {df.columns.tolist()}")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    out = (df.groupby("API", as_index=False)
           .agg(valisure_mean_score=(score_col, "mean"),
                valisure_min_score=(score_col, "min"),
                valisure_max_score=(score_col, "max"),
                valisure_n_companies=("FEI", "nunique"),
                valisure_n_failing=(score_col, lambda s: int((s < 70).sum()))))
    out = out.rename(columns={"API": "api"})
    out["drug_norm"] = out["api"]
    log.info("Valisure rollup: %d APIs", len(out))
    log.info("Score summary:\n%s", out[["api","valisure_mean_score","valisure_min_score","valisure_n_companies"]].to_string())
    return out

# %%
def main():
    out = build_valisure_features()
    write_table(out, OUT_DATA / "valisure_drug.parquet", log)
    return out


if __name__ == "__main__":
    main()

# %%
