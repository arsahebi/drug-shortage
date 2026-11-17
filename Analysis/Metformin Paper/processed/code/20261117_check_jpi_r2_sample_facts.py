"""
Check the sample facts added to the manuscript for the J Pharm Innov R&R
(Reviewer 2), on the same rows the figures use.

  R2.18  tested strengths and formulations (Methods, Product Quality)
  R2.30  tested lots per NDC-year (Limitations)

Sample: prepare("manual", "all") from build_variant_graphs.py, then rows with a
matched FEI in the U.S., India, or China -> 110 NDC-years / 84 NDCs.
Strength comes from step 3 (Valisure year-specific value, else step 2);
IR_ER comes from Valisure's per-year formulation file via prepare();
n_lots is the number of tested lots/samples per NDC-year from step 3.

Read-only: prints counts, writes nothing.
"""
import re

import pandas as pd

from build_variant_graphs import prepare

d = prepare("manual", "all")
d = d[d.matched_fei.notna() & d.CountryCode.isin(["USA", "IND", "CHN"])].copy()
print(f"Analysis sample: {len(d)} NDC-years, {d.NDC11.nunique()} NDCs\n")

# Raw strength labels vary ("500 mg", "500", "750mg"); group by the number.
d["strength_mg"] = d["Strength"].astype(str).map(
    lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else None)

print("R2.18 strength (mg):")
print(d.strength_mg.value_counts(dropna=False).sort_index().to_string(), "\n")
print("R2.18 formulation:")
print(d.IR_ER.value_counts(dropna=False).to_string(), "\n")
print("R2.18 strength x formulation:")
print(pd.crosstab(d.strength_mg, d.IR_ER.fillna("not recorded")).to_string(), "\n")
missing = d[d.IR_ER.isna()][["NDC11", "TestYear", "Strength"]]
if len(missing):
    print("Formulation not recorded:")
    print(missing.to_string(index=False), "\n")

print("R2.30 tested lots per NDC-year:")
print(d.n_lots.value_counts(dropna=False).sort_index().to_string())
