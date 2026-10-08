# %%
"""
Step 1 (ProPublica) — Build NDC→FEI Map
========================================
Third linkage method, alongside manual (step1_build_ndc_fei_map_manual.py) and
rule-based (step1_build_ndc_fei_map_rulebased.py). Used to check the
manuscript's claim that results are unchanged under ProPublica's linkage.

Source: Data/19 - ProPublica/raw/ndc_fei.csv (Rx Inspector public release,
  last updated Nov 2025). One row per (product NDC, FEI); product NDCs have
  two segments (labeler-product), so the join key is the 5-4 NDC9.

Universe: the 112 Valisure-tested NDC11s (taken from the manual map, which
  carries every tested NDC11 whether or not it was matched).

Rule
----
Keep every ProPublica facility except those flagged api_mfr == TRUE (API-only
makers, not the finished-dosage plant). No metformin NDC in our universe is
currently linked to an API-only row, but the filter is kept so the rule is
explicit.

Country: ProPublica gives the ISO-2 country of the registered address. It is
carried as `pp_country` (ISO-3) so step2 can apply the Canada/Bangladesh
exclusion and build_variant_graphs can assign a country to FEIs that are not
in the Redica metformin pull (6 of the 33 ProPublica FEIs).

Output: step1_ndc_fei_map_propublica.csv
  NDC, NDC11, NDC8, NDC9, FEI, manufacturer_name, pp_country, linkage_method,
  fei_count, facility_distance_km
"""

from pathlib import Path

import pandas as pd

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
SRC  = BASE / "Data/19 - ProPublica/raw/ndc_fei.csv"
UNIV = BASE / "Analysis/Metformin Paper/processed/step1_ndc_fei_map_manual.csv"
OUT  = BASE / "Analysis/Metformin Paper/processed/step1_ndc_fei_map_propublica.csv"

ISO2_TO_3 = {"IN": "IND", "US": "USA", "CN": "CHN", "CA": "CAN", "BD": "BGD"}


def norm_ndc11(x):
    lab, prod, pkg = str(x).strip().split("-")
    return f"{lab.zfill(5)}-{prod.zfill(4)}-{pkg.zfill(2)}"


def norm_ndc9(x):
    p = str(x).strip().split("-")
    return f"{p[0].zfill(5)}-{p[1].zfill(4)}"


def ndc_display(n11):
    lab, prod, pkg = n11.split("-")
    return f"{lab}-{prod.lstrip('0').zfill(3)}-{pkg}"


def ndc8(n11):
    lab, prod, _ = n11.split("-")
    return f"{lab}-{prod.lstrip('0').zfill(3)}"


# ── load ──────────────────────────────────────────────────────────────────────
pp = pd.read_csv(SRC, dtype=str)
pp["NDC9"] = pp["ndc"].apply(norm_ndc9)
pp = pp[pp["api_mfr"].fillna("").str.upper() != "TRUE"]

univ = pd.read_csv(UNIV, dtype=str)
ndcs = pd.DataFrame({"NDC11": sorted(set(univ["NDC11"].apply(norm_ndc11)))})
ndcs["NDC9"] = ndcs["NDC11"].str[:10]
print(f"Valisure-tested NDC11s: {len(ndcs)}  (NDC9s: {ndcs['NDC9'].nunique()})")

# ── emit one row per (NDC11, FEI) ────────────────────────────────────────────
out = ndcs.merge(
    pp[["NDC9", "fei", "registrant", "country", "linkage_method"]],
    on="NDC9", how="left",
).rename(columns={"fei": "FEI", "registrant": "manufacturer_name"})
out["pp_country"] = out["country"].map(ISO2_TO_3)
out["NDC"]  = out["NDC11"].apply(ndc_display)
out["NDC8"] = out["NDC11"].apply(ndc8)

n_per_ndc = out[out["FEI"].notna()].groupby("NDC11")["FEI"].nunique()
out["fei_count"] = out["NDC11"].map(
    lambda n: "Not Applicable" if n not in n_per_ndc
    else ("Single - ProPublica" if n_per_ndc[n] == 1 else "Multi FEI - ProPublica")
)
out["facility_distance_km"] = None

COLS = ["NDC", "NDC11", "NDC8", "NDC9", "FEI", "manufacturer_name", "pp_country",
        "linkage_method", "fei_count", "facility_distance_km"]
out = (out[COLS]
       .drop_duplicates(["NDC11", "FEI"])
       .sort_values(["NDC11", "FEI"], na_position="last")
       .reset_index(drop=True))
out.to_csv(OUT, index=False)

matched = out[out["FEI"].notna()]
print(f"\nSaved: {OUT}")
print(f"Rows                  : {len(out)}")
print(f"NDC11 with an FEI     : {matched['NDC11'].nunique()}")
print(f"NDC11 with >1 FEI     : {(n_per_ndc > 1).sum()}")
print(f"Unique FEIs           : {matched['FEI'].nunique()}")
print(f"\nCountry of linked (NDC11, FEI) pairs:")
print(matched["pp_country"].value_counts().to_string())
print(f"\nLinkage method of linked (NDC11, FEI) pairs:")
print(matched["linkage_method"].value_counts().to_string())
# %%
