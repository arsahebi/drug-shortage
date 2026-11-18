"""
Check the facts added for the J Pharm Innov R&R, Reviewer 1 comment 1.3
(market share follows price competition among many approved generics).

  1. Approved metformin ANDAs in the Orange Book (single ingredient,
     METFORMIN HYDROCHLORIDE): active (RX) vs discontinued, firms, dosage forms.
  2. Share of IQVIA market volume by country of manufacture in the analysis
     sample, overall and by test year.

Orange Book: Data/01 - Orange Book/input_data/products.txt (FDA download,
March 2025). Rerun after replacing that file with a newer download.
Sample: prepare("manual", "all") from build_variant_graphs.py, then rows with a
matched FEI in the U.S., India, or China (110 NDC-years / 84 NDCs); volume
shares use the rows with IQVIA volume.

Read-only: prints counts, writes nothing.
"""
# %%
from pathlib import Path

import pandas as pd

from build_variant_graphs import BASE, VOL_COL, prepare

OB_PRODUCTS = Path(BASE) / "Data/01 - Orange Book/input_data/products.txt"

# %% 1. Orange Book ANDAs
ob = pd.read_csv(OB_PRODUCTS, sep="~", dtype=str)
met = ob[ob.Ingredient.str.upper().eq("METFORMIN HYDROCHLORIDE")
         & ob.Appl_Type.eq("A")]
active = met[met.Type.eq("RX")]
print(f"Orange Book file: {OB_PRODUCTS.name}")
print(f"Metformin ANDAs: {active.Appl_No.nunique()} active (RX) from "
      f"{active.Applicant.nunique()} firms; "
      f"{met[met.Type.eq('DISCN')].Appl_No.nunique()} discontinued")
print(active.groupby("DF;Route").Appl_No.nunique().to_string(), "\n")

# %% 2. Volume share by country
d = prepare("manual", "all")
d = d[d.matched_fei.notna() & d.CountryCode.isin(["USA", "IND", "CHN"])]
v = d[d[VOL_COL].notna()]
print(f"Analysis sample: {len(d)} NDC-years; {len(v)} with IQVIA volume")
print(v.groupby("CountryCode").agg(ndc_years=("NDC11", "size"),
                                   ndcs=("NDC11", "nunique")).to_string())
share = v.groupby("CountryCode")[VOL_COL].sum() / v[VOL_COL].sum()
print("\nShare of volume:\n" + share.round(3).to_string())
by_year = (v.groupby(["TestYear", "CountryCode"])[VOL_COL].sum().unstack()
           .div(v.groupby("TestYear")[VOL_COL].sum(), axis=0))
print("\nShare of volume by test year:\n" + by_year.round(3).to_string())
