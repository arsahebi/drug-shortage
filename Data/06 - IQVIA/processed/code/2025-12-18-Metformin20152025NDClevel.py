# %%
import pandas as pd
from pathlib import Path
import os

# 1. File paths
DATA_DIR = "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data"
IQVIA_DIR = os.path.join(DATA_DIR, "06 - IQVIA")
file_old = os.path.join(IQVIA_DIR, "raw/Metformin Jan 2015 - Mar 2025 No NDC.xlsx")
file_new = os.path.join(IQVIA_DIR, "raw/Metformin Jul 2019 - Jun 2025 NDC Level.xlsx")
out_file = os.path.join(IQVIA_DIR, "processed/2025-12-18-Metformin20152025NDClevel.xlsx")

# 2. Read all sheets from both workbooks
old_sheets = pd.read_excel(file_old, sheet_name=None)
new_sheets = pd.read_excel(file_new, sheet_name=None)

# 3. Key columns to match on
keys = [
    "Combined Molecule",
    "Corporation",
    "Manufacturer",
    "Product Sum",
    "Prod Form2",
    "Strength"
]

# 4. Prepare ExcelWriter for output
with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    for sheet_name, df_old in old_sheets.items():
        # If the same sheet exists in the new file, merge NDCs
        if sheet_name in new_sheets:
            df_new = new_sheets[sheet_name]
            # Aggregate NDCs by the key columns
            df_ndc = (
                df_new
                .groupby(keys)["NDC"]
                .apply(lambda s: ",".join(s.astype(str).unique()))
                .reset_index(name="NDC")
            )
            # Merge back onto the old sheet
            df_merged = df_old.merge(df_ndc, on=keys, how="left")
            df_merged.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # If no matching sheet, just write the original
            df_old.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"All sheets processed and saved to: {out_file}")

# %%
