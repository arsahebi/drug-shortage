# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

DATA_DIR = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data")
PROC = DATA_DIR / "15 - FDA - Adverse Event" / "processed"
OUT_FIGS = PROC / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# Find the ANDA-linked FAERS CSV (latest by name)
candidates = sorted(PROC.glob("faers_all_drugs_anda_linked_*.csv"))
if not candidates:
    raise FileNotFoundError("No faers_all_drugs_anda_linked_*.csv found in processed folder")
faers_csv = candidates[-1]
print("Using:", faers_csv)

# Load a subset for quick EDA
df = pd.read_csv(fi:=str(faers_csv), usecols=["primaryid","appl_no","severity","year","period"], dtype=str)

# Basic plots
plt.figure(figsize=(8,4))
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df_year = df.groupby('year').size()
df_year.plot(kind='bar')
plt.title('FAERS rows by year')
plt.tight_layout()
plt.savefig(OUT_FIGS / 'faers_rows_by_year.png')
plt.close()

plt.figure(figsize=(6,6))
df['severity'] = df['severity'].fillna('No outcome reported')
df['severity'].value_counts().head(20).plot(kind='barh')
plt.title('Severity distribution (top 20)')
plt.tight_layout()
plt.savefig(OUT_FIGS / 'faers_severity_dist.png')
plt.close()

print('Saved figures to', OUT_FIGS)
