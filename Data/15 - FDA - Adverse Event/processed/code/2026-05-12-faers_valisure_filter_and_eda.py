# %%
# 2026-05-12: FAERS filter + EDA for Valisure drugs
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

DATA_DIR = Path("/Users/asahebi/Library/Application Support/Code/User/workspaceStorage")
# override to project Data
DATA_DIR = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data")
PROC = DATA_DIR / "15 - FDA - Adverse Event" / "processed"
VAL_RAW = DATA_DIR / "08 - Valisure" / "raw"
OUT_FIGS = PROC / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# load latest merged FAERS
candidates = sorted(PROC.glob("faers_all_drugs_anda_linked_*.csv"))
if not candidates:
    raise FileNotFoundError("No faers_all_drugs_anda_linked_*.csv found in processed folder")
faers_csv = candidates[-1]
print('Using', faers_csv)

# load valisure FEI / drug list
fei_xlsx = VAL_RAW / 'FEIs_March 2026.xlsx'
if not fei_xlsx.exists():
    raise FileNotFoundError(f'Missing {fei_xlsx}')

# try to read sheets; expect a sheet with NDC/FEI/API names
val_df = pd.read_excel(fei_xlsx, dtype=str)
# heuristics: find column with drug name tokens
name_cols = [c for c in val_df.columns if re.search(r'drug|name|api|product', c, re.I)]
print('Valisure FEI file columns:', val_df.columns.tolist())
if name_cols:
    name_col = name_cols[0]
else:
    # fallback to first column
    name_col = val_df.columns[0]

# build a list of target drug names based on Valisure sheet — unique tokens
val_drugs = val_df[name_col].dropna().unique().tolist()
print('Valisure candidate drug names (sample 20):', val_drugs[:20])

# load FAERS subset (small columns)
usecols = ['primaryid','appl_no','prod_ai','drugname','severity','year','period']
df = pd.read_csv(str(fi:=str(faers_csv)), usecols=[c for c in usecols if c in pd.read_csv(fi, nrows=0).columns], dtype=str)

# normalize function
import unicodedata
import string

def normalize(s):
    if pd.isna(s):
        return ''
    s = str(s).lower()
    s = unicodedata.normalize('NFKD', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# prepare normalized lists
val_norm = [normalize(x) for x in val_drugs if x]
# filter FAERS by matching any valisure token in `prod_ai` or `drugname`
for col in ['prod_ai','drugname']:
    if col not in df.columns:
        df[col] = ''

# create combined normalized field
df['combined'] = (df['prod_ai'].fillna('') + ' ' + df['drugname'].fillna('')).map(normalize)

# build regex from val_norm tokens (escape)
terms = [re.escape(t) for t in val_norm if t]
# sort by length to prefer longer matches
terms = sorted(terms, key=lambda x: -len(x))
pattern = re.compile(r'(' + r'|'.join(terms) + r')') if terms else re.compile(r'\b$^')

mask = df['combined'].str.contains(pattern, na=False)
filtered = df[mask].copy()
print('Filtered rows:', len(filtered))
print('Non-filtered rows:', len(df) - len(filtered))

# Save filtered subset for the 14 Valisure drugs
out_fname_csv = PROC / f"faers_valisure_14_drugs_2026-05-12.csv"
out_fname_parquet = PROC / f"faers_valisure_14_drugs_2026-05-12.parquet"
filtered.to_csv(out_fname_csv, index=False)
try:
    filtered.to_parquet(out_fname_parquet, index=False)
except Exception:
    # parquet may fail if pyarrow/fastparquet not installed; ignore
    pass
print('Saved filtered data to', out_fname_csv)

# quick EDA on filtered set
if not filtered.empty:
    filtered['year'] = pd.to_numeric(filtered['year'], errors='coerce')
    plt.figure(figsize=(8,4))
    filtered.groupby('year').size().plot(kind='bar')
    plt.title('Filtered FAERS rows by year (Valisure list)')
    plt.tight_layout()
    plt.savefig(OUT_FIGS / 'faers_valisure_rows_by_year.png')
    plt.close()

    plt.figure(figsize=(6,6))
    filtered['severity'] = filtered['severity'].fillna('No outcome reported')
    filtered['severity'].value_counts().head(20).plot(kind='barh')
    plt.title('Severity distribution (Valisure list)')
    plt.tight_layout()
    plt.savefig(OUT_FIGS / 'faers_valisure_severity_dist.png')
    plt.close()

    print('Saved Valisure-filtered figures to', OUT_FIGS)
else:
    print('No matches found for Valisure drug list in FAERS merged CSV')
