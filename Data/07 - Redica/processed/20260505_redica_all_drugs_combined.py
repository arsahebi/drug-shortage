# %%
import pandas as pd
from pathlib import Path
import ast
import warnings

# Suppress potential openpyxl warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# =============================================================================
# 1. PATHS & SETTINGS
# =============================================================================
BASE_DIR = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
REDICA_DETAILED = BASE_DIR / "Data/07 - Redica/raw/Valisure_Sites_Red_Flag_Events.xlsx"
DATA_AVAILABILITY = BASE_DIR / "Data/07 - Redica/raw/Valisure_Sites_Data_Availability.xlsx"
SITE_LIST = BASE_DIR / "Data/07 - Redica/raw/Site List.xlsx"

# Output Paths
OUT_DIR = BASE_DIR / "Data/07 - Redica/processed/redica_all_drugs_combined.csv"
MISMATCH_DIR = BASE_DIR / "Data/07 - Redica/processed/redica_all_drugs_combined_mismatches_report.csv"

# =============================================================================
# 2. DATA LOADING & PRE-PROCESSING
# =============================================================================
xls_red = pd.ExcelFile(REDICA_DETAILED)
df = xls_red.parse(xls_red.sheet_names[0])

def parse_cell(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

df['Agency List'] = df['Agency List'].apply(parse_cell)
df['Industry List'] = df['Industry List'].apply(parse_cell)
df['Risk Event Attribute'] = df['Risk Event Attribute'].apply(parse_cell)
df['Risk Event Attribute Value'] = df['Risk Event Attribute Value'].apply(parse_cell)

# 2.3. Filtering Logic
# Note: Including 'Human Drugs' OR empty list [] to capture untagged FDA inspections
n_orig = len(df)
df = df[df['Agency List'].apply(lambda x: 'US - FDA' in x)]
df = df[df['Industry List'].apply(lambda x: 'Human Drugs' in x or len(x) == 0)]
print(f"Filtering: {n_orig - len(df)} rows removed (Non-FDA or Non-Drug Industry).")

# =============================================================================
# 3. INSPECTION EVENT EXTRACTION (GROUPING BY SITE + DATE)
# =============================================================================
grouped = df.groupby(['Site Redica Id', 'Site Display Name', 'Event Date'])
extracted_rows = []

for (site_id, site_name, event_date), group in grouped:
    is_483, is_warning_letter = 0, 0
    classification = None
    crit_483, maj_483, oth_483 = 0, 0, 0
    
    for idx, row in group.iterrows():
        attrs = row['Risk Event Attribute']
        vals = row['Risk Event Attribute Value']
        
        if "Inspection Outcome" in attrs:
            if "483" in vals: is_483 = 1
            elif "No 483" in vals: is_483 = 0
            if "Warning Letter" in vals: is_warning_letter = 1
            # Collect (classification, program) pairs; "Drug Quality Assurance"
            # takes priority over pre-approval programs like "Generic Drug Evaluation"
            cls_candidates = []
            for v in vals:
                if not isinstance(v, str):
                    continue
                if v == "NA":
                    cls_candidates.append(("NA", ""))
                elif "NAI" in v:
                    program = v.split(":", 1)[1].strip() if ":" in v else ""
                    cls_candidates.append(("NAI", program))
                elif "VAI" in v:
                    program = v.split(":", 1)[1].strip() if ":" in v else ""
                    cls_candidates.append(("VAI", program))
                elif "OAI" in v:
                    program = v.split(":", 1)[1].strip() if ":" in v else ""
                    cls_candidates.append(("OAI", program))
            dqa = next((cls for cls, prog in cls_candidates if prog == "Drug Quality Assurance"), None)
            if dqa:
                classification = dqa
            elif cls_candidates:
                for priority in ("OAI", "VAI", "NAI", "NA"):
                    match = next((cls for cls, _ in cls_candidates if cls == priority), None)
                    if match:
                        classification = match
                        break
        
        if "Post Inspection Document: 483" in attrs:
            for v in vals:
                if isinstance(v, dict):
                    crit_483 = v.get('critical', 0)
                    maj_483 = v.get('major', 0)
                    oth_483 = v.get('other', 0)

    extracted_rows.append({
        'Site Redica Id': site_id,
        'Site Display Name': site_name,
        'Event Date': event_date,
        'Classification': classification,
        '483': is_483,
        '483 critical': crit_483,
        '483 major': maj_483,
        '483 other': oth_483,
        'Warning Letter': is_warning_letter
    })

final_df = pd.DataFrame(extracted_rows)

# =============================================================================
# 4. MERGING METADATA & FEI
# =============================================================================
# Merge Site Totals
df_avail = pd.read_excel(DATA_AVAILABILITY)
cols_avail = ['Site Redica Id', 'Total Inspections', 'FDA Inspections', '483s Issued', 
              'Total Observations', 'Warning Letters Issued', 'Import Alerts Issued']
final_combined = pd.merge(final_df, df_avail[cols_avail], on='Site Redica Id', how='left')

# Merge FEI and Reorder
df_site = pd.read_excel(SITE_LIST)
final_combined = pd.merge(final_combined, df_site[['Site Redica Id', 'FEI']], on='Site Redica Id', how='left')

cols = final_combined.columns.tolist()
if 'FEI' in cols:
    cols.insert(0, cols.pop(cols.index('FEI')))
    final_combined = final_combined[cols]

# =============================================================================
# 5. DISCREPANCY ANALYSIS (THE MISMATCH PART)
# =============================================================================
# Calculate the sum of our extracted 1s per Site
site_stats = final_combined.groupby(['Site Redica Id', 'FEI', 'Site Display Name']).agg({
    '483': 'sum',
    '483s Issued': 'first'
}).reset_index()

# Find differences between calculated sum and metadata total
mismatches = site_stats[site_stats['483'] != site_stats['483s Issued']].copy()
mismatches['Difference'] = mismatches['483'] - mismatches['483s Issued']
mismatches = mismatches.rename(columns={'483': 'Calculated_Sum', '483s Issued': 'Metadata_Total'})

# =============================================================================
# 6. SAVE OUTPUTS
# =============================================================================
final_combined.to_csv(OUT_DIR, index=False)
mismatches.to_csv(MISMATCH_DIR, index=False)

print(f"SUCCESS:")
print(f"1. Main Data: {OUT_DIR}")
print(f"2. Mismatches Report ({len(mismatches)} sites): {MISMATCH_DIR}")

# Summary for the team
print("\nQuick Summary of Discrepancies:")
print(mismatches[['FEI', 'Calculated_Sum', 'Metadata_Total', 'Difference']].head())

"""
NOTES REGARDING DISCREPANCIES (FOR THE TEAM):
--------------------------------------------
When comparing the sum of extracted '483' flags vs. the '483s Issued' column from 
the summary metadata, you may still notice differences. This is expected due to:

1. SOURCE DATA TAGGING: The metadata summary counts ALL FDA inspections. Our detailed 
   event extraction only captures inspections where 'Industry List' is ['Human Drugs'] 
   or []. Some FDA inspections are tagged as 'Biologics' or 'Devices' in the detailed 
   logs and are excluded by our filters, causing our sum to be lower than the summary.

2. MISSING AUDIT LOGS: The 'Data Availability' file is a summary profile of the site. 
   The 'Red Flag Events' file is a detailed audit trail. Occasionally, a site profile 
   records that a 483 occurred historically, but the specific detailed "Red Flag" row 
   (needed for unpacking) is missing from the audit log spreadsheet.

3. DATE CONSOLIDATION: We group by 'Event Date'. If the FDA issued multiple 483 documents 
   on different dates for the same single inspection event, our code may count them 
   differently than the pre-calculated site summary.

4. UPDATED LOGIC: We have added logic to include 'Empty Industry Lists' []. This reduced 
   discrepancies for sites where the FDA inspection was simply not tagged with an 
   industry type in the detailed log.

QUESTION FOR REVIEW:
Are the 'critical', 'major', and 'other' counts related to the total number of 
observations in a 483? (i.e., does critical + major + other = Total Observations?)

"""

print(f"Extraction complete. Results saved to: {OUT_DIR}")
print(final_combined.head())
# %%
