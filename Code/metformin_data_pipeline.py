
"""metformin_data_pipeline.py

End‑to‑end data assembly for the metformin quality study.

1. Reads:
   • Redica event logs (one workbook per Site‑ID)
   • Pre‑aggregated Metformin Manufacturer Site Score Table
   • NDC ↔ manufacturer cross‑walk (with Redica Site‑ID and Site Score)
   • Valisure contaminants dataset
   • IQVIA NPA monthly volume extract

2. Produces:
   • `master_table.parquet` — one row per NDC with:
       - labeler / manufacturer metadata
       - Redica compliance score (site‑level, 2018‑09‑01 cut‑off by default)
       - Valisure NDMA / DMF scores (lot‑ or mfr‑level aggregation configurable)
       - IQVIA script volume (summed over a user‑defined window)
   • a few helper CSVs for quick EDA

Usage
-----
python metformin_data_pipeline.py --data-root "G:/My Drive/.../06 - Metformin Data" \
                                  --out-dir   "G:/My Drive/.../06 - Metformin Data/derived" \
                                  --iqvia-window-start 2021-01-01 --iqvia-window-end 2021-12-31

All arguments are optional; see `--help`.

Author: ChatGPT – generated 2025‑06‑18
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
import glob
import warnings

# ------------------------------------------------------------------
# 1. Helpers
# ------------------------------------------------------------------

DATE_FMT = "%Y-%m-%d"

def _parse_date(s: str | pd.Timestamp) -> pd.Timestamp:
    """Robust date parser that tolerates NA/blank."""
    if pd.isna(s) or s == "":
        return pd.NaT
    return pd.to_datetime(s)

# ------------------------------------------------------------------
# 2. Readers
# ------------------------------------------------------------------

def read_redica_event_logs(event_dir: Path,
                           cutoff_end: pd.Timestamp | None = None,
                           cutoff_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Read and concatenate all per‑site Redica workbooks found in *event_dir*.

    Parameters
    ----------
    event_dir : Path
        Folder containing workbooks whose names start with the numeric Redica Site‑ID.
    cutoff_end, cutoff_start : optional
        Restrict events to this date range (inclusive).
    Returns
    -------
    DataFrame with columns:
    ['site_id', 'event_date', 'red_flag_criticality', 'red_flag_type',
     'red_flag_value', 'red_flag_agency', 'site_score']
    """
    dfs = []
    pattern = re.compile(r"^(\d{6,})\s*-")  # capture leading Site‑ID
    for fp in event_dir.glob("*.xls*"):
        m = pattern.match(fp.name)
        if not m:
            continue
        site_id = int(m.group(1))
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            warnings.warn(f"Failed reading {fp}: {e}")
            continue
        df.columns = (c.strip().lower().replace(' ', '_') for c in df.columns)
        rename = {
            'event_end_date': 'event_date',
            'site_score': 'site_score'
        }
        df = df.rename(columns=rename)
        df['site_id'] = site_id
        df['event_date'] = df['event_date'].apply(_parse_date)
        dfs.append(df[['site_id', 'event_date', 'red_flag_criticality',
                       'red_flag_type', 'red_flag_value', 'red_flag_agency',
                       'site_score']])
    if not dfs:
        raise FileNotFoundError(f"No Redica workbooks found in {event_dir}")
    out = pd.concat(dfs, ignore_index=True)
    if cutoff_start:
        out = out[out['event_date'] >= cutoff_start]
    if cutoff_end:
        out = out[out['event_date'] <= cutoff_end]
    return out

def read_site_score_table(fp: Path) -> pd.DataFrame:
    """Load pre‑aggregated manufacturer Site Score table."""
    df = pd.read_excel(fp)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.rename(columns={'red_flag_score': 'site_score'})
    return df[['site_app_id', 'site_display_name', 'fei', 'site_score']]

def read_ndc_mapping(fp: Path) -> pd.DataFrame:
    """Load master NDC ↔ manufacturer cross‑walk."""
    df = pd.read_excel(fp)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # normalise NDC to 11‑digit (strip hyphens, pad)
    df['ndc_11'] = df['ndc'].astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(11)
    return df

def read_valisure(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['ndc_11'] = df['ndc'].astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(11)
    return df

def read_iqvia_npa(fp: Path,
                   window_start: pd.Timestamp | None = None,
                   window_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load IQVIA NPA extract and sum TRx within *window* per NDC."""
    df = pd.read_excel(fp)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # columns like trx_nov_2018 … trx_oct_2024
    trx_cols = [c for c in df.columns if c.startswith('trx_')]
    if not trx_cols:
        raise ValueError("No TRx columns detected in IQVIA sheet")
    # Melt to long format, parse date from column name
    long = (df.melt(id_vars=['ndc'], value_vars=trx_cols, var_name='col', value_name='trx')
              .assign(period=lambda d: pd.to_datetime(
                  d['col'].str.replace('trx_', '').str.title(), format='%b_%Y')))
    if window_start:
        long = long[long['period'] >= window_start]
    if window_end:
        long = long[long['period'] <= window_end]
    agg = (long.groupby('ndc', as_index=False)['trx']
               .sum(min_count=1)
               .rename(columns={'trx': 'trx_volume'}))
    agg['ndc_11'] = agg['ndc'].astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(11)
    return agg[['ndc_11', 'trx_volume']]

# ------------------------------------------------------------------
# 3. Master assembler
# ------------------------------------------------------------------

def build_master_table(data_root: Path,
                       redica_cutoff_start: str = '2018-01-01',
                       redica_cutoff_end: str = '2021-09-01',
                       iqvia_window_start: str = '2021-01-01',
                       iqvia_window_end: str = '2021-12-31') -> pd.DataFrame:
    """Return merged DataFrame ready for EDA."""
    dr = Path(data_root)
    # locate files
    ndc_map_fp = next(dr.glob('*NDCs labelers*Redica*.xlsx'))
    val_fp = next(dr.glob('Valisure*Scoring*.xlsx'))
    iqvia_fp = next(dr.glob('Metformin Selected NDCs NPA*.xlsx'))
    site_score_fp = next(dr.glob('Metformin Manufacturer Site Score Table*.xlsx'))

    redica_dir = dr  # event log workbooks live in same folder
    reddit = read_redica_event_logs(redica_dir,
                                    cutoff_start=pd.to_datetime(redica_cutoff_start),
                                    cutoff_end=pd.to_datetime(redica_cutoff_end))
    # aggregate per site
    site_scores = (reddit.groupby('site_id', as_index=False)['site_score']
                          .sum()
                          .rename(columns={'site_score': 'site_score_agg'}))

    # master cross‑walk
    ndc_map = read_ndc_mapping(ndc_map_fp)

    # merge Redica score (collapse to site_id in cross‑walk)
    master = ndc_map.merge(site_scores,
                           how='left',
                           left_on='redica_site_id',
                           right_on='site_id')
    master['site_score'] = master['site_score_agg'].fillna(master.get('site_score'))

    # Valisure
    val_df = read_valisure(val_fp)
    master = master.merge(val_df,
                          how='left',
                          on='ndc_11',
                          suffixes=('', '_val'))

    # IQVIA
    iqvia_df = read_iqvia_npa(iqvia_fp,
                              window_start=pd.to_datetime(iqvia_window_start),
                              window_end=pd.to_datetime(iqvia_window_end))
    master = master.merge(iqvia_df, how='left', on='ndc_11')

    # housekeeping
    master = master.drop(columns=[c for c in ['site_id', 'site_score_agg'] if c in master.columns])
    return master

# ------------------------------------------------------------------
# 4. CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Build master metformin dataset')
    p.add_argument('--data-root', type=Path, required=True,
                   help='Folder containing all raw data files')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='Destination folder for master_table.parquet (default: data-root/derived)')
    p.add_argument('--iqvia-window-start', type=str, default='2021-01-01')
    p.add_argument('--iqvia-window-end', type=str, default='2021-12-31')
    p.add_argument('--redica-start', type=str, default='2018-01-01')
    p.add_argument('--redica-end', type=str, default='2021-09-01')
    args = p.parse_args()

    out_dir = args.out_dir or args.data_root / 'derived'
    out_dir.mkdir(parents=True, exist_ok=True)

    master = build_master_table(args.data_root,
                                redica_cutoff_start=args.redica_start,
                                redica_cutoff_end=args.redica_end,
                                iqvia_window_start=args.iqvia_window_start,
                                iqvia_window_end=args.iqvia_window_end)
    out_fp = out_dir / 'master_table.parquet'
    master.to_parquet(out_fp, index=False)
    print(f"✅ Master table written to {out_fp}  ({len(master):,} rows)")
    # also light CSV preview for Excel
    master.head(500).to_csv(out_dir / 'master_preview_sample.csv', index=False)
    print("⚠️  Only first 500 rows saved to CSV for quick peek (full set is parquet).")

if __name__ == '__main__':
    main()
