# %%
from pathlib import Path
import pandas as pd

# ---------------- helpers ----------------
FIFTY_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY",
    "LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND",
    "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
}
STATES_50_DC = FIFTY_STATES | {"DC"}
TERRITORIES = {"PR","VI","GU","AS","MP"}

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _prep_ndc_cols(df: pd.DataFrame) -> pd.DataFrame:
    # ensure ndc11 and ndc9 exist and are zero-padded
    if "ndc11" not in df.columns:
        if "ndc" in df.columns:
            df["ndc11"] = (
                df["ndc"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11)
            )
        else:
            df["ndc11"] = (
                df.get("labeler_code","").astype(str).str.zfill(5) +
                df.get("product_code","").astype(str).str.zfill(4) +
                df.get("package_size","").astype(str).str.zfill(2)
            )
    if "ndc9" not in df.columns:
        df["ndc9"] = (
            df.get("labeler_code","").astype(str).str.zfill(5) +
            df.get("product_code","").astype(str).str.zfill(4)
        )
    return df

def sdud_to_monthly_national(df: pd.DataFrame, agg_level: str = "ndc11") -> pd.DataFrame:
    """
    Aggregate SDUD to national totals (50 states + DC), explode quarters to months (÷3).
    Returns monthly rows with volumes and amounts.
    """
    df = df.copy()

    # keep 50 states + DC, drop unknowns and territories
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper()
        df = df[df["state"].isin(STATES_50_DC)]
        df = df[~df["state"].isin(TERRITORIES)]
        df = df[df["state"] != "XX"]

    # numeric fields
    num_cols = [
        "units_reimbursed",
        "num_prescriptions",
        "total_amount_reimbursed",
        "medicaid_amount_reimbursed",
        "non_medicaid_amount_reimbursed",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = _to_num(df[c])

    # ensure Y/Q and NDC keys
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year","quarter"])
    df = _prep_ndc_cols(df)

    # quarterly national totals by NDC
    id_cols = ["year","quarter", agg_level]
    q_nat = df.groupby(id_cols, as_index=False)[num_cols].sum()

    # quarter → months
    q_to_months = {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}
    q_nat["month"] = q_nat["quarter"].map(q_to_months)
    m_nat = q_nat.explode("month").reset_index(drop=True)

    # divide by 3 to approximate monthly values
    m_nat[num_cols] = m_nat[num_cols].div(3.0)

    # add a month_start date (useful for joins/plots)
    m_nat["month"] = m_nat["month"].astype(int)
    m_nat["year"] = m_nat["year"].astype(int)
    m_nat["month_start"] = pd.to_datetime(dict(year=m_nat["year"], month=m_nat["month"], day=1))

    # order columns
    out_cols = ["year","month","quarter", agg_level] + num_cols + ["month_start"]
    return m_nat[out_cols].sort_values([agg_level,"year","month"]).reset_index(drop=True)

# ---------------- run ----------------
# Chnage directory accordingly
DATA_DIR = "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data/04 - Medicaid - SDUD/processed"
in_parquet = str(Path(DATA_DIR) / "2025-12-18-SDUDcanonical.parquet")
out_parquet = str(Path(DATA_DIR) / "2025-12-18-SDUDmonthly.parquet")
out_csv = str(Path(DATA_DIR) / "2025-12-18-SDUDmonthly.csv")  # optional

print("Loading sdud_master.parquet ...")
sdud_master = pd.read_parquet(in_parquet)  # requires pyarrow or fastparquet

sdud_monthly_nat = sdud_to_monthly_national(sdud_master, agg_level="ndc11")
print(sdud_monthly_nat.head())

# Save (Parquet recommended for size/speed)
sdud_monthly_nat.to_parquet(out_parquet, index=False)
# Optional CSV
sdud_monthly_nat.to_csv(out_csv, index=False)
print(f"Saved monthly national SDUD to:\n  {out_parquet}")

# %%
