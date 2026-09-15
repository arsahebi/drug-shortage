"""
04_valisure_faers_validation.py
────────────────────────────────────────────────────────────────────────────
Validate FAERS as a manufacturing quality proxy by correlating facility-level
FAERS adverse event counts with Valisure independent chemical testing scores.

If FAERS tracks manufacturing quality, FEIs with lower Valisure scores
(more quality failures) should have higher FAERS serious AE counts.

Linkage chain:
  Valisure ANDA (Application Number)
    → FDA NDC product file (APPLICATIONNUMBER → PRODUCTNDC)
    → NDC-FEI mapping (NDC → FEI_NUMBER)
    → FEI-level FAERS AE counts (from panel parquet)

Outputs
───────
  outputs/tables/valisure_faers_fei_merged.csv   — merged FEI-level data
  outputs/tables/valisure_faers_correlation.csv  — correlation summary
  outputs/figures/valisure_faers_scatter.png     — scatter: score vs AE count
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent.parent.parent
DATA   = ROOT / "Data"
OUT    = HERE / "outputs"
OUT_TABS = OUT / "tables"
OUT_FIGS = OUT / "figures"

VALISURE_SCORE = DATA / "08 - Valisure" / "raw" / "Discrete Scoring_DoD First 13 Drug Scores with ANDAs & NDCs.xlsx"
NDC_PRODUCT    = DATA / "03 - FDA - NDC" / "product.csv"
NDC_FEI_MAP    = DATA / "17 - NDC, FEI Mapping" / "ndc_fei_from_labels.csv"
PANEL          = OUT / "fei_ae_panel.parquet"


def _load_valisure_scores() -> pd.DataFrame:
    """Load all Valisure drug sheets, keep ANDA + Score + drug."""
    xl = pd.ExcelFile(VALISURE_SCORE)
    dfs = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(VALISURE_SCORE, sheet_name=sheet)
        df["drug"] = sheet
        dfs.append(df)
    all_scores = pd.concat(dfs, ignore_index=True)

    an_col   = next((c for c in all_scores.columns if "application" in c.lower()), None)
    score_col = next((c for c in all_scores.columns if c.strip().lower() == "score"), None)
    co_col   = next((c for c in all_scores.columns if "company" in c.lower()), None)

    keep = {an_col: "anda", score_col: "valisure_score", co_col: "company", "drug": "drug"}
    df = all_scores[[c for c in keep if c]].rename(columns=keep)
    df["valisure_score"] = pd.to_numeric(df["valisure_score"], errors="coerce")
    df = df.dropna(subset=["anda", "valisure_score"])
    # normalise ANDA: keep digits only, zero-pad to 6
    df["anda_norm"] = df["anda"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    return df


def _load_ndc_anda_fei() -> pd.DataFrame:
    """
    Build ANDA → FEI mapping via:
      FDA NDC product (APPLICATIONNUMBER → NDC)
      NDC-FEI crosswalk (NDC → FEI)
    """
    prod = pd.read_csv(NDC_PRODUCT, low_memory=False, encoding="latin-1",
                       usecols=["APPLICATIONNUMBER", "PRODUCTNDC", "LABELERNAME"])
    prod["anda_norm"] = prod["APPLICATIONNUMBER"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    prod["ndc_norm"]  = prod["PRODUCTNDC"].astype(str).str.replace("-", "").str[:9]

    nf = pd.read_csv(NDC_FEI_MAP, low_memory=False,
                     usecols=["manufacture_ndc", "FEI_NUMBER"])
    nf = nf.rename(columns={"manufacture_ndc": "ndc_raw", "FEI_NUMBER": "fei"})
    nf["ndc_norm"] = nf["ndc_raw"].astype(str).str.replace("-", "").str.strip().str[:9]
    nf["fei"]      = pd.to_numeric(nf["fei"], errors="coerce").astype("Int64")
    nf = nf.dropna(subset=["fei"])

    merged = prod.merge(nf[["ndc_norm", "fei"]].drop_duplicates(), on="ndc_norm", how="inner")
    anda_fei = merged[["anda_norm", "fei"]].drop_duplicates()
    print(f"  ANDA→FEI map: {len(anda_fei)} pairs, {anda_fei['fei'].nunique()} unique FEIs")
    return anda_fei


def _fei_level_valisure(valisure: pd.DataFrame, anda_fei: pd.DataFrame) -> pd.DataFrame:
    """Join Valisure scores to FEIs; aggregate to FEI level."""
    merged = valisure.merge(anda_fei, on="anda_norm", how="inner")
    print(f"  Valisure–FEI matches: {len(merged)} rows, {merged['fei'].nunique()} FEIs")

    fei_scores = (
        merged.groupby("fei", as_index=False)
              .agg(
                  valisure_mean_score=("valisure_score", "mean"),
                  valisure_min_score=("valisure_score", "min"),
                  valisure_fail_rate=("valisure_score", lambda x: (x < 90).mean()),
                  n_valisure_products=("anda_norm", "nunique"),
                  n_drugs_tested=("drug", "nunique"),
              )
    )
    return fei_scores


def _fei_level_ae(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse panel to FEI level: mean AE across years."""
    ae = (
        panel.groupby("fei", as_index=False)
             .agg(
                 mean_ae_t0=("n_ae_t0", "mean"),
                 mean_ae_t1=("n_ae_t1", "mean"),
                 total_ae=("n_ae_t0", "sum"),
                 n_panel_years=("panel_year", "nunique"),
             )
    )
    return ae


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 5:
        return np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p), int(n)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                 xlabel: str, ylabel: str, title: str, out: Path) -> None:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 3:
        return
    r, p, n = _spearman(sub[x_col], sub[y_col])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(sub[x_col], np.log1p(sub[y_col]), alpha=0.6, s=35,
               color="#2563eb", edgecolors="none")

    z = np.polyfit(sub[x_col], np.log1p(sub[y_col]), 1)
    xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
    ax.plot(xr, np.polyval(z, xr), color="#dc2626", linewidth=1.5,
            label=f"ρ={r:.2f}{sig}  (n={n})")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out.name}")


FAERS_PARQ   = DATA / "15 - FDA - Adverse Event" / "processed" / "faers_valisure_14_drugs_2026-05-12.parquet"
VALISURE_FEI = DATA / "08 - Valisure" / "raw" / "FEIs_March 2026.xlsx"

# Canonical drug name mapping: Valisure sheet name → FAERS prod_ai first-word key
DRUG_KEY_MAP = {
    "Metformin":         "metformin",
    "Atorvastatin":      "atorvastatin",
    "Bupropion":         "bupropion",
    "Pantoprazole":      "pantoprazole",
    "Vancomycin":        "vancomycin",
    "Tacrolimus":        "tacrolimus",
    "Lisinopril":        "lisinopril",
    "Metoprolol":        "metoprolol",
    "Potassium chloride": "potassium",
    "Magnesium sulfate": "magnesium",
    "Metronidazole":     "metronidazole",
    "Calcium Gluconate": "calcium",
    "Ampicillin":        "ampicillin",
}


def _drug_level_valisure(valisure: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Valisure scores to drug level."""
    valisure["drug_key"] = valisure["drug"].map(DRUG_KEY_MAP)
    grp = (
        valisure.dropna(subset=["drug_key"])
                .groupby("drug_key", as_index=False)
                .agg(
                    valisure_mean_score=("valisure_score", "mean"),
                    valisure_min_score=("valisure_score", "min"),
                    valisure_fail_rate_90=("valisure_score", lambda x: (x < 90).mean()),
                    valisure_fail_rate_70=("valisure_score", lambda x: (x < 70).mean()),
                    n_products=("anda", "nunique"),
                )
    )
    return grp


def _drug_level_faers() -> pd.DataFrame:
    """Aggregate FAERS to drug level: total serious AEs per drug."""
    df = pd.read_parquet(FAERS_PARQ)
    df["drug_key"] = df["prod_ai"].str.strip().str.lower().str.split().str[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # serious AEs only
    serious_pat = r"death|died|fatal|life.?threat|hospital|disab|congenital|required intervention|other serious"
    df["serious"] = df["severity"].astype(str).str.lower().str.contains(serious_pat, regex=True, na=False)

    grp = (
        df.groupby("drug_key", as_index=False)
          .agg(
              n_total_ae=("primaryid", "count"),
              n_serious_ae=("serious", "sum"),
              n_years=("year", "nunique"),
          )
    )
    grp["ae_per_year"] = grp["n_total_ae"] / grp["n_years"]
    grp["serious_per_year"] = grp["n_serious_ae"] / grp["n_years"]
    return grp


def main() -> None:
    OUT_TABS.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    print("Loading Valisure scores…")
    valisure = _load_valisure_scores()
    print(f"  {len(valisure)} ANDA-level rows across {valisure['drug'].nunique()} drugs")

    # ── Drug-level approach (primary) ─────────────────────────────────────
    print("\nBuilding drug-level Valisure summary…")
    drug_val = _drug_level_valisure(valisure)
    print(drug_val.to_string(index=False))

    print("\nBuilding drug-level FAERS summary…")
    drug_ae = _drug_level_faers()

    print("\nMerging drug-level Valisure + FAERS…")
    merged = drug_val.merge(drug_ae, on="drug_key", how="inner")
    print(f"  Merged: {len(merged)} drugs")
    print(merged[["drug_key", "valisure_mean_score", "valisure_fail_rate_90",
                   "ae_per_year", "serious_per_year"]].to_string(index=False))
    merged.to_csv(OUT_TABS / "valisure_faers_drug_merged.csv", index=False)

    # ── Correlations ──────────────────────────────────────────────────────
    pairs = [
        ("valisure_mean_score",    "ae_per_year",      "Valisure mean score",       "Total AEs per year"),
        ("valisure_mean_score",    "serious_per_year", "Valisure mean score",       "Serious AEs per year"),
        ("valisure_fail_rate_90",  "ae_per_year",      "Valisure fail rate (< 90)", "Total AEs per year"),
        ("valisure_fail_rate_90",  "serious_per_year", "Valisure fail rate (< 90)", "Serious AEs per year"),
        ("valisure_fail_rate_70",  "serious_per_year", "Valisure fail rate (< 70)", "Serious AEs per year"),
    ]

    corr_rows = []
    for x_col, y_col, xlabel, ylabel in pairs:
        r, p, n = _spearman(merged[x_col], merged[y_col])
        sig = "***" if (not np.isnan(p) and p < 0.001) else \
              "**"  if (not np.isnan(p) and p < 0.01)  else \
              "*"   if (not np.isnan(p) and p < 0.05)  else ""
        corr_rows.append({"x": x_col, "y": y_col,
                           "spearman_r": round(r, 4) if not np.isnan(r) else np.nan,
                           "p_value":    round(p, 4) if not np.isnan(p) else np.nan,
                           "sig": sig, "n": n})

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_TABS / "valisure_faers_correlation.csv", index=False)

    print("\nValisure–FAERS correlation results (drug level):")
    print(corr_df.to_string(index=False))

    # ── Scatter plots ─────────────────────────────────────────────────────
    plot_scatter(merged, "valisure_mean_score", "ae_per_year",
                 "Valisure mean score (higher = better quality)",
                 "log(1 + AEs per year)",
                 "Valisure score vs FAERS AE rate (drug level, n={})" .format(len(merged)),
                 OUT_FIGS / "valisure_faers_score_vs_ae.png")

    plot_scatter(merged, "valisure_fail_rate_90", "serious_per_year",
                 "Valisure failure rate (score < 90)",
                 "log(1 + serious AEs per year)",
                 "Valisure failure rate vs serious FAERS AEs (drug level, n={})".format(len(merged)),
                 OUT_FIGS / "valisure_faers_failrate_vs_serious_ae.png")

    print("\nNote: n =", len(merged), "drugs. FAERS volume is dominated by drug market size,")
    print("      not quality alone. Results are indicative; treat as hypothesis-generating.")
    print("\nDone. Outputs saved to", OUT_TABS, "and", OUT_FIGS)


if __name__ == "__main__":
    main()
