# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from scipy import stats
from typing import Optional, Dict, Any
from scipy.stats import norm

# -------------------- PATHS --------------------
BASE_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/06 - Metformin Data/Derived"
)
DF_FILE = BASE_DIR / "Q&As1234_v8_new.xlsx"   # <-- update to your new file name as needed

# -------------------- CONSTANTS --------------------
DMF_COL = "DMF (ng/DAY) Valisure"
NDMA_COL = "NDMA (ng/DAY) Valisure"
DISS_COL = "Difference Factor"

COUNTRY_CODE_ORDER = ["IND", "USA", "CHN"]
SCORE_ORDER = [0.0, 1.5, 3.5]
SCORE_LABELS = {0.0: "NAI", 1.5: "VAI", 3.5: "OAI"}

country_code_colors = {
    "IND": "#ef4444",
    "CHN": "#f59e0b",
    "USA": "#3b82f6",
}

# -------------------- HELPERS --------------------
def pick_price_series(df: pd.DataFrame) -> pd.Series:
    """Prefer NADAC price if available; otherwise fall back to SDUD total-per-unit."""
    s = pd.Series(np.nan, index=df.index)
    if "nadac_price" in df.columns:
        s = df["nadac_price"].copy()
    if "sdud_price_total_per_unit" in df.columns:
        s = s.fillna(df["sdud_price_total_per_unit"])
    return s


def add_jitter_by_category(d, cat_col, value_col=None, radius=0.06):
    """Adds column 'jx' with symmetric offsets within [-radius, +radius]."""
    d = d.copy()
    d["jx"] = 0.0

    if value_col is None:
        for cat, idx in d.groupby(cat_col).groups.items():
            idx = list(idx)
            k = len(idx)
            if k > 1:
                offsets = np.linspace(-radius, radius, k)
                d.loc[idx, "jx"] = offsets
    else:
        for (cat, val), idx in d.groupby([cat_col, value_col]).groups.items():
            idx = list(idx)
            k = len(idx)
            if k > 1:
                offsets = np.linspace(-radius, radius, k)
                d.loc[idx, "jx"] = offsets
    return d


def add_n_labels_under_categories(ax, x_positions, n_values, y_axes_frac=0.03, fontsize=9):
    """Places 'n=..' at a fixed vertical position in axes coordinates (works with log y)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, n in zip(x_positions, n_values):
        ax.text(
            x, y_axes_frac, f"n={int(n)}",
            transform=trans, ha="center", va="bottom",
            fontsize=fontsize, color="#374151"
        )

def log1p_safe(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log1p(s.clip(lower=0))

def log_safe_pos(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0, np.nan)
    return np.log(s)

# -------------------- NDC NORMALIZATION --------------------
def _digits_only(x) -> str:
    if pd.isna(x):
        return ""
    return "".join(ch for ch in str(x).strip() if ch.isdigit())


def ndc10_to_ndc11_and_542(ndc10_digits: str):
    """
    Convert a 10-digit NDC (no hyphens) into:
      - NDC11 (11 digits)
      - NDC_542 (5-4-2 hyphenated)

    NOTE: A bare 10-digit NDC is ambiguous (could be 4-4-2, 5-3-2, or 5-4-1).
    We apply a deterministic heuristic that works well in practice:
      1) Try 4-4-2 if the first char is '0' (common for 4-digit labeler needing pad)
      2) Else try 5-3-2 (very common)
      3) Else fall back to 5-4-1
    """
    s = ndc10_digits
    if len(s) != 10:
        return (np.nan, np.nan)

    # Heuristic choose a split
    if s[0] == "0":
        # 4-4-2 -> pad labeler to 5
        a, b, c = s[:4], s[4:8], s[8:]
        ndc11 = a.zfill(5) + b + c
    else:
        # default 5-3-2 -> pad product to 4
        a, b, c = s[:5], s[5:8], s[8:]
        ndc11 = a + b.zfill(4) + c
        if len(ndc11) != 11:
            # fall back 5-4-1 -> pad package to 2
            a, b, c = s[:5], s[5:9], s[9:]
            ndc11 = a + b + c.zfill(2)

    if len(ndc11) != 11:
        return (np.nan, np.nan)

    ndc_542 = f"{ndc11[:5]}-{ndc11[5:9]}-{ndc11[9:]}"
    return (ndc11, ndc_542)


def ndc542_to_ndc11(ndc_542: str):
    if pd.isna(ndc_542):
        return np.nan
    s = str(ndc_542).strip().replace(" ", "")
    parts = s.split("-")
    if len(parts) != 3:
        return np.nan
    a = _digits_only(parts[0])
    b = _digits_only(parts[1])
    c = _digits_only(parts[2])
    if not a or not b or not c:
        return np.nan
    return a.zfill(5) + b.zfill(4) + c.zfill(2)


def ensure_ndc542_and_ndc11(d: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures:
      - NDC_542 exists (5-4-2 hyphenated)
      - NDC11 exists (11-digit)
    Accepts any of these inputs if present:
      - NDC_542 (preferred)
      - NDC11
      - NDC (10 digits)
      - product_ndc (common FDA naming)
    """
    out = d.copy()

    # 1) locate candidate raw NDC column (10-digit) if needed
    candidate_10 = None
    for c in ["NDC", "ndc", "product_ndc", "Product NDC", "productNDC"]:
        if c in out.columns:
            candidate_10 = c
            break

    # 2) Start by cleaning NDC11 if present
    if "NDC11" in out.columns:
        out["NDC11"] = out["NDC11"].astype(str).str.replace(r"\D", "", regex=True)
        out.loc[out["NDC11"].str.len() != 11, "NDC11"] = np.nan
    else:
        out["NDC11"] = np.nan

    # 3) If NDC_542 exists, use it to fill NDC11
    if "NDC_542" in out.columns:
        out["NDC11"] = out["NDC11"].fillna(out["NDC_542"].apply(ndc542_to_ndc11))

    # 4) If still missing NDC11 and we have a 10-digit NDC column, build both NDC11 + NDC_542
    if candidate_10 is not None:
        raw10 = out[candidate_10].apply(_digits_only)
        # Keep only 10-digit values
        raw10 = raw10.where(raw10.str.len() == 10, "")
        ndc11_542 = raw10.apply(lambda s: ndc10_to_ndc11_and_542(s) if s else (np.nan, np.nan))
        out["NDC11"] = out["NDC11"].fillna(ndc11_542.apply(lambda t: t[0]))
        # Create/keep NDC_542
        if "NDC_542" not in out.columns:
            out["NDC_542"] = ndc11_542.apply(lambda t: t[1])
        else:
            out["NDC_542"] = out["NDC_542"].fillna(ndc11_542.apply(lambda t: t[1]))

    # 5) If NDC_542 still missing but NDC11 exists, derive it
    if "NDC_542" not in out.columns:
        out["NDC_542"] = np.nan
    out.loc[out["NDC_542"].isna() & out["NDC11"].notna(), "NDC_542"] = (
        out.loc[out["NDC_542"].isna() & out["NDC11"].notna(), "NDC11"]
        .apply(lambda s: f"{str(s)[:5]}-{str(s)[5:9]}-{str(s)[9:]}")
    )

    return out


# -------------------- BUILD TABLES --------------------
def build_ndc_year_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    NDC–Year TEST GRANULARITY (matches Q&As134_v8_new structure)

    What this does (vs the prior 'test_df' you had):
      1) Normalizes NDC fields:
         - keeps/builds NDC_542 and NDC11 (handles 10-digit NDC inputs too)
      2) Applies measurement rules:
         - NDMA in 2024 -> NaN
         - Difference Factor only in 2024 -> for 2020/2022 force NaN
         - DO NOT convert NA to 0 for Difference Factor
      3) Collapses inspection-expanded rows:
         - one row per (NDC11, Year)
         - inspection columns are used only to compute PriorScore for that test-year
      4) PriorScore for each (NDC11, Year):
         - set TestDate = Dec 31 of that Year
         - pick most recent Event Start Date <= TestDate within that (NDC11,Year) group
         - use Event Score if present else Score
         - snap to {0.0, 1.5, 3.5}
    """
    d = df.copy()

    # --- dates ---
    if "Event Start Date" in d.columns:
        d["Event Start Date"] = pd.to_datetime(d["Event Start Date"], errors="coerce")

    # --- normalize NDC fields (creates NDC11 + NDC_542) ---
    d = ensure_ndc542_and_ndc11(d)
    d = d.dropna(subset=["NDC11"]).copy()

    # --- Year ---
    d["Year"] = pd.to_numeric(d.get("Year"), errors="coerce")
    d = d.dropna(subset=["Year"]).copy()
    d["Year"] = d["Year"].astype(int)

    # --- define TestDate (proxy) ---
    d["TestDate"] = pd.to_datetime(d["Year"].astype(str) + "-12-31")

    # --- price + volume ---
    d["price"] = pd.to_numeric(pick_price_series(d), errors="coerce")
    d["volume"] = pd.to_numeric(d.get("iqvia_extended_units"), errors="coerce")

    # --- quality metrics ---
    d["DMF"] = pd.to_numeric(d.get(DMF_COL), errors="coerce")
    d["NDMA"] = pd.to_numeric(d.get(NDMA_COL), errors="coerce")
    d.loc[d["Year"] == 2024, "NDMA"] = np.nan  # not measured in 2024

    # Difference factor column name differs across files
    # In Q&As1234_v8_new.xlsx it's "Dissolution" (per your file)
    # In Q&As134_v8_new.xlsx it's "Difference Factor"
    if "Difference Factor" in d.columns and DISS_COL not in d.columns:
        # keep your constant name DISS_COL="Difference Factor"
        pass

    # Support either column name
    diss_source_col = None
    if DISS_COL in d.columns:
        diss_source_col = DISS_COL
    elif "Dissolution" in d.columns:
        diss_source_col = "Dissolution"

    d["Dissolution"] = pd.to_numeric(d.get(diss_source_col), errors="coerce")
    d.loc[d["Year"].isin([2020, 2022]), "Dissolution"] = np.nan  # only measured in 2024
    # IMPORTANT: do NOT fill NA with 0

    # --- compute ScoreUsed on each inspection row (used for "prior") ---
    if "Event Score" in d.columns or "Score" in d.columns:
        d["ScoreUsed"] = pd.to_numeric(d.get("Event Score"), errors="coerce")
        d.loc[d["ScoreUsed"].isna(), "ScoreUsed"] = pd.to_numeric(d.get("Score"), errors="coerce")
    else:
        d["ScoreUsed"] = np.nan

    # --- collapse to one row per (NDC11, Year) ---
    keys = ["NDC11", "Year"]

    def prior_score_for_group(g: pd.DataFrame):
        # most recent inspection before/equal TestDate (which is year-end)
        test_date = g["TestDate"].iloc[0]
        gg = g.dropna(subset=["Event Start Date"]).copy()
        gg = gg[gg["Event Start Date"] <= test_date]
        if gg.empty:
            return (np.nan, pd.NaT)
        gg = gg.sort_values("Event Start Date")
        last = gg.iloc[-1]
        return (last["ScoreUsed"] if pd.notna(last["ScoreUsed"]) else np.nan, last["Event Start Date"])

    # first build the core aggregated metrics
    agg = (
        d.groupby(keys, as_index=False)
         .agg(
            # identifiers / descriptors
            NDC_542=("NDC_542", "first"),
            NDC=("NDC", "first"),
            NDC8=("NDC8", "first"),
            Strength=("Strength", "first"),

            FEI=("FEI", "first"),
            Firm=("Firm", "first"),
            CountryCode=("CountryCode", "first"),
            CountryName=("CountryName", "first"),

            # keep test date
            TestDate=("TestDate", "first"),

            # economics
            price=("price", "mean"),
            volume=("volume", "mean"),

            # quality
            DMF=("DMF", "mean"),
            NDMA=("NDMA", "mean"),
            Dissolution=("Dissolution", "mean"),

            # optional: how many inspection rows were collapsed
            n_event_rows=("Event Start Date", "size"),
         )
    )

    # now compute PriorScore_raw + PriorEventDate per (NDC11,Year)
    prior_raw = []
    prior_date = []
    for _, g in d.groupby(keys, sort=False):
        s, dt = prior_score_for_group(g)
        prior_raw.append(s)
        prior_date.append(dt)

    agg["PriorScore_raw"] = prior_raw
    agg["PriorEventDate"] = prior_date

    def nearest_score(x):
        if pd.isna(x):
            return np.nan
        arr = np.array(SCORE_ORDER, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(x)))])

    agg["PriorScore"] = agg["PriorScore_raw"].apply(nearest_score)
    agg["PriorScore_cat"] = pd.Categorical(agg["PriorScore"], categories=SCORE_ORDER, ordered=True)

    return agg



def build_events_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Event Start Date"] = pd.to_datetime(d.get("Event Start Date"), errors="coerce")
    for c in ["EventYear", "Score", "Event Score"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    events = (
        d[["FEI", "Event Start Date", "EventYear", "Event Score", "Score"]]
        .dropna(subset=["FEI", "Event Start Date"])
        .drop_duplicates(subset=["FEI", "Event Start Date"])
        .sort_values(["FEI", "Event Start Date"])
        .copy()
    )
    return events

def attach_prior_score_to_tests(test_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Robust FEI-wise prior score attachment.

    Why: pandas merge_asof requires the 'on' key to be globally sorted even with by=...
    So we do merge_asof PER FEI group (within-group TestDate is sorted), then concat.

    Returns test_df with:
      - PriorScore_raw
      - PriorScore snapped to {0.0, 1.5, 3.5}
      - PriorScore_cat categorical
    """
    t = test_df.copy()

    # --- required keys ---
    t = t.dropna(subset=["FEI", "TestDate"]).copy()
    t["FEI"] = t["FEI"].astype(str)
    t["TestDate"] = pd.to_datetime(t["TestDate"], errors="coerce")
    t = t.dropna(subset=["TestDate"]).copy()

    e = events.copy()
    e = e.dropna(subset=["FEI", "Event Start Date"]).copy()
    e["FEI"] = e["FEI"].astype(str)
    e["Event Start Date"] = pd.to_datetime(e["Event Start Date"], errors="coerce")
    e = e.dropna(subset=["Event Start Date"]).copy()

    # choose score
    e["ScoreUsed"] = pd.to_numeric(e.get("Event Score"), errors="coerce")
    e.loc[e["ScoreUsed"].isna(), "ScoreUsed"] = pd.to_numeric(e.get("Score"), errors="coerce")

    e = e.rename(columns={"Event Start Date": "EventDate"})
    e = e[["FEI", "EventDate", "ScoreUsed"]].copy()

    # sort events inside FEI
    e = e.sort_values(["FEI", "EventDate"], kind="mergesort").reset_index(drop=True)

    # Build a dict of event tables per FEI for fast access
    e_by_fei = {fei: g.sort_values("EventDate", kind="mergesort").reset_index(drop=True)
               for fei, g in e.groupby("FEI", sort=False)}

    out_parts = []

    for fei, g in t.groupby("FEI", sort=False):
        gg = g.sort_values("TestDate", kind="mergesort").reset_index(drop=True)

        ev = e_by_fei.get(fei)
        if ev is None or ev.empty:
            gg["PriorScore_raw"] = np.nan
            out_parts.append(gg)
            continue

        # within one FEI: no need for by=
        merged = pd.merge_asof(
            gg,
            ev,
            left_on="TestDate",
            right_on="EventDate",
            direction="backward",
            allow_exact_matches=True,
        )

        merged["PriorScore_raw"] = pd.to_numeric(merged["ScoreUsed"], errors="coerce")
        out_parts.append(merged)

    merged_all = pd.concat(out_parts, ignore_index=True)

    def nearest_score(x):
        if pd.isna(x):
            return np.nan
        arr = np.array(SCORE_ORDER, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(x)))])

    merged_all["PriorScore"] = merged_all["PriorScore_raw"].apply(nearest_score)
    merged_all["PriorScore_cat"] = pd.Categorical(merged_all["PriorScore"], categories=SCORE_ORDER, ordered=True)

    # optional: drop helper cols if you don’t want them
    # merged_all = merged_all.drop(columns=["EventDate", "ScoreUsed"], errors="ignore")

    return merged_all

from scipy.stats import spearmanr, pearsonr

def report_price_volume_relationship(
    ndc_year_df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "volume",
    by_year: bool = True,
    by_country: bool = True,
) -> None:
    """
    Quick check for the claim "market is focused on price":
    reports correlation between price and volume.

    - Spearman (rank correlation): robust to skew/outliers
    - Pearson on log10(price) vs log10(volume): common for economic data
    """

    d = ndc_year_df.copy()
    d[price_col] = pd.to_numeric(d.get(price_col), errors="coerce")
    d[volume_col] = pd.to_numeric(d.get(volume_col), errors="coerce")

    # basic validity: price finite, volume positive for log
    base = d[np.isfinite(d[price_col]) & np.isfinite(d[volume_col]) & (d[volume_col] > 0)].copy()

    def _one_report(tag: str, df_: pd.DataFrame) -> None:
        n = len(df_)
        if n < 3:
            print(f"{tag}: n={n} (too small)")
            return

        x = df_[price_col].astype(float).values
        y = df_[volume_col].astype(float).values

        # Spearman (raw)
        rho_s, p_s = spearmanr(x, y)

        # Pearson on log-log (requires positive price too)
        mask_pos = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if mask_pos.sum() >= 3:
            lx = np.log10(x[mask_pos])
            ly = np.log10(y[mask_pos])
            r_p, p_p = pearsonr(lx, ly)
            pearson_str = f"Pearson log-log: r={r_p:+.3f}, p={p_p:.3e}, n={int(mask_pos.sum())}"
        else:
            pearson_str = "Pearson log-log: n<3 (or nonpositive price)"

        print(
            f"{tag}: "
            f"Spearman: ρ={rho_s:+.3f}, p={p_s:.3e}, n={n} | "
            f"{pearson_str}"
        )

    print("\n" + "=" * 80)
    print("PRICE vs VOLUME relationship")
    print("=" * 80)

    _one_report("Overall", base)

    if by_year and "Year" in base.columns:
        for yr, g in base.groupby("Year", sort=True):
            _one_report(f"Year={int(yr)}", g)

    if by_country and "CountryCode" in base.columns:
        for cc, g in base.groupby("CountryCode", sort=True):
            _one_report(f"Country={cc}", g)

    if by_year and by_country and ("Year" in base.columns) and ("CountryCode" in base.columns):
        for (yr, cc), g in base.groupby(["Year", "CountryCode"], sort=True):
            _one_report(f"Year={int(yr)} | Country={cc}", g)


# -------------------- SIGNIFICANCE --------------------
def ols_tests_all_dependence_modes(
    df: pd.DataFrame,
    y_col: str,
    group_col: str,
    *,
    transform: Optional[str] = None,
    ref_level=None,
    cluster_fei_col: str = "FEI",
    cluster_ndc_col: str = "NDC11",
    label: str = "",
) -> dict:

    d = df.copy()

    # ---- outcome transform ----
    y = d[y_col]
    if transform == "log1p":
        d["_y"] = log1p_safe(y)
    elif transform == "log":
        d["_y"] = log_safe_pos(y)
    else:
        d["_y"] = pd.to_numeric(y, errors="coerce")

    d = d.dropna(subset=["_y", group_col]).copy()

    if cluster_fei_col in d.columns:
        d[cluster_fei_col] = d[cluster_fei_col].astype(str)
    if cluster_ndc_col in d.columns:
        d[cluster_ndc_col] = d[cluster_ndc_col].astype(str)

    # set reference category
    if ref_level is not None:
        d[group_col] = pd.Categorical(d[group_col])
        if ref_level in list(d[group_col].cat.categories):
            cats = [ref_level] + [c for c in d[group_col].cat.categories if c != ref_level]
            d[group_col] = d[group_col].cat.reorder_categories(cats, ordered=False)

    formula = f"_y ~ C({group_col})"

    # ---- IID ----
    m_iid = smf.ols(formula, data=d).fit()

    # ---- helper: pairwise vs ref from statsmodels result ----
    def _pairwise_vs_ref(model) -> pd.DataFrame:
        params = model.params
        pvals = model.pvalues
        term_prefix = f"C({group_col})"
        term_names = [t for t in params.index if t.startswith(term_prefix)]
        if len(term_names) == 0:
            return pd.DataFrame()
        raw_p = pvals.loc[term_names].values
        rej, p_holm, _, _ = multipletests(raw_p, method="holm")
        return pd.DataFrame({
            "contrast_vs_ref": term_names,
            "coef": params.loc[term_names].values,
            "p_raw": raw_p,
            "p_holm": p_holm,
            "reject_0.05_holm": rej,
        })

    # ---- helper: pairwise vs ref from two-way dict ----
    def _pairwise_2way(tw: dict) -> pd.DataFrame:
        if tw is None:
            return pd.DataFrame()
        ct = tw["coef_table"]
        term_prefix = f"C({group_col})"
        rows = ct[ct.index.str.startswith(term_prefix)]
        if rows.empty:
            return pd.DataFrame()
        raw_p = rows["P>|z|"].values
        rej, p_holm, _, _ = multipletests(raw_p, method="holm")
        return pd.DataFrame({
            "contrast_vs_ref": rows.index.tolist(),
            "coef": rows["coef"].values,
            "p_raw": raw_p,
            "p_holm": p_holm,
            "reject_0.05_holm": rej,
        })

    # ---- helper: overall group test ----
    def _overall_test(model) -> str:
        try:
            w = model.wald_test_terms(skip_single=False)
            if f"C({group_col})" in w.summary_frame().index:
                row = w.summary_frame().loc[f"C({group_col})"]
                return f"chi2={float(row['statistic']):.4g}, p={float(row['pvalue']):.4g}, df={int(row['df_constraint'])}"
            return str(w)
        except Exception as e:
            return f"(overall test unavailable: {e})"

    # ---- cluster by FEI ----
    m_fei = None
    if cluster_fei_col in d.columns and d[cluster_fei_col].nunique() >= 2:
        m_fei = smf.ols(formula, data=d).fit(
            cov_type="cluster",
            cov_kwds={"groups": d[cluster_fei_col]},
        )

    # ---- cluster by NDC11 ----
    m_ndc = None
    if cluster_ndc_col in d.columns and d[cluster_ndc_col].nunique() >= 2:
        m_ndc = smf.ols(formula, data=d).fit(
            cov_type="cluster",
            cov_kwds={"groups": d[cluster_ndc_col]},
        )

    # ---- two-way cluster FEI + NDC11 ----
    two_way = None
    if (
        cluster_fei_col in d.columns and cluster_ndc_col in d.columns
        and d[cluster_fei_col].nunique() >= 2 and d[cluster_ndc_col].nunique() >= 2
    ):
        try:
            base = smf.ols(formula, data=d).fit()

            fei_codes = pd.Categorical(d[cluster_fei_col]).codes.astype(np.int64)
            ndc_codes = pd.Categorical(d[cluster_ndc_col]).codes.astype(np.int64)

            cov_result = cov_cluster_2groups(base, fei_codes, ndc_codes)

            # cov_cluster_2groups returns a 3-tuple (cov_2way, cov_1, cov_2)
            cov = cov_result[0] if isinstance(cov_result, tuple) else cov_result
            cov = np.asarray(cov)

            if cov.ndim != 2:
                raise ValueError(f"Two-way cov has unexpected shape: {cov.shape}")

            params = base.params.copy()
            se = pd.Series(np.sqrt(np.diag(cov)), index=params.index)
            z = params / se
            p = 2 * (1 - norm.cdf(np.abs(z)))
            ci_low  = params - 1.96 * se
            ci_high = params + 1.96 * se

            coef_table = pd.DataFrame({
                "coef":    params,
                "std err": se,
                "z":       z,
                "P>|z|":   p,
                "[0.025":  ci_low,
                "0.975]":  ci_high,
            })

            two_way = {"base": base, "cov": cov, "coef_table": coef_table}

        except Exception as e:
            print(f"[WARNING] Two-way clustering failed: {e}. Skipping two-way cluster results.")
            two_way = None

    # ---- print block ----
    print("\n" + "=" * 100)
    print(f"{label} | y={y_col} | transform={transform} | group={group_col}")
    print(f"n_rows={len(d):,}")
    if cluster_fei_col in d.columns:
        print(f"n_FEI={d[cluster_fei_col].nunique():,}")
    if cluster_ndc_col in d.columns:
        print(f"n_NDC11={d[cluster_ndc_col].nunique():,}")
    print("=" * 100)

    def _print_model(name: str, model):
        if model is None:
            print(f"\n[{name}] not available (missing clusters, columns, or failed).")
            return

        print("\n" + "-" * 100)
        print(f"[{name}]")
        print("-" * 100)

        # statsmodels result object
        if hasattr(model, "summary"):
            print(model.summary().tables[1])
            print("\nOverall group test:", _overall_test(model))
            pw = _pairwise_vs_ref(model)
            if pw.empty:
                print("\nPairwise vs reference: (none)")
            else:
                print("\nPairwise vs reference (Holm-adjusted):")
                print(pw.to_string(index=False))
            return

        # dict result (two-way)
        if isinstance(model, dict) and "coef_table" in model:
            print(model["coef_table"].to_string())

            # overall Wald test using two-way cov
            try:
                params  = model["base"].params
                cov     = model["cov"]
                term_prefix = f"C({group_col})"
                term_names  = [t for t in params.index if t.startswith(term_prefix)]
                if not term_names:
                    print("\nOverall group test: (no group terms)")
                else:
                    idx = [list(params.index).index(t) for t in term_names]
                    b   = params.values[idx]
                    V   = cov[np.ix_(idx, idx)]
                    chi2 = float(b.T @ np.linalg.inv(V) @ b)
                    df_  = len(idx)
                    pval = 1 - stats.chi2.cdf(chi2, df_)
                    print(f"\nOverall group test (two-way clustered Wald): chi2={chi2:.4g}, p={pval:.4g}, df={df_}")
            except Exception as e:
                print(f"\nOverall group test failed: {e}")

            # pairwise using two-way cov p-values
            pw = _pairwise_2way(model)
            if pw.empty:
                print("\nPairwise vs reference: (none)")
            else:
                print("\nPairwise vs reference (Holm-adjusted, two-way clustered SEs):")
                print(pw.to_string(index=False))
            return

        print("\n(Unknown model object type)")

    _print_model("IID (assume independent NDC-year tests)", m_iid)
    _print_model("Cluster by FEI (facility)", m_fei)
    _print_model("Cluster by NDC11 (same product over years)", m_ndc)
    _print_model("Two-way cluster FEI + NDC11", two_way)

    return {
        "data_used":     d,
        "m_iid":         m_iid,
        "m_fei":         m_fei,
        "m_ndc":         m_ndc,
        "m_2way":        two_way,
        "pairwise_iid":  _pairwise_vs_ref(m_iid),
        "pairwise_fei":  _pairwise_vs_ref(m_fei)  if m_fei    is not None else None,
        "pairwise_ndc":  _pairwise_vs_ref(m_ndc)  if m_ndc    is not None else None,
        "pairwise_2way": _pairwise_2way(two_way)   if two_way  is not None else None,
    }


def spearman_with_dependence_sensitivity(df, x_col, y_col, *, label="", collapse_by_ndc=True):
    d = df[[x_col, y_col, "NDC11"]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col]).copy()

    # IID Spearman (assumes each row independent)
    rho_iid, p_iid = spearmanr(d[x_col].values, d[y_col].values)

    print("\n" + "=" * 90)
    print(label)
    print(f"IID Spearman (assumes independent rows): rho={rho_iid:+.3f}, p={p_iid:.3e}, n={len(d):,}")

    if collapse_by_ndc:
        # collapse to NDC level (e.g., mean across years)
        g = d.groupby("NDC11", as_index=False).agg(
            x=(x_col, "mean"),
            y=(y_col, "mean"),
            n_years=("NDC11", "size"),
        )
        rho_c, p_c = spearmanr(g["x"].values, g["y"].values)
        print(f"Collapsed-by-NDC Spearman (1 row per NDC11): rho={rho_c:+.3f}, p={p_c:.3e}, n={len(g):,}")
        print("Note: collapsing reduces dependence but changes estimand (NDC-average association).")

    print("=" * 90)

# -------------------- PLOTS --------------------
def plot_obs1_country_bars(test_df: pd.DataFrame) -> None:
    """Observation 1: averages by country across ALL test records (bar plot uses means)."""
    code_order = ["IND", "CHN", "USA"]
    code_to_name = {"IND": "India", "CHN": "China", "USA": "USA"}

    d = test_df[test_df["CountryCode"].isin(code_order)].copy()

    g = (
        d.groupby("CountryCode", as_index=False)
        .agg(
            DMF=("DMF", "mean"),
            NDMA=("NDMA", "mean"),
            Dissolution=("Dissolution", "mean"),
            n_DMF=("DMF", lambda s: int(s.notna().sum())),
            n_NDMA=("NDMA", lambda s: int(s.notna().sum())),
            n_Dissolution=("Dissolution", lambda s: int(s.notna().sum())),
        )
    )

    g["CountryCode"] = pd.Categorical(g["CountryCode"], categories=code_order, ordered=True)
    g = g.sort_values("CountryCode").reset_index(drop=True)

    bar_color = "#93c5fd"
    edge_color = "#2563eb"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)

    metrics = [
        ("DMF", "DMF (ng/day)", "{:,.0f}", "n_DMF"),
        ("NDMA", "NDMA (ng/day)", "{:,.1f}", "n_NDMA"),
        ("Dissolution", "Difference Factor", "{:,.2f}", "n_Dissolution"),
    ]

    x = np.arange(len(g))
    xlabels = [code_to_name[str(c)] for c in g["CountryCode"].astype(str).tolist()]

    for ax, (col, title, fmt, ncol) in zip(axes, metrics):
        vals = g[col].values
        nvals = g[ncol].values

        bars = ax.bar(x, vals, color=bar_color, edgecolor=edge_color, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Country")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

        finite_vals = vals[np.isfinite(vals)]
        ymax = float(np.nanmax(finite_vals)) if len(finite_vals) else 1.0
        y_top = ymax * 1.12
        y_pad = max(y_top * 0.10, 0.05) if col != "Dissolution" else max(y_top * 0.18, 0.015)
        ax.set_ylim(-y_pad, y_top)

        for rect, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt.format(val),
                    ha="center", va="bottom", fontsize=10)

        y_n = -y_pad * 0.55
        for xi, n in zip(x, nvals):
            ax.text(xi, y_n, f"n={int(n)}", ha="center", va="center", fontsize=9, color="#374151")

    # fig.suptitle("Observation 1: Quality by country (NDC Level)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    plt.show()


def plot_obs2_volume_price_boxes_with_country_jitter(test_with_prior: pd.DataFrame) -> None:
    """
    Observation 2 (UPDATED):
      - Uses ALL TEST RECORDS (dots = tests, not unique NDCs)
      - X = PriorScore (most recent inspection BEFORE each test date)
      - Y = price and volume
      - n = count of plotted TEST ROWS per score (not nunique NDC)
    """
    d = test_with_prior.copy()
    d = d[d["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()

    d["price"] = pd.to_numeric(d.get("price"), errors="coerce")
    d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce")

    fig, (ax_price, ax_vol) = plt.subplots(1, 2, figsize=(12, 4))
    # fig.suptitle("Observation 2: Market Response to Prior Inspection Scores (NDC Level)",
    #              fontsize=14, fontweight="bold")

    def prepare_score_x_mapping(dd: pd.DataFrame, score_col: str):
        present = [s for s in SCORE_ORDER if (dd[score_col] == s).any()]
        mapping = {s: i for i, s in enumerate(present, start=1)}
        return present, mapping

    def add_n_linear_space(ax, present_scores, n_by_score):
        y0, y1 = ax.get_ylim()
        y1 = max(y1, 0.0)
        y_top = y1 * 1.05 if y1 != 0 else 1.0
        y_pad = max(y_top * 0.12, 0.05)
        ax.set_ylim(-y_pad, y_top)
        y_text = -0.55 * y_pad
        for i, s in enumerate(present_scores, start=1):
            ax.text(i, y_text, f"n={int(n_by_score.get(s, 0))}",
                    ha="center", va="center", fontsize=9, color="#374151")

    def add_n_log_space(ax, present_scores, n_by_score):
        y0, y1 = ax.get_ylim()
        if not np.isfinite(y0) or y0 <= 0:
            y0 = 1.0
        if not np.isfinite(y1) or y1 <= y0:
            y1 = y0 * 10
        y_floor = y0 / 30.0
        ax.set_ylim(y_floor, y1)
        y_text = y_floor * 1.8
        for i, s in enumerate(present_scores, start=1):
            ax.text(i, y_text, f"n={int(n_by_score.get(s, 0))}",
                    ha="center", va="center", fontsize=9, color="#374151")

    # ---------- PRICE ----------
    price_df = d[d["PriorScore"].notna() & d["price"].notna()].copy()
    if not price_df.empty:
        present_scores, score_to_x = prepare_score_x_mapping(price_df, "PriorScore")
        box_data = [price_df.loc[price_df["PriorScore"] == s, "price"].values for s in present_scores]

        ax_price.boxplot(
            box_data,
            positions=list(range(1, len(present_scores) + 1)),
            widths=0.35,
            showfliers=False,
            medianprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        for code in COUNTRY_CODE_ORDER:
            d_cc = price_df[price_df["CountryCode"] == code].copy()
            if d_cc.empty:
                continue
            d_cc["xcat"] = d_cc["PriorScore"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", "price", radius=0.02)
            ax_price.scatter(
                d_cc["xcat"] + d_cc["jx"], d_cc["price"],
                s=25, alpha=0.7, c=country_code_colors.get(code, "#6b7280"),
                edgecolor="none",
            )

        ax_price.set_xlabel("Prior Inspection Outcome", fontsize=11)
        ax_price.set_ylabel("Price per Unit ($)", fontsize=11)
        ax_price.set_title("Price vs Prior Inspection Outcome", fontsize=11, fontweight="bold")
        ax_price.set_xticks(range(1, len(present_scores) + 1))
        ax_price.set_xticklabels([SCORE_LABELS.get(float(s), str(s)) for s in present_scores])
        ax_price.grid(True, axis="y", alpha=0.3)

        n_by_score_price = price_df.groupby("PriorScore").size().to_dict()
        add_n_linear_space(ax_price, present_scores, n_by_score_price)

    # ---------- VOLUME ----------
    vol_df = d[d["PriorScore"].notna() & d["volume"].notna() & (d["volume"] > 0)].copy()
    if not vol_df.empty:
        present_scores, score_to_x = prepare_score_x_mapping(vol_df, "PriorScore")
        box_data = [vol_df.loc[vol_df["PriorScore"] == s, "volume"].values for s in present_scores]

        ax_vol.boxplot(
            box_data,
            positions=list(range(1, len(present_scores) + 1)),
            widths=0.35,
            showfliers=False,
            medianprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        for code in COUNTRY_CODE_ORDER:
            d_cc = vol_df[vol_df["CountryCode"] == code].copy()
            if d_cc.empty:
                continue
            d_cc["xcat"] = d_cc["PriorScore"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", "volume", radius=0.02)
            ax_vol.scatter(
                d_cc["xcat"] + d_cc["jx"], d_cc["volume"],
                s=25, alpha=0.7, c=country_code_colors.get(code, "#6b7280"),
                edgecolor="none",
            )

        ax_vol.set_xlabel("Prior Inspection Outcome", fontsize=11)
        ax_vol.set_ylabel("Market Volume (Extended Units)", fontsize=11)
        ax_vol.set_yscale("log")
        ax_vol.set_title("Market Volume vs Prior Inspection Outcome", fontsize=11, fontweight="bold")
        ax_vol.set_xticks(range(1, len(present_scores) + 1))
        ax_vol.set_xticklabels([SCORE_LABELS.get(float(s), str(s)) for s in present_scores])
        ax_vol.grid(True, which="major", axis="both", alpha=0.3)
        ax_vol.grid(False, which="minor")

        n_by_score_vol = vol_df.groupby("PriorScore").size().to_dict()
        add_n_log_space(ax_vol, present_scores, n_by_score_vol)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=country_code_colors.get(code, "#6b7280"), label=code)
        for code in COUNTRY_CODE_ORDER
    ]
    fig.legend(handles=legend_handles, title="CountryCode",
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()

def print_fig2_summary_stats(
    df: pd.DataFrame,
    score_col: str = "PriorScore",
    price_col: str = "price",
    vol_col: str = "volume",
) -> None:
    """
    Prints mean/median by PriorScore category for the same inclusion rules used in Figure 2:
      - price panel: PriorScore notna AND price notna
      - volume panel: PriorScore notna AND volume notna AND volume > 0
    """

    def _summ(dd: pd.DataFrame, val_col: str) -> pd.DataFrame:
        out = (
            dd.groupby(score_col)[val_col]
              .agg(
                  n="count",
                  mean="mean",
                  median="median",
                  p25=lambda s: s.quantile(0.25),
                  p75=lambda s: s.quantile(0.75),
              )
              .reindex(SCORE_ORDER)
              .reset_index()
        )
        out["Outcome"] = out[score_col].map(SCORE_LABELS)
        return out[["Outcome", score_col, "n", "mean", "median", "p25", "p75"]]

    d = df.copy()
    d[score_col] = pd.to_numeric(d.get(score_col), errors="coerce")
    d[price_col] = pd.to_numeric(d.get(price_col), errors="coerce")
    d[vol_col] = pd.to_numeric(d.get(vol_col), errors="coerce")

    price_df = d[d[score_col].notna() & d[price_col].notna()].copy()
    vol_df   = d[d[score_col].notna() & d[vol_col].notna() & (d[vol_col] > 0)].copy()

    print("\n" + "=" * 90)
    print("Figure 2 summary stats (by Prior Inspection Outcome)")
    print("=" * 90)

    if price_df.empty:
        print("PRICE: no data after filters")
    else:
        s_price = _summ(price_df, price_col)
        print("\nPRICE per unit ($):")
        print(s_price.to_string(index=False, float_format=lambda x: f"{x:,.6g}"))

    if vol_df.empty:
        print("\nVOLUME: no data after filters")
    else:
        s_vol = _summ(vol_df, vol_col)
        print("\nMARKET VOLUME (Extended Units):")
        print(s_vol.to_string(index=False, float_format=lambda x: f"{x:,.6g}"))

    print("=" * 90 + "\n")


def plot_obs3_scatter_volume_price_vs_quality(test_df: pd.DataFrame) -> None:
    """
    Observation 3 (UPDATED):
      - Uses ALL TEST RECORDS (dots = tests)
      - Difference Factor NA stays NA (not plotted)
      - Spearman + p-value computed on EXACTLY the plotted sample
    """
    d = test_df[test_df["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()

    def add_trend_and_corr(ax, x: np.ndarray, y: np.ndarray, xscale: str, linthresh: float = 1.0):
        x = x.astype(float)
        y = y.astype(float)

        mask_plot = np.isfinite(x) & np.isfinite(y) & (y > 0)
        xf = x[mask_plot]
        yf = y[mask_plot]
        n_corr = int(len(xf))

        if n_corr <= 2:
            ax.text(
                0.02, 0.98,
                f"n={n_corr}\nCorrelation: n<3",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.50),
            )
            return

        if xscale == "symlog":
            def T(u):
                u = np.asarray(u, dtype=float)
                out = u.copy()
                big = u >= linthresh
                out[big] = linthresh * (1.0 + np.log10(u[big] / linthresh))
                return out

            def Tinv(t):
                t = np.asarray(t, dtype=float)
                out = t.copy()
                big = t >= linthresh
                out[big] = linthresh * (10 ** (t[big] / linthresh - 1.0))
                return out

            Xfit = T(xf)
        else:
            def T(u): return np.asarray(u, dtype=float)
            def Tinv(t): return np.asarray(t, dtype=float)
            Xfit = xf

        try:
            Yfit = np.log10(yf)
            b, a = np.polyfit(Xfit, Yfit, 1)

            x_line_fit = np.linspace(np.nanmin(Xfit), np.nanmax(Xfit), 200)
            y_line = 10 ** (a + b * x_line_fit)
            x_line = Tinv(x_line_fit)

            ax.plot(x_line, y_line, "r--", alpha=0.55, linewidth=2)

            rho, pval = spearmanr(xf, yf)
            ax.text(
                0.02, 0.98,
                f"n={n_corr}\nCorrelation: ρ={rho:+.3f}\np={pval:.3e}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.50),
            )
        except Exception:
            ax.text(
                0.02, 0.98,
                f"n={n_corr}\nCorrelation: error",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.50),
            )

    def one_figure(ycol: str, ylabel: str, title: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        xcols = ["DMF", "NDMA", "Dissolution"]
        xlabels = ["DMF (ng/day)", "NDMA (ng/day)", "Difference Factor"]

        for ax, xcol, xlabel in zip(axes, xcols, xlabels):
            dd = d.dropna(subset=[xcol, ycol]).copy()

            for code in COUNTRY_CODE_ORDER:
                d_cc = dd[dd["CountryCode"] == code]
                if d_cc.empty:
                    continue
                ax.scatter(
                    d_cc[xcol].values,
                    d_cc[ycol].values,
                    s=55,
                    alpha=0.65,
                    c=country_code_colors.get(code, "#6b7280"),
                    edgecolor="white",
                    linewidth=0.4,
                )

            xscale = "linear"
            linthresh = 1.0
            if xcol in ["DMF", "NDMA"]:
                ax.set_xscale("symlog", linthresh=linthresh)
                xscale = "symlog"

            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            add_trend_and_corr(
                ax,
                dd[xcol].values.astype(float),
                dd[ycol].values.astype(float),
                xscale=xscale,
                linthresh=linthresh,
            )

        legend_handles = [
            Line2D(
                [0], [0],
                marker="o", linestyle="",
                color=country_code_colors.get(code, "#6b7280"),
                label={"IND": "India", "USA": "United States of America", "CHN": "China"}.get(code, code),
                markeredgecolor="white", markeredgewidth=0.5, markersize=8,
            )
            for code in COUNTRY_CODE_ORDER
        ]
        fig.legend(handles=legend_handles, title="Country",
                   loc="lower center", bbox_to_anchor=(0.5, -0.02),
                   ncol=3, framealpha=0.9)

        plt.tight_layout(rect=[0, 0.06, 1, 0.92])
        plt.show()

    one_figure(
        ycol="volume",
        ylabel="Market Volume (Extended Units)",
        title = "",
        #title="Observation 3: Market Volume vs Tested Quality (NDC Level)",
    )
    one_figure(
        ycol="price",
        ylabel="Price per Unit ($)",
        title="",
        #title="Observation 3: Price vs Tested Quality (NDC Level)",
    )


# -------------------- LOAD DATA --------------------
df = pd.read_excel(DF_FILE)
print(f"Loaded: {DF_FILE.name} | rows={len(df):,} | cols={len(df.columns)}")

# -------------------- BUILD NDC–YEAR TABLE (dedup inspections) --------------------
ndc_year_df = build_ndc_year_table(df)

print(f"NDC–Year rows (tests): {len(ndc_year_df):,}")
print("Unique NDC11:", int(ndc_year_df["NDC11"].nunique()))
print("Avg event-rows collapsed per NDC–Year:", float(ndc_year_df["n_event_rows"].mean()))

# -------------------- FILTER TO PLOTTED COUNTRIES ONLY --------------------
# We only want countries that appear in the manuscript figures (remove CAN/BGD, etc.)
PLOT_COUNTRIES = ["IND", "CHN", "USA"]

ndc_year_df = ndc_year_df[ndc_year_df["CountryCode"].isin(PLOT_COUNTRIES)].copy()

print("After country filter:", ndc_year_df["CountryCode"].value_counts(dropna=False).to_dict())
print("Rows:", len(ndc_year_df), "| Unique FEI:", ndc_year_df["FEI"].astype(str).nunique())

# -------------------- QUICK CHECK: PRICE vs VOLUME --------------------
report_price_volume_relationship(
    ndc_year_df,
    by_year=True,     # gives 2020/2022/2024 separately
    by_country=True,  # gives IND/USA/CHN separately
)



# -------------------- CALL PLOTS --------------------
plot_obs1_country_bars(ndc_year_df)  # uses DMF/NDMA/Dissolution means
plot_obs2_volume_price_boxes_with_country_jitter(ndc_year_df)  # now expects PriorScore already in df
print_fig2_summary_stats(ndc_year_df)
plot_obs3_scatter_volume_price_vs_quality(ndc_year_df)


# Figure 1: Country differences
res_fig1_dmf = ols_tests_all_dependence_modes(
    ndc_year_df, y_col="DMF", group_col="CountryCode",
    transform="log1p", ref_level="USA",
    label="Figure 1 panel: DMF by country"
)

res_fig1_ndma = ols_tests_all_dependence_modes(
    ndc_year_df, y_col="NDMA", group_col="CountryCode",
    transform="log1p", ref_level="USA",
    label="Figure 1 panel: NDMA by country"
)

res_fig1_diss = ols_tests_all_dependence_modes(
    ndc_year_df, y_col="Dissolution", group_col="CountryCode",
    transform=None, ref_level="USA",
    label="Figure 1 panel: Difference factor by country"
)

# Figure 2: Prior inspection outcome differences
res_fig2_price = ols_tests_all_dependence_modes(
    ndc_year_df, y_col="price", group_col="PriorScore_cat",
    transform="log", ref_level=0.0,    # NAI baseline
    label="Figure 2 panel: Price by prior inspection outcome"
)

res_fig2_vol = ols_tests_all_dependence_modes(
    ndc_year_df[ndc_year_df["volume"].notna() & (ndc_year_df["volume"] > 0)].copy(),
    y_col="volume", group_col="PriorScore_cat",
    transform="log", ref_level=0.0,
    label="Figure 2 panel: Volume by prior inspection outcome"
)

# Figure 3: price vs DMF, NDMA, Dissolution Difference
spearman_with_dependence_sensitivity(
    ndc_year_df, x_col="DMF", y_col="price",
    label="Fig 3 sensitivity: price vs DMF"
)

spearman_with_dependence_sensitivity(
    ndc_year_df, x_col="DMF", y_col="price",
    label="Fig 3 sensitivity: price vs NDMA"
)

spearman_with_dependence_sensitivity(
    ndc_year_df, x_col="DMF", y_col="price",
    label="Fig 3 sensitivity: price vs Dissolution Difference"
)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------------------
# PURPOSE
# Given a list of warning-letter cases (FEI + warning-letter date), we:
#   1) pull all NDCs in the Q&A file for that FEI (Q&A NDC is 5-3-2 like 68180-337-07)
#   2) convert those NDCs to monthly-panel format (ndc11 digits: 68180033707)
#   3) filter the monthly IQVIA panel by those ndc11s and sum iqvia_extended_units by month
#   4) plot units over time and draw a vertical line at the warning-letter date
#
# This backs the claim: warning letters (public compliance signal) do not obviously change volume.
# --------------------------------------------------------------------------------------

# -------------------- INPUT: Warning-letter FEIs + dates --------------------
# (Edit/extend as you like; this is enough to run the plotting pipeline.)
WARNING_LETTER_CASES = [
    {
        "label": "Lupin Ltd. (FEI 3004819820) — WL 2017-11-06",
        "fei": "3004819820",
        "wl_date": "2017-11-06",
        "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lupin-limited-535014-11062017",
    },
    {
        "label": "Aurobindo Pharma Limited Unit XI (FEI 3004611182) — WL 2019-06-20",
        "fei": "3004611182",
        "wl_date": "2019-06-20",
        "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/aurobindo-pharma-limited-577033-06202019",
    },
    {
        "label": "Zydus / Cadila (FEI 3002984011) — WL 2019-10-29",
        "fei": "3002984011",
        "wl_date": "2019-10-29",
        "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/cadila-healthcare-limited-584856-10292019",
    },
    {
        "label": "Novel Laboratories dba Lupin (FEI 3006271438) — WL 2021-06-11",
        "fei": "3006271438",
        "wl_date": "2021-06-11",
        "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/novel-laboratories-inc-dba-lupin-613385-06112021",
    },
    {
        "label": "Lupin Limited Tarapur (FEI 3002807512) — WL 2022-09-27",
        "fei": "3002807512",
        "wl_date": "2022-09-27",
        "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lupin-limited-633703-09272022",
    },
    {
        "label": "Sun Pharma (FEI 3002809586) — WL 2024-06-18",
        "fei": "3002809586",
        "wl_date": "2024-06-18",
        "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sun-pharmaceutical-industries-limited-677337-06182024",
    },
]

# -------------------- HELPERS --------------------
# --- assumes df (Q&A) is ALREADY LOADED in memory exactly like you said ---
# df = pd.read_excel(DF_FILE)
# --- load monthly panel once ---
DATA_ROOT = "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data"
MONTHLY_STEM = "2025-12-18-iqvia_with_sdud_nadac.cleaned"  # in processed folder

monthly_path = (
    Path(DATA_ROOT)
    / "04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)"
    / "processed"
    / f"{MONTHLY_STEM}.csv"
)
# if it's not .csv, switch extension to .parquet or .xlsx
if not monthly_path.exists():
    # try a few common options
    for ext in [".parquet", ".pq", ".xlsx", ".csv.gz", ""]:
        p = monthly_path.with_suffix(ext) if ext else monthly_path.parent / MONTHLY_STEM
        if p.exists():
            monthly_path = p
            break

if monthly_path.suffix.lower() in [".parquet", ".pq"]:
    dfm = pd.read_parquet(monthly_path)
elif monthly_path.suffix.lower() in [".xlsx"]:
    dfm = pd.read_excel(monthly_path)
else:
    dfm = pd.read_csv(monthly_path)

print("Loaded monthly:", monthly_path)
print("Monthly cols:", list(dfm.columns))


# --- clean monthly minimal ---
dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
dfm = dfm.dropna(subset=["date"]).copy()
dfm["ndc11"] = dfm["ndc11"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
dfm["iqvia_extended_units"] = pd.to_numeric(dfm["iqvia_extended_units"], errors="coerce")

# --- NDC 5-3-2 -> ndc11 digits (pad middle to 4) ---
def ndc_532_to_ndc11(s):
    # expects '68180-337-07'
    if pd.isna(s):
        return None
    parts = str(s).strip().split("-")
    if len(parts) != 3:
        return None
    a, b, c = parts
    a = "".join(ch for ch in a if ch.isdigit()).zfill(5)
    b = "".join(ch for ch in b if ch.isdigit()).zfill(4)  
    c = "".join(ch for ch in c if ch.isdigit()).zfill(2)
    out = a + b + c
    return out if len(out) == 11 else None


# --- main loop: for each FEI -> get NDCs from Q&A -> plot monthly IQVIA units ---
for case in WARNING_LETTER_CASES:
    fei = str(case["fei"])
    wl_date = pd.Timestamp(case["wl_date"])
    label = case.get("label", fei)

    # 1) get all NDCs for FEI from Q&A (your df has columns FEI + NDC)
    dd = df.copy()
    dd["FEI"] = dd["FEI"].astype(str)
    ndcs_532 = dd.loc[dd["FEI"] == fei, "NDC"].dropna().astype(str).str.strip().unique().tolist()
    ndcs_532 = sorted(set(ndcs_532))

    if len(ndcs_532) == 0:
        print(f"[SKIP] No NDCs in Q&A for FEI={fei} ({label})")
        continue

    # 2) convert to ndc11 list
    ndc11_list = [ndc_532_to_ndc11(x) for x in ndcs_532]
    ndc11_list = sorted({x for x in ndc11_list if isinstance(x, str) and len(x) == 11})

    if len(ndc11_list) == 0:
        print(f"[SKIP] Could not convert NDCs to ndc11 for FEI={fei} ({label})")
        continue

    # 3) filter monthly and aggregate
    m = dfm[dfm["ndc11"].isin(ndc11_list)].copy()
    if m.empty:
        print(f"[SKIP] Monthly has no matches for FEI={fei} ({label}) | ndc11_count={len(ndc11_list)}")
        continue

    series = (
        m.groupby("date", as_index=False)
         .agg(units=("iqvia_extended_units", "sum"),
              n_rows=("date", "size"),
              n_ndc11=("ndc11", "nunique"))
         .sort_values("date")
         .reset_index(drop=True)
    )

    # 4) plot (window +-24 months around WL for readability)
    start = wl_date - pd.DateOffset(months=24)
    end = wl_date + pd.DateOffset(months=24)
    sw = series[(series["date"] >= start) & (series["date"] <= end)].copy()
    if sw.empty:
        sw = series.copy()

    plt.figure(figsize=(12, 4))
    plt.plot(sw["date"], sw["units"], marker="o", linewidth=2)
    plt.axvline(wl_date, linestyle="--", linewidth=2)

    plt.title(f"IQVIA extended units (sum over FEI-linked NDCs)\n{label} | FEI={fei} | NDCs={len(ndcs_532)} | ndc11={len(ndc11_list)}")
    plt.xlabel("Month")
    plt.ylabel("Extended Units")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"{label} | FEI={fei} | WL={wl_date.date()} | NDCs(Q&A)={len(ndcs_532)} | ndc11={len(ndc11_list)} | monthly_rows={len(m):,}")

# %%
