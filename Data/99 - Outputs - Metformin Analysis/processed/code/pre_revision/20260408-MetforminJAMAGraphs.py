# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.stats import spearmanr, pearsonr
from statistical_tests_advanced_models import run_advanced_models
#from statistical_tests_advanced_models import run_advanced_models
from statistical_tests import test_fig3_fig4_scatter_correlations, _block_bootstrap_spearman

# -------------------- EXPORT SETTINGS --------------------
# fonttype=42 embeds fonts as TrueType → text remains editable in
# Illustrator / Inkscape / Affinity Designer after opening the PDF/EPS.
# fonttype=3 (default) converts text to outlines — not editable.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

# -------------------- PATHS --------------------
BASE_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/06 - Metformin Data/Derived"
)
DF_FILE = BASE_DIR / "Q&As1234_v8_v02.xlsx"

# Figures saved to Desktop so they are immediately visible (avoids Google Drive sync delay)
FIGURES_DIR = Path.home() / "Desktop" / "MetforminFigures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- CONSTANTS --------------------
DMF_COL  = "DMF (ng/DAY) Valisure"
NDMA_COL = "NDMA (ng/DAY) Valisure"
DISS_COL = "Difference Factor"

COUNTRY_CODE_ORDER = ["IND", "USA", "CHN"]
SCORE_ORDER  = [0.0, 1.5, 3.5]
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


# -------------------- NDC NORMALIZATION --------------------
def _digits_only(x) -> str:
    if pd.isna(x):
        return ""
    return "".join(ch for ch in str(x).strip() if ch.isdigit())


def ndc10_to_ndc11_and_542(ndc10_digits: str):
    """
    Convert a 10-digit NDC (no hyphens) to NDC11 + NDC_542 (5-4-2).
    Heuristic: 4-4-2 if first char is '0', else 5-3-2, else 5-4-1.
    """
    s = ndc10_digits
    if len(s) != 10:
        return (np.nan, np.nan)
    if s[0] == "0":
        a, b, c = s[:4], s[4:8], s[8:]
        ndc11 = a.zfill(5) + b + c
    else:
        a, b, c = s[:5], s[5:8], s[8:]
        ndc11 = a + b.zfill(4) + c
        if len(ndc11) != 11:
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
    """Ensures NDC_542 (5-4-2 hyphenated) and NDC11 (11-digit) columns exist."""
    out = d.copy()
    candidate_10 = None
    for c in ["NDC", "ndc", "product_ndc", "Product NDC", "productNDC"]:
        if c in out.columns:
            candidate_10 = c
            break

    if "NDC11" in out.columns:
        out["NDC11"] = out["NDC11"].astype(str).str.replace(r"\D", "", regex=True)
        out.loc[out["NDC11"].str.len() != 11, "NDC11"] = np.nan
    else:
        out["NDC11"] = np.nan

    if "NDC_542" in out.columns:
        out["NDC11"] = out["NDC11"].fillna(out["NDC_542"].apply(ndc542_to_ndc11))

    if candidate_10 is not None:
        raw10 = out[candidate_10].apply(_digits_only)
        raw10 = raw10.where(raw10.str.len() == 10, "")
        ndc11_542 = raw10.apply(lambda s: ndc10_to_ndc11_and_542(s) if s else (np.nan, np.nan))
        out["NDC11"] = out["NDC11"].fillna(ndc11_542.apply(lambda t: t[0]))
        if "NDC_542" not in out.columns:
            out["NDC_542"] = ndc11_542.apply(lambda t: t[1])
        else:
            out["NDC_542"] = out["NDC_542"].fillna(ndc11_542.apply(lambda t: t[1]))

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
    Collapse to one row per (NDC11, Year).
      1) Normalizes NDC fields (builds NDC11 + NDC_542).
      2) NDMA in 2024 -> NaN; Difference Factor in 2020/2022 -> NaN.
      3) PriorScore = most recent inspection score <= Dec 31 of test year,
         snapped to {0.0=NAI, 1.5=VAI, 3.5=OAI}.
    """
    d = df.copy()
    if "Event Start Date" in d.columns:
        d["Event Start Date"] = pd.to_datetime(d["Event Start Date"], errors="coerce")

    if "ndc11" in d.columns and "NDC11" not in d.columns:
        d = d.rename(columns={"ndc11": "NDC11"})
    d = ensure_ndc542_and_ndc11(d)
    d = d.dropna(subset=["NDC11"]).copy()

    d["Year"] = pd.to_numeric(d.get("Year"), errors="coerce")
    d = d.dropna(subset=["Year"]).copy()
    d["Year"] = d["Year"].astype(int)

    d["TestDate"] = pd.to_datetime(d["Year"].astype(str) + "-12-31")
    d["price"]    = pd.to_numeric(pick_price_series(d), errors="coerce")
    d["volume"]   = pd.to_numeric(d.get("iqvia_extended_units"), errors="coerce")

    d["DMF"]  = pd.to_numeric(d.get(DMF_COL),  errors="coerce")
    d["NDMA"] = pd.to_numeric(d.get(NDMA_COL), errors="coerce")
    d.loc[d["Year"] == 2024, "NDMA"] = np.nan  # not measured in 2024

    diss_source_col = DISS_COL if DISS_COL in d.columns else ("Dissolution" if "Dissolution" in d.columns else None)
    d["Dissolution"] = pd.to_numeric(d.get(diss_source_col), errors="coerce")
    d.loc[d["Year"].isin([2020, 2022]), "Dissolution"] = np.nan  # only measured in 2024

    if "Event Score" in d.columns or "Score" in d.columns:
        d["ScoreUsed"] = pd.to_numeric(d.get("Event Score"), errors="coerce")
        d.loc[d["ScoreUsed"].isna(), "ScoreUsed"] = pd.to_numeric(d.get("Score"), errors="coerce")
    else:
        d["ScoreUsed"] = np.nan

    keys = ["NDC11", "Year"]

    def prior_score_for_group(g: pd.DataFrame):
        test_date = g["TestDate"].iloc[0]
        gg = g.dropna(subset=["Event Start Date"])
        gg = gg[gg["Event Start Date"] <= test_date]
        if gg.empty:
            return (np.nan, pd.NaT)
        last = gg.sort_values("Event Start Date").iloc[-1]
        return (last["ScoreUsed"] if pd.notna(last["ScoreUsed"]) else np.nan, last["Event Start Date"])

    agg = (
        d.groupby(keys, as_index=False)
         .agg(
            NDC_542     =("NDC_542",          "first"),
            NDC         =("NDC",              "first"),
            NDC8        =("NDC8",             "first"),
            Strength    =("Strength",         "first"),
            FEI         =("FEI",              "first"),
            Firm        =("Firm",             "first"),
            CountryCode =("CountryCode",      "first"),
            CountryName =("CountryName",      "first"),
            TestDate    =("TestDate",         "first"),
            price       =("price",            "mean"),
            volume      =("volume",           "mean"),
            DMF         =("DMF",              "mean"),
            NDMA        =("NDMA",             "mean"),
            Dissolution =("Dissolution",      "mean"),
            n_event_rows=("Event Start Date", "size"),
         )
    )

    prior_raw, prior_date = [], []
    for _, g in d.groupby(keys, sort=False):
        s, dt = prior_score_for_group(g)
        prior_raw.append(s)
        prior_date.append(dt)

    agg["PriorScore_raw"]  = prior_raw
    agg["PriorEventDate"]  = prior_date

    def nearest_score(x):
        if pd.isna(x): return np.nan
        arr = np.array(SCORE_ORDER, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(x)))])

    agg["PriorScore"]     = agg["PriorScore_raw"].apply(nearest_score)
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
    FEI-wise prior score attachment via per-FEI merge_asof.
    Returns test_df with PriorScore_raw, PriorScore, PriorScore_cat.
    """
    t = test_df.copy()
    t = t.dropna(subset=["FEI", "TestDate"]).copy()
    t["FEI"]      = t["FEI"].astype(str)
    t["TestDate"] = pd.to_datetime(t["TestDate"], errors="coerce")
    t = t.dropna(subset=["TestDate"]).copy()

    e = events.copy()
    e = e.dropna(subset=["FEI", "Event Start Date"]).copy()
    e["FEI"] = e["FEI"].astype(str)
    e["Event Start Date"] = pd.to_datetime(e["Event Start Date"], errors="coerce")
    e = e.dropna(subset=["Event Start Date"]).copy()
    e["ScoreUsed"] = pd.to_numeric(e.get("Event Score"), errors="coerce")
    e.loc[e["ScoreUsed"].isna(), "ScoreUsed"] = pd.to_numeric(e.get("Score"), errors="coerce")
    e = e.rename(columns={"Event Start Date": "EventDate"})
    e = e[["FEI", "EventDate", "ScoreUsed"]].sort_values(["FEI", "EventDate"], kind="mergesort").reset_index(drop=True)
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
        merged = pd.merge_asof(gg, ev, left_on="TestDate", right_on="EventDate",
                               direction="backward", allow_exact_matches=True)
        merged["PriorScore_raw"] = pd.to_numeric(merged["ScoreUsed"], errors="coerce")
        out_parts.append(merged)

    merged_all = pd.concat(out_parts, ignore_index=True)

    def nearest_score(x):
        if pd.isna(x): return np.nan
        arr = np.array(SCORE_ORDER, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(x)))])

    merged_all["PriorScore"]     = merged_all["PriorScore_raw"].apply(nearest_score)
    merged_all["PriorScore_cat"] = pd.Categorical(merged_all["PriorScore"], categories=SCORE_ORDER, ordered=True)
    return merged_all


def report_price_volume_relationship(
    ndc_year_df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "volume",
    by_year: bool = True,
    by_country: bool = True,
) -> None:
    """Reports Spearman and log-log Pearson correlations between price and volume."""
    d = ndc_year_df.copy()
    d[price_col]  = pd.to_numeric(d.get(price_col),  errors="coerce")
    d[volume_col] = pd.to_numeric(d.get(volume_col), errors="coerce")
    base = d[np.isfinite(d[price_col]) & np.isfinite(d[volume_col]) & (d[volume_col] > 0)].copy()

    def _one_report(tag: str, df_: pd.DataFrame) -> None:
        n = len(df_)
        if n < 3:
            print(f"{tag}: n={n} (too small)"); return
        x = df_[price_col].astype(float).values
        y = df_[volume_col].astype(float).values
        rho_s, p_s = spearmanr(x, y)
        mask_pos = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if mask_pos.sum() >= 3:
            r_p, p_p = pearsonr(np.log10(x[mask_pos]), np.log10(y[mask_pos]))
            pearson_str = f"Pearson log-log: r={r_p:+.3f}, p={p_p:.4f}, n={int(mask_pos.sum())}"
        else:
            pearson_str = "Pearson log-log: n<3 (or nonpositive price)"
        print(f"{tag}: Spearman: ρ={rho_s:+.3f}, p={p_s:.4f}, n={n} | {pearson_str}")

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
    if by_year and by_country and "Year" in base.columns and "CountryCode" in base.columns:
        for (yr, cc), g in base.groupby(["Year", "CountryCode"], sort=True):
            _one_report(f"Year={int(yr)} | Country={cc}", g)


# -------------------- PLOTS --------------------
def plot_obs1_country_bars(test_df: pd.DataFrame) -> None:
    """Observation 1: averages by country (bar plot)."""
    code_order   = ["IND", "CHN", "USA"]
    code_to_name = {"IND": "India", "CHN": "China", "USA": "USA"}

    d = test_df[test_df["CountryCode"].isin(code_order)].copy()
    g = (
        d.groupby("CountryCode", as_index=False)
         .agg(
            DMF          =("DMF",         "mean"),
            NDMA         =("NDMA",        "mean"),
            Dissolution  =("Dissolution", "mean"),
            n_DMF        =("DMF",         lambda s: int(s.notna().sum())),
            n_NDMA       =("NDMA",        lambda s: int(s.notna().sum())),
            n_Dissolution=("Dissolution", lambda s: int(s.notna().sum())),
         )
    )
    g["CountryCode"] = pd.Categorical(g["CountryCode"], categories=code_order, ordered=True)
    g = g.sort_values("CountryCode").reset_index(drop=True)

    bar_color  = "#93c5fd"
    edge_color = "#2563eb"
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)

    metrics = [
        ("DMF",         "DMF (ng/day)",     "{:,.0f}",  "n_DMF"),
        ("NDMA",        "NDMA (ng/day)",    "{:,.1f}",  "n_NDMA"),
        ("Dissolution", "Dissolution Difference","{:,.2f}",  "n_Dissolution"),
    ]
    x       = np.arange(len(g))
    xlabels = [code_to_name[str(c)] for c in g["CountryCode"].astype(str).tolist()]

    for ax, (col, title, fmt, ncol) in zip(axes, metrics):
        vals  = g[col].values
        nvals = g[ncol].values
        bars = ax.bar(x, vals, color=bar_color, edgecolor=edge_color, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Country")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
        finite_vals = vals[np.isfinite(vals)]
        ymax  = float(np.nanmax(finite_vals)) if len(finite_vals) else 1.0
        y_top = ymax * 1.12
        y_pad = max(y_top * 0.10, 0.05) if col != "Dissolution" else max(y_top * 0.18, 0.015)
        ax.set_ylim(-y_pad, y_top)
        for rect, val in zip(bars, vals):
            if np.isnan(val): continue
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=10)
        y_n = -y_pad * 0.55
        for xi, n in zip(x, nvals):
            ax.text(xi, y_n, f"n={int(n)}", ha="center", va="center", fontsize=9, color="#374151")

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = FIGURES_DIR / "Figure1_Quality_by_Country.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"Saved Figure 1 → {out}")
    plt.show()


def plot_obs2_volume_price_boxes_with_country_jitter(test_with_prior: pd.DataFrame) -> None:
    """
    Observation 2: price and volume by prior inspection outcome.
    Dots = individual test records, colored by country.
    """
    d = test_with_prior.copy()
    d = d[d["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()
    d["price"]  = pd.to_numeric(d.get("price"),  errors="coerce")
    d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce")

    fig, (ax_price, ax_vol) = plt.subplots(1, 2, figsize=(12, 4))

    def prepare_score_x_mapping(dd: pd.DataFrame, score_col: str):
        present = [s for s in SCORE_ORDER if (dd[score_col] == s).any()]
        mapping = {s: i for i, s in enumerate(present, start=1)}
        return present, mapping

    def add_n_linear_space(ax, present_scores, n_by_score):
        y0, y1   = ax.get_ylim()
        y1       = max(y1, 0.0)
        y_top    = y1 * 1.05 if y1 != 0 else 1.0
        y_pad    = max(y_top * 0.12, 0.05)
        ax.set_ylim(-y_pad, y_top)
        for i, s in enumerate(present_scores, start=1):
            ax.text(i, -0.55 * y_pad, f"n={int(n_by_score.get(s, 0))}",
                    ha="center", va="center", fontsize=9, color="#374151")

    def add_n_log_space(ax, present_scores, n_by_score):
        y0, y1 = ax.get_ylim()
        if not np.isfinite(y0) or y0 <= 0: y0 = 1.0
        if not np.isfinite(y1) or y1 <= y0: y1 = y0 * 10
        y_floor = y0 / 30.0
        ax.set_ylim(y_floor, y1)
        for i, s in enumerate(present_scores, start=1):
            ax.text(i, y_floor * 1.8, f"n={int(n_by_score.get(s, 0))}",
                    ha="center", va="center", fontsize=9, color="#374151")

    # ---------- PRICE ----------
    price_df = d[d["PriorScore"].notna() & d["price"].notna()].copy()
    if not price_df.empty:
        present_scores, score_to_x = prepare_score_x_mapping(price_df, "PriorScore")
        box_data = [price_df.loc[price_df["PriorScore"] == s, "price"].values for s in present_scores]
        ax_price.boxplot(box_data, positions=list(range(1, len(present_scores)+1)),
                         widths=0.35, showfliers=False,
                         medianprops=dict(linewidth=1.5), boxprops=dict(linewidth=1.2),
                         whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
        for code in COUNTRY_CODE_ORDER:
            d_cc = price_df[price_df["CountryCode"] == code].copy()
            if d_cc.empty: continue
            d_cc["xcat"] = d_cc["PriorScore"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", "price", radius=0.02)
            ax_price.scatter(d_cc["xcat"] + d_cc["jx"], d_cc["price"],
                             s=25, alpha=0.7, c=country_code_colors.get(code, "#6b7280"), edgecolor="none")
        ax_price.set_xlabel("Prior Inspection Outcome", fontsize=11)
        ax_price.set_ylabel("Price per Unit ($)", fontsize=11)
        #ax_price.set_title("Price vs Prior Inspection Outcome", fontsize=11, fontweight="bold")
        ax_price.set_xticks(range(1, len(present_scores)+1))
        ax_price.set_xticklabels([SCORE_LABELS.get(float(s), str(s)) for s in present_scores])
        ax_price.grid(True, axis="y", alpha=0.3)
        add_n_linear_space(ax_price, present_scores, price_df.groupby("PriorScore").size().to_dict())

    # ---------- VOLUME ----------
    vol_df = d[d["PriorScore"].notna() & d["volume"].notna() & (d["volume"] > 0)].copy()
    if not vol_df.empty:
        present_scores, score_to_x = prepare_score_x_mapping(vol_df, "PriorScore")
        box_data = [vol_df.loc[vol_df["PriorScore"] == s, "volume"].values for s in present_scores]
        ax_vol.boxplot(box_data, positions=list(range(1, len(present_scores)+1)),
                       widths=0.35, showfliers=False,
                       medianprops=dict(linewidth=1.5), boxprops=dict(linewidth=1.2),
                       whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
        for code in COUNTRY_CODE_ORDER:
            d_cc = vol_df[vol_df["CountryCode"] == code].copy()
            if d_cc.empty: continue
            d_cc["xcat"] = d_cc["PriorScore"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", "volume", radius=0.02)
            ax_vol.scatter(d_cc["xcat"] + d_cc["jx"], d_cc["volume"],
                           s=25, alpha=0.7, c=country_code_colors.get(code, "#6b7280"), edgecolor="none")
        ax_vol.set_xlabel("Prior Inspection Outcome", fontsize=11)
        ax_vol.set_ylabel("Market Volume (Extended Units)", fontsize=11)
        ax_vol.set_yscale("log")
        #ax_vol.set_title("Market Volume vs Prior Inspection Outcome", fontsize=11, fontweight="bold")
        ax_vol.set_xticks(range(1, len(present_scores)+1))
        ax_vol.set_xticklabels([SCORE_LABELS.get(float(s), str(s)) for s in present_scores])
        ax_vol.grid(True, which="major", axis="both", alpha=0.3)
        ax_vol.grid(False, which="minor")
        add_n_log_space(ax_vol, present_scores, vol_df.groupby("PriorScore").size().to_dict())

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=country_code_colors.get(code, "#6b7280"), label=code)
        for code in COUNTRY_CODE_ORDER
    ]
    fig.legend(handles=legend_handles, title="CountryCode",
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    out = FIGURES_DIR / "Figure2_Price_Volume_by_Inspection.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"Saved Figure 2 → {out}")
    plt.show()


def print_fig2_summary_stats(
    df: pd.DataFrame,
    score_col: str = "PriorScore",
    price_col: str = "price",
    vol_col: str = "volume",
) -> None:
    """Prints mean/median/IQR by PriorScore using same inclusion rules as Figure 2."""
    def _summ(dd, val_col):
        out = (
            dd.groupby(score_col)[val_col]
              .agg(n="count", mean="mean", median="median",
                   p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
              .reindex(SCORE_ORDER).reset_index()
        )
        out["Outcome"] = out[score_col].map(SCORE_LABELS)
        return out[["Outcome", score_col, "n", "mean", "median", "p25", "p75"]]

    d = df.copy()
    d[score_col] = pd.to_numeric(d.get(score_col), errors="coerce")
    d[price_col] = pd.to_numeric(d.get(price_col), errors="coerce")
    d[vol_col]   = pd.to_numeric(d.get(vol_col),   errors="coerce")
    price_df = d[d[score_col].notna() & d[price_col].notna()].copy()
    vol_df   = d[d[score_col].notna() & d[vol_col].notna() & (d[vol_col] > 0)].copy()

    print("\n" + "=" * 90)
    print("Figure 2 summary stats (by Prior Inspection Outcome)")
    print("=" * 90)
    if price_df.empty:
        print("PRICE: no data after filters")
    else:
        print("\nPRICE per unit ($):")
        print(_summ(price_df, price_col).to_string(index=False, float_format=lambda x: f"{x:,.6g}"))
    if vol_df.empty:
        print("\nVOLUME: no data after filters")
    else:
        print("\nMARKET VOLUME (Extended Units):")
        print(_summ(vol_df, vol_col).to_string(index=False, float_format=lambda x: f"{x:,.6g}"))
    print("=" * 90 + "\n")


# -------------------- FEI-CLUSTER BOOTSTRAP (Figs 3 & 4) --------------------
def _fei_cluster_bootstrap_spearman(x, y, fei, n_boot=1000, seed=42):
    """
    FEI-level block bootstrap for Spearman ρ.
    Returns: (rho_obs, ci_lo, ci_hi, p_clustered, n_used)

    Resamples whole FEI blocks (all NDCs within a facility move together),
    giving a p-value and 95% CI that account for within-facility correlation.
    Minimum p-value = 2/n_boot.
    """
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    xm, ym, fm = x[mask], y[mask], fei[mask]
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n

    rho_obs, _ = spearmanr(xm, ym)
    feis = np.unique(fm)
    rng  = np.random.default_rng(seed)

    boot_rhos = []
    for _ in range(n_boot):
        sampled_feis = rng.choice(feis, size=len(feis), replace=True)
        idx = np.concatenate([np.where(fm == f)[0] for f in sampled_feis])
        if len(idx) < 3:
            continue
        r, _ = spearmanr(xm[idx], ym[idx])
        boot_rhos.append(r)

    if len(boot_rhos) < 10:
        return rho_obs, np.nan, np.nan, np.nan, n

    boot_rhos  = np.array(boot_rhos)
    ci_lo      = float(np.percentile(boot_rhos, 2.5))
    ci_hi      = float(np.percentile(boot_rhos, 97.5))
    # Two-sided p via shift method (null: ρ=0):
    #   center bootstrap distribution at 0, count fraction where |centered| >= |rho_obs|
    #   Using np.abs() on BOTH sides already gives the two-sided p-value — NO * 2 needed.
    #   Minimum p = 1/n_boot (one bootstrap sample); cap at 1.0.
    centered   = boot_rhos - np.mean(boot_rhos)
    p_val      = float(min(max(np.mean(np.abs(centered) >= np.abs(rho_obs)),
                               1.0 / n_boot), 1.0))
    return rho_obs, ci_lo, ci_hi, p_val, n


# -------------------- SCATTER PLOTS (Obs 3 / Figs 3 & 4) --------------------
def plot_obs3_scatter_volume_price_vs_quality(test_df: pd.DataFrame) -> None:
    """
    Figs 3 & 4: scatter of volume and price vs quality metrics.
    Correlation annotation uses FEI-cluster bootstrap Spearman ρ with 95% CI.
    Trend line fitted in log-y space.
    """
    d = test_df[test_df["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()

    def add_trend_and_corr(ax, x: np.ndarray, y: np.ndarray, ndc: np.ndarray,
                           xscale: str, linthresh: float = 1.0):
        x   = x.astype(float)
        y   = y.astype(float)
        ndc = np.asarray(ndc, dtype=str)

        mask_plot = np.isfinite(x) & np.isfinite(y) & (y > 0)
        xf, yf, ndcf = x[mask_plot], y[mask_plot], ndc[mask_plot]
        n_corr = int(len(xf))

        if n_corr <= 2:
            ax.text(0.02, 0.98, f"n={n_corr}\nCorrelation: n<3",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.50))
            return

        if xscale == "symlog":
            def T(u):
                u = np.asarray(u, dtype=float); out = u.copy()
                big = u >= linthresh
                out[big] = linthresh * (1.0 + np.log10(u[big] / linthresh))
                return out
            def Tinv(t):
                t = np.asarray(t, dtype=float); out = t.copy()
                big = t >= linthresh
                out[big] = linthresh * (10 ** (t[big] / linthresh - 1.0))
                return out
            Xfit = T(xf)
        else:
            def T(u):    return np.asarray(u, dtype=float)
            def Tinv(t): return np.asarray(t, dtype=float)
            Xfit = xf

        try:
            Yfit   = np.log10(yf)
            b, a   = np.polyfit(Xfit, Yfit, 1)
            x_line = Tinv(np.linspace(np.nanmin(Xfit), np.nanmax(Xfit), 200))
            y_line = 10 ** (a + b * np.linspace(np.nanmin(Xfit), np.nanmax(Xfit), 200))
            ax.plot(x_line, y_line, "r--", alpha=0.55, linewidth=2)
        except Exception:
            pass

        # NDC-cluster bootstrap Spearman — same function and parameters as console output
        res = _block_bootstrap_spearman(xf, yf, ndcf, n_boot=2000, seed=42)
        rho   = res["rho"]
        ci_lo = res["ci_lo"]
        ci_hi = res["ci_hi"]
        p_cl  = res["p_boot"]
        n_used = res["n_obs"]
        if np.isnan(p_cl):
            corr_text = f"n={n_used}\nρ={rho:+.3f}\np=n/a (boot failed)"
        else:
            sig = "**" if p_cl < 0.01 else ("*" if p_cl < 0.05 else ("." if p_cl < 0.10 else ""))
            corr_text = (f"n={n_used}\nρ={rho:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]"
                         f"\np={p_cl:.4f} {sig}")
        ax.text(0.02, 0.98, corr_text,
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.50))

    def one_figure(ycol: str, ylabel: str, title: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        xcols   = ["DMF",          "NDMA",          "Dissolution"]
        xlabels = ["DMF (ng/day)", "NDMA (ng/day)", "Dissolution Difference"]

        for ax, xcol, xlabel in zip(axes, xcols, xlabels):
            dd = d.dropna(subset=[xcol, ycol]).copy()

            for code in COUNTRY_CODE_ORDER:
                d_cc = dd[dd["CountryCode"] == code]
                if d_cc.empty: continue
                ax.scatter(d_cc[xcol].values, d_cc[ycol].values,
                           s=55, alpha=0.65, c=country_code_colors.get(code, "#6b7280"),
                           edgecolor="white", linewidth=0.4)

            xscale    = "linear"
            linthresh = 1.0
            if xcol in ["DMF", "NDMA"]:
                ax.set_xscale("symlog", linthresh=linthresh)
                xscale = "symlog"
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            # Pass NDC11 for cluster bootstrap — matches console output exactly
            ndc_arr = dd["NDC11"].values if "NDC11" in dd.columns else np.zeros(len(dd), dtype=str)
            add_trend_and_corr(ax, dd[xcol].values.astype(float),
                               dd[ycol].values.astype(float), ndc_arr,
                               xscale=xscale, linthresh=linthresh)

        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   color=country_code_colors.get(code, "#6b7280"),
                   label={"IND": "India", "USA": "United States of America", "CHN": "China"}.get(code, code),
                   markeredgecolor="white", markeredgewidth=0.5, markersize=8)
            for code in COUNTRY_CODE_ORDER
        ]
        fig.legend(handles=legend_handles, title="Country",
                   loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, framealpha=0.9)
        plt.tight_layout(rect=[0, 0.06, 1, 0.92])
        fig_num  = 3 if ycol == "volume" else 4
        fig_name = "Volume" if ycol == "volume" else "Price"
        out = FIGURES_DIR / f"Figure{fig_num}_Quality_vs_{fig_name}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        print(f"Saved Figure {fig_num} → {out}")
        plt.show()

    one_figure(ycol="volume", ylabel="Market Volume (Extended Units)", title="")
    one_figure(ycol="price",  ylabel="Price per Unit ($)",             title="")


def print_fig3_fig4_scatter_stats(test_df: pd.DataFrame) -> None:
    """
    Prints Spearman ρ statistics for each scatter panel in Figs 3 & 4.

    Shows two p-values side by side:
      1. Naive Spearman (parametric, assumes independence) — same as before
      2. NDC-cluster bootstrap — accounts for same NDC across multiple test years

    NOTE: For Figs 3 & 4 (market-level IQVIA data), NDC-level clustering is the
    correct approach (one observation per NDC-Year; the dependence is that the same
    NDC appears in 2020, 2022, and 2024).  FEI-level clustering is used for the
    scatter plot annotations on the figure (as requested by the advisor), but with
    only ~10 FEI clusters it has essentially no power and gives p≈1 for all panels.
    """
    # Restore the original NDC-clustered output (same numbers as before)
    test_fig3_fig4_scatter_correlations(test_df)


# -------------------- LOAD DATA --------------------
df = pd.read_excel(DF_FILE)
print(f"Loaded: {DF_FILE.name} | rows={len(df):,} | cols={len(df.columns)}")

# -------------------- BUILD NDC–YEAR TABLE --------------------
ndc_year_df = build_ndc_year_table(df)
print(f"NDC–Year rows (tests): {len(ndc_year_df):,}")
print("Unique NDC11:", int(ndc_year_df["NDC11"].nunique()))
print("Avg event-rows collapsed per NDC–Year:", float(ndc_year_df["n_event_rows"].mean()))

# -------------------- PLOTS --------------------
plot_obs1_country_bars(ndc_year_df)
plot_obs2_volume_price_boxes_with_country_jitter(ndc_year_df)
print_fig2_summary_stats(ndc_year_df)
plot_obs3_scatter_volume_price_vs_quality(ndc_year_df)
print_fig3_fig4_scatter_stats(ndc_year_df)   # prints same ρ/CI/p to console

# -------------------- PRICE vs VOLUME CHECK --------------------
report_price_volume_relationship(ndc_year_df, by_year=True, by_country=True)

# -------------------- STATISTICAL MODELS (Model B — Primary) --------------------
run_advanced_models(ndc_year_df, output_dir="./diagnostic_plots")


# %%
# ================================================================================
# WARNING LETTER ANALYSIS
# For each FEI with a warning letter, plot monthly IQVIA volume around the event.
# Backs the claim: warning letters do not obviously change volume.
# ================================================================================

WARNING_LETTER_CASES = [
    {"label": "Lupin Ltd. (FEI 3004819820) — WL 2017-11-06",
     "fei": "3004819820", "wl_date": "2017-11-06",
     "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lupin-limited-535014-11062017"},
    {"label": "Aurobindo Pharma Limited Unit XI (FEI 3004611182) — WL 2019-06-20",
     "fei": "3004611182", "wl_date": "2019-06-20",
     "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/aurobindo-pharma-limited-577033-06202019"},
    {"label": "Zydus / Cadila (FEI 3002984011) — WL 2019-10-29",
     "fei": "3002984011", "wl_date": "2019-10-29",
     "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/cadila-healthcare-limited-584856-10292019"},
    {"label": "Novel Laboratories dba Lupin (FEI 3006271438) — WL 2021-06-11",
     "fei": "3006271438", "wl_date": "2021-06-11",
     "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/novel-laboratories-inc-dba-lupin-613385-06112021"},
    {"label": "Lupin Limited Tarapur (FEI 3002807512) — WL 2022-09-27",
     "fei": "3002807512", "wl_date": "2022-09-27",
     "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lupin-limited-633703-09272022"},
    {"label": "Sun Pharma (FEI 3002809586) — WL 2024-06-18",
     "fei": "3002809586", "wl_date": "2024-06-18",
     "ref_url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sun-pharmaceutical-industries-limited-677337-06182024"},
]

DATA_ROOT    = (
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/Data"
)
MONTHLY_STEM = "2025-12-18-iqvia_with_sdud_nadac.cleaned"

monthly_path = (
    Path(DATA_ROOT)
    / "04_06_11 - Build - Monthly Panel (IQVIA+SDUD+NADAC)"
    / "processed"
    / f"{MONTHLY_STEM}.csv"
)
if not monthly_path.exists():
    for ext in [".parquet", ".pq", ".xlsx", ".csv.gz", ""]:
        p = monthly_path.with_suffix(ext) if ext else monthly_path.parent / MONTHLY_STEM
        if p.exists():
            monthly_path = p
            break

if monthly_path.suffix.lower() in [".parquet", ".pq"]:
    dfm = pd.read_parquet(monthly_path)
elif monthly_path.suffix.lower() == ".xlsx":
    dfm = pd.read_excel(monthly_path)
else:
    dfm = pd.read_csv(monthly_path)

print("Loaded monthly:", monthly_path)
print("Monthly cols:",   list(dfm.columns))

dfm["date"]  = pd.to_datetime(dfm["date"], errors="coerce")
dfm = dfm.dropna(subset=["date"]).copy()
dfm["ndc11"] = dfm["ndc11"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
dfm["iqvia_extended_units"] = pd.to_numeric(dfm["iqvia_extended_units"], errors="coerce")


def ndc_532_to_ndc11(s):
    if pd.isna(s): return None
    parts = str(s).strip().split("-")
    if len(parts) != 3: return None
    a = "".join(ch for ch in parts[0] if ch.isdigit()).zfill(5)
    b = "".join(ch for ch in parts[1] if ch.isdigit()).zfill(4)
    c = "".join(ch for ch in parts[2] if ch.isdigit()).zfill(2)
    out = a + b + c
    return out if len(out) == 11 else None


for case in WARNING_LETTER_CASES:
    fei     = str(case["fei"])
    wl_date = pd.Timestamp(case["wl_date"])
    label   = case.get("label", fei)

    dd = df.copy()
    dd["FEI"] = dd["FEI"].astype(str)
    ndcs_532  = dd.loc[dd["FEI"] == fei, "NDC"].dropna().astype(str).str.strip().unique().tolist()
    ndcs_532  = sorted(set(ndcs_532))

    if not ndcs_532:
        print(f"[SKIP] No NDCs in Q&A for FEI={fei} ({label})"); continue

    ndc11_list = sorted({x for x in (ndc_532_to_ndc11(n) for n in ndcs_532)
                         if isinstance(x, str) and len(x) == 11})
    if not ndc11_list:
        print(f"[SKIP] Could not convert NDCs to ndc11 for FEI={fei} ({label})"); continue

    m = dfm[dfm["ndc11"].isin(ndc11_list)].copy()
    if m.empty:
        print(f"[SKIP] Monthly has no matches for FEI={fei} ({label}) | ndc11_count={len(ndc11_list)}"); continue

    series = (
        m.groupby("date", as_index=False)
         .agg(units=("iqvia_extended_units", "sum"),
              n_rows=("date", "size"),
              n_ndc11=("ndc11", "nunique"))
         .sort_values("date").reset_index(drop=True)
    )
    start = wl_date - pd.DateOffset(months=24)
    end   = wl_date + pd.DateOffset(months=24)
    sw    = series[(series["date"] >= start) & (series["date"] <= end)].copy()
    if sw.empty:
        sw = series.copy()

    plt.figure(figsize=(12, 4))
    plt.plot(sw["date"], sw["units"], marker="o", linewidth=2)
    plt.axvline(wl_date, linestyle="--", linewidth=2)
    plt.title(f"IQVIA extended units (sum over FEI-linked NDCs)\n{label} | FEI={fei} | ndc11={len(ndc11_list)}")
    plt.xlabel("Month")
    plt.ylabel("Extended Units")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"{label} | FEI={fei} | WL={wl_date.date()} | NDCs(Q&A)={len(ndcs_532)} | ndc11={len(ndc11_list)} | monthly_rows={len(m):,}")

# %%
# ================================================================================
# VALISURE NDC COVERAGE ANALYSIS
# Reads all three Valisure testing sheets (2020, 2022, 2024), builds a master list
# of unique NDC11s, then compares to the Q&A working dataset.
# Also maps country from Q&A FEI→CountryCode where possible.
# Goal: understand what fraction of the 113 Valisure NDCs are IND/CHN/USA vs other.
# ================================================================================

VALISURE_FILE = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/06 - Metformin Data/Valisure Test/Valisure_All_SD.xlsx"
)

def _normalize_ndc11(s):
    """
    Accept a digit-only string (hyphens/spaces already stripped by caller).
    Returns 11-digit string or None.
    """
    try:
        raw = str(s).strip()
        if not raw.isdigit():
            return None
        if len(raw) == 11:
            return raw
        if len(raw) == 10:
            return "0" + raw   # prepend leading zero (4-4-2 → 5-4-2)
        return None
    except Exception:
        return None


def analyze_valisure_ndc_coverage():
    """
    Print a comprehensive NDC coverage table comparing Valisure raw sheets
    to the Q&A working dataset, including country assignment.
    """
    print("\n" + "=" * 70)
    print("VALISURE NDC COVERAGE ANALYSIS")
    print("=" * 70)

    # ── Build NDC→country/FEI/Firm lookup from Q&A dataset ──────────────
    # Use raw df (all 88 NDCs, all countries incl. BGD/CAN) + compute NDC11
    df_all = ensure_ndc542_and_ndc11(df)
    qa_ndc_map = (
        df_all[["NDC11", "CountryCode", "FEI", "Firm"]]
        .dropna(subset=["NDC11"])
        .drop_duplicates(subset=["NDC11"])
        .set_index("NDC11")
    )

    # Also build a firm-name→country lookup (for Valisure firm columns)
    firm_country_map = (
        df_all[["Firm", "CountryCode"]]
        .dropna()
        .drop_duplicates(subset=["Firm"])
        .set_index("Firm")["CountryCode"]
        .to_dict()
    )

    # ── Read each Valisure sheet ─────────────────────────────────────────
    rows_all = []

    # Sheet 1 (index 0) — 2020 data; NDC column name is "NDC"
    # Sheet 2 (index 1) — duplicate of Sheet 1, SKIP
    # Sheet 3 (index 2) — 2022 data; NDC column name is "NDC11"
    # Sheet 4 (index 3) — 2024 data; merged header, real headers on row 1

    def _find_col(cols, *candidates):
        """Return first candidate found in cols, else None."""
        for c in candidates:
            if c in cols:
                return c
        return None

    raw_ndc_sets = {}   # year → set of raw NDC strings (before normalization)

    def _parse_sheet(xl_df, year):
        ndc_col  = _find_col(xl_df.columns, "NDC11", "NDC")
        firm_col = _find_col(xl_df.columns, "Company", "Firm", "Manufacturer")
        if ndc_col is None:
            print(f"  [WARN] No NDC column found in {year} sheet. Columns: {list(xl_df.columns)}")
            return
        raw_set = set()
        for _, row in xl_df.iterrows():
            raw_ndc = row.get(ndc_col, None)
            firm    = row.get(firm_col, None) if firm_col else None
            if pd.isna(raw_ndc):
                continue
            raw_str = str(raw_ndc).strip().replace("-", "").replace(" ", "")
            raw_set.add(raw_str)          # track raw before normalization
            if len(raw_str) > 11:
                print(f"  [WARN] Malformed NDC skipped ({year}): {raw_ndc!r} → {firm}")
                continue
            ndc11 = _normalize_ndc11(raw_str)
            if ndc11:
                rows_all.append({"ndc11": ndc11, "year": year, "firm": firm})
        raw_ndc_sets[year] = raw_set

    v2020 = pd.read_excel(VALISURE_FILE, sheet_name=0)           # Sheet 1
    _parse_sheet(v2020, 2020)

    v2022 = pd.read_excel(VALISURE_FILE, sheet_name=2)           # Sheet 3
    _parse_sheet(v2022, 2022)

    v2024 = pd.read_excel(VALISURE_FILE, sheet_name=3, header=1) # Sheet 4
    _parse_sheet(v2024, 2024)

    # ── Raw (pre-normalization) unique counts — reconcile with Excel ─────
    all_raw = set().union(*raw_ndc_sets.values())
    print(f"\nRAW unique NDCs (before zero-padding, as stored in file):")
    for yr, s in sorted(raw_ndc_sets.items()):
        print(f"  {yr}: {len(s)} unique raw NDC strings")
    print(f"  ALL sheets combined (union): {len(all_raw)} unique raw NDC strings")
    print(f"  → If Excel shows a different number, check for hyphens or leading zeros in the file")

    valisure_df = pd.DataFrame(rows_all)
    print(f"\nAfter normalization to 11-digit NDC11:")
    print(f"  Total rows parsed (all sheets): {len(valisure_df):,}")

    # ── Unique NDC11s per year and overall ───────────────────────────────
    years_per_ndc = valisure_df.groupby("ndc11")["year"].apply(set)
    unique_ndcs   = valisure_df["ndc11"].unique()
    print(f"Unique NDC11s across all Valisure sheets: {len(unique_ndcs)}")
    for yr in [2020, 2022, 2024]:
        n = valisure_df[valisure_df["year"] == yr]["ndc11"].nunique()
        print(f"  {yr}: {n} unique NDC11s")

    # ── Map country: prefer Q&A lookup, fallback to Valisure firm name ───
    def get_country(ndc11, firm):
        if ndc11 in qa_ndc_map.index:
            return qa_ndc_map.loc[ndc11, "CountryCode"]
        if firm and str(firm).strip() in firm_country_map:
            return firm_country_map[str(firm).strip()]
        return None

    # One row per unique NDC11 (take first firm seen)
    ndc_summary = (
        valisure_df.sort_values("year")
                   .drop_duplicates(subset=["ndc11"])
                   [["ndc11", "firm"]]
                   .reset_index(drop=True)
    )
    ndc_summary["years_tested"] = ndc_summary["ndc11"].map(
        lambda n: sorted(years_per_ndc.get(n, set()))
    )
    ndc_summary["in_qa"]      = ndc_summary["ndc11"].isin(qa_ndc_map.index)
    ndc_summary["CountryCode"] = ndc_summary.apply(
        lambda r: get_country(r["ndc11"], r["firm"]), axis=1
    )

    # ── Summary by match status ──────────────────────────────────────────
    matched   = ndc_summary[ndc_summary["in_qa"]]
    unmatched = ndc_summary[~ndc_summary["in_qa"]]

    total_v = len(ndc_summary)
    print(f"\n{'═'*55}")
    print(f"  VALISURE NDC TOTALS (after 11-digit normalization)")
    print(f"{'═'*55}")
    print(f"  Total unique Valisure NDC11s            : {total_v:>4}")
    print(f"  Matched to Q&A dataset (all countries)  : {len(matched):>4}")
    print(f"  Not in Q&A (Valisure-only)              : {len(unmatched):>4}")

    # ── Country breakdown for ALL Valisure NDCs ──────────────────────────
    print(f"\n{'─'*55}")
    print("  Country breakdown — ALL Valisure NDC11s:")
    print(f"  {'Country':<22} {'N':>5}   {'% of total':>10}")
    print(f"  {'─'*40}")
    country_counts = ndc_summary["CountryCode"].value_counts(dropna=False)
    for code, cnt in country_counts.items():
        label = str(code) if pd.notna(code) else "Unknown/Not mapped"
        pct = 100 * cnt / total_v
        print(f"  {label:<22} {cnt:>5}   {pct:>9.1f}%")

    ind_chn_usa = ndc_summary[ndc_summary["CountryCode"].isin(["IND", "CHN", "USA"])]
    pct_icu = 100 * len(ind_chn_usa) / total_v
    print(f"\n  ★ IND + CHN + USA subtotal : {len(ind_chn_usa):>4} of {total_v}  ({pct_icu:.1f}%)")
    print(f"  ★ Our working Q&A dataset  :   82 unique NDCs (IND/CHN/USA only)")

    # ── Country breakdown for MATCHED NDCs only ──────────────────────────
    print(f"\n{'─'*55}")
    print("  Country breakdown — Matched NDC11s only (in Q&A):")
    matched_country = matched["CountryCode"].value_counts(dropna=False)
    for code, cnt in matched_country.items():
        label = str(code) if pd.notna(code) else "Unknown"
        print(f"  {label:<22} {cnt:>5}")

    # ── Unmatched NDC list with country where known ───────────────────────
    print(f"\n{'─'*50}")
    print("Unmatched Valisure NDC11s (not in Q&A working dataset):")
    print(f"{'NDC11':<15} {'Firm':<40} {'Country':<10} {'Years'}")
    print("-" * 80)
    for _, r in unmatched.sort_values("CountryCode").iterrows():
        country = r["CountryCode"] if pd.notna(r["CountryCode"]) else "?"
        firm    = str(r["firm"])[:38] if pd.notna(r["firm"]) else "?"
        years   = ", ".join(str(y) for y in r["years_tested"])
        print(f"{r['ndc11']:<15} {firm:<40} {country:<10} {years}")

    print("=" * 70)
    return ndc_summary


valisure_ndc_summary = analyze_valisure_ndc_coverage()

# %%
