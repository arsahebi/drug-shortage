# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------- PATHS --------------------
BASE_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/"
    "My Drive/North Carolina State University/Project - Drug Shortage/"
    "Data/06 - Metformin Data/Derived"
)
DF_FILE = BASE_DIR / "Q&As1234_v9_v4.xlsx"

# -------------------- CONSTANTS --------------------
DMF_COL = "DMF (ng/DAY) Valisure"
NDMA_COL = "NDMA (ng/DAY) Valisure"
DISS_COL = "Difference Factor"  # per your note: “Difference Factor” column

COUNTRY_CODE_ORDER = ["IND", "USA", "CHN"]
SCORE_ORDER = [0.0, 1.5, 3.5]

# Colors
country_colors = {
    "India": "#ef4444",
    "China": "#f59e0b",
    "United States of America": "#3b82f6",
}
country_code_colors = {
    "IND": "#ef4444",
    "CHN": "#f59e0b",
    "USA": "#3b82f6",
}
# -------------------- HELPERS --------------------
from matplotlib.transforms import blended_transform_factory

def add_n_labels_under_categories(ax, x_positions, n_values, y_axes_frac=0.03, fontsize=9):
    """
    Places 'n=..' at a fixed vertical position in axes coordinates (works with log y).
    """
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, n in zip(x_positions, n_values):
        ax.text(
            x, y_axes_frac, f"n={int(n)}",
            transform=trans,
            ha="center", va="bottom",
            fontsize=fontsize, color="#374151"
        )


def pick_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Prefer NADAC price if available; otherwise fall back to SDUD total-per-unit.
    """
    s = pd.Series(np.nan, index=df.index)
    if "nadac_price" in df.columns:
        s = df["nadac_price"].copy()
    if "sdud_price_total_per_unit" in df.columns:
        s = s.fillna(df["sdud_price_total_per_unit"])
    return s
def build_ndc_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    NDC-level:
      - Ensure NDC11 exists
      - Pick most recent Year per NDC11
      - Aggregate multiple rows within that year to a single NDC row (means)
      - Keep volume as iqvia_extended_units
      - Apply measurement rules:
          * DMF measured in 2020/2022/2024 -> keep numeric as-is
          * NDMA measured ONLY in 2020/2022 -> force NDMA=NaN for Year==2024
          * Difference Factor measured ONLY in 2024:
                - force Dissolution=NaN for Year in {2020, 2022}
                - for Year==2024, treat non-numeric / N/A as 0.0 (per your earlier rule)
    """
    d = df.copy()

    # ---- dates ----
    if "Event Start Date" in d.columns:
        d["Event Start Date"] = pd.to_datetime(d["Event Start Date"], errors="coerce")

    # ---- build/clean NDC11 ----
    def _ndc542_to_ndc11(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if not s:
            return np.nan
        s = s.replace(" ", "")
        parts = s.split("-")

        if len(parts) != 3:
            digits = "".join(ch for ch in s if ch.isdigit())
            if len(digits) == 11:
                return digits
            return np.nan

        a, b, c = parts
        a = "".join(ch for ch in a if ch.isdigit())
        b = "".join(ch for ch in b if ch.isdigit())
        c = "".join(ch for ch in c if ch.isdigit())
        if not a or not b or not c:
            return np.nan
        return a.zfill(5) + b.zfill(4) + c.zfill(2)

    if "NDC11" in d.columns:
        d["NDC11"] = d["NDC11"].astype(str).str.replace(r"\D", "", regex=True)
        d.loc[d["NDC11"].str.len() != 11, "NDC11"] = np.nan
    else:
        d["NDC11"] = np.nan

    if "NDC_542" in d.columns:
        d["NDC11"] = d["NDC11"].fillna(d["NDC_542"].apply(_ndc542_to_ndc11))

    d = d.dropna(subset=["NDC11"])

    # ---- years ----
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d = d.dropna(subset=["Year"])
    d["Year"] = d["Year"].astype(int)

    # ---- keep only the most recent year per NDC ----
    d = d.sort_values(["NDC11", "Year", "Event Start Date"], ascending=[True, False, False])
    most_recent_year = d.groupby("NDC11")["Year"].transform("max")
    d = d[d["Year"] == most_recent_year].copy()

    # ---- price + volume ----
    d["price"] = pick_price_series(d)
    d["iqvia_extended_units"] = pd.to_numeric(d.get("iqvia_extended_units"), errors="coerce")

    # ---- numeric quality columns ----
    d["DMF_num"] = pd.to_numeric(d.get(DMF_COL), errors="coerce")

    # NDMA measured only 2020/2022 -> remove 2024 (force NaN)
    d["NDMA_num"] = pd.to_numeric(d.get(NDMA_COL), errors="coerce")
    d.loc[d["Year"] == 2024, "NDMA_num"] = np.nan

    # Difference Factor measured only 2024
    d["Dissolution_num"] = pd.to_numeric(d.get(DISS_COL), errors="coerce")

    # years with no DF measurement -> NaN
    d.loc[d["Year"].isin([2020, 2022]), "Dissolution_num"] = np.nan

    # 2024: if blank/N/A/non-numeric -> 0.0 (your earlier "N/A should =0" rule)
    # d.loc[(d["Year"] == 2024) & (d["Dissolution_num"].isna()), "Dissolution_num"] = 0.0

    agg = (
        d.groupby("NDC11", as_index=False)
        .agg(
            FEI=("FEI", "first"),
            Firm=("Firm", "first"),
            CountryCode=("CountryCode", "first"),
            CountryName=("CountryName", "first"),
            Year=("Year", "first"),
            DMF=("DMF_num", "mean"),
            NDMA=("NDMA_num", "mean"),
            Dissolution=("Dissolution_num", "mean"),
            volume=("iqvia_extended_units", "mean"),
            price=("price", "mean"),
        )
    )
    return agg


def build_facility_level_table(df: pd.DataFrame, ndc_df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Event Start Date"] = pd.to_datetime(d["Event Start Date"], errors="coerce")
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

    def wavg_price(g: pd.DataFrame) -> float:
        w = g["volume"].fillna(0).values
        x = g["price"].values
        if np.nansum(w) > 0:
            mask = ~np.isnan(x)
            if mask.sum() == 0:
                return float(np.nan)
            return float(np.average(x[mask], weights=w[mask]))
        return float(np.nanmean(x))

    f = (
        ndc_df.groupby("FEI", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "Firm": g["Firm"].iloc[0],
                    "CountryName": g["CountryName"].iloc[0],
                    "CountryCode": g["CountryCode"].iloc[0],
                    "TestYear": int(g["Year"].max()),
                    "volume": float(g["volume"].sum(skipna=True)),  # <-- total facility volume
                    "price": wavg_price(g),                        # <-- volume-weighted price
                    "DMF": float(g["DMF"].mean(skipna=True)),
                    "NDMA": float(g["NDMA"].mean(skipna=True)),
                    "Dissolution": float(g["Dissolution"].mean(skipna=True)),
                }
            )
        )
        .reset_index(drop=True)
    )

    # prior score logic (same as before)
    prior_scores_raw = []
    prior_event_date = []
    prior_event_year = []
    prior_event_score_used = []

    for _, row in f.iterrows():
        fei = row["FEI"]
        test_year = row["TestYear"]
        test_date = pd.Timestamp(int(test_year), 12, 31)

        e = events[(events["FEI"] == fei) & (events["Event Start Date"] <= test_date)]
        if len(e) == 0:
            prior_scores_raw.append(np.nan)
            prior_event_date.append(pd.NaT)
            prior_event_year.append(np.nan)
            prior_event_score_used.append(np.nan)
            continue

        last = e.sort_values("Event Start Date").iloc[-1]
        used = last["Event Score"] if ("Event Score" in last.index and pd.notna(last["Event Score"])) else last["Score"]

        prior_scores_raw.append(float(used) if pd.notna(used) else np.nan)
        prior_event_date.append(last["Event Start Date"])
        prior_event_year.append(last.get("EventYear", np.nan))
        prior_event_score_used.append(float(used) if pd.notna(used) else np.nan)

    f["PriorScore_raw"] = prior_scores_raw
    f["PriorEventDate"] = prior_event_date
    f["PriorEventYear"] = prior_event_year
    f["PriorEventScoreUsed"] = prior_event_score_used

    def nearest_score(x):
        if pd.isna(x):
            return np.nan
        arr = np.array(SCORE_ORDER, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(x)))])

    f["PriorScore"] = f["PriorScore_raw"].apply(nearest_score)
    f["PriorScore_cat"] = pd.Categorical(f["PriorScore"], categories=SCORE_ORDER, ordered=True)
    return f

def plot_obs1_country_bars(ndc_df: pd.DataFrame) -> None:
    """
    Observation 1 (UPDATED):
      - Only three countries: India, China, USA (x-axis labels in this exact order)
      - Light blue bars (similar to your example)
      - Write averages above bars for ALL three metrics (DMF, NDMA, Difference Factor)
      - Add number of observations (n) under each bar, where n = NON-NaN count for that metric
        (so n can differ across DMF / NDMA / Dissolution panels)
    """
    # enforce requested order + labels
    code_order = ["IND", "CHN", "USA"]
    code_to_name = {"IND": "India", "CHN": "China", "USA": "USA"}

    d = ndc_df[ndc_df["CountryCode"].isin(code_order)].copy()

    # Aggregate means + metric-specific non-NaN counts
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

    # light-blue style
    bar_color = "#93c5fd"   # light blue
    edge_color = "#2563eb"  # blue edge

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

        # headroom for top labels + add bottom whitespace for n=...
        finite_vals = vals[np.isfinite(vals)]
        ymax = float(np.nanmax(finite_vals)) if len(finite_vals) else 1.0
        y_top = ymax * 1.12

        # reserve some space under 0 for the "n=..." labels
        if col == "Dissolution":
            y_pad = max(y_top * 0.18, 0.015)   # small absolute pad works better here
        else:
            y_pad = max(y_top * 0.10, 0.05)
        ax.set_ylim(-y_pad, y_top)

        # labels above bars (means)
        for rect, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # n labels under bars (inside plot area, below 0)
        y_n = -y_pad * 0.55
        for xi, n in zip(x, nvals):
            ax.text(
                xi,
                y_n,
                f"n={int(n)}",
                ha="center",
                va="center",
                fontsize=9,
                color="#374151",
            )

    fig.suptitle("Observation 1: Quality by country (NDC-level means)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    plt.show()


def add_jitter_by_category(d, cat_col, value_col=None, radius=0.06):
    """
    Adds column 'jx' with symmetric offsets within [-radius, +radius].

    - If value_col is None: jitter within each category (old behavior, but narrower).
    - If value_col is given: jitter only when there are *multiple* points
      with the same (category, value). Singletons stay on the category line.
    """
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


from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_obs2_volume_price_boxes_with_country_jitter(ndc_df: pd.DataFrame, facility_df: pd.DataFrame) -> None:
    """
    Observation 2 (FIXED: true NDC-level):
      - Uses ndc_df (one row per NDC11) to avoid inflated counts from event-expanded df
      - Y1: price box + country-colored jitter points
      - Y2: volume box + country-colored jitter points (log scale)
      - X: PriorScore_fac (facility-level prior score mapped to {0,1.5,3.5})
      - Adds n= inside plot area between x-axis and bottom (like Obs1)
        * price: linear -> negative y-space
        * volume: log -> lower ymin and place n near bottom
    REQUIREMENTS:
      - ndc_df has: NDC11, FEI, Year, CountryCode, price, volume
      - facility_df has: FEI, TestYear, PriorScore
    """

    # ---- merge facility prior score onto NDC rows (correct granularity) ----
    prior_map = facility_df[["FEI", "TestYear", "PriorScore"]].rename(
        columns={"TestYear": "Year", "PriorScore": "PriorScore_fac"}
    )

    d = ndc_df.merge(prior_map, on=["FEI", "Year"], how="left").copy()
    d = d[d["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()

    # ensure numeric
    d["price"] = pd.to_numeric(d.get("price"), errors="coerce")
    d["volume"] = pd.to_numeric(d.get("volume"), errors="coerce")

    fig, (ax_price, ax_vol) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Observation 2: Market Response to Prior Inspection Scores (NDC Level)", fontsize=14, fontweight="bold")

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

    # ---------- PRICE PANEL ----------
    price_df = d[d["PriorScore_fac"].notna() & d["price"].notna()].copy()
    if not price_df.empty:
        present_scores, score_to_x = prepare_score_x_mapping(price_df, "PriorScore_fac")
        box_data = [price_df.loc[price_df["PriorScore_fac"] == s, "price"].values for s in present_scores]

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

            d_cc["xcat"] = d_cc["PriorScore_fac"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", "price", radius=0.02)

            ax_price.scatter(
                d_cc["xcat"] + d_cc["jx"],
                d_cc["price"],
                s=35,
                alpha=0.7,
                c=country_code_colors.get(code, "#6b7280"),
                edgecolor="none",
            )

        ax_price.set_xlabel("Prior Non-Compliance Score", fontsize=11)
        ax_price.set_ylabel("Price per Unit ($)", fontsize=11)
        ax_price.set_title("Price vs Prior Inspection Score", fontsize=11, fontweight="bold")
        ax_price.set_xticks(range(1, len(present_scores) + 1))
        ax_price.set_xticklabels([str(s) for s in present_scores])
        ax_price.grid(True, axis="y", alpha=0.3)

        # n = unique NDCs (already unique rows, but still explicitly count NDC11)
        n_by_score_price = price_df.groupby("PriorScore_fac")["NDC11"].nunique().to_dict()
        add_n_linear_space(ax_price, present_scores, n_by_score_price)

    # ---------- VOLUME PANEL ----------
    vol_df = d[d["PriorScore_fac"].notna() & d["volume"].notna() & (d["volume"] > 0)].copy()
    if not vol_df.empty:
        present_scores, score_to_x = prepare_score_x_mapping(vol_df, "PriorScore_fac")
        box_data = [vol_df.loc[vol_df["PriorScore_fac"] == s, "volume"].values for s in present_scores]

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

            d_cc["xcat"] = d_cc["PriorScore_fac"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", "volume", radius=0.02)

            ax_vol.scatter(
                d_cc["xcat"] + d_cc["jx"],
                d_cc["volume"],
                s=35,
                alpha=0.7,
                c=country_code_colors.get(code, "#6b7280"),
                edgecolor="none",
            )

        ax_vol.set_xlabel("Prior Non-Compliance Score", fontsize=11)
        ax_vol.set_ylabel("Market Volume (Extended Units)", fontsize=11)
        ax_vol.set_yscale("log")
        ax_vol.set_title("Market Volume vs Prior Inspection Score", fontsize=11, fontweight="bold")
        ax_vol.set_xticks(range(1, len(present_scores) + 1))
        ax_vol.set_xticklabels([str(s) for s in present_scores])
        ax_vol.grid(True, which="major", axis="both", alpha=0.3)
        ax_vol.grid(False, which="minor")

        n_by_score_vol = vol_df.groupby("PriorScore_fac")["NDC11"].nunique().to_dict()
        add_n_log_space(ax_vol, present_scores, n_by_score_vol)

    # shared legend
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=country_code_colors.get(code, "#6b7280"), label=code)
        for code in COUNTRY_CODE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        title="CountryCode",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()


from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_obs2a_quality_boxes_with_country_dots(facility_df: pd.DataFrame) -> None:
    """
    Observation 2a:
      - Facility-level boxplots + jittered points by country (IND/USA/CHN)
      - Adds n=... INSIDE the plot area, between x-axis and bottom (like Obs 1)
      - Uses negative y-space (all panels are linear y)
    """

    d = facility_df[facility_df["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Observation 2a: Prior Inspection: Non-Compliance vs Tested Quality (Facility-Level)",
        fontsize=14,
        fontweight="bold",
    )

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
            n = int(n_by_score.get(s, 0))
            ax.text(
                i,
                y_text,
                f"n={n}",
                ha="center",
                va="center",
                fontsize=9,
                color="#374151",
            )

    def panel(ax, ycol: str, ylabel: str, title: str, yscale=None):
        dd = d[d["PriorScore"].notna() & d[ycol].notna()].copy()
        if dd.empty:
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return

        present_scores, score_to_x = prepare_score_x_mapping(dd, "PriorScore")
        box_data = [dd.loc[dd["PriorScore"] == s, ycol].values for s in present_scores]

        ax.boxplot(
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
            d_cc = dd[dd["CountryCode"] == code].copy()
            d_cc = d_cc[d_cc["PriorScore"].isin(present_scores)].copy()
            if d_cc.empty:
                continue

            d_cc["xcat"] = d_cc["PriorScore"].map(score_to_x)
            d_cc = add_jitter_by_category(d_cc, "xcat", ycol, radius=0.02)

            ax.scatter(
                d_cc["xcat"] + d_cc["jx"],
                d_cc[ycol],
                s=45,
                alpha=0.7,
                c=country_code_colors.get(code, "#6b7280"),
                edgecolor="none",
            )

        ax.set_xlabel("Prior Non-Compliance Score", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, len(present_scores) + 1))
        ax.set_xticklabels([str(s) for s in present_scores])
        ax.grid(True, axis="y", alpha=0.3)

        if yscale is not None:
            ax.set_yscale(yscale)

        # n labels inside plot (linear)
        n_by_score = dd.groupby("PriorScore")[ycol].apply(lambda s: int(s.notna().sum())).to_dict()
        add_n_linear_space(ax, present_scores, n_by_score)

    panel(ax1, "DMF", "DMF (ng/day)", "DMF vs Prior Inspection Score")
    panel(ax2, "NDMA", "NDMA (ng/day)", "NDMA vs Prior Inspection Score")
    panel(ax3, "Dissolution", "Difference Factor", "Difference Factor vs Prior Inspection Score")

    legend_handles = [
        Line2D(
            [], [],
            marker="o",
            linestyle="",
            color=country_code_colors.get(code, "#6b7280"),
            label=code,
        )
        for code in COUNTRY_CODE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        title="CountryCode",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.show()


from matplotlib.lines import Line2D
from scipy.stats import spearmanr

def plot_obs3_scatter_volume_price_vs_quality(ndc_df: pd.DataFrame) -> None:
    """
    Observation 3 (UPDATED per measurement rules + correct n counting):
      - Uses ndc_df after build_ndc_level_table year-based measurement rules
      - Dots near 0 are OK (symlog) and will be COUNTED if x==0 (as long as not NaN)
      - n in the box counts ALL plotted points (finite x and y, y>0), including x==0
      - Correlation uses the SAME sample as n (so it matches what you see)
    """
    d = ndc_df[ndc_df["CountryCode"].isin(COUNTRY_CODE_ORDER)].copy()

    def add_trend_and_corr(ax, x: np.ndarray, y: np.ndarray, xscale: str, linthresh: float = 1.0):
        x = x.astype(float)
        y = y.astype(float)

        # must have y>0 for log-y
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

        # ---- transform x into "display coordinate" for fitting so line looks straight ----
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
            def T(u):
                return np.asarray(u, dtype=float)

            def Tinv(t):
                return np.asarray(t, dtype=float)

            Xfit = xf

        try:
            # fit straight line in (T(x), log10(y))
            Yfit = np.log10(yf)
            b, a = np.polyfit(Xfit, Yfit, 1)

            x_line_fit = np.linspace(np.nanmin(Xfit), np.nanmax(Xfit), 200)
            y_line = 10 ** (a + b * x_line_fit)
            x_line = Tinv(x_line_fit)

            ax.plot(x_line, y_line, "r--", alpha=0.55, linewidth=2)

            # Spearman on SAME sample used for n (xf, yf)
            rho, pval = spearmanr(xf, yf)
            ax.text(
                0.02,
                0.98,
                f"n={n_corr}\nCorrelation: ρ={rho:+.3f}\np={pval:.3e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
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

            # scatter points (these are what we want n to reflect)
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
                marker="o",
                linestyle="",
                color=country_code_colors.get(code, "#6b7280"),
                label={"IND": "India", "USA": "United States of America", "CHN": "China"}.get(code, code),
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=8,
            )
            for code in COUNTRY_CODE_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            title="Country",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            framealpha=0.9,
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.92])
        plt.show()

    one_figure(
        ycol="volume",
        ylabel="Market Volume (Extended Units)",
        title="Observation 3: Market Volume vs Tested Quality (NDC-level)",
    )
    one_figure(
        ycol="price",
        ylabel="Price per Unit ($)",
        title="Observation 3: Price vs Tested Quality (NDC-level)",
    )


# -------------------- LOAD DATA --------------------
df = pd.read_excel(DF_FILE)
print(f"Loaded: {DF_FILE.name} | rows={len(df):,} | cols={len(df.columns)}")

# -------------------- BUILD TABLES --------------------
ndc_df = build_ndc_level_table(df)
facility_df = build_facility_level_table(df, ndc_df)

print(f"NDC-level rows: {len(ndc_df):,}")
print(f"Facility-level rows: {len(facility_df):,}")

# -------------------- CALL ALL PLOTTING FUNCTIONS (NO main) --------------------
plot_obs1_country_bars(ndc_df)
plot_obs2_volume_price_boxes_with_country_jitter(ndc_df, facility_df)
ndma_obs2a_raw = facility_df.loc[
    facility_df["CountryCode"].isin(COUNTRY_CODE_ORDER)
    & facility_df["PriorScore"].notna()
    & facility_df["NDMA"].notna(),
    ["FEI", "Firm", "CountryCode", "CountryName", "TestYear", "NDMA",
     "PriorScore", "PriorScore_raw", "PriorEventDate", "PriorEventYear", "PriorEventScoreUsed"]
].copy()
ndma_obs2a_raw.to_excel(BASE_DIR / "obs2a_ndma_raw.xlsx", index=False)
print("Wrote:", BASE_DIR / "obs2a_ndma_raw.xlsx", "| rows:", len(ndma_obs2a_raw))
plot_obs2a_quality_boxes_with_country_dots(facility_df)
plot_obs3_scatter_volume_price_vs_quality(ndc_df)

# %%
