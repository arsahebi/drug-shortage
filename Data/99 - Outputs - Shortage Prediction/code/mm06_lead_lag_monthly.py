# %%
"""
Module MM06 — Monthly lead-lag analysis and recall circularity assessment.

Reads master_panel_monthly.parquet and produces:

    outputs/tables/lead_lag_monthly.csv
        For each (signal_group, signal_col, month_offset) in range(-12, +1):
        mean signal value in pre-shortage windows vs control baseline mean.

    outputs/figures/lead_lag_monthly_redica.png
    outputs/figures/lead_lag_monthly_faers.png
    outputs/figures/lead_lag_monthly_recalls.png
        Event-study trajectory for each signal group.

    outputs/tables/recall_circularity.csv
        Each matched recall event with: gap to nearest shortage onset,
        circularity classification, and recall reason bucket.

    outputs/figures/recall_circularity_analysis.png
        Distribution of recall-to-onset gaps by reason (CGMP vs other).

    outputs/tables/monthly_analysis_summary.md
        Plain-language summary of findings.

CAVEATS baked into this analysis:
  - Only 14 drugs, ~21 shortage-start months → very wide CIs; interpret
    trajectories as "suggestive" / "exploratory" only.
  - FAERS is quarterly (non-zero only in months 1, 4, 7, 10). The *_w3m
    rolling sums are used for FAERS to avoid artificial monthly sparsity.
  - Recall results are reported separately for CGMP vs. non-CGMP reasons
    because CGMP recalls may be genuine upstream signals while other
    recall reasons are often coincident with the shortage itself.
"""

from __future__ import annotations
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DATA, OUT_FIGS, OUT_LOGS, OUT_TABS
from utils import get_logger, read_table

log = get_logger("mm06_lead_lag_monthly", OUT_LOGS / "mm06_lead_lag_monthly.log")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# Month offsets to examine around shortage onset
LOOKBACK  = 12   # months before onset
LOOKAHEAD =  0   # include onset month itself
# Control exclusion: drug-months within ±N months of ANY onset are excluded
CONTROL_EXCL_MONTHS = 12


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _midx(year: int | float, month: int | float) -> int:
    """Linear month index: year*12 + (month-1).  Enables integer arithmetic."""
    return int(year) * 12 + int(month) - 1


def _to_midx_series(df: pd.DataFrame) -> pd.Series:
    return df["year"] * 12 + df["month"] - 1


# ─────────────────────────────────────────────────────────────────────────────
# Event-study builder
# ─────────────────────────────────────────────────────────────────────────────

def build_event_study(panel: pd.DataFrame, signal_cols: list[str],
                      lookback: int = LOOKBACK) -> pd.DataFrame:
    """
    For each shortage onset (drug, month T), record signal values at
    offsets -lookback … 0.  Returns a long DataFrame with columns:
        drug_norm, onset_period, offset, <signal_cols>
    """
    panel = panel.copy()
    panel["midx"] = _to_midx_series(panel)
    # Map midx → row for fast lookup
    rows_out: list[dict] = []

    for drug, g in panel.groupby("drug_norm"):
        g_idx = g.set_index("midx")
        onsets = g_idx.index[g_idx["shortage_start"] == 1].tolist()
        for T in onsets:
            onset_period = g_idx.at[T, "period"] if T in g_idx.index else ""
            for k in range(-lookback, LOOKAHEAD + 1):
                idx = T + k
                if idx not in g_idx.index:
                    continue
                row: dict = {"drug_norm": drug, "onset_period": onset_period, "offset": k}
                for c in signal_cols:
                    row[c] = g_idx.at[idx, c] if c in g_idx.columns else np.nan
                rows_out.append(row)

    return pd.DataFrame(rows_out)


# ─────────────────────────────────────────────────────────────────────────────
# Control baseline
# ─────────────────────────────────────────────────────────────────────────────

def build_control_baseline(panel: pd.DataFrame, signal_cols: list[str],
                            excl_months: int = CONTROL_EXCL_MONTHS) -> pd.Series:
    """
    Mean of each signal across drug-months that are NOT within ±excl_months
    of any shortage onset for that drug.
    """
    p = panel.copy()
    p["midx"] = _to_midx_series(p)

    onset_map: dict[str, list[int]] = {}
    for drug, g in p.groupby("drug_norm"):
        onset_map[drug] = g.loc[g["shortage_start"] == 1, "midx"].tolist()

    def is_control(row: pd.Series) -> bool:
        onsets = onset_map.get(row["drug_norm"], [])
        if not onsets:
            return True
        return all(abs(row["midx"] - T) > excl_months for T in onsets)

    ctrl_mask = p.apply(is_control, axis=1)
    log.info("Control rows: %d / %d (%.1f%%)",
             ctrl_mask.sum(), len(p), 100 * ctrl_mask.mean())
    ctrl = p[ctrl_mask]
    return ctrl[[c for c in signal_cols if c in ctrl.columns]].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Lead-lag aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lead_lag_table(es: pd.DataFrame, baseline: pd.Series,
                           signal_cols: list[str],
                           signal_group: str) -> pd.DataFrame:
    """Return long-form lead-lag table per signal × offset."""
    rows: list[dict] = []
    for c in signal_cols:
        if c not in es.columns:
            continue
        bl_val = float(baseline.get(c, np.nan))
        grp = es.groupby("offset")[c]
        mean_ = grp.mean()
        sem_  = grp.sem()
        n_    = grp.count()
        for off in sorted(mean_.index):
            rows.append({
                "signal_group":    signal_group,
                "signal":          c,
                "offset_months":   int(off),
                "mean":            round(mean_[off], 4),
                "se":              round(sem_[off],  4),
                "n_events":        int(n_[off]),
                "baseline_mean":   round(bl_val, 4),
                "vs_baseline":     round(mean_[off] - bl_val, 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_group(es: pd.DataFrame, baseline: pd.Series,
                signal_cols: list[str], title: str, fname: str,
                n_onsets: int) -> None:
    ncols = 2
    nrows = int(np.ceil(len(signal_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), constrained_layout=True)
    ax_flat = axes.flat if nrows > 1 or ncols > 1 else [axes]

    for ax, col in zip(ax_flat, signal_cols):
        if col not in es.columns:
            ax.set_visible(False)
            continue
        grp = es.groupby("offset")[col]
        m   = grp.mean()
        s   = grp.sem()
        ax.errorbar(m.index, m.values, yerr=s.values,
                    marker="o", capsize=3, linewidth=1.5,
                    label=f"Pre-shortage mean (n={n_onsets} events)")
        bl = baseline.get(col, np.nan)
        if not np.isnan(bl):
            ax.axhline(bl, color="C2", linestyle="--", linewidth=1.2,
                       label=f"Control baseline (no shortage ±{CONTROL_EXCL_MONTHS}m)")
        ax.axvline(0, color="red", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_xlabel("Months relative to shortage onset (0 = onset month)")
        ax.set_ylabel(f"Mean {col}")
        ax.set_title(col.replace("_w3m", " (3m rolling)").replace("_", " "), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for ax in list(ax_flat)[len(signal_cols):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{title}\n"
        f"Exploratory — {n_onsets} shortage-onset months, 14 drugs, 2015–2024. "
        "Error bars = ±1 SE; interpret as suggestive only.",
        fontsize=10, y=1.01,
    )
    fig.savefig(OUT_FIGS / fname)
    plt.close(fig)
    log.info("Wrote %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# Recall circularity analysis
# ─────────────────────────────────────────────────────────────────────────────

def recall_circularity_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each matched recall event, compute its gap (months) relative to the
    nearest shortage onset for the same drug, and classify it:
        'pre_shortage'   : gap < -3  months (potentially leading signal)
        'coincident'     : -3 ≤ gap ≤ 0  (immediately before/at onset)
        'post_onset'     : gap > 0  (after shortage began)
        'no_shortage'    : drug has no shortage onset in the panel

    Also flags whether the recall month has shortage_ongoing = 1 for that drug
    (most direct circularity indicator).

    Returns a row-level DataFrame of recall events with timing metadata.
    """
    detail_path = OUT_DATA / "recall_matched_events_monthly.csv"
    if not detail_path.exists():
        log.warning("recall_matched_events_monthly.csv not found; "
                    "run mm02_recall_monthly first")
        return pd.DataFrame()

    events = pd.read_csv(detail_path, parse_dates=["recall_dt"])
    if events.empty:
        log.warning("No recall events found for circularity analysis")
        return pd.DataFrame()

    events["recall_midx"] = events["year"] * 12 + events["month"] - 1

    # Build onset index per drug
    panel["midx"] = _to_midx_series(panel)
    onset_map: dict[str, list[int]] = {}
    for drug, g in panel.groupby("drug_norm"):
        onset_map[drug] = g.loc[g["shortage_start"] == 1, "midx"].tolist()

    # Build ongoing-shortage index per drug-midx for fast lookup
    ongoing_set: set[tuple[str, int]] = set(
        zip(panel.loc[panel["shortage_ongoing"] == 1, "drug_norm"],
            panel.loc[panel["shortage_ongoing"] == 1, "midx"])
    )

    rows_out: list[dict] = []
    for _, ev in events.iterrows():
        drug = ev["drug_norm"]
        r_midx = int(ev["recall_midx"])
        onsets = onset_map.get(drug, [])
        during = (drug, r_midx) in ongoing_set

        if not onsets:
            nearest_gap = np.nan
            label = "no_shortage"
        else:
            gaps = [r_midx - T for T in onsets]
            nearest_gap = int(min(gaps, key=abs))
            if nearest_gap < -3:
                label = "pre_shortage"
            elif nearest_gap <= 0:
                label = "coincident"
            else:
                label = "post_onset"

        # Reason bucket (first truthy reason wins for labelling)
        reason_label = "other"
        for col, name in [("n_cgmp", "cgmp"), ("n_contam", "contamination"),
                          ("n_potency", "potency"), ("n_stability", "stability"),
                          ("n_dissolution", "dissolution")]:
            if int(ev.get(col, 0)) > 0:
                reason_label = name
                break

        rows_out.append({
            "drug_norm":       drug,
            "year":            ev["year"],
            "month":           ev["month"],
            "recall_dt":       ev.get("recall_dt"),
            "reason_bucket":   reason_label,
            "is_cgmp":         int(ev.get("n_cgmp", 0)) > 0,
            "class_I":         int(ev.get("is_I", 0)) > 0,
            "nearest_onset_gap_months": nearest_gap,
            "timing_class":    label,
            "during_shortage": int(during),
        })

    df = pd.DataFrame(rows_out)
    log.info("Recall circularity: %d events | %s",
             len(df), df["timing_class"].value_counts().to_dict())
    return df


def _plot_circularity(circ: pd.DataFrame, n_onsets: int) -> None:
    if circ.empty:
        log.warning("No circularity data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left: gap distribution CGMP vs non-CGMP
    ax = axes[0]
    bins = range(-15, 16)
    cgmp_gaps     = circ.loc[circ["is_cgmp"],      "nearest_onset_gap_months"].dropna()
    noncgmp_gaps  = circ.loc[~circ["is_cgmp"],     "nearest_onset_gap_months"].dropna()
    if not cgmp_gaps.empty:
        ax.hist(cgmp_gaps,    bins=bins, alpha=0.6, label=f"CGMP recalls (n={len(cgmp_gaps)})",    color="C0")
    if not noncgmp_gaps.empty:
        ax.hist(noncgmp_gaps, bins=bins, alpha=0.6, label=f"Non-CGMP recalls (n={len(noncgmp_gaps)})", color="C1")
    ax.axvline(0,  color="red",   linestyle="--", linewidth=1.2, label="Shortage onset (0)")
    ax.axvline(-3, color="green", linestyle=":",  linewidth=1,   label="Pre-shortage threshold (−3m)")
    ax.set_xlabel("Recall months relative to nearest shortage onset\n(negative = before onset)")
    ax.set_ylabel("Number of recalls")
    ax.set_title("Recall timing relative to shortage onset\n(CGMP vs. non-CGMP)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: timing class breakdown by reason
    ax2 = axes[1]
    tbl = (circ.groupby(["timing_class", "reason_bucket"])
           .size().unstack(fill_value=0))
    tbl.plot(kind="bar", ax=ax2, width=0.7)
    ax2.set_xlabel("Timing class relative to shortage onset")
    ax2.set_ylabel("Number of recalls")
    ax2.set_title("Recall count by timing class and reason\n"
                  "(pre_shortage = >3m before; coincident = within 3m)")
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(title="Reason", fontsize=7, title_fontsize=7)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Recall Circularity Analysis — {n_onsets} shortage-onset months, 14 drugs\n"
        "Exploratory; small sample (recall n is also small).",
        fontsize=10,
    )
    fig.savefig(OUT_FIGS / "recall_circularity_analysis.png")
    plt.close(fig)
    log.info("Wrote recall_circularity_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary note
# ─────────────────────────────────────────────────────────────────────────────

def _generate_summary(lead_lag: pd.DataFrame, circ: pd.DataFrame,
                      n_drugs: int, n_onsets: int) -> str:
    """Generate a markdown summary note based on computed results."""
    lines: list[str] = [
        "# Monthly Lead-Lag Analysis — Exploratory Summary\n",
        f"**Data:** {n_drugs} Valisure-tested APIs × 120 months (Jan 2015–Dec 2024), "
        f"{n_onsets} shortage-onset months.\n",
        "> **Framing note:** With only ~21 shortage-onset events the estimates below are "
        "highly uncertain. All findings are exploratory and suggestive; no causal "
        "inference is intended.\n",
        "---\n",
        "## 1. Signals that precede shortage onset (month offsets −12 to 0)\n",
    ]

    if not lead_lag.empty:
        # For each signal, find the offset where vs_baseline is most positive
        # and whether that peak is at offset < -3 (clearly leading)
        summary_rows = []
        for sig, sg in lead_lag.groupby("signal"):
            peak_row = sg.loc[sg["vs_baseline"].idxmax()]
            pre_rows = sg[sg["offset_months"] < -3]
            avg_pre_elevation = pre_rows["vs_baseline"].mean() if not pre_rows.empty else 0.0
            summary_rows.append({
                "signal":            sig,
                "peak_offset":       int(peak_row["offset_months"]),
                "peak_vs_baseline":  float(peak_row["vs_baseline"]),
                "avg_pre_elevation": float(avg_pre_elevation),
            })
        sr = pd.DataFrame(summary_rows).sort_values("avg_pre_elevation", ascending=False)

        lines.append("| Signal | Avg elevation (−12 to −4m) | Peak offset | Peak vs baseline |\n")
        lines.append("|---|---|---|---|\n")
        for _, r in sr.iterrows():
            direction = "↑" if r["avg_pre_elevation"] > 0 else "—"
            lines.append(
                f"| {r['signal']} | {direction} {r['avg_pre_elevation']:+.3f} "
                f"| {r['peak_offset']:+d}m | {r['peak_vs_baseline']:+.3f} |\n"
            )
        lines.append("\n")

        # Identify any clearly leading signals (avg elevation > 0 at −12 to −4)
        leading = sr[sr["avg_pre_elevation"] > 0].sort_values("avg_pre_elevation", ascending=False)
        if not leading.empty:
            top = leading.iloc[0]
            lines.append(
                f"**Suggestive leading signals:** `{top['signal']}` shows the largest average "
                f"elevation before shortage onset (+{top['avg_pre_elevation']:.3f} vs baseline "
                f"in months −12 to −4). This is exploratory; the wide error bars "
                f"(small n={n_onsets} events) prevent firm conclusions.\n\n"
            )
        else:
            lines.append(
                "**No signal shows clear average elevation in the −12 to −4 month window.** "
                "Elevation appears concentrated near or at the onset month, "
                "which is consistent with coincidence rather than leading behavior.\n\n"
            )
    else:
        lines.append("*(Lead-lag table is empty — check upstream modules.)*\n\n")

    lines += [
        "---\n",
        "## 2. Recall circularity\n",
    ]

    if not circ.empty:
        n_total  = len(circ)
        n_cgmp   = circ["is_cgmp"].sum()
        tc       = circ["timing_class"].value_counts()
        n_pre    = int(tc.get("pre_shortage", 0))
        n_coinc  = int(tc.get("coincident", 0))
        n_post   = int(tc.get("post_onset", 0))
        n_no_sh  = int(tc.get("no_shortage", 0))

        # CGMP specifically
        cgmp_tc = circ.loc[circ["is_cgmp"], "timing_class"].value_counts()
        cgmp_pre = int(cgmp_tc.get("pre_shortage", 0))

        lines.append(
            f"Total matched recalls: **{n_total}** "
            f"({n_cgmp} CGMP, {n_total - n_cgmp} non-CGMP).\n\n"
        )
        lines.append("**Timing breakdown (all recalls):**\n\n")
        lines.append(f"- Pre-shortage (gap < −3m): {n_pre} recalls ({100*n_pre/max(n_total,1):.0f}%)\n")
        lines.append(f"- Coincident (gap −3 to 0m): {n_coinc} recalls ({100*n_coinc/max(n_total,1):.0f}%)\n")
        lines.append(f"- Post-onset (gap > 0m): {n_post} recalls ({100*n_post/max(n_total,1):.0f}%)\n")
        lines.append(f"- No shortage for this drug: {n_no_sh} recalls\n\n")

        lines.append(
            f"**CGMP recalls pre-shortage:** {cgmp_pre} of {n_cgmp} CGMP recalls "
            f"({100*cgmp_pre/max(n_cgmp,1):.0f}%) fall >3 months before the nearest "
            "shortage onset, suggesting a possible upstream manufacturing signal. "
            "However the absolute counts are very small; this is suggestive only.\n\n"
        )
        lines.append(
            "**Circularity concern:** Recalls that fall during an active shortage "
            f"({circ['during_shortage'].sum()} of {n_total}) are likely mechanically "
            "circular — the shortage may have prompted or coincided with the recall "
            "rather than caused it. These are flagged as `recall_during_shortage = 1` "
            "in master_panel_monthly.csv and should be excluded from causal analysis.\n\n"
        )
    else:
        lines.append("*(No recall events found for circularity analysis.)*\n\n")

    lines += [
        "---\n",
        "## 3. Data quality notes\n",
        "- **FAERS resolution:** Quarterly (non-zero in months 1, 4, 7, 10 only). "
        "Lead-lag analysis uses 3-month rolling sums (*_w3m). Interpret with care.\n",
        "- **Valisure scores:** 2024 snapshot only — NOT time-varying. "
        "Excluded from all lead-lag analysis.\n",
        "- **Recall sparsity:** Very few matched recall events across 14 drugs. "
        "CGMP/other breakdowns are based on small counts.\n",
        "- **Sample size:** ~21 shortage-onset months drives all event-study estimates. "
        "Standard errors are large; all signals should be treated as hypotheses "
        "for future validation, not confirmed findings.\n",
    ]

    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OAI forward-looking event study  (Wang et al. 2025 context)
# ─────────────────────────────────────────────────────────────────────────────

OAI_LOOKAHEAD   = 12   # months after OAI event to track
OAI_EXCL_MONTHS = 12   # exclusion window for control baseline


def oai_forward_study(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Use each month where redica_n_oai >= 1 as an "event" and track two
    shortage-state metrics at offsets -6 to +OAI_LOOKAHEAD:

        shortage_ongoing  — is the drug in shortage at that specific month?
                            This is the primary metric (mirrors Wang et al.'s
                            1-year forward indicator).  High baseline power
                            because shortages can last months to years.

        shortage_start    — did a NEW shortage begin? (kept for reference but
                            too sparse for meaningful monthly analysis.)

    Additionally computes per-OAI-event:
        any_shortage_fwd12 — was the drug in shortage at ANY point in months
                             +1 to +12 after the OAI?  This is the closest
                             analog to Wang et al.'s binary outcome variable.

    Returns:
        offset_df    — long-form by offset (primary chart data)
        event_df     — one row per OAI event with any_shortage_fwd12

    Context: Wang et al. (MSOM 2025) find OAI outcomes *reduce* shortage risk
    ~96% (IV-adjusted, 1-year window).  Our observational analog uses
    shortage_ongoing as the forward outcome — more powerful than shortage_start
    because shortages persist.  We cannot replicate IV instruments.
    """
    if "redica_n_oai" not in panel.columns:
        log.warning("redica_n_oai not in panel; skipping OAI forward study")
        return pd.DataFrame(), pd.DataFrame()

    p = panel.copy()
    p["midx"] = _to_midx_series(p)

    # ── OAI events ──────────────────────────────────────────────────────────
    oai_events: list[tuple[str, int]] = []
    for drug, g in p.groupby("drug_norm"):
        g_idx = g.set_index("midx")
        for m, row in g_idx.iterrows():
            if (row.get("redica_n_oai", 0) or 0) >= 1:
                oai_events.append((drug, int(m)))

    if not oai_events:
        log.warning("No OAI months found in panel")
        return pd.DataFrame(), pd.DataFrame()
    log.info("OAI forward study: %d OAI event months, %d unique drugs",
             len(oai_events), len({d for d, _ in oai_events}))

    # ── Event windows ────────────────────────────────────────────────────────
    OFFSETS = list(range(-6, OAI_LOOKAHEAD + 1))
    rows_out: list[dict] = []
    event_rows: list[dict] = []

    for drug, T in oai_events:
        g_idx = p[p["drug_norm"] == drug].set_index("midx")
        # Per-offset metrics
        for k in OFFSETS:
            idx = T + k
            if idx not in g_idx.index:
                continue
            rows_out.append({
                "drug_norm":       drug,
                "oai_midx":        T,
                "offset":          k,
                "shortage_ongoing": int(g_idx.at[idx, "shortage_ongoing"]),
                "shortage_start":  int(g_idx.at[idx, "shortage_start"]),
            })
        # Any shortage in forward 12 months
        fwd_ongoing = [
            int(g_idx.at[T + k, "shortage_ongoing"])
            for k in range(1, OAI_LOOKAHEAD + 1)
            if T + k in g_idx.index
        ]
        any_fwd12 = int(any(v == 1 for v in fwd_ongoing))
        months_in_shortage_fwd12 = int(sum(fwd_ongoing))
        # Pre-OAI state: was the drug already in shortage at T?
        in_shortage_at_oai = int(g_idx.at[T, "shortage_ongoing"]) if T in g_idx.index else 0
        event_rows.append({
            "drug_norm":              drug,
            "oai_midx":               T,
            "in_shortage_at_oai":     in_shortage_at_oai,
            "any_shortage_fwd12":     any_fwd12,
            "months_in_shortage_fwd12": months_in_shortage_fwd12,
        })

    es = pd.DataFrame(rows_out)
    ev = pd.DataFrame(event_rows)

    if es.empty:
        return pd.DataFrame(), ev

    # ── Control baseline: shortage_ongoing rate in drug-months far from any OAI
    #    Only exclude ±OAI_EXCL_MONTHS from OAI; no onset exclusion needed
    #    because shortage_ongoing already captures the state naturally. ────────
    oai_midx_by_drug: dict[str, list[int]] = {}
    for drug, T in oai_events:
        oai_midx_by_drug.setdefault(drug, []).append(T)

    ctrl_ongoing: list[float] = []
    for _, row in p.iterrows():
        drug = row["drug_norm"]
        m    = int(row["midx"])
        oai_list = oai_midx_by_drug.get(drug, [])
        if not any(abs(m - T) <= OAI_EXCL_MONTHS for T in oai_list):
            ctrl_ongoing.append(float(row["shortage_ongoing"]))

    baseline_ongoing = float(np.mean(ctrl_ongoing)) if ctrl_ongoing else np.nan
    log.info("OAI forward study: baseline shortage_ongoing rate = %.4f "
             "(%d control rows)", baseline_ongoing, len(ctrl_ongoing))

    # ── Aggregate by offset ──────────────────────────────────────────────────
    summary_rows: list[dict] = []
    for off in OFFSETS:
        sub = es[es["offset"] == off]
        if sub.empty:
            continue
        m_ong  = float(sub["shortage_ongoing"].mean())
        se_ong = float(sub["shortage_ongoing"].sem()) if len(sub) > 1 else 0.0
        m_st   = float(sub["shortage_start"].mean())
        summary_rows.append({
            "offset":                   off,
            "mean_in_shortage":         round(m_ong, 4),
            "se_in_shortage":           round(se_ong, 4),
            "mean_shortage_start":      round(m_st,  4),
            "n_events":                 int(len(sub)),
            "baseline_in_shortage":     round(baseline_ongoing, 4),
            "vs_baseline":              round(m_ong - baseline_ongoing, 4),
        })

    offset_df = pd.DataFrame(summary_rows)

    # ── Summary stats on event-level any_shortage_fwd12 ─────────────────────
    if not ev.empty:
        n_total  = len(ev)
        n_fwd    = int(ev["any_shortage_fwd12"].sum())
        n_at_oai = int(ev["in_shortage_at_oai"].sum())
        mean_mo  = float(ev["months_in_shortage_fwd12"].mean())
        log.info("OAI events: %d total | %d (%.0f%%) had drug in shortage at OAI month | "
                 "%d (%.0f%%) had ≥1 shortage month in next 12m | mean months in shortage: %.1f",
                 n_total, n_at_oai, 100*n_at_oai/n_total,
                 n_fwd, 100*n_fwd/n_total, mean_mo)

    return offset_df, ev


def valisure_quality_split(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Split drugs by Valisure mean score (median cut) into high / low quality.
    For each group compute the per-drug shortage_start rate and OAI count.
    Returns a per-drug summary with quality_tier column.

    Context: Wang et al. have no quality score; Valisure scores give us a
    direct proxy for underlying quality independent of inspection outcome.
    """
    if "valisure_mean_score" not in panel.columns:
        log.warning("valisure_mean_score not in panel; skipping quality split")
        return pd.DataFrame()

    drug_summ = (panel.groupby("drug_norm").agg(
        shortage_starts    = ("shortage_start",      "sum"),
        shortage_months    = ("shortage_ongoing",     "sum"),
        valisure_score     = ("valisure_mean_score",  "first"),
        oai_total          = ("redica_n_oai",         "sum"),
        inspections_total  = ("redica_n_inspections", "sum"),
        recalls_total      = ("recall_total",         "sum"),
        recalls_cgmp       = ("recall_cgmp",          "sum"),
    ).reset_index())

    med = drug_summ["valisure_score"].median()
    drug_summ["quality_tier"] = drug_summ["valisure_score"].apply(
        lambda x: "high_quality" if (pd.notna(x) and x >= med) else "low_quality"
    )
    drug_summ["valisure_score"] = drug_summ["valisure_score"].round(1)
    drug_summ = drug_summ.sort_values("valisure_score", ascending=False)
    log.info("Valisure quality split: median=%.1f  high=%d low=%d drugs",
             med,
             (drug_summ["quality_tier"] == "high_quality").sum(),
             (drug_summ["quality_tier"] == "low_quality").sum())
    return drug_summ


def _plot_oai_forward(offset_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    """
    Two-panel figure:
      Left:  shortage_ongoing rate at each offset around OAI (primary metric)
      Right: per-event any_shortage_fwd12 breakdown (already in shortage vs not)
    """
    if offset_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # ── Left: offset trajectory ──────────────────────────────────────────────
    ax = axes[0]
    pre  = offset_df[offset_df["offset"] <= 0]
    post = offset_df[offset_df["offset"] > 0]
    bl   = float(offset_df["baseline_in_shortage"].iloc[0])

    ax.fill_between(offset_df["offset"],
                    offset_df["mean_in_shortage"] - offset_df["se_in_shortage"],
                    offset_df["mean_in_shortage"] + offset_df["se_in_shortage"],
                    alpha=0.15, color="C0")
    ax.plot(pre["offset"],  pre["mean_in_shortage"],  "o--", color="C3",
            linewidth=1.5, markersize=5, label="Pre-OAI context (months −6 to 0)")
    ax.plot(post["offset"], post["mean_in_shortage"], "o-",  color="C0",
            linewidth=2.0, markersize=5, label=f"Post-OAI (months +1 to +{OAI_LOOKAHEAD})")
    ax.axhline(bl, color="grey", linestyle=":", linewidth=1.5,
               label=f"Control baseline (no OAI ±{OAI_EXCL_MONTHS}m): {bl:.3f}")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.axvspan(0.5, OAI_LOOKAHEAD + 0.5, alpha=0.05, color="C0")

    ax.set_xlabel("Months relative to OAI inspection (0 = OAI month)")
    ax.set_ylabel("Fraction of events: drug in shortage (shortage_ongoing)")
    ax.set_title(
        "Primary metric: shortage_ongoing rate\naround OAI inspection events",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Right: event-level breakdown ─────────────────────────────────────────
    ax2 = axes[1]
    if not event_df.empty:
        # Split: already in shortage at OAI vs not
        already = event_df[event_df["in_shortage_at_oai"] == 1]
        fresh   = event_df[event_df["in_shortage_at_oai"] == 0]
        cats    = ["Already in shortage\nat OAI month",
                   "NOT in shortage\nat OAI month"]
        n_total = [len(already), len(fresh)]
        n_fwd   = [int(already["any_shortage_fwd12"].sum()),
                   int(fresh["any_shortage_fwd12"].sum())]
        n_no    = [t - f for t, f in zip(n_total, n_fwd)]

        x = np.arange(len(cats))
        w = 0.5
        ax2.bar(x, n_no,  w, label="No shortage in next 12m",    color="C2",   alpha=0.8)
        ax2.bar(x, n_fwd, w, label="≥1 shortage month in next 12m", color="C3", alpha=0.8,
                bottom=n_no)
        ax2.set_xticks(x)
        ax2.set_xticklabels(cats, fontsize=9)
        for i, (nf, nt) in enumerate(zip(n_fwd, n_total)):
            if nt > 0:
                ax2.text(i, nt + 0.3, f"{nf}/{nt}\n({100*nf//nt}%)",
                         ha="center", fontsize=9, color="C3", fontweight="bold")
        ax2.set_ylabel("Number of OAI events")
        ax2.set_title(
            "Post-OAI shortage status (any_shortage_fwd12)\nby drug state at time of OAI",
            fontsize=10,
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis="y")

    n_oai = len(event_df) if not event_df.empty else "?"
    fig.suptitle(
        f"OAI Forward-Looking Event Study — {n_oai} OAI event months, 14 drugs, 2015–2024\n"
        "Primary outcome: shortage_ongoing (is the drug in shortage?).  "
        "Observational — no IV.  Cf. Wang et al. MSOM 2025 (OAI → −96% shortage risk, IV-adjusted).",
        fontsize=9, y=1.02,
    )
    fig.savefig(OUT_FIGS / "oai_forward_study.png")
    plt.close(fig)
    log.info("Wrote oai_forward_study.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    panel = read_table(OUT_DATA / "master_panel_monthly.parquet")

    n_drugs  = panel["drug_norm"].nunique()
    n_onsets = int(panel["shortage_start"].sum())
    log.info("Panel: %d drugs, %d onset months, %d total rows",
             n_drugs, n_onsets, len(panel))

    # ── Signal groups ───────────────────────────────────────────────────────────
    # Use w3m sums for FAERS (quarterly sparsity); raw monthly for Redica & recalls.
    SIGNAL_GROUPS = {
        "redica": [
            "redica_n_483_critical",
            "redica_n_oai",
            "redica_n_warning_letters",
            "redica_n_inspections",
        ],
        "faers": [
            "faers_n_reports_w3m",
            "faers_n_serious_w3m",
            "faers_severity_score_w3m",
        ],
        "recalls": [
            "recall_total",
            "recall_cgmp",
            "recall_contam",
            "recall_potency",
            "recall_class_I",
        ],
    }

    all_signal_cols = [c for cols in SIGNAL_GROUPS.values() for c in cols
                       if c in panel.columns]
    missing = [c for cols in SIGNAL_GROUPS.values() for c in cols
               if c not in panel.columns]
    if missing:
        log.warning("Columns not found in panel (will be skipped): %s", missing)
    SIGNAL_GROUPS = {g: [c for c in cols if c in panel.columns]
                     for g, cols in SIGNAL_GROUPS.items()}

    # ── Build event study and control baseline ──────────────────────────────────
    es       = build_event_study(panel, all_signal_cols, lookback=LOOKBACK)
    baseline = build_control_baseline(panel, all_signal_cols)

    if es.empty:
        log.warning("No shortage-onset months found; lead-lag analysis skipped")
        return

    log.info("Event-study rows: %d (onset events: %d)",
             len(es), es.groupby(["drug_norm", "onset_period"]).ngroups)

    # ── Lead-lag table (all groups) ─────────────────────────────────────────────
    ll_parts: list[pd.DataFrame] = []
    for gname, cols in SIGNAL_GROUPS.items():
        if not cols:
            continue
        ll_parts.append(compute_lead_lag_table(es, baseline, cols, gname))

    lead_lag = pd.concat(ll_parts, ignore_index=True) if ll_parts else pd.DataFrame()
    if not lead_lag.empty:
        lead_lag.to_csv(OUT_TABS / "lead_lag_monthly.csv", index=False)
        log.info("Wrote lead_lag_monthly.csv (%d rows)", len(lead_lag))

    # ── Figures ─────────────────────────────────────────────────────────────────
    group_titles = {
        "redica":  "Redica Regulatory Signals (483 observations, OAI, Warning Letters)",
        "faers":   "FAERS Adverse Events (3-month rolling sums; quarterly resolution)",
        "recalls": "FDA Recalls by Class and Reason",
    }
    for gname, cols in SIGNAL_GROUPS.items():
        if not cols:
            continue
        _plot_group(
            es, baseline, cols,
            title=group_titles.get(gname, gname),
            fname=f"lead_lag_monthly_{gname}.png",
            n_onsets=n_onsets,
        )

    # ── Recall circularity ──────────────────────────────────────────────────────
    circ = recall_circularity_analysis(panel)
    if not circ.empty:
        circ.to_csv(OUT_TABS / "recall_circularity.csv", index=False)
        log.info("Wrote recall_circularity.csv (%d rows)", len(circ))
        _plot_circularity(circ, n_onsets)

    # ── OAI forward-looking study (Wang et al. 2025 context) ───────────────────
    fwd_offset, fwd_events = oai_forward_study(panel)
    if not fwd_offset.empty:
        fwd_offset.to_csv(OUT_TABS / "oai_forward_study.csv", index=False)
        log.info("Wrote oai_forward_study.csv (%d offset rows)", len(fwd_offset))
    if not fwd_events.empty:
        fwd_events.to_csv(OUT_TABS / "oai_forward_study_events.csv", index=False)
        log.info("Wrote oai_forward_study_events.csv (%d event rows)", len(fwd_events))
    if not fwd_offset.empty:
        _plot_oai_forward(fwd_offset, fwd_events)

    # ── Valisure quality split ─────────────────────────────────────────────────
    qs = valisure_quality_split(panel)
    if not qs.empty:
        qs.to_csv(OUT_TABS / "valisure_quality_split.csv", index=False)
        log.info("Wrote valisure_quality_split.csv (%d rows)", len(qs))

    # ── Markdown summary ────────────────────────────────────────────────────────
    note = _generate_summary(lead_lag, circ, n_drugs, n_onsets)
    note_path = OUT_TABS / "monthly_analysis_summary.md"
    note_path.write_text(note, encoding="utf-8")
    log.info("Wrote monthly_analysis_summary.md")

    return lead_lag


if __name__ == "__main__":
    main()

# %%
