"""
Module MM07 — Combined dashboard generator.

Reads existing pipeline outputs (annual + monthly) and writes a single
self-contained HTML file:  dashboard.html  in OUT_ROOT.

Replaces/supersedes the two legacy dashboard files:
  99 - Outputs - Shortage Prediction/dashboard.html          (old location)
  99 - Outputs - Shortage Prediction/dashboard-simple.html   (old location)

Run from the code/ directory:
    python3 mm07_dashboard.py
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from config import OUT_DATA, OUT_TABS, OUT_ROOT, OUT_LOGS, OUT_MODELS, OUT_FIGS, TEXT_FEATURES_CSV, TEXT_TIMESERIES_CSV, VALISURE_FEI
from utils import get_logger

log = get_logger("mm07_dashboard", OUT_LOGS / "mm07_dashboard.log")

DASH_OUT = OUT_ROOT / "dashboard.html"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(name: str, subdir: Path = OUT_DATA) -> pd.DataFrame | None:
    """Read CSV; return None if missing (dashboard falls back to placeholders)."""
    p = subdir / name
    if not p.exists():
        log.warning("File not found (section will show placeholder): %s", p.name)
        return None
    return pd.read_csv(p)


class _NpEncoder(json.JSONEncoder):
    """Encode numpy scalars as plain Python types so json.dumps doesn't choke."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _j(obj) -> str:
    """Serialize to compact JSON for embedding in JS."""
    return json.dumps(obj, cls=_NpEncoder, allow_nan=False)


# ─────────────────────────────────────────────────────────────────────────────
# Data computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_data() -> dict:
    d: dict = {}

    # ── Annual panel (pilot = has_valisure) ────────────────────────────────────
    ap = _read("master_panel.csv")
    if ap is not None:
        pilot = ap[ap["has_valisure"] == 1].copy()
        by_year = pilot.groupby("year")["shortage_started"].sum().reset_index()
        d["by_year"] = [{"year": int(r.year), "starts": int(r.shortage_started)}
                        for _, r in by_year.iterrows()]

        by_drug_raw = (pilot.groupby("drug_norm").agg(
            starts=("shortage_started", "sum"),
            faers_sev=("faers_severity_score", "mean"),
            faers_serious=("faers_n_serious", "mean"),
            faers_reports=("faers_n_reports", "mean"),
            redica_oai=("redica_n_oai", "mean"),
            redica_483=("redica_n_483_critical", "mean"),
            redica_vai=("redica_n_vai", "mean"),
            redica_wl=("redica_n_warning_letters", "mean"),
            redica_insp=("redica_n_inspections", "mean"),
        ).reset_index())

        # Valisure scores from valisure_drug summary
        vs = _read("eda_valisure_drug_summary.csv", OUT_TABS)
        if vs is None:
            vs = _read("valisure_drug.csv")
        if vs is not None:
            vs_map = vs.set_index("drug_norm")[["valisure_mean_score", "valisure_n_failing"]].to_dict("index")
        else:
            vs_map = {}

        d["by_drug"] = []
        for _, r in by_drug_raw.sort_values("starts", ascending=False).iterrows():
            vm = vs_map.get(r.drug_norm, {})
            d["by_drug"].append({
                "drug":          r.drug_norm,
                "starts":        int(r.starts),
                "faers":         round(float(r.faers_sev), 1),
                "faers_serious": round(float(r.faers_serious or 0), 1),
                "faers_reports": round(float(r.faers_reports or 0), 1),
                "val":           round(float(vm.get("valisure_mean_score", 0) or 0), 1),
                "fails":         int(vm.get("valisure_n_failing", 0) or 0),
                "redica_oai":    round(float(r.redica_oai or 0), 2),
                "redica_483":    round(float(r.redica_483 or 0), 2),
                "redica_vai":    round(float(r.redica_vai or 0), 2),
                "redica_wl":     round(float(r.redica_wl or 0), 2),
                "redica_insp":   round(float(r.redica_insp or 0), 2),
            })

        d["annual_rows"]    = int((ap["has_valisure"] == 1).sum())
        d["annual_events"]  = int(pilot["shortage_started"].sum())
        d["annual_n_drugs"] = int(pilot["drug_norm"].nunique())
    else:
        d["by_year"] = []; d["by_drug"] = []
        d["annual_rows"] = 140; d["annual_events"] = 21; d["annual_n_drugs"] = 14

    # ── Annual lift table ──────────────────────────────────────────────────────
    lt = _read("eda_lead_time_means.csv", OUT_TABS)
    _RECALL_SIGNALS = {
        "recall_total","recall_cgmp","recall_class_I","recall_class_II","recall_class_III",
        "recall_contam","recall_potency","recall_mislabel","recall_stability","recall_foreign",
        "recall_dissolution","recall_total_w3","recall_cgmp_w3","recall_class_I_w3",
    }
    if lt is not None:
        d["lift"] = [{
            "signal": r.iloc[0],
            "mean0":  round(float(r["y_next=0"]),  4),
            "mean1":  round(float(r["y_next=1"]),  4),
            "lift":   round(float(r["lift"]),       3),
        } for _, r in lt.iterrows() if r.iloc[0] not in _RECALL_SIGNALS]
    else:
        d["lift"] = []

    # ── 483 raw text feature lift (annual, drug-year level) ───────────────────
    _TEXT_LIFT_COLS = [
        "repeat_llm_only_share", "contamination_llm_only_share",
        "oos_oot_regex_share", "severity_critmajor_share", "remediation_none_share",
        "repeat_cross_insp_share", "scope_facilitywide_share", "cultural_root_cause_share",
    ]
    if ap is not None:
        q = ap.dropna(subset=["y_next_year_shortage"])
        for c in [c for c in _TEXT_LIFT_COLS if c in q.columns]:
            grp = q.groupby("y_next_year_shortage")[c].mean()
            m0  = float(grp.get(0, 0))
            m1  = float(grp.get(1, 0))
            d["lift"].append({"signal": c, "mean0": round(m0, 3), "mean1": round(m1, 3),
                               "lift": round(m1 / m0, 3) if m0 > 0 else 1.0})

    # ── Annual event study (lead_time_valisure.csv) ────────────────────────────
    ls = _read("lead_time_valisure.csv", OUT_TABS)
    if ls is not None:
        d["annual_lead"] = {
            "rel":           ls["rel_year"].tolist(),
            "recall_total":  ls["recall_total"].round(3).tolist(),
            "recall_cgmp":   ls["recall_cgmp"].round(3).tolist(),
            "faers_sev":     ls["faers_severity_score"].round(1).tolist(),
            "faers_serious": ls["faers_n_serious"].round(1).tolist(),
        }
    else:
        d["annual_lead"] = {"rel": [], "recall_total": [], "recall_cgmp": [], "faers_sev": [], "faers_serious": []}

    # ── Monthly panel stats + shortage duration per drug ──────────────────────
    mp = _read("master_panel_monthly.csv")
    if mp is not None:
        d["monthly_rows"]         = len(mp)
        d["monthly_onset_months"] = int(mp["shortage_start"].sum())
        d["monthly_ongoing_months"] = int(mp["shortage_ongoing"].sum())
        dur = mp.groupby("drug_norm")["shortage_ongoing"].sum().reset_index()
        dur.columns = ["drug_norm", "duration_months"]
        dur_map = dur.set_index("drug_norm")["duration_months"].to_dict()
        for entry in d.get("by_drug", []):
            entry["duration"] = int(dur_map.get(entry["drug"], 0))
    else:
        d["monthly_rows"] = 1680; d["monthly_onset_months"] = "?"; d["monthly_ongoing_months"] = "?"
        for entry in d.get("by_drug", []):
            entry["duration"] = 0

    # ── Text features from 483 timeseries: latest snapshot per FEI, averaged by drug ──
    _TEXT_DETAIL_COLS = [
        "severity_critmajor_share", "contamination_llm_only_share", "repeat_llm_only_share",
        "oos_oot_regex_share", "remediation_none_share",
        "repeat_cross_insp_share", "scope_facilitywide_share", "cultural_root_cause_share",
    ]
    drug_text_detail: dict = {}
    _ts_src = TEXT_TIMESERIES_CSV if TEXT_TIMESERIES_CSV.exists() else TEXT_FEATURES_CSV
    if _ts_src.exists() and VALISURE_FEI.exists():
        tf_raw  = pd.read_csv(_ts_src)
        # If timeseries, take most recent snapshot per FEI
        if "snapshot_date" in tf_raw.columns:
            tf_raw["snapshot_date"] = pd.to_datetime(tf_raw["snapshot_date"])
            tf_raw = tf_raw.loc[tf_raw.groupby("fei")["snapshot_date"].idxmax()].copy()
        bridge  = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
        bridge  = (bridge[["API", "FEI_NUMBER"]].dropna()
                   .rename(columns={"FEI_NUMBER": "fei", "API": "drug_norm"}))
        _avail  = [c for c in _TEXT_DETAIL_COLS if c in tf_raw.columns]
        merged  = tf_raw[["fei"] + _avail].merge(bridge, on="fei", how="inner")
        drug_text_detail = merged.groupby("drug_norm")[_avail].mean().round(3).to_dict(orient="index")
    for entry in d.get("by_drug", []):
        detail = drug_text_detail.get(entry["drug"], {})
        entry["sev_high"]      = round(float(detail.get("severity_critmajor_share", 0)), 3)
        entry["contam_share"]  = round(float(detail.get("contamination_llm_only_share", 0)), 3)
        entry["repeat_share"]  = round(float(detail.get("repeat_llm_only_share", 0)), 3)
        entry["oos_share"]     = round(float(detail.get("oos_oot_regex_share", 0)), 3)
        entry["remed_none"]    = round(float(detail.get("remediation_none_share", 0)), 3)
        entry["cross_repeat"]  = round(float(detail.get("repeat_cross_insp_share", 0)), 3)
        entry["scope_fw"]      = round(float(detail.get("scope_facilitywide_share", 0)), 3)
        entry["cultural_rc"]   = round(float(detail.get("cultural_root_cause_share", 0)), 3)

    # ── Text detail group comparison (violation categories + remediation) ──────
    if ap is not None:
        dm2 = ap.groupby("drug_norm")["shortage_started"].sum().reset_index()
        dm2.columns = ["drug_norm", "starts"]
        detail_cols_map = {
            "repeat_share":  "Repeat violations",
            "cross_repeat":  "Cross-insp. repeat",
            "contam_share":  "Contamination",
            "oos_share":     "OOS/OOT references",
            "sev_high":      "Critical+Major sev.",
            "scope_fw":      "Facility-wide scope",
            "cultural_rc":   "Cultural root cause",
            "remed_none":    "No remediation",
        }
        # Build drug-level detail from by_drug entries
        by_drug_dict = {e["drug"]: e for e in d.get("by_drug", [])}
        has_s = [e for e in d.get("by_drug", []) if e["starts"] > 0]
        no_s  = [e for e in d.get("by_drug", []) if e["starts"] == 0]
        if has_s and no_s:
            labels_detail = list(detail_cols_map.values())
            keys = list(detail_cols_map.keys())
            d["text_detail_group"] = {
                "labels":      labels_detail,
                "shortage":    [round(float(pd.Series([e[k] for e in has_s]).mean()) * 100, 1) for k in keys],
                "no_shortage": [round(float(pd.Series([e[k] for e in no_s]).mean()) * 100, 1) for k in keys],
                "n_shortage":    len(has_s),
                "n_no_shortage": len(no_s),
            }
        else:
            d["text_detail_group"] = {}
    else:
        d["text_detail_group"] = {}

    # ── Monthly lead-lag ───────────────────────────────────────────────────────
    ll = _read("lead_lag_monthly.csv", OUT_TABS)
    if ll is not None:
        offsets = [int(x) for x in sorted(ll["offset_months"].unique())]
        groups = {}
        for sig, g in ll.groupby("signal"):
            g = g.set_index("offset_months").reindex(offsets)
            groups[sig] = {
                "offsets":   offsets,
                "means":     [round(float(v), 4) if pd.notna(v) else 0.0 for v in g["mean"]],
                "ses":       [round(float(v), 4) if pd.notna(v) else 0.0 for v in g["se"]],
                "baseline":  float(g["baseline_mean"].dropna().iloc[0]) if not g["baseline_mean"].dropna().empty else 0.0,
                "group":     str(ll.loc[ll["signal"] == sig, "signal_group"].iloc[0]),
            }
        d["monthly_lead"] = groups
    else:
        d["monthly_lead"] = {}

    # ── VAI lead-lag (not in lead_lag_monthly.csv — compute from monthly panel) ─
    if mp is not None and "redica_n_vai" in mp.columns:
        mp2 = mp.copy()
        mp2["midx"] = mp2["year"] * 12 + mp2["month"]
        onsets = mp2[mp2["shortage_start"] == 1][["drug_norm", "midx"]].values.tolist()
        _ll_offsets = list(range(-12, 1))
        if onsets:
            records_vai = []
            for drug, omidx in onsets:
                drug_idx = mp2[mp2["drug_norm"] == drug].set_index("midx")["redica_n_vai"]
                for off in _ll_offsets:
                    tmidx = omidx + off
                    if tmidx in drug_idx.index and pd.notna(drug_idx[tmidx]):
                        records_vai.append({"offset": off, "val": float(drug_idx[tmidx])})
            tagged = {(drug, omidx + k) for drug, omidx in onsets for k in range(-12, 13)}
            bl_rows = mp2[~mp2.apply(lambda r: (r["drug_norm"], r["midx"]) in tagged, axis=1)]
            bl_vai  = float(bl_rows["redica_n_vai"].dropna().mean()) if len(bl_rows) else 0.0
            if records_vai:
                df_vai = pd.DataFrame(records_vai)
                grp_vai = df_vai.groupby("offset")["val"]
                d["monthly_lead"]["redica_n_vai"] = {
                    "offsets":  _ll_offsets,
                    "means":    [round(float(grp_vai.mean().get(o, 0)), 4) for o in _ll_offsets],
                    "ses":      [round(float(grp_vai.sem().get(o, 0)), 4) for o in _ll_offsets],
                    "baseline": round(bl_vai, 4),
                    "group":    "redica",
                }

    # ── m11: time-varying repeat-violation 483 count — monthly lead-lag ───────
    hr = _read("text_highrisk_483_monthly.csv")
    if hr is not None and mp is not None and "n_repeat_483_last_24mo" in hr.columns:
        mp3 = mp.merge(hr, on=["drug_norm", "year", "month"], how="left")
        mp3["n_repeat_483_last_24mo"] = mp3["n_repeat_483_last_24mo"].fillna(0)
        mp3["midx"] = mp3["year"] * 12 + mp3["month"]
        onsets = mp3[mp3["shortage_start"] == 1][["drug_norm", "midx"]].values.tolist()
        _off = list(range(-12, 1))
        recs = []
        for drug, om in onsets:
            di = mp3[mp3["drug_norm"] == drug].set_index("midx")["n_repeat_483_last_24mo"]
            for off in _off:
                t = om + off
                if t in di.index and pd.notna(di[t]):
                    recs.append({"offset": off, "val": float(di[t])})
        tagged = {(dr, om + k) for dr, om in onsets for k in range(-12, 13)}
        bl_rows = mp3[~mp3.apply(lambda r: (r["drug_norm"], r["midx"]) in tagged, axis=1)]
        bl = float(bl_rows["n_repeat_483_last_24mo"].mean()) if len(bl_rows) else 0.0
        if recs:
            dfh = pd.DataFrame(recs)
            g = dfh.groupby("offset")["val"]
            d["monthly_lead"]["n_repeat_483_last_24mo"] = {
                "offsets":  _off,
                "means":    [round(float(g.mean().get(o, 0)), 4) for o in _off],
                "ses":      [round(float(g.sem().get(o, 0)), 4) for o in _off],
                "baseline": round(bl, 4),
                "group":    "text",
            }

    # ── m12: text-signal validation grid (curated cells) ──────────────────────
    # Curated from outputs/tables/text_signal_grid.csv (41 features × 5 outcomes
    # × 3 horizons). Selection rule: consistent direction across horizons,
    # n_hi >= 10, and a coherent mechanism. recall@12m excluded (only 2 events).
    _ESC_FEATS = [
        ("repeat_llm_only_share",        "Repeat violations (LLM)"),
        ("repeat_cross_insp_share",      "Cross-insp. repeat (algo)"),
        ("contamination_llm_only_share", "Contamination (LLM)"),
        ("oos_oot_regex_share",          "OOS/OOT references"),
        ("severity_critmajor_share",     "Critical+Major severity"),
        ("scope_facilitywide_share",     "Facility-wide scope"),
    ]
    _REC_FEATS = [
        ("vc_buildingsequipment_share",  "Buildings/equipment violations"),
        ("repeat_llm_only_share",        "Repeat violations (LLM)"),
        ("repeat_cross_insp_share",      "Cross-insp. repeat (algo)"),
        ("capital_root_cause_share",     "Capital root cause"),
        ("cultural_root_cause_share",    "Cultural root cause"),
    ]
    grid = _read("text_signal_grid.csv", OUT_TABS)
    d["grid_esc"], d["grid_rec"], d["grid_extra"] = [], [], {}
    if grid is not None:
        def _cells(feats, outcome):
            out = []
            for feat, label in feats:
                row = grid[(grid["feature"] == feat) & (grid["outcome"] == outcome)]
                if len(row):
                    r = row.iloc[0]
                    out.append({"label": label,
                                "hi": round(float(r["hi_rate"]) * 100, 1),
                                "lo": round(float(r["lo_rate"]) * 100, 1),
                                "lift": float(r["lift"]),
                                "n_hi": int(r["n_hi"]), "n_lo": int(r["n_lo"])})
            return out
        d["grid_esc"] = _cells(_ESC_FEATS, "esc_24")
        d["grid_rec"] = _cells(_REC_FEATS, "rec_24")

        def _cell1(feat, outcome, col="effect"):
            row = grid[(grid["feature"] == feat) & (grid["outcome"] == outcome)]
            return float(row.iloc[0][col]) if len(row) else None
        d["grid_extra"] = {
            "sev_esc12_lift":     _cell1("severity_critmajor_share", "esc_12", "lift"),
            "remed_none_shdur":   _cell1("remediation_none_share", "sh_dur_36"),
            "invest_shdelta":     _cell1("investigation_llm_share", "sh_dur_delta_12"),
            "scope_fw_esc24":     _cell1("scope_facilitywide_share", "esc_24", "lift"),
            "cross_repeat_esc24": _cell1("repeat_cross_insp_share", "esc_24", "lift"),
            "esc24_base": round(float(
                (grid[grid["outcome"] == "esc_24"].iloc[0]["hi_rate"] * grid[grid["outcome"] == "esc_24"].iloc[0]["n_hi"]
                 + grid[grid["outcome"] == "esc_24"].iloc[0]["lo_rate"] * grid[grid["outcome"] == "esc_24"].iloc[0]["n_lo"])
                / (grid[grid["outcome"] == "esc_24"].iloc[0]["n_hi"] + grid[grid["outcome"] == "esc_24"].iloc[0]["n_lo"])) * 100, 1),
            "rec24_base": round(float(
                (grid[grid["outcome"] == "rec_24"].iloc[0]["hi_rate"] * grid[grid["outcome"] == "rec_24"].iloc[0]["n_hi"]
                 + grid[grid["outcome"] == "rec_24"].iloc[0]["lo_rate"] * grid[grid["outcome"] == "rec_24"].iloc[0]["n_lo"])
                / (grid[grid["outcome"] == "rec_24"].iloc[0]["n_hi"] + grid[grid["outcome"] == "rec_24"].iloc[0]["n_lo"])) * 100, 1),
            "n_snapshots": int(grid.iloc[0]["n_hi"] + grid.iloc[0]["n_lo"]),
        }

    # ── m11: FEI drill-down summary ────────────────────────────────────────────
    fs = _read("fei_timeline_summary.csv", OUT_FIGS / "timelines")
    if fs is not None:
        fs["firm_name"] = fs["firm_name"].fillna("")
        d["fei_summary"] = fs.replace({np.nan: None}).to_dict(orient="records")
    else:
        d["fei_summary"] = []

    # ── m11: FEI event-level data for interactive timeline (Section 5B) ────────
    ev_all = _read("fei_events_all.csv")
    d["fei_events"] = {}
    if ev_all is not None:
        ev_all["event_date"] = pd.to_datetime(ev_all["event_date"], errors="coerce")
        ev_all["year_dec"] = (ev_all["event_date"].dt.year +
                              (ev_all["event_date"].dt.dayofyear - 1) / 365.25).round(3)
        for fei_id, grp in ev_all.groupby("fei"):
            firm_nm = grp["firm_name"].dropna().iloc[0] if grp["firm_name"].notna().any() else str(fei_id)
            apis_v  = grp["apis"].dropna().iloc[0] if "apis" in grp.columns and grp["apis"].notna().any() else ""
            ev483, evinsp, evrec, shbands = [], [], [], []
            for _, r in grp.sort_values("event_date").iterrows():
                if pd.isna(r.get("year_dec")):
                    continue
                yr    = float(r["year_dec"])
                etype = str(r["event_type"])
                lbl   = str(r.get("event_label", ""))[:70]
                if etype == "483_snapshot":
                    ev483.append({
                        "yr": yr,
                        "hr": bool(r["high_risk_483"]) if pd.notna(r.get("high_risk_483")) else False,
                        "n_obs": int(r["n_obs_total"]) if pd.notna(r.get("n_obs_total")) else 1,
                        "sev":  round(float(r["severity_critmajor_share"]), 2)
                                if pd.notna(r.get("severity_critmajor_share")) else 0.0,
                        "label": lbl,
                    })
                elif etype == "inspection_outcome":
                    evinsp.append({"yr": yr, "cls": str(r["classification"])
                                   if pd.notna(r.get("classification")) else "?", "label": lbl})
                elif etype == "recall":
                    evrec.append({"yr": yr, "cls": str(r["recall_class"])
                                  if pd.notna(r.get("recall_class")) else "?", "label": lbl})
                elif etype == "shortage_start":
                    drug    = str(r.get("shortage_drug", ""))[:40]
                    end_yr  = None
                    resolved = r.get("shortage_resolved_date")
                    if pd.notna(resolved) and str(resolved) not in ("nan", "NaT", ""):
                        try:
                            rd = pd.to_datetime(resolved, errors="coerce")
                            if pd.notna(rd):
                                end_yr = round(float(rd.year + (rd.dayofyear - 1) / 365.25), 3)
                        except Exception:
                            pass
                    shbands.append({"start": yr, "end": end_yr or 2025.5, "drug": drug})
            d["fei_events"][str(fei_id)] = {
                "firm": str(firm_nm)[:55], "apis": str(apis_v)[:60],
                "events_483": ev483, "events_insp": evinsp,
                "events_recall": evrec, "shortage_bands": shbands,
                "n_483": len(ev483),
                "n_hr":     sum(1 for e in ev483 if e["hr"]),
                "n_oai":    sum(1 for e in evinsp if e["cls"] == "OAI"),
                "n_recalls": len(evrec),
            }
        log.info("Interactive timeline events: %d FEIs", len(d["fei_events"]))

    # ── Drug-level text feature × shortage/FAERS scatter ──────────────────────
    d["text_vs_shortage"] = [
        {
            "drug":     e["drug"],
            "contam":   round(float(e.get("contam_share", 0)) * 100, 1),
            "repeat":   round(float(e.get("repeat_share", 0)) * 100, 1),
            "sev":      round(float(e.get("sev_high",    0)) * 100, 1),
            "cross":    round(float(e.get("cross_repeat", 0)) * 100, 1),
            "duration": int(e.get("duration", 0)),
            "starts":   int(e.get("starts", 0)),
            "faers_serious": round(float(e.get("faers_serious", 0) or 0), 1),
        }
        for e in d.get("by_drug", [])
    ]

    # ── RF model results: feature importance + ablation ───────────────────────
    fi_raw = _read("rf_importance_valisure.csv", OUT_MODELS)
    abl_raw = _read("text_features_ablation.csv", OUT_MODELS)
    met_raw = _read("metrics_valisure.csv", OUT_MODELS)
    if fi_raw is not None:
        fi_raw = fi_raw.sort_values("importance", ascending=False)
        d["rf_importance"] = [{"feature": r["feature"], "imp": round(float(r["importance"]), 4)}
                               for _, r in fi_raw.iterrows()]
    else:
        d["rf_importance"] = []
    if abl_raw is not None:
        d["ablation"] = abl_raw.to_dict(orient="records")
    else:
        d["ablation"] = []
    if met_raw is not None:
        d["model_metrics"] = met_raw.to_dict(orient="records")
    else:
        d["model_metrics"] = []

    # ── OAI forward study (Wang et al. 2025) ──────────────────────────────────
    fw = _read("oai_forward_study.csv", OUT_TABS)
    fwe = _read("oai_forward_study_events.csv", OUT_TABS)
    if fw is not None:
        fw = fw.sort_values("offset")
        d["oai_fwd"] = {
            "offsets":   [int(x) for x in fw["offset"].tolist()],
            "rates":     [round(float(v), 5) for v in fw["mean_in_shortage"]],
            "ses":       [round(float(v), 5) for v in fw["se_in_shortage"]],
            "baseline":  round(float(fw["baseline_in_shortage"].iloc[0]), 5),
            "n_events":  int(fw.loc[fw["offset"] == 1, "n_events"].iloc[0])
                         if 1 in fw["offset"].values else int(fw["n_events"].max()),
        }
    else:
        d["oai_fwd"] = {}

    # Event-level summary for "already in shortage" breakdown
    if fwe is not None:
        n_total   = len(fwe)
        n_already = int((fwe["in_shortage_at_oai"] == 1).sum())
        n_fresh   = n_total - n_already
        n_already_fwd = int(fwe.loc[fwe["in_shortage_at_oai"] == 1, "any_shortage_fwd12"].sum())
        n_fresh_fwd   = int(fwe.loc[fwe["in_shortage_at_oai"] == 0, "any_shortage_fwd12"].sum())
        mean_mo   = round(float(fwe["months_in_shortage_fwd12"].mean()), 1)
        d["oai_events"] = {
            "n_total": n_total, "n_already": n_already, "n_fresh": n_fresh,
            "n_already_fwd": n_already_fwd, "n_fresh_fwd": n_fresh_fwd,
            "mean_months_fwd": mean_mo,
        }
    else:
        d["oai_events"] = {}

    # ── Valisure quality split ─────────────────────────────────────────────────
    qs = _read("valisure_quality_split.csv", OUT_TABS)
    if qs is not None:
        d["quality_split"] = qs.to_dict(orient="records")
    else:
        d["quality_split"] = []

    return d


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

def _lift_rows(lift: list[dict]) -> str:
    label_map = {
        "redica_n_oai": "Redica — OAI inspections",
        "redica_n_warning_letters": "Redica — warning letters",
        "redica_n_483_critical": "Redica — 483 critical obs.",
        "faers_severity_score": "FAERS — severity score",
        "faers_n_serious": "FAERS — serious reports",
        "faers_n_reports": "FAERS — all reports",
        "repeat_llm_only_share":      "483 Text — Repeat violations (LLM)",
        "contamination_llm_only_share": "483 Text — Contamination (LLM)",
        "oos_oot_regex_share":         "483 Text — OOS/OOT references",
        "severity_critmajor_share":    "483 Text — Critical+Major severity",
        "remediation_none_share":      "483 Text — No remediation",
        "repeat_cross_insp_share":     "483 Text — Cross-inspection repeat",
        "scope_facilitywide_share":    "483 Text — Facility-wide scope",
        "cultural_root_cause_share":   "483 Text — Cultural root cause",
    }
    cls_map = lambda x: "high" if x >= 5 else ("mid" if x >= 1.5 else "low")
    rows = ""
    for r in lift:
        lv = r["lift"]
        label = label_map.get(r["signal"], r["signal"])
        reads = "Clearly elevated" if lv >= 5 else ("Slight lift" if lv >= 1.3 else "Roughly flat")
        rows += (f'<tr><td>{label}</td>'
                 f'<td class="num">{r["mean0"]:.3f}</td>'
                 f'<td class="num">{r["mean1"]:.3f}</td>'
                 f'<td class="num"><span class="lift {cls_map(lv)}">{lv:.1f}×</span></td>'
                 f'<td>{reads}</td></tr>\n')
    return rows


def generate_html(d: dict) -> str:
    ml = d.get("monthly_lead", {})

    # Prepare JS data for monthly lead-lag charts
    SIGNAL_LABELS = {
        "redica_n_483_critical": "483 Critical Observations",
        "redica_n_oai": "OAI Inspections",
        "redica_n_vai": "VAI Inspections",
        "redica_n_warning_letters": "Warning Letters",
        "redica_n_inspections": "Total Inspections",
        "faers_n_reports_w3m": "FAERS Reports (3m rolling)",
        "faers_n_serious_w3m": "FAERS Serious (3m rolling)",
        "faers_severity_score_w3m": "FAERS Severity Score (3m rolling)",
        "n_repeat_483_last_24mo":  "Repeat-violation 483s, trailing 24m",
        "n_flagged_483_last_24mo": "Red-flagged 483s (≥2 risk markers), trailing 24m",
    }

    def _ll_overlay_js(canvas_id: str, sig1: str, sig2: str, col1: str, col2: str) -> str:
        """Overlay two lead-lag signals normalized to baseline (% deviation) on same axis."""
        if sig1 not in ml or sig2 not in ml:
            return f'/* overlay {sig1}/{sig2} not in data */'
        i1, i2 = ml[sig1], ml[sig2]
        offsets = i1["offsets"]
        bl1 = i1["baseline"] or 1e-6
        bl2 = i2["baseline"] or 1e-6
        norm1 = [round((m - bl1) / bl1 * 100, 2) for m in i1["means"]]
        norm2 = [round((m - bl2) / bl2 * 100, 2) for m in i2["means"]]
        lbl1 = SIGNAL_LABELS.get(sig1, sig1)
        lbl2 = SIGNAL_LABELS.get(sig2, sig2)
        return f"""
new Chart(document.getElementById({_j(canvas_id)}), {{
  type:'line',
  data:{{
    labels:{_j(offsets)},
    datasets:[
      {{label:{_j(lbl1 + " (% vs baseline)")},data:{_j(norm1)},
        borderColor:'rgb({col1})',backgroundColor:'transparent',
        tension:0.2,pointRadius:3,borderWidth:2,fill:false}},
      {{label:{_j(lbl2 + " (% vs baseline)")},data:{_j(norm2)},
        borderColor:'rgb({col2})',backgroundColor:'transparent',
        tension:0.2,pointRadius:3,borderWidth:2,borderDash:[5,4],fill:false}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10}}}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Months to shortage onset (0 = onset month)'}},grid:{{display:false}}}},
      y:{{title:{{display:true,text:'% deviation from control baseline'}},
          grid:{{color:'#EEEEEE'}},
          ticks:{{callback:v=>v+'%'}}}}
    }}
  }}
}});"""

    def _ll_chart_js(canvas_id: str, sig: str, color: str) -> str:
        if sig not in ml:
            return f'/* {sig} not in data */'
        info = ml[sig]
        offsets = info["offsets"]
        means   = info["means"]
        ses     = info["ses"]
        bl      = info["baseline"]
        label   = SIGNAL_LABELS.get(sig, sig)
        means_hi = [round((m or 0) + (s or 0), 5) for m, s in zip(means, ses)]
        means_lo = [round((m or 0) - (s or 0), 5) for m, s in zip(means, ses)]
        return f"""
new Chart(document.getElementById({_j(canvas_id)}), {{
  type:'line',
  data:{{
    labels:{_j(offsets)},
    datasets:[
      {{label:'±1 SE upper',data:{_j(means_hi)},borderColor:'transparent',backgroundColor:'rgba({color},0.15)',fill:'+1',pointRadius:0,tension:0.2}},
      {{label:'±1 SE lower',data:{_j(means_lo)},borderColor:'transparent',fill:false,pointRadius:0,tension:0.2}},
      {{label:{_j(label)},data:{_j(means)},borderColor:'rgb({color})',backgroundColor:'rgb({color})',fill:false,tension:0.2,pointRadius:3,borderWidth:2}},
      {{label:'Control baseline',data:{_j([round(bl,5)]*len(offsets))},borderColor:'rgba({color},0.5)',borderDash:[5,4],pointRadius:0,fill:false,borderWidth:1.5}}
    ]
  }},
  options:{{maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{title:{{display:true,text:'Months to shortage onset (0 = onset month)'}},grid:{{display:false}}}},
             y:{{beginAtZero:true,grid:{{color:'#EEEEEE'}}}}}}}}
}});"""

    redica_js = (
        _ll_chart_js("llR1", "redica_n_oai",          "224, 122, 95") +
        _ll_chart_js("llR2", "redica_n_vai",           "28, 114, 147") +
        _ll_chart_js("llR3", "redica_n_inspections",   "80, 140, 60")
    )
    faers_js = (
        _ll_chart_js("llF1", "faers_severity_score_w3m", "2, 99, 176") +
        _ll_chart_js("llF2", "faers_n_serious_w3m",       "28, 114, 147") +
        _ll_chart_js("llF3", "faers_n_reports_w3m",       "224, 122, 95")
    )
    cross_signal_js = ""
    text_ll_js = (
        _ll_chart_js("llT5", "n_repeat_483_last_24mo",   "180, 80, 180") +
        _ll_chart_js("llT6", "n_flagged_483_last_24mo",  "224, 122, 95")
    )

    # (composite-index charts removed — indices dropped from the analysis)
    text_js = ""
    text_group_js = ""

    # ── m12: forward-validation charts (escalation + recalls) ─────────────────
    gx = d.get("grid_extra", {})
    fwd_n = int(gx.get("n_snapshots") or 0)
    esc_base = gx.get("esc24_base", 0)
    rec_base = gx.get("rec24_base", 0)

    def _split_chart_js(canvas_id: str, cells: list[dict], x_title: str) -> str:
        if not cells:
            return f'/* {canvas_id}: no grid data */'
        labels = [c["label"] for c in cells]
        hi = [c["hi"] for c in cells]
        lo = [c["lo"] for c in cells]
        lifts = [c["lift"] for c in cells]
        return f"""
(function(){{
const LIFTS = {_j(lifts)};
new Chart(document.getElementById({_j(canvas_id)}),{{
  type:'bar',
  data:{{
    labels:{_j(labels)},
    datasets:[
      {{label:'Above median',data:{_j(hi)},backgroundColor:'rgba(224,122,95,0.85)',borderRadius:3}},
      {{label:'At/below median',data:{_j(lo)},backgroundColor:'rgba(28,114,147,0.75)',borderRadius:3}}
    ]
  }},
  options:{{maintainAspectRatio:false,indexAxis:'y',
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}},
      tooltip:{{callbacks:{{afterBody:items=>'Lift: '+LIFTS[items[0].dataIndex]+'x'}}}}}},
    scales:{{
      x:{{beginAtZero:true,title:{{display:true,text:{_j(x_title)}}},ticks:{{callback:v=>v+'%'}}}},
      y:{{grid:{{display:false}},ticks:{{font:{{size:10.5}}}}}}
    }}
  }}
}});
}})();"""

    fwd_js = (
        _split_chart_js("escChart", d.get("grid_esc", []),
                        "% followed by OAI or Warning Letter within 24 months") +
        _split_chart_js("recChart", d.get("grid_rec", []),
                        "% followed by a drug recall at the facility within 24 months")
    )

    # dynamic narrative bullets for 6A
    def _fmt_cell(cells, label):
        for c in cells:
            if c["label"] == label:
                return f"{c['hi']}% vs {c['lo']}% ({c['lift']}×)"
        return "—"
    esc_repeat    = _fmt_cell(d.get("grid_esc", []), "Repeat violations (LLM)")
    esc_cross     = _fmt_cell(d.get("grid_esc", []), "Cross-insp. repeat (algo)")
    esc_scope_fw  = _fmt_cell(d.get("grid_esc", []), "Facility-wide scope")
    esc_contam    = _fmt_cell(d.get("grid_esc", []), "Contamination (LLM)")
    rec_bldg      = _fmt_cell(d.get("grid_rec", []), "Buildings/equipment violations")
    sev12_lift    = gx.get("sev_esc12_lift")
    sev12_txt     = f"{sev12_lift}×" if sev12_lift else "—"
    remed_rho     = gx.get("remed_none_shdur")
    remed_txt     = f"ρ = +{remed_rho}" if remed_rho else "—"
    invest_rho    = gx.get("invest_shdelta")
    invest_txt    = f"ρ = +{invest_rho}" if invest_rho else "—"
    cross_lift    = gx.get("cross_repeat_esc24")
    cross_txt     = f"{cross_lift}×" if cross_lift else "—"
    scope_fw_lift = gx.get("scope_fw_esc24")
    scope_fw_txt  = f"{scope_fw_lift}×" if scope_fw_lift else "—"

    # ── m11: FEI drill-down table (text-covered facilities) ───────────────────
    fei_sum = d.get("fei_summary", [])
    cov = [r for r in fei_sum if (r.get("n_483_snapshots") or 0) > 0]
    cov.sort(key=lambda r: ((r.get("n_high_risk_483_snapshots") or 0),
                            (r.get("n_oai_inspections") or 0)), reverse=True)
    n_total_feis = len(fei_sum)
    n_cov_feis = len(cov)
    fei_rows = []
    for r in cov:
        flags = []
        if r.get("any_recall_within_24mo"):
            flags.append("recall ≤24m")
        if r.get("any_shortage_within_24mo"):
            flags.append("shortage ≤24m")
        hr_n = r.get("n_high_risk_483_snapshots") or 0
        hr_style = ' style="color:#C0392B;font-weight:700;"' if hr_n > 0 else ""
        fei_rows.append(
            f"<tr><td>{r['fei']}</td><td>{r.get('firm_name','')}</td>"
            f"<td>{r.get('apis_made','')}</td>"
            f"<td class='num'>{r.get('n_483_snapshots',0)}</td>"
            f"<td class='num'{hr_style}>{hr_n}</td>"
            f"<td class='num'>{r.get('n_oai_inspections',0)}</td>"
            f"<td class='num'>{r.get('n_vai_inspections',0)}</td>"
            f"<td class='num'>{r.get('n_drug_recalls',0)}</td>"
            f"<td class='num'>{r.get('n_class_I_recalls',0)}</td>"
            f"<td>{', '.join(flags) or '—'}</td></tr>")
    fei_table_rows = "\n".join(fei_rows)

    # (static timeline PNG embeds removed — Section 5B is now interactive)

    # ── Drug-level cross-signal scatter: Redica vs FAERS ─────────────────────
    tf2 = d.get("by_drug", [])
    def _scatter_color(starts):
        if starts == 0:  return "rgba(28,114,147,0.80)"
        if starts >= 3:  return "rgba(203,75,75,0.85)"
        return "rgba(2,99,176,0.80)"

    sc_oai_faers = [{"x": r["redica_oai"], "y": r["faers_serious"],
                     "name": r["drug"], "starts": r["starts"]} for r in tf2]
    sc_483_faers = [{"x": r["redica_483"], "y": r["faers_reports"],
                     "name": r["drug"], "starts": r["starts"]} for r in tf2]
    sc_vai_faers = [{"x": r["redica_vai"], "y": r["faers_serious"],
                     "name": r["drug"], "starts": r["starts"]} for r in tf2]
    scatter_colors = [_scatter_color(r["starts"]) for r in tf2]

    drug_scatter_js = f"""
new Chart(document.getElementById('scOaiFaers'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(sc_oai_faers)},
    backgroundColor:{_j(scatter_colors)},pointRadius:7,pointHoverRadius:9}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: OAI=${{ctx.raw.x}}, FAERS serious=${{ctx.raw.y}}`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Mean OAI inspections/yr (Redica)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean serious FAERS reports/yr'}},beginAtZero:true}}
    }}}}
}});
new Chart(document.getElementById('sc483Faers'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(sc_483_faers)},
    backgroundColor:{_j(scatter_colors)},pointRadius:7,pointHoverRadius:9}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: 483-crit=${{ctx.raw.x}}, FAERS=${{ctx.raw.y}}`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Mean 483 critical obs/yr (Redica)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean total FAERS reports/yr'}},beginAtZero:true}}
    }}}}
}});
new Chart(document.getElementById('scVaiFaers'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(sc_vai_faers)},
    backgroundColor:{_j(scatter_colors)},pointRadius:7,pointHoverRadius:9}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: VAI=${{ctx.raw.x}}, FAERS serious=${{ctx.raw.y}}`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Mean VAI inspections/yr (Redica)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean serious FAERS reports/yr'}},beginAtZero:true}}
    }}}}
}});"""

    # ── Text detail group comparison: violation categories + remediation ─────
    tdg = d.get("text_detail_group", {})
    text_detail_group_js = ""
    if tdg:
        n_s2  = tdg.get("n_shortage", "?")
        n_ns2 = tdg.get("n_no_shortage", "?")
        text_detail_group_js = f"""
new Chart(document.getElementById('textDetailChart'),{{
  type:'bar',
  data:{{
    labels:{_j(tdg["labels"])},
    datasets:[
      {{label:'≥1 shortage (n={n_s2} drugs)',data:{_j(tdg["shortage"])},
        backgroundColor:'rgba(224,122,95,0.8)',borderRadius:3}},
      {{label:'No shortage (n={n_ns2} drugs)',data:{_j(tdg["no_shortage"])},
        backgroundColor:'rgba(28,114,147,0.7)',borderRadius:3}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}}}}}}}},
    scales:{{
      x:{{grid:{{display:false}},ticks:{{font:{{size:9}}}}}},
      y:{{beginAtZero:true,max:100,
           title:{{display:true,text:'Mean % of observations (0–100)'}}}}
    }}
  }}
}});"""

    # ── RF feature importance + ablation ──────────────────────────────────────
    fi_data = d.get("rf_importance", [])
    abl_data = d.get("ablation", [])
    model_js = ""
    if fi_data:
        fi_sorted = sorted(fi_data, key=lambda x: x["imp"])
        fi_labels = [r["feature"] for r in fi_sorted]
        _TEXT_FEATS = (
            "severity_critmajor_share", "scope_facilitywide_share", "scope_multipleproducts_share",
            "cultural_root_cause_share", "capital_root_cause_share",
            "remediation_none_share", "remediation_weak_share",
            "repeat_llm_share", "contamination_llm_share", "data_integrity_llm_share", "investigation_llm_share",
            "repeat_llm_only_share", "contamination_llm_only_share",
            "oos_oot_regex_share", "wl_ref_regex_share",
            "repeat_cross_insp_share", "vc_labcontrols_share", "vc_buildingsequipment_share",
        )
        fi_colors = [
            "rgba(28,114,147,0.85)" if r["feature"] in _TEXT_FEATS
            else "rgba(2,99,176,0.65)" for r in fi_sorted
        ]
        fi_vals = [r["imp"] for r in fi_sorted]
        abl_l2   = next((r for r in abl_data if r["model"] == "L2_logit"), {})
        abl_rf   = next((r for r in abl_data if r["model"] == "RandomForest"), {})
        auc_with = round(float(abl_rf.get("auc_with_text", 0)), 3)
        auc_no   = round(float(abl_rf.get("auc_without_text", 0)), 3)
        delta_rf = round(float(abl_rf.get("auc_delta", 0)), 3)
        model_js = f"""
new Chart(document.getElementById('fiChart'),{{
  type:'bar',
  data:{{
    labels:{_j(fi_labels)},
    datasets:[{{
      label:'RF importance',
      data:{_j(fi_vals)},
      backgroundColor:{_j(fi_colors)},
      borderRadius:3
    }}]
  }},
  options:{{indexAxis:'y',maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{beginAtZero:true,title:{{display:true,text:'Importance'}}}},
      y:{{grid:{{display:false}},ticks:{{font:{{size:10}}}}}}
    }}
  }}
}});
(function(){{
  const el = document.getElementById('modelKeyNums');
  if (!el) return;
  el.innerHTML = `
    <strong>RandomForest CV (GroupKFold by drug)</strong><br>
    AUC with text features: <strong>{auc_with}</strong> &nbsp;|&nbsp; without: <strong>{auc_no}</strong>
    &nbsp;→ <strong style="color:var(--accent);">Δ = {delta_rf:+.3f}</strong><br>
    <br>
    <strong>Honest read:</strong> at drug-year level (n=126 rows, 19 events), the raw 483 text
    features do <em>not</em> improve prediction. The validated text signal is at the
    <strong>facility level</strong> (Section 5) — aggregating to drug-year across many facilities
    dilutes it.<br>
    <br>
    <span style="color:var(--muted);font-size:11px;">
      Logit AUC with text: {round(float(abl_l2.get("auc_with_text",0)),3)} &nbsp;|&nbsp;
      without: {round(float(abl_l2.get("auc_without_text",0)),3)}.
      Recalls excluded from features — concurrent/lagging w.r.t. shortage onset, not leading indicators.<br>
      Next step: a facility-level model using the dated 483 snapshots directly.
    </span>
  `;
}})();"""

    # ── OAI forward study JS ───────────────────────────────────────────────────
    oai_fwd    = d.get("oai_fwd", {})
    oai_events = d.get("oai_events", {})
    oai_fwd_js = ""
    if oai_fwd:
        oai_offsets  = oai_fwd["offsets"]
        oai_rates    = oai_fwd["rates"]
        oai_ses      = oai_fwd["ses"]
        oai_baseline = oai_fwd["baseline"]
        oai_hi = [round((r or 0) + (s or 0), 6) for r, s in zip(oai_rates, oai_ses)]
        oai_lo = [round((r or 0) - (s or 0), 6) for r, s in zip(oai_rates, oai_ses)]
        oai_bl_line  = [round(oai_baseline, 6)] * len(oai_offsets)
        oai_colors = [
            "rgba(203,75,75,0.85)" if o <= 0 else "rgba(2,99,176,0.85)"
            for o in oai_offsets
        ]
        # Event-level stacked bar data
        n_already      = oai_events.get("n_already", 0)
        n_fresh        = oai_events.get("n_fresh", 0)
        n_already_fwd  = oai_events.get("n_already_fwd", 0)
        n_fresh_fwd    = oai_events.get("n_fresh_fwd", 0)
        n_already_no   = n_already - n_already_fwd
        n_fresh_no     = n_fresh   - n_fresh_fwd
        mean_mo        = oai_events.get("mean_months_fwd", 0)
        oai_fwd_js = f"""
new Chart(document.getElementById('oaiFwdChart'),{{
  type:'line',
  data:{{
    labels:{_j(oai_offsets)},
    datasets:[
      {{label:'\\u00b11 SE band',data:{_j(oai_hi)},borderColor:'transparent',
        backgroundColor:'rgba(2,99,176,0.12)',fill:'+1',pointRadius:0,tension:0.2}},
      {{label:'_lo',data:{_j(oai_lo)},borderColor:'transparent',
        fill:false,pointRadius:0,tension:0.2}},
      {{label:'% months drug is in shortage',data:{_j(oai_rates)},
        borderColor:'rgba(2,99,176,0.6)',backgroundColor:{_j(oai_colors)},
        fill:false,tension:0.15,pointRadius:5,borderWidth:2,
        pointBackgroundColor:{_j(oai_colors)}}},
      {{label:'Control baseline (no OAI \u00b112m): {oai_baseline:.3f}',
        data:{_j(oai_bl_line)},
        borderColor:'#888',borderDash:[5,4],pointRadius:0,fill:false,borderWidth:1.5}}
    ]
  }},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{position:'bottom',labels:{{
      boxWidth:12,font:{{size:10.5}},
      filter:function(item){{return item.text!=='_lo';}}
    }}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Months relative to OAI inspection (0 = OAI month)'}},
           grid:{{display:false}}}},
      y:{{beginAtZero:true,max:1,
           title:{{display:true,text:'Fraction of events: drug in shortage'}},
           grid:{{color:'#EEEEEE'}},
           ticks:{{callback:function(v){{return (v*100).toFixed(0)+'%';}}}}}}
    }}
  }}
}});"""

    qs_js = ""  # quality-split chart removed with the condensed Wang section

    # (Recall circularity analysis removed — recalls excluded from predictive features
    #  as they are concurrent/lagging w.r.t. shortage onset, not leading indicators.)

    # ── Interactive FEI timeline (Section 5B) ─────────────────────────────────
    fei_events_data = d.get("fei_events", {})
    timeline_js = f"""
const FEI_EVENTS = {_j(fei_events_data)};
let _tChart = null;
(function(){{
  const sel = document.getElementById('feiSelector');
  if (!sel || !Object.keys(FEI_EVENTS).length) {{
    if (sel) {{ sel.innerHTML='<option>No data — run m11 first</option>'; sel.disabled=true; }}
    return;
  }}
  const feiList = Object.keys(FEI_EVENTS).sort((a,b) => {{
    const da=FEI_EVENTS[a], db=FEI_EVENTS[b];
    return (db.n_hr+db.n_oai*2+db.n_recalls)-(da.n_hr+da.n_oai*2+da.n_recalls);
  }});
  feiList.forEach(fei => {{
    const ev=FEI_EVENTS[fei];
    const opt=document.createElement('option');
    opt.value=fei;
    opt.textContent=ev.firm+' (FEI '+fei+')'+(ev.apis?' · '+ev.apis:'');
    sel.appendChild(opt);
  }});
  function buildTimeline(fei) {{
    const ev=FEI_EVENTS[fei]; if(!ev) return;
    const stats=document.getElementById('feiStats');
    if(stats) stats.innerHTML=
      '<strong>'+ev.n_483+'</strong> 483s (<span style="color:#C0392B;font-weight:700;">'+
      ev.n_hr+' high-risk</span>) · <strong>'+ev.n_oai+'</strong> OAI · <strong>'+
      ev.n_recalls+'</strong> recalls';
    const datasets=[];
    const ns=ev.shortage_bands.length;
    ev.shortage_bands.forEach((s,i)=>{{
      const y=0.07*(i-(ns-1)/2);
      datasets.push({{label:i===0?'Shortage period':'_hide_',type:'line',
        data:[{{x:s.start,y:y,label:s.drug+' — shortage start'}},
              {{x:s.end,  y:y,label:s.drug+' — resolved'}}],
        borderColor:'rgba(100,100,100,0.5)',borderWidth:10,pointRadius:2,fill:false,order:0}});
    }});
    const hr=ev.events_483.filter(e=>e.hr);
    if(hr.length) datasets.push({{
      label:'483 Red-flagged',type:'scatter',
      data:hr.map(e=>({{x:e.yr,y:2,label:e.label,n_obs:e.n_obs,sev:e.sev}})),
      backgroundColor:'rgba(192,57,43,0.88)',borderColor:'#922B21',borderWidth:1,
      pointRadius:ctx=>Math.max(6,Math.min(18,(ctx.raw?.n_obs||1)*1.8)),
      pointHoverRadius:14,order:2}});
    const ok=ev.events_483.filter(e=>!e.hr);
    if(ok.length) datasets.push({{
      label:'483 Normal',type:'scatter',
      data:ok.map(e=>({{x:e.yr,y:2,label:e.label,n_obs:e.n_obs,sev:e.sev}})),
      backgroundColor:'rgba(28,114,147,0.7)',borderColor:'#065A82',borderWidth:1,
      pointRadius:ctx=>Math.max(5,Math.min(14,(ctx.raw?.n_obs||1)*1.5)),
      pointHoverRadius:10,order:2}});
    const clsC={{OAI:'rgba(192,57,43,0.9)',VAI:'rgba(230,126,34,0.9)',NAI:'rgba(39,174,96,0.9)'}};
    ['OAI','VAI','NAI'].forEach(cls=>{{
      const evs=ev.events_insp.filter(e=>e.cls===cls); if(!evs.length) return;
      datasets.push({{label:'Inspection: '+cls,type:'scatter',
        data:evs.map(e=>({{x:e.yr,y:3,label:e.label}})),
        backgroundColor:clsC[cls],borderColor:clsC[cls],
        pointStyle:'triangle',pointRadius:9,pointHoverRadius:12,order:1}});
    }});
    const rclsC={{'Class I':'rgba(139,0,0,0.9)','Class II':'rgba(204,85,0,0.9)','Class III':'rgba(184,134,11,0.9)'}};
    Object.entries(rclsC).forEach(([cls,color])=>{{
      const evs=ev.events_recall.filter(e=>e.cls===cls); if(!evs.length) return;
      datasets.push({{label:'Recall: '+cls,type:'scatter',
        data:evs.map(e=>({{x:e.yr,y:1,label:e.label}})),
        backgroundColor:color,borderColor:color,
        pointStyle:'rectRot',pointRadius:9,pointHoverRadius:12,order:1}});
    }});
    if(_tChart) _tChart.destroy();
    const ctx=document.getElementById('timelineChart'); if(!ctx) return;
    _tChart=new Chart(ctx,{{
      data:{{datasets}},
      options:{{maintainAspectRatio:false,
        scales:{{
          x:{{type:'linear',min:2008,max:2026,
              title:{{display:true,text:'Year'}},
              ticks:{{stepSize:1,callback:v=>Number.isInteger(v)?v:''}},
              grid:{{color:'#EEE'}}}},
          y:{{min:-0.7,max:3.7,
              ticks:{{stepSize:1,callback:v=>{{
                const m={{0:'Shortages',1:'Recalls',2:'483 Snapshots',3:'Inspections'}};
                return m[Math.round(v)]||'';
              }}}},
              grid:{{color:'#EEE'}}}}
        }},
        plugins:{{
          legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10.5}},
            filter:i=>!i.text.startsWith('_')}}}},
          tooltip:{{callbacks:{{label:ctx=>{{
            const r=ctx.raw;
            if(r.label) return r.label;
            if(r.n_obs!==undefined) return r.n_obs+' obs, sev='+r.sev;
            return '';
          }}}}}}
        }}
      }}
    }});
    const story=[];
    if(ev.n_hr>0){{
      const pct=Math.round(ev.n_hr/ev.n_483*100);
      story.push('<strong>'+ev.n_hr+'/'+ev.n_483+' ('+pct+'%)</strong> of 483 snapshots are red-flagged.');
    }}
    if(ev.n_oai>0) story.push('<strong>'+ev.n_oai+' OAI</strong> inspection outcome(s).');
    if(ev.n_recalls>0) story.push('<strong>'+ev.n_recalls+'</strong> drug recall event(s) at this facility.');
    if(ev.shortage_bands.length>0){{
      const drugs=[...new Set(ev.shortage_bands.map(s=>s.drug))].join(', ');
      story.push('Shortage periods linked to: <em>'+drugs+'</em>.');
    }}
    if(ev.n_hr>0&&ev.n_oai>0) story.push('Look for red circles (●) preceding OAI triangles (▲) on the timeline.');
    const stEl=document.getElementById('feiStory');
    if(stEl) stEl.innerHTML=story.length?story.join(' '):'No high-risk events for this facility.';
  }}
  buildTimeline(feiList[0]);
  sel.addEventListener('change',()=>buildTimeline(sel.value));
}})();"""

    # ── Drug-level text feature × shortage / FAERS scatter ────────────────────
    tvs = d.get("text_vs_shortage", [])
    tvs_colors = [
        "rgba(192,57,43,0.85)" if r["starts"] >= 3
        else ("rgba(2,99,176,0.75)" if r["starts"] >= 1 else "rgba(100,100,100,0.6)")
        for r in tvs
    ]
    tvs_dur  = [{"x": r["contam"], "y": r["duration"],
                 "name": r["drug"], "starts": r["starts"], "faers": r["faers_serious"]} for r in tvs]
    tvs_faers = [{"x": r["cross"],  "y": r["faers_serious"],
                  "name": r["drug"], "starts": r["starts"], "dur": r["duration"]} for r in tvs]
    text_vs_shortage_js = f"""
new Chart(document.getElementById('tvsChart1'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(tvs_dur)},
    backgroundColor:{_j(tvs_colors)},pointRadius:8,pointHoverRadius:11}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: ${{ctx.raw.y}}mo shortage, ${{ctx.raw.x}}% contam, ${{ctx.raw.starts}} starts`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Contamination flag share (% of obs)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Total months in shortage (2015–2024)'}},beginAtZero:true}}
    }}}}
}});
new Chart(document.getElementById('tvsChart2'),{{
  type:'scatter',
  data:{{datasets:[{{label:'Drug',data:{_j(tvs_faers)},
    backgroundColor:{_j(tvs_colors)},pointRadius:8,pointHoverRadius:11}}]}},
  options:{{maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw.name}}: FAERS serious=${{ctx.raw.y}}/yr, cross-insp repeat=${{ctx.raw.x}}%, ${{ctx.raw.dur}}mo shortage`}}}}}},
    scales:{{
      x:{{title:{{display:true,text:'Cross-inspection repeat share (% of obs)'}},beginAtZero:true}},
      y:{{title:{{display:true,text:'Mean serious FAERS reports / year'}},beginAtZero:true}}
    }}}}
}});"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>From Regulatory Text to Shortage Risk · Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--navy:#21295C;--deep:#065A82;--teal:#1C7293;--accent:#E07A5F;
      --cream:#F4F1EC;--paper:#FBFAF7;--ink:#1A2233;--muted:#5A6577;
      --rule:#E2DDD2;--white:#FFFFFF;}}
*{{box-sizing:border-box;}}
html,body{{margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      background:var(--paper);color:var(--ink);line-height:1.5;}}
.wrap{{max-width:1200px;margin:0 auto;padding:32px 28px 48px;}}
header.hero{{background:var(--navy);color:var(--white);padding:36px 28px;
             border-radius:10px;margin-bottom:28px;position:relative;overflow:hidden;}}
header.hero::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--accent);}}
.eyebrow{{color:#E7C8B9;font-size:11px;letter-spacing:4px;font-weight:700;
          text-transform:uppercase;margin-bottom:10px;}}
header.hero h1{{margin:0 0 10px;font-family:Georgia,serif;font-size:28px;font-weight:700;line-height:1.25;}}
header.hero p{{margin:0;color:#CADCFC;font-size:14px;max-width:900px;}}
section{{margin-bottom:28px;}}
.section-head{{display:flex;align-items:baseline;gap:14px;margin-bottom:14px;}}
.step-num{{display:inline-block;background:var(--accent);color:#fff;
           width:28px;height:28px;border-radius:50%;text-align:center;
           line-height:28px;font-weight:700;font-size:13px;flex-shrink:0;}}
h2{{margin:0;font-family:Georgia,serif;font-size:20px;color:var(--navy);font-weight:700;}}
.sub{{color:var(--muted);font-size:13px;margin:4px 0 14px 42px;}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
.stat{{background:var(--white);border:1px solid var(--rule);border-top:4px solid var(--deep);
       border-radius:6px;padding:16px 18px;}}
.stat .num{{font-family:Georgia,serif;font-size:30px;font-weight:700;color:var(--navy);line-height:1.1;}}
.stat .num small{{font-family:sans-serif;color:var(--muted);font-size:14px;font-weight:400;margin-left:4px;}}
.stat .lbl{{color:var(--ink);font-size:12.5px;margin-top:6px;}}
.stat.accent{{border-top-color:var(--accent);}}
.stat.teal{{border-top-color:var(--teal);}}
.sources{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}}
.src{{background:var(--white);border:1px solid var(--rule);border-left:4px solid var(--deep);
      border-radius:6px;padding:12px 14px;}}
.src .name{{font-family:Georgia,serif;font-weight:700;color:var(--navy);font-size:14px;}}
.src .role{{color:var(--muted);font-size:12px;margin-top:4px;}}
.card{{background:var(--white);border:1px solid var(--rule);border-radius:8px;padding:18px 20px;}}
.card h3{{margin:0 0 4px;font-family:Georgia,serif;font-size:16px;color:var(--navy);}}
.card .csub{{color:var(--muted);font-size:12px;margin-bottom:12px;}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.chart-row.three{{grid-template-columns:1fr 1fr 1fr;}}
.chart-host{{position:relative;height:280px;}}
.chart-host.tall{{height:360px;}}
.note{{background:var(--cream);border-left:4px solid var(--accent);
       padding:12px 14px;border-radius:4px;font-size:13px;color:var(--ink);margin-top:14px;}}
.note strong{{color:var(--navy);}}
.note.dark{{background:var(--navy);border-left-color:var(--accent);color:#CADCFC;}}
.note.dark strong{{color:var(--white);}}
.divider{{border:none;border-top:2px solid var(--accent);margin:8px 0 20px;opacity:0.3;}}
table.signals{{width:100%;border-collapse:collapse;font-size:13px;}}
table.signals th,table.signals td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--rule);}}
table.signals th{{background:var(--cream);color:var(--navy);font-weight:700;
                  font-size:11px;text-transform:uppercase;letter-spacing:1px;}}
table.signals td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.lift{{display:inline-block;padding:2px 8px;border-radius:12px;font-weight:700;font-size:12px;}}
.lift.low{{background:#E8EEF2;color:var(--deep);}}
.lift.mid{{background:#FCE6DD;color:#B14A30;}}
.lift.high{{background:var(--accent);color:white;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.col-card{{background:var(--white);border:1px solid var(--rule);border-radius:8px;overflow:hidden;}}
.col-card .col-head{{padding:10px 16px;color:white;font-size:12px;font-weight:700;
                     letter-spacing:3px;text-transform:uppercase;}}
.col-card .col-head.l{{background:var(--navy);}} .col-card .col-head.n{{background:var(--accent);}}
.col-card ul{{margin:0;padding:14px 20px 18px;}}
.col-card li{{padding:4px 0;font-size:13px;color:var(--ink);}}
.badge{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;margin-right:4px;}}
.badge.new{{background:#dcfce7;color:#166534;}}
.badge.upd{{background:#dbeafe;color:#1e40af;}}
footer{{text-align:center;color:var(--muted);font-size:11px;margin-top:26px;
        padding-top:14px;border-top:1px solid var(--rule);}}
@media(max-width:900px){{
  .stats{{grid-template-columns:repeat(2,1fr);}}
  .sources{{grid-template-columns:repeat(2,1fr);}}
  .chart-row,.chart-row.three,.two-col{{grid-template-columns:1fr;}}
}}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <div class="eyebrow">Research Dashboard · June 2026 · NLP &amp; LLM Framework</div>
  <h1>From Regulatory Text to Shortage Risk</h1>
  <p>14 generic APIs · 129 manufacturing facilities · 2015–2024 ·
     LLM extraction of FDA Form 483 narrative text → validated facility-level risk signals.</p>
</header>

<!-- ═══ KEY FINDINGS ═══ -->
<section>
  <div class="note dark" style="margin-top:0;font-size:13.5px;line-height:1.8;">
    <strong>Three findings to take away:</strong><br>
    <strong>1.</strong> When FDA issues a 483, <em>what the text says</em> predicts what happens next:
    facilities whose 483s show <strong>repeat violations, cross-inspection repeats, contamination,
    or facility-wide-scope findings</strong> are 2–7× more likely to face an OAI or Warning Letter
    — and buildings/equipment findings precede recalls — within 24 months (Section 5).<br>
    <strong>2.</strong> 483s with <strong>no remediation response</strong> in the text are followed by more
    months of drug shortage over the next 3 years (ρ = +0.33) — text quality of the response matters,
    not just the violation (Section 5).<br>
    <strong>3.</strong> These signals live at the <strong>facility level</strong>. Aggregated to drug-year
    (n=19 shortage events), they do not yet improve a prediction model (Section 6) — the next step is
    facility-level modeling, not more aggregation.
  </div>
</section>

<!-- ═══ SECTION 1: PANEL SNAPSHOT ═══ -->
<section>
  <div class="section-head"><span class="step-num">1</span><h2>Panel at a glance</h2></div>
  <div class="sub">Both the annual pilot panel and the new monthly panel cover the same 14 Valisure-tested APIs and 2015–2024.</div>
  <div class="stats">
    <div class="stat">
      <div class="num">{d["annual_n_drugs"]}<small>APIs</small></div>
      <div class="lbl">Valisure-tested pilot universe</div>
    </div>
    <div class="stat">
      <div class="num">{d["annual_rows"]}<small>rows</small></div>
      <div class="lbl">Annual panel (drug × year, 2015–2024)</div>
    </div>
    <div class="stat accent">
      <div class="num">{d["annual_events"]}<small>starts</small></div>
      <div class="lbl">New shortage episodes (annual)</div>
    </div>
    <div class="stat teal">
      <div class="num">{d["monthly_rows"]}<small>rows</small></div>
      <div class="lbl">Monthly panel (drug × month) <span class="badge new">new</span></div>
    </div>
  </div>
</section>

<!-- ═══ SECTION 2: SOURCES ═══ -->
<section>
  <div class="section-head"><span class="step-num">2</span><h2>Data sources &amp; linkage</h2></div>
  <div class="sources">
    <div class="src"><div class="name">FDA Form 483s</div><div class="role">38 FEIs · 622 observations · LLM-extracted (primary text source)</div></div>
    <div class="src"><div class="name">FDA Inspections</div><div class="role">1,019 inspections · 961 CFR citations · 127/129 FEIs</div></div>
    <div class="src"><div class="name">Valisure</div><div class="role">14-API universe · 2024 independent quality test scores (static)</div></div>
    <div class="src"><div class="name">UUtah</div><div class="role">Shortage start / end dates → annual &amp; monthly shortage state</div></div>
    <div class="src"><div class="name">FAERS</div><div class="role">Adverse-event reports + severity (quarterly resolution)</div></div>
  </div>
  <div class="note" style="margin-top:12px;">
    <strong>Primary text source:</strong> FDA Form 483 PDF narratives, processed via LangChain OCR + LLM extraction (GPT-4o-mini).
    Structured citations provide CFR-level signals for 127/129 facilities.
    Valisure scores are a 2024 static snapshot used for cross-sectional validation only.
    <strong>Recalls excluded as predictors</strong> — timing analysis confirms they are concurrent or lagging w.r.t. shortage onset, not leading indicators.
  </div>
</section>

<!-- ═══ SECTION 3: SHORTAGE PATTERNS ═══ -->
<section>
  <div class="section-head"><span class="step-num">3</span><h2>Shortage patterns (2015–2024)</h2></div>
  <div class="chart-row">
    <div class="card">
      <h3>New shortage starts by year</h3>
      <div class="csub">Annual count across all 14 pilot APIs</div>
      <div class="chart-host"><canvas id="chartYear"></canvas></div>
    </div>
    <div class="card">
      <h3>Total months in shortage per drug (2015–2024)</h3>
      <div class="csub">Sorted by shortage duration · red ≥ 60 months · teal = no shortage · tooltip shows # onset events</div>
      <div class="chart-host tall"><canvas id="chartDrug"></canvas></div>
    </div>
  </div>
  <div class="note">
    <strong>Pattern:</strong> 21 shortage-start events across 14 drugs ({d["monthly_onset_months"]} monthly onset months).
    Metoprolol and Vancomycin were in shortage for the entire 10-year window (120 months), though
    Vancomycin had zero onset events — it entered shortage before 2015 and never resolved.
    Metronidazole and Potassium chloride had the most separate onset episodes.
    Three drugs had zero total months in shortage (Bupropion, Atorvastatin, Tacrolimus — short episodes only).
  </div>
</section>


<!-- ═══ SECTION 5: EDA — REGULATORY & FAERS RELATIONSHIPS ═══ -->
<section>
  <div class="section-head">
    <span class="step-num">4</span>
    <h2>EDA — Signals before shortage onset</h2>
  </div>
  <div class="sub">
    Monthly event study: mean signal value at each offset month relative to shortage onset (month 0).
    Control baseline = drug-months with no shortage onset within ±12 months.
    Shaded band = ±1 SE. <strong>N = {d["monthly_onset_months"]} onset months across 14 drugs — interpret as exploratory only.</strong>
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:0 0 4px 0;font-size:16px;">
    4A · 483 text features — shortage vs no-shortage drugs</h3>
  <div class="sub" style="margin-left:0;">
    Raw LLM/regex-extracted feature shares, averaged across all facility 483s for each drug.
    Two groups: drugs with ≥1 shortage onset vs drugs with zero onsets (2015–2024).
    <br><strong>≥1 shortage onset:</strong> Ampicillin, Atorvastatin, Calcium Gluconate,
    Lisinopril, Magnesium Sulfate, Metformin, Metoprolol, Metronidazole, Pantoprazole, Potassium Chloride, Tacrolimus.
    <br><strong>0 shortage onsets:</strong> Ampicillin+Sulbactam, Bupropion, Vancomycin.
  </div>
  <div class="card">
    <h3>483 text features — % of observations, drugs with vs without shortages</h3>
    <div class="csub">8 features including new: cross-inspection repeat, facility-wide scope, cultural root cause. Each bar = mean share across a drug group's 483s.</div>
    <div class="chart-host"><canvas id="textDetailChart"></canvas></div>
  </div>
  <div class="note dark">
    Shortage drugs show more contamination-flagged observations and higher rates of no/weak remediation.
    This is a cross-sectional comparison (each drug's full 483 history) — the temporal evidence is in
    Sections 4C and 5.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:16px 0 4px;font-size:16px;">
    4B · Monthly lead-lag — Regulatory signals (OAI, VAI, Total Inspections)</h3>
  <div class="sub" style="margin-left:0;">
    OAI = Official Action Indicated (most severe; triggers mandatory remediation).
    VAI = Voluntary Action Indicated (violations noted, manufacturer commits to fix voluntarily).
    Total inspections shows inspection frequency context.
    X-axis = months to shortage onset (0 = onset month). Shaded = ±1 SE.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:4px;">
    <div class="card"><h3>OAI Inspections</h3><div class="chart-host"><canvas id="llR1"></canvas></div></div>
    <div class="card"><h3>VAI Inspections</h3><div class="chart-host"><canvas id="llR2"></canvas></div></div>
    <div class="card"><h3>Total Inspections</h3><div class="chart-host"><canvas id="llR3"></canvas></div></div>
  </div>
  <div class="note">
    OAI events are sparse; no consistent ramp-up before onset is visible at this sample size.
    VAI inspections provide a broader signal of manufacturing concerns being addressed voluntarily.
    Total inspections are flat — shortage onset is not preceded by a surge in inspection activity.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:16px 0 4px;font-size:16px;">
    4C · Monthly lead-lag — time-varying text signals before shortage onset</h3>
  <div class="sub" style="margin-left:0;">
    Both charts show a trailing-24-month count of 483 documents at the drug's facilities,
    aggregated to drug-month. Solid line = mean in the 12 months before shortage onset;
    dashed = control baseline (drug-months far from any shortage onset). A solid line above baseline
    means the drug entered shortage with more recent high-risk 483s than usual.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:4px;">
    <div class="card">
      <h3>Repeat-violation 483s, trailing 24 months</h3>
      <div class="csub">Count of 483 documents containing ≥1 repeat-violation finding, across all drug facilities.</div>
      <div class="chart-host"><canvas id="llT5"></canvas></div>
    </div>
    <div class="card">
      <h3>Red-flagged 483s, trailing 24 months</h3>
      <div class="csub">Count of 483 documents meeting ≥2 of 4 risk markers (repeat, contamination, OOS/OOT, Critical+Major severity).</div>
      <div class="chart-host"><canvas id="llT6"></canvas></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;justify-content:center;padding:24px;">
      <h3 style="margin-bottom:10px;">How to read this</h3>
      <div style="font-size:12px;line-height:1.7;color:var(--ink);">
        These are the only <strong>time-varying</strong> text features — each month's value depends
        on when recent 483s were issued and what they said.<br><br>
        The facility-level validation (Section 5) shows the <em>content</em> of these 483s predicts
        escalation and recalls 2–7×. Here we ask: does that content also accumulate
        <strong>before drug shortages</strong>?<br><br>
        n = {d["monthly_onset_months"]} shortage-onset months, 14 drugs — exploratory.
      </div>
    </div>
  </div>
</section>

<!-- ═══ SECTION 4D: TEXT FEATURES vs. SHORTAGE & ADVERSE EVENTS (drug-level) ═══ -->
<section>
  <div class="section-head"><span class="step-num" style="background:var(--teal);">4D</span>
    <h2>Text risk vs. shortage burden and adverse events — drug-level</h2></div>
  <div class="sub">
    Each point = one of the 14 pilot APIs. X = text risk feature aggregated across the drug's facilities;
    Y = outcome. <strong>Red = ≥3 shortage starts, blue = 1–2 starts, gray = no starts.</strong>
    Hover for drug name and values. These are observational associations at n=14; treat as directional only.
  </div>
  <div class="chart-row">
    <div class="card">
      <h3>Contamination flag share vs. months in shortage</h3>
      <div class="csub">X = % of 483 observations flagged for contamination (LLM, semantic lift).
        Y = total months drug was in shortage 2015–2024.</div>
      <div class="chart-host tall"><canvas id="tvsChart1"></canvas></div>
    </div>
    <div class="card">
      <h3>Cross-inspection repeat share vs. FAERS serious reports</h3>
      <div class="csub">X = % of 483 observations where same deficiency cited across inspections (algorithmic).
        Y = mean annual serious adverse-event reports (FAERS).</div>
      <div class="chart-host tall"><canvas id="tvsChart2"></canvas></div>
    </div>
  </div>
  <div class="note">
    <strong>Pattern:</strong> drugs whose facilities show higher contamination flags tend to have longer
    shortage histories; drugs with more cross-inspection repeats also see more serious adverse events —
    consistent with a systemic quality problem. Caveat: only 10/14 drugs have any 483 text coverage;
    the remaining 4 plot at zero (not zero risk — just no text data).
  </div>
</section>

<!-- ═══ SECTION 5: FACILITY-LEVEL TEXT SIGNAL VALIDATION ═══ -->
<section>
  <div class="section-head"><span class="step-num">5</span><h2>Facility-level validation — is the 483 text a real signal?</h2></div>
  <div class="sub">
    <strong>Scope: the text-covered subset.</strong> Of the 129 FEIs manufacturing the 14 Valisure APIs,
    <strong>{n_cov_feis} have publicly available 483 PDFs</strong> ({fwd_n} dated snapshots scored by the LLM).
    Everything in this section uses only that subset; all other sections use all 129 FEIs.
    The question here: given that FDA issued a 483, does the <em>narrative content</em> of that document
    predict what happens to the facility next — beyond the mere fact that a 483 exists?
  </div>
  <div class="note" style="margin:0 0 14px;">
    <strong>Two definitions used throughout this section:</strong><br>
    <strong>Snapshot</strong> = one FDA Form 483 document for one inspection of one facility,
    dated by the inspection end date. The LLM scores every observation in the document; the snapshot
    carries the shares (e.g., "40% of observations flagged contamination").<br>
    <strong>Red flag</strong> (red markers in the timelines below) = a snapshot meeting <strong>≥2 of 4
    validated risk markers</strong>: any repeat-violation finding; contamination share above the sample
    median (25%); OOS/OOT-reference share above median (17%); high-severity share above median (67%).
    These four markers are exactly the features validated in 5A — the flag is shorthand, not a new index.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:0 0 4px 0;font-size:16px;">
    5A · Forward validation — what the 483 text predicts, by outcome</h3>
  <div class="sub" style="margin-left:0;">
    Each of the {fwd_n} snapshots is split at the median of each text feature; bars compare outcome
    rates for the two halves. Every facility here received a 483 — the document's <em>existence</em>
    is held constant, so any difference comes from <strong>what the text says</strong>.
    Base rates: escalation {esc_base}%, recall {rec_base}%. Features shown are those with consistent
    direction across 12/24/36-month horizons (full grid: <code>outputs/tables/text_signal_grid.csv</code>).
  </div>
  <div class="chart-row">
    <div class="card">
      <h3>Regulatory escalation — OAI or Warning Letter within 24m</h3>
      <div class="csub">Orange = above feature median; teal = at/below. Hover for lift.</div>
      <div class="chart-host tall"><canvas id="escChart"></canvas></div>
    </div>
    <div class="card">
      <h3>Drug recall at the facility within 24m</h3>
      <div class="csub">Same median split. Recall@12m excluded (only 2 events).</div>
      <div class="chart-host tall"><canvas id="recChart"></canvas></div>
    </div>
  </div>
  <div class="note dark">
    <strong>Different text features predict different failures.</strong>
    <em>Repeat violations</em> ({esc_repeat} escalation) and <em>cross-inspection repeats</em>
    ({cross_txt} — same deficiency cited at an earlier inspection, detected algorithmically) are the
    strongest escalation predictors; <em>contamination</em> ({esc_contam}) and
    <em>facility-wide scope</em> ({scope_fw_txt}) also predict escalation.
    <em>Buildings/equipment violations</em> ({rec_bldg}) and capital/cultural root causes predict recalls —
    physical-asset and management failures produce defective product.
    <em>Critical+Major severity</em> works best at the short 12-month horizon ({sev12_txt} escalation lift):
    FDA acts fast on documented defects.
    For <strong>shortage burden</strong>: 483s with <em>no remediation response</em> are followed by
    more months of shortage over 3 years ({remed_txt}), and <em>failed investigations</em> by a worsening
    shortage trajectory ({invest_txt}).
    n = {fwd_n} snapshots across {n_cov_feis} FEIs — exploratory, suggestive rather than definitive.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:16px 0 4px;font-size:16px;">
    5B · Facility timeline explorer — interactive</h3>
  <div class="sub" style="margin-left:0;">
    Select any of the {n_cov_feis} text-covered facilities. Four lanes:
    <strong>▲ Inspections</strong> (red=OAI, orange=VAI, green=NAI) ·
    <strong>● 483 Snapshots</strong> (red=red-flagged, blue=normal; circle size = observation count) ·
    <strong>◆ Recalls</strong> (dark-red=Class I) ·
    <strong>── Shortages</strong> (gray bar spanning shortage period).
    Hover any point for details. Sorted by risk (high-risk 483s + OAI count).
  </div>
  <div class="card" style="margin-bottom:4px;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;">
      <label style="font-weight:700;color:var(--navy);white-space:nowrap;">Facility:</label>
      <select id="feiSelector" style="font-size:13px;padding:6px 10px;border:1px solid var(--rule);
              border-radius:4px;background:var(--white);color:var(--ink);flex:1;min-width:260px;max-width:560px;">
      </select>
      <div id="feiStats" style="font-size:12px;color:var(--muted);"></div>
    </div>
    <div style="position:relative;height:310px;"><canvas id="timelineChart"></canvas></div>
    <div id="feiStory" class="note" style="margin-top:12px;font-size:13px;"></div>
  </div>
  <div class="note" style="margin-top:8px;">
    Caution: temporal co-occurrence only. A facility may make many drugs; a shortage may have
    causes unrelated to this facility. These timelines illustrate the mechanism, they do not establish attribution.
  </div>

  <hr class="divider"/>
  <h3 style="font-family:Georgia,serif;color:var(--navy);margin:16px 0 4px;font-size:16px;">
    5C · Facility drill-down — all {n_cov_feis} text-covered FEIs</h3>
  <div class="sub" style="margin-left:0;">
    Sorted by red-flagged snapshot count, then OAI inspections. Red-flag definition at the top of this
    section. Per-FEI timeline charts for all facilities are in <code>outputs/figures/timelines/</code>.
  </div>
  <table class="signals">
    <thead><tr>
      <th>FEI</th><th>Facility</th><th>APIs</th>
      <th>483 snapshots</th><th>Red-flagged</th><th>OAI</th><th>VAI</th>
      <th>Recalls</th><th>Class I</th><th>Events ≤24m after red flag</th>
    </tr></thead>
    <tbody>
    {fei_table_rows}
    </tbody>
  </table>
</section>

<!-- ═══ SECTION 6: PREDICTIVE MODEL RESULTS ═══ -->
<section>
  <div class="section-head"><span class="step-num">6</span><h2>Predictive Model — drug-year baseline (honest negative)</h2></div>
  <div class="sub">
    <strong>Setup:</strong> Drug × year panel (14 APIs, 2015–2024, n=126 drug-years, 19 shortage-onset events).
    Outcome = <em>y_next_year_shortage</em>: does this drug start a shortage in the following year?
    Cross-validation: GroupKFold by drug (model tested on drugs it has never seen).
    Recall features excluded (concurrent/lagging with onset = leakage).
  </div>
  <div class="chart-row">
    <div class="card">
      <h3>RandomForest feature importance</h3>
      <div class="csub">Teal bars = 483 text features (raw LLM/regex shares). Navy = structured data features.</div>
      <div class="chart-host tall"><canvas id="fiChart"></canvas></div>
    </div>
    <div class="card" style="display:flex;flex-direction:column;justify-content:center;padding:24px;">
      <h3 style="margin-bottom:12px;">Model results</h3>
      <div id="modelKeyNums" style="font-size:13px;line-height:2;color:var(--ink);"></div>
    </div>
  </div>
  <div class="note dark">
    <strong>Clear message:</strong> at the drug-year level, with only 19 shortage events, the text
    features do <em>not</em> improve prediction. This is not a failure of the text signal — Section 5
    shows it is strong at the facility level — but of the aggregation: averaging one facility's
    repeat-violation 483 across 34 facilities making the same drug erases it.
    The next model iteration should predict <em>facility-level</em> outcomes (escalation, recall)
    from dated snapshots, then roll facility risk up to drugs.
  </div>
</section>

<!-- ═══ SECTION 7: WANG ET AL. 2025 CONTEXT (condensed) ═══ -->
<section>
  <div class="section-head">
    <span class="step-num">7</span>
    <h2>Context — OAI escalation and shortage risk (Wang et al., MSOM 2025)</h2>
  </div>
  <div class="note">
    Wang et al. (2025) find, with IV adjustment, that an OAI outcome <em>reduces</em> future shortage
    risk by ~96% — forced remediation fixes the underlying problem. Our 14-drug panel is too sparse for
    the same IV analysis, but the directional pattern is consistent: post-OAI shortage rates in our data
    trend toward the control baseline. The Section 5 finding (483 text content predicts escalation
    <em>before</em> OAI) is the upstream counterpart: if 483 text signals which facilities will face
    OAI, and OAI triggers remediation, the text is an early indicator of the quality-shock cycle.
    Full OAI forward-study chart available on request; removed here to keep the dashboard focused.
  </div>
</section>

<!-- ═══ SECTION 8: LIMITATIONS & NEXT STEPS ═══ -->
<section>
  <div class="section-head"><span class="step-num">8</span><h2>Limitations &amp; next steps</h2></div>
  <div class="two-col">
    <div class="col-card">
      <div class="col-head l">Limitations</div>
      <ul>
        <li>Only 14 drugs, 19 shortage-onset events, and 79 scored 483 snapshots → wide confidence intervals. All findings are exploratory.</li>
        <li>Text coverage: 37 of 129 facilities have public 483 PDFs — text signals exist only for that subset.</li>
        <li>Severity is now 4-tier (Critical/Major/Moderate/Minor), anchored on documented product impact. Recalibrated from the prior 3-tier scale; full extraction re-run on the 483 corpus reflects the new scheme.</li>
        <li>FAERS is quarterly in this dataset; Valisure scores are a single 2024 cross-section — neither supports monthly lead-lag.</li>
        <li>Shortage ↔ facility links are via API name only — a shortage may be caused by a different manufacturer of the same drug.</li>
        <li>All associations are descriptive — no causal identification.</li>
      </ul>
    </div>
    <div class="col-card">
      <div class="col-head n">Next steps</div>
      <ul>
        <li><strong>Facility-level model:</strong> predict escalation/recall from dated 483 snapshots, then roll facility risk up to drugs (the validated direction of the signal).</li>
        <li>Recalibrated 4-tier severity and new scope/root-cause fields are now in the extraction schema; expand 483 PDF coverage beyond the current 37 FEIs.</li>
        <li>Cross-inspection repeat detection (algorithmic, no API cost) is now in the pipeline — it's the second-strongest escalation predictor alongside LLM repeat flags.</li>
        <li>Expand pilot universe as Valisure tests additional APIs.</li>
        <li>Obtain inspector-level FDA data via FOIA to replicate Wang et al.'s IV analysis on the Valisure-drug subset.</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  Drug Shortage Prediction · Annual + Monthly Pipeline · June 2026 ·
  Canonical key: <code>drug_norm</code> · 14 Valisure APIs · exploratory only
</footer>
</div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const BY_YEAR   = {_j(d["by_year"])};
const BY_DRUG   = {_j(d["by_drug"])};

// ── Chart defaults ─────────────────────────────────────────────────────────
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
Chart.defaults.color = "#1A2233";
Chart.defaults.font.size = 11.5;
const C = {{navy:"#21295C",deep:"#065A82",teal:"#1C7293",accent:"#E07A5F",muted:"#5A6577"}};

// ── Annual: shortage starts by year ───────────────────────────────────────
new Chart(document.getElementById("chartYear"),{{
  type:"bar",
  data:{{labels:BY_YEAR.map(d=>d.year),
         datasets:[{{label:"Shortage starts",data:BY_YEAR.map(d=>d.starts),
                    backgroundColor:C.deep,borderRadius:4}}]}},
  options:{{responsive:true,maintainAspectRatio:false,
            plugins:{{legend:{{display:false}}}},
            scales:{{x:{{grid:{{display:false}}}},y:{{beginAtZero:true,ticks:{{stepSize:1}}}}}}}}
}});

// ── Annual: shortage duration by drug (total months in shortage 2015–2024) ─
const sorted = [...BY_DRUG].sort((a,b)=>a.duration-b.duration);
new Chart(document.getElementById("chartDrug"),{{
  type:"bar",
  data:{{labels:sorted.map(d=>d.drug),
         datasets:[{{label:"Months in shortage",
                    data:sorted.map(d=>d.duration),
                    backgroundColor:sorted.map(d=>d.duration>=60?C.accent:(d.duration===0?C.teal:C.deep)),
                    borderRadius:3}}]}},
  options:{{indexAxis:"y",responsive:true,maintainAspectRatio:false,
            plugins:{{legend:{{display:false}},
              tooltip:{{callbacks:{{label:ctx=>`${{sorted[ctx.dataIndex].drug}}: ${{ctx.raw}} months (${{sorted[ctx.dataIndex].starts}} shortage starts)`}}}}}},
            scales:{{x:{{beginAtZero:true,title:{{display:true,text:"Months in shortage (2015–2024)"}}}},
                     y:{{grid:{{display:false}}}}}}}}
}});

// ── Monthly lead-lag charts ────────────────────────────────────────────────
{redica_js}

// ── Text index monthly lead-lag (flat = persistent risk gap) ─────────────
{text_ll_js}

// ── 483 text analysis charts ──────────────────────────────────────────────
{text_js}

// ── 483 text group comparison (shortage vs no-shortage drugs) ─────────────
{text_group_js}

// ── Text detail group (violation categories + remediation) ────────────────
{text_detail_group_js}

// ── m11 facility-level forward validation ─────────────────────────────────
{fwd_js}

// ── RF model results ──────────────────────────────────────────────────────
{model_js}

// ── Drug-level text × shortage / FAERS scatter ────────────────────────────
{text_vs_shortage_js}

// ── Interactive FEI timeline (Section 5B) ─────────────────────────────────
{timeline_js}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data…")
    d = compute_data()
    log.info("Generating HTML…")
    html = generate_html(d)
    DASH_OUT.write_text(html, encoding="utf-8")
    size_kb = DASH_OUT.stat().st_size // 1024
    log.info("Wrote %s  (%d KB)", DASH_OUT, size_kb)
    print(f"Dashboard written → {DASH_OUT}  ({size_kb} KB)")
    return DASH_OUT


if __name__ == "__main__":
    main()
