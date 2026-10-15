"""
eval/code/08_score_round2.py

Scores Abdel's round-2 labels (the held-out validation set) against the
Claude Sonnet 5 v2 extraction, and measures his test-retest reliability on
the 20 hidden repeats of round 1.

Why a new script rather than re-running 04 or human_eval_02
  - 04_score_models_vs_human.py scores 5 fields and drops blank remediation as
    missing. human_eval_02_score.py has the full 11-field mapping, but scores
    against the round-1 sample-50 answer key. Neither is wired for round 2.
  - This script reuses human_eval_02's FIELDS (model column -> human column)
    and its _normalize, so the field mapping is identical, and pulls the model
    side from the current canonical step01 file, as 04 does. Both older
    scripts are left unchanged, so the round-1 table in the manuscript does
    not move.

Conventions
  - Blank remediation is a value. Abdel confirmed a blank (exported as "None")
    means he saw nothing about remediation; the extraction leaves the field
    blank on 912 of 1,067 observations. Both sides map blank to "None".
  - severity_tier is reported two ways: the raw 4 tiers, and with
    Major+Moderate collapsed (the 04 standard; see the 2026-09-16 validation
    report). The pooled figure uses the raw tiers, so it is the stricter one.
  - Round 2 oversampled rare model labels (06_build_round2_labeling_set.py),
    and 06 says overall accuracy must be weighted back to the corpus. The
    weighted estimate post-stratifies the 50 new rows into exclusive cells:
    each unlabeled-pool observation goes to the first rare stratum it matches,
    in 06's order, else to a residual cell, and each labeled row is weighted
    by cell size / labeled rows in the cell. Its interval is a Wilson interval
    on the Kish effective sample size.
  - Test-retest compares round 2 with round 1 on the 20 repeats. Round 1 was
    labeled before the rules were amended (data integrity now covers testing
    into compliance; Grade A/B EM excursions count as confirmed contamination;
    a complaint naming a batch shows distribution), so a disagreement on those
    fields can be a rule change rather than inconsistency.

Inputs
  labels_from_abdel/20261014_round2_labels_abdel.xlsx              (primary)
  labels_from_abdel/20261014_round2_labels_abdel_ai_assisted.xlsx  (sensitivity)
  labels_from_abdel/20260902_round1_labels_abdel.xlsx              (test-retest)
  validation_data/DO_NOT_SHARE_round2_key.csv
  step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv

Outputs (eval/results_and_notes/)
  round2_agreement.csv        per field x label version x subset, with Wilson CIs
  round2_test_retest.csv      per field, round 2 vs round 1 on the 20 repeats
  round2_ai_edit_direction.csv  each logged AI edit: did it move toward Claude?
  round2_scoring_summary.md   the tables above in readable form

Run:
  /Users/asahebi/Projects/drug-shortage-research/venv/bin/python 08_score_round2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE      = Path(__file__).parent
EVAL_ROOT = HERE.parent
DATA      = EVAL_ROOT.parent
sys.path.insert(0, str(HERE))
from human_eval_02_score import FIELDS, _normalize  # noqa: E402

LABELS    = EVAL_ROOT / "labels_from_abdel"
R2_OWN    = LABELS / "20261014_round2_labels_abdel.xlsx"
R2_AI     = LABELS / "20261014_round2_labels_abdel_ai_assisted.xlsx"
R1        = LABELS / "20260902_round1_labels_abdel.xlsx"
KEY_CSV   = EVAL_ROOT / "validation_data" / "DO_NOT_SHARE_round2_key.csv"
CLAUDE    = DATA / "step01_redica_483_obs_llm_signals_anthropic_claudesonnet5_v2.csv"
OUT       = EVAL_ROOT / "results_and_notes"

JOIN = ["fei", "insp_date", "obs_num"]
COLLAPSE = {"major": "major_or_moderate", "moderate": "major_or_moderate"}
# The split the models actually use (severity_critmajor_share in step02).
CRITMAJOR = {"critical": "critical_or_major", "major": "critical_or_major",
             "moderate": "moderate_or_minor", "minor": "moderate_or_minor"}

# 06's strata, same order, same masks (on the model's labels).
STRATA = [
    ("severity_Minor",       lambda d: d["severity_tier"] == "Minor"),
    ("remediation_nonblank", lambda d: d["remediation_signal"].notna()),
    ("violation_rare",       lambda d: d["violation_category"].isin(
                                 ["MaterialsSystem", "PackagingLabelingSystem", "Other"])),
    ("scope_Unclear",        lambda d: d["scope"] == "Unclear"),
    ("rootcause_Unclear",    lambda d: d["root_cause_type"] == "Unclear"),
]


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = float((a.values == b.values).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return np.nan if pe >= 1 else (po - pe) / (1 - pe)


def norm(v) -> str:
    """human_eval_02's _normalize, with blank meaning 'None' (see docstring)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return "none"
    return _normalize(v)


def read_labels(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Labeling", keep_default_na=False, na_values=[""])
    d["insp_date"] = pd.to_datetime(d["insp_date"]).dt.strftime("%Y-%m-%d")
    return d


def post_strat_weights(claude: pd.DataFrame, r1: pd.DataFrame, new_rows: pd.DataFrame) -> pd.Series:
    """Weight for each new labeled row = pool cell size / labeled rows in that cell."""
    r1_keys = set(zip(r1.fei, r1.insp_date, r1.obs_num))
    pool = claude[[k not in r1_keys for k in zip(claude.fei, claude.insp_date, claude.obs_num)]].copy()

    def cell(d: pd.DataFrame) -> pd.Series:
        out = pd.Series("residual", index=d.index)
        for name, f in reversed(STRATA):      # first match wins, so assign in reverse
            out[f(d).values] = name
        return out

    pool["cell"] = cell(pool)
    sizes = pool["cell"].value_counts()
    nr = new_rows.merge(claude[JOIN + [s for s in ["severity_tier", "remediation_signal",
                                                     "violation_category", "scope",
                                                     "root_cause_type"]]],
                        on=JOIN, how="left", suffixes=("", "_m"))
    nr["cell"] = cell(nr).values
    counts = nr["cell"].value_counts()
    w = nr["cell"].map(sizes) / nr["cell"].map(counts)
    return pd.Series(w.values, index=new_rows.index), nr["cell"].values, sizes, counts


def agreement_rows(lab: pd.DataFrame, version: str, subset: str,
                   w: pd.Series | None = None) -> tuple[list[dict], pd.DataFrame]:
    """Per-field rows, plus the row x field hit matrix (raw severity) for pooling."""
    rows, hits = [], {}
    specs = [(f, mc, hc, False) for f, mc, hc, _ in FIELDS]
    specs.insert(2, ("severity_tier (Major+Moderate collapsed)", "severity_tier", "human_severity_tier", COLLAPSE))
    specs.insert(3, ("severity_tier (Critical+Major vs rest)", "severity_tier", "human_severity_tier", CRITMAJOR))
    for field, mc, hc, collapse in specs:
        h = lab[hc].map(norm)
        m = lab[mc].map(norm)
        if collapse:
            h, m = h.replace(collapse), m.replace(collapse)
        hit = (h == m).astype(float)
        if not collapse:
            hits[field] = hit
        n, k = len(hit), hit.sum()
        lo, hi = wilson(k, n)
        r = dict(version=version, subset=subset, field=field, n=n, agree=int(k),
                 pct=k / n, ci_lo=lo, ci_hi=hi, kappa=cohen_kappa(h, m))
        if w is not None:
            ww = w.loc[lab.index]
            pw = float((hit * ww).sum() / ww.sum())
            n_eff = ww.sum() ** 2 / (ww ** 2).sum()
            lo_w, hi_w = wilson(pw * n_eff, n_eff)
            r.update(pct_weighted=pw, n_eff=n_eff, ci_lo_weighted=lo_w, ci_hi_weighted=hi_w)
        rows.append(r)
    return rows, pd.DataFrame(hits)


def pooled(hits: pd.DataFrame, version: str, subset: str, w: pd.Series | None = None,
           n_boot: int = 4000, seed: int = 20261015) -> dict:
    """Pooled agreement over the 11 fields. The interval resamples observations
    (rows), since the 11 fields of one observation are not independent."""
    rng = np.random.default_rng(seed)
    per_row = hits.mean(axis=1).values
    n_rows = len(per_row)
    idx = rng.integers(0, n_rows, size=(n_boot, n_rows))
    boot = per_row[idx].mean(axis=1)
    out = dict(version=version, subset=subset, field=f"POOLED ({hits.shape[1]} fields)",
               n=int(hits.size), agree=int(hits.values.sum()), pct=float(per_row.mean()),
               ci_lo=float(np.quantile(boot, 0.025)), ci_hi=float(np.quantile(boot, 0.975)),
               kappa=np.nan, note=f"n = {n_rows} observations x {hits.shape[1]} fields; "
                                  "95% CI from an observation-level bootstrap")
    if w is not None:
        ww = w.loc[hits.index].values
        bw = (per_row[idx] * ww[idx]).sum(axis=1) / ww[idx].sum(axis=1)
        out.update(pct_weighted=float((per_row * ww).sum() / ww.sum()),
                   n_eff=float(ww.sum() ** 2 / (ww ** 2).sum()),
                   ci_lo_weighted=float(np.quantile(bw, 0.025)),
                   ci_hi_weighted=float(np.quantile(bw, 0.975)))
    return out


def main() -> None:
    key = pd.read_csv(KEY_CSV)
    key["insp_date"] = pd.to_datetime(key["insp_date"]).dt.strftime("%Y-%m-%d")
    claude = pd.read_csv(CLAUDE)
    claude["insp_date"] = pd.to_datetime(claude["insp_date"]).dt.strftime("%Y-%m-%d")
    r1 = read_labels(R1)

    model_cols = sorted({mc for _, mc, _, _ in FIELDS})
    labsets = {}
    for version, path in [("own", R2_OWN), ("ai_assisted", R2_AI)]:
        lab = read_labels(path)
        assert (lab["row_id"].values == key["row_id"].values).all(), "row order differs from key"
        assert (lab[JOIN].values == key[JOIN].values).all(), "fei/date/obs differ from key"
        lab = lab.merge(key[["row_id", "source", "round1_row_id", "stratum"]], on="row_id")
        lab = lab.merge(claude[JOIN + model_cols], on=JOIN, how="left", validate="1:1")
        assert lab["violation_category"].notna().all(), "unmatched rows against step01"
        labsets[version] = lab

    own = labsets["own"]
    new_idx = own.index[own["source"] == "new"]
    w, cells, sizes, counts = post_strat_weights(claude, r1, own.loc[new_idx])
    print("Post-stratification cells (pool size / labeled new rows):")
    for c in sizes.index:
        print(f"  {c:22s} {sizes[c]:5d} / {counts.get(c, 0)}")

    rows = []
    for version, lab in labsets.items():
        for subset, idx, ww in [("all70", lab.index, None),
                                ("new50", new_idx, w),
                                ("random_natural20", lab.index[lab["stratum"] == "random_natural"], None),
                                ("repeat20", lab.index[lab["source"] == "repeat"], None)]:
            rr, hits = agreement_rows(lab.loc[idx], version, subset, ww)
            rows += rr + [pooled(hits, version, subset, ww)]
    agr = pd.DataFrame(rows)
    agr.to_csv(OUT / "round2_agreement.csv", index=False)

    # ── test-retest: round 2 (own) vs round 1, on the 20 repeats ──────────────
    rep = own[own["source"] == "repeat"].merge(
        r1, left_on="round1_row_id", right_on="row_id", suffixes=("", "_r1"))
    assert len(rep) == 20
    assert (rep["fei"] == rep["fei_r1"]).all() and (rep["obs_num"] == rep["obs_num_r1"]).all()
    tr, tr_hits = [], {}
    for field, _, hc, _ in FIELDS:
        for collapse in ([False, COLLAPSE, CRITMAJOR] if field == "severity_tier" else [False]):
            a, b = rep[hc].map(norm), rep[hc + "_r1"].map(norm)
            if collapse:
                a, b = a.replace(collapse), b.replace(collapse)
            k, n = int((a == b).sum()), len(a)
            if not collapse:
                tr_hits[field] = (a == b).astype(float)
            lo, hi = wilson(k, n)
            tag = ("" if not collapse else " (Major+Moderate collapsed)" if collapse is COLLAPSE
                   else " (Critical+Major vs rest)")
            tr.append(dict(field=field + tag,
                           n=n, agree=k, pct=k / n, ci_lo=lo, ci_hi=hi, kappa=cohen_kappa(a, b),
                           rule_amended_since_round1=field in {"data_integrity_flag", "patient_risk_flag",
                                                               "contamination_flag",
                                                               "contamination_risk_flag"}))
    trd = pd.DataFrame(tr)
    pr = pooled(pd.DataFrame(tr_hits), "own", "repeat20_vs_round1")
    trd = pd.concat([trd, pd.DataFrame([{k: pr[k] for k in
                                         ["field", "n", "agree", "pct", "ci_lo", "ci_hi", "note"]}])])
    trd.to_csv(OUT / "round2_test_retest.csv", index=False)

    # ── direction of the logged AI edits ────────────────────────────────────
    log = pd.read_excel(R2_AI, sheet_name="AI_Revision_Log", header=2)
    log.columns = [str(c).strip() for c in log.columns]
    log = log.dropna(subset=["row_id"])
    hum2model = {hc: mc for _, mc, hc, _ in FIELDS}
    ed = []
    for _, r in log.iterrows():
        fld = str(r["field"]).strip()
        hc = fld if fld.startswith("human_") else "human_" + fld
        row = own[own["row_id"] == int(r["row_id"])].iloc[0]
        mc = hum2model.get(hc)
        orig, new = norm(r["original_value"]), norm(r["revised_value"])
        mv = norm(row[mc]) if mc else None
        if mc is None:
            direction = "free-text field"
        elif new == mv and orig != mv:
            direction = "to Claude"
        elif orig == mv and new != mv:
            direction = "away from Claude"
        else:
            direction = "neither"
        ed.append(dict(row_id=int(r["row_id"]), field=hc, original=orig, revised=new,
                       claude=mv, direction=direction))
    edd = pd.DataFrame(ed)
    edd.to_csv(OUT / "round2_ai_edit_direction.csv", index=False)

    # ── readable summary ────────────────────────────────────────────────────
    def fmt(df, cols):
        return "```\n" + df[cols].to_string(index=False, float_format="%.3f") + "\n```"

    lines = ["# Round-2 human validation: scored results",
             "",
             "Source: `eval/code/08_score_round2.py`. Claude Sonnet 5 v2 extraction against "
             "Abdel's blinded round-2 labels. Percent agreement with Wilson 95% intervals; n beside "
             "every figure. See the script docstring for conventions.",
             ""]
    for version in ["own", "ai_assisted"]:
        tag = "PRIMARY: his own labels" if version == "own" else "SENSITIVITY ONLY: after his Claude self-check"
        for subset in ["new50", "all70", "random_natural20"]:
            d = agr[(agr.version == version) & (agr.subset == subset)]
            cols = ["field", "n", "agree", "pct", "ci_lo", "ci_hi", "kappa"]
            if subset == "new50":
                cols += ["pct_weighted", "n_eff", "ci_lo_weighted", "ci_hi_weighted"]
            lines += [f"## {tag} | {subset}", "", fmt(d, cols), ""]
            if version == "ai_assisted":
                break
    lines += ["## Test-retest: round 2 vs round 1, 20 repeats (Abdel against himself)", "",
              fmt(trd, ["field", "n", "agree", "pct", "ci_lo", "ci_hi", "kappa", "rule_amended_since_round1"]), "",
              "## Direction of the logged AI edits", "",
              "```\n" + edd.direction.value_counts().to_string() + "\n```", "",
              "Categorical and flag fields only:", "",
              "```\n" + edd[edd.direction != "free-text field"].direction.value_counts().to_string() + "\n```",
              ""]
    (OUT / "round2_scoring_summary.md").write_text("\n".join(lines))

    show = agr[agr.field.str.startswith("POOLED")][["version", "subset", "n", "pct", "ci_lo", "ci_hi",
                                                     "pct_weighted", "ci_lo_weighted", "ci_hi_weighted"]]
    print(show.to_string(index=False, float_format="%.3f"))
    print(agr[(agr.version == "own") & (agr.subset == "new50")][
        ["field", "n", "pct", "ci_lo", "ci_hi", "kappa", "pct_weighted"]].to_string(index=False, float_format="%.3f"))
    print(trd.to_string(index=False, float_format="%.3f"))
    print(edd.direction.value_counts().to_string())


if __name__ == "__main__":
    main()
