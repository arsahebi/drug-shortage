# %%
"""
Build one document covering all 6 max-sample variants (rulebased x manual,
each pooled / IR / ER), with every figure, its primary statistics, and a
one-sentence interpretation. Same style as 20260912_metformin_figures_rulebased.docx.

Figures 1 and 4 report Model B (RE + two-way CGM clustered SE, NDC x FEI) --
the established primary specification per Metformin JAMA 2026 02 27_StatTests.docx.
Figures 2 and 3 report Spearman rho with an NDC-cluster bootstrap, unchanged from
that same memo. All numbers are parsed directly from the stats_log.txt each
build_variant_graphs.py run wrote, not retyped, so this document cannot drift
from what was actually computed.

Output: outputs/20260920_metformin_all_variants.docx
"""

import re
from pathlib import Path

import doc_style as ds

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
VOUT = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs/variants"
OUT  = BASE / "Data/99 - Outputs - Metformin Analysis/processed/outputs/20260920_metformin_all_variants.docx"

VARIANTS = [("rulebased", "all"), ("rulebased", "IR"), ("rulebased", "ER"),
            ("manual", "all"), ("manual", "IR"), ("manual", "ER")]
MAP_LABEL = {"rulebased": "Rule-based NDC-FEI map", "manual": "Manual NDC-FEI map"}
DOSE_LABEL = {"all": "all dosage forms pooled", "IR": "immediate-release only", "ER": "extended-release only"}


# ── parsing ───────────────────────────────────────────────────────────────────
_COEF_LINE = re.compile(
    r"\s*(\w+): beta=([+-]\d+\.\d+), SE=(\d+\.\d+).*?p([<=])(\d+\.\d+)(\*{1,2}|\.)?([^\n]*)")


def _coefs_after(anchor_match):
    """Coefficient lines directly following a '-- PRIMARY [...]:' anchor, up to
    the next blank line. Anchoring on the literal '-- PRIMARY' text (rather than
    the header text after it) matters because the header itself contains a
    bracketed '[...]' that would otherwise terminate a '[^\\[]*' capture early."""
    tail = anchor_match.string[anchor_match.end():]
    block = tail.split("\n\n", 1)[0]
    out = {}
    for line in block.split("\n"):
        mm = _COEF_LINE.match(line)
        if not mm:
            continue
        name, beta, se, rel, pval, sig, extra = mm.groups()
        out[name] = {
            "beta": float(beta), "se": float(se),
            "p": ("<" + pval) if rel == "<" else float(pval),
            "sig": sig or "", "unreliable": "UNRELIABLE" in extra,
        }
    return out


def parse_log(path):
    t = Path(path).read_text()
    r = {}

    # Fig1: price and volume by outcome
    for key, label in [("Medicaid price", "price"), ("IQVIA extended units", "volume")]:
        m = re.search(rf"\[{re.escape(key)}[^\]]*\] n=(\d+), by outcome: (\{{[^}}]+\}})", t)
        kw = re.search(rf"\[{re.escape(key)}.*?Kruskal-Wallis[^:]*: p=([\d.]+)", t, re.S)
        anchor = re.search(rf"-- PRIMARY \[log\({re.escape(key)}[^\]]*\), ref=NAI\]:", t)
        coefs = _coefs_after(anchor) if anchor else {}
        r[label] = {
            "n": int(m.group(1)) if m else None,
            "by": eval(m.group(2)) if m else {},
            "kw": float(kw.group(1)) if kw else None,
            "modelB": coefs,
        }

    # Fig2 / Fig3: correlations
    for metric in ["DMF (ng/day)", "NDMA (ng/day)", "Difference Factor"]:
        for xcol, fig in [("IQVIA extended units", "fig2"), ("Medicaid price", "fig3")]:
            m = re.search(rf"\[{re.escape(metric)} vs {re.escape(xcol)}[^\]]*\] n=(\d+).*?\n\s*n=\d+ \(NDCs=(\d+)\)\s+rho=([+-][\d.]+)\s+p_naive=([\d.]+)\s+p_boot=([\d.]+)\s*(\*{{0,2}})", t)
            key = f"{fig}_{metric}"
            if m:
                r[key] = {"n": int(m.group(1)), "ndcs": int(m.group(2)), "rho": float(m.group(3)),
                           "p_naive": float(m.group(4)), "p_boot": float(m.group(5)), "sig": m.group(6)}
            else:
                r[key] = None

    # Fig4: quality by country
    for metric in ["DMF (ng/day)", "NDMA (ng/day)", "Difference Factor"]:
        m = re.search(rf"\[{re.escape(metric)} by country\] n=(\d+), by country: (\{{[^}}]+\}})", t)
        kw = re.search(rf"\[{re.escape(metric)} by country\].*?Kruskal-Wallis[^:]*: p=([\d.]+)", t, re.S)
        anchor = re.search(rf"-- PRIMARY[^\[]*\[log1p\({re.escape(metric)}\), ref=USA\]:", t)
        coefs = _coefs_after(anchor) if anchor else {}
        r[f"fig4_{metric}"] = {
            "n": int(m.group(1)) if m else None,
            "by": eval(m.group(2)) if m else {},
            "kw": float(kw.group(1)) if kw else None,
            "modelB": coefs,
        }

    # Figure S1
    m = re.search(r"\[Figure S1\] n=(\d+), mean=([\d.]+), median=([\d.]+), >36mo: (\d+)", t)
    r["figS1"] = ({"n": int(m.group(1)), "mean": float(m.group(2)), "median": float(m.group(3)),
                   "over36": int(m.group(4))} if m else None)
    return r


# ── doc helpers ────────────────────────────────────────────────────────────────
def pfmt(v):
    if v is None:
        return "n/a"
    return v if isinstance(v, str) else (f"{v:.4f}" if v >= 0.001 else "<0.001")


def coef_line(coefs, name, ref):
    if name not in coefs:
        return f"{name} vs {ref}: n/a"
    c = coefs[name]
    tag = " [UNRELIABLE]" if c["unreliable"] else (" *" if c["sig"] else "")
    return f"{name} vs {ref}: beta={c['beta']:+.3f} (SE {c['se']:.3f}), p={pfmt(c['p'])}{tag}"


def build():
    doc = ds.new_document()
    doc.add_heading("Metformin Analysis: All Variants", 0)
    ds.p(doc, "September 20, 2026", italic=True, size=10)
    ds.rule(doc)

    ds.p(doc,
         "Six versions of Figures 1 through 4 and S1: the rule-based and manual NDC-FEI maps, "
         "each pooled and split by dosage form (immediate vs. extended release). Every figure "
         "uses the largest sample its own axes allow; the only universal exclusion is Canada "
         "and Bangladesh. Figures 1 and 4 report Model B: an OLS point estimate with "
         "Cameron-Gelbach-Miller (2011) two-way clustered standard errors on NDC and facility "
         "simultaneously, matching Metformin_2026 03 10_Appendix.docx. A random-NDC-intercept "
         "mixed model is fit alongside it only to report the ICC that documents within-NDC "
         "correlation and justifies clustering; its coefficient is not the one reported. "
         "Difference Factor (2024 only) has no repeated-measures structure for a NDC cluster "
         "to describe, so it uses FEI-only clustered SE with no mixed-model step at all. "
         "Figures 2 and 3 report Spearman correlation with an NDC-cluster bootstrap, unchanged "
         "from that same specification. A coefficient resting on fewer than 3 observations or "
         "2 facilities is marked UNRELIABLE rather than reported as a finding.",
         italic=True, size=9)

    for map_label, dose in VARIANTS:
        log_path = VOUT / f"{map_label}_{dose}" / "stats_log.txt"
        fig_dir = VOUT / f"{map_label}_{dose}"
        d = parse_log(log_path)

        doc.add_page_break()
        doc.add_heading(f"{MAP_LABEL[map_label]}, {DOSE_LABEL[dose]}", 1)

        # Figure 1
        doc.add_heading("Figure 1, price and volume by prior inspection outcome", 2)
        ds.figure(doc, fig_dir / "Figure1_Price_Volume_by_Outcome.png",
                  f"n(price)={d['price']['n']}, n(volume)={d['volume']['n']}", width=5.6)
        rows = []
        for label, key in [("Price", "price"), ("Volume", "volume")]:
            rows.append([label, str(d[key]["n"]), pfmt(d[key]["kw"]),
                         coef_line(d[key]["modelB"], "VAI", "NAI"),
                         coef_line(d[key]["modelB"], "OAI", "NAI")])
        ds.table(doc, ["Metric", "n", "KW p", "VAI vs NAI (Model B)", "OAI vs NAI (Model B)"], rows,
                 size=8.5, widths=[0.7, 0.4, 0.6, 2.0, 2.0])
        sig_any = any(d[k]["modelB"].get(g, {}).get("sig") for k in ["price", "volume"] for g in ["VAI", "OAI"])
        ds.p(doc, "No significant relationship between inspection outcome and price or volume "
             "under Model B." if not sig_any else
             "At least one outcome contrast reaches significance under Model B; see table.", size=9)

        # Figure 2 / Figure 3
        for fig_num, fig_label, fname in [("2", "volume", "Figure2_Volume_vs_Quality"),
                                            ("3", "price", "Figure3_Price_vs_Quality")]:
            doc.add_heading(f"Figure {fig_num}, quality vs {fig_label}", 2)
            ds.figure(doc, fig_dir / f"{fname}.png", f"Figure {fig_num}", width=5.6)
            rows = []
            any_sig = False
            for metric in ["DMF (ng/day)", "NDMA (ng/day)", "Difference Factor"]:
                v = d.get(f"fig{fig_num}_{metric}")
                if v is None:
                    rows.append([metric, "n/a", "n/a", "n/a"])
                    continue
                any_sig = any_sig or bool(v["sig"])
                rows.append([metric, str(v["n"]), f"{v['rho']:+.3f}", f"{pfmt(v['p_boot'])}{v['sig']}"])
            ds.table(doc, ["Metric", "n", "rho", "p_boot (NDC-clustered)"], rows, size=8.5,
                      widths=[1.5, 0.5, 0.7, 1.5])
            ds.p(doc, f"{'At least one metric is significantly correlated with ' + fig_label + '.' if any_sig else 'No metric is significantly correlated with ' + fig_label + '.'}", size=9)

        # Figure 4
        doc.add_heading("Figure 4, quality by country of manufacture", 2)
        ds.figure(doc, fig_dir / "Figure4_Quality_by_Country.png", "Figure 4", width=5.6)
        rows = []
        any_sig = False
        for metric in ["DMF (ng/day)", "NDMA (ng/day)", "Difference Factor"]:
            v = d[f"fig4_{metric}"]
            ind = v["modelB"].get("IND", {})
            chn = v["modelB"].get("CHN", {})
            any_sig = any_sig or bool(ind.get("sig")) or bool(chn.get("sig"))
            rows.append([metric, str(v["n"]) if v["n"] else "0", pfmt(v["kw"]),
                         coef_line(v["modelB"], "IND", "USA"),
                         coef_line(v["modelB"], "CHN", "USA")])
        ds.table(doc, ["Metric", "n", "KW p", "India vs USA (Model B)", "China vs USA (Model B)"],
                  rows, size=8.5, widths=[0.7, 0.4, 0.6, 2.1, 2.1])
        ds.p(doc, "At least one country contrast reaches significance under Model B; see table "
             "and note any UNRELIABLE flag before citing it." if any_sig else
             "No country contrast reaches significance under Model B in this variant.", size=9)

        # Figure S1
        doc.add_heading("Figure S1, months since last inspection", 2)
        ds.figure(doc, fig_dir / "FigureS1_Months_Since_Inspection.png", "Figure S1", width=4.8)
        s1 = d["figS1"]
        if s1:
            ds.p(doc, f"n={s1['n']}, mean={s1['mean']:.1f} months, median={s1['median']:.1f}, "
                 f"{s1['over36']} rows exceed 36 months.", size=9)
        else:
            ds.p(doc, "Insufficient data.", size=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    build()
# %%
