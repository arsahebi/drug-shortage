# %%
"""
Build one document covering all 6 max-sample variants (rulebased x manual,
each pooled / IR / ER). Figures follow the style and structure of
Metformin Health Affairs Scholars 2026 05 29_ASF.docx: points colored by
country, sample sizes shown on the plot itself (n = NDC-year observations),
correlation panels annotated in-plot, quality-by-country as bar charts of
means. Each figure gets its regression coefficients, the ICC that justifies
clustering, and one short sentence of interpretation -- no comparison across
alternative test approaches, no model-name labels.

All numbers are parsed directly from the stats_log.txt each
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
    the next blank line."""
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


def _icc_before(anchor_match):
    """ICC value from the 'MixedLM (diagnostic only): ICC=...' line that
    precedes a Model B anchor, if one was fit (absent for the Difference
    Factor cross-section, which has no repeated-measures structure)."""
    head = anchor_match.string[:anchor_match.start()]
    m = re.search(r"MixedLM \(diagnostic only\): ICC=([\d.]+)\s*$", head.rstrip().rsplit("\n", 1)[-1])
    if not m:
        # search a couple lines back in case of intervening blank line
        lines = head.rstrip().split("\n")
        for ln in reversed(lines[-3:]):
            m = re.search(r"ICC=([\d.]+)", ln)
            if m:
                break
    return float(m.group(1)) if m else None


def parse_log(path):
    t = Path(path).read_text()
    r = {}

    # Fig1: price and volume by outcome
    for key, label in [("Price per Unit ($)", "price"), ("Market Volume (Extended Units)", "volume")]:
        m = re.search(rf"\[{re.escape(key)}\] n=(\d+), by outcome: (\{{[^}}]+\}})", t)
        anchor = re.search(rf"-- PRIMARY \[log\({re.escape(key)}\), ref=NAI\]:", t)
        coefs = _coefs_after(anchor) if anchor else {}
        icc = _icc_before(anchor) if anchor else None
        r[label] = {
            "n": int(m.group(1)) if m else None,
            "by": eval(m.group(2)) if m else {},
            "icc": icc,
            "coefs": coefs,
        }

    # Fig2 / Fig3: correlations (already annotated in-plot; parsed for the
    # short finding sentence only)
    for metric in ["DMF (ng/day)", "NDMA (ng/day)", "Dissolution Difference"]:
        for xcol, fig in [("Market Volume (Extended Units)", "fig2"), ("Price per Unit ($)", "fig3")]:
            m = re.search(rf"\[{re.escape(metric)} vs {re.escape(xcol)}[^\]]*\] n=(\d+).*?\n\s*n=\d+ \(NDCs=(\d+)\)\s+rho=([+-][\d.]+)\s+95% CI[^\n]*p_naive=([\d.]+)\s+p_boot=([\d.]+)\s*(\*{{0,2}})", t)
            key = f"{fig}_{metric}"
            r[key] = ({"n": int(m.group(1)), "ndcs": int(m.group(2)), "rho": float(m.group(3)),
                       "p_naive": float(m.group(4)), "p_boot": float(m.group(5)), "sig": m.group(6)}
                      if m else None)

    # Fig4: quality by country
    for metric in ["DMF (ng/day)", "NDMA (ng/day)", "Dissolution Difference"]:
        m = re.search(rf"\[{re.escape(metric)} by country\] n=(\d+), by country: (\{{[^}}]+\}})", t)
        anchor = re.search(rf"-- PRIMARY[^\[]*\[log1p\({re.escape(metric)}\), ref=USA\]:", t)
        coefs = _coefs_after(anchor) if anchor else {}
        icc = _icc_before(anchor) if anchor else None
        r[f"fig4_{metric}"] = {
            "n": int(m.group(1)) if m else None,
            "by": eval(m.group(2)) if m else {},
            "icc": icc,
            "coefs": coefs,
        }

    return r


# ── doc helpers ────────────────────────────────────────────────────────────────
def pfmt(v):
    if v is None:
        return "n/a"
    return v if isinstance(v, str) else (f"{v:.3f}" if v >= 0.001 else "<0.001")


def coef_phrase(coefs, name, ref):
    if name not in coefs:
        return f"{name} vs {ref} n/a"
    c = coefs[name]
    if c["unreliable"]:
        return f"{name} vs {ref} not estimable (too few observations)"
    return f"{name} vs {ref} beta={c['beta']:+.3f}, p={pfmt(c['p'])}{c['sig']}"


def any_significant(coefs):
    return any(c["sig"] and not c["unreliable"] for c in coefs.values())


def fig1_finding(d):
    sig_bits = []
    for label, key in [("price", "price"), ("volume", "volume")]:
        for name in ["VAI", "OAI"]:
            c = d[key]["coefs"].get(name)
            if c and c["sig"] and not c["unreliable"]:
                sig_bits.append(f"{name} {label} ({'higher' if c['beta']>0 else 'lower'} than NAI)")
    if not sig_bits:
        return "No significant relationship between prior inspection outcome and price or volume."
    return "Significant: " + "; ".join(sig_bits) + "."


def fig4_finding(d):
    sig_bits = []
    for metric, mlabel in [("DMF (ng/day)", "DMF"), ("NDMA (ng/day)", "NDMA"), ("Dissolution Difference", "Dissolution")]:
        coefs = d[f"fig4_{metric}"]["coefs"]
        for name in ["IND", "CHN"]:
            c = coefs.get(name)
            if c and c["sig"] and not c["unreliable"]:
                sig_bits.append(f"{mlabel} {'India' if name=='IND' else 'China'} ({'higher' if c['beta']>0 else 'lower'} than USA)")
    if not sig_bits:
        return "No significant difference in quality by country of manufacture."
    return "Significant: " + "; ".join(sig_bits) + "."


def fig23_finding(d, fig):
    sig_bits = []
    for metric, mlabel in [("DMF (ng/day)", "DMF"), ("NDMA (ng/day)", "NDMA"), ("Dissolution Difference", "Dissolution")]:
        v = d.get(f"{fig}_{metric}")
        if v and v["sig"]:
            sig_bits.append(f"{mlabel} (rho={v['rho']:+.2f}, p={pfmt(v['p_boot'])})")
    if not sig_bits:
        return "No significant correlation."
    return "Significant: " + "; ".join(sig_bits) + "."


def build():
    doc = ds.new_document()
    doc.add_heading("Metformin Analysis: All Variants", 0)
    ds.p(doc, "September 21, 2026", italic=True, size=10)
    ds.rule(doc)

    ds.p(doc,
         "Six versions of Figures 1 through 4: the rule-based and manual NDC-FEI maps, each "
         "pooled and split by dosage form (immediate vs. extended release). Every figure uses "
         "the largest sample its own axes allow; the only universal exclusion is Canada and "
         "Bangladesh. n is the count of NDC-year observations, shown on each plot. Figures 1 "
         "and 4 report an OLS regression with Cameron-Gelbach-Miller (2011) two-way clustered "
         "standard errors on NDC and facility; the ICC reported alongside it documents the "
         "within-NDC correlation that justifies clustering. Difference Factor is 2024 only, so "
         "it has no ICC and uses facility-only clustering. Figures 2 and 3 report a Spearman "
         "correlation with an NDC-cluster bootstrap, annotated directly on each panel.",
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
                  "Points are NDC-year observations colored by country of manufacture; "
                  "sample sizes shown beneath each box.", width=5.8)
        for label, key in [("Price", "price"), ("Volume", "volume")]:
            icc_str = f", ICC={d[key]['icc']:.2f}" if d[key]["icc"] is not None else ""
            ds.p(doc, f"{label}: n={d[key]['n']}{icc_str}. "
                 f"{coef_phrase(d[key]['coefs'], 'VAI', 'NAI')}. "
                 f"{coef_phrase(d[key]['coefs'], 'OAI', 'NAI')}.", size=9)
        ds.p(doc, fig1_finding(d), bold=True, size=9)

        # Figure 2 / Figure 3
        for fig_num, fname in [("2", "Figure2_Volume_vs_Quality"), ("3", "Figure3_Price_vs_Quality")]:
            fig_key = f"fig{fig_num}"
            ylab = "volume" if fig_num == "2" else "price"
            doc.add_heading(f"Figure {fig_num}, quality vs {ylab}", 2)
            ds.figure(doc, fig_dir / f"{fname}.png",
                      "Points are NDC-year observations colored by country of manufacture; "
                      "n, rho, 95% CI and p shown on each panel.", width=5.8)
            ds.p(doc, fig23_finding(d, fig_key), bold=True, size=9)

        # Figure 4
        doc.add_heading("Figure 4, quality by country of manufacture", 2)
        ds.figure(doc, fig_dir / "Figure4_Quality_by_Country.png",
                  "Bars show the mean of each quality metric by country; sample sizes shown "
                  "beneath each bar.", width=5.8)
        for metric, mlabel in [("DMF (ng/day)", "DMF"), ("NDMA (ng/day)", "NDMA"),
                                ("Dissolution Difference", "Dissolution Difference")]:
            v = d[f"fig4_{metric}"]
            icc_str = f", ICC={v['icc']:.2f}" if v["icc"] is not None else ""
            ds.p(doc, f"{mlabel}: n={v['n'] or 0}{icc_str}. "
                 f"{coef_phrase(v['coefs'], 'IND', 'USA')}. "
                 f"{coef_phrase(v['coefs'], 'CHN', 'USA')}.", size=9)
        ds.p(doc, fig4_finding(d), bold=True, size=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    build()
# %%
