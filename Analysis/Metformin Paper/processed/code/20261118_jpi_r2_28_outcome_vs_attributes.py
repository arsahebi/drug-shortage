"""
J Pharm Innov R&R, Reviewer 2 comment 2.28: relationship between the
facility's prior FDA inspection outcome (NAI / VAI / OAI) and the tested
product attributes (DMF, NDMA, dissolution difference).

Sample: prepare("manual", "all") from build_variant_graphs.py, then rows with
a matched FEI in the U.S., India, or China (110 NDC-years / 84 NDCs), the same
rows as the paper's figures. Two inspection definitions, as in Figure 1:
  all    most recent prior inspection, regardless of time
  recent only if that inspection is within 36 months of the test year

Statistics: Kruskal-Wallis across outcomes, plus the paper's pairwise tests
(NDC-clustered permutation, FEI-clustered permutation, Mann-Whitney) written
to the stats log. Descriptive: products sharing a facility are not
independent.

Outputs (processed/outputs/jpi_revision/):
  R2_28_outcome_vs_attributes.xlsx   summary table, both definitions
  R2_28_outcome_vs_attributes.png    three panels, all prior inspections
  R2_28_stats_log.txt                pairwise tests
  20261118_R2_28_outcome_vs_attributes.docx   note for the team
  Supplementary_Table_S1_inspection_vs_attributes.docx   table for the paper
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal

import build_variant_graphs as b
import doc_style as ds
import fig1_style as fs

OUT = b.PROC / "outputs" / "jpi_revision"
NDMA_LIMIT = 96  # ng/day, FDA acceptable intake
ATTRS = [(b.DMF_COL, "DMF (ng/day)", "log"),
         (b.NDMA_COL, "NDMA (ng/day)", "symlog"),
         (b.DIFF_COL, "Dissolution difference", "linear")]


def sample():
    d = b.prepare("manual", "all")
    d = d[d.matched_fei.notna() & d.CountryCode.isin(["USA", "IND", "CHN"])].copy()
    recent, _ = b.mask_recent(d)
    return d, recent


def summary(d, label):
    rows = []
    for col, name, _ in ATTRS:
        s = d[d.prior_outcome.notna() & d[col].notna()]
        groups = {o: s.loc[s.prior_outcome == o, col].astype(float) for o in fs.OUTCOME_ORDER}
        valid = [g for g in groups.values() if len(g) >= 2]
        kw = kruskal(*valid)[1] if len(valid) >= 2 else np.nan
        for o, g in groups.items():
            rows.append({
                "inspection": label, "attribute": name, "outcome": o, "n": len(g),
                "median": g.median() if len(g) else np.nan,
                "q1": g.quantile(.25) if len(g) else np.nan,
                "q3": g.quantile(.75) if len(g) else np.nan,
                "mean": g.mean() if len(g) else np.nan,
                "above_limit": int((g > NDMA_LIMIT).sum()) if col == b.NDMA_COL else np.nan,
                "kruskal_p": kw})
    return pd.DataFrame(rows)


def figure(d, path):
    fs.N_IN_TICKS = True
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
        has_unknown = False
        for ax, (col, name, scale) in zip(axes, ATTRS):
            s = d[d.prior_outcome.notna() & d[col].notna()]
            groups, unk = fs.draw_on(ax, s, col, "prior_outcome", fs.OUTCOME_ORDER,
                                     fs.OUTCOME_ORDER, "Prior Inspection Outcome",
                                     "CountryCode", fs.COUNTRY_ORDER, fs.COUNTRY_COLORS, name)
            has_unknown |= unk
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=1)
                ax.axhline(NDMA_LIMIT, color="#6b7280", linestyle=":", linewidth=1)
            elif scale == "linear":
                ax.set_yscale("linear")
            ax.set_title(f"{name}{fs._kw_title_p(groups)}", fontsize=9, fontweight="bold")
        handles = fs.legend_handles(fs.COUNTRY_ORDER, fs.COUNTRY_COLORS, fs.COUNTRY_LABELS,
                                    "Unknown", has_unknown)
        fig.legend(handles=handles, title="Country", loc="lower center", ncol=len(handles),
                   fontsize=8, title_fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        fs.save(fig, [path])
        plt.close(fig)


def stats_log(d, recent, path):
    lines = []
    for label, df in [("All prior inspections", d), ("Inspections within 36 months", recent)]:
        lines.append(f"=== {label} ===")
        for col, name, _ in ATTRS:
            s = df[df.prior_outcome.notna() & df[col].notna()]
            lines.append(f"\n[{name}] n={len(s)}")
            b.pairwise_group_tests(lines.append, s, col, "prior_outcome", fs.OUTCOME_ORDER,
                                   fei_col="prior_fei")
        lines.append("")
    path.write_text("\n".join(lines))


def fmt(v, dec):
    return "" if pd.isna(v) else f"{v:,.{dec}f}"


def team_doc(tab, d, png, path):
    doc = ds.new_document()
    doc.add_heading("Comment 2.28: Prior Inspection Outcome and Tested Attributes", level=1)
    ds.p(doc, "Reviewer 2 asked about the relationship between the facility's inspection "
              "classification and the product attribute tests. This note shows that "
              "relationship in our data so the team can decide whether to report it in a "
              "supplementary table, in one sentence, or only as a limitation.")
    n_out = int(d.prior_outcome.notna().sum())
    ds.p(doc, f"Sample: the {len(d)} NDC-years used in the paper (manual linkage, U.S., India "
              f"and China); {n_out} have a prior inspection on record. Inspection outcome is "
              "the facility's most recent FDA inspection before the test year, as in Figure 1.")
    for label in tab.inspection.unique():
        doc.add_heading(label, level=2)
        rows = []
        for _, r in tab[tab.inspection == label].iterrows():
            dec = 3 if r.attribute.startswith("Dissolution") else 0
            rows.append([r.attribute, r.outcome, str(int(r.n)),
                         fmt(r["median"], dec),
                         f"{fmt(r.q1, dec)} to {fmt(r.q3, dec)}" if r.n else "",
                         "" if pd.isna(r.above_limit) else f"{int(r.above_limit)} of {int(r.n)}",
                         fmt(r.kruskal_p, 2)])
        ds.table(doc, ["Attribute", "Outcome", "n", "Median", "IQR", "NDMA > 96 ng/day",
                       "Kruskal-Wallis p"], rows, align_right_from=2)
    ds.figure(doc, png, "Tested attributes by prior inspection outcome (all prior "
                        "inspections). Points are NDC-years colored by country. DMF on a log "
                        "scale; NDMA on a symmetric log scale with the 96 ng/day limit dotted; "
                        "dissolution difference on a linear scale.")
    doc.add_heading("Reading", level=2)
    ds.bullet(doc, "No tested attribute differs significantly by prior inspection outcome, "
                   "with either inspection definition.")
    ds.bullet(doc, "No OAI-linked product had dissolution testing, so that comparison is "
                   "NAI vs VAI only.")
    ds.bullet(doc, "Group sizes are small (OAI has 11 NDC-years) and products sharing a "
                   "facility are not independent, so this is descriptive. The pairwise "
                   "clustered tests are in R2_28_stats_log.txt.")
    doc.add_heading("Options for the manuscript", level=2)
    ds.bullet(doc, "A. Supplementary table plus one Results sentence: \"Tested attribute "
                   "levels did not differ significantly by the facility's most recent "
                   "inspection outcome (Supplementary Table X).\"")
    ds.bullet(doc, "B. One sentence only, in Results or Limitations, with the Kruskal-Wallis "
                   "p values and no table.")
    ds.bullet(doc, "C. Limitation and future research only.")
    doc.save(path)


def supplementary_table(tab, d, path):
    """Supplementary Table S1 for the manuscript (option A chosen by the team)."""
    doc = ds.new_document()
    ds.p(doc, "Supplementary Table S1. Tested product attributes by the facility's most "
              "recent FDA inspection outcome", bold=True)
    rows = []
    for label in tab.inspection.unique():
        rows.append([label, "", "", "", "", "", ""])
        for _, r in tab[tab.inspection == label].iterrows():
            dec = 3 if r.attribute.startswith("Dissolution") else 0
            rows.append([r.attribute, r.outcome, str(int(r.n)), fmt(r["median"], dec),
                         f"{fmt(r.q1, dec)} to {fmt(r.q3, dec)}" if r.n else "",
                         "" if pd.isna(r.above_limit) else f"{int(r.above_limit)} of {int(r.n)}",
                         fmt(r.kruskal_p, 2) if r.outcome == "NAI" else ""])
    ds.table(doc, ["Attribute", "Prior outcome", "n", "Median", "Interquartile range",
                   "NDMA > 96 ng/day", "Kruskal-Wallis p"], rows, size=9, align_right_from=2)
    n_out = int(d.prior_outcome.notna().sum())
    ds.p(doc, f"Notes: NDC-years in the analysis sample with a prior FDA inspection on record "
              f"({n_out} of {len(d)}). Prior outcome is the classification of the facility's "
              "most recent FDA inspection before the test year (NAI, no action indicated; VAI, "
              "voluntary action indicated; OAI, official action indicated); the second section "
              "keeps only inspections within 36 months of the test year, as in Figure 1b. DMF "
              "and NDMA are daily exposures in ng/day; 96 ng/day is the FDA acceptable intake "
              "limit for NDMA. NDMA was tested in 2020 and 2022 and dissolution in 2024. No "
              "product linked to a facility with an OAI outcome had dissolution testing. "
              "Kruskal-Wallis tests compare the outcome groups for each attribute; products "
              "sharing a facility are not independent, so the tests are descriptive.", size=9)
    doc.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, recent = sample()
    tab = pd.concat([summary(d, "All prior inspections"),
                     summary(recent, "Inspections within 36 months")], ignore_index=True)
    tab.to_excel(OUT / "R2_28_outcome_vs_attributes.xlsx", index=False)
    png = OUT / "R2_28_outcome_vs_attributes.png"
    figure(d, png)
    stats_log(d, recent, OUT / "R2_28_stats_log.txt")
    team_doc(tab, d, png, OUT / "20261118_R2_28_outcome_vs_attributes.docx")
    supplementary_table(tab, d, OUT / "Supplementary_Table_S1_inspection_vs_attributes.docx")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
# %%
