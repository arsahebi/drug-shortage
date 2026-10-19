"""
20261019_faers_product_quality_cases.py
─────────────────────────────────────────────────────────────────────────────
Which FAERS reports describe a PRODUCT QUALITY problem, not a side effect.

A 483 is about how a drug is made, so the closest FAERS outcome is a report
whose reaction terms (MedDRA preferred terms, REAC files) describe the
product itself: contamination, odour, colour, physical defect, potency,
packaging, counterfeit, recall, substitution problems and so on.

Rule, kept simple so it can be checked by eye:
  1. keep a term if it contains the word "product", or is one of three
     drug-quality terms that do not ("poor quality drug administered",
     "counterfeit drug administered", "manufacturing materials contamination")
  2. drop it if it is a USE or MEDICATION ERROR term rather than a defect
     (EXCLUDE pattern below: wrong product, dose omission, confusion,
     storage error, off-label use, therapeutic product effect, ...)
The printed term table shows every term that passed; read it before use.

Input : raw/faers_ascii_*/ascii/REAC*.txt   (primaryid$caseid$pt$drug_rec_act)
        processed/faers_all_drugs_anda_linked_2015Q1_2026Q1.csv (primaryids to keep)
Output: processed/faers_product_quality_reactions.csv  (primaryid, pt)
        processed/faers_product_quality_terms.csv      (pt, reports, kept, category)
"""

from pathlib import Path
import glob
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                      # Data/15 - FDA - Adverse Event
RAW = BASE / "raw"
FAERS = BASE / "processed" / "faers_all_drugs_anda_linked_2015Q1_2026Q1.csv"
OUT = BASE / "processed" / "faers_product_quality_reactions.csv"
# every candidate term (kept or removed) with report counts and a category,
# for checking the rule by eye (2026-10-19)
OUT_TERMS = BASE / "processed" / "faers_product_quality_terms.csv"

ALSO_KEEP = {"poor quality drug administered", "counterfeit drug administered",
             "manufacturing materials contamination"}
EXCLUDE = (r"error|use issue|use in |misuse|exposure|administered at|administered to|"
           r"dose omission|prescri|dispens|wrong |incorrect|inappropriate|schedule|"
           r"accidental|intentional|technique|monitoring|preparation|storage|"
           r"therapeutic product|availability|supply|confusion|communication|"
           r"use complaint|expired|contraindicated|cross-reactivity|duration|off label|"
           r"unapproved|abuse|dependence|withdrawal|multiple use|interrupted|"
           r"hypersensitivity|injury|unknown indication|knowledge|difficult to|"
           r"intolerance|transfusion|conception|vaping|advertising|information|"
           r"unauthorised|origin unknown|discontinued|enhancing|design|calcium phosphate")

# Categories for reading the term table only; they do not change what is kept.
# First match wins, in this order.
KEPT_GROUPS = [
    ("Switch to another maker's version", r"substitution"),
    ("Patch does not stick", r"adhesion"),
    ("Contamination, sterility, impurity", r"contamina|steril|impurit|infectious agent|cleaning"),
    ("Recall, counterfeit, tampering", r"recall|counterfeit|tamper|adulterat"),
    ("Packaging, label, container, device", r"packag|label|container|closure|blister|leak|"
                                            r"delivery mechanism|dropper|barcode|desiccant|"
                                            r"lot number|identification number|expiration|child resistant"),
    ("Physical defect (look, smell, taste, form)", r"physical|odour|taste|colour|solubil|residue|"
                                                   r"deposit|coating|shape|size|friable|gel"),
    ("General quality or potency", r"quality|potency|formulation|dosage form|ineffective|"
                                   r"reconstitution|raw material|process control|complaint"),
    ("Product mix-up at the maker", r"commingling"),
    ("Distribution, shipping, temperature", r"distribution|shipping|temperature"),
]
REMOVED_GROUPS = [
    ("Borderline: packaging or design", r"difficult to open|package associated injury|design issue|"
                                        r"difficult to swallow|difficult to remove"),
    ("Medication or use error", r"error|wrong |incorrect|inappropriate|dose omission|technique|"
                                r"administered at|administered to|preparation|storage|monitoring|"
                                r"prescri|dispens|schedule|interrupted|multiple use|duration|"
                                r"contraindicated|expired|discontinued|unauthorised"),
    ("Off-label or unapproved use", r"off label|unapproved|unknown indication|use in |use issue"),
    ("Misuse, abuse, accidental exposure", r"misuse|abuse|dependence|withdrawal|intentional|accidental|exposure"),
    ("Drug did not work as expected", r"therapeutic product"),
    ("Shortage or supply", r"availability|supply"),
    ("Confusion or information", r"confusion|communication|information|knowledge|advertising|use complaint"),
    ("About the patient, not the product", r"hypersensitivity|intolerance|transfusion|conception|"
                                           r"vaping|calcium phosphate|performance enhancing|origin unknown"),
]


def category(term: str, groups: list[tuple[str, str]]) -> str:
    for name, pattern in groups:
        if pd.Series([term]).str.contains(pattern).iloc[0]:
            return name
    return "Other"


def main() -> None:
    ids = set(pd.read_csv(FAERS, usecols=["primaryid"])["primaryid"].astype(str))
    print(f"primaryids in the ANDA-linked FAERS file: {len(ids):,}")

    files = sorted({f for pat in ("*/ascii/REAC*.txt", "*/ascii/reac*.txt",
                                  "*/ASCII/REAC*.txt", "*/ASCII/reac*.txt")
                    for f in glob.glob(str(RAW / pat))})
    print(f"REAC files: {len(files)}")

    parts = []
    for fn in files:
        r = pd.read_csv(fn, sep="$", dtype=str, usecols=[0, 2],
                        encoding="latin-1", on_bad_lines="skip")
        r.columns = ["primaryid", "pt"]
        parts.append(r[r["primaryid"].isin(ids)])
    r = pd.concat(parts, ignore_index=True)
    r["pt"] = r["pt"].str.strip().str.lower()

    keep = r["pt"].str.contains(r"\bproduct\b") | r["pt"].isin(ALSO_KEEP)
    drop = r["pt"].str.contains(EXCLUDE)
    q = r[keep & ~drop].drop_duplicates()

    print("\nterms kept (reports per term):")
    print(q["pt"].value_counts().to_string())
    print(f"\nreports with a product-quality term: {q['primaryid'].nunique():,} "
          f"of {r['primaryid'].nunique():,} with any reaction")
    q.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}")

    cand = r[keep].drop_duplicates()
    terms = (cand.groupby("pt")["primaryid"].nunique().rename("reports").reset_index())
    terms["kept"] = ~terms["pt"].str.contains(EXCLUDE)
    terms["category"] = [category(t, KEPT_GROUPS if k else REMOVED_GROUPS)
                         for t, k in zip(terms["pt"], terms["kept"])]
    terms = terms.sort_values(["kept", "reports"], ascending=[False, False])
    terms.to_csv(OUT_TERMS, index=False)
    print("\nby category (reports per term summed):")
    print(terms.groupby(["kept", "category"])["reports"].agg(["size", "sum"]).to_string())
    print(f"Saved -> {OUT_TERMS}")


if __name__ == "__main__":
    main()
