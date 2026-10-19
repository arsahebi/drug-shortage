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
"""

from pathlib import Path
import glob
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                      # Data/15 - FDA - Adverse Event
RAW = BASE / "raw"
FAERS = BASE / "processed" / "faers_all_drugs_anda_linked_2015Q1_2026Q1.csv"
OUT = BASE / "processed" / "faers_product_quality_reactions.csv"

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


if __name__ == "__main__":
    main()
