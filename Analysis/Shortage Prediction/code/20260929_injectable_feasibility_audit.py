"""
Injectable drug shortage feasibility audit.

Question this answers
---------------------
Rob Handfield asked whether it is even possible to get at injectable drugs in the
databases this project already works with, in support of a proposed PhD study on
"effective redundancy" -- the idea that a market can list several manufacturers
and still be fragile, because those manufacturers share a corporate parent, share
a manufacturing site, or are all in shortage at the same time.

This script answers that question with numbers rather than opinion. It does not
estimate the proposed model. It audits whether the four ingredients that model
needs can actually be built from data we hold:

  A. Shortage episodes for injectables, with onset and resolution dates  (UUDIS)
  B. A marketed-product universe we can evaluate as of a date            (FDA NDC Directory)
  C. Corporate ownership, time-varying                                   (firm-name crosswalk)
  D. The physical manufacturing site behind each NDC                     (ProPublica Rx Inspector + DailyMed)

and then demonstrates, end to end on real episodes, how far the independent
owners and independent plants behind a drug fall short of the raw label count.

A caution that governs how every Stage E number should be read: UUDIS carries a
free-text drug name and nothing else, no NDC and no application number, so the
link to the product universe is name-based and lands at ACTIVE INGREDIENT level.
The counts are therefore an ingredient footprint and an upper bound on the true
alternatives, not the supplier count of the product that went short. Stage D
grades every link and records how coarse each episode's market definition is
forced to be.

Each stage prints a verdict and writes its evidence to
`outputs/tables/injectable_feasibility/`. Stages C and D are deliberately
reported as partial findings rather than smoothed over.

Run:
    python 20260929_injectable_feasibility_audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import config as C
from utils import get_logger, normalize_drug_name, write_table

LOG = get_logger("injectable_audit", C.OUT_LOGS / "20260929_injectable_feasibility_audit.log")

TAB = C.OUT_TABS / "injectable_feasibility"
FIG = C.OUT_FIGS / "injectable_feasibility"
for _p in (TAB, FIG):
    _p.mkdir(parents=True, exist_ok=True)

# Validated categorical slots 1/2/3 (see dataviz reference palette; checked with
# validate_palette.js -- worst adjacent CVD dE 9.2, normal-vision dE 27.6).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

# Censoring date for episodes still active in the UUDIS extract.
STUDY_END = pd.Timestamp("2025-12-31")

# The window the proposed study would use. openFDA starts in 2012; UUDIS goes
# back to 1996, so we report both the full history and the comparable window.
PILOT_START_YEAR = 2012

# Date of the FDA NDC Directory extract on disk. The directory is a live
# snapshot, not an archive, so this bounds how far back supplier counts are
# trustworthy -- see the survivorship warning in stage_b_product_universe.
NDC_SNAPSHOT_DATE = pd.Timestamp("2025-04-20")

# ProPublica uses ISO-2 country codes, the DailyMed parse yields ISO-3. Map the
# ISO-3 codes that actually appear so the two bridges agree.
_ISO3_TO_ISO2 = {
    "USA": "US", "IND": "IN", "CHN": "CN", "CAN": "CA", "DEU": "DE", "TWN": "TW",
    "NOR": "NO", "KOR": "KR", "ITA": "IT", "AUT": "AT", "CHE": "CH", "IRL": "IE",
    "GBR": "GB", "FRA": "FR", "ESP": "ES", "JPN": "JP", "ISR": "IL", "SWE": "SE",
    "BEL": "BE", "NLD": "NL", "DNK": "DK", "HUN": "HU", "POL": "PL", "PRI": "PR",
    "SGP": "SG", "AUS": "AU", "BRA": "BR", "MEX": "MX", "ARG": "AR", "SVN": "SI",
    "HRV": "HR", "CZE": "CZ", "PRT": "PT", "FIN": "FI", "GRC": "GR", "ROU": "RO",
}


# ==============================================================================
# Stage A -- shortage episodes
# ==============================================================================

def load_uudis() -> pd.DataFrame:
    """UUDIS episode file -> one clean row per shortage episode.

    The raw sheet carries its real header on the second row. `parenteral` is a
    y/n flag maintained by UUDIS, which is what makes an injectable-only study
    possible without inferring dosage form from a drug-name string.
    """
    LOG.info("Stage A: loading UUDIS episodes from %s", C.UUTAH_FILE.name)
    df = pd.read_excel(C.UUTAH_FILE, header=1)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "Drug Shortages": "drug_text",
        "status12312025": "status_raw",
        "AHFS": "ahfs",
        "Reason": "reason_raw",
        "yr": "onset_year",
        "Date Notified": "date_onset",
        "Date Resolved": "date_resolved",
        "Sole source yes or no": "sole_source_raw",
        "parenteral": "parenteral_raw",
        "Control Substance Schedule": "dea_schedule",
    })

    df["date_onset"] = pd.to_datetime(df["date_onset"], errors="coerce")
    df["date_resolved"] = pd.to_datetime(df["date_resolved"], errors="coerce")
    df["status"] = df["status_raw"].astype(str).str.strip().str.lower()
    df["parenteral"] = df["parenteral_raw"].astype(str).str.strip().str.lower()
    df["reason"] = df["reason_raw"].astype(str).str.strip().str.lower()

    # Sole-source is coded inconsistently (YES/NO plus "NN", "NE", blanks). Map
    # what is unambiguous and keep the rest as missing rather than guessing --
    # the messiness is itself a finding for the feasibility write-up.
    ss = df["sole_source_raw"].astype(str).str.strip().str.upper()
    df["sole_source"] = np.where(ss == "YES", 1.0, np.where(ss == "NO", 0.0, np.nan))

    # Survival framing: resolved episodes are events, still-active episodes are
    # right-censored at the extract date.
    df["event_observed"] = df["date_resolved"].notna().astype(int)
    end = df["date_resolved"].fillna(STUDY_END)
    df["duration_days"] = (end - df["date_onset"]).dt.days

    df["drug_norm"] = df["drug_text"].map(normalize_drug_name)
    return df


def stage_a_episode_inventory(uu: pd.DataFrame) -> dict:
    """Count what is actually there, and be explicit about what is unusable."""
    LOG.info("Stage A: episode inventory")

    inj = uu[uu["parenteral"] == "y"]
    oral = uu[uu["parenteral"] == "n"]

    # Data-quality flags. Negative durations mean a resolution date that precedes
    # the notification date -- a real cleaning task, not a blocker.
    bad_dur = uu["duration_days"] < 0
    no_onset = uu["date_onset"].isna()

    summary = pd.DataFrame([
        {"metric": "episodes, all dosage forms", "value": len(uu)},
        {"metric": "episodes, parenteral", "value": len(inj)},
        {"metric": "episodes, non-parenteral", "value": len(oral)},
        {"metric": "parenteral share", "value": round(len(inj) / len(uu), 3)},
        {"metric": f"parenteral episodes, onset {PILOT_START_YEAR}+",
         "value": int((inj["onset_year"] >= PILOT_START_YEAR).sum())},
        {"metric": "parenteral, resolved (event observed)", "value": int(inj["event_observed"].sum())},
        {"metric": "parenteral, still active (censored)", "value": int((1 - inj["event_observed"]).sum())},
        {"metric": "parenteral, median duration (days)", "value": float(inj.loc[inj["event_observed"] == 1, "duration_days"].median())},
        {"metric": "non-parenteral, median duration (days)", "value": float(oral.loc[oral["event_observed"] == 1, "duration_days"].median())},
        {"metric": "earliest onset year", "value": int(uu["onset_year"].min())},
        {"metric": "latest onset year", "value": int(uu["onset_year"].max())},
        {"metric": "DQ: rows with no onset date", "value": int(no_onset.sum())},
        {"metric": "DQ: rows with negative duration", "value": int(bad_dur.sum())},
        {"metric": "DQ: sole-source unusable/blank", "value": int(uu["sole_source"].isna().sum())},
        {"metric": "DQ: reason == 'unknown'", "value": int((uu["reason"] == "unknown").sum())},
    ])
    summary.to_csv(TAB / "a_episode_inventory.csv", index=False)

    by_year = (uu[uu["onset_year"] >= PILOT_START_YEAR]
               .groupby(["onset_year", "parenteral"]).size()
               .unstack(fill_value=0).rename(columns={"y": "parenteral", "n": "non_parenteral"}))
    by_year.to_csv(TAB / "a_episodes_by_year.csv")

    # Reason mix matters for the proposed model: manufacturing/quality causes are
    # the ones our facility-level signals could speak to.
    reason_mix = (inj["reason"].value_counts().head(15).rename_axis("reason")
                  .reset_index(name="episodes"))
    reason_mix.to_csv(TAB / "a_parenteral_reason_mix.csv", index=False)

    LOG.info("  %d episodes total; %d parenteral (%.0f%%)", len(uu), len(inj), 100 * len(inj) / len(uu))
    LOG.info("  parenteral median duration %.0f d vs non-parenteral %.0f d",
             inj.loc[inj["event_observed"] == 1, "duration_days"].median(),
             oral.loc[oral["event_observed"] == 1, "duration_days"].median())
    LOG.info("  VERDICT A: injectable episodes are directly available and pre-flagged.")
    return {"n_total": len(uu), "n_inj": len(inj), "n_oral": len(oral)}


# ==============================================================================
# Stage B -- marketed injectable product universe
# ==============================================================================

def load_ndc_injectables() -> pd.DataFrame:
    """FDA NDC Directory, restricted to injectable dosage forms.

    STARTMARKETINGDATE / ENDMARKETINGDATE are what make "who was marketing this
    on the day the shortage began" answerable at all.
    """
    LOG.info("Stage B: loading FDA NDC Directory")
    # The FDA ships this file latin-1, not UTF-8.
    ndc = pd.read_csv(C.NDC_PRODUCT_CSV, low_memory=False, encoding="latin-1")
    form = ndc["DOSAGEFORMNAME"].astype(str).str.upper()
    inj = ndc[form.str.contains("INJECT", na=False)].copy()

    inj["start_mkt"] = pd.to_datetime(inj["STARTMARKETINGDATE"], format="%Y%m%d", errors="coerce")
    inj["end_mkt"] = pd.to_datetime(inj["ENDMARKETINGDATE"], format="%Y%m%d", errors="coerce")
    inj["product_ndc"] = inj["PRODUCTNDC"].astype(str).str.strip()
    inj["labeler"] = inj["LABELERNAME"].astype(str).str.strip()

    # Active ingredient key. SUBSTANCENAME is semicolon-delimited for
    # combination products; normalize each part so "SODIUM CHLORIDE" and
    # "sodium chloride injection, USP" land on the same key.
    inj["ingredient_norm"] = (inj["SUBSTANCENAME"].astype(str)
                              .str.split(";").str[0].map(normalize_drug_name))
    inj["n_ingredients"] = inj["SUBSTANCENAME"].astype(str).str.count(";") + 1
    return inj


def stage_b_product_universe(inj: pd.DataFrame) -> pd.DataFrame:
    LOG.info("Stage B: injectable product universe")

    # Survivorship warning. The NDC Directory is a live snapshot: FDA drops
    # delisted products rather than retaining them with a closing date, so a
    # supplier who exited before the file date is invisible. Onset-date supplier
    # counts are therefore a floor, and the bias grows the further back you go.
    # The fix for a real study is an archived NDC snapshot per year (FDA
    # publishes these; openFDA also exposes historical NDC records).
    n_stale = int((inj["end_mkt"].notna() & (inj["end_mkt"] < NDC_SNAPSHOT_DATE)).sum())
    LOG.warning("  SURVIVORSHIP: this snapshot retains %d products delisted before its "
                "own file date, so a supplier who exited earlier is invisible and "
                "historical supplier counts are a lower bound. Use per-year archived "
                "NDC snapshots for the real study.", n_stale)

    summary = pd.DataFrame([
        {"metric": "injectable product NDCs", "value": len(inj)},
        {"metric": "DQ: products delisted before the snapshot date, still listed",
         "value": n_stale},
        {"metric": "distinct labelers", "value": inj["labeler"].nunique()},
        {"metric": "distinct normalized ingredients", "value": inj["ingredient_norm"].nunique()},
        {"metric": "single-ingredient products", "value": int((inj["n_ingredients"] == 1).sum())},
        {"metric": "with a parseable start-marketing date", "value": int(inj["start_mkt"].notna().sum())},
        {"metric": "generic (ANDA)", "value": int((inj["MARKETINGCATEGORYNAME"] == "ANDA").sum())},
        {"metric": "brand (NDA)", "value": int((inj["MARKETINGCATEGORYNAME"] == "NDA").sum())},
        {"metric": "biologic (BLA)", "value": int((inj["MARKETINGCATEGORYNAME"] == "BLA").sum())},
    ])
    summary.to_csv(TAB / "b_product_universe.csv", index=False)
    LOG.info("  %d injectable NDCs / %d labelers / %d ingredients",
             len(inj), inj["labeler"].nunique(), inj["ingredient_norm"].nunique())
    LOG.info("  VERDICT B: a date-evaluable marketed-product universe exists.")
    return inj


# ==============================================================================
# Stage C -- ownership and manufacturing-site layers
# ==============================================================================

# Corporate boilerplate carried by NDC labeler names but not by the crosswalk's
# standardized names ("americanregent", "hospira"). Stripping it is what takes
# the label->firm match from 48% of NDC rows to 80%.
_FIRM_SUFFIX_RX = re.compile(
    r"\b(incorporated|inc|llc|l\.l\.c|lp|llp|plc|ltd|limited|corp|corporation|co|"
    r"company|companies|gmbh|ag|sa|s\.a|nv|bv|pty|pvt|private|holdings?|group|"
    r"usa|us|u\.s|america|american|north|international|intl|labs?|laboratories|"
    r"laboratory|pharma|pharm|pharmaceutical|pharmaceuticals|healthcare|health|"
    r"sciences|science|therapeutics|products|division|dba|subsidiary|of|the|and|a)\b")
# "Wyeth Pharmaceuticals LLC, a subsidiary of Pfizer Inc." -> keep only "Wyeth".
_FIRM_TAIL_RX = re.compile(r",\s*(?:a|an)\s+\w+\s+of\b|\bdba\b|\bdivision of\b|\bsubsidiary of\b")


def firm_key(name: str) -> str:
    """Collapse a firm name to the crosswalk's standardized-name shape."""
    if not isinstance(name, str):
        return ""
    s = _FIRM_TAIL_RX.split(name.lower())[0]
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _FIRM_SUFFIX_RX.sub(" ", s)
    return re.sub(r"\s+", "", s)


def load_ownership() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Firm-name standardization + dated parent/subsidiary links.

    Two files, two jobs:
      std_firm_names  -- messy label name  -> standardized firm name
      lookup (sheet)  -- standardized sub  -> standardized parent, with a date

    The date is the part that matters. Ownership in generic injectables churns,
    so "are these two labelers independent" has to be asked as of the shortage
    onset, not as of today.
    """
    LOG.info("Stage C: loading firm-name and ownership crosswalks")
    std = pd.read_excel(C.FIRM_STD_NAMES_XLSX, sheet_name="Sheet1")
    std = std[["OriginalName", "NewStandardizedName"]].dropna()

    links = pd.read_excel(C.FIRM_LOOKUP_XLSX, sheet_name="Final_table_for_lookup")
    links = links.dropna(subset=["Std_parent", "Std_subsidiary"])
    # Before_2000 rows have no usable day; treat them as owned from 1999-12-31.
    yr = pd.to_numeric(links["Year"], errors="coerce")
    links["acquired_year"] = np.where(yr.between(1900, 2030), yr, np.nan)
    links["acquired_date"] = pd.to_datetime(
        dict(year=links["acquired_year"].fillna(1999).astype(int),
             month=pd.to_numeric(links["Month"], errors="coerce").fillna(12).clip(1, 12).astype(int),
             day=pd.to_numeric(links["Day"], errors="coerce").fillna(31).clip(1, 28).astype(int)),
        errors="coerce")
    return std, links


def load_ndc_fei() -> pd.DataFrame:
    """NDC -> manufacturing site (FEI), from both bridges we hold.

    ProPublica's Rx Inspector release is the primary source: it reaches 50% of
    injectable NDCs and 80% of the generic (ANDA) injectables that matter for a
    sterile-injectable study, against 11% for the DailyMed parse. It is
    generics-only by design, excluding brand NDAs, gases and intradermal
    products, so the DailyMed map is unioned in rather than dropped. It still
    contributes a few hundred NDCs ProPublica does not carry.

    ProPublica's `linkage_method` is kept: SPL_DUNS_FEI links come from an
    identifier printed on the label, ADDRESS_MATCH links come from fuzzy address
    matching and are the weaker of the two. Anything that turns on site identity
    should be checked both ways.
    """
    LOG.info("Stage C: loading NDC->FEI site bridges")

    pp = pd.read_csv(C.PROPUBLICA_NDC_FEI_CSV, low_memory=False)
    pp = pd.DataFrame({
        "product_ndc": pp["ndc"].astype(str).str.strip(),
        "fei": pd.to_numeric(pp["fei"], errors="coerce"),
        "country": pp["country"].astype(str).str.strip(),
        "linkage_method": pp["linkage_method"],
        # ProPublica flags API-only sites; a raw-ingredient supplier is a real
        # dependency but is not an alternative finished-dose source.
        "api_only": pp["api_mfr"].notna(),
        "source": "propublica",
    })

    dm = pd.read_csv(C.NDC_FEI_MAP_CSV, low_memory=False, encoding="latin-1")
    dm = dm.dropna(subset=["manufacture_ndc"])
    dm = pd.DataFrame({
        "product_ndc": dm["manufacture_ndc"].astype(str).str.strip(),
        # FEI_NUMBER arrives mixed: floats, digit strings, whitespace-only cells.
        "fei": pd.to_numeric(dm["FEI_NUMBER"], errors="coerce"),
        # DailyMed addresses carry a trailing ISO-3 code; fold it to ISO-2 so the
        # two bridges do not report "US" and "USA" as different countries.
        "country": (dm["ADDRESS"].astype(str).str.extract(r"\(([A-Z]{3})\)\s*$")[0]
                    .map(_ISO3_TO_ISO2)),
        "linkage_method": "SPL_DUNS_FEI",
        "api_only": False,
        "source": "dailymed",
    })

    both = pd.concat([pp, dm], ignore_index=True).dropna(subset=["fei"])
    both = both.drop_duplicates(subset=["product_ndc", "fei"])
    LOG.info("  ProPublica %d rows / %d NDCs; DailyMed %d rows / %d NDCs; "
             "union %d NDC-FEI pairs",
             len(pp), pp["product_ndc"].nunique(), len(dm), dm["product_ndc"].nunique(),
             len(both))
    return both


def stage_c_independence_layers(inj: pd.DataFrame, std: pd.DataFrame,
                                links: pd.DataFrame, fei: pd.DataFrame) -> pd.DataFrame:
    """Attach standardized owner and manufacturing site to each injectable NDC."""
    LOG.info("Stage C: attaching ownership and site to injectable NDCs")

    # --- owner ---
    # Two passes: the crosswalk's own raw strings first (exact, highest trust),
    # then the collapsed firm key. Standardized names are seeded into both
    # directions so a labeler already written in standardized form resolves too.
    name_map: dict[str, str] = {}
    for orig, newn in zip(std["OriginalName"], std["NewStandardizedName"]):
        name_map.setdefault(str(orig).lower().strip(), newn)
        name_map.setdefault(firm_key(orig), newn)
    for newn in std["NewStandardizedName"].unique():
        name_map.setdefault(str(newn).lower().strip(), newn)
        name_map.setdefault(firm_key(newn), newn)
    name_map.pop("", None)

    inj["labeler_key"] = inj["labeler"].str.lower().str.strip()
    matched = inj["labeler_key"].map(name_map)
    matched = matched.fillna(inj["labeler"].map(lambda s: name_map.get(firm_key(s))))
    inj["firm_std_matched"] = matched.notna().astype(int)
    # Unmatched firms keep their own collapsed key, so they still count as one
    # distinct owner rather than collapsing together under missing.
    inj["firm_std"] = matched.fillna(inj["labeler"].map(firm_key))

    # --- site ---
    # Finished-dose sites only. An API-only plant is a real dependency, and one
    # worth studying, but it is not an alternative source of the product.
    fin = fei[~fei["api_only"].fillna(False)]
    site = (fin.groupby("product_ndc")
               .agg(fei_list=("fei", lambda s: sorted({int(v) for v in s})),
                    country_list=("country", lambda s: sorted(set(s.dropna()))),
                    site_sources=("source", lambda s: sorted(set(s))),
                    weak_link_share=("linkage_method",
                                     lambda s: round((s == "ADDRESS_MATCH").mean(), 2))))
    inj = inj.merge(site, on="product_ndc", how="left")
    inj["has_site"] = inj["fei_list"].notna().astype(int)

    # The rollup can only fire for firms that appear as a subsidiary in the
    # lookup, so report that separately -- it bounds Stage E, not the crosswalk.
    firms = set(inj["firm_std"].dropna())
    n_sub = len(firms & set(links["Std_subsidiary"]))

    cov = pd.DataFrame([
        {"layer": "labeler -> standardized firm name (NDC rows)",
         "covered": int(inj["firm_std_matched"].sum()), "of": len(inj),
         "pct": round(100 * inj["firm_std_matched"].mean(), 1)},
        {"layer": "distinct labelers resolved to a standardized firm",
         "covered": inj.loc[inj["firm_std_matched"] == 1, "labeler"].nunique(),
         "of": inj["labeler"].nunique(),
         "pct": round(100 * inj.loc[inj["firm_std_matched"] == 1, "labeler"].nunique()
                      / inj["labeler"].nunique(), 1)},
        {"layer": "injectable firms with a parent link in the lookup",
         "covered": n_sub, "of": len(firms),
         "pct": round(100 * n_sub / len(firms), 1)},
    ])

    # Break the site bridge out by source, and by marketing category -- the
    # headline coverage number is misleading on its own, because ProPublica
    # deliberately excludes brand NDAs and the study population is generics.
    ndcs = set(inj["product_ndc"])
    for src, label in [("propublica", "NDC -> site, ProPublica alone"),
                       ("dailymed", "NDC -> site, DailyMed parse alone")]:
        s = set(fin.loc[fin["source"] == src, "product_ndc"]) & ndcs
        cov.loc[len(cov)] = {"layer": label, "covered": len(s), "of": len(inj),
                             "pct": round(100 * len(s) / len(inj), 1)}
    cov.loc[len(cov)] = {"layer": "NDC -> site, both sources unioned",
                         "covered": int(inj["has_site"].sum()), "of": len(inj),
                         "pct": round(100 * inj["has_site"].mean(), 1)}
    anda = inj[inj["MARKETINGCATEGORYNAME"] == "ANDA"]
    cov.loc[len(cov)] = {"layer": "NDC -> site, generic (ANDA) injectables only",
                         "covered": int(anda["has_site"].sum()), "of": len(anda),
                         "pct": round(100 * anda["has_site"].mean(), 1)}
    cov.to_csv(TAB / "c_independence_layer_coverage.csv", index=False)

    n_fei = len({f for lst in inj["fei_list"].dropna() for f in lst})
    ctry = (fin[fin["product_ndc"].isin(ndcs)]
            .drop_duplicates("fei")["country"].value_counts().head(12))
    ctry.rename_axis("country").reset_index(name="distinct_sites").to_csv(
        TAB / "c_injectable_site_countries.csv", index=False)

    LOG.info("  firm-name crosswalk covers %.0f%% of injectable labels; "
             "site bridge covers %.0f%% of all injectable NDCs and %.0f%% of "
             "generic (ANDA) injectables (%d distinct plants)",
             100 * inj["firm_std_matched"].mean(), 100 * inj["has_site"].mean(),
             100 * anda["has_site"].mean(), n_fei)
    LOG.info("  VERDICT C: ownership layer is broad; site layer is usable for "
             "generic injectables via ProPublica, thin for brand NDAs by design.")
    return inj


# ==============================================================================
# Stage D -- can UUDIS episodes be joined to the product universe?
# ==============================================================================

# Packaging / presentation words that appear in UUDIS drug strings but carry no
# ingredient identity. `normalize_drug_name` handles dose forms and units; these
# are the extras specific to how UUDIS writes injectable presentations.
_UUDIS_NOISE = {
    "large", "small", "volume", "bags", "bag", "vials", "vial", "syringes",
    "syringe", "flush", "premixed", "premix", "concentrated", "concentrate",
    "soln", "free", "preservative", "pvc", "dehp", "ml", "l", "and", "or",
    "irrigation", "adult", "pediatric", "kit", "pf", "usp",
}
_PAREN_RX = re.compile(r"\([^)]*\)")

# Strength and container, pulled back out of the UUDIS string. These are what
# separate "0.9% sodium chloride flush syringes" from "14.6% sodium chloride
# injection" -- two episodes that the ingredient key alone collapses together.
# Only about 11% of episodes carry either, which is itself the finding: for the
# rest, the source text cannot support a market definition finer than ingredient.
_STRENGTH_RX = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|mg\s*/\s*ml|mcg\s*/\s*ml|units?\s*/\s*ml|meq\s*/\s*ml|"
    r"mg|mcg|g|units?|meq|mmol))", re.I)
_CONTAINER_RX = re.compile(
    r"\b(vial|syringe|bag|ampul\w*|premix\w*|pen|cartridge|bottle|"
    r"flush|irrigation|large volume|small volume)\w*\b", re.I)


def uudis_presentation(text: str) -> tuple[str, str, str]:
    """Pull strength and container out of a UUDIS string, and grade it.

    Returns (strength, container, granularity) where granularity is:
      'ingredient_strength_form' -- both present, a real market definition
      'ingredient_strength'      -- strength only
      'ingredient_form'          -- container only
      'ingredient_only'          -- neither; the string is a bare drug name

    This does not change any join. It records how coarse each episode's market
    definition is forced to be, so a downstream model can condition on it or
    restrict to the rows where a finer definition is actually available.
    """
    if not isinstance(text, str):
        return "", "", "ingredient_only"
    s = _PAREN_RX.sub(" ", text)
    strengths = sorted({m.group(1).lower().replace(" ", "") for m in _STRENGTH_RX.finditer(s)})
    containers = sorted({m.group(1).lower() for m in _CONTAINER_RX.finditer(s)})
    st, ct = "|".join(strengths), "|".join(containers)
    if st and ct:
        gran = "ingredient_strength_form"
    elif st:
        gran = "ingredient_strength"
    elif ct:
        gran = "ingredient_form"
    else:
        gran = "ingredient_only"
    return st, ct, gran


def uudis_ingredient_key(text: str) -> str:
    """Reduce a UUDIS presentation string to its lead active ingredient.

    'Sodium chloride 14.6% concentrated soln'  -> 'sodium chloride'
    'Prothrombin complex concentrate (Kcentra)' -> 'prothrombin complex'

    Deliberately conservative: the parenthetical brand name is dropped rather
    than used, so a match means the ingredient text agreed on its own.
    """
    if not isinstance(text, str):
        return ""
    s = _PAREN_RX.sub(" ", text)
    s = normalize_drug_name(s)
    toks = [t for t in s.split() if t not in _UUDIS_NOISE]
    return " ".join(toks)


def stage_d_linkage(uu: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Join test. This is the study's go/no-go, so report it honestly.

    UUDIS carries a free-text presentation string and nothing else: no NDC, no
    ANDA, no application number. Every link to the product universe is therefore
    name-based, and this stage grades each link rather than reporting a single
    match rate, because a match rate says nothing about whether the matches are
    right.

    Three tiers:
      exact          -- the normalized UUDIS ingredient equals an NDC ingredient
      prefix         -- NDC ingredient is a prefix of it, i.e. salt-form
                        resolution (doxycycline -> doxycycline hyclate). Sound.
      lone_candidate -- one candidate shared the first token and it was taken on
                        faith. This tier contains real errors: 'Calcium Acetate'
                        resolves to calcium chloride, 'Copper injection' to
                        copper oxodotreotide cu. It is reported and carried, but
                        excluded from the headline and from `modelable`.
    """
    LOG.info("Stage D: UUDIS -> NDC ingredient linkage")

    ing = set(inj["ingredient_norm"].dropna()) - {""}
    # Index ingredients by first token so a UUDIS string can be matched by its
    # leading ingredient when the full string carries extra qualifiers.
    by_first: dict[str, list[str]] = {}
    for g in ing:
        by_first.setdefault(g.split()[0], []).append(g)

    def match(key: str) -> tuple[str | None, str | None]:
        if not key:
            return None, None
        if key in ing:
            return key, "exact"
        head = key.split()[0]
        cands = by_first.get(head, [])
        if not cands:
            return None, None
        # Prefer a candidate that is a prefix of the UUDIS key (e.g. UUDIS
        # 'sodium chloride bacteriostatic' -> NDC 'sodium chloride').
        pref = [c for c in cands if key.startswith(c)]
        if pref:
            return max(pref, key=len), "prefix"
        return (cands[0], "lone_candidate") if len(cands) == 1 else (None, None)

    epi = uu[uu["parenteral"] == "y"].copy()
    epi["ingredient_key"] = epi["drug_text"].map(uudis_ingredient_key)
    hits = epi["ingredient_key"].map(match)
    epi["ingredient_matched"] = [h[0] for h in hits]
    epi["match_method"] = [h[1] for h in hits]
    epi["matched"] = epi["ingredient_matched"].notna().astype(int)
    # Only exact and prefix are defensible without manual review.
    epi["match_confident"] = epi["match_method"].isin(["exact", "prefix"]).astype(int)

    # How coarse is each episode's market definition forced to be?
    pres = epi["drug_text"].map(uudis_presentation)
    epi["uudis_strength"] = [p[0] for p in pres]
    epi["uudis_container"] = [p[1] for p in pres]
    epi["granularity"] = [p[2] for p in pres]

    modelable = (epi["match_confident"].astype(bool) & epi["date_onset"].notna()
                 & (epi["duration_days"] >= 0))
    epi["modelable"] = modelable.astype(int)

    n_fine = int((epi["granularity"] != "ingredient_only").sum())
    summary = pd.DataFrame([
        {"metric": "parenteral episodes", "value": len(epi)},
        {"metric": "matched, any method", "value": int(epi["matched"].sum())},
        {"metric": "  of which exact", "value": int((epi["match_method"] == "exact").sum())},
        {"metric": "  of which prefix (salt-form)", "value": int((epi["match_method"] == "prefix").sum())},
        {"metric": "  of which lone_candidate (UNVERIFIED, excluded)",
         "value": int((epi["match_method"] == "lone_candidate").sum())},
        {"metric": "confident match rate (exact + prefix)",
         "value": round(epi["match_confident"].mean(), 3)},
        {"metric": f"confident AND onset {PILOT_START_YEAR}+ AND clean duration",
         "value": int((epi["modelable"] & (epi["onset_year"] >= PILOT_START_YEAR)).sum())},
        {"metric": "distinct ingredients matched", "value": int(epi["ingredient_matched"].nunique())},
        {"metric": "episodes whose text supports a finer market than ingredient",
         "value": n_fine},
        {"metric": "  share", "value": round(n_fine / len(epi), 3)},
    ])
    summary.to_csv(TAB / "d_linkage_summary.csv", index=False)

    (epi["granularity"].value_counts().rename_axis("granularity")
        .reset_index(name="episodes").to_csv(TAB / "d_market_granularity.csv", index=False))

    # Write the unverified tier out in full. It is a manual-review worklist, not
    # a rounding error: this is where the false matches live.
    (epi.loc[epi["match_method"] == "lone_candidate",
             ["drug_text", "ingredient_key", "ingredient_matched", "onset_year"]]
        .drop_duplicates().sort_values("drug_text")
        .to_csv(TAB / "d_unverified_matches_REVIEW.csv", index=False))

    epi.loc[epi["matched"] == 0, ["drug_text", "ingredient_key", "onset_year"]] \
       .head(200).to_csv(TAB / "d_unmatched_examples.csv", index=False)

    LOG.info("  %d/%d episodes matched confidently (%.0f%%); %d more matched only "
             "by an unverified lone-candidate rule and are excluded",
             epi["match_confident"].sum(), len(epi), 100 * epi["match_confident"].mean(),
             int((epi["match_method"] == "lone_candidate").sum()))
    LOG.info("  market granularity: %d/%d episodes (%.0f%%) name a strength or "
             "container; the rest can only be defined at ingredient level",
             n_fine, len(epi), 100 * n_fine / len(epi))
    LOG.info("  VERDICT D: PARTLY. The link is name-only and lands at ingredient "
             "level. Defining the market that actually went short is the study's "
             "hardest measurement problem, not a refinement.")
    return epi


# ==============================================================================
# Stage E -- ingredient footprint at onset, demonstrated
# ==============================================================================

def owner_as_of(firm: str, onset: pd.Timestamp, links: pd.DataFrame) -> str:
    """Roll a standardized firm up to its parent, if it was owned by then.

    One hop only. A longer chain would need iteration, which is exactly the kind
    of thing the pilot semester should test rather than assume.
    """
    hit = links[(links["Std_subsidiary"] == firm) & (links["acquired_date"] <= onset)]
    if len(hit) == 0:
        return firm
    return hit.sort_values("acquired_date").iloc[-1]["Std_parent"]


def stage_e_redundancy(epi: pd.DataFrame, inj: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Measure the ingredient's injectable footprint at shortage onset.

    READ THE NAMES LITERALLY. Because UUDIS carries only a drug name, every
    count here is over *all injectable NDCs sharing the active ingredient*, not
    over the product that actually went short. "0.9% sodium chloride flush
    syringes" and "14.6% sodium chloride injection" are different markets and
    get near-identical counts, because both roll up to sodium chloride. These
    are an upper bound on the true alternatives, and the overcount is worst
    exactly where it matters most, on commodity products with many
    presentations.

      footprint_ndcs        -- injectable NDCs of the ingredient marketed at onset
      footprint_labelers    -- distinct label holders among them
      footprint_owners      -- distinct corporate parents, as of the onset date
      footprint_sites       -- distinct finished-dose plants (FEI)
      concurrent_shortages  -- other episodes on the same ingredient already in
                               shortage on the onset date

    Only the last one is measured at the right granularity, because it comes
    from the episode file rather than the product file. It is also the half of
    the construct no product database can produce on its own.

    Turning footprint into a real supplier count needs strength and presentation
    matched through to the NDC, which the UUDIS text supports for roughly a
    tenth of episodes (see `granularity` from Stage D).
    """
    LOG.info("Stage E: ingredient footprint at onset (see docstring: NOT a "
             "market-level supplier count)")

    mat = epi[(epi["modelable"] == 1) & (epi["onset_year"] >= PILOT_START_YEAR)].copy()

    # Every dated episode on a matched ingredient, for the concurrency check --
    # including ones outside the pilot window, since an older episode can still
    # be open when a new one starts.
    open_epi = epi[epi["matched"].astype(bool) & epi["date_onset"].notna()].copy()
    open_epi["date_end"] = open_epi["date_resolved"].fillna(STUDY_END)
    conc_by_ing = {k: v for k, v in open_epi.groupby("ingredient_matched")}
    prod = inj.dropna(subset=["ingredient_norm"])
    prod = prod[prod["ingredient_norm"] != ""]
    by_ing = {k: v for k, v in prod.groupby("ingredient_norm")}

    # Cache parent lookups -- the same firm recurs across hundreds of episodes.
    owner_cache: dict[tuple[str, int], str] = {}

    rows = []
    for _, e in mat.iterrows():
        g = by_ing.get(e["ingredient_matched"])
        if g is None:
            continue
        onset = e["date_onset"]
        # Marketed on the onset date: started before it, and not yet ended.
        live = g[(g["start_mkt"].notna()) & (g["start_mkt"] <= onset) &
                 ((g["end_mkt"].isna()) | (g["end_mkt"] >= onset))]
        if len(live) == 0:
            continue

        owners = set()
        for f in live["firm_std"].dropna().unique():
            k = (f, onset.year)
            if k not in owner_cache:
                owner_cache[k] = owner_as_of(f, onset, links)
            owners.add(owner_cache[k])

        sites = {s for lst in live["fei_list"].dropna() for s in lst}

        # Other presentations of the same ingredient already in shortage when
        # this one began. Compare on the drug string so a repeat episode of the
        # same presentation is not counted against itself.
        o = conc_by_ing[e["ingredient_matched"]]
        concurrent = int(((o["date_onset"] < onset) & (o["date_end"] >= onset) &
                          (o["drug_text"] != e["drug_text"])).sum())

        rows.append({
            "drug_text": e["drug_text"],
            "ingredient": e["ingredient_matched"],
            "onset_year": e["onset_year"],
            "date_onset": onset,
            "duration_days": e["duration_days"],
            "event_observed": e["event_observed"],
            "sole_source_uudis": e["sole_source"],
            "reason": e["reason"],
            "footprint_ndcs": len(live),
            "footprint_labelers": live["firm_std"].nunique(),
            "footprint_owners": len(owners),
            "footprint_sites": len(sites) if sites else np.nan,
            "site_coverage": round(live["has_site"].mean(), 3),
            "concurrent_shortages": concurrent,
        })

    red = pd.DataFrame(rows)
    if red.empty:
        LOG.warning("  no episodes survived the onset-date product join")
        return red

    red["owner_gap"] = red["footprint_labelers"] - red["footprint_owners"]
    red["consolidated"] = (red["owner_gap"] > 0).astype(int)

    write_table(red, C.OUT_DATA / "injectable_redundancy_demo.parquet", LOG)
    red.to_csv(TAB / "e_redundancy_by_episode.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "episodes with an onset-date supplier count", "value": len(red)},
        {"metric": "mean footprint labelers", "value": round(red["footprint_labelers"].mean(), 2)},
        {"metric": "mean effective owners", "value": round(red["footprint_owners"].mean(), 2)},
        {"metric": "episodes where owners < labelers", "value": int(red["consolidated"].sum())},
        {"metric": "share where the label count overstates independent ownership",
         "value": round(red["consolidated"].mean(), 3)},
        {"metric": "max labelers collapsing to one owner", "value": int(red["owner_gap"].max())},
        {"metric": "episodes with >=1 mapped manufacturing site",
         "value": int(red["footprint_sites"].notna().sum())},
        {"metric": "episodes with >=1 alternative already in shortage",
         "value": int((red["concurrent_shortages"] > 0).sum())},
        {"metric": "share with a concurrently unavailable alternative",
         "value": round((red["concurrent_shortages"] > 0).mean(), 3)},
        {"metric": "mean concurrent shortages on the same ingredient",
         "value": round(red["concurrent_shortages"].mean(), 2)},
    ])
    summary.to_csv(TAB / "e_redundancy_summary.csv", index=False)

    # Does the construct move with the outcome at all? Descriptive and
    # unadjusted, on resolved episodes -- not the proposed survival model. Read
    # it as "the naive version of H1 does not survive contact with the data",
    # which is a useful thing for the pilot to know before it starts.
    obs = red[red["event_observed"] == 1]
    if len(obs) > 30:
        bins = pd.cut(obs["footprint_owners"], [0, 1, 2, 4, np.inf],
                      labels=["1 owner", "2 owners", "3-4 owners", "5+ owners"])
        dur = (obs.groupby(bins, observed=True)["duration_days"]
                  .agg(episodes="size", median_days="median", mean_days="mean").round(1))
        dur.to_csv(TAB / "e_duration_by_footprint_owners.csv")
        LOG.info("  median duration by effective owners:\n%s", dur.to_string())

        cbins = pd.cut(obs["concurrent_shortages"], [-0.1, 0, 1, 3, np.inf],
                       labels=["none", "1", "2-3", "4+"])
        cdur = (obs.groupby(cbins, observed=True)["duration_days"]
                   .agg(episodes="size", median_days="median", mean_days="mean").round(1))
        cdur.to_csv(TAB / "e_duration_by_concurrent_shortages.csv")
        LOG.info("  median duration by concurrent shortages:\n%s", cdur.to_string())

        _confounding_check(obs)

    LOG.info("  footprint labelers %.2f vs independent owners %.2f; "
             "%.0f%% of episodes overstate independence via ownership; "
             "%.0f%% had an alternative already in shortage",
             red["footprint_labelers"].mean(), red["footprint_owners"].mean(),
             100 * red["consolidated"].mean(),
             100 * (red["concurrent_shortages"] > 0).mean())
    LOG.info("  VERDICT E: the construct is constructible end to end, but at "
             "INGREDIENT granularity. Only concurrent_shortages is measured at "
             "the granularity of the product that actually went short.")
    return red


def _confounding_check(obs: pd.DataFrame) -> None:
    """Why the unadjusted association runs the wrong way.

    Across episodes, more suppliers looks like *longer* shortages -- the opposite
    of the proposed hypothesis. It is market-size confounding: big commodity
    injectables (sodium chloride, dextrose) carry both the most suppliers and the
    longest episodes. Demeaning within ingredient collapses the association,
    which is the empirical case for the ingredient-level frailty/clustering the
    proposal already specifies. Worth knowing before the pilot, not after.
    """
    o = obs.copy()
    o["log_dur"] = np.log1p(o["duration_days"].clip(lower=0))
    g = o.groupby("ingredient")
    o["dur_w"] = o["log_dur"] - g["log_dur"].transform("mean")
    o["own_w"] = o["footprint_owners"] - g["footprint_owners"].transform("mean")
    within = o[g["ingredient"].transform("size") >= 3]

    raw = o[["duration_days", "footprint_owners"]].corr(method="spearman").iloc[0, 1]
    size = o[["duration_days", "footprint_ndcs"]].corr(method="spearman").iloc[0, 1]
    wcorr = np.corrcoef(within["own_w"], within["dur_w"])[0, 1] if len(within) > 10 else np.nan

    out = pd.DataFrame([
        {"check": "Spearman(duration, effective owners), across episodes",
         "value": round(raw, 3), "n": len(o)},
        {"check": "Spearman(duration, market size in NDCs), across episodes",
         "value": round(size, 3), "n": len(o)},
        {"check": "Pearson(duration, owners), demeaned within ingredient",
         "value": round(wcorr, 3), "n": len(within)},
    ])
    out.to_csv(TAB / "e_confounding_check.csv", index=False)
    LOG.info("  confounding check: raw rho=%.3f, market-size rho=%.3f, "
             "within-ingredient r=%.3f", raw, size, wcorr)


# ==============================================================================
# Figures
# ==============================================================================

def _style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_km_parenteral(uu: pd.DataFrame) -> None:
    """Injectable vs non-injectable time to resolution.

    Two series, both direct-labelled at the curve end, so identity never rests
    on color alone.
    """
    d = uu[(uu["duration_days"] >= 0) & uu["date_onset"].notna() &
           (uu["onset_year"] >= PILOT_START_YEAR)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _style(ax)

    ends = []
    for grp, color, label in [("y", BLUE, "Injectable"), ("n", ORANGE, "Oral / other")]:
        g = d[d["parenteral"] == grp]
        km = KaplanMeierFitter().fit(g["duration_days"], g["event_observed"], label=label)
        km.plot_survival_function(ax=ax, color=color, lw=2, ci_show=True, ci_alpha=0.12,
                                  show_censors=False)
        med = km.median_survival_time_
        ends.append((label, color, len(g), med))

    ax.legend_.remove()
    ax.set_xlim(0, 1095)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Days since shortage onset", color=MUTED, fontsize=10)
    ax.set_ylabel("Share still in shortage", color=MUTED, fontsize=10)
    ax.set_title("Injectable shortages resolve more slowly", color=INK, fontsize=13,
                 fontweight="bold", loc="left", pad=14)

    for i, (label, color, n, med) in enumerate(ends):
        ax.text(1105, 0.62 - 0.14 * i, f"{label}\nn={n:,} · median {med:.0f} d",
                color=color, fontsize=9.5, fontweight="bold", va="center")

    g1 = d[d["parenteral"] == "y"]
    g2 = d[d["parenteral"] == "n"]
    lr = logrank_test(g1["duration_days"], g2["duration_days"],
                      g1["event_observed"], g2["event_observed"])
    ax.text(0, -0.2, f"UUDIS episodes with onset {PILOT_START_YEAR}+. "
                     f"Log-rank p = {lr.p_value:.2g}. Still-active episodes censored at {STUDY_END:%Y-%m-%d}.",
            transform=ax.transAxes, color=MUTED, fontsize=8, va="top")

    fig.subplots_adjust(right=0.76, bottom=0.22)
    fig.savefig(FIG / "km_injectable_vs_oral.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOG.info("  figure: km_injectable_vs_oral.png (log-rank p=%.3g)", lr.p_value)


def fig_footprint_vs_owners(red: pd.DataFrame) -> None:
    """How far the ownership rollup moves the supplier count.

    On current lookup coverage it barely moves at all -- the points sit on the
    diagonal. That is a statement about the 2020 parent lookup's reach into
    injectable firms, not evidence that these markets are unconsolidated, and
    the figure is titled to say so.
    """
    if red.empty:
        return
    agg = (red.groupby("footprint_labelers")
              .agg(effective=("footprint_owners", "mean"), episodes=("drug_text", "size"))
              .reset_index())
    agg = agg[agg["footprint_labelers"] <= 15]

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    _style(ax)
    lim = agg["footprint_labelers"].max() + 1
    ax.plot([0, lim], [0, lim], color=GRID, lw=2, ls="--", zorder=1)
    ax.scatter(agg["footprint_labelers"], agg["effective"],
               s=np.clip(agg["episodes"] * 6, 40, 400), color=BLUE,
               edgecolor="white", linewidth=2, zorder=3)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Labelers marketing the ingredient at onset", color=MUTED, fontsize=10)
    ax.set_ylabel("Independent corporate owners", color=MUTED, fontsize=10)
    ax.set_title("Ownership rollup barely separates from raw label counts",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.text(lim * 0.80, lim * 0.70, "every labeler independent", color=MUTED,
            fontsize=8.5, rotation=45, ha="center", va="center",
            rotation_mode="anchor")
    ax.text(0, -0.2, f"{len(red):,} matched injectable episodes, onset {PILOT_START_YEAR}+. "
                     "Point size = episodes. Ownership evaluated as of the onset date.\n"
                     "Points sit on the diagonal because only a minority of injectable firms "
                     "carry a parent link in the 2020 lookup: a coverage limit, not evidence\n"
                     "these markets are unconsolidated. Counts are over all injectable NDCs of "
                     "the ingredient, not the presentation that went short.",
            transform=ax.transAxes, color=MUTED, fontsize=8, va="top")

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIG / "nominal_vs_footprint_owners.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOG.info("  figure: nominal_vs_footprint_owners.png")


def fig_duration_by_owners(red: pd.DataFrame) -> None:
    """Descriptive: does effective redundancy track resolution speed?"""
    obs = red[red["event_observed"] == 1]
    if len(obs) < 30:
        return
    bins = pd.cut(obs["footprint_owners"], [0, 1, 2, 4, np.inf],
                  labels=["1", "2", "3-4", "5+"])
    g = obs.groupby(bins, observed=True)["duration_days"].agg(["size", "median"])

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    _style(ax)
    bars = ax.bar(g.index.astype(str), g["median"], color=BLUE, width=0.55)
    for b, (n, med) in zip(bars, g.itertuples(index=False)):
        ax.text(b.get_x() + b.get_width() / 2, med + 6, f"{med:.0f} d",
                ha="center", color=INK, fontsize=9.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 6, f"n={n}", ha="center",
                color="white", fontsize=8.5)

    ax.set_xlabel("Independent owners of the ingredient at onset", color=MUTED, fontsize=10)
    ax.set_ylabel("Median days to resolution", color=MUTED, fontsize=10)
    ax.set_title("A bigger ingredient footprint, longer shortages",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.text(0, -0.22, "Resolved injectable episodes only; descriptive and unadjusted. Owners are "
                      "counted over all injectable\nNDCs of the active ingredient, not the "
                      "presentation that went short. Runs opposite to the hypothesis\nbecause "
                      "commodity injectables carry both the largest footprints and the longest "
                      "episodes; demeaned\nwithin ingredient the association falls to r = 0.06 "
                      "(see e_confounding_check.csv).",
            transform=ax.transAxes, color=MUTED, fontsize=8, va="top")

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(FIG / "duration_by_footprint_owners.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOG.info("  figure: duration_by_footprint_owners.png")


# ==============================================================================
# Summary
# ==============================================================================

def write_summary(uu: pd.DataFrame, inj: pd.DataFrame, epi: pd.DataFrame,
                  red: pd.DataFrame) -> None:
    """One markdown page a non-analyst can read, backed by the CSVs above."""
    n_inj = int((uu["parenteral"] == "y").sum())
    inj_res = uu[(uu["parenteral"] == "y") & (uu["event_observed"] == 1)]
    oral_res = uu[(uu["parenteral"] == "n") & (uu["event_observed"] == 1)]

    lines = [
        "# Injectable shortage data: feasibility audit",
        "",
        f"Generated by `20260929_injectable_feasibility_audit.py`.",
        "",
        "| Stage | Question | Verdict |",
        "|---|---|---|",
        f"| A | Are injectable shortage episodes available? | YES. {n_inj:,} parenteral "
        f"episodes, {int(uu['onset_year'].min())}–{int(uu['onset_year'].max())}, pre-flagged by UUDIS |",
        f"| B | Can we see who was marketing the drug at onset? | PARTLY. {len(inj):,} injectable "
        f"NDCs and {inj['labeler'].nunique():,} labelers with marketing start/end dates, but only at "
        "ACTIVE INGREDIENT level, and from a live snapshot with no delisted products |",
        f"| C | Can suppliers be resolved to owners and sites? | PARTLY. Ownership broad, "
        f"site bridge {100 * inj['has_site'].mean():.0f}% of injectable NDCs |",
        f"| D | Can shortage episodes be joined to products? | PARTLY. "
        f"{100 * epi['match_confident'].mean():.0f}% matched confidently on name alone; "
        f"only {100 * (epi['granularity'] != 'ingredient_only').mean():.0f}% of episodes name a "
        "strength or container |",
        f"| E | Is effective redundancy constructible? | YES, at ingredient granularity. "
        f"Demonstrated on {len(red):,} episodes |",
        "",
        "## Headline numbers",
        "",
        f"- Parenteral episodes: **{n_inj:,}** of {len(uu):,} ({100 * n_inj / len(uu):.0f}%).",
        f"- Median time to resolution: **{inj_res['duration_days'].median():.0f} days** for "
        f"injectables vs {oral_res['duration_days'].median():.0f} for everything else.",
        f"- Labelers marketing the ingredient at onset: **{red['footprint_labelers'].mean():.2f}** on average; "
        f"independent owners: **{red['footprint_owners'].mean():.2f}**.",
        f"- **{100 * red['consolidated'].mean():.0f}%** of episodes had fewer independent "
        "owners than labelers (a lower bound; see limit 2).",
        f"- **{100 * (red['concurrent_shortages'] > 0).mean():.0f}%** of episodes began while "
        "another presentation of the same ingredient was already in shortage.",
        "",
        "## The measurement problem that governs everything above",
        "",
        "UUDIS carries a free-text drug name and nothing else: no NDC, no ANDA, no "
        "application number. Every link to the product universe is therefore name-based, "
        "and it resolves at **active ingredient** level. So the counts above are the "
        "ingredient's injectable footprint, not the suppliers of the product that went "
        "short. `0.9% sodium chloride flush syringes` and `14.6% sodium chloride injection` "
        "are different markets and receive near-identical counts.",
        "",
        "Only 10% of episodes name a strength or a container in their text, so for the rest "
        "the source does not support anything finer (`d_market_granularity.csv`). Of the "
        "matches themselves, exact and salt-form matches are sound; a third tier that "
        "guessed from a single first-token candidate is excluded from all counts and "
        "written out for manual review (`d_unverified_matches_REVIEW.csv`) because it "
        "contains real errors, such as Calcium Acetate resolving to calcium chloride.",
        "",
        "`concurrent_shortages` is the exception. It comes from the episode file rather "
        "than the product file, so it is measured at the granularity of the thing that "
        "actually went short.",
        "",
        "## One finding the pilot should know before it starts",
        "",
        "Unadjusted, more suppliers goes with **longer** shortages, the opposite of the "
        "proposed hypothesis. It is market-size confounding: commodity injectables "
        "(sodium chloride, dextrose) carry both the most suppliers and the longest episodes. "
        "Demeaning within ingredient collapses the association to near zero "
        "(`e_confounding_check.csv`). The naive cross-sectional specification would produce a "
        "clean, wrong-signed result, which is the empirical case for the ingredient-level "
        "frailty or clustering the proposal already specifies.",
        "",
        "## Known limits",
        "",
        f"1. The NDC→FEI site bridge covers {100 * inj['has_site'].mean():.0f}% of injectable "
        "NDCs. It was parsed from DailyMed labels for the project's 14-drug universe; the "
        "full SPL archive is in `Data/02 - DailyMed - Labels` and the parser exists, so this "
        "is a re-run rather than new research.",
        "2. The ownership rollup resolves 80% of injectable NDC rows to a standardized firm, "
        "but only a minority of those firms appear as a subsidiary in the 2020 parent lookup, because "
        "it was built for a different sample. Measured consolidation is therefore a floor, and "
        "extending the lookup for injectable firms is a defined task.",
        f"3. UUDIS sole-source is unusable for {int(uu['sole_source'].isna().sum()):,} rows, "
        f"and reason is 'unknown' for {int((uu['reason'] == 'unknown').sum()):,}.",
        f"4. {int((uu['duration_days'] < 0).sum())} episodes have a resolution date before "
        "the notification date and are dropped here.",
        "5. The parent rollup follows one ownership hop. Multi-hop chains are untested.",
        "6. The FDA NDC Directory is a live snapshot, not an archive: no already-delisted "
        "product is retained, so a supplier who exited before the extract date is invisible "
        "and every historical count is a floor. Archived per-year NDC snapshots are the fix.",
        "",
        "## Outputs",
        "",
        f"- Tables: `{TAB.relative_to(C.OUT_ROOT)}/`",
        f"- Figures: `{FIG.relative_to(C.OUT_ROOT)}/`",
        f"- Episode-level demo: `data/injectable_redundancy_demo.parquet`",
    ]
    out = TAB / "SUMMARY.md"
    out.write_text("\n".join(lines))
    LOG.info("Wrote %s", out)


def main() -> int:
    LOG.info("=" * 78)
    LOG.info("Injectable feasibility audit")
    LOG.info("=" * 78)

    uu = load_uudis()
    stage_a_episode_inventory(uu)

    inj = stage_b_product_universe(load_ndc_injectables())

    std, links = load_ownership()
    inj = stage_c_independence_layers(inj, std, links, load_ndc_fei())

    epi = stage_d_linkage(uu, inj)
    red = stage_e_redundancy(epi, inj, links)

    LOG.info("Figures")
    fig_km_parenteral(uu)
    fig_footprint_vs_owners(red)
    fig_duration_by_owners(red)

    write_summary(uu, inj, epi, red)
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
