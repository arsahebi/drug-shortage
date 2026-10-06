"""
Shared utilities: logging, string normalization, drug matching.
"""

from __future__ import annotations
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---- Logging -----------------------------------------------------------------

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Standardized logger. If log_file is given, also write to it."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---- Drug-name normalization -------------------------------------------------

_STRIP_RX = re.compile(r"[^a-z0-9 ]+")
_WS_RX    = re.compile(r"\s+")

# Salt/route/form noise tokens we strip when comparing names
_NOISE = {
    "hydrochloride", "hcl", "sulfate", "sulphate", "phosphate", "tartrate",
    "succinate", "fumarate", "maleate", "mesylate", "besylate", "citrate",
    "sodium", "potassium", "calcium", "magnesium",
    "injection", "injectable", "tablet", "tablets", "capsule", "capsules",
    "extended", "release", "er", "xr", "sr", "ir",
    "oral", "iv", "intravenous", "intramuscular",
    "solution", "suspension", "syrup",
    "and", "the", "of", "for", "with",
    "mg", "ml", "mcg", "g", "kg",
}

# common anions to preserve when paired with cations like 'sodium'/'potassium'
_SALT_ANIONS = {
    "chloride", "bicarbonate", "phosphate", "acetate", "citrate",
    "tartrate", "sulfate", "sulphate", "nitrate", "carbonate",
    "hyaluronate",
}

def normalize_drug_name(name: str) -> str:
    """Lowercase, strip salts/forms/numbers/units, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _STRIP_RX.sub(" ", s)
    s = _WS_RX.sub(" ", s).strip()
    raw_toks = [t for t in s.split() if not t.isdigit()]
    toks = [t for t in raw_toks if t not in _NOISE]

    # Preserve common salt pairs like 'sodium chloride' — if 'sodium' (or
    # 'potassium', etc.) was present in the raw tokens together with a known
    # anion, re-insert the cation if it was removed by the noise list so the
    # compound identity is preserved.
    for cation in ("sodium", "potassium", "calcium", "magnesium"):
        if cation in raw_toks and any(a in raw_toks for a in _SALT_ANIONS) and cation not in toks:
            # insert cation before the first anion occurrence in toks if present
            inserted = False
            for i, t in enumerate(toks):
                if t in _SALT_ANIONS:
                    toks.insert(i, cation)
                    inserted = True
                    break
            if not inserted:
                toks.insert(0, cation)

    return " ".join(toks)


# ---- Valisure-canonical matching --------------------------------------------

_MATCH_DROP_TOKENS = {
    "and", "the", "of", "for", "with",
    "hydrochloride", "hcl", "sodium", "usp",
    "injection", "injectable", "tablet", "tablets", "capsule", "capsules",
    "extended", "release", "er", "xr", "sr", "ir",
    "oral", "iv", "intravenous", "intramuscular",
    "solution", "suspension", "syrup",
    "mg", "ml", "mcg", "g", "kg", "meq",
}


def normalize_match_text(name: str) -> str:
    """Normalize text for matching to exact Valisure API names.

    This is deliberately less aggressive than `normalize_drug_name`: it keeps
    meaningful two-part API names such as potassium chloride, magnesium sulfate,
    and calcium gluconate intact.
    """
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _STRIP_RX.sub(" ", s)
    s = _WS_RX.sub(" ", s).strip()
    toks = [t for t in s.split() if not t.isdigit() and t not in _MATCH_DROP_TOKENS]
    return " ".join(toks)


def load_valisure_api_names(valisure_csv: Path) -> list[str]:
    """Return exact API names as written in the processed Valisure file."""
    df = pd.read_csv(valisure_csv, usecols=lambda c: c.strip().lower() == "api")
    api_col = next(c for c in df.columns if c.strip().lower() == "api")
    return df[api_col].dropna().astype(str).str.strip().drop_duplicates().tolist()


class ValisureDrugMatcher:
    """Match source text to exact Valisure API names."""

    def __init__(self, api_names: Iterable[str]):
        self.api_names = [a.strip() for a in api_names if isinstance(a, str) and a.strip()]
        self._patterns: list[tuple[str, re.Pattern]] = []
        for api in sorted(self.api_names, key=lambda x: (len(normalize_match_text(x).split()), len(x)), reverse=True):
            toks = normalize_match_text(api).split()
            if not toks:
                continue
            # Allow a few descriptive tokens between API words, e.g.
            # "ampicillin sodium and sulbactam sodium" -> Ampicillin; Sulbactam.
            gap = r"(?:\s+\w+){0,3}\s+"
            pat = r"(?<!\w)" + gap.join(re.escape(t) for t in toks) + r"(?!\w)"
            self._patterns.append((api, re.compile(pat)))

    def match(self, text: str) -> str | None:
        haystack = normalize_match_text(text)
        for api, pattern in self._patterns:
            if pattern.search(haystack):
                return api
        return None


def contains_any(haystack: str, needles: Iterable[str]) -> str | None:
    """Return the first needle found as a whole word in haystack (after normalize), else None."""
    h = " " + normalize_drug_name(haystack) + " "
    for n in needles:
        if f" {n} " in h:
            return n
    return None


# ---- Table IO ----------------------------------------------------------------

def write_table(df: pd.DataFrame, parquet_path: Path, logger: logging.Logger | None = None) -> None:
    """Write CSV for inspection and parquet when a parquet engine is installed."""
    csv_path = parquet_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except ImportError as exc:
        if logger:
            logger.warning("Wrote %s but skipped parquet %s: %s", csv_path.name, parquet_path.name, exc)
    else:
        if logger:
            logger.info("Wrote %s and %s", csv_path.name, parquet_path.name)


def read_table(parquet_path: Path) -> pd.DataFrame:
    """Read parquet, falling back to CSV when parquet support is unavailable or CSV is newer."""
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists() and (
        not parquet_path.exists() or csv_path.stat().st_mtime >= parquet_path.stat().st_mtime
    ):
        return pd.read_csv(csv_path)
    try:
        return pd.read_parquet(parquet_path)
    except ImportError:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise


# ---- Year helpers ------------------------------------------------------------

def to_year(s: pd.Series) -> pd.Series:
    """Parse a Series of dates/strings/numbers into integer year (NaN-safe)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64")
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.year.astype("Int64")


# ---- SDUD volume helper ------------------------------------------------------

def load_sdud_fei_volume(
    fei_list: list[int],
    drug_list: list[str],
    year_range: tuple[int, int],
) -> pd.DataFrame:
    """Estimate Medicaid (SDUD) unit volume share per FEI × drug × year.

    Joins SDUD monthly data (NDC11 level) to the NDC→FEI crosswalk, aggregates
    to FEI × drug_norm × year, and computes each FEI's share of total units for
    that drug × year across all FEIs in fei_list.

    Parameters
    ----------
    fei_list   : FEI numbers to include (filters the NDC→FEI mapping)
    drug_list  : Normalized drug names (drug_norm) to include
    year_range : (start_year, end_year) inclusive

    Returns
    -------
    DataFrame with columns: fei, drug_norm, year, sdud_units, sdud_volume_share
    sdud_volume_share = FEI's units / total units for that drug × year.
    FEIs with no matching NDCs or no SDUD volume will have NaN share.
    """
    from config import SDUD_MONTHLY_CSV, NDC_FEI_MAP_CSV

    # Load NDC→FEI bridge
    bridge = pd.read_csv(NDC_FEI_MAP_CSV, low_memory=False)
    bridge.columns = [c.strip() for c in bridge.columns]
    ndc_col = next((c for c in bridge.columns if "ndc" in c.lower() and "manufacture" in c.lower()),
                   next((c for c in bridge.columns if "ndc" in c.lower()), None))
    fei_col = next((c for c in bridge.columns if "fei" in c.lower()), None)

    if ndc_col is None or fei_col is None:
        raise ValueError(f"Cannot find NDC/FEI columns in {NDC_FEI_MAP_CSV}")

    bridge = bridge[[ndc_col, fei_col]].dropna().rename(columns={ndc_col: "ndc", fei_col: "fei"})
    bridge["fei"] = pd.to_numeric(bridge["fei"], errors="coerce").astype("Int64")
    bridge["ndc"] = bridge["ndc"].astype(str).str.strip()
    bridge = bridge[bridge["fei"].isin(fei_list)].drop_duplicates()

    if bridge.empty:
        return pd.DataFrame(columns=["fei", "drug_norm", "year", "sdud_units", "sdud_volume_share"])

    # Build NDC11 lookup: strip dashes/dots, zero-pad to 11 digits
    def _ndc_to_11(s: str) -> str:
        digits = re.sub(r"[^0-9]", "", s)
        return digits.zfill(11)[-11:]

    bridge["ndc11"] = bridge["ndc"].map(_ndc_to_11)

    # Load SDUD monthly
    sdud = pd.read_csv(SDUD_MONTHLY_CSV, low_memory=False)
    sdud.columns = [c.strip() for c in sdud.columns]
    sdud["ndc11"] = sdud["ndc11"].astype(str).str.strip().map(_ndc_to_11)
    sdud["year"]  = pd.to_numeric(sdud["year"], errors="coerce").astype("Int64")
    sdud = sdud[
        (sdud["year"] >= year_range[0]) & (sdud["year"] <= year_range[1])
    ]

    # Join SDUD → bridge to get FEI
    merged = sdud.merge(bridge[["ndc11", "fei"]], on="ndc11", how="inner")

    # Normalize drug names using Valisure matcher if drug_list provided
    # (SDUD has no drug name column — the FEI bridge tells us which drug a FEI makes)
    # So we join drug_norm through the FEI→drug mapping from Valisure
    from config import VALISURE_FEI, VALISURE_CSV
    try:
        fei_map = pd.read_excel(VALISURE_FEI, sheet_name="API Only_FEI Mapping")
        fei_map.columns = [c.strip() for c in fei_map.columns]
        fei_col_v = next((c for c in fei_map.columns if "fei" in c.lower()), None)
        api_col_v = next((c for c in fei_map.columns if c.lower() == "api"), None)
        if fei_col_v and api_col_v:
            fm = fei_map[[fei_col_v, api_col_v]].dropna().rename(
                columns={fei_col_v: "fei", api_col_v: "api"}
            )
            fm["fei"] = pd.to_numeric(fm["fei"], errors="coerce").astype("Int64")
            matcher = ValisureDrugMatcher(load_valisure_api_names(VALISURE_CSV))
            fm["drug_norm"] = fm["api"].astype(str).map(matcher.match)
            fm = fm.dropna(subset=["drug_norm"])
            merged = merged.merge(fm[["fei", "drug_norm"]], on="fei", how="inner")
            if drug_list:
                merged = merged[merged["drug_norm"].isin(drug_list)]
    except Exception:
        merged["drug_norm"] = "unknown"

    # Aggregate to FEI × drug_norm × year
    agg = merged.groupby(["fei", "drug_norm", "year"], as_index=False).agg(
        sdud_units=("units_reimbursed", "sum")
    )

    # Compute volume share within each drug × year
    total = agg.groupby(["drug_norm", "year"])["sdud_units"].transform("sum")
    agg["sdud_volume_share"] = agg["sdud_units"] / total.replace(0, float("nan"))

    return agg[["fei", "drug_norm", "year", "sdud_units", "sdud_volume_share"]]
