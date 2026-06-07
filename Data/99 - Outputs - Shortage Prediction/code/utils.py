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
