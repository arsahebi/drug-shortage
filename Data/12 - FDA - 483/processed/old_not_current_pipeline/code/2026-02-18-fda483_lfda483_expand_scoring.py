# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# Imports & Config
# =========================
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer


# ---- Set this once ----
DATA_DIR = Path(
    "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive"
    "/North Carolina State University/Project - Drug Shortage/Data/12 - FDA - 483"
)

JSON_DIR = DATA_DIR / "processed" / "json"
OTHER_DIR = DATA_DIR / "raw"
OUT_DIR  = DATA_DIR / "processed" / "scoring"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_DIR:", DATA_DIR)
print("JSON_DIR exists:", JSON_DIR.exists())
print("OUT_DIR:", OUT_DIR)


#%% =========================
# Helper functions
# =========================
def normalize_outcome(x: str) -> str | None:
    """
    Accepts:
      - "OAI", "VAI", "NAI"
      - "Official Action Indicated (OAI)"
      - "Voluntary Action Indicated (VAI)"
      - "No Action Indicated (NAI)"
    Returns: "OAI"/"VAI"/"NAI" or None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()

    m = re.search(r"\b(OAI|VAI|NAI)\b", s)
    if m:
        return m.group(1)

    if "OFFICIAL ACTION INDICATED" in s:
        return "OAI"
    if "VOLUNTARY ACTION INDICATED" in s:
        return "VAI"
    if "NO ACTION INDICATED" in s:
        return "NAI"

    return None


def fei_from_filename(p: Path) -> str | None:
    """Extract FEI prefix from filenames like '3006370533 - Inspection Data.xlsx'."""
    m = re.match(r"^\s*(\d{6,12})\s*-\s*", p.name)
    return m.group(1) if m else None


def load_all_inspection_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("* - Inspection Data.xlsx"))
    if not files:
        raise FileNotFoundError(f"No '* - Inspection Data.xlsx' found in: {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_excel(f)

        # normalize FEI
        if "FEI Number" not in df.columns:
            raise ValueError(f"Missing 'FEI Number' in {f}")
        df["FEI Number"] = df["FEI Number"].astype(str).str.strip()

        # parse date
        if "Inspection End Date" in df.columns:
            df["Inspection End Date"] = pd.to_datetime(df["Inspection End Date"], errors="coerce")

        # normalize classification
        if "Classification" in df.columns:
            df["outcome"] = df["Classification"].apply(normalize_outcome)
        else:
            df["outcome"] = None

        df["_source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_all_citation_data(data_dir: Path) -> pd.DataFrame:
    """Concatenate all citation files that exist. Missing facility files are OK."""
    files = sorted(data_dir.glob("* - Citation Data.xlsx"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_excel(f)
        if "FEI Number" in df.columns:
            df["FEI Number"] = df["FEI Number"].astype(str).str.strip()
        if "Inspection End Date" in df.columns:
            df["Inspection End Date"] = pd.to_datetime(df["Inspection End Date"], errors="coerce")
        df["_source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_all_483_metadata(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("* - 483 Data.xlsx"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_excel(f)
        if "FEI Number" in df.columns:
            df["FEI Number"] = df["FEI Number"].astype(str).str.strip()
        if "Record Date" in df.columns:
            df["Record Date"] = pd.to_datetime(df["Record Date"], errors="coerce")
        if "Publish Date" in df.columns:
            df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")
        df["_source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)



def load_json_observations(json_dir: Path) -> pd.DataFrame:
    rows = []

    # e.g. "3002809586 - 02232018 - Sun Pharmaceutical.json"
    fname_pat = re.compile(
        r"^\s*(?P<fei>\d+)\s*-\s*(?P<doi>\d{8})\s*-\s*(?P<firm>.+?)\s*$"
    )

    for p in sorted(json_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8", errors="ignore"))

        header = d.get("header") or {}
        fei = header.get("fei_number")
        firm = header.get("firm_name")
        doi = header.get("date_of_inspection")  # might be a range string
        obs = d.get("observations") or []

        # ---- fallback: parse from filename if missing ----
        if (fei is None or firm is None or doi is None):
            stem = p.stem  # filename without ".json"
            m = fname_pat.match(stem)
            if m:
                if fei is None:
                    fei = m.group("fei")
                if doi is None:
                    doi = m.group("doi")
                if firm is None:
                    firm = m.group("firm").strip()

        for o in obs:
            rows.append(
                {
                    "json_file": p.name,
                    "fei_number": str(fei).strip() if fei is not None else None,
                    "firm_name": firm,
                    "date_of_inspection_raw": doi,
                    "obs_num": o.get("observation_number"),
                    "obs_text": o.get("text", "") or "",
                }
            )

    return pd.DataFrame(rows)



# ---- Citation domain mapping (expand later) ----
DOMAIN_PATTERNS = {
    "qc_unit": [r"\bquality control unit\b", r"\bapprove or reject\b", r"\b211\.22\b"],
    "deviations": [r"\bdeviation\b", r"\b211\.100\b", r"\bnot followed\b"],
    "lab_controls": [r"\b211\.160\b", r"\bspecification\b", r"\blaborator"],
    "data_integrity": [r"\baudit trail\b", r"\bunauthorized\b", r"\bmetadata\b", r"\bcomputer system\b"],
    "sterility_micro": [r"\bsteril", r"\baseptic\b", r"\bmicrobiolog", r"\bendotoxin\b"],
}


def citation_domain_flags(short_desc: str, long_desc: str) -> dict:
    t = f"{short_desc or ''} {long_desc or ''}".lower()
    out = {}
    for dom, pats in DOMAIN_PATTERNS.items():
        out[f"cit_dom_{dom}"] = int(any(re.search(p, t) for p in pats))
    return out


def build_citation_features(df_cit: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per Inspection ID with:
      - n_citations
      - n_unique_cfr
      - domain counts from citation text
    """
    if df_cit is None or df_cit.empty:
        return pd.DataFrame(columns=["Inspection ID"])

    req = {"Inspection ID", "Act/CFR Number", "Short Description", "Long Description"}
    missing = req - set(df_cit.columns)
    if missing:
        raise ValueError(f"Citation data missing columns: {missing}")

    dom = [
        citation_domain_flags(a, b)
        for a, b in zip(df_cit["Short Description"], df_cit["Long Description"])
    ]
    df_dom = pd.DataFrame(dom)
    tmp = pd.concat([df_cit.reset_index(drop=True), df_dom], axis=1)

    agg = tmp.groupby("Inspection ID", dropna=False).agg(
        n_citations=("Act/CFR Number", "size"),
        n_unique_cfr=("Act/CFR Number", pd.Series.nunique),
        **{c: (c, "sum") for c in df_dom.columns},
    ).reset_index()

    return agg


#%% =========================
# Load: JSON observations
# =========================
df_obs = load_json_observations(JSON_DIR)
print("df_obs shape:", df_obs.shape)
df_obs.head()


#%% =========================
# Load: Inspection / Citation / 483 metadata
# =========================
df_insp = load_all_inspection_data(OTHER_DIR)
print("df_insp shape:", df_insp.shape)
print("outcome value counts:\n", df_insp["outcome"].value_counts(dropna=False))

df_insp = df_insp[df_insp["outcome"].isin(["NAI", "VAI", "OAI"])].copy()

df_cit = load_all_citation_data(OTHER_DIR)
print("df_cit shape:", df_cit.shape)

df_483meta = load_all_483_metadata(OTHER_DIR)
print("df_483meta shape:", df_483meta.shape)


#%% =========================
# Build citation features (inspection-level)
# =========================
df_cit_feat = build_citation_features(df_cit)
print("df_cit_feat shape:", df_cit_feat.shape)
df_cit_feat.head()


#%% =========================
# Join obs -> inspection labels
# =========================
df_insp2 = df_insp.rename(columns={"FEI Number": "fei_number"})
df_insp2["fei_number"] = df_insp2["fei_number"].astype(str).str.strip()

df = df_obs.merge(df_insp2, on="fei_number", how="left", suffixes=("", "_insp"))
df = df[df["outcome"].isin(["NAI", "VAI", "OAI"])].copy()

print("df joined shape:", df.shape)
df[["fei_number", "firm_name", "Inspection ID", "Inspection End Date", "outcome"]].head()


#%% =========================
# Aggregate to inspection-level text
# =========================
if "Inspection ID" in df.columns:
    group_cols = ["Inspection ID", "fei_number", "Inspection End Date", "outcome"]
else:
    group_cols = ["fei_number", "Inspection End Date", "outcome"]

full_text = df.groupby(group_cols)["obs_text"].apply(lambda s: "\n\n".join(s.astype(str))).reset_index()
full_text["n_obs"] = df.groupby(group_cols)["obs_num"].size().values

print("full_text shape:", full_text.shape)
full_text.head()


#%% =========================
# Merge in citation features (if available)
# =========================
if "Inspection ID" in full_text.columns and not df_cit_feat.empty:
    full_text = full_text.merge(df_cit_feat, on="Inspection ID", how="left")

# Fill citation feature NAs with 0
for c in full_text.columns:
    if c in ["n_citations", "n_unique_cfr"] or c.startswith("cit_dom_"):
        full_text[c] = full_text[c].fillna(0)

# Track whether citation file existed for that FEI at all (optional)
# (Helpful for interpreting "0" vs "missing online")
cit_feis = set(df_cit["FEI Number"].astype(str).str.strip()) if not df_cit.empty and "FEI Number" in df_cit.columns else set()
full_text["citation_file_available_for_fei"] = full_text["fei_number"].astype(str).isin(cit_feis).astype(int)

full_text[[*group_cols, "n_obs", "n_citations" if "n_citations" in full_text.columns else "n_obs"]].head()


#%% =========================
# Merge in "has published 483" flag using 483 metadata (optional)
# =========================
full_text["has_483_published"] = 0

if not df_483meta.empty and {"FEI Number", "Record Date"}.issubset(df_483meta.columns) and "Inspection End Date" in full_text.columns:
    df_483meta2 = df_483meta.rename(columns={"FEI Number": "fei_number"})
    df_483meta2["fei_number"] = df_483meta2["fei_number"].astype(str).str.strip()

    tmp = full_text.merge(df_483meta2[["fei_number", "Record Date"]], on="fei_number", how="left")
    tmp["day_diff"] = (tmp["Inspection End Date"] - tmp["Record Date"]).abs().dt.days

    mark = tmp.groupby(group_cols)["day_diff"].min().reset_index()
    mark["has_483_published"] = (mark["day_diff"].fillna(99999) <= 60).astype(int)

    full_text = full_text.drop(columns=["has_483_published"], errors="ignore").merge(
        mark[group_cols + ["has_483_published"]], on=group_cols, how="left"
    )
    full_text["has_483_published"] = full_text["has_483_published"].fillna(0).astype(int)

full_text[[*group_cols, "has_483_published"]].head()


#%% =========================
# Build ML features (rule + citations + embeddings)
# =========================
rule_cols = ["n_obs", "has_483_published", "citation_file_available_for_fei"]

cit_cols = [c for c in full_text.columns if c in ["n_citations", "n_unique_cfr"] or c.startswith("cit_dom_")]
use_cols = rule_cols + cit_cols

X_rule = full_text[use_cols].to_numpy(dtype=float)

print("rule feature columns:", use_cols)
print("X_rule shape:", X_rule.shape)


#%% =========================
# Embeddings from inspection text
# =========================
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_emb = emb_model.encode(
    full_text["obs_text"].fillna("").tolist(),
    show_progress_bar=True,
    normalize_embeddings=True,
)
X = np.hstack([X_rule, X_emb])

y_oai = (full_text["outcome"] == "OAI").astype(int).to_numpy()
groups = full_text["fei_number"].astype(str).to_numpy()

print("X shape:", X.shape, "y_oai mean:", y_oai.mean())


#%% =========================
# Train model: CV p(OAI)
# =========================
probs = np.zeros(len(full_text))
clf = LogisticRegression(max_iter=3000)

unique_groups = np.unique(groups)
if len(unique_groups) <= 1:
    clf.fit(X, y_oai)
    probs = clf.predict_proba(X)[:, 1]
    auc = np.nan
else:
    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    for tr, te in gkf.split(X, y_oai, groups=groups):
        clf.fit(X[tr], y_oai[tr])
        probs[te] = clf.predict_proba(X[te])[:, 1]

    auc = roc_auc_score(y_oai, probs) if len(np.unique(y_oai)) > 1 else np.nan

full_text["p_oai"] = probs
print("CV AUC (OAI vs not):", auc)

full_text[[*group_cols, "p_oai"]].head()


#%% =========================
# Overall severity score (0-100)
# =========================
# Normalize citations if present; otherwise treat as 0 contribution.
if "n_citations" in full_text.columns:
    denom = full_text["n_citations"].max()
    cit_norm = full_text["n_citations"] / (denom if denom and denom > 0 else 1)
else:
    cit_norm = 0.0

full_text["overall_severity_score"] = np.clip(
    100 * (0.15 * cit_norm + 0.85 * full_text["p_oai"]),
    0, 100
)

full_text[[*group_cols, "n_obs", "n_citations" if "n_citations" in full_text.columns else "n_obs",
          "p_oai", "overall_severity_score"]].head()


#%% =========================
# Save output
# =========================
out_path = OUT_DIR / "inspection_scores_v2.csv"
full_text.to_csv(out_path, index=False)
print("Saved:", out_path)

# %%
