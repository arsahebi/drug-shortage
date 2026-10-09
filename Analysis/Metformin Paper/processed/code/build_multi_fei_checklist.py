# %%
"""
Checklist of multi-facility NDCs for manual DailyMed verification.

For the manual and ProPublica NDC-FEI maps, lists every Valisure-tested NDC
linked to more than one facility, one row per (NDC, FEI), with what each
source says the facility does, so each link can be checked by hand on
DailyMed. Motivation: the manuscript says these extra establishments are
finished-dosage form (FDF) manufacturers, not API makers; ProPublica's
address-matched links may also pick up "Manufactured for" labelers.

Evidence columns per facility:
  - FDA GDUFA self-identification lists (Data/16 - FDA - FEI, FY2023-FY2026):
    the facility's registered business operations (FDF MANUFACTURE,
    API MANUFACTURE, PACK, ...), name, and address.
  - DailyMed establishment operations for this NDC from the July 2025 label
    parse (Data/17 - NDC-FEI Linkage/processed/all_daily_med.csv).
  - ProPublica linkage method and API-only flag (ProPublica sheet).
A rule-based "flag" column marks rows worth a closer look; the
"checked"/"verdict"/"notes" columns are left blank for the reviewer.

Output: outputs/20261009_multi_fei_ndc_checklist.xlsx
"""

import re
from pathlib import Path

import pandas as pd
from lxml import etree

BASE = Path("/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage")
PROC = BASE / "Analysis/Metformin Paper/processed"
GDUFA_DIR = BASE / "Data/16 - FDA - FEI"
DAILYMED = BASE / "Data/17 - NDC-FEI Linkage/processed/all_daily_med.csv"
PROPUBLICA = BASE / "Data/19 - ProPublica/raw/ndc_fei.csv"
INSPECTIONS = BASE / "Data/14 - FDA - Inspection/raw/Inspections Details.xlsx"
OUT = PROC / "outputs/20261009_multi_fei_ndc_checklist.xlsx"
GDUFA_YEARS = [2023, 2024, 2025, 2026]


def n11(x):
    p = str(x).strip().split("-")
    return f"{p[0].zfill(5)}-{p[1].zfill(4)}-{p[2].zfill(2)}"


def n9(x):
    p = str(x).strip().split("-")
    return f"{p[0].zfill(5)}-{p[1].zfill(4)}"


def clean_fei(x):
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


# ── GDUFA self-ID lists (mixed formats: xlsx, SpreadsheetML 2003 XML) ────────
def _read_spreadsheetml(path):
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    root = etree.parse(str(path), etree.XMLParser(recover=True, huge_tree=True)).getroot()
    rows = []
    for r in root.iterfind(".//ss:Worksheet[1]//ss:Row", ns):
        vals, col = [], 0
        for c in r.iterfind("ss:Cell", ns):
            idx = c.get(f"{{{ns['ss']}}}Index")
            if idx:
                while col < int(idx) - 1:
                    vals.append(None); col += 1
            d = c.find("ss:Data", ns)
            vals.append("".join(d.itertext()) if d is not None else None); col += 1
        rows.append(vals)
    hdr_i = next(i for i, r in enumerate(rows) if r and any(str(v).strip().upper() == "FEI NUMBER" for v in r if v))
    hdr = [str(v).strip().upper() if v else f"c{i}" for i, v in enumerate(rows[hdr_i])]
    return pd.DataFrame([r + [None] * (len(hdr) - len(r)) for r in rows[hdr_i + 1:]], columns=hdr)


def load_gdufa():
    frames = []
    for yr in GDUFA_YEARS:
        f = next(GDUFA_DIR.glob(f"FY-{yr}--*"))
        try:
            d = pd.read_excel(f, dtype=str)
        except Exception:
            d = _read_spreadsheetml(f)
        d.columns = [str(c).strip().upper() for c in d.columns]
        d = d.rename(columns={"FEI": "FEI NUMBER"})
        d["FY"] = yr
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    g = g[g["FEI NUMBER"].notna()].copy()
    g["FEI"] = g["FEI NUMBER"].map(clean_fei)
    ops = (g.groupby(["FEI", "BUSINESS OPERATIONS"])["FY"]
             .apply(lambda s: f"{min(s)}-{max(s)}" if min(s) != max(s) else str(min(s)))
             .reset_index())
    ops = ops.groupby("FEI").apply(
        lambda x: "; ".join(f"{o} (FY{y})" for o, y in zip(x["BUSINESS OPERATIONS"], x["FY"]))
    ).rename("gdufa_ops")
    latest = g.sort_values("FY").drop_duplicates("FEI", keep="last").set_index("FEI")
    out = pd.DataFrame({"gdufa_ops": ops})
    out["gdufa_name"] = latest["FACILITY NAME"]
    out["gdufa_address"] = latest["BUSINESS ADDRESS"]
    out["has_fdf"] = out["gdufa_ops"].str.contains("FDF MANUFACTURE", na=False)
    out["has_api"] = out["gdufa_ops"].str.contains("API MANUFACTURE", na=False)
    return out


# ── other facility evidence ──────────────────────────────────────────────────
def load_dailymed_ops(ndc9s, feis):
    keep = []
    for ch in pd.read_csv(DAILYMED, dtype=str, usecols=["ndc", "name", "opr_type", "FEI"],
                          chunksize=500_000):
        ch = ch[ch["FEI"].isin(feis)]
        if len(ch):
            keep.append(ch)
    d = pd.concat(keep)
    d["NDC9"] = d["ndc"].map(n9)
    on_ndc = (d[d["NDC9"].isin(ndc9s)].groupby(["NDC9", "FEI"])["opr_type"]
                .apply(lambda s: ", ".join(sorted(set(s)))).rename("dailymed_ops_this_ndc"))
    ever_api = d[d["opr_type"] == "api manufacture"]["FEI"].unique()
    name = d.dropna(subset=["name"]).drop_duplicates("FEI").set_index("FEI")["name"]
    return on_ndc, set(ever_api), name


def load_inspection_locations(feis):
    x = pd.read_excel(INSPECTIONS, dtype=str, usecols=["FEI Number", "Legal Name", "City", "State", "Country/Area"])
    x["FEI"] = x["FEI Number"].map(clean_fei)
    x = x[x["FEI"].isin(feis)].drop_duplicates("FEI").set_index("FEI")
    loc = x["City"].fillna("") + ", " + x["State"].fillna("").replace("-", "") + ", " + x["Country/Area"].fillna("")
    return x["Legal Name"], loc.str.replace(", ,", ",").str.strip(", ")


def multi_fei_rows(map_label):
    m = pd.read_csv(PROC / f"step1_ndc_fei_map_{map_label}.csv", dtype=str).dropna(subset=["FEI"])
    m["NDC11"] = m["NDC11"].map(n11)
    m["FEI"] = m["FEI"].map(clean_fei)
    k = m.groupby("NDC11")["FEI"].nunique()
    m = m[m["NDC11"].isin(k[k > 1].index)].drop_duplicates(["NDC11", "FEI"]).copy()
    m["n_facilities"] = m["NDC11"].map(k)
    m["NDC9"] = m["NDC11"].str[:10]
    return m


def build():
    maps = {lab: multi_fei_rows(lab) for lab in ["manual", "propublica"]}
    feis = set().union(*[set(m["FEI"]) for m in maps.values()])
    ndc9s = set().union(*[set(m["NDC9"]) for m in maps.values()])

    gd = load_gdufa()
    dm_ops, dm_ever_api, dm_name = load_dailymed_ops(ndc9s, feis)
    insp_name, insp_loc = load_inspection_locations(feis)
    pp = pd.read_csv(PROPUBLICA, dtype=str)
    pp["NDC9"] = pp["ndc"].map(n9)
    pp_link = pp.drop_duplicates(["NDC9", "fei"]).set_index(["NDC9", "fei"])
    pp_fei = pp.drop_duplicates("fei").set_index("fei")

    pairs = {lab: set(zip(m["NDC11"], m["FEI"])) for lab, m in maps.items()}
    all_pairs = {lab: set(zip(*[pd.read_csv(PROC / f"step1_ndc_fei_map_{lab}.csv", dtype=str)
                                .dropna(subset=["FEI"])[c].map(f) for c, f in
                                [("NDC11", n11), ("FEI", clean_fei)]]))
                 for lab in ["manual", "propublica", "rulebased"]}

    sheets = {}
    for lab, m in maps.items():
        rows = []
        for _, r in m.sort_values(["NDC11", "FEI"]).iterrows():
            f, key9 = r["FEI"], (r["NDC9"], r["FEI"])
            g = gd.loc[f] if f in gd.index else None
            name = (g["gdufa_name"] if g is not None else None) or insp_name.get(f) or dm_name.get(f) \
                or (pp_fei.loc[f, "registrant"] if f in pp_fei.index else None)
            loc = (g["gdufa_address"] if g is not None else None) or insp_loc.get(f) \
                or (pp_fei.loc[f, "country"] if f in pp_fei.index else None)
            ops_here = dm_ops.get(key9, "")
            flags = []
            if g is None:
                flags.append("not in GDUFA FY2023-26 lists")
            elif not g["has_fdf"]:
                flags.append("GDUFA: no FDF MANUFACTURE")
            if g is not None and g["has_api"] and not g["has_fdf"]:
                flags.append("GDUFA: API only")
            if f in dm_ever_api:
                flags.append("DailyMed lists it as API manufacture somewhere")
            if ops_here and "manufacture" not in ops_here.split(", "):
                flags.append("on this label, not listed as manufacture")
            if not ops_here:
                flags.append("not on this NDC's July 2025 label parse")
            row = {
                "NDC": r["NDC"], "NDC11": r["NDC11"],
                "DailyMed search": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={r['NDC9']}",
                "Facilities linked to this NDC": int(r["n_facilities"]),
                "FEI": f, "Facility name": name, "Address / location": loc,
                "GDUFA business operations": g["gdufa_ops"] if g is not None else "",
                "DailyMed operations for this NDC (Jul 2025)": ops_here,
            }
            if lab == "propublica":
                pl = pp_link.loc[key9] if key9 in pp_link.index else None
                row["ProPublica linkage method"] = pl["linkage_method"] if pl is not None else ""
                row["ProPublica API-only flag"] = (pl["api_mfr"] if pl is not None and pd.notna(pl["api_mfr"]) else "")
                row["Same link in manual map"] = "yes" if (r["NDC11"], f) in all_pairs["manual"] else "no"
            else:
                row["Manual source note"] = r.get("fei_count", "")
                row["Same link in ProPublica"] = "yes" if (r["NDC11"], f) in all_pairs["propublica"] else "no"
            row["Same link in rule-based"] = "yes" if (r["NDC11"], f) in all_pairs["rulebased"] else "no"
            row["Auto flag"] = "; ".join(flags)
            row["Checked (Y/N)"] = ""
            row["Verdict (FDF / API / labeler / other)"] = ""
            row["Notes"] = ""
            rows.append(row)
        sheets[lab] = pd.DataFrame(rows)

    readme = pd.DataFrame({"Item": [
        "Purpose", "Rows", "GDUFA business operations", "DailyMed operations",
        "ProPublica linkage method", "Auto flag", "Your columns"], "Description": [
        "Every Valisure-tested NDC linked to more than one facility, for manual DailyMed verification "
        "that each linked facility makes the finished dosage form (not API, not a 'Manufactured for' labeler).",
        f"Manual: {sheets['manual']['NDC11'].nunique()} NDCs, {len(sheets['manual'])} NDC-facility rows. "
        f"ProPublica: {sheets['propublica']['NDC11'].nunique()} NDCs, {len(sheets['propublica'])} rows. "
        "Canada/Bangladesh facilities are included here (they are dropped later in the analysis).",
        "Operations the facility self-identified to FDA under GDUFA, FY2023-FY2026, with fiscal years. "
        "FDF MANUFACTURE = finished dosage form maker; API MANUFACTURE = active ingredient maker.",
        "Establishment operations listed for this NDC on its DailyMed label, from the July 2025 label parse. "
        "Blank = facility not on that label at that time (label may have changed, or link came from elsewhere).",
        "SPL_DUNS_FEI = facility identifier printed on the label; ADDRESS_MATCH = matched by address from "
        "ANDA records or label text (where 'Manufactured for' labelers can slip in).",
        "Rule-based hints of rows worth a closer look; empty = all sources agree it is an FDF manufacturer.",
        "Checked, Verdict, Notes are blank for the reviewer."]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        sheets["manual"].to_excel(xw, sheet_name="Manual", index=False)
        sheets["propublica"].to_excel(xw, sheet_name="ProPublica", index=False)
        from openpyxl.styles import Font, PatternFill, Alignment
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True)
                c.fill = PatternFill("solid", fgColor="DDE4F0")
                c.alignment = Alignment(wrap_text=True, vertical="top")
            for col in ws.columns:
                hdr = str(col[0].value or "")
                width = 60 if hdr in ("Description", "GDUFA business operations", "Address / location", "Auto flag") \
                    else 28 if hdr in ("Facility name", "DailyMed operations for this NDC (Jul 2025)", "Verdict (FDF / API / labeler / other)", "Notes", "Item") \
                    else 16
                ws.column_dimensions[col[0].column_letter].width = width
                for c in col[1:]:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                    if hdr == "DailyMed search" and c.value:
                        c.hyperlink = c.value
                        c.value = "open"
                        c.font = Font(color="0563C1", underline="single")
    print(f"Saved: {OUT}")
    for lab, s in sheets.items():
        print(f"{lab}: {s['NDC11'].nunique()} NDCs, {len(s)} rows, {s['FEI'].nunique()} FEIs, "
              f"flagged rows: {(s['Auto flag'] != '').sum()}")
    return sheets


if __name__ == "__main__":
    build()
# %%
