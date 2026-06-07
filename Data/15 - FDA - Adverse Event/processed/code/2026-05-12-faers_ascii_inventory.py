# %%
# ============================================
# FAERS + Orange Book: Metformin, ANDA vs NDA,
# severity over time
# ============================================

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- Paths (adjust if needed) ----------------

DATA_DIR = "/Users/asahebi/Library/CloudStorage/GoogleDrive-asahebi@ncsu.edu/My Drive/North Carolina State University/Project - Drug Shortage/Data"
FAERS_DIR = Path(os.path.join(DATA_DIR, "15 - FDA - Adverse Event/raw"))
OUT_DIR = Path(os.path.join(DATA_DIR, "15 - FDA - Adverse Event/processed"))

def find_ascii_dir(folder_root: Path) -> Path:
    for name in ["ASCII", "ascii", "Ascii"]:
        d = folder_root / name
        if d.is_dir():
            return d
    raise FileNotFoundError(f"No ASCII/ascii folder in {folder_root}")

def count_data_rows_txt(path: Path, has_header: bool = True) -> int:
    """
    Count lines in a big text file fast, then subtract header (default).
    Robust to missing trailing newline.
    """
    # count '\n' in chunks
    n_newlines = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)  # 8MB
            if not chunk:
                break
            n_newlines += chunk.count(b"\n")

    if n_newlines == 0:
        # could be empty file or single-line file
        return 0 if has_header else 1

    # if file doesn't end with '\n', there's one extra line
    try:
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        total_lines = n_newlines + (1 if last != b"\n" else 0)
    except OSError:
        total_lines = n_newlines  # fallback

    data_rows = total_lines - (1 if has_header else 0)
    return max(data_rows, 0)

# ---------------- 1) Discover quarters ----------------
quarters = []
pat = re.compile(r"^faers_ascii_(\d{4})q([1-4])$", re.IGNORECASE)

for folder in Path(FAERS_DIR).iterdir():
    if folder.is_dir():
        m = pat.match(folder.name)
        if m:
            quarters.append((int(m.group(1)), int(m.group(2)), folder))

quarters = sorted(quarters, key=lambda x: (x[0], x[1]))
print("Found quarters:", [(y, q) for y, q, _ in quarters])

# ---------------- 2) Scan all .txt tables per quarter ----------------
rows = []
for year, qtr, qfolder in quarters:
    ascii_dir = find_ascii_dir(qfolder)

    txt_files = sorted([p for p in ascii_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not txt_files:
        print(f"WARNING: no .txt files in {ascii_dir}")
        continue

    for p in txt_files:
        size_mb = p.stat().st_size / (1024 * 1024)
        nrec = count_data_rows_txt(p, has_header=True)

        # table name is prefix before yyQq, e.g. DEMO15Q1.txt -> DEMO
        table = re.split(r"\d{2}[qQ][1-4]", p.stem, maxsplit=1)[0].upper()

        rows.append({
            "year": year,
            "quarter": qtr,
            "period": f"{year}Q{qtr}",
            "table": table,
            "file": p.name,
            "records": nrec,
            "size_mb": size_mb,
        })

df = pd.DataFrame(rows)

# ---------------- 3) Summaries ----------------
per_q = (df.groupby("period", as_index=False)[["records", "size_mb"]].sum()
           .sort_values("period", key=lambda s: s.map(lambda x: (int(x[:4]), int(x[-1])))))

per_table = (df.groupby("table", as_index=False)[["records", "size_mb"]].sum()
               .sort_values("records", ascending=False))

total_records = int(df["records"].sum())
total_size_mb = float(df["size_mb"].sum())

print("\n=== TOTAL (all quarters, all .txt tables) ===")
print(f"Total records: {total_records:,}")
print(f"Total size_mb: {total_size_mb:,.2f}")

print("\n=== By quarter (sum across tables) [head] ===")
print(per_q.head(10).to_string(index=False))

print("\n=== By table (sum across quarters) ===")
print(per_table.to_string(index=False))

# ---------------- 4) Save outputs ----------------
out_dir = Path(DATA_DIR) / "15 - FDA - Adverse Event" / "processed" / "out"
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "faers_ascii_file_inventory_all_drugs.csv", index=False)
per_q.to_csv(out_dir / "faers_ascii_totals_by_quarter_all_drugs.csv", index=False)
per_table.to_csv(out_dir / "faers_ascii_totals_by_table_all_drugs.csv", index=False)

print("\nSaved:")
print(" -", out_dir / "faers_ascii_file_inventory_all_drugs.csv")
print(" -", out_dir / "faers_ascii_totals_by_quarter_all_drugs.csv")
print(" -", out_dir / "faers_ascii_totals_by_table_all_drugs.csv")

# %%
