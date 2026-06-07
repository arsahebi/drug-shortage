Project Drug Shortage — Minimal Code Style

Purpose
- Keep scripts short, readable, and auditable.
- Prefer a few small helpers over many generic utilities.
- Make file layout predictable so tools/scripts can find data.

Repository layout (per dataset)
- Data/<dataset>/raw/        # original files exactly as received
- Data/<dataset>/processed/  # cleaned outputs, summaries
- Data/<dataset>/processed/code/  # scripts for that dataset

- Script style
- Use small modules/scripts per dataset. Prefer simple top-level scripts that run interactively (no `if __name__ == "__main__"` needed), since many workflows run code in interactive sessions.
- If a script needs to be importable, add a `main()` but keep the execution flow readable and minimal.
- Minimize indirection: only create helper functions when reused or to clarify a step.
- Favor explicit, readable pandas ops over heavy generic abstractions.
- Normalize column names early (`df.columns = df.columns.str.strip().str.lower()`) when working with tabular data.
- Use `default=str` when `json.dump` to avoid serialization errors from datetimes.

File discovery pattern
- Use `Path('<dataset>/raw').glob('**/*')` and filter by suffix.
- Support `.csv`, `.xlsx`, `.xls`, and `.docx` minimally.

Outputs
- Every explorer/processor should write a small summary JSON + CSV to `processed/`.
- For large final tables, prefer Parquet with compression (`df.to_parquet(..., compression='snappy')`).

Testing and reproducibility
- Keep scripts runnable via `python path/to/script.py`.
- If external packages are required (e.g., `python-docx`), raise a clear error and document the dependency.

Naming and metadata
- Use ISO-like prefixes for scripts: `YYYY-MM-DD-brief-desc.py`.
- Store generated files in `processed/` and include the generator script in `processed/code/`.

Example minimal pattern (FAERS-style)
- small helper(s)
- discover files
- per-file load -> normalize -> summarize
- concat/write outputs

When in doubt, prefer clarity and a few lines of explicit code over over-generalized utility functions.
