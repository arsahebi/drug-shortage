"""
eval/code/07_extraction_reproducibility.py

How reproducible is the extraction when nothing changes?

Why this exists
───────────────
Round 1 found that re-running the identical prompt on the identical 50
observations gave only 72-96% self-agreement by field, moving accuracy-vs-human
by 12-16 points with no prompt change at all
(eval/results_and_notes/20260902_human_eval_round1_findings_and_fixes.md).
That noise is why a single-pass accuracy figure cannot be reported as a point
estimate.

The obvious fix, temperature=0, is NOT available. claude-sonnet-5 rejects the
parameter ("`temperature` is deprecated for this model", HTTP 400) and
gpt-5-mini is a reasoning model whose Responses API also rejects it. Verified
2026-09-29. So the variance has to be measured and managed rather than switched
off, which is what this script is for: it re-runs the same 50 observations N
times through the CURRENT prompt and reports per-field self-agreement.

Use it two ways. Before a paid full-corpus run, to know the noise floor of the
prompt you are about to pay for. And to size the majority-vote protocol: the
number of passes needed before a per-field number is stable enough to report.

How it works
────────────
The prompt, the tool schema, and the model id are read straight out of
01_extract_observation_signals.py by parsing it (ast.literal_eval), never by
importing it -- that file is script-style and importing it would launch a real
extraction. Reading rather than copying means this diagnostic cannot silently
drift away from the prompt it is supposed to be testing.

The 50 observations are the exact rows Abdul labeled in round 1, recovered by
joining round 1's workbook back to the Redica observation file on
(fei, insp_date, obs_num).

Run from this folder:
  python 07_extraction_reproducibility.py            # 2 passes
  python 07_extraction_reproducibility.py --runs 3   # 3 passes

Output:
  eval/results_and_notes/extraction_reproducibility.csv
"""

from __future__ import annotations

import argparse
import ast
import copy
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
EVAL_ROOT = HERE.parent
DATA = EVAL_ROOT.parent

EXTRACT_PY = DATA / "01_extract_observation_signals.py"
REDICA_OBS = DATA / "step00_redica_483_observations.csv"
ROUND1_XLS = EVAL_ROOT / "sent_to_abdul" / "labeling_template_v2.xlsx"
OUT_DIR = EVAL_ROOT / "results_and_notes"

# the fields Abdul labels and the paper reports; rationale/quote free text is
# excluded because it is expected to vary in wording without varying in meaning
COMPARE_FIELDS = [
    "violation_category", "severity_tier", "scope", "root_cause_type",
    "remediation_signal", "repeat_flag_llm", "patient_risk_flag_llm",
    "contamination_flag_llm", "contamination_risk_flag_llm",
    "investigation_flag_llm", "data_integrity_flag_llm",
]


def _eval(node: ast.AST, ns: dict):
    """
    literal_eval, but also resolving bare names and copy.deepcopy() against
    constants already read. The v2 tool definition is assembled rather than
    written as a literal -- _ANTHROPIC_SCHEMA_V2 is a deepcopy of the OpenAI
    schema that is then patched field by field -- so plain literal_eval drops
    the whole thing.
    """
    if isinstance(node, ast.Name):
        return ns[node.id]
    if isinstance(node, ast.Dict):
        return {_eval(k, ns): _eval(v, ns) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [_eval(e, ns) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, ns) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval(node.left, ns) + _eval(node.right, ns)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "deepcopy"):
        return copy.deepcopy(_eval(node.args[0], ns))
    # the schemas build their enums with sorted(VALID_...), so a few pure
    # builtins have to be callable here
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = {"sorted": sorted, "list": list, "set": set,
              "tuple": tuple, "frozenset": frozenset}.get(node.func.id)
        if fn is not None:
            return fn(*[_eval(a, ns) for a in node.args])
    return ast.literal_eval(node)


def _patch_subscript(target: ast.Subscript, value, ns: dict) -> None:
    """Apply `X["a"]["b"] = value` to an already-resolved constant."""
    keys, node = [], target
    while isinstance(node, ast.Subscript):
        keys.append(ast.literal_eval(node.slice))
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in ns:
        return
    path = list(reversed(keys))
    obj = ns[node.id]
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value


def _load_constants() -> dict:
    """Pull the v2 prompt, tool schema, and model id out of the extractor."""
    tree = ast.parse(EXTRACT_PY.read_text())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        try:
            if isinstance(target, ast.Name):
                out[target.id] = _eval(node.value, out)
            elif isinstance(target, ast.Subscript):
                _patch_subscript(target, _eval(node.value, out), out)
        except Exception:
            pass
    if "MAX_TOKENS" not in out:
        sys.exit(f"Could not read MAX_TOKENS from {EXTRACT_PY.name}")
    return out


def _key(df: pd.DataFrame) -> pd.Series:
    return (df["fei"].astype(str).str.strip() + "|"
            + df["insp_date"].astype(str).str.strip() + "|"
            + df["obs_num"].astype(str).str.strip())


def _round1_rows() -> pd.DataFrame:
    obs = pd.read_csv(REDICA_OBS, low_memory=False)
    r1 = pd.read_excel(ROUND1_XLS, sheet_name="Labeling")
    obs["key"], r1["key"] = _key(obs), _key(r1)
    rows = obs[obs["key"].isin(set(r1["key"]))].copy()
    if len(rows) != len(r1):
        print(f"  WARNING: matched {len(rows)} of {len(r1)} round-1 rows")
    return rows.sort_values("key").reset_index(drop=True)


def _extract_one_openai(client, C: dict, text: str, cfr, model: str) -> dict:
    """Same v2 prompt and schema the pipeline sends, via the Responses API."""
    import json
    cfr_str = str(cfr).strip() if pd.notna(cfr) and str(cfr).strip() else "not specified"
    prompt = C["_PROMPT_TEMPLATE_V2"].format(
        obs_text_clean=text.strip(), cfr_codes=cfr_str,
        patient_risk_rule=C["_PATIENT_RISK_RULE_OPENAI_V2"])
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system",
             "content": ("You extract structured risk signals from FDA Form 483 "
                         "observations. Return only schema-valid JSON.")},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=C["MAX_TOKENS"],
        text={"format": {"type": "json_schema",
                         "name": "form_483_observation_signal",
                         "strict": True,
                         "schema": C["OPENAI_JSON_SCHEMA_V2"]}},
    )
    txt = getattr(resp, "output_text", "") or ""
    try:
        return json.loads(txt)
    except Exception:
        return {}


def _extract_one(client, C: dict, text: str, cfr, model: str) -> dict:
    tool = C["ANTHROPIC_TOOL_V2"]
    cfr_str = str(cfr).strip() if pd.notna(cfr) and str(cfr).strip() else "not specified"
    variable = C["_ANTHROPIC_PROMPT_VARIABLE_TEMPLATE"].format(
        obs_text_clean=text.strip(), cfr_codes=cfr_str)
    resp = client.messages.create(
        model=model,
        max_tokens=C["MAX_TOKENS"],
        system=[{
            "type": "text",
            "text": ("You extract structured risk signals from FDA Form 483 "
                     f"observations. Use the {tool['name']} tool to return your analysis."),
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user",
                   "content": [{"type": "text", "text": C["_ANTHROPIC_PROMPT_FIXED_V2"] + variable}]}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2,
                    help="number of identical passes; must be >= 2 to measure anything")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None,
                    help="default: claude-sonnet-5 / gpt-5-mini for the chosen provider")
    args = ap.parse_args()

    if args.runs < 2:
        sys.exit("--runs must be at least 2: self-agreement needs two passes to compare.")
    model = args.model or ("claude-sonnet-5" if args.provider == "anthropic" else "gpt-5-mini")
    if args.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set.")
        from anthropic import Anthropic
        client, call = Anthropic(), _extract_one
        needed = ["_ANTHROPIC_PROMPT_FIXED_V2", "_ANTHROPIC_PROMPT_VARIABLE_TEMPLATE",
                  "ANTHROPIC_TOOL_V2"]
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        client, call = OpenAI(), _extract_one_openai
        needed = ["_PROMPT_TEMPLATE_V2", "_PATIENT_RISK_RULE_OPENAI_V2",
                  "OPENAI_JSON_SCHEMA_V2"]

    C = _load_constants()
    missing = [n for n in needed if n not in C]
    if missing:
        sys.exit(f"Could not read {missing} from {EXTRACT_PY.name}")
    rows = _round1_rows()
    print(f"round-1 observations recovered      : {len(rows)}")
    print(f"provider / model                    : {args.provider} / {model}")
    print(f"passes                              : {args.runs}\n")

    runs = []
    for r in range(args.runs):
        recs = []
        t0 = time.time()
        for i, obs in rows.iterrows():
            res = call(client, C, str(obs["obs_text"]), obs.get("cfr_codes", ""), model)
            res["key"] = obs["key"]
            recs.append(res)
            if (i + 1) % 10 == 0:
                print(f"  run {r + 1}: {i + 1}/{len(rows)}")
        runs.append(pd.DataFrame(recs).set_index("key"))
        print(f"  run {r + 1} done in {time.time() - t0:.0f}s\n")

    base = runs[0]
    out = []
    for f in COMPARE_FIELDS:
        if f not in base.columns:
            print(f"  [skip] {f}: not returned by the tool")
            continue
        agrees = []
        for other in runs[1:]:
            a = base[f].astype(str).str.strip().str.lower()
            b = other[f].reindex(base.index).astype(str).str.strip().str.lower()
            agrees.append(100 * (a == b).mean())
        out.append({
            "field": f,
            "n": len(base),
            "pairwise_self_agreement_pct": round(sum(agrees) / len(agrees), 1),
            "min_pair_pct": round(min(agrees), 1),
            "runs": args.runs,
        })

    res = pd.DataFrame(out).sort_values("pairwise_self_agreement_pct")
    print("\nSelf-agreement (identical prompt, identical input, default decoding):")
    print(res.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"extraction_reproducibility_{args.provider}.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    worst = res["pairwise_self_agreement_pct"].min()
    print()
    if worst >= 99.5:
        print("All fields reproduce; a single pass is enough.")
    elif worst >= 95:
        print(f"Worst field {worst}%. Close to stable; 3 passes with a majority vote "
              "should be plenty.")
    else:
        print(f"Worst field {worst}%. Single-pass numbers are not reportable. Use "
              "repeated passes with a per-field majority vote, and state the "
              "self-agreement alongside every accuracy figure.")


if __name__ == "__main__":
    main()
