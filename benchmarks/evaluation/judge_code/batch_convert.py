#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch converter for QA evaluation prompts.

Reads records containing at least "query" and "answer" fields, plus a generated answer.
If the generated answer field isn't present under common variants, it will try to use "generate".

Supported input formats:
- JSON Lines (.jsonl): one JSON object per line
- JSON array (.json): a top-level list of JSON objects
- CSV (.csv): with headers like query, answer, generated answer (or generate)

Usage:
    python batch_convert.py input.jsonl output.jsonl --start-id 1
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

TEMPLATE = (
    "You are given the following information:\n"
    "- the query\n"
    "- a generated answer\n"
    "- a reference answer\n\n"
    "Your task is to evaluate the correctness of the generated answer.\n\n"
    "## Query\n"
    "{q}\n\n"
    "## Reference Answer\n"
    "{gold_ans}\n\n"
    "## Generated Answer\n"
    "{gen_ans}\n\n"
    "Your response should be formatted as following:\n"
    "<judge>True or False</judge>\n\n"
    "If the generated answer is correct, please set \"judge\" to True. Otherwise, please set \"judge\" to False.\n\n"
    "Please note that the generated answer may contain additional information beyond the reference answer."
)

DEFAULT_SYSTEM = "You are an expert evaluation system for a question answering chatbot."

# Common key aliases normalized by removing spaces and lowercasing
GEN_ANSWER_KEYS = [
    "generatedanswer",
    "generateanswer",
    "genanswer",
    "gen_ans",
    "generated_answer",
    "generated",
    "prediction",
    "modelanswer",
    "model_output",
    "modeloutput",
    "output",
    "response",
    "gen",
    "generate",
]

# Helper to extract a readable string from various generated-answer shapes
def _stringify_generated(val: Any) -> str:
    """Return a string for generated answer. If it's a dict, prefer typical keys like 'answer'.
    Falls back to JSON serialization for non-string types."""
    if isinstance(val, dict):
        # Prefer common inner keys conveying the actual text
        for k in [
            "answer",
            "generated answer",
            "generate answer",
            "generated_answer",
            "output",
            "prediction",
            "text",
            "content",
        ]:
            if k in val and val[k] not in (None, ""):
                return str(val[k])
        # Fallback: serialize the whole dict
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, (list, tuple)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def norm_key(k: str) -> str:
    return k.replace(" ", "").lower()

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError("Top-level JSON must be a list for .json input.")

def load_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def read_records(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return load_jsonl(path)
    elif ext == ".json":
        return load_json_array(path)
    elif ext == ".csv":
        return load_csv(path)
    else:
        # Try JSONL as a fallback
        try:
            return load_jsonl(path)
        except Exception:
            pass
        # Try JSON array
        try:
            return load_json_array(path)
        except Exception:
            pass
        # Try CSV
        try:
            return load_csv(path)
        except Exception:
            pass
        raise ValueError(f"Unsupported input format for file: {path}")

def extract_field(record: Dict[str, Any], wanted_keys: List[str]) -> Any:
    """Return the first present field value from `wanted_keys` variants (normalized)."""
    normalized_map = {norm_key(k): k for k in record.keys()}
    for alias in wanted_keys:
        if alias in normalized_map:
            return record[normalized_map[alias]]
    return None

def get_generated_answer(record: Dict[str, Any]) -> str:
    """Resolve the generated answer.
    Priority:
    1) Exact keys like 'generated answer' / 'generate answer' / 'generated_answer' / 'generatedAnswer'.
       If the found value is a dict, prefer inner key 'answer'; else stringify.
    2) Flexible aliases listed in GEN_ANSWER_KEYS (e.g., 'generated', 'generate', 'model_output', etc.).
       If the value is a dict, prefer inner 'answer' and other common keys; else stringify.
    3) Explicit fallback to 'generate' if present (stringify as above).
    Raises KeyError if nothing is found.
    """
    # First try exact keys
    direct_candidates = [
        "generated answer",
        "generate answer",
        "generated_answer",
        "generatedAnswer",
    ]
    for k in direct_candidates:
        if k in record and record[k] not in (None, ""):
            val = record[k]
            return _stringify_generated(val)

    # Flexible search among a variety of aliases
    normalized_map = {norm_key(k): k for k in record.keys()}
    for alias in GEN_ANSWER_KEYS:
        if alias in normalized_map:
            val = record[normalized_map[alias]]
            if val not in (None, ""):
                # If it's an object like {"answer": ...}, prefer the inner answer
                if isinstance(val, dict) and "answer" in val and val["answer"] not in (None, ""):
                    return str(val["answer"])
                return _stringify_generated(val)

    # Fallback: try raw 'generate' explicitly if present (user requested this behavior)
    if "generate" in record and record["generate"] not in (None, ""):
        return _stringify_generated(record["generate"])

    raise KeyError("No generated answer found (tried common variants and 'generate').")

def get_required(record: Dict[str, Any], names: List[str]) -> str:
    for n in names:
        if n in record and record[n] not in (None, ""):
            return str(record[n])
    # Try normalized lookup
    normalized_map = {norm_key(k): k for k in record.keys()}
    for n in names:
        nk = norm_key(n)
        if nk in normalized_map and record[normalized_map[nk]] not in (None, ""):
            return str(record[normalized_map[nk]])
    raise KeyError(f"Missing required field. Looked for any of: {names}")

def build_item(idx: int, q: str, gold: str, gen: str, system: str, top_p: float, temperature: float, penalty_score: float) -> Dict[str, Any]:
    content = TEMPLATE.format(q=q, gold_ans=gold, gen_ans=gen)
    return {
        "id": str(idx),
        "request_body": {
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "top_p": top_p,
            "temperature": temperature
        }
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch convert QA examples into evaluation JSON.")
    ap.add_argument("input", help="Path to input file (.jsonl, .json, or .csv).")
    ap.add_argument("output", help="Path to output file (.jsonl for line-delimited JSON; .json for array).")
    ap.add_argument("--start-id", type=int, default=1, help="Starting numeric ID (default: 1).")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM, help="System prompt string.")
    ap.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--penalty-score", type=float, default=1.1, dest="penalty_score")
    args = ap.parse_args()

    # Load records
    try:
        records = read_records(args.input)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    out_items: List[Dict[str, Any]] = []
    id_counter = args.start_id

    for rec in records:
        try:
            q = get_required(rec, ["query", "question", "q"])
            gold = get_required(rec, ["answer", "reference answer", "reference_answer", "gold", "gold_answer"])
            gen = get_generated_answer(rec)
        except Exception as e:
            # If generated answer isn't found but "generate" exists, the helper already handles it.
            print(f"Skipping record due to error: {e}\nRecord: {rec}", file=sys.stderr)
            continue

        item = build_item(id_counter, q, gold, gen, args.system, args.top_p, args.temperature, args.penalty_score)
        out_items.append(item)
        id_counter += 1

    # Write output
    out_ext = os.path.splitext(args.output)[1].lower()
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            if out_ext == ".jsonl":
                for obj in out_items:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                # default to JSON array
                json.dump(out_items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(out_items)} items to {args.output}")

if __name__ == "__main__":
    main()
