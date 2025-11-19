from __future__ import annotations
import argparse
import json
from typing import Dict, Tuple, List, Any
import os

Key = Tuple[str, int]


def load_summary_map(path: str) -> Dict[Key, str]:
    mapping: Dict[Key, str] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"summary.jsonl line {lineno} is not valid JSON: {e}") from e

            if not isinstance(obj, dict):
                continue
            doc_id = obj.get("doc_id")
            page_index = obj.get("page_index")
            summary = obj.get("summary")
            if isinstance(page_index, str):
                try:
                    page_index = int(page_index)
                except Exception:
                    page_index = None
            if not isinstance(doc_id, str) or not isinstance(page_index, int) or not isinstance(summary, str):
                # Skip lines that don't have the required fields
                continue
            mapping[(doc_id, page_index)] = summary
            if "_" in doc_id:
                simple_id = doc_id.split("_", 1)[0]
                mapping.setdefault((simple_id, page_index), summary)
    return mapping


def pages_from_entry(entry: Dict[str, Any]) -> List[int]:
    if "page_id" in entry and isinstance(entry["page_id"], int):
        return [entry["page_id"]]
    if "page_ids" in entry and isinstance(entry["page_ids"], list):
        return [p for p in entry["page_ids"] if isinstance(p, int)]
    # some data may use "page_id" as a list with single int
    if "page_id" in entry and isinstance(entry["page_id"], list):
        return [p for p in entry["page_id"] if isinstance(p, int)]
    return []


def split_doc_id(entry: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = entry.get("doc_id")
    if isinstance(doc_id, str) and "_" in doc_id:
        parts = doc_id.split("_", 1)
        entry["doc_id"] = parts[0]
        entry["doc_full_name"] = parts[1]
    return entry

def attach_summary(entry: Dict[str, Any], s_map: Dict[Key, str], missing: str) -> Dict[str, Any]:
    pages = pages_from_entry(entry)
    doc_id = entry.get("doc_id")
    if not isinstance(doc_id, str) or not pages:
        return entry

    doc_full = entry.get("doc_full_name")
    summaries: List[str] = []
    for p in pages:
        keys_to_try = [(doc_id, p)]
        if isinstance(doc_full, str):
            keys_to_try.append((doc_full, p))
        found = False
        for key in keys_to_try:
            if key in s_map:
                summaries.append(s_map[key])
                found = True
                break
        if not found:
            if missing == "error":
                raise KeyError(f"Missing summary for (doc_id={doc_id}, page_index={p})")
            elif missing == "ignore":
                # If any missing and ignore, skip adding the summary field entirely
                return entry
            else:  # empty
                summaries.append("")

    # Always write list form to align with page_ids-based structure
    entry["summary"] = summaries
    return entry


def process(spsc_path: str, summary_path: str, out_path: str, missing: str) -> None:
    s_map = load_summary_map(summary_path)

    with open(spsc_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"SPSC.jsonl line {lineno} is not valid JSON: {e}\nLine: {raw[:200]}...") from e

            if not isinstance(obj, dict):
                # pass-through anything non-dict
                fout.write(raw + "\n")
                continue

            obj = split_doc_id(obj)
            try:
                updated = attach_summary(obj, s_map, missing)
            except KeyError:
                continue
            fout.write(json.dumps(updated, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Attach page summaries to SPSC records.")
    ap.add_argument('--spsc', '--input', dest='spsc', required=True, help='Path to SPSC.jsonl input')
    ap.add_argument('--summary', required=True, help='Path to summary.jsonl input')
    ap.add_argument('--out', required=True, help='Path to write output JSONL with summaries attached')
    ap.add_argument('--missing', choices=['empty', 'ignore', 'error'], default='empty',
                    help='How to handle missing summaries (default: empty)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    process(args.spsc, args.summary, args.out, args.missing)


if __name__ == '__main__':
    main()
