from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# Prefer image and table crops over text
PREFERRED_LABELS = {"image", "table"}


def to_int(x: Any) -> int:
    if isinstance(x, bool):  # avoid True->1 surprises
        return int(x)
    if isinstance(x, (int,)):
        return x
    try:
        return int(str(x).strip())
    except Exception:
        raise ValueError(f"Cannot convert to int: {x!r}")


def normalize_bbox(bbox: Iterable[Any]) -> List[int]:
    vals = list(bbox)
    if len(vals) != 4:
        raise ValueError(f"bbox must have 4 elements, got {vals}")
    return [to_int(v) for v in vals]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_by_doc(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        doc_id = str(r.get("doc_id", "")).strip()
        if not doc_id:
            # try to derive from doc_full_name if present
            df = r.get("doc_full_name")
            if isinstance(df, str) and df:
                doc_id = df.split("_")[0]
        if not doc_id:
            # skip if still missing
            continue
        grouped[doc_id].append(r)
    return grouped


def pick_samples_for_doc(rows: List[Dict[str, Any]], per_doc: int) -> List[Dict[str, Any]]:
    """Return up to per_doc unique SPSC candidates, biasing toward table/image crops."""
    seen: set[Tuple[int, int]] = set()
    unique: List[Dict[str, Any]] = []
    for r in rows:
        try:
            page_idx = to_int(r.get("page_index"))
        except Exception:
            if "page_id" in r:
                page_idx = to_int(r["page_id"]) if isinstance(r["page_id"], (int, str)) else to_int(r["page_id"][0])
            else:
                continue
        crop_idx = r.get("subimg_index") or r.get("crop_id")
        if crop_idx is None:
            continue
        crop_idx = to_int(crop_idx)
        key = (page_idx, crop_idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    if not unique:
        return []

    preferred: List[Dict[str, Any]] = []
    non_text: List[Dict[str, Any]] = []
    text_only: List[Dict[str, Any]] = []

    for r in unique:
        label = str(r.get("label", "")).lower()
        if label in PREFERRED_LABELS:
            preferred.append(r)
        elif label and label != "text":
            non_text.append(r)
        else:
            text_only.append(r)

    random.shuffle(preferred)
    random.shuffle(non_text)
    random.shuffle(text_only)

    ordered = preferred + non_text + text_only

    picked: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[int, int]] = set()

    for r in ordered:
        try:
            page_idx = to_int(r.get("page_index") or (r.get("page_id")[0] if isinstance(r.get("page_id"), list) else r.get("page_id")))
            crop_idx = to_int(r.get("subimg_index") or r.get("crop_id"))
        except Exception:
            continue
        key = (page_idx, crop_idx)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        picked.append(r)
        if len(picked) >= per_doc:
            break

    return picked


def transform_record(r: Dict[str, Any]) -> Dict[str, Any]:
    category = r.get("category", "")
    doc_id = str(r.get("doc_id", "")).strip()
    page_idx = r.get("page_index")
    if page_idx is None and "page_id" in r:
        page_idx = r["page_id"][0] if isinstance(r["page_id"], list) else r["page_id"]
    page_idx = to_int(page_idx)

    crop_idx = r.get("subimg_index")
    if crop_idx is None:
        crop_idx = r.get("crop_id")
    crop_idx = to_int(crop_idx)

    label = str(r.get("label", "")).lower() or "text"

    bbox = r.get("bbox")
    if bbox is None:
        # allow alternative keys
        bbox = r.get("xyxy") or r.get("box")
    bbox = normalize_bbox(bbox)

    return {
        "category": category,
        "doc_id": doc_id,
        "page_id": [page_idx],
        "crop_ids": [[crop_idx]],
        "type": [[label]],
        "bbox": [[bbox]],  # triple nested as required [[[x1,y1,x2,y2]]]
    }


def main():
    ap = argparse.ArgumentParser(description="Sample SPSC items per doc from a JSONL file.")
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--per-doc", type=int, default=10, help="Number of samples per doc (default: 1)")
    ap.add_argument("--max-total", type=int, default=0, help="Optional cap on total samples across all docs (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    rows = read_jsonl(args.input)
    grouped = group_by_doc(rows)

    outputs: List[Dict[str, Any]] = []
    for doc_id, doc_rows in grouped.items():
        picked = pick_samples_for_doc(doc_rows, args.per_doc)
        for r in picked:
            try:
                out = transform_record(r)
            except Exception as e:
                # skip malformed records
                continue
            outputs.append(out)
            if args.max_total and len(outputs) >= args.max_total:
                break
        if args.max_total and len(outputs) >= args.max_total:
            break

    with open(args.output, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Simple summary to stdout
    label_counts = defaultdict(int)
    for o in outputs:
        lbl = o.get("type", [[""]])[0][0]
        label_counts[lbl] += 1
    summary = {
        "docs": len(grouped),
        "samples": len(outputs),
        "by_label": dict(label_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
