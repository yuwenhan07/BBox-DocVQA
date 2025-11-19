from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

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


def group_by_doc(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        category = str(r.get("category", "")).strip()
        if not category:
            continue
        doc_id = str(r.get("doc_id", "")).strip()
        if not doc_id:
            # try to derive from doc_full_name if present
            df = r.get("doc_full_name")
            if isinstance(df, str) and df:
                doc_id = df.split("_")[0]
        if not doc_id:
            # skip if still missing
            continue
        grouped[(category, doc_id)].append(r)
    return grouped


def pick_samples_for_doc(rows: List[Dict[str, Any]], per_doc: int) -> List[Dict[str, Any]]:
    """Return up to `per_doc` SPMC entries (one page per entry, 2-3 crops each).

    For each page in the doc, we randomly pick 2 or 3 unique crops.
    Preference: use labels in PREFERRED_LABELS first, then fill with others.
    """
    # Deduplicate by (page_index, subimg_index) and group by page
    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    seen: set[Tuple[int, int]] = set()

    for r in rows:
        # Resolve page index
        page_idx_val = r.get("page_index")
        if page_idx_val is None and "page_id" in r:
            page_idx_val = r["page_id"][0] if isinstance(r["page_id"], list) else r["page_id"]
        try:
            page_idx = to_int(page_idx_val)
        except Exception:
            continue

        # Resolve crop id
        crop_idx_val = r.get("subimg_index")
        if crop_idx_val is None:
            crop_idx_val = r.get("crop_id")
        if crop_idx_val is None:
            continue
        try:
            crop_idx = to_int(crop_idx_val)
        except Exception:
            continue

        key = (page_idx, crop_idx)
        if key in seen:
            continue
        seen.add(key)
        by_page[page_idx].append(r)

    # Build SPMC entries; iterate pages in random order
    pages = list(by_page.keys())
    random.shuffle(pages)

    spmc_entries: List[Dict[str, Any]] = []

    for page in pages:
        if len(spmc_entries) >= per_doc:
            break
        crops = by_page[page]
        if len(crops) < 2:
            # need at least 2 crops to form SPMC
            continue

        # Partition by preferred labels
        preferred = [r for r in crops if str(r.get("label", "")).lower() in PREFERRED_LABELS]
        others = [r for r in crops if r not in preferred]

        # Randomize within buckets
        random.shuffle(preferred)
        random.shuffle(others)

        # Choose 2 or 3 randomly, but cap by availability
        k = random.choice([2, 3])
        k = min(k, max(2, len(crops)))

        picked = preferred[:k]
        if len(picked) < k:
            picked.extend(others[: (k - len(picked))])
        if len(picked) < 2:
            continue

        # Transform this page's picks into one SPMC record
        try:
            spmc_entries.append(transform_group(page, picked))
        except Exception:
            # skip malformed group
            continue

    return spmc_entries
def transform_group(page_idx: int, crops: List[Dict[str, Any]]) -> Dict[str, Any]:
    # All crops are from the same doc; pick common fields from the first
    r0 = crops[0]
    category = r0.get("category", "")
    doc_id = str(r0.get("doc_id", "")).strip()

    crop_ids: List[int] = []
    labels: List[str] = []
    bboxes: List[List[int]] = []

    for r in crops:
        # crop id
        cid = r.get("subimg_index")
        if cid is None:
            cid = r.get("crop_id")
        cid = to_int(cid)
        crop_ids.append(cid)

        # label
        lab = str(r.get("label", "")).lower() or "text"
        labels.append(lab)

        # bbox
        bb = r.get("bbox") or r.get("xyxy") or r.get("box")
        bb = normalize_bbox(bb)
        bboxes.append(bb)

    return {
        "category": category,
        "doc_id": doc_id,
        "page_id": [to_int(page_idx)],
        "crop_ids": [crop_ids],      # one page, multiple crops
        "type": [labels],            # labels aligned with crop_ids
        "bbox": [bboxes],            # bboxes aligned with crop_ids
    }


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
    ap = argparse.ArgumentParser(description="Sample SPMC items per doc from a JSONL file (2-3 crops per page, 7 entries per doc by default).")
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--per-doc", type=int, default=8, help="Number of SPMC items per doc (default: 7)")
    ap.add_argument("--max-total", type=int, default=0, help="Optional cap on total samples across all docs (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    rows = read_jsonl(args.input)
    grouped = group_by_doc(rows)

    outputs: List[Dict[str, Any]] = []
    for (_, doc_id), doc_rows in grouped.items():
        spmc_items = pick_samples_for_doc(doc_rows, args.per_doc)
        for item in spmc_items:
            outputs.append(item)
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
        tfield = o.get("type")
        if isinstance(tfield, list):
            for tlist in tfield:            # per page
                if isinstance(tlist, list):
                    for lbl in tlist:       # per crop within the page
                        if isinstance(lbl, str) and lbl:
                            label_counts[lbl.lower()] += 1
    crop_size_counts = defaultdict(int)
    for o in outputs:
        if "crop_ids" in o and isinstance(o["crop_ids"], list) and len(o["crop_ids"]) > 0:
            num_crops = len(o["crop_ids"][0])
            crop_size_counts[num_crops] += 1
    summary = {
        "docs": len(grouped),
        "samples": len(outputs),
        "by_label": dict(label_counts),
        "by_crop_count": dict(crop_size_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
