from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
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


def pick_samples_for_doc(rows: List[Dict[str, Any]], per_doc: int, three_prob: float = 0.2) -> List[Dict[str, Any]]:
    """Return up to `per_doc` MPMC entries (exactly 2 pages per entry, 1 crop per page).

    Each entry samples two distinct pages within the document; from every selected page, we keep a single crop
    preferring labels in `PREFERRED_LABELS`. The `three_prob` argument is kept for backwards compatibility but ignored.
    """
    # Deduplicate by (page_index, subimg_index) and group by page
    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    page_has_preferred: Dict[int, bool] = {}
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
        label_lower = str(r.get("label", "")).lower()
        if label_lower in PREFERRED_LABELS:
            page_has_preferred[page_idx] = True
        else:
            page_has_preferred.setdefault(page_idx, False)

    pages_all = [p for p, lst in by_page.items() if len(lst) >= 1]
    if not pages_all:
        return []

    # Randomize page order for variety
    random.shuffle(pages_all)

    mpmc_entries: List[Dict[str, Any]] = []

    if len(pages_all) < 2:
        return mpmc_entries

    page_pairs = list(combinations(pages_all, 2))
    random.shuffle(page_pairs)
    page_pairs.sort(
        key=lambda pair: sum(1 for pg in pair if page_has_preferred.get(pg, False)),
        reverse=True,
    )

    for pair in page_pairs:
        if len(mpmc_entries) >= per_doc:
            break

        chosen_crops: List[Dict[str, Any]] = []
        for pg in pair:
            candidates = by_page.get(pg, [])
            if not candidates:
                break
            preferred = [r for r in candidates if str(r.get("label", "")).lower() in PREFERRED_LABELS]
            non_text = [r for r in candidates if str(r.get("label", "")).lower() != "text"]
            if preferred:
                pool = preferred
            elif non_text:
                pool = non_text
            else:
                pool = list(candidates)
            random.shuffle(pool)
            chosen_crops.append(pool[0])
        else:
            try:
                mpmc_entries.append(transform_group_mpmc(list(pair), chosen_crops))
            except Exception:
                continue

    return mpmc_entries
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

def transform_group_mpmc(pages: List[int], crops: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(pages) == len(crops) and len(crops) >= 2
    r0 = crops[0]
    category = r0.get("category", "")
    doc_id = str(r0.get("doc_id", "")).strip()

    page_ids: List[int] = []
    crop_ids: List[List[int]] = []
    labels: List[List[str]] = []
    bboxes: List[List[List[int]]] = []

    for pg, r in zip(pages, crops):
        page_ids.append(to_int(pg))
        # crop id
        cid = r.get("subimg_index")
        if cid is None:
            cid = r.get("crop_id")
        cid = to_int(cid)
        # label
        lab = str(r.get("label", "")).lower() or "text"
        # bbox
        bb = r.get("bbox") or r.get("xyxy") or r.get("box")
        bb = normalize_bbox(bb)

        crop_ids.append([cid])
        labels.append([lab])
        bboxes.append([bb])

    return {
        "category": category,
        "doc_id": doc_id,
        "page_ids": page_ids,   # plural for multi-page
        "crop_ids": crop_ids,   # aligned with pages, one per page
        "type": labels,         # aligned with pages
        "bbox": bboxes,         # aligned with pages
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
    ap = argparse.ArgumentParser(description="Sample MPMC items per doc from a JSONL file (2 pages per entry, 1 crop per page).")
    ap.add_argument("--input", required=True, help="Path to input JSONL")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--per-doc", type=int, default=4, help="Number of MPMC items per doc (default: 4)")
    ap.add_argument("--three-prob", type=float, default=0.2, help="Deprecated; retained for CLI compatibility and ignored.")
    ap.add_argument("--max-total", type=int, default=0, help="Optional cap on total samples across all docs (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    rows = read_jsonl(args.input)
    grouped = group_by_doc(rows)

    outputs: List[Dict[str, Any]] = []
    for doc_id, doc_rows in grouped.items():
        spmc_items = pick_samples_for_doc(doc_rows, args.per_doc, three_prob=args.three_prob)
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
    page_count_counts = defaultdict(int)
    for o in outputs:
        if "page_ids" in o and isinstance(o["page_ids"], list):
            num_pages = len(o["page_ids"])
            page_count_counts[num_pages] += 1
    summary = {
        "docs": len(grouped),
        "samples": len(outputs),
        "by_label": dict(label_counts),
        "by_page_count": dict(page_count_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
