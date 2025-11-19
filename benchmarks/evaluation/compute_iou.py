import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

Box = Tuple[float, float, float, float]
Page = Union[int, str]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_page(value: Page) -> Page:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


def page_sort_key(page: Page) -> Tuple[int, Union[int, str]]:
    if isinstance(page, int):
        return (0, page)
    if isinstance(page, float):
        return (0, int(page))
    if isinstance(page, str):
        return (1, page)
    return (2, str(page))


def is_box(candidate: object) -> bool:
    return (
        isinstance(candidate, (list, tuple))
        and len(candidate) == 4
        and all(isinstance(v, (int, float)) for v in candidate)
    )


def extract_boxes(node: Iterable) -> List[Box]:
    boxes: List[Box] = []

    def _collect(candidate):
        if isinstance(candidate, (list, tuple)):
            if len(candidate) == 4 and all(isinstance(v, (int, float)) for v in candidate):
                x1, y1, x2, y2 = candidate
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
            else:
                for child in candidate:
                    _collect(child)

    _collect(node)
    return boxes


def build_page_boxes(pages: Sequence[Page], boxes_source) -> Dict[Page, List[Box]]:
    page_map: Dict[Page, List[Box]] = {}
    normalized_pages = [normalize_page(page) for page in pages]
    for page in normalized_pages:
        page_map[page] = []

    if boxes_source is None:
        return page_map

    def add_boxes_to_page(target_page: Page, source) -> None:
        boxes = extract_boxes(source)
        if boxes:
            page_map.setdefault(target_page, [])
            page_map[target_page].extend(boxes)

    if isinstance(boxes_source, dict):
        for raw_key, value in boxes_source.items():
            page = normalize_page(raw_key)
            add_boxes_to_page(page, value)
        return page_map

    if isinstance(boxes_source, list):
        if is_box(boxes_source):
            target_page = normalized_pages[0] if normalized_pages else normalize_page(0)
            add_boxes_to_page(target_page, boxes_source)
            return page_map

        list_elements_are_dicts = all(
            isinstance(item, dict) and {"page", "bbox"} <= set(item.keys()) for item in boxes_source
        )
        if list_elements_are_dicts:
            for item in boxes_source:
                page = normalize_page(item.get("page"))
                add_boxes_to_page(page, item.get("bbox", []))
            return page_map

        if len(boxes_source) == len(normalized_pages):
            for idx, page in enumerate(normalized_pages):
                add_boxes_to_page(page, boxes_source[idx])
            return page_map

        for idx, item in enumerate(boxes_source):
            page = normalized_pages[idx] if idx < len(normalized_pages) else normalize_page(idx)
            add_boxes_to_page(page, item)
        return page_map

    if normalized_pages:
        page = normalized_pages[0]
        add_boxes_to_page(page, boxes_source)
    return page_map


def map_to_page_list(page_map: Dict[Page, List[Box]], page_order: Sequence[Page]) -> List[List[Box]]:
    normalized_order = [normalize_page(page) for page in page_order]
    seen = set()
    pages_list: List[List[Box]] = []
    for page in normalized_order:
        pages_list.append(list(page_map.get(page, [])))
        seen.add(page)
    for page in sorted(page_map.keys(), key=page_sort_key):
        if page not in seen:
            pages_list.append(list(page_map[page]))
    return pages_list


def parse_prediction_boxes(entry: dict) -> Tuple[Dict[Page, List[Box]], bool, Optional[str]]:
    pages = entry.get("evidence_page") or []
    raw_generate = entry.get("generate")
    parsed_generate = None
    raw_text: Optional[str] = None

    if isinstance(raw_generate, dict):
        parsed_generate = raw_generate
    elif isinstance(raw_generate, str):
        raw_text = raw_generate
        try:
            cleaned = raw_generate.replace("```json", "").replace("```", "").strip()
            parsed_generate = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed_generate = None
    elif raw_generate is not None:
        raw_text = json.dumps(raw_generate, ensure_ascii=False)

    if parsed_generate is None and entry.get("generate_parsed") is not None:
        parsed_generate = entry.get("generate_parsed")

    boxes_source = None
    if parsed_generate is not None:
        if isinstance(parsed_generate, dict):
            if "bboxes" in parsed_generate:
                boxes_source = parsed_generate["bboxes"]
            elif "bbox" in parsed_generate:
                boxes_source = parsed_generate["bbox"]
        else:
            boxes_source = parsed_generate

    parsed_ok = True
    if boxes_source is None:
        parsed_ok = False

    page_boxes = build_page_boxes(pages, boxes_source)
    return page_boxes, parsed_ok, raw_text


# ---------------------------------------------------------------------------
# IoU helpers – aligned with eval_iou.py logic
# ---------------------------------------------------------------------------

def _area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_box(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    ua = _area(a) + _area(b) - inter
    if ua <= 0.0:
        return 0.0
    return inter / ua


def iou_single_page_single_box(
    gt: List[Tuple[float, float, float, float]], pr: List[Tuple[float, float, float, float]]
) -> float:
    if not gt or not pr:
        return 0.0
    return iou_box(gt[0], pr[0])


def iou_single_page_multi_box(
    gt: List[Tuple[float, float, float, float]], pr: List[Tuple[float, float, float, float]]
) -> float:
    if not gt:
        return 0.0
    vals: List[float] = []
    for g in gt:
        if not pr:
            vals.append(0.0)
        else:
            vals.append(max(iou_box(g, p) for p in pr))
    return sum(vals) / len(vals)


def iou_multi_page(
    gt_pages: List[List[Tuple[float, float, float, float]]],
    pr_pages: List[List[Tuple[float, float, float, float]]],
) -> float:
    total_pages = max(len(gt_pages), len(pr_pages))
    if total_pages == 0:
        return 0.0
    page_ious: List[float] = []
    for idx in range(total_pages):
        gt = gt_pages[idx] if idx < len(gt_pages) else []
        pr = pr_pages[idx] if idx < len(pr_pages) else []
        if len(gt) <= 1 and len(pr) <= 1:
            page_ious.append(iou_single_page_single_box(gt, pr))
        else:
            page_ious.append(iou_single_page_multi_box(gt, pr))
    return sum(page_ious) / len(page_ious)


def compute_iou_per_rules(
    gt_pages: List[List[Tuple[float, float, float, float]]],
    pr_pages: List[List[Tuple[float, float, float, float]]],
) -> float:
    if (
        len(gt_pages) == 1
        and len(pr_pages) == 1
        and len(gt_pages[0]) == 1
        and len(pr_pages[0]) == 1
    ):
        return iou_single_page_single_box(gt_pages[0], pr_pages[0])
    if len(gt_pages) == 1 and len(pr_pages) == 1:
        return iou_single_page_multi_box(gt_pages[0], pr_pages[0])
    return iou_multi_page(gt_pages, pr_pages)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IoU of predicted bboxes vs. GT within a single JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_gpt_v2/gpt_pages_bbox_and_answer_v2.jsonl"),
        help="JSONL file where each line contains GT fields ('bbox') and prediction fields ('generate').",
    )
    parser.add_argument(
        "--out_file",
        type=Path,
        help="Optional JSONL path to store per-sample IoU results (defaults to '<input>_iou_results.jsonl').",
    )
    parser.add_argument(
        "--failed",
        type=Path,
        default=Path("failed_generates.jsonl"),
        help="Where to store raw generate outputs that could not be parsed.",
    )
    args = parser.parse_args()

    entries = load_jsonl(args.input)

    total_iou = 0.0
    hits_0_5 = 0
    hits_0_7 = 0
    n_bad = 0
    parsed_failures = 0

    ious_all: List[List[float]] = [[] for _ in range(3)]
    hit_0_5_all: List[List[float]] = [[] for _ in range(3)]
    hit_0_7_all: List[List[float]] = [[] for _ in range(3)]
    results: List[dict] = []
    failed_records: List[dict] = []

    for idx, entry in enumerate(entries):
        pages = entry.get("evidence_page") or []
        gt_map = build_page_boxes(pages, entry.get("bbox"))
        pred_map, parsed_ok, raw_generate = parse_prediction_boxes(entry)
        if not parsed_ok and raw_generate is not None:
            parsed_failures += 1
            failed_records.append(
                {"index": idx, "doc_name": entry.get("doc_name"), "query": entry.get("query"), "generate": raw_generate}
            )

        gt_order: List[Page] = []
        seen_gt = set()
        for page in pages:
            norm = normalize_page(page)
            if norm not in seen_gt:
                gt_order.append(norm)
                seen_gt.add(norm)
        for extra_page in sorted(gt_map.keys(), key=page_sort_key):
            if extra_page not in seen_gt:
                gt_order.append(extra_page)
                seen_gt.add(extra_page)
        if not gt_order and gt_map:
            gt_order = sorted(gt_map.keys(), key=page_sort_key)

        page_order = list(gt_order)
        seen_pages = set(page_order)
        for extra_page in sorted(pred_map.keys(), key=page_sort_key):
            if extra_page not in seen_pages:
                page_order.append(extra_page)
                seen_pages.add(extra_page)

        gt_pages_stats = map_to_page_list(gt_map, gt_order)
        gt_pages_eval = map_to_page_list(gt_map, page_order)
        pr_pages_eval = map_to_page_list(pred_map, page_order)

        iou_val = compute_iou_per_rules(gt_pages_eval, pr_pages_eval)
        total_iou += iou_val

        hit_0_5 = float(iou_val >= 0.5)
        hit_0_7 = float(iou_val >= 0.7)
        hits_0_5 += int(hit_0_5)
        hits_0_7 += int(hit_0_7)
        if iou_val < 1e-6:
            n_bad += 1

        if len(gt_pages_stats) == 1 and len(gt_pages_stats[0]) == 1:
            type_idx = 0
        elif len(gt_pages_stats) == 1 and len(gt_pages_stats[0]) > 1:
            type_idx = 1
        else:
            type_idx = 2

        ious_all[type_idx].append(iou_val)
        hit_0_5_all[type_idx].append(hit_0_5)
        hit_0_7_all[type_idx].append(hit_0_7)

        results.append(
            {
                "index": idx,
                "doc_name": entry.get("doc_name"),
                "query": entry.get("query"),
                "iou": round(iou_val, 6),
                "hit_0_5": int(hit_0_5),
                "hit_0_7": int(hit_0_7),
                "gt_pages": gt_pages_eval,
                "pred_pages": pr_pages_eval,
            }
        )

    sample_count = len(entries)
    mean_iou = total_iou / sample_count if sample_count else 0.0
    iou_at_0_5 = hits_0_5 / sample_count if sample_count else 0.0
    iou_at_0_7 = hits_0_7 / sample_count if sample_count else 0.0

    print("file_name:", args.input)
    print("=" * 50)
    print(
        f"[RESULT] Samples: {sample_count}, Bad parses: {parsed_failures}, Good ratio: "
        f"{(1 - parsed_failures / sample_count) if sample_count else 0.0:.4f}"
    )
    print(f"         Mean IoU:  {mean_iou:.6f}")
    print(f"         IoU@0.5:   {iou_at_0_5:.6f}")
    print(f"         IoU@0.7:   {iou_at_0_7:.6f}")
    print("-" * 50)

    type_names = [
        "Type 1 (1 page, 1 box)",
        "Type 2 (1 page, multi boxes)",
        "Type 3 (multi pages)",
    ]
    for type_idx in range(3):
        count = len(ious_all[type_idx])
        if count == 0:
            continue
        mean_iou_type = sum(ious_all[type_idx]) / count
        iou_0_5_type = sum(hit_0_5_all[type_idx]) / count
        iou_0_7_type = sum(hit_0_7_all[type_idx]) / count
        print(
            f"[TYPE {type_idx + 1}] Samples: {count}, Mean IoU: {mean_iou_type:.6f}, "
            f"IoU@0.5: {iou_0_5_type:.6f}, IoU@0.7: {iou_0_7_type:.6f} -- {type_names[type_idx]}"
        )
    print("=" * 50)

    import os
    os.makedirs(args.out_file.parent, exist_ok=True) if args.out_file else None
    out_path = args.out_file or Path(str(args.input).replace(".jsonl", "_iou_results.jsonl"))
    with out_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    if failed_records:
        with args.failed.open("w", encoding="utf-8") as handle:
            for record in failed_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
