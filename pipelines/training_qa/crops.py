import argparse
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop bounding boxes from images based on JSONL annotations."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Path to the input dataset root directory.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Path to the output directory where cropped images will be saved.",
    )
    parser.add_argument(
        "--jsonl_path",
        required=True,
        help="Path to JSONL file containing annotations.",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Path to output JSONL manifest listing cropped images (default: OUTPUT_ROOT/crops.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=128,
        help="Number of parallel threads to use (default: 128).",
    )
    return parser.parse_args()


# Helper: normalize various bbox schemas into a unified list of tuples
# Returns: List of (x1, y1, x2, y2, label, subimg_index_or_None)
def _normalize_bboxes(item: Dict[str, Any]) -> List[Tuple[int, int, int, int, str, Any]]:
    out: List[Tuple[int, int, int, int, str, Any]] = []
    label_default = item.get("label", "")
    subidx_default = item.get("subimg_index", None)
    bbox_field = item.get("bbox")

    if bbox_field is None:
        return out

    def _coerce_xyxy(xy):
        x1, y1, x2, y2 = xy
        return int(x1), int(y1), int(x2), int(y2)

    # Case A: bbox is a dict with 'xyxy' or 'x1'...'y2'
    if isinstance(bbox_field, dict):
        if "xyxy" in bbox_field:
            xy = bbox_field["xyxy"]
            x1, y1, x2, y2 = _coerce_xyxy(xy)
        else:
            x1 = bbox_field.get("x1") or bbox_field.get("left") or bbox_field.get("xmin")
            y1 = bbox_field.get("y1") or bbox_field.get("top") or bbox_field.get("ymin")
            x2 = bbox_field.get("x2") or bbox_field.get("right") or bbox_field.get("xmax")
            y2 = bbox_field.get("y2") or bbox_field.get("bottom") or bbox_field.get("ymax")
            if None in (x1, y1, x2, y2):
                return out
            x1, y1, x2, y2 = _coerce_xyxy([x1, y1, x2, y2])
        lbl = bbox_field.get("label", label_default)
        subidx = bbox_field.get("subimg_index", subidx_default)
        out.append((x1, y1, x2, y2, lbl if lbl is not None else "", subidx))
        return out

    # Case B: bbox is a flat list of 4 numbers: [x1, y1, x2, y2]
    if (
        isinstance(bbox_field, list)
        and len(bbox_field) == 4
        and all(isinstance(v, (int, float)) for v in bbox_field)
    ):
        x1, y1, x2, y2 = _coerce_xyxy(bbox_field)
        out.append((x1, y1, x2, y2, label_default, subidx_default))
        return out

    # Case C: bbox is a list of dicts or list of lists
    if isinstance(bbox_field, list) and len(bbox_field) > 0:
        first = bbox_field[0]
        # List of dicts
        if isinstance(first, dict):
            for d in bbox_field:
                if "xyxy" in d:
                    x1, y1, x2, y2 = _coerce_xyxy(d["xyxy"])
                else:
                    x1 = d.get("x1") or d.get("left") or d.get("xmin")
                    y1 = d.get("y1") or d.get("top") or d.get("ymin")
                    x2 = d.get("x2") or d.get("right") or d.get("xmax")
                    y2 = d.get("y2") or d.get("bottom") or d.get("ymax")
                    if None in (x1, y1, x2, y2):
                        continue
                    x1, y1, x2, y2 = _coerce_xyxy([x1, y1, x2, y2])
                lbl = d.get("label", label_default)
                subidx = d.get("subimg_index", subidx_default)
                out.append((x1, y1, x2, y2, lbl if lbl is not None else "", subidx))
            return out
        # List of lists: [[x1,y1,x2,y2], ...]
        if isinstance(first, (list, tuple)) and len(first) == 4:
            for xy in bbox_field:
                if not (isinstance(xy, (list, tuple)) and len(xy) == 4):
                    continue
                x1, y1, x2, y2 = _coerce_xyxy(xy)
                out.append((x1, y1, x2, y2, label_default, subidx_default))
            return out

    # Fallback: nothing recognized
    return out


def _resolve_doc_dir(
    category: str,
    doc_id: str,
    data_root: str,
    cache: Dict[Tuple[str, str], str],
) -> str:
    key = (category, doc_id)
    if key in cache:
        return cache[key]

    category_path = os.path.join(data_root, category)
    if not os.path.isdir(category_path):
        print(f"❌ Category path not found: {category_path}")
        cache[key] = ""
        return ""

    matched_dirs = [
        d
        for d in os.listdir(category_path)
        if d.startswith(doc_id) and os.path.isdir(os.path.join(category_path, d))
    ]
    if not matched_dirs:
        print(f"❌ No folder starting with {doc_id} found in {category_path}")
        cache[key] = ""
        return ""

    cache[key] = matched_dirs[0]
    return cache[key]


def prepare_tasks(
    jsonl_path: str,
    data_root: str,
    output_root: str,
) -> List[Dict[str, Any]]:
    doc_dir_cache: Dict[Tuple[str, str], str] = {}
    counters: Dict[Tuple[str, str, int], int] = defaultdict(int)
    tasks_map: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"❌ JSON parse error at line {line_num}: {err}")
                continue

            category = item.get("category")
            doc_id = item.get("doc_id")
            doc_full_name = item.get("doc_full_name", "")
            page_index_raw = item.get("page_index")

            if category is None or doc_id is None or page_index_raw is None:
                print(f"⚠️  Missing required fields at line {line_num}, skipping.")
                continue

            try:
                page_index_int = int(page_index_raw)
            except Exception:
                print(
                    f"⚠️  Invalid page_index for doc_id={doc_id} at line {line_num}, skipping."
                )
                continue
            page_index = page_index_int

            norm_bboxes = _normalize_bboxes(item)
            if not norm_bboxes:
                print(f"⚠️  No valid bbox for doc_id={doc_id}, page_index={page_index}")
                continue

            matched_dir = _resolve_doc_dir(category, doc_id, data_root, doc_dir_cache)
            if not matched_dir:
                continue

            image_path = os.path.join(
                data_root,
                category,
                matched_dir,
                f"{doc_id}_{page_index}.png",
            )
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                continue

            page_key = (category, doc_id, page_index)
            task_key = (category, doc_id, page_index, image_path)
            task = tasks_map.setdefault(
                task_key,
                {
                    "category": category,
                    "doc_id": doc_id,
                    "doc_full_name": doc_full_name,
                    "page_index": page_index,
                    "image_path": image_path,
                    "jobs": [],
                },
            )

            for tup in norm_bboxes:
                x1, y1, x2, y2, region_label, subidx = tup
                if subidx is None or subidx == "":
                    subidx = counters[page_key]
                    counters[page_key] += 1
                else:
                    try:
                        subidx = int(subidx)
                    except Exception:
                        print(
                            f"⚠️  Invalid subimg_index for doc_id={doc_id}, page_index={page_index} at line {line_num}; assigning automatically."
                        )
                        subidx = counters[page_key]
                        counters[page_key] += 1

                save_dir = os.path.join(output_root, category, doc_id)
                filename = f"{page_index}_{subidx}.png"
                save_path = os.path.join(save_dir, filename)
                record = {
                    "category": category,
                    "doc_id": doc_id,
                    "doc_full_name": doc_full_name,
                    "page_index": page_index,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "subimg_index": int(subidx),
                    "subimage_path": os.path.relpath(save_path, output_root).replace(
                        "\\", "/"
                    ),
                    "label": region_label if region_label is not None else "",
                }
                task["jobs"].append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "save_dir": save_dir,
                        "save_path": save_path,
                        "record": record,
                    }
                )

    return [task for task in tasks_map.values() if task["jobs"]]


def process_task(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    messages: List[str] = []
    image_path = task["image_path"]

    try:
        with Image.open(image_path) as image:
            for job in task["jobs"]:
                try:
                    x1, y1, x2, y2 = job["bbox"]
                    os.makedirs(job["save_dir"], exist_ok=True)
                    cropped = image.crop((x1, y1, x2, y2))
                    cropped.save(job["save_path"])
                    records.append(job["record"])
                    messages.append(f"✅ Saved: {job['save_path']}")
                except Exception as err:
                    messages.append(f"❌ Failed to save {job['save_path']}: {err}")
    except Exception as err:
        messages.append(f"❌ Failed to open {image_path}: {err}")

    return records, messages


def main():
    args = parse_args()
    data_root = args.input_root
    output_root = args.output_root
    jsonl_path = args.jsonl_path
    manifest_path = (
        args.manifest_path
        if args.manifest_path
        else os.path.join(output_root, "crops.jsonl")
    )

    os.makedirs(output_root, exist_ok=True)
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    tasks = prepare_tasks(jsonl_path, data_root, output_root)
    if not tasks:
        print("⚠️  No crops to process.")
        with open(manifest_path, "w", encoding="utf-8"):
            pass
        return

    manifest_lock = threading.Lock()

    with open(manifest_path, "w", encoding="utf-8") as mf:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(process_task, task) for task in tasks]
            for future in as_completed(futures):
                records, messages = future.result()
                for msg in messages:
                    print(msg)
                if records:
                    with manifest_lock:
                        for rec in records:
                            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
