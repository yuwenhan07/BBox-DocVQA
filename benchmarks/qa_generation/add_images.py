import argparse
import base64
import io
import json
import os
import sys
from typing import List, Dict, Any


def b64_of_file(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def resolve_image_paths(row: Dict[str, Any], root: str) -> List[str]:
    """Given one JSON row, produce ordered list of image paths to encode.

    Expected keys: category (str), doc_id (str), page_ids (List[int]),
    crop_ids (List[List[int]]).
    """
    category = row.get('category')
    doc_id = row.get('doc_id')
    page_ids = row.get('page_ids') or row.get('page_id')
    crops_nested = row.get('crop_ids')

    if isinstance(page_ids, int):
        page_ids = [page_ids]
    if not isinstance(page_ids, list) or not isinstance(crops_nested, list):
        raise ValueError('Row missing page_ids/crop_ids in expected formats')

    if len(crops_nested) != len(page_ids):
        # Allow tolerant case: if crops is flat, pairwise zip
        if all(isinstance(x, int) for x in crops_nested):
            crops_nested = [[x] for x in crops_nested]
        else:
            raise ValueError(
                f"Length mismatch: page_ids has {len(page_ids)} items but crop_ids has {len(crops_nested)}"
            )

    base_dir = os.path.join(root, str(category), str(doc_id))
    paths: List[str] = []
    for p, crops in zip(page_ids, crops_nested):
        # normalize crops to list
        if isinstance(crops, int):
            crops = [crops]
        for c in crops:
            fname = f"{p}_{c}.png"
            paths.append(os.path.join(base_dir, fname))
    return paths


def process(input_path: str, root: str, output_path: str, strict: bool = False) -> None:
    missing_any = False
    total_rows = 0
    written_rows = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip invalid JSON line {total_rows}: {e}", file=sys.stderr)
                continue

            try:
                paths = resolve_image_paths(row, root)
            except Exception as e:
                print(f"[ERROR] Row {total_rows}: {e}", file=sys.stderr)
                if strict:
                    raise
                fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                written_rows += 1
                continue

            images_b64: List[str] = []
            for p in paths:
                if not os.path.isfile(p):
                    print(f"[WARN] Missing file: {p}", file=sys.stderr)
                    missing_any = True
                    if strict:
                        raise FileNotFoundError(p)
                    images_b64.append("")
                    continue
                try:
                    images_b64.append(b64_of_file(p))
                except Exception as e:
                    print(f"[WARN] Failed to encode {p}: {e}", file=sys.stderr)
                    if strict:
                        raise
                    images_b64.append("")

            # attach list in-order
            row['images_b64'] = images_b64
            fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            written_rows += 1

    print(f"[INFO] Processed rows: {total_rows}; written: {written_rows}", file=sys.stderr)
    if missing_any and strict:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Append base64-encoded crop images to JSONL rows.')
    parser.add_argument('--input', required=True, help='Path to input JSONL file')
    parser.add_argument('--root', required=False, default='/Users/yuwenhan/Documents/VLM/boundingdoc/data/benchmark_80_1_crops',
                        help='Root directory containing {category}/{doc_id}/{page_id}_{crop_id}.png')
    parser.add_argument('--out', required=False, default=None, help='Output JSONL path')
    parser.add_argument('--strict', action='store_true', help='Fail if any image files are missing')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.out or (os.path.splitext(input_path)[0] + '.with_images.jsonl')

    # Make sure we can write the parent directory of the output
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    process(input_path=input_path, root=args.root, output_path=output_path, strict=args.strict)


if __name__ == '__main__':
    main()
