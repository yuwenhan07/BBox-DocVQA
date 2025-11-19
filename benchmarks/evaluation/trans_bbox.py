#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
from typing import Any, List, Union
from PIL import Image

Number = Union[int, float]


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    """Return resized (h_bar, w_bar) using the same logic as preprocessing."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {aspect_ratio}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def _is_box(x: Any) -> bool:
    return isinstance(x, list) and len(x) == 4 and all(isinstance(v, (int, float)) for v in x)


def _map_box_to_original(box: List[Number], scale_x: float, scale_y: float, W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    X1 = int(round(x1 * scale_x))
    Y1 = int(round(y1 * scale_y))
    X2 = int(round(x2 * scale_x))
    Y2 = int(round(y2 * scale_y))

    x_min, x_max = sorted((X1, X2))
    y_min, y_max = sorted((Y1, Y2))
    return [
        max(0, min(x_min, W)),
        max(0, min(y_min, H)),
        max(0, min(x_max, W)),
        max(0, min(y_max, H))
    ]


def _deep_map_bboxes(obj: Any, scale_x: float, scale_y: float, W: int, H: int) -> Any:
    if _is_box(obj):
        return _map_box_to_original(obj, scale_x, scale_y, W, H)
    if isinstance(obj, list):
        return [_deep_map_bboxes(el, scale_x, scale_y, W, H) for el in obj]
    return obj


def process_jsonl(dataroot: str, in_jsonl: str, out_jsonl: str,
                  factor: int = 28, min_pixels: int = 56 * 56,
                  max_pixels: int = 14 * 14 * 4 * 1280):
    with open(in_jsonl, "r", encoding="utf-8") as fin, open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                fout.write(raw + "\n")
                continue
            try:
                data = json.loads(raw)
            except Exception:
                fout.write(raw + "\n")
                continue

            try:
                category = data["category"]
                doc_id = data["doc_name"]
                pages = data.get("evidence_page", [])
                gen = data.get("generate", {})
                bbs = gen.get("bboxes", None)
                if bbs is None:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                # 读取任意一页图片（假设所有页尺寸一致）
                page = pages[0] if pages else None
                if page is None:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                img_path = os.path.join(dataroot, category, doc_id, f"{doc_id}_{page}.png")
                if not os.path.exists(img_path):
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                with Image.open(img_path) as im:
                    W, H = im.size

                h_bar, w_bar = smart_resize(H, W, factor=factor,
                                            min_pixels=min_pixels, max_pixels=max_pixels)
                scale_x = W / float(w_bar)
                scale_y = H / float(h_bar)

                mapped = _deep_map_bboxes(bbs, scale_x=scale_x, scale_y=scale_y, W=W, H=H)
                gen["bboxes"] = mapped
                data["generate"] = gen

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception:
                fout.write(raw + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map generate.bboxes from resized coords back to original using dataroot structure.")
    parser.add_argument("--dataroot", required=True, help="Root folder containing images: {category}/{doc_id}/{doc_id}_{page}.png")
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--factor", type=int, default=28)
    parser.add_argument("--min_pixels", type=int, default=56*56)
    parser.add_argument("--max_pixels", type=int, default=14*14*4*1280)
    args = parser.parse_args()

    process_jsonl(
        dataroot=args.dataroot,
        in_jsonl=args.input,
        out_jsonl=args.output,
        factor=args.factor,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )