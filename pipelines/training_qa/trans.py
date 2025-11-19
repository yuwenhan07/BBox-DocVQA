#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from typing import Any, Dict, Optional, Tuple

def extract_doc_id(doc_name: str) -> str:
    """
    从 doc_name 提取 doc_id：取第一个下划线 '_' 之前的部分。
    例：'2509.24553_Focusing_the_Axion...' -> '2509.24553'
    """
    return doc_name.split("_", 1)[0] if "_" in doc_name else doc_name

def pick_first_bbox(bbox_field: Any) -> Optional[Tuple[int, int, int, int]]:
    """
    原始字段可能像 [[[x1,y1,x2,y2]]] 或 [[x1,y1,x2,y2]] 或 [x1,y1,x2,y2]
    统一取第一个框 [x1,y1,x2,y2]。
    """
    if bbox_field is None:
        return None
    v = bbox_field
    while isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
        v = v[0]
    if isinstance(v, list) and len(v) == 4 and all(isinstance(n, (int, float)) for n in v):
        return tuple(int(n) for n in v)  # type: ignore
    return None

def pick_first_page(pages: Any) -> Optional[int]:
    """
    evidence_page 可能是 [6] 或 ["6"] 等，统一取第一个并转 int。
    """
    if pages is None:
        return None
    if isinstance(pages, list) and len(pages) > 0:
        try:
            return int(pages[0])
        except Exception:
            return None
    try:
        return int(pages)
    except Exception:
        return None

def pick_label(subimg_type_field: Any) -> Optional[str]:
    """
    label 从 subimg_type/subimg_tpye（兼容拼写）中取第一个字符串；
    若为 [['image']] 或 ['image'] 或 'image' 都兼容。
    """
    v = subimg_type_field
    if v is None:
        return None
    while isinstance(v, list) and len(v) > 0:
        if isinstance(v[0], list):
            v = v[0]
        else:
            v = v[0]
            break
    if isinstance(v, str):
        return v
    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
        return v[0]
    return None

def convert_item(raw: Dict[str, Any], default_subimg_index: int = 0) -> Optional[Dict[str, Any]]:
    """
    将单条原始记录转换为目标格式；字段缺失会返回 None。
    需要的原始字段：
      - doc_name（必要）
      - evidence_page（必要，取第一个）
      - bbox（必要，取第一个框）
      - subimg_tpye / subimg_type（必要，取第一个）
      - category（必要）
    """
    doc_name = raw.get("doc_name")
    category = raw.get("category")
    evidence_page = raw.get("evidence_page")
    bbox_field = raw.get("bbox")
    subimg_type_field = raw.get("subimg_tpye", raw.get("subimg_type"))

    if not doc_name or not category:
        return None

    page_index = pick_first_page(evidence_page)
    bbox = pick_first_bbox(bbox_field)
    label = pick_label(subimg_type_field)

    if page_index is None or bbox is None or label is None:
        return None

    doc_id = extract_doc_id(doc_name)
    doc_full_name = doc_name
    # 这里仍保留默认 subimg_index，但稍后会被覆盖为分组计数后的值
    subimg_index = int(default_subimg_index)
    subimage_path = f"{category}/{doc_id}/{page_index}_{subimg_index}.png"

    x1, y1, x2, y2 = bbox

    return {
        "category": category,
        "doc_id": doc_id,
        "doc_full_name": doc_full_name,
        "page_index": page_index,
        "bbox": [x1, y1, x2, y2],
        "subimg_index": subimg_index,
        "subimage_path": subimage_path,
        "label": label
    }

def main():
    parser = argparse.ArgumentParser(description="批量将原始 jsonl 转换为目标格式（去掉 query/answer 等）。")
    parser.add_argument("--input", required=True, help="输入 .jsonl 文件路径")
    parser.add_argument("--output", required=True, help="输出 .jsonl 文件路径")
    parser.add_argument("--subimg-index", type=int, default=0, help="生成的 subimg_index（默认 0）")
    parser.add_argument("--skip-invalid", action="store_true", help="遇到缺字段的行直接跳过（默认报错终止）")
    parser.add_argument("--no-dedup", action="store_true", help="关闭去重；默认开启去重（按 category, doc_id, page_index, bbox, label 去重）")
    args = parser.parse_args()

    total = 0
    ok = 0
    bad = 0
    dup = 0
    seen = set()
    # 新增：为 (doc_id, page_index) 维护递增 subimg_index 的计数器
    group_counters = {}  # key: (doc_id, page_index) -> next_index(int)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                raw = json.loads(line)
            except Exception as e:
                bad += 1
                if args.skip_invalid:
                    continue
                print(f"[ERROR] 第 {total} 行不是合法 JSON：{e}", file=sys.stderr)
                sys.exit(1)

            converted = convert_item(raw, default_subimg_index=args.subimg_index)
            if converted is None:
                bad += 1
                if args.skip_invalid:
                    continue
                print(f"[ERROR] 第 {total} 行缺少必需字段或格式不合法：{raw}", file=sys.stderr)
                sys.exit(1)

            # 去重：默认按 (category, doc_id, page_index, bbox, label)
            if not args.no_dedup:
                key = (
                    converted.get("category"),
                    converted.get("doc_id"),
                    int(converted.get("page_index")),
                    tuple(converted.get("bbox", [])),
                    converted.get("label"),
                )
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)

            # —— 关键逻辑：在通过去重后才给出 subimg_index 递增编号 ——
            doc_id = converted["doc_id"]
            page_index = int(converted["page_index"])
            grp_key = (doc_id, page_index)  # 按你的要求仅按 (doc_id, page_index) 分组

            next_idx = group_counters.get(grp_key, 0)
            converted["subimg_index"] = next_idx
            # subimage_path 也需要用新的下标重写
            converted["subimage_path"] = f"{converted['category']}/{doc_id}/{page_index}_{next_idx}.png"
            group_counters[grp_key] = next_idx + 1
            # —— 关键逻辑结束 ——

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            ok += 1

    print(f"完成。总计 {total} 行，成功 {ok} 行，去重 {dup} 行，跳过/出错 {bad} 行。", file=sys.stderr)

if __name__ == "__main__":
    main()