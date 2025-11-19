#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from boundingdoc.data_transform import transform_jsonl_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform and merge QA JSONL files")
    parser.add_argument("--input_dir", required=True, type=Path, help="Root directory; recursively scans for *.jsonl files")
    parser.add_argument("--output", required=True, type=Path, help="Output file path (JSONL)")
    parser.add_argument("--type_map", type=Path, help="Optional JSON file mapping evidence.image to type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform_jsonl_tree(args.input_dir, args.output, args.type_map)


if __name__ == "__main__":
    main()
