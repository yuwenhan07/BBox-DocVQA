#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from boundingdoc.pdf import PdfToPngConfig, convert_pdfs_to_pngs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert PDF files into PNG images")
    parser.add_argument("pdfs", type=Path, help="PDF file or directory containing PDF files")
    parser.add_argument("output", type=Path, help="Output directory for the generated images")
    parser.add_argument("--dpi", type=int, default=300, help="Conversion DPI (default: 300)")
    parser.add_argument(
        "--thread-count",
        type=int,
        default=None,
        help="Per-PDF page rendering threads for pdf2image/pdftoppm",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of PDFs to convert concurrently",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = PdfToPngConfig(
            dpi=args.dpi,
            thread_count=args.thread_count,
            max_workers=args.max_workers,
        )
        generated = convert_pdfs_to_pngs(args.pdfs, args.output, config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    if not generated:
        print("WARNING: No PNG images were generated.")
        return
    root = args.output.resolve()
    print(f"SUCCESS: Generated {len(generated)} PNG images. Output directory: {root}")


if __name__ == "__main__":
    main()
