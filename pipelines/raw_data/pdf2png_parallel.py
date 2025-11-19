#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert PDFs to PNGs in parallel batches.
- Input: a directory containing many PDFs (recursive)
- Output: a folder named after each PDF file (without extension), images named sequentially starting from 1.png
- Uses multiprocessing to fully utilize multi-core CPUs
Dependencies:
    pip install pymupdf
Note:
    PyMuPDF (fitz) is usually faster and easier to install than calling external tools.
    You can modify to use pdftoppm if necessary.
"""

import argparse
import concurrent.futures as futures
import itertools
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# ---- Utility functions ----

def is_pdf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf"

def scan_pdfs(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*.pdf") if p.is_file()]
    else:
        return [p for p in root.glob("*.pdf") if p.is_file()]

def safe_dirname_from_pdf(pdf_path: Path) -> str:
    # Remove extension and replace unsafe characters with underscores
    name = pdf_path.stem
    # Replace illegal filename characters (try to be cross-platform)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = name.strip().strip(".")
    return name or "untitled_pdf"

def make_unique_dir(base_dir: Path) -> Path:
    """If directory name conflicts, append _1, _2, ... to ensure uniqueness"""
    if not base_dir.exists():
        return base_dir
    for i in itertools.count(1):
        candidate = base_dir.parent / f"{base_dir.name}_{i}"
        if not candidate.exists():
            return candidate

def convert_one_pdf(args: Tuple[Path, Path, Path, int, int, bool, bool]) -> Tuple[Path, int, str]:
    """
    Single PDF conversion task.
    Returns: (pdf_path, number of pages converted, error message or empty string)
    """
    pdf_path, input_root, out_root, dpi, jpeg_quality, overwrite, keep_empty = args
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        return (pdf_path, 0, f"Failed to import PyMuPDF: {e}. Please install first: pip install pymupdf")

    try:
        # Build output directory to mirror input_root's relative structure, then add a folder named after the PDF file
        folder_name = safe_dirname_from_pdf(pdf_path)
        try:
            rel_parent = pdf_path.relative_to(input_root).parent
        except Exception:
            # If pdf_path is not under input_root for some reason, fall back to no relative parent
            rel_parent = Path("")
        out_dir = (out_root / rel_parent / folder_name)

        if out_dir.exists() and overwrite:
            shutil.rmtree(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        # Calculate zoom matrix
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(pdf_path)
        num_pages = doc.page_count

        # Determine resume index if not overwriting
        existing = sorted([p for p in out_dir.glob("*.png") if p.is_file()], key=lambda p: p.name)
        start_idx = 1
        if existing and not overwrite:
            nums = []
            for p in existing:
                m = re.match(r"(\d+)\.png$", p.name)
                if m:
                    nums.append(int(m.group(1)))
            if nums:
                start_idx = max(nums) + 1

        saved = 0
        for i in range(num_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            idx = i + 1  # page number starts from 1
            out_file = out_dir / f"{idx}.png"
            if out_file.exists() and not overwrite:
                continue
            pix.save(out_file.as_posix())
            saved += 1

        doc.close()

        # Handle empty directories
        if not keep_empty:
            try:
                if not any(out_dir.iterdir()):
                    out_dir.rmdir()
            except Exception:
                pass

        return (pdf_path, saved, "")
    except Exception as e:
        return (pdf_path, 0, f"Processing failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch convert PDFs to PNGs in parallel")
    parser.add_argument("input_dir", type=Path, help="Directory containing PDFs (supports recursive search)")
    parser.add_argument("-o", "--out-root", type=Path, default=None,
                        help="Output root directory (default: same as input_dir)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse subdirectories, only process current directory")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300, higher is clearer but uses more memory/disk)")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel processes (default uses all CPU cores)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if target files already exist")
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty output folders even if no images are exported")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    out_root: Path = args.out_root or input_dir
    out_root.mkdir(parents=True, exist_ok=True)

    recursive = not args.no_recursive
    dpi = max(36, args.dpi)  # reasonable lower bound
    workers = max(1, args.workers)
    overwrite = bool(args.overwrite)
    keep_empty = bool(args.keep_empty)

    pdfs = scan_pdfs(input_dir, recursive=recursive)
    if not pdfs:
        print("No PDF files found in the specified directory.")
        return

    print(f"Found {len(pdfs)} PDF files")
    print(f"Output root directory: {out_root}")
    print(f"DPI: {dpi}, parallel processes: {workers}, overwrite: {overwrite}")
    sys.stdout.flush()

    # Assemble arguments
    job_args = [(p, input_dir, out_root, dpi, 90, overwrite, keep_empty) for p in pdfs]

    # Run multiprocessing
    ok = 0
    total_pages = 0
    failed = []
    with futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for pdf_path, saved, err in ex.map(convert_one_pdf, job_args):
            rel = pdf_path.relative_to(input_dir) if pdf_path.is_relative_to(input_dir) else pdf_path
            if err:
                failed.append((rel, err))
                print(f"[Failed] {rel} -> {err}")
            else:
                ok += 1
                total_pages += saved
                print(f"[Done] {rel} -> Exported {saved} images")

    print("-" * 60)
    print(f"Completed files: {ok}/{len(pdfs)}, total images exported: {total_pages}")
    if failed:
        print("Failure list:")
        for rel, err in failed:
            print(f"  - {rel}: {err}")


if __name__ == "__main__":
    main()

