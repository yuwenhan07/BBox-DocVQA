#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import threading
import glob
import shlex

PRINT_LOCK = threading.Lock()

def log(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel read JSONL, convert to paper commands and download arXiv PDFs"
    )
    p.add_argument("--jsonl", required=True, help="Input jsonl file path")
    p.add_argument("--out_base", required=True,
                   help="Output root directory, e.g. ../../data/arxiv_pdf_725")
    p.add_argument("--paper_cmd", default="paper",
                   help="Paper command name or absolute path, default 'paper'")
    p.add_argument("--paper_threads", "-n", type=int, default=8,
                   help="The -n parameter passed to paper (internal parallelism of single task), default 8")
    p.add_argument("--workers", type=int, default=16,
                   help="Number of concurrent download tasks (thread pool size), default 16")
    p.add_argument("--retries", type=int, default=3,
                   help="Number of retries on failure, default 3")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout for single download task (seconds), default 600")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip if target PDF already exists")
    p.add_argument("--dry_run", action="store_true",
                   help="Only print commands to execute, do not actually run")
    return p.parse_args()

def read_jsonl(jsonl_path: str) -> List[Dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"[WARN] JSON parsing failed at line {ln}: {e}")
                continue
            items.append(obj)
    return items

def build_task_records(items: List[Dict]) -> Dict[str, Tuple[str, Dict]]:
    """
    Returns {paper_id: (category, raw_record)}, only keep the first record for the same paper_id
    """
    dedup: Dict[str, Tuple[str, Dict]] = {}
    for obj in items:
        paper_id = str(obj.get("paper_id") or obj.get("doc_id") or "").strip()
        if not paper_id:
            # Try to extract arXiv id from doc_id (like '2509.24220_...')
            doc_id = str(obj.get("doc_id") or "").strip()
            if doc_id and "_" in doc_id:
                paper_id = doc_id.split("_", 1)[0]
        if not paper_id:
            log(f"[WARN] Skipping one record: no paper_id/doc_id found -> {obj}")
            continue
        category = str(obj.get("category") or "misc").strip()
        if paper_id not in dedup:
            dedup[paper_id] = (category, obj)
    return dedup

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def predict_pdf_path(out_dir: Path, paper_id: str) -> Optional[Path]:
    """
    Predict the filename of the downloaded PDF.
    Here we assume the tool names it as <paper_id>.pdf; if your tool names differently,
    you can switch to glob matching: any pdf file starting with paper_id is considered existing.
    """
    candidate = out_dir / f"{paper_id}.pdf"
    if candidate.exists():
        return candidate

    # Fallback: if tool naming is uncertain, try glob to find pdf starting with paper_id
    matches = list(out_dir.glob(f"{paper_id}*.pdf"))
    if matches:
        return matches[0]
    return candidate  # Return default predicted path (for logging)

def build_command(paper_cmd: str, paper_id: str, out_dir: Path, paper_threads: int) -> List[str]:
    # Build command: paper <paper_id> -p -n <paper_threads> -d <out_dir>
    return [paper_cmd, paper_id, "-p", "-n", str(paper_threads), "-d", str(out_dir)]

def run_one(cmd: List[str], timeout: int) -> Tuple[int, str, str]:
    """
    Run a single external command, return (returncode, stdout, stderr)
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"Command not found: {shlex.join(cmd)}"
    except Exception as e:
        return 1, "", f"Exception: {repr(e)}"

def worker(task: Tuple[str, str, Path, List[str], bool, int, int]) -> Tuple[str, bool, str]:
    """
    Single task (for thread pool)
    Returns: (paper_id, success, message)
    """
    paper_id, category, out_dir, cmd, skip_existing, retries, timeout = task

    ensure_dir(out_dir)
    pdf_path = predict_pdf_path(out_dir, paper_id)

    if skip_existing and pdf_path and pdf_path.exists():
        return paper_id, True, f"Exists, skipped -> {pdf_path}"

    attempt = 0
    backoff = 2.0
    while attempt <= retries:
        attempt += 1
        log(f"[{paper_id}] Starting download (attempt {attempt}/{retries+1}): {shlex.join(cmd)}")
        rc, so, se = run_one(cmd, timeout=timeout)
        # Success: returncode==0, or even if non-zero but PDF exists in target directory
        if rc == 0 or (pdf_path and pdf_path.exists()):
            return paper_id, True, f"Success (rc={rc}) -> {pdf_path if pdf_path else out_dir}"
        err = se.strip() or so.strip() or f"unknown error (rc={rc})"
        if attempt <= retries:
            log(f"[{paper_id}] Failed: {err}; retrying after {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            return paper_id, False, f"Failed (retried {retries} times): {err}"

def main():
    args = parse_args()

    items = read_jsonl(args.jsonl)
    if not items:
        log("[ERROR] Input jsonl is empty or failed to parse")
        sys.exit(1)

    tasks_map = build_task_records(items)
    if not tasks_map:
        log("[ERROR] No paper_id extracted")
        sys.exit(1)

    # Prepare tasks
    tasks: List[Tuple[str, str, Path, List[str], bool, int, int]] = []
    for paper_id, (category, _raw) in tasks_map.items():
        out_dir = Path(args.out_base) / category
        cmd = build_command(args.paper_cmd, paper_id, out_dir, args.paper_threads)
        tasks.append((paper_id, category, out_dir, cmd, args.skip_existing, args.retries, args.timeout))

    log(f"[INFO] Number of papers to process: {len(tasks)}")
    if args.dry_run:
        for _, category, out_dir, cmd, *_ in tasks:
            print(shlex.join(cmd))
        return

    # Execute in parallel
    ok, fail = 0, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, t): t[0] for t in tasks}
        for fut in cf.as_completed(futures):
            paper_id = futures[fut]
            try:
                _pid, success, msg = fut.result()
            except Exception as e:
                success, msg = False, f"Exception: {repr(e)}"
            if success:
                ok += 1
                log(f"[OK] {paper_id}: {msg}")
            else:
                fail += 1
                log(f"[FAIL] {paper_id}: {msg}")

    log(f"[DONE] Success {ok}, Fail {fail}, Total {len(tasks)}")
    if fail > 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
