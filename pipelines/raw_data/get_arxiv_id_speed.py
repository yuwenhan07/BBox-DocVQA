#!/usr/bin/env python3
"""
Batch fetch arXiv IDs by field and download with the `paper` CLI.

Examples
--------
# 1) Use defaults (several cs/* fields, 100 per field), save under ./downloads/<field>
python get_arxiv_id.py

# 2) Specify fields and base directory
python get_arxiv_id.py --fields cs.CV,cs.CL,cs.LG --per-field 100 --base-dir ~/PapersByField

# 3) Only fetch IDs (no download), print to stdout
python get_arxiv_id.py --dry-run

# 4) Download PDF only and use 4 threads for `paper`
python get_arxiv_id.py --paper-flags "-p -n 4"

python get_arxiv_id_speed.py --majors --per-field 10 --start-date 2025-07-01  --paper-flags "-p -n 4" --workers 20 --timeout 900 --skip-existing --base-dir ./pdfs

Notes
-----
- Fields should be valid arXiv categories (e.g., cs.CV, cs.CL, cs.LG, stat.ML, math.OC, physics.optics, etc.).
- Requires: requests, feedparser, and the `paper` CLI available in PATH.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
import concurrent.futures as cf
import threading
import glob
PRINT_LOCK = threading.Lock()

def log(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)
from datetime import datetime, date
from typing import List, Optional

import requests
import feedparser

ARXIV_API = "http://export.arxiv.org/api/query"
DEFAULT_FIELDS = [
    "cs.CV",   # Computer Vision and Pattern Recognition
    "cs.CL",   # Computation and Language (NLP)
    "cs.LG",   # Machine Learning
    "cs.AI",   # Artificial Intelligence
    "stat.ML", # Statistics - Machine Learning
    "math.OC", # Optimization and Control
]

# Major (top-level) arXiv categories
MAJOR_CATEGORIES = [
    "physics",  # includes many physics-related subcategories
    "math",
    "cs",
    "q-bio",
    "q-fin",
    "stat",
    "eess",
    "econ",
]

# Mapping from major categories to concrete subcategories for robust fetching
MAJOR_TO_SUBCATS = {
    "physics": [
        # Classic physics umbrella categories
        "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
        "math-ph", "nlin", "nucl-ex", "nucl-th", "quant-ph",
        # physics.* specific topical categories
        "physics.acc-ph", "physics.app-ph", "physics.ao-ph", "physics.atm-clus",
        "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
        "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn",
        "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det",
        "physics.med-ph", "physics.optics", "physics.soc-ph", "physics.space-ph",
        # Plasma physics (spelling varies historically, include both common forms)
        "physics.plasm-ph", "physics.plasma-ph",
    ],
    "math": [
        "math.AG","math.AT","math.AP","math.CT","math.CA","math.CO","math.CV","math.DG",
        "math.DS","math.FA","math.GM","math.GN","math.GT","math.GR","math.HO","math.IT",
        "math.KT","math.LO","math.MG","math.NA","math.NT","math.OA","math.OC","math.PR",
        "math.QA","math.RA","math.RT","math.SP","math.ST","math.SG",
    ],
    "cs": [
        "cs.AI","cs.CL","cs.CC","cs.CE","cs.CG","cs.GT","cs.CV","cs.CY","cs.CR","cs.DS",
        "cs.DB","cs.DL","cs.DM","cs.DC","cs.ET","cs.FL","cs.GL","cs.GR","cs.AR","cs.HC",
        "cs.IR","cs.IT","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS","cs.NA","cs.NE","cs.NI",
        "cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC","cs.SD","cs.SE","cs.SI","cs.SY",
    ],
    "q-bio": [
        "q-bio.BM","q-bio.CB","q-bio.GN","q-bio.MN","q-bio.NC","q-bio.OT",
        "q-bio.PE","q-bio.QM","q-bio.SC","q-bio.TO",
    ],
    "q-fin": [
        "q-fin.CP","q-fin.EC","q-fin.GN","q-fin.MF","q-fin.PM","q-fin.PR","q-fin.RM","q-fin.ST","q-fin.TR",
    ],
    "stat": [
        "stat.AP","stat.CO","stat.ME","stat.ML","stat.OT","stat.TH",
    ],
    "eess": [
        "eess.AS","eess.IV","eess.SP","eess.SY",
    ],
    "econ": [
        "econ.EM","econ.GN","econ.TH",
    ],
}
def _fetch_arxiv_entries(field: str, n: int = 100, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[tuple[str, Optional[date]]]:
    """Fetch up to n entries for a subcategory, returning (id, published_date)."""
    results: List[tuple[str, Optional[date]]] = []
    start_index = 0
    page_size = min(max(n, 100), 300)
    headers = {"User-Agent": "paper-batch-downloader/1.1 (mailto:example@example.com)"}
    while len(results) < n:
        params = {
            "search_query": f"cat:{field}",
            "start": start_index,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            r = requests.get(ARXIV_API, params=params, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"[ERROR] arXiv request failed for {field}: {e}", file=sys.stderr)
            break
        feed = feedparser.parse(r.text)
        entries = getattr(feed, "entries", [])
        if not entries:
            break
        for entry in entries:
            arxiv_id = entry.id.split("/abs/")[-1]
            if "v" in arxiv_id:
                base, _, _ = arxiv_id.partition("v")
                arxiv_id = base
            pub_d: Optional[date] = None
            pub_str = getattr(entry, "published", "") or getattr(entry, "updated", "")
            try:
                pub_dt = datetime.strptime(pub_str[:19], "%Y-%m-%dT%H:%M:%S")
                pub_d = pub_dt.date()
            except Exception:
                pub_d = None
            if start_date and pub_d and pub_d < start_date:
                continue
            if end_date and pub_d and pub_d > end_date:
                continue
            results.append((arxiv_id, pub_d))
            if len(results) >= n:
                break
        start_index += page_size
        if start_index > 5000:
            print(f"[WARN] Reached paging cap for {field}; collected {len(results)} entries so far.")
            break
    return results[:n]

def fetch_arxiv_ids_for_major(major: str, n: int = 100, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[str]:
    """Expand a major category into its subcategories, merge results, deduplicate, and return newest-first IDs."""
    subcats = MAJOR_TO_SUBCATS.get(major, [])
    if not subcats:
        # Fall back: try the major itself (may return empty on arXiv API)
        return fetch_arxiv_ids(major, n, start_date=start_date, end_date=end_date)

    # First pass: fetch a modest number from each subcat to ensure breadth
    per_sub_initial = max(25, (n // max(len(subcats), 1)) + 10)
    pool: List[tuple[str, Optional[date]]] = []
    seen: set[str] = set()

    for sub in subcats:
        entries = _fetch_arxiv_entries(sub, per_sub_initial, start_date=start_date, end_date=end_date)
        for _id, d in entries:
            if _id not in seen:
                seen.add(_id)
                pool.append((_id, d))
        if len(pool) >= n * 2:
            break

    # If still short, do a second pass (deeper) until we have enough
    if len(pool) < n:
        for sub in subcats:
            # Fetch a bit more per subcategory
            entries = _fetch_arxiv_entries(sub, per_sub_initial, start_date=start_date, end_date=end_date)
            for _id, d in entries:
                if _id not in seen:
                    seen.add(_id)
                    pool.append((_id, d))
            if len(pool) >= n * 3:
                break

    # Sort by date desc, unknown dates go last
    pool.sort(key=lambda x: (x[1] is None, x[1]), reverse=True)
    ids_sorted = [pid for pid, _d in pool[:n]]
    return ids_sorted


def fetch_arxiv_ids(field: str, n: int = 100, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[str]:
    """Fetch up to n arXiv IDs for a given category (field), newest first.
    
    If start_date/end_date are provided, filter entries by the 'published' date (inclusive).
    Date filtering is done client-side for robustness.
    """
    ids: List[str] = []
    start_index = 0
    # Use a reasonable page size to balance performance and politeness.
    page_size = min(max(n, 100), 300)  # between 100 and 300
    headers = {"User-Agent": "paper-batch-downloader/1.1 (mailto:example@example.com)"}

    while len(ids) < n:
        params = {
            "search_query": f"cat:{field}",
            "start": start_index,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            r = requests.get(ARXIV_API, params=params, headers=headers, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"[ERROR] arXiv request failed for {field}: {e}", file=sys.stderr)
            break

        feed = feedparser.parse(r.text)
        entries = getattr(feed, "entries", [])
        if not entries:
            break

        for entry in entries:
            arxiv_id = entry.id.split("/abs/")[-1]
            if "v" in arxiv_id:
                base, _, _ = arxiv_id.partition("v")
                arxiv_id = base

            # Date filtering (inclusive)
            if start_date or end_date:
                pub_str = getattr(entry, "published", "") or getattr(entry, "updated", "")
                try:
                    # example: '2025-01-23T17:54:02Z'
                    pub_dt = datetime.strptime(pub_str[:19], "%Y-%m-%dT%H:%M:%S")
                    pub_d = pub_dt.date()
                except Exception:
                    # If parsing fails, skip date filter for this entry
                    pub_d = None
                if start_date and pub_d and pub_d < start_date:
                    continue
                if end_date and pub_d and pub_d > end_date:
                    continue

            ids.append(arxiv_id)
            if len(ids) >= n:
                break

        start_index += page_size
        # Safety stop to avoid excessive paging in pathological cases
        if start_index > 5000:
            print(f"[WARN] Reached paging cap for {field}; collected {len(ids)} IDs so far.")
            break

    return ids[:n]


def chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_paper_download(ids: List[str], out_dir: str, paper_flags: str = "", retry: int = 1) -> None:
    if not ids:
        print("[WARN] No IDs to download.")
        return

    ensure_dir(out_dir)

    # `paper` can receive many IDs; to be safe, batch into chunks
    for batch in chunked(ids, 25):  # 25 IDs per call is a safe default
        cmd = ["paper", *batch, "-d", out_dir]
        if paper_flags:
            cmd.extend(shlex.split(paper_flags))
        attempt = 0
        while True:
            attempt += 1
            print(f"[INFO] Downloading {len(batch)} papers to {out_dir} (attempt {attempt})\n       $ {' '.join(shlex.quote(x) for x in cmd)}")
            proc = subprocess.run(cmd)
            if proc.returncode == 0:
                break
            if attempt > retry:
                print(f"[ERROR] `paper` failed after {retry} retries for batch: {batch}", file=sys.stderr)
                break
            # time.sleep(2)


# --- Parallel per-ID download helpers ---------------------------------------

def predict_pdf_path(out_dir: str, paper_id: str):
    """
    Try to predict or discover the resulting PDF path for a given paper_id.
    We first check `<paper_id>.pdf`, then fall back to glob `<paper_id>*.pdf`.
    Returns a string path (which may not exist yet).
    """
    candidate = os.path.join(out_dir, f"{paper_id}.pdf")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(out_dir, f"{paper_id}*.pdf"))
    if matches:
        return matches[0]
    return candidate


def build_command(paper_cmd: str, paper_id: str, out_dir: str, paper_flags: str):
    """Build a `paper` command for a single arXiv id."""
    cmd = [paper_cmd, paper_id, "-d", out_dir]
    if paper_flags:
        cmd.extend(shlex.split(paper_flags))
    return cmd


def run_one(cmd, timeout: int):
    """Run a single external command and return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"Command not found: {shlex.join(cmd)}"
    except Exception as e:
        return 1, "", f"Exception: {repr(e)}"


def worker(task):
    """
    Single download task executed in a thread pool.
    Task tuple: (paper_id, out_dir, cmd, skip_existing, retries, timeout)
    Returns: (paper_id, success_bool, message)
    """
    paper_id, out_dir, cmd, skip_existing, retries, timeout = task

    ensure_dir(out_dir)
    pdf_path = predict_pdf_path(out_dir, paper_id)

    if skip_existing and os.path.exists(pdf_path):
        return paper_id, True, f"Exists, skipped -> {pdf_path}"

    attempt = 0
    backoff = 2.0
    while attempt <= retries:
        attempt += 1
        log(f"[{paper_id}] Starting download (attempt {attempt}/{retries+1}): {shlex.join(cmd)}")
        rc, so, se = run_one(cmd, timeout=timeout)
        if rc == 0 or os.path.exists(pdf_path):
            return paper_id, True, f"Success (rc={rc}) -> {pdf_path}"
        err = (se or so or "").strip() or f"unknown error (rc={rc})"
        if attempt <= retries:
            log(f"[{paper_id}] Failed: {err}; retrying after {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            return paper_id, False, f"Failed (retried {retries} times): {err}"


def main():
    parser = argparse.ArgumentParser(description="Batch fetch arXiv IDs by field and download with `paper`.")
    parser.add_argument("--fields", type=str, default=",".join(DEFAULT_FIELDS),
                        help="Comma-separated arXiv categories (e.g., cs.CV,cs.CL,cs.LG)")
    parser.add_argument("--per-field", type=int, default=100, help="How many papers (IDs) per field")
    parser.add_argument("--base-dir", type=str, default=os.path.join(os.getcwd(), "downloads"),
                        help="Base directory to save; each field gets its own subfolder")
    parser.add_argument("--dry-run", action="store_true", help="Only print IDs; do not call `paper`.")
    parser.add_argument("--paper-flags", type=str, default="",
                        help="Extra flags to pass to `paper` (e.g., '-p -n 4 -s')")
    parser.add_argument("--retries", type=int, default=1, help="Retries per batch if `paper` fails")
    parser.add_argument("--majors", action="store_true",
                        help="Use top-level arXiv major categories (physics, math, cs, q-bio, q-fin, stat, eess, econ). Overrides --fields.")
    parser.add_argument("--start-date", type=str, default="",
                        help="Filter by inclusive start date (YYYY-MM-DD) on 'published' date.")
    parser.add_argument("--end-date", type=str, default="",
                        help="Filter by inclusive end date (YYYY-MM-DD) on 'published' date.")
    parser.add_argument("--paper-cmd", type=str, default="paper",
                        help="Paper command name or absolute path (default: 'paper')")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of concurrent download tasks (thread pool size), default 16")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout for a single download task in seconds, default 600")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if target PDF already exists in output folder")

    args = parser.parse_args()

    # Parse optional date filters
    start_date_obj: Optional[date] = None
    end_date_obj: Optional[date] = None
    if args.start_date:
        try:
            start_date_obj = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except Exception:
            print(f"[WARN] Ignoring invalid --start-date: {args.start_date}", file=sys.stderr)
    if args.end_date:
        try:
            end_date_obj = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except Exception:
            print(f"[WARN] Ignoring invalid --end-date: {args.end_date}", file=sys.stderr)

    # Decide which categories to use
    if args.majors:
        fields = MAJOR_CATEGORIES[:]
    else:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    ensure_dir(args.base_dir)

    parallel_tasks = []  # (paper_id, out_dir, cmd, skip_existing, retries, timeout)
    all_counts = []
    for field in fields:
        print(f"\n[INFO] Fetching up to {args.per_field} IDs for field: {field}")
        if args.majors:
            ids = fetch_arxiv_ids_for_major(field, args.per_field, start_date=start_date_obj, end_date=end_date_obj)
        else:
            ids = fetch_arxiv_ids(field, args.per_field, start_date=start_date_obj, end_date=end_date_obj)
        print(f"[INFO] Got {len(ids)} IDs for {field}")

        # Print IDs or prepare parallel tasks
        if args.dry_run:
            for _id in ids:
                out_dir = os.path.join(args.base_dir, field)
                cmd = build_command(args.paper_cmd, _id, out_dir, args.paper_flags)
                print(shlex.join(cmd))
        else:
            for _id in ids:
                out_dir = os.path.join(args.base_dir, field)
                cmd = build_command(args.paper_cmd, _id, out_dir, args.paper_flags)
                parallel_tasks.append((_id, out_dir, cmd, args.skip_existing, args.retries, args.timeout))
        all_counts.append((field, len(ids)))

    # Execute parallel downloads (if not dry-run)
    if not args.dry_run and parallel_tasks:
        log(f"[INFO] Total download tasks prepared: {len(parallel_tasks)} (workers={args.workers})")
        ok, fail = 0, 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(worker, t): t[0] for t in parallel_tasks}
            for fut in cf.as_completed(futures):
                pid = futures[fut]
                try:
                    _pid, success, msg = fut.result()
                except Exception as e:
                    success, msg = False, f"Exception: {repr(e)}"
                if success:
                    ok += 1
                    log(f"[OK] {pid}: {msg}")
                else:
                    fail += 1
                    log(f"[FAIL] {pid}: {msg}")
        log(f"[DONE] Success {ok}, Fail {fail}, Total {len(parallel_tasks)}")
        if fail > 0:
            sys.exit(2)

    print("\n[SUMMARY]")
    for field, count in all_counts:
        print(f"{field}: {count} IDs")
    print(f"Saved under: {args.base_dir}")


if __name__ == "__main__":
    main()
