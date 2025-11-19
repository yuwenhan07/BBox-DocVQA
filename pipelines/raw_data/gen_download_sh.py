from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _find_first(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """Return the first available key from the dict."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_paper_id(doc_id: Optional[str]) -> Optional[str]:
    """Extract the paper_id (part before the underscore)."""
    if not doc_id or not isinstance(doc_id, str):
        return None
    return doc_id.split("_", 1)[0]


def collect_unique_entries(jsonl_path: Path) -> Dict[str, str]:
    """Return {doc_id: category}, defaulting to 'cs' if missing."""
    uniq: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                print(f"[WARN] {jsonl_path}:{ln} Skipping bad line: {e}", file=sys.stderr)
                continue
            doc_id = _find_first(obj, ("doc_id", "doc_name", "docid"))
            if not isinstance(doc_id, str):
                continue
            category = _find_first(obj, ("category",)) or "cs"
            if doc_id not in uniq:
                uniq[doc_id] = str(category)
    return uniq


def main(jsonl_file: str, out_root: str = "download", output_file: str = "download.sh") -> None:
    """Generate a shell script with all paper download commands."""
    path = Path(jsonl_file)
    if not path.exists():
        print(f"[ERROR] JSONL file not found: {jsonl_file}")
        sys.exit(1)

    uniq_map = collect_unique_entries(path)
    if not uniq_map:
        print("[WARN] No doc_id found.")
        return

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        count = 0
        for doc_id, category in uniq_map.items():
            pid = parse_paper_id(doc_id)
            if not pid:
                continue
            out_dir = Path(out_root) / category
            cmd = f"paper {pid} -p -n 8 -d {out_dir}"
            f.write(cmd + "\n")
            count += 1

    print(f"[INFO] Saved {count} commands to {output_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_download_sh.py <jsonl_path> [out_root] [output_file]")
        sys.exit(1)

    # Optional args: <jsonl> [out_root] [output_file]
    args = list(sys.argv[1:])
    while len(args) < 3:
        args.append(None)
    jsonl_file, out_root, output_file = args[0], args[1] or "download", args[2] or "download.sh"

    main(jsonl_file, out_root, output_file)