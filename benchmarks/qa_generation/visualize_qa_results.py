#!/usr/bin/env python3
"""
Visualize BoundingDoc QA predictions with page overlays.

Given a JSONL file that contains QA outputs (e.g. gen_qa/*.jsonl), this script
draws the bounding boxes over the original page images stored under the
../../data/benchmark_80_1/ directory. The layout places the QA pair at the top,
renders each referenced page with its corresponding boxes, and shows the page
summary underneath.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# -----------------------------
# Data structures
# -----------------------------
@dataclass
class NormalizedSample:
    category: Optional[str]
    doc_id: str
    doc_full_name: Optional[str]
    pages: List[int]
    bboxes: List[List[Tuple[int, int, int, int]]]
    summaries: List[str]
    question: Optional[str]
    answer: Optional[str]
    raw: Dict


# -----------------------------
# JSON helpers
# -----------------------------
def load_jsonl(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"JSONL file is empty: {path}")
    return samples


def _as_box(box: Sequence) -> Tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"Invalid bbox: {box}")
    return tuple(int(round(v)) for v in box)


def _extract_boxes(raw) -> List[Tuple[int, int, int, int]]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        if len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
            return [_as_box(raw)]
        boxes: List[Tuple[int, int, int, int]] = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                boxes.extend(_extract_boxes(item))
            else:
                raise ValueError(f"Invalid bbox: {raw}")
        return boxes
    raise ValueError(f"Unsupported bbox type: {type(raw)}")


def normalize_sample(sample: Dict) -> NormalizedSample:
    if "doc_id" not in sample:
        raise ValueError("Missing doc_id field")

    raw_pages = sample.get("page_ids") or sample.get("page_id")
    if raw_pages is None:
        raise ValueError("Missing page_ids/page_id field")
    if isinstance(raw_pages, Iterable) and not isinstance(raw_pages, (str, bytes)):
        pages = [int(p) for p in raw_pages]
    else:
        pages = [int(raw_pages)]

    raw_bbox = sample.get("bbox")
    if raw_bbox is None:
        raise ValueError("Missing bbox field")

    bboxes: List[List[Tuple[int, int, int, int]]] = []
    if len(pages) == 1:
        bboxes = [_extract_boxes(raw_bbox)]
    else:
        if not isinstance(raw_bbox, Iterable):
            raise ValueError("Multi-page samples require a list of bbox entries")
        for page_boxes in raw_bbox:
            bboxes.append(_extract_boxes(page_boxes))
        if len(bboxes) != len(pages):
            raise ValueError("Number of pages does not match number of bbox entries")

    raw_summary = sample.get("summary") or sample.get("summaries") or []
    if isinstance(raw_summary, str):
        summaries = [raw_summary]
    elif isinstance(raw_summary, Iterable):
        summaries = [str(s) for s in raw_summary]
    else:
        summaries = []
    if not summaries:
        summaries = ["" for _ in pages]
    if len(summaries) != len(pages):
        if len(summaries) == 1:
            summaries = summaries * len(pages)
        else:
            summaries = (summaries + [""] * len(pages))[: len(pages)]

    return NormalizedSample(
        category=sample.get("category"),
        doc_id=str(sample["doc_id"]),
        doc_full_name=sample.get("doc_fullname") or sample.get("doc_full_name"),
        pages=pages,
        bboxes=bboxes,
        summaries=summaries,
        question=sample.get("q"),
        answer=sample.get("a"),
        raw=sample,
    )


# -----------------------------
# File resolution helpers
# -----------------------------
def resolve_doc_dir(
    data_root: Path,
    category: Optional[str],
    doc_id: str,
    doc_full_name: Optional[str],
) -> Path:
    base = data_root / category if category else data_root
    if not base.exists():
        raise FileNotFoundError(f"Category directory not found: {base}")

    if doc_full_name:
        candidate = base / doc_full_name
        if candidate.is_dir():
            return candidate

    entries = [p for p in base.iterdir() if p.is_dir()]
    exact = [p for p in entries if p.name == doc_id]
    if exact:
        return exact[0]

    starts = sorted([p for p in entries if p.name.startswith(f"{doc_id}_")])
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:
        # Choose the directory with most PNG files as heuristic
        starts.sort(key=lambda p: len(list(p.glob("*.png"))), reverse=True)
        return starts[0]

    flat_pngs = list(base.glob(f"{doc_id}_*.png"))
    if flat_pngs:
        return base

    raise FileNotFoundError(
        f"Could not locate document directory in {base}. Expected {doc_id} or {doc_id}_* folders."
    )


def load_page_image(
    data_root: Path,
    category: Optional[str],
    doc_id: str,
    doc_full_name: Optional[str],
    page_id: int,
) -> Tuple[Image.Image, Path]:
    doc_dir = resolve_doc_dir(data_root, category, doc_id, doc_full_name)
    patterns = [
        f"{doc_id}_{page_id}.png",
        f"{doc_id}_{str(page_id).zfill(2)}.png",
        f"{doc_id}_{str(page_id).zfill(3)}.png",
        f"{doc_id}_{str(page_id).zfill(4)}.png",
    ]
    for name in patterns:
        candidate = doc_dir / name
        if candidate.exists():
            return Image.open(candidate).convert("RGB"), candidate
    # fallback: any file ending with _{page_id}.png
    alternatives = [p for p in doc_dir.glob(f"*_{page_id}.png")]
    if alternatives:
        return Image.open(alternatives[0]).convert("RGB"), alternatives[0]
    raise FileNotFoundError(f"Page image missing: {doc_dir}/{doc_id}_{page_id}.png")


# -----------------------------
# Rendering
# -----------------------------
def _wrap_text(text: str, width: int = 70) -> str:
    wrapped = textwrap.fill(text.strip(), width=width) if text else ""
    return wrapped


def render_sample(
    sample: NormalizedSample,
    data_root: Path,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Path:
    num_pages = len(sample.pages)
    height_ratios = [0.9] + [5, 1.3] * num_pages
    fig = plt.figure(figsize=(14, 4 + num_pages * 5.5))
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)

    qa_ax = fig.add_subplot(gs[0, 0])
    qa_ax.axis("off")
    question = sample.question or "(question not provided)"
    answer = sample.answer or "(answer not provided)"
    title_parts = [
        f"Category: {sample.category or 'unknown'}",
        f"Doc: {sample.doc_id}",
        f"Pages: {', '.join(str(p) for p in sample.pages)}",
    ]
    qa_text = "\n".join(
        [
            " / ".join(title_parts),
            "",
            f"Q: {_wrap_text(question)}",
            "",
            f"A: {_wrap_text(answer)}",
        ]
    )
    qa_ax.text(0, 1, qa_text, va="top", ha="left", fontsize=11, family="monospace")

    for idx, page_id in enumerate(sample.pages):
        img, img_path = load_page_image(
            data_root=data_root,
            category=sample.category,
            doc_id=sample.doc_id,
            doc_full_name=sample.doc_full_name,
            page_id=page_id,
        )

        img_ax = fig.add_subplot(gs[1 + idx * 2, 0])
        img_ax.imshow(img)
        img_ax.set_title(f"Page {page_id} — {img_path.name}", fontsize=11)
        img_ax.axis("off")

        for (x1, y1, x2, y2) in sample.bboxes[idx]:
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            rect = Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=max(1, img.width // 800),
                edgecolor="red",
                facecolor="none",
            )
            img_ax.add_patch(rect)

        summary_ax = fig.add_subplot(gs[2 + idx * 2, 0])
        summary_ax.axis("off")
        summary_ax.text(
            0.5,
            1,
            _wrap_text(sample.summaries[idx]),
            va="top",
            ha="center",
            fontsize=10.5,
        )

    fig.tight_layout()

    if save_path is None:
        doc_slug = sample.raw.get("doc_full_name") or sample.doc_id
        safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in doc_slug)
        save_path = Path(f"{sample.category or 'unknown'}_{safe_slug}_{'-'.join(str(p) for p in sample.pages)}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# -----------------------------
# Filtering utilities
# -----------------------------
def filter_samples(
    samples: Iterable[Dict],
    categories: Optional[Sequence[str]],
    limit: Optional[int],
) -> Iterator[Tuple[int, Dict]]:
    seen = 0
    for idx, sample in enumerate(samples):
        cat = sample.get("category")
        if categories and cat not in categories:
            continue
        yield idx, sample
        seen += 1
        if limit is not None and seen >= limit:
            break


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BoundingDoc QA visualizer (Matplotlib edition)")
    parser.add_argument(
        "--qa-file",
        type=Path,
        required=True,
        help="JSONL file containing QA results, e.g., gen_qa/MPMC_test.jsonl",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../data/benchmark_80_1"),
        help="Root directory containing the original page PNGs (with category subdirectories)",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Only render the specified category; can be repeated, e.g., --category cs --category econ",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Render starting from the filtered sample at this index (0-based). Defaults to the first match.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to render (when used with --index, renders limit samples starting from that index).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Directory to store the rendered PNG files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save images without opening a window (useful for headless environments).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching samples without rendering.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    samples = load_jsonl(args.qa_file)
    filtered = list(filter_samples(samples, args.category, None))
    if not filtered:
        categories = ", ".join(args.category or ["(unspecified)"])
        raise SystemExit(f"No samples found for categories {categories}.")

    if args.list:
        for idx, sample in filtered:
            pages = sample.get("page_ids") or sample.get("page_id")
            if isinstance(pages, (list, tuple)):
                pages_disp = ",".join(str(p) for p in pages)
            else:
                pages_disp = str(pages)
            print(
                f"[{idx}] category={sample.get('category')} doc_id={sample.get('doc_id')} pages={pages_disp} type={sample.get('type')}"
            )
        return

    if args.index is not None:
        filtered = filtered[args.index :]
        if not filtered:
            raise SystemExit(f"Index {args.index} is out of range.")

    limit = args.limit if args.limit is not None else len(filtered)
    to_render = filtered[:limit]

    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in to_render:
        normalized = normalize_sample(sample)
        output_path = None
        if save_dir:
            doc_slug = normalized.doc_full_name or normalized.doc_id
            safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in doc_slug)
            filename = f"{normalized.category or 'unknown'}_{safe_slug}_{'-'.join(str(p) for p in normalized.pages)}.png"
            output_path = save_dir / filename

        result_path = render_sample(
            normalized,
            data_root=args.data_root,
            save_path=output_path,
            show=not args.no_show,
        )
        print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
