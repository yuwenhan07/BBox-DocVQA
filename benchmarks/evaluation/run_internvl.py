import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image as lmdeploy_load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
from tqdm import tqdm

from PROMPT import SYSTEM_PROMPT_IMAGE, SYSTEM_PROMPT_IMAGE_BBOX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InternVL inference on BoundingDoc benchmark entries using LMDeploy."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to write the JSONL predictions.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing benchmark_80_png/. Defaults to ../../data.",
    )
    parser.add_argument(
        "--mode",
        choices=["crops_answer", "page_answer", "pages_bbox_answer"],
        default="crops_answer",
        help=(
            "Use cropped subimages ('crops_answer'), full pages with bbox prompt ('pages_bbox_answer'), "
            "or full pages answered with the subimage prompt ('page_answer')."
        ),
    )
    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL3-78B",
        help="InternVL checkpoint identifier.",
    )
    parser.add_argument(
        "--chat-template",
        default="internvl2_5",
        help="Chat template name passed to LMDeploy ChatTemplateConfig.",
    )
    parser.add_argument(
        "--chat-tokenizer",
        default=None,
        help="Optional path passed to ChatTemplateConfig.model_path (defaults to --model when it points to a local directory).",
    )
    parser.add_argument("--session-len", type=int, default=16384, help="Session length for Turbomind backend.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of samples per LMDeploy call.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N records (<=0 means all).")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N records.")
    parser.add_argument("--cuda", default="0,1,2,3", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each success.")
    parser.add_argument(
        "--failed-out",
        type=Path,
        default=None,
        help="Optional path to store indices that failed generation.",
    )
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size for Turbomind backend.")
    return parser.parse_args()


def log_debug(index: int, prompt: str, image_paths: Sequence[Path]) -> None:
    label = f"[DEBUG] sample {index}"
    tqdm.write(f"{label} prompt:\n{prompt}", file=sys.stderr)
    if image_paths:
        joined = "\n  ".join(str(path) for path in image_paths)
        tqdm.write(f"{label} images:\n  {joined}", file=sys.stderr)
    else:
        tqdm.write(f"{label} images: <none>", file=sys.stderr)


def resolve_data_root(arg_path: Optional[Path]) -> Path:
    if arg_path is not None:
        root = arg_path.expanduser().resolve()
    else:
        root = (Path(__file__).resolve().parent.parent / "data").resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    if not (root / "benchmark_80_png").exists():
        raise FileNotFoundError(f"Expected benchmark_80_png/ under {root}")
    return root


def load_jsonl(path: Path, start: int, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def normalize_page_ids(example: Dict[str, Any]) -> List[int]:
    page_ids = example.get("evidence_page")
    if page_ids is None:
        page_ids = example.get("page_ids")
    if page_ids is None:
        page_ids = example.get("page_id")
    if page_ids is None:
        return []
    if isinstance(page_ids, int):
        return [page_ids]
    return [int(pid) for pid in page_ids]


def _align_groups(groups: List[List[int]], target_len: int) -> List[List[int]]:
    if len(groups) < target_len:
        groups = groups + [[] for _ in range(target_len - len(groups))]
    elif len(groups) > target_len:
        groups = groups[:target_len]
    return groups


def find_doc_dir(root: Path, category: str, doc_id: str) -> Path:
    category_dir = root / "benchmark_80_png" / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Missing category directory: {category_dir}")
    candidates = [
        folder
        for folder in category_dir.iterdir()
        if folder.is_dir() and folder.name.startswith(f"{doc_id}")
    ]
    if not candidates:
        raise FileNotFoundError(f"No directory found for doc_id={doc_id} in {category_dir}")
    candidates.sort(key=lambda p: len(p.name))
    return candidates[0]


def page_paths(root: Path, example: Dict[str, Any]) -> List[Path]:
    category = str(example["category"])
    doc_id = str(example.get("doc_id") or example.get("doc_name"))
    page_ids: Sequence[int] = normalize_page_ids(example)
    doc_dir = find_doc_dir(root, category, doc_id)
    paths: List[Path] = []
    for page in page_ids:
        path = doc_dir / f"{doc_id}_{int(page)}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing page image: {path}")
        paths.append(path)
    return paths


def _extract_crop_paths(page_paths_list: Iterable[Path], example: Dict[str, Any]) -> List[Path]:
    raw_bboxes = example.get("bbox")
    if not isinstance(raw_bboxes, list):
        raw_bboxes = []

    groups: List[List[Any]] = []
    for boxes in raw_bboxes:
        if not boxes:
            groups.append([])
            continue
        if isinstance(boxes, (list, tuple)):
            groups.append([box for box in boxes if isinstance(box, (list, tuple))])
        else:
            groups.append([])

    page_paths_seq = list(page_paths_list)
    aligned_boxes = _align_groups(groups, len(page_paths_seq))

    save_dir = Path("crops_output")
    save_dir.mkdir(parents=True, exist_ok=True)

    category = str(example.get("category", "unknown"))
    doc_id = str(example.get("doc_id") or example.get("doc_name") or "unknown")
    timestamp = int(time.time() * 1000)

    saved_paths: List[Path] = []
    counter = 0
    for page_path, boxes in zip(page_paths_seq, aligned_boxes):
        if not boxes:
            continue
        image = Image.open(page_path).convert("RGB")
        width, height = image.size
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            try:
                left, top, right, bottom = [int(round(float(coord))) for coord in box[:4]]
            except (TypeError, ValueError):
                continue
            left = max(0, min(left, width))
            top = max(0, min(top, height))
            right = max(0, min(right, width))
            bottom = max(0, min(bottom, height))
            if right <= left or bottom <= top:
                continue
            crop_path = save_dir / f"{category}_{doc_id}_{timestamp}_{counter}.png"
            counter += 1
            image.crop((left, top, right, bottom)).save(crop_path)
            saved_paths.append(crop_path)

    if saved_paths:
        return saved_paths
    return page_paths_seq


def collect_image_paths(page_paths_list: List[Path], example: Dict[str, Any], mode: str) -> List[Path]:
    if mode == "crops_answer":
        return _extract_crop_paths(page_paths_list, example)
    return page_paths_list


def build_prompt(example: Dict[str, Any], mode: str) -> str:
    question = example.get("query", "").strip()
    prompt_lines: List[str] = []
    if mode in ("crops_answer", "page_answer"):
        prompt_lines.append(SYSTEM_PROMPT_IMAGE.strip())
    else:
        prompt_lines.append(SYSTEM_PROMPT_IMAGE_BBOX.strip())
    if question:
        prompt_lines.append(f"Query: {question}")
    return "\n\n".join(prompt_lines)


ImageInput = Union[Any, List[Any]]


def _load_for_lmdeploy(path: Path) -> Any:
    return lmdeploy_load_image(str(path))


def build_pipe_request(image_paths: List[Path], prompt: str) -> Tuple[str, ImageInput]:
    if not image_paths:
        return prompt, []
    if len(image_paths) == 1:
        return prompt, _load_for_lmdeploy(image_paths[0])
    numbered = [f"Image-{idx}: {IMAGE_TOKEN}" for idx in range(1, len(image_paths) + 1)]
    numbered.append(prompt)
    prompt_with_tokens = "\n".join(numbered)
    images = [_load_for_lmdeploy(path) for path in image_paths]
    return prompt_with_tokens, images


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def extract_text(output: Any) -> str:
    if output is None:
        return ""
    if hasattr(output, "text"):
        return getattr(output, "text") or ""
    if isinstance(output, dict):
        text_val = output.get("text")
        if isinstance(text_val, str):
            return text_val
    return str(output)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[fatal] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data_root = resolve_data_root(args.data_root)
    device_list = [dev.strip() for dev in args.cuda.split(",") if dev.strip()]
    if args.tp > len(device_list):
        print(
            f"[warn] tensor_parallel_size ({args.tp}) exceeds number of visible devices ({len(device_list)}).",
            file=sys.stderr,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list)

    rows = load_jsonl(input_path, args.start_index, args.limit)
    if not rows:
        print("No records to process.")
        return

    backend_cfg = TurbomindEngineConfig(session_len=args.session_len, tp=args.tp)
    chat_cfg_kwargs: Dict[str, Any] = {"model_name": args.chat_template}
    template_path: Optional[str] = None
    if args.chat_tokenizer:
        template_candidate = Path(args.chat_tokenizer).expanduser()
        if template_candidate.exists():
            template_path = str(template_candidate.resolve())
    else:
        model_path = Path(args.model).expanduser()
        if model_path.exists():
            template_path = str(model_path.resolve())
    if template_path:
        chat_cfg_kwargs["model_path"] = template_path
    chat_cfg = ChatTemplateConfig(**chat_cfg_kwargs)
    pipe = pipeline(args.model, backend_config=backend_cfg, chat_template_config=chat_cfg)

    failures: List[int] = []
    total = len(rows)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        progress = tqdm(range(total), desc="InternVL inference", unit="sample")
        batch_inputs: List[Tuple[str, ImageInput]] = []
        batch_meta: List[Tuple[int, Dict[str, Any]]] = []

        for idx in progress:
            example = rows[idx]
            try:
                page_img_paths = page_paths(data_root, example)
                prompt_text = build_prompt(example, args.mode)
                image_paths = collect_image_paths(page_img_paths, example, args.mode)
                # log_debug(idx + args.start_index, prompt_text, image_paths)
                request = build_pipe_request(image_paths, prompt_text)
            except Exception as err:
                print(f"[WARN] Skipping index {idx} due to error: {err}", file=sys.stderr)
                failures.append(idx + args.start_index)
                continue

            batch_inputs.append(request)
            batch_meta.append((idx, example))

            if len(batch_inputs) >= args.batch_size:
                responses = ensure_list(pipe(batch_inputs))
                for (meta_idx, meta_example), response in zip(batch_meta, responses):
                    text = extract_text(response).strip()
                    if not text:
                        failures.append(meta_idx + args.start_index)
                        continue
                    out_row = {k: v for k, v in meta_example.items() if k != "summary"}
                    out_row["generate"] = text
                    out_row["model"] = args.model
                    out_f.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                out_f.flush()
                batch_inputs.clear()
                batch_meta.clear()

        if batch_inputs:
            responses = ensure_list(pipe(batch_inputs))
            for (meta_idx, meta_example), response in zip(batch_meta, responses):
                text = extract_text(response).strip()
                if not text:
                    failures.append(meta_idx + args.start_index)
                    continue
                out_row = {k: v for k, v in meta_example.items() if k != "summary"}
                out_row["generate"] = text
                out_row["model"] = args.model
                out_f.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if args.sleep > 0:
                    time.sleep(args.sleep)
            out_f.flush()

    if failures:
        failed_path = args.failed_out or output_path.with_suffix(output_path.suffix + ".failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"Finished with {len(failures)} failed index(es). See {failed_path}.")
    else:
        print("Finished without failures.")


if __name__ == "__main__":
    main()
