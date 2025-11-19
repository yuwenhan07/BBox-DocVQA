import argparse
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info
from PROMPT import SYSTEM_PROMPT_IMAGE, SYSTEM_PROMPT_IMAGE_BBOX

try:
    from transformers import AutoProcessor as TransformersAutoProcessor  # type: ignore
except ImportError:
    TransformersAutoProcessor = None

try:
    from modelscope import AutoProcessor as ModelscopeAutoProcessor  # type: ignore
except ImportError:
    ModelscopeAutoProcessor = None


if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen-VL models (Qwen2.x / Qwen3) on BoundingDoc benchmark entries using vLLM."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to write the JSONL predictions.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing benchmark_80_png/ and benchmark_80_crops/. Defaults to ../../data.",
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
        default="/models/Qwen3-VL-Instruct",
        help="Local or Hugging Face Qwen-VL checkpoint path.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for vLLM.generate.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N records (<=0 means all).")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N records.")
    parser.add_argument("--cuda", default="0,1,2,3", help="CUDA_VISIBLE_DEVICES value for vLLM.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each success.")
    parser.add_argument("--save-raw", action="store_true", help="Keep raw model text in output under 'raw'.")
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="tensor_parallel_size for vLLM (set 0 to auto-detect from available CUDA devices).",
    )
    parser.add_argument(
        "--mm-encoder-tp-mode",
        type=str,
        default=None,
        help="Pass-through for vLLM LLM(..., mm_encoder_tp_mode=...). Required for some Qwen3 checkpoints.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism (useful for Qwen3 MoE checkpoints).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed passed to vLLM.",
    )
    parser.add_argument(
        "--processor-source",
        choices=["auto", "transformers", "modelscope"],
        default="auto",
        help="Where to load AutoProcessor from. 'auto' tries transformers first (needed for Qwen3) then modelscope.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="If supported, call processor.apply_chat_template(..., enable_thinking=False) to suppress <think> outputs.",
    )
    parser.add_argument(
        "--failed-out",
        type=Path,
        default=None,
        help="Optional path to store indices that failed generation.",
    )
    parser.add_argument(
        "--gpu-mem-utilization",
        type=float,
        default=0.8,
        help="Value forwarded to vLLM gpu_memory_utilization (fraction of GPU VRAM to use).",
    )
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


def load_processor(model_path: str, source: str):
    """
    Load AutoProcessor either from transformers (for Qwen3) or modelscope (legacy Qwen2.x).
    """
    failures: List[str] = []

    if source in ("auto", "transformers"):
        if TransformersAutoProcessor is None:
            failures.append("transformers.AutoProcessor not available in environment")
        else:
            try:
                return TransformersAutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception as exc:  # pragma: no cover - informative error path
                failures.append(f"transformers load failed: {exc}")
                if source == "transformers":
                    msg = "; ".join(failures)
                    raise RuntimeError(f"Unable to load processor via transformers: {msg}") from exc

    if source in ("auto", "modelscope"):
        if ModelscopeAutoProcessor is None:
            failures.append("modelscope.AutoProcessor not available in environment")
        else:
            try:
                return ModelscopeAutoProcessor.from_pretrained(model_path)
            except Exception as exc:  # pragma: no cover - informative error path
                failures.append(f"modelscope load failed: {exc}")
                if source == "modelscope":
                    msg = "; ".join(failures)
                    raise RuntimeError(f"Unable to load processor via modelscope: {msg}") from exc

    msg = "; ".join(failures) if failures else f"unknown source '{source}'"
    raise RuntimeError(f"Could not load AutoProcessor for {model_path}: {msg}")


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


def derive_crop_indices(example: Dict[str, Any], page_ids: Sequence[int]) -> List[List[int]]:
    crop_ids = example.get("bbox_crop_ids") or example.get("crop_ids")
    if crop_ids:
        # normalize nested structure to lists of ints
        result: List[List[int]] = []
        for crops in crop_ids:
            if isinstance(crops, int):
                result.append([int(crops)])
            else:
                result.append([int(c) for c in crops])
        return _align_groups(result, len(page_ids))

    bboxes = example.get("bbox")
    if not bboxes:
        return [[] for _ in page_ids]
    result: List[List[int]] = []
    for boxes in bboxes:
        if boxes is None:
            result.append([])
        else:
            result.append(list(range(len(boxes))))
    return _align_groups(result, len(page_ids))


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
        # print(f"[DEBUG] ================== page_path: {path} ==================")
        if not path.exists():
            raise FileNotFoundError(f"Missing page image: {path}")
        paths.append(path)
    return paths


def load_images(paths: Iterable[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        pil = Image.open(path).convert("RGB")
        images.append(pil)
    return images


def extract_crops_from_pages(
    page_images: Sequence[Image.Image],
    example: Dict[str, Any],
) -> List[Image.Image]:
    """
    Slice the page images using bounding boxes provided in the example.
    Falls back to returning the full pages if no valid boxes are present.
    """
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

    aligned_boxes = _align_groups(groups, len(page_images))

    crops: List[Image.Image] = []
    for image, boxes in zip(page_images, aligned_boxes):
        if not boxes:
            continue
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
            crops.append(image.crop((left, top, right, bottom)))

    if crops:
        # save crops by category and doc_id
        category = str(example.get("category", "unknown"))
        doc_id = str(example.get("doc_id") or example.get("doc_name") or "unknown")
        page_id = str(example.get("evidence_page") or "unknown")

        save_dir = Path("crops_output") / category / doc_id
        save_dir.mkdir(parents=True, exist_ok=True)

        ts_ms = int(time.time() * 1000)
        for i, crop_img in enumerate(crops, start=1):
            unique = f"{ts_ms}_{uuid4().hex[:8]}"
            filename = f"{doc_id}_{page_id}_{i}_{unique}.png"
            out_path = save_dir / filename
            # Extra safety: if by any chance this path exists, append an incrementing counter
            counter = 1
            while out_path.exists():
                filename = f"{doc_id}_{page_id}_{i}_{unique}_{counter}.png"
                out_path = save_dir / filename
                counter += 1
            crop_img.save(out_path)
        return crops
    return list(page_images)


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


def build_messages(images: List[Image.Image], prompt_text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def prepare_inputs(
    messages: List[Dict[str, Any]],
    processor: Any,
    chat_template_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chat_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if chat_template_overrides:
        try:
            apply_params = set(inspect.signature(processor.apply_chat_template).parameters)
        except (TypeError, ValueError):
            apply_params = set()
        for key, value in chat_template_overrides.items():
            if key in apply_params:
                chat_kwargs[key] = value
            else:
                tqdm.write(
                    f"[warn] apply_chat_template does not support argument '{key}', ignoring.",
                    file=sys.stderr,
                )

    prompt = processor.apply_chat_template(messages, **chat_kwargs)
    call_kwargs: Dict[str, Any] = {}
    try:
        signature_params = set(inspect.signature(process_vision_info).parameters)
    except (TypeError, ValueError):
        signature_params = set()
    if "image_patch_size" in signature_params:
        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
        if patch_size is not None:
            call_kwargs["image_patch_size"] = patch_size
    if "return_video_kwargs" in signature_params:
        call_kwargs["return_video_kwargs"] = True
    if "return_video_metadata" in signature_params:
        call_kwargs["return_video_metadata"] = True

    vision_info = process_vision_info(messages, **call_kwargs)

    image_inputs = None
    video_inputs = None
    video_kwargs: Dict[str, Any] = {}

    if isinstance(vision_info, tuple):
        if len(vision_info) == 3:
            image_inputs, video_inputs, video_kwargs = vision_info
        elif len(vision_info) == 2:
            image_inputs, video_inputs = vision_info
        elif len(vision_info) == 1:
            image_inputs = vision_info[0]
    elif isinstance(vision_info, dict):
        image_inputs = vision_info.get("image")
        video_inputs = vision_info.get("video")
        video_kwargs = vision_info.get("video_kwargs", {})

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def process_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    batch_inputs: List[Dict[str, Any]],
) -> List[str]:
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    results: List[str] = []
    for item in outputs:
        text = ""
        if item.outputs:
            text = item.outputs[0].text.strip()
        results.append(text)
    return results


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    # remove ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        # keep only the inside of the first fenced block
        m = re.match(r"```[a-zA-Z0-9_-]*\n(.*?)\n```", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
    return text

def _clean_json_like(text: str) -> str:
    # remove code fences first
    text = _strip_code_fences(text)
    # keep only from the first opening brace/bracket to the last closing one
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    first_arr = text.find("[")
    last_arr = text.rfind("]")
    span_candidates = []
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        span_candidates.append((first_obj, last_obj + 1))
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        span_candidates.append((first_arr, last_arr + 1))
    if span_candidates:
        start, end = min(span_candidates, key=lambda p: p[0])
        text = text[start:end]

    # remove // line comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    # remove /* block comments */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # collapse excessive whitespace
    return text.strip()

def _parse_json_like(text: str):
    """
    Try to coerce model output into valid JSON (object or array).
    Returns Python object on success, else None.
    """
    try:
        cleaned = _clean_json_like(text)
        if not cleaned:
            return None
        return json.loads(cleaned)
    except Exception:
        return None

def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[fatal] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data_root = resolve_data_root(args.data_root)
    device_list = [dev.strip() for dev in args.cuda.split(",") if dev.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list)

    visible_device_count = len(device_list)
    tp_size = args.tp
    auto_tp = tp_size <= 0
    if auto_tp:
        tp_size = max(visible_device_count, 1)
        try:
            import torch  # type: ignore

            detected = torch.cuda.device_count()
            if detected:
                tp_size = detected
        except Exception:
            pass
        tqdm.write(f"[info] Auto-detected tensor_parallel_size={tp_size}", file=sys.stderr)

    if visible_device_count and tp_size > visible_device_count:
        print(
            f"[warn] tensor_parallel_size ({tp_size}) exceeds number of visible devices ({visible_device_count}).",
            file=sys.stderr,
        )

    rows = load_jsonl(input_path, args.start_index, args.limit)
    if not rows:
        print("No records to process.")
        return

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    llm_kwargs: Dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": tp_size,
        "trust_remote_code": True,
    }
    mm_processor_kwargs = {}
    mm_processor_kwargs["max_pixels"] = 1280 * 28 * 28
    llm_kwargs["max_model_len"] = 20480
    llm_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
    llm_kwargs["gpu_memory_utilization"] = args.gpu_mem_utilization
    if args.mm_encoder_tp_mode:
        llm_kwargs["mm_encoder_tp_mode"] = args.mm_encoder_tp_mode
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed
    llm = LLM(**llm_kwargs)

    processor = load_processor(args.model, args.processor_source)
    tqdm.write(
        f"[info] Loaded processor ({processor.__class__.__module__}.{processor.__class__.__name__})",
        file=sys.stderr,
    )

    chat_template_overrides: Dict[str, Any] = {}
    if args.disable_thinking:
        chat_template_overrides["enable_thinking"] = False

    failures: List[int] = []
    total = len(rows)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        progress = tqdm(range(total), desc="Qwen inference", unit="sample")
        batch_inputs: List[Dict[str, Any]] = []
        batch_meta: List[Tuple[int, Dict[str, Any]]] = []

        for idx in progress:
            example = rows[idx]
            try:
                page_img_paths = page_paths(data_root, example)
                prompt_text = build_prompt(example, args.mode)
                # log_debug(idx + args.start_index, prompt_text, page_img_paths)
                if args.mode == "crops_answer":
                    page_images = load_images(page_img_paths)
                    images = extract_crops_from_pages(page_images, example)
                    del page_images
                else:
                    images = load_images(page_img_paths)
                messages = build_messages(images, prompt_text)
                request = prepare_inputs(messages, processor, chat_template_overrides)
            except Exception as err:
                print(f"[WARN] Skipping index {idx} due to error: {err}", file=sys.stderr)
                failures.append(idx + args.start_index)
                continue

            batch_inputs.append(request)
            batch_meta.append((idx, example))

            if len(batch_inputs) >= args.batch_size:
                results = process_batch(llm, sampling, batch_inputs)
                for (meta_idx, meta_example), text in zip(batch_meta, results):
                    if not text:
                        failures.append(meta_idx + args.start_index)
                        continue
                    out_row = {k: v for k, v in meta_example.items() if k != "summary"}
                    parsed = _parse_json_like(text)
                    if parsed is not None:
                        # store structured JSON if we could parse it
                        out_row["generate"] = parsed
                        if args.save_raw:
                            out_row["raw"] = text
                    else:
                        # fall back to the raw text
                        out_row["generate"] = text
                    out_row["model"] = args.model
                    out_f.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                out_f.flush()
                batch_inputs.clear()
                batch_meta.clear()

        if batch_inputs:
            results = process_batch(llm, sampling, batch_inputs)
            for (meta_idx, meta_example), text in zip(batch_meta, results):
                if not text:
                    failures.append(meta_idx + args.start_index)
                    continue
                out_row = {k: v for k, v in meta_example.items() if k != "summary"}
                parsed = _parse_json_like(text)
                if parsed is not None:
                    out_row["generate"] = parsed
                    if args.save_raw:
                        out_row["raw"] = text
                else:
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
