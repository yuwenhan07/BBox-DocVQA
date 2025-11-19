import argparse
import ast
import base64
import inspect
import json
import multiprocessing as mp
import os
import re
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image
from modelscope import AutoProcessor
from tqdm import tqdm
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info

# Ensure CUDA-safe multiprocessing for vLLM (avoid fork with initialized CUDA)
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert-level question-and-answer generator. Based strictly on the visible content of ALL provided crop images "
    "(they may come from MULTIPLE PAGES), generate exactly ONE challenging, grounded question with its answer.\n"
    "Primary objective: The question must rely on fine-grained details visible ONLY in the crops; a reader using full pages "
    "without inspecting these precise crops should likely answer incorrectly.\n"
    "Rules:\n"
    "1) Use ONLY the crops' visible content—no outside knowledge or assumptions; do not infer text that is not clearly visible.\n"
    "2) Prefer questions that require integrating or comparing details across multiple crops (numbers, labels, legends, figure indices, footnotes, coordinates). "
    "If only one crop is available, make the question hinge on a subtle/localized detail within that crop.\n"
    "3) Avoid generic or high-level questions answerable from page-level context, titles, or common sense. The question should become easy ONLY if the crops are inspected carefully.\n"
    "4) Make the question as challenging and reasoning-intensive as possible while remaining fully grounded in the crops.\n"
    "5) The answer MUST be concise (no more than 10 words), precise, and directly supported by the crops.\n"
    "6) Output VALID JSON ONLY, in this exact format:\n"
    "{\n"
    "  \"q\": \"...\",\n"
    "  \"a\": \"...\",\n"
    "}\n"
    "7) If the crops lack enough information to form such a question, output exactly: "
    "{\"q\": \"\", \"a\": \"\"}.\n"
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def resolve_image_paths(row: Dict[str, Any], root: str) -> List[str]:
    """
    Determine absolute image paths for the row. Prefer explicit subimage_paths if present.
    Falls back to {root}/{category}/{doc_id}/{page_index}_{crop_id}.png.
    """
    paths: List[str] = []
    sub_paths = row.get("subimage_paths")
    category = row.get("category")
    doc_id = row.get("doc_id")

    if isinstance(sub_paths, list) and sub_paths:
        for group in sub_paths:
            if not isinstance(group, list):
                continue
            for rel in group:
                if not isinstance(rel, str):
                    continue
                abs_path = rel if os.path.isabs(rel) else os.path.join(root, rel)
                if isinstance(category, str):
                    rel_norm = os.path.relpath(abs_path, root).replace("\\", "/")
                    if not rel_norm.startswith(f"{category}/"):
                        raise ValueError(
                            f"Path category mismatch for doc_id={doc_id}: expected prefix {category}/, got {rel_norm}"
                        )
                paths.append(abs_path)
        if paths:
            return paths

    page_ids = row.get("page_ids") or row.get("page_id")
    crop_ids = row.get("crop_ids")
    if isinstance(page_ids, int):
        page_ids = [page_ids]
    if isinstance(crop_ids, int):
        crop_ids = [[crop_ids]]
    if not (isinstance(page_ids, list) and isinstance(crop_ids, list)):
        return paths
    if len(page_ids) != len(crop_ids):
        return paths

    base_dir = os.path.join(root, str(category), str(doc_id))
    for page, crops in zip(page_ids, crop_ids):
        if isinstance(crops, int):
            crops = [crops]
        for crop in crops:
            paths.append(os.path.join(base_dir, f"{page}_{crop}.png"))
    return paths


def load_images_from_paths(paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for idx, path in enumerate(paths, start=1):
        try:
            pil = Image.open(path).convert("RGB")
            images.append(pil)
        except Exception as err:
            print(f"[WARN] Failed to load image #{idx} ({path}): {err}", file=sys.stderr)
    return images


def decode_base64_images(images_b64: List[str]) -> List[Image.Image]:
    decoded: List[Image.Image] = []
    for idx, img_b64 in enumerate(images_b64, start=1):
        if not isinstance(img_b64, str) or not img_b64:
            continue
        try:
            img_bytes = base64.b64decode(img_b64)
            pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            decoded.append(pil)
        except Exception as err:
            print(f"[WARN] Failed to decode image #{idx}: {err}", file=sys.stderr)
    return decoded


def build_prompt_text(system_prompt: str, doc_id: Any, page_ids: Sequence[Any]) -> str:
    doc_str = str(doc_id) if doc_id is not None else ""
    pages_str = ", ".join(str(p) for p in page_ids) if page_ids else ""
    header_lines = []
    if doc_str:
        header_lines.append(f"Document ID: {doc_str}")
    if pages_str:
        header_lines.append(f"Pages covered: {pages_str}")
    header = "\n".join(header_lines)
    if header:
        header += "\n\n"
    return (
        f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{header}Use only the provided crop images to craft the question-answer pair."
    )


def build_conversation(images: List[Image.Image], prompt_text: str) -> List[Dict[str, Any]]:
    if not images:
        return []
    content: List[Dict[str, Any]] = []
    for pil in images:
        content.append({"type": "image", "image": pil})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video_kwargs: Dict[str, Any] = {}
    image_inputs = None
    video_inputs = None

    try:
        sig_params = set(inspect.signature(process_vision_info).parameters)
    except (TypeError, ValueError):
        sig_params = set()

    call_kwargs: Dict[str, Any] = {}
    if "image_patch_size" in sig_params:
        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
        if patch_size is not None:
            call_kwargs["image_patch_size"] = patch_size
    if "return_video_kwargs" in sig_params:
        call_kwargs["return_video_kwargs"] = True
    if "return_video_metadata" in sig_params:
        call_kwargs["return_video_metadata"] = True

    vision_info = process_vision_info(messages, **call_kwargs)
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
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def count_existing_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def extract_single_qa(json_text: str) -> Dict[str, Any]:
    def _parse_payload(candidate: str) -> Dict[str, Any]:
        candidate = candidate.strip()
        if not candidate:
            return {}
        try:
            return json.loads(candidate)
        except Exception:
            try:
                data = ast.literal_eval(candidate)
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}

    payload = _parse_payload(json_text)
    try:
        if not payload:
            raise ValueError
    except Exception:
        payload = {}
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", json_text, re.IGNORECASE)
        candidates = [*fenced] if fenced else [json_text]
        if fenced:
            candidates.append(json_text)

        seen = set()
        for text in candidates:
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            parsed = _parse_payload(text)
            if parsed:
                payload = parsed
                break

        if not payload:
            raw = json_text
            for match in re.finditer(r"\{", raw):
                start = match.start()
                depth = 0
                for pos in range(start, len(raw)):
                    char = raw[pos]
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = raw[start : pos + 1]
                            parsed = _parse_payload(candidate)
                            if parsed:
                                payload = parsed
                                break
                    if depth < 0:
                        break
                if payload:
                    break

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]

    if isinstance(payload, dict) and "q" in payload and "a" in payload:
        result: Dict[str, Any] = {
            "q": payload.get("q", ""),
            "a": payload.get("a", ""),
        }
        if "type" in payload:
            result["type"] = payload.get("type")
        return result
    if isinstance(payload, dict) and isinstance(payload.get("qas"), list) and payload["qas"]:
        qa0 = payload["qas"][0]
        if isinstance(qa0, dict) and "q" in qa0 and "a" in qa0:
            result = {
                "q": qa0.get("q", ""),
                "a": qa0.get("a", ""),
            }
            if "type" in qa0:
                result["type"] = qa0.get("type")
            return result
    return {}


def process_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    batch_inputs: List[Dict[str, Any]],
    batch_meta: List[Dict[str, Any]],
    out_f,
    keep_raw: bool,
    sleep_seconds: float,
) -> Tuple[int, List[int]]:
    if not batch_inputs:
        return 0, []

    try:
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    except Exception as err:
        print(f"[ERROR] vLLM generation failed for batch -> {err}", file=sys.stderr)
        return 0, [meta["index"] for meta in batch_meta]

    success = 0
    failed: List[int] = []
    for output, meta in zip(outputs, batch_meta):
        idx = meta["index"]
        raw_text = ""
        if output.outputs:
            raw_text = output.outputs[0].text.strip()
        if not raw_text:
            print(f"[WARN] Empty generation output at index {idx}", file=sys.stderr)
            failed.append(idx)
            continue

        qa = extract_single_qa(raw_text)
        if not qa:
            preview = raw_text.strip().replace("\n", " ")
            if len(preview) > 300:
                preview = preview[:300] + "..."
            print(f"[WARN] Failed to parse JSON at index {idx}. Preview: {preview}", file=sys.stderr)
            failed.append(idx)
            continue

        row = meta["row"]
        out_row: Dict[str, Any] = {
            "query": qa.get("q", ""),
            "answer": qa.get("a", ""),
        }

        doc_name = row.get("doc_name") or row.get("doc_id") or row.get("docId") or row.get("docid")
        if doc_name is not None:
            out_row["doc_name"] = str(doc_name)

        evidence_page = row.get("page_ids") or row.get("page_id")
        if isinstance(evidence_page, int):
            evidence_page = [evidence_page]
        if isinstance(evidence_page, list):
            out_row["evidence_page"] = list(evidence_page)

        if "bbox" in row and row["bbox"] is not None:
            out_row["bbox"] = row["bbox"]
        if "type" in row and row["type"] is not None:
            out_row["subimg_tpye"] = row["type"]
        if "category" in row and row["category"] is not None:
            out_row["category"] = str(row["category"])

        if keep_raw:
            out_row["raw"] = raw_text

        out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        success += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    out_f.flush()
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Generate MPMC QA per item using local Qwen-VL with vLLM.")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--out", required=True, help="Path to output JSONL")
    parser.add_argument("--model", default="/yuwenhan/models/Qwen2.5-VL-72B-Instruct", help="Local Qwen-VL checkpoint path")
    parser.add_argument("--tp", type=int, default=4, help="tensor_parallel_size for vLLM")
    parser.add_argument("--cuda-devices", default="0,1,2,3", help="CUDA_VISIBLE_DEVICES to expose to vLLM")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for vLLM.generate calls")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw model output in the JSONL")
    parser.add_argument("--sys-prompt-file", default=None, help="Optional file containing custom system prompt text")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each successful generation")
    parser.add_argument("--limit", type=int, default=10, help="Limit processed records (<=0 means all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output without overwriting")
    parser.add_argument("--failed-out", default=None, help="Optional path to save JSON list of failed indices")
    parser.add_argument(
        "--image-root",
        default="../../data/arxiv_png_800_docid_crops",
        help="Root directory containing cropped images when subimage_paths are relative.",
    )
    # Compatibility flags (ignored but kept to avoid breaking existing scripts)
    parser.add_argument("--max-retries", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--retry-wait", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--retry-multiplier", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.sys_prompt_file:
        with open(args.sys_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

    rows = load_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
        print(f"Processing only first {len(rows)} records due to --limit={args.limit}")

    start_idx = 0
    if args.resume:
        processed = count_existing_lines(args.out)
        if processed > 0:
            print(f"Resuming from existing {processed} line(s) in {args.out}")
        start_idx = min(processed, len(rows))
        if start_idx >= len(rows):
            print("All records already processed. Nothing to do.")
            return

    mode = "a" if args.resume and os.path.exists(args.out) else "w"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    total_rows = len(rows)
    processed_rows = start_idx
    failed_indices: List[int] = []
    batch_inputs: List[Dict[str, Any]] = []
    batch_meta: List[Dict[str, Any]] = []

    with open(args.out, mode, encoding="utf-8") as out_f:
        progress_iter = tqdm(
            range(start_idx, total_rows),
            desc="Generating QA",
            unit="sample",
            initial=start_idx,
            total=total_rows,
        )
        for idx in progress_iter:
            row = rows[idx]

            crop_ids = row.get("crop_ids") or []
            flat_crops: List[int] = []
            if isinstance(crop_ids, list):
                for group in crop_ids:
                    if isinstance(group, list):
                        flat_crops.extend([c for c in group if isinstance(c, int)])
            expected_images = len(flat_crops) if flat_crops else 0

            decoded_images: List[Image.Image] = []
            images_b64 = row.get("images_b64")
            if isinstance(images_b64, list) and any(isinstance(b64, str) and b64 for b64 in images_b64):
                b64_list = [b64 for b64 in images_b64 if isinstance(b64, str) and b64]
                expected_from_b64 = expected_images or len(b64_list)
                if len(b64_list) < expected_from_b64:
                    print(
                        f"[WARN] Row {idx}: only {len(b64_list)} base64 images provided but {expected_from_b64} expected; continuing with available images.",
                        file=sys.stderr,
                    )
                if expected_from_b64 > 0:
                    b64_list = b64_list[:expected_from_b64]
                decoded_images = decode_base64_images(b64_list)
            else:
                try:
                    image_paths = resolve_image_paths(row, args.image_root)
                except Exception as err:
                    print(f"[ERROR] Failed to resolve image paths at index {idx}: {err}", file=sys.stderr)
                    failed_indices.append(idx)
                    continue

                if not image_paths:
                    print(f"[WARN] No image paths found at index {idx}; skipping.", file=sys.stderr)
                    failed_indices.append(idx)
                    continue

                expected_from_paths = expected_images or len(image_paths)
                if len(image_paths) < expected_from_paths:
                    print(
                        f"[WARN] Row {idx}: only {len(image_paths)} image paths resolved but {expected_from_paths} expected; continuing with available images.",
                        file=sys.stderr,
                    )
                if expected_from_paths > 0:
                    image_paths = image_paths[:expected_from_paths]
                decoded_images = load_images_from_paths(image_paths)

            if not decoded_images:
                failed_indices.append(idx)
                continue

            page_ids = row.get("page_ids") or row.get("page_id") or []
            if isinstance(page_ids, (int, str)):
                page_ids = [page_ids]
            doc_id = row.get("doc_id") or row.get("docId") or row.get("docid")
            prompt_text = build_prompt_text(system_prompt, doc_id, page_ids)
            messages = build_conversation(decoded_images, prompt_text)
            if not messages:
                failed_indices.append(idx)
                continue

            try:
                request_inputs = prepare_inputs_for_vllm(messages, processor)
            except Exception as err:
                print(f"[ERROR] Failed to prepare vLLM inputs at index {idx}: {err}", file=sys.stderr)
                failed_indices.append(idx)
                continue

            batch_inputs.append(request_inputs)
            batch_meta.append({"index": idx, "row": row})

            if len(batch_inputs) >= args.batch_size:
                success, failed = process_batch(
                    llm,
                    sampling_params,
                    batch_inputs,
                    batch_meta,
                    out_f,
                    keep_raw=args.keep_raw,
                    sleep_seconds=args.sleep,
                )
                processed_rows += success
                failed_indices.extend(failed)
                batch_inputs.clear()
                batch_meta.clear()

        if batch_inputs:
            success, failed = process_batch(
                llm,
                sampling_params,
                batch_inputs,
                batch_meta,
                out_f,
                keep_raw=args.keep_raw,
                sleep_seconds=args.sleep,
            )
            processed_rows += success
            failed_indices.extend(failed)
            batch_inputs.clear()
            batch_meta.clear()

    if failed_indices:
        failed_path = args.failed_out or f"{args.out}.failed.json"
        with open(failed_path, "w", encoding="utf-8") as failed_f:
            json.dump(failed_indices, failed_f, ensure_ascii=False, indent=2)
        print(f"Recorded {len(failed_indices)} failed indice(s) to {failed_path}")

    handled_rows = processed_rows + len(failed_indices)
    if handled_rows >= total_rows:
        print(f"Done. Wrote {processed_rows} lines to {args.out}. Failed: {len(failed_indices)}.")
    else:
        remaining = total_rows - handled_rows
        print(
            f"Stopped early with {remaining} record(s) remaining. "
            f"Processed {processed_rows} success(es), {len(failed_indices)} failure(s)."
        )


if __name__ == "__main__":
    main()
