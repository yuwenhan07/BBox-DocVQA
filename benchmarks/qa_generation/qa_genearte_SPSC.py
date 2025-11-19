import os
import json
import argparse
import time
from typing import Any, Callable, Dict, List
from tqdm import tqdm

SYSTEM_PROMPT = (
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
    "  \"a\": \"...\"\n"
    "}\n"
    "7) If the crops lack enough information to form such a question, output exactly: "
    "{\"q\": \"\", \"a\": \"\"}.\n"
)

# --- OpenAI Responses API ---
from openai import OpenAI
client = OpenAI()

# ---------- Helpers ----------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_user_content(system_prompt: str, images_b64: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": system_prompt},
    ]
    # print(f"DEBUG: User content: {content}")
    for img_b64 in images_b64:
        if isinstance(img_b64, str) and img_b64:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"})
    return content


def call_openai_responses(model: str, system_prompt: str, images_b64: List[str]) -> str:
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": build_user_content(system_prompt, images_b64),
        }],
        # request_kwargs={
        #     "reasoning": {"effort": "minimal"},
        # },
    )
    text = getattr(resp, "output_text", None)
    if text is None:
        try:
            # Fallback to raw structure if output_text is absent
            text = resp.output[0].content[0].text
            print(f"DEBUG: Fallback output_text: {text}")
        except Exception:
            text = ""
    return text or ""


def with_retries(fn: Callable[[], str], max_retries: int, base_delay: float, backoff: float) -> str:
    attempt = 0
    while True:
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as err:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Exceeded max retries ({max_retries}) for OpenAI request") from err
            sleep_for = base_delay * (backoff ** (attempt - 1))
            sleep_for = max(sleep_for, base_delay)
            print(f"[WARN] OpenAI request failed (attempt {attempt}/{max_retries}); retrying in {sleep_for:.2f}s. Error: {err}")
            time.sleep(sleep_for)


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
    """Parse either flat {q,a} or legacy {qas:[{q,a}]} into a single QA dict."""
    try:
        payload = json.loads(json_text)
    except Exception:
        return {}
    if isinstance(payload, dict) and "q" in payload and "a" in payload:
        return {"q": payload.get("q", ""), "a": payload.get("a", "")}
    if isinstance(payload, dict) and isinstance(payload.get("qas"), list) and payload["qas"]:
        qa0 = payload["qas"][0]
        if isinstance(qa0, dict) and "q" in qa0 and "a" in qa0:
            return {"q": qa0.get("q", ""), "a": qa0.get("a", "")}
    return {}


# -------------- Main --------------
def main():
    parser = argparse.ArgumentParser(description="Generate SPSC QA per item and write line-by-line.")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--out", required=True, help="Path to output JSONL")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model name")
    parser.add_argument("--sys-prompt-file", default=None, help="Optional file containing custom system prompt text")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between API calls")
    parser.add_argument("--limit", type=int, default=10, help="Limit processed records (<=0 means all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file without overwriting")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries for OpenAI API failures per sample")
    parser.add_argument("--retry-wait", type=float, default=5.0, help="Initial wait (seconds) before retrying the API")
    parser.add_argument("--retry-multiplier", type=float, default=2.0, help="Exponential backoff multiplier for retries")
    parser.add_argument("--failed-out", default=None, help="Optional path to save JSON list of failed indices")
    args = parser.parse_args()

    # system prompt: keep original text but allow override via file
    system_prompt = SYSTEM_PROMPT
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

    total_rows = len(rows)
    processed_rows = start_idx
    failed_indices: List[int] = []
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
            images_b64 = row.get("images_b64") or []
            if not isinstance(images_b64, list):
                images_b64 = []

            try:
                json_text = with_retries(
                    lambda: call_openai_responses(args.model, system_prompt, images_b64),
                    max_retries=args.max_retries,
                    base_delay=args.retry_wait,
                    backoff=args.retry_multiplier,
                )
            except RuntimeError as err:
                print(f"[ERROR] Stopping at index {idx} due to repeated API failures: {err}")
                print("Skipping this record. It will be logged as failed.")
                failed_indices.append(idx)
                continue
            except Exception as err:
                print(f"[ERROR] Unexpected failure at index {idx}: {err}")
                print("Skipping this record. It will be logged as failed.")
                failed_indices.append(idx)
                continue

            qa = extract_single_qa(json_text)

            doc_name = str(row.get("doc_id", ""))
            evidence_page = row.get("page_ids") or row.get("page_id") or []
            if isinstance(evidence_page, int):
                evidence_page = [evidence_page]
            bbox = row.get("bbox") or []
            subimg_type = row.get("type") or []
            if isinstance(subimg_type, list):
                subimg_tpye = subimg_type  # preserve original nested structure (and spelling)
            else:
                subimg_tpye = []
            out_row = {
                "query": qa.get("q", ""),
                "answer": qa.get("a", ""),
                "doc_name": doc_name,
                "evidence_page": evidence_page,
                "bbox": bbox,
                "subimg_tpye": subimg_tpye,
                "category": row.get("category", ""),
            }

            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_rows += 1

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

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
        print(f"Stopped early with {remaining} record(s) remaining. Processed {processed_rows} success(es), {len(failed_indices)} failure(s).")


if __name__ == "__main__":
    main()
