import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, Iterable
from openai import OpenAI

# ===== 基本配置 =====
BASE_URL = "xxxxx"
API_KEY = "xxxxx"
DEFAULT_MODEL = "deepseek-v3.1-250821"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ===== 工具函数 =====

def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

_judge_re = re.compile(r"<judge>(.*?)</judge>", re.IGNORECASE | re.DOTALL)

def to_bool_or_none(text: str):
    if text is None:
        return None
    v = text.lower()
    # 只要包含"false"（无论大小写），返回False
    if "false" in v:
        return False
    # 只要包含"true"（无论大小写），返回True
    if "true" in v:
        return True
    # 既不包含true也不包含false，返回原始文本
    return text

# ===== 主流程 =====

def run(input_path: str, output_path: str):
    in_p, out_p = Path(input_path), Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_p.open("w", encoding="utf-8") as fout:
        for item in read_jsonl(in_p):
            rid = item.get("id")
            rb = item.get("request_body", {})
            params = {
                "model": rb.get("model", DEFAULT_MODEL),
                "messages": build_messages(rb),
                "temperature": rb.get("temperature", 0.0),
                "top_p": rb.get("top_p", 1.0),
                "max_tokens": rb.get("max_tokens", 1024),
            }
            try:
                resp = client.chat.completions.create(**params)
                content = resp.choices[0].message.content if resp and resp.choices else ""
                m = _judge_re.search(content)
                if m:
                    judge = to_bool_or_none(m.group(1))
                else:
                    judge = to_bool_or_none(content)
                # 确保最终输出为小写的true/false字符串
                if judge is True:
                    judge_str = "true"
                elif judge is False:
                    judge_str = "false"
                else:
                    judge_str = judge
            except Exception:
                judge_str = None

            # 立即写入输出文件
            record = {"id": str(rid) if rid is not None else None, "judge": judge_str}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            total += 1
            print(f"[{total}] 写入完成 id={rid}")

    print(f"Done: {total} lines → {out_p}")

def build_messages(rb: Dict[str, Any]):
    msgs = rb.get("messages", []) or []
    sys_prompt = rb.get("system")
    if sys_prompt:
        msgs = [{"role": "system", "content": sys_prompt}] + msgs
    return msgs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python run.py <输入.jsonl> <输出.jsonl>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
    