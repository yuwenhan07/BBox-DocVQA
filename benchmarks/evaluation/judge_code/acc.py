import os
import json

# Base directory containing subfolders like 3B, 7B, 32B, 72B, gpt-5
base_dir = "../judgement_output/qwen_pixel_input"  

# Define the ranges
# ranges = [(1, 320), (321, 875), (876, 1624)]
ranges = [(1, 749), (750, 1305), (1306, 1623)]

# Walk through all subdirectories and files
for root, _, files in os.walk(base_dir):
    for filename in files:
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(root, filename)

        # Per-file counters
        counts = [0, 0, 0]
        corrects = [0, 0, 0]

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # id may be string or int in the file; normalize to int
                id_ = int(item["id"])  # will raise if missing; intentional to catch bad rows
                # judge may be string or boolean; normalize to boolean
                j = item.get("judge")
                if isinstance(j, str):
                    judge = j.strip().lower() == "true"
                else:
                    judge = bool(j)

                for i, (start, end) in enumerate(ranges):
                    if start <= id_ <= end:
                        counts[i] += 1
                        if judge:
                            corrects[i] += 1
                        break

        # Pretty print results for this file
        model_folder = os.path.basename(root)
        print(f"\n[Model: {model_folder}] File: {filename}")
        for i, (start, end) in enumerate(ranges):
            total = counts[i]
            corr = corrects[i]
            acc = (corr / total) if total else 0.0
            print(f"  {start:>4}-{end:<4}: acc={acc:.4f}  (correct={corr}, total={total})")
        # Compute and print total accuracy
        total_counts = sum(counts)
        total_corrects = sum(corrects)
        total_acc = (total_corrects / total_counts) if total_counts else 0.0
        print(f"  TOTAL: acc={total_acc:.4f}  (correct={total_corrects}, total={total_counts})")