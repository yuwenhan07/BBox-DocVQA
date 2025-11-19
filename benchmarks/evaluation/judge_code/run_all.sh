#!/bin/bash
set -e

INPUT_DIR="../outputs_qwen3/"
OUTPUT_DIR="../judgement_input/input_qwen3_32B/"

mkdir -p "$OUTPUT_DIR"

# 遍历所有子目录和 .jsonl 文件
for file in $(find "$INPUT_DIR" -type f -name "*.jsonl"); do
    # 提取文件名和上级目录名
    subdir=$(basename "$(dirname "$file")")
    filename=$(basename "$file" .jsonl)
    out_subdir="$OUTPUT_DIR/$subdir"
    mkdir -p "$out_subdir"

    # 输出文件路径
    outfile="$out_subdir/${filename}_converted.jsonl"

    echo "Converting $file → $outfile"
    python batch_convert.py "$file" "$outfile"
done

echo "✅ All conversions done! Converted files are under $OUTPUT_DIR/"