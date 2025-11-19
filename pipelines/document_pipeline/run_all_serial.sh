#!/bin/bash

# 定义类别数组
categories=("econ" "eess" "math" "physics" "q-bio" "q-fin" "stat")

# 公共参数
BASE_PDF_DIR="/yuwenhan/Proj/boundingdoc/data/arxiv_pdf_4044"
WORK_ROOT_BASE="/yuwenhan/Proj/boundingdoc/data/arxiv_step_work_4044"
SAM_CKPT="/yuwenhan/models/sam/sam_vit_h_4b8939.pth"
JUDGE_MODEL="/yuwenhan/models/Qwen2.5-VL-72B-Instruct"
LOG_DIR="logs"

# 创建日志目录
mkdir -p $LOG_DIR

echo "===== 开始顺序处理所有类别 ====="

# 循环顺序执行每个类别
for cat in "${categories[@]}"; do
    echo ">>> 开始处理类别: $cat <<<"
    
    # 日志文件名
    LOG_FILE="${LOG_DIR}/run_${cat}_step1,2_$(date +%Y%m%d_%H%M%S).log"

    # 运行命令（前台执行，不加 &）
    python process_sam_judge.py ${BASE_PDF_DIR}/${cat} \
        --work_root ${WORK_ROOT_BASE}/${cat}/work \
        --output_dir ${WORK_ROOT_BASE}/${cat}/judge_output \
        --sam_checkpoint ${SAM_CKPT} \
        --sam_devices 4 5 \
        --sam_num_workers 8 \
        --judge_model ${JUDGE_MODEL} \
        --judge_backend vllm \
        --judge_gpu_devices 0,1,2,3 \
        --judge_batch_size 32 \
        --pdf_thread_count 32 \
        --max_workers 32 \
        > "$LOG_FILE" 2>&1

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 类别 $cat 完成，日志：$LOG_FILE"
    else
        echo "❌ 类别 $cat 出错，日志：$LOG_FILE"
        echo "中止执行。"
        exit 1
    fi

    # 每个类别结束后稍微休息一下，防止资源没完全释放
    echo "等待 30 秒再继续下一个类别..."
    sleep 30
done

echo "🎉 所有类别已顺序处理完毕！"
