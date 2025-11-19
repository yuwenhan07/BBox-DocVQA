# BBox-DocVQA

BBox-DocVQA is a large-scale bounding-box-grounded document VQA dataset and pipeline that operationalises the **Segment-Judge-Generate** workflow introduced in our CVPR 2026 submission, _"BBox-DocVQA: A Large-Scale Bounding-Box-Grounded Dataset for Enhancing Reasoning in Document Visual Question Answering."_ The repository now mirrors the stages described in the paper so you can (1) scrape raw PDFs, (2) turn them into spatially grounded QA pairs, and (3) evaluate VLMs under fine-grained supervision.

## Highlights from the Paper
- **Fine-grained grounding:** Every QA pair is tied to explicit bounding boxes, enabling spatial reasoning checks beyond page-level DocVQA benchmarks.
- **Scale and coverage:** 3.6K documents, 32K QA pairs and 44K images spanning multi-domain arXiv content; 30,780 auto-generated training QA pairs + 1,623 manually curated benchmark items.
- **Task diversity:** Single-page/single-box (SPSBB), single-page/multi-box (SPMBB), and multi-page/multi-box (MPMBB) scenarios with text, table, and figure regions (Table 2 in the paper).
- **Measured impact:** Fine-tuning VLMs such as Qwen2.5-VL and InternVL on BBox-DocVQA improves both bounding-box localization and answer accuracy.
- For detailed methodology, refer to `docs/BBox_DocVQA_paper.pdf`.

## Repository Layout
```
.
|-- docs/
|   \-- BBox_DocVQA_paper.pdf      # CVPR 2026 submission (reference for this repo)
|-- pipelines/
|   |-- raw_data/                  # Scripts for harvesting arXiv PDFs and rasterising pages
|   |-- document_pipeline/         # Segment-Judge-Generate implementation and helpers
|   \-- training_qa/               # Multi-page / multi-bbox QA generation for training sets
|-- benchmarks/
|   |-- qa_generation/             # Scripts used to curate the fine-grained benchmark
|   \-- evaluation/                # IoU + QA evaluation utilities for VLM baselines
|-- data/
|   |-- trainingdata_30k.jsonl     # Auto-generated QA pairs (Section 3.3.1)
|   \-- benchmark_v2.jsonl         # Manually verified benchmark split (Section 3.3.2)
\-- README.md
```

## Environment & Dependencies
- Python 3.10+ and CUDA-ready PyTorch GPUs are required for SAM, Qwen-VL, VLLM, and LMDeploy workloads.
- Install everything with `pip install -r requirements.txt`. The list covers `torch`, `transformers`, `modelscope`, `vllm`, `segment-anything`, `opencv-python-headless`, `numpy`, `Pillow`, `pdf2image`, `PyMuPDF`, `qwen-vl-utils`, `lmdeploy`, `matplotlib`, `openai`, `requests`, `feedparser`, and `tqdm`.
- Recommended bootstrap:
  ```bash
  conda create -n bbox-docvqa python=3.10
  conda activate bbox-docvqa
  pip install -r requirements.txt
  ```
- **System extras:** Poppler binaries for `pdf2image`, the SAM checkpoint (e.g., `sam_vit_h_4b8939.pth`), Qwen/InternVL weights, and optionally an `OPENAI_API_KEY` for the OpenAI-based benchmark scripts. Allocate at least 64 GB GPU memory for 72B models or adjust configs accordingly.

## Pipelines
### 1. Raw document harvesting - `pipelines/raw_data`
1. Query arXiv IDs per field:
   ```bash
   python pipelines/raw_data/get_arxiv_id_speed.py \
     --majors \
     --per-field 100 \
     --start-date 2023-01-01 \
     --end-date 2024-12-31 \
     --workers 20 \
     --base-dir data/downloads
   ```
2. Download PDFs and cache metadata with `download_reuse.py` / `gen_download_sh.py`.
3. Rasterise PDFs into per-page PNGs (Section 2.1 of the paper):
   ```bash
   python pipelines/raw_data/pdf2png_parallel.py data/arxiv_pdf \
     -o data/arxiv_png \
     -w 128 --dpi 300
   ```

### 2. Segment-Judge-Generate - `pipelines/document_pipeline`
This folder hosts the production-ready rewrite of the pipeline described in Section 3:
- `boundingdoc/` bundles reusable components for PDF rendering, SAM cropping, judge agents, and QA generation.
- `process_doc.py` runs the full PDF -> crops -> judge -> QA flow for one or more PDFs. Example:
  ```bash
  python pipelines/document_pipeline/process_doc.py data/arxiv_pdf/sample.pdf \
    --work_root work/full_pipeline \
    --output_dir outputs/qa_jsonl \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --sam_devices 0 1 \
    --sam_num_workers 8 \
    --judge_model Qwen/Qwen2.5-VL-14B-Instruct \
    --judge_backend vllm \
    --judge_gpu_devices 0,1,2,3 \
    --qa_model Qwen/Qwen2.5-VL-14B-Instruct \
    --qa_backend vllm \
    --qa_workers 4 \
    --dpi 300
  ```
- Stage-specific entrypoints (`step0_pdf2png.py` ... `step4_data_tran.py`) are available if you need to resume or debug a specific component.
- Use `run_all_serial.sh` as a template for large-scale batch processing by discipline.

### 3. Training QA generation - `pipelines/training_qa`
Implements the multi-granularity QA templates from Section 3.3:
- `do_sample_*.py` selects SPSBB / SPMBB / MPMBB crops from the judged results.
- `qa_generate_*.py` invokes Qwen2.5-VL via vLLM to craft grounded QA pairs across multiple crops/pages.
- `add_images.py`, `add_summary.py`, and `trans.py` post-process outputs into the released `trainingdata_30k.jsonl`.

Run, for instance:
```bash
python pipelines/training_qa/qa_generate_MPMC.py \
  --input data/training_samples_mpmc.jsonl \
  --image-root data/crop_bank \
  --model Qwen/Qwen2.5-VL-14B-Instruct \
  --output outputs/trainingdata_mpmc.jsonl
```

### 4. Benchmark QA generation - `benchmarks/qa_generation`
Contains the scripts (`qa_genearte_*.py`, `visualize_qa_results.py`, etc.) used to produce the 1,623 human-verified QA items highlighted in Figure 3-5. These tools mirror the training generators but enforce stricter quality filters and visualization for annotators.

### 5. Benchmark evaluation - `benchmarks/evaluation`
- `run_internvl.py`, `eval_qwen.py`, and `PROMPT.py` show how to prompt VLMs for simultaneous answer + bbox prediction.
- `compute_iou.py` and `trans_bbox.py` implement the IoU checks aligned with the paper's metrics.
- `judge_code/` holds the lightweight QA judge for sanity-checking generations.

## Data Assets
- `data/trainingdata_30k.jsonl`: Automatically generated QA pairs over 3,671 arXiv papers (30,780 samples). Columns include document/category metadata, bbox coordinates, crop paths, and QA text.
- `data/benchmark_v2.jsonl`: Final fine-grained benchmark with 1,623 manually verified QA pairs sampled uniformly across domains. Use it for reporting numbers comparable to Table 1/2.
- For new data releases or large files, add them under `data/` or point to an external storage bucket, then update this README accordingly.

## Citation
Please cite the paper when reusing the dataset or code:
> paper is under reviewing...

For questions or pull requests, feel free to open an issue once the repository goes public. Enjoy building spatially grounded DocVQA systems!
