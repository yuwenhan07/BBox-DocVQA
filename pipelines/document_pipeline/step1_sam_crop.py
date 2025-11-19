#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from boundingdoc.sam_crop import SamCropConfig, SamCropper, run_parallel_sam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch crop pages with SAM, grouped by subdirectories"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Root directory or single report folder containing page PNG files",
    )
    parser.add_argument(
        "--output_root",
        default=Path("./output"),
        type=Path,
        help="Output root directory (creates {output_root}/{set}/{page}/crops)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the SAM checkpoint",
    )
    parser.add_argument("--device", default="cuda", help="Computation device: cuda or cpu")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional list of CUDA device identifiers (e.g. 0 1 2 or cuda:0 cuda:1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (defaults to number of devices)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help="Maximum queued images waiting per worker (defaults to config)",
    )
    parser.add_argument( 
        "--sample-size",
        type=int,
        default=5,
        help="Maximum number of page PNGs per document to process with SAM (<=0 means no limit)",
    )
    parser.add_argument("--pad_px", type=int, default=10, help="Extra padding around crops (pixels)")
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.05,
        help="Minimum area ratio required to keep a crop",
    )
    parser.add_argument(
        "--max_ratio",
        type=float,
        default=0.70,
        help="Maximum area ratio allowed to keep a crop",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = args.devices if args.devices is not None else [args.device]
    normalized_devices: list[str] = []
    for dev in devices:
        dev_str = str(dev).strip()
        if not dev_str:
            continue
        if dev_str.isdigit():
            dev_str = f"cuda:{dev_str}"
        normalized_devices.append(dev_str)

    primary_device = normalized_devices[0] if normalized_devices else args.device

    config = SamCropConfig(
        checkpoint=args.checkpoint,
        device=primary_device,
        pad_px=args.pad_px,
        area_min_ratio=args.min_ratio,
        area_max_ratio=args.max_ratio,
        devices=tuple(normalized_devices) if normalized_devices else None,
        num_workers=args.num_workers,
        queue_size=args.queue_size or 32,
        max_pages_per_doc=args.sample_size,
    )

    use_parallel = len(normalized_devices) > 1 or (args.num_workers and args.num_workers > 1)
    if use_parallel:
        stats = run_parallel_sam(
            config,
            args.input_dir,
            args.output_root,
            devices=normalized_devices,
            num_workers=args.num_workers,
            queue_size=args.queue_size,
        )
    else:
        cropper = SamCropper(config)
        stats = cropper.run(args.input_dir, args.output_root)
    print(
        f"Completed: processed {stats.processed_sets} set(s) containing {stats.processed_images} image(s);"
        f" successful crops for {stats.succeeded_images} images. Output root: {args.output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
