"""
pipeline.py
-------------
Main entrypoint for the segmentation → bounds → VNNLIB generation pipeline.

Inputs:
    - Command-line args:
        • --config path/to/config.yaml
        • --indices N1 N2 ...
        • --images path1 path2 ...

    - YAML config containing:
        • dataset paths
        • SAM2 model config
        • verification parameters (eps, timeout, etc.)
        • output directory

Outputs:
    - vnnlib/ folder with generated property files
    - onnx/ folder containing copied model
    - debug_vis/ folder with segment overlays
    - instances.csv listing (onnx_path, vnnlib_path, timeout)
    - input_change_stats.csv reporting semantic coverage per segment

Purpose:
    Orchestrates the full pipeline:
        1) Load config + prepare directories
        2) Load dataset metadata
        3) Load ONNX model
        4) Load SAM2
        5) For each selected image:
             - run segmentation
             - compute bounds
             - write VNNLIBs
             - log stats
        6) Write CSV outputs

    Serves as the user interface layer while all heavy logic is
    handled by supporting modules.
"""

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor

from config_utils import load_config, ensure_dir
from dataset_utils import load_imagenet_vggnet16_metadata,get_imagenet_label_for_image
from sam2_utils import load_sam2_predictor, run_segmentation_model
from vnnlib_utils import bounds_for_segment,report_changed_inputs,write_vnnlib_for_segment
from vis_utils import visualize_segments


def process_single_image(
    img_path: Path,
    predictor: SAM2ImagePredictor,
    cfg: Dict,
    get_label_fn: Callable[[Path], int],
    onnx_rel_path: str,
    vnnlib_dir: Path,
    debug_dir: Optional[Path] = None,
    stats_rows: Optional[List[Dict]] = None,
) -> List[Tuple[str, str, int]]:
    """
    Returns list of instances.csv rows for this image.
    Each row: (onnx_rel_path, vnnlib_rel_path, timeout)
    """
    print("\n==============================")
    print(f"Processing image: {img_path}")
    timeout = int(cfg["output"]["timeout"])
    eps = float(cfg["verification"]["epsilon"])
    dilation_radius = int(cfg["verification"]["dilation_radius"])
    num_classes = int(cfg["verification"]["num_classes"])

    label = get_label_fn(img_path)
    print(f"True label index: {label} / {num_classes}")

    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image) / 255.0

    seg_cfg = cfg["segmentation"]
    segments = run_segmentation_model(
        predictor=predictor,
        image_np=image_np,
        grid_size=seg_cfg.get("grid_size", 6),
        iou_threshold=seg_cfg.get("iou_threshold", 0.8),
        score_threshold=seg_cfg.get("score_threshold", 0.0),
        max_segments=seg_cfg.get("max_segments", None),
    )

    if debug_dir is not None:
        img_basename = img_path.stem
        visualize_segments(
            image_np=image_np,
            segments=segments,
            out_dir=debug_dir,
            img_basename=img_basename,
        )

    img_basename = img_path.stem
    csv_rows: List[Tuple[str, str, int]] = []

    # 1) global epsilon VNNLIB
    lb_global = np.clip(image_np - eps, 0.0, 1.0)
    ub_global = np.clip(image_np + eps, 0.0, 1.0)
    lb_global_flat = lb_global.transpose(2, 0, 1).reshape(-1)
    ub_global_flat = ub_global.transpose(2, 0, 1).reshape(-1)

    global_stats = report_changed_inputs(
        lb_flat=lb_global_flat,
        ub_flat=ub_global_flat,
        tag=f"{img_basename}_global",
        extra={
            "image": img_basename,
            "is_global": True,
            "segment_index": -1,
            "eps": eps,
            "dilation_radius": dilation_radius,
            "score": None,
            "bbox": None,
        },
    )
    if stats_rows is not None:
        stats_rows.append(global_stats)

    global_vnnlib = vnnlib_dir / f"{img_basename}_global_eps_{eps:.4f}.vnnlib"
    write_vnnlib_for_segment(
        lb_flat=lb_global_flat,
        ub_flat=ub_global_flat,
        label=label,
        num_classes=num_classes,
        out_path=global_vnnlib,
    )
    csv_rows.append(
        (
            onnx_rel_path,
            str(Path("vnnlib") / global_vnnlib.name),
            timeout,
        )
    )

        # 2) segmented VNNLIBs:
    #    for each segment we now create TWO variants:
    #      - fix mask      -> object frozen, background can change
    #      - fix non-mask  -> background frozen, object can change
    for seg_idx, seg in enumerate(segments):
        mask = seg["mask"]

        # ------------------------------------------------
        # (A) FIX MASK: only NON-MASK region can move
        #     -> "freeze_mask=True" => eps outside mask
        # ------------------------------------------------
        lb_fix_mask, ub_fix_mask = bounds_for_segment(
            image_np=image_np,
            mask=mask,
            eps=eps,
            dilation_radius=dilation_radius,
            freeze_mask=True,   # mask is fixed
        )

        seg_stats_fix_mask = report_changed_inputs(
            lb_flat=lb_fix_mask,
            ub_flat=ub_fix_mask,
            tag=f"{img_basename}_seg{seg_idx}_fixmask",
            extra={
                "image": img_basename,
                "is_global": False,
                "segment_index": seg_idx,
                "eps": eps,
                "dilation_radius": dilation_radius,
                "score": float(seg["score"]),
                "bbox": seg["bbox"],
                "pattern": "fix_mask",
            },
        )
        if stats_rows is not None:
            stats_rows.append(seg_stats_fix_mask)

        seg_vnnlib_fix_mask = (
            vnnlib_dir
            / f"{img_basename}_seg{seg_idx}_fixmask_eps_{eps:.4f}.vnnlib"
        )
        write_vnnlib_for_segment(
            lb_flat=lb_fix_mask,
            ub_flat=ub_fix_mask,
            label=label,
            num_classes=num_classes,
            out_path=seg_vnnlib_fix_mask,
        )
        csv_rows.append(
            (
                onnx_rel_path,
                str(Path("vnnlib") / seg_vnnlib_fix_mask.name),
                timeout,
            )
        )

        # ------------------------------------------------
        # (B) FIX NON-MASK: only MASK region can move
        #     -> "freeze_mask=False" => eps inside mask
        # ------------------------------------------------
        lb_fix_nonmask, ub_fix_nonmask = bounds_for_segment(
            image_np=image_np,
            mask=mask,
            eps=eps,
            dilation_radius=dilation_radius,
            freeze_mask=False,  # non-mask is fixed
        )

        seg_stats_fix_nonmask = report_changed_inputs(
            lb_flat=lb_fix_nonmask,
            ub_flat=ub_fix_nonmask,
            tag=f"{img_basename}_seg{seg_idx}_fixnonmask",
            extra={
                "image": img_basename,
                "is_global": False,
                "segment_index": seg_idx,
                "eps": eps,
                "dilation_radius": dilation_radius,
                "score": float(seg["score"]),
                "bbox": seg["bbox"],
                "pattern": "fix_nonmask",
            },
        )
        if stats_rows is not None:
            stats_rows.append(seg_stats_fix_nonmask)

        seg_vnnlib_fix_nonmask = (
            vnnlib_dir
            / f"{img_basename}_seg{seg_idx}_fixnonmask_eps_{eps:.4f}.vnnlib"
        )
        write_vnnlib_for_segment(
            lb_flat=lb_fix_nonmask,
            ub_flat=ub_fix_nonmask,
            label=label,
            num_classes=num_classes,
            out_path=seg_vnnlib_fix_nonmask,
        )
        csv_rows.append(
            (
                onnx_rel_path,
                str(Path("vnnlib") / seg_vnnlib_fix_nonmask.name),
                timeout,
            )
        )

    print(f"[PIPELINE] Created {len(csv_rows)} VNNLIBs for {img_path.name}")
    return csv_rows


def main():
    parser = argparse.ArgumentParser(
        description="Image dataset → SAM2 segmentation → VNNLIB pipeline"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        help="Dataset indices (0-based) to process",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        help="Explicit image paths to process",
    )

    # TODO: delete later
    import sys
    sys.argv += ["--config", "/Users/zd3504phd/Desktop/XAIV/configs/xaiv/vggnet16_benchmark2022_segmented.yaml"]

    args = parser.parse_args()
    cfg = load_config(args.config)

    # -----------------------------
    # Prepare benchmark directories
    # -----------------------------
    out_root = ensure_dir(Path(cfg["output"]["benchmark_dir"]))
    onnx_dir = ensure_dir(out_root / "onnx")
    vnnlib_dir = ensure_dir(out_root / "vnnlib")
    instances_csv_path = out_root / "instances.csv"
    debug_vis_dir = ensure_dir(out_root / "debug_vis")
    stats_csv_path = out_root / "input_change_stats.csv"
    change_stats_rows: List[Dict] = []

    # -----------------------------
    # Handle ONNX model
    # -----------------------------
    onnx_source = Path(cfg["onnx"]["source_model"])
    onnx_target_name = cfg["onnx"].get("target_name", onnx_source.name)
    onnx_target = onnx_dir / onnx_target_name

    if not onnx_target.exists():
        print(f"[ONNX] Copying {onnx_source} -> {onnx_target}")
        onnx_target.write_bytes(onnx_source.read_bytes())
    else:
        print(f"[ONNX] Using existing {onnx_target}")

    onnx_rel_path = str(Path("onnx") / onnx_target.name)

    # -----------------------------
    # Dataset metadata
    # -----------------------------
    dataset_type = cfg["dataset"].get("type", "tinyimagenet")

    if dataset_type == "tinyimagenet":
        all_images, img_to_wnid, wnid_to_idx, num_classes = load_tinyimagenet_metadata(
            cfg["dataset"]
        )

        def get_label(img_path: Path) -> int:
            return get_true_label_for_image(img_path, img_to_wnid, wnid_to_idx)

    elif dataset_type == "imagenet_vggnet16":
        all_images, img_to_label, num_classes = load_imagenet_vggnet16_metadata(
            cfg["dataset"]
        )

        def get_label(img_path: Path) -> int:
            return get_imagenet_label_for_image(img_path, img_to_label)

    else:
        raise ValueError(f"Unknown dataset.type = {dataset_type}")

    print(f"[Dataset] Found {len(all_images)} images; num_classes = {num_classes}")
    cfg.setdefault("verification", {})
    cfg["verification"]["num_classes"] = num_classes

    # -----------------------------
    # Which images to process?
    # -----------------------------
    images_to_process: List[Path] = []

    if args.images:
        images_to_process.extend(Path(p) for p in args.images)

    if args.indices is not None and len(args.indices) > 0:
        for idx in args.indices:
            images_to_process.append(all_images[idx])
    else:
        cfg_indices = cfg.get("run", {}).get("indices", [])
        cfg_paths = cfg.get("run", {}).get("image_paths", [])
        for idx in cfg_indices:
            images_to_process.append(all_images[idx])
        for p in cfg_paths:
            images_to_process.append(Path(p))

    if not images_to_process:
        raise ValueError("No images specified. Use --indices/--images or config.run.*")

    # -----------------------------
    # Load SAM2 model
    # -----------------------------
    predictor = load_sam2_predictor(cfg["segmentation"])

    # -----------------------------
    # Run pipeline
    # -----------------------------
    all_rows: List[Tuple[str, str, int]] = []

    for img_path in images_to_process:
        rows = process_single_image(
            img_path=img_path,
            predictor=predictor,
            cfg=cfg,
            get_label_fn=get_label,
            onnx_rel_path=onnx_rel_path,
            vnnlib_dir=vnnlib_dir,
            debug_dir=debug_vis_dir,
            stats_rows=change_stats_rows,
        )
        all_rows.extend(rows)

    # -----------------------------
    # Write instances.csv
    # -----------------------------
    print(f"\n[CSV] Writing {len(all_rows)} rows to {instances_csv_path}")
    with instances_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row)

    if change_stats_rows:
        fieldnames = [
            "image",
            "tag",
            "is_global",
            "segment_index",
            "eps",
            "dilation_radius",
            "score",
            "bbox",
            "num_changed",
            "num_fixed",
            "total",
            "percent_changed",
            "pattern",
        ]
        print(f"[CSV] Writing {len(change_stats_rows)} rows to {stats_csv_path}")
        with stats_csv_path.open("w", newline="") as f_stats:
            writer = csv.DictWriter(f_stats, fieldnames=fieldnames)
            writer.writeheader()
            for row in change_stats_rows:
                for key in fieldnames:
                    row.setdefault(key, None)
                writer.writerow(row)
    else:
        print("[CSV] No change-stats rows collected, not writing stats CSV.")

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()