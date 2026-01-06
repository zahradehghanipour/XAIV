"""
pipeline.py
-------------
Main entrypoint for the segmentation → bounds → VNNLIB generation pipeline.

Inputs:
    - Command-line args:
        • --config path/to/config.yaml
        • --indices N1 N2 ...
        • --images path1 path2 ...

Outputs:
    - vnnlib/ folder with generated property files
    - debug_vis/ folder with segment overlays
    - instances.csv listing (onnx_path, vnnlib_path, timeout)
    - input_change_stats.csv reporting semantic coverage per segment

Purpose:
    Orchestrates the full pipeline:
        For each selected image:
            - run segmentation
            - compute bounds
            - write VNNLIBs
            - log stats
        Write CSV outputs
"""

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor

from config_utils import load_config, ensure_dir
from dataset_utils import load_imagenet_vggnet16_metadata,get_imagenet_label_for_image
from sam2_utils import load_sam2_predictor, run_segmentation_model
from vnnlib_utils import bounds_for_segment,report_changed_inputs,write_vnnlib_for_segment
from vis_utils import visualize_segments

import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import onnxruntime as ort

VGG16_MEAN = [0.485, 0.456, 0.406]
VGG16_STD = [0.229, 0.224, 0.225]

# pipeline.py
import json

def load_manual_prompts(prompts_path: Path) -> dict:
    prompts = {}
    for p in prompts_path.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            prompts.update(json.load(f))
    return prompts

def prompts_for_image(prompts_db: dict, image_key: str):
    """
    prompts_db example:
    {
      "myimg.jpg": {
        "foreground": [[x,y], [x,y]],
        "background": [[x,y]]
      }
    }
    """
    entry = prompts_db.get(image_key)
    if entry is None:
        return None, None

    fg = entry.get("foreground", [])
    bg = entry.get("background", [])

    # labels: 1 for FG, 0 for BG
    coords = np.array(fg + bg, dtype=np.float32)
    labels = np.array([1] * len(fg) + [0] * len(bg), dtype=np.int64)

    if coords.size == 0:
        return None, None
    return coords, labels

def vgg16_preprocess(img_pil, normalize=True, size=224):
    img_r = F.resize(img_pil, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    img_c = F.center_crop(img_r, [size, size])
    img_t = F.to_tensor(img_c)  # [3,H,W] in [0,1]
    if normalize:
        img_t = F.normalize(img_t, mean=VGG16_MEAN, std=VGG16_STD)
    return img_t

def debug_print_mask_binary(mask_t: torch.Tensor, tag: str = ""):
    """
    Prints a binary sanity check (0/1) in percentage.
    mask_t is expected bool or {0,1}. Shape [1,H,W] or [H,W].
    """
    if mask_t.ndim == 3:
        m = mask_t[0]
    else:
        m = mask_t

    # Convert to int for unique/value checks
    m_int = m.to(torch.int32)
    uniq = torch.unique(m_int).cpu().tolist()

    total = m.numel()
    ones = int(m.sum().item())
    zeros = total - ones
    ones_pct = 100.0 * ones / total if total else 0.0
    zeros_pct = 100.0 * zeros / total if total else 0.0

    print(f"[DEBUG][MASK]{'['+tag+']' if tag else ''} unique={uniq} "
          f"ones={ones_pct:.2f}% zeros={zeros_pct:.2f}% (H={m.shape[-2]}, W={m.shape[-1]})")

def save_preprocess_debug_vis(
    out_dir: Path,
    prefix: str,
    img_pil_before: Image.Image,
    mask_pil_before: Image.Image,
    img_t_after: torch.Tensor,
    mask_t_after: torch.Tensor,
):
    """
    Saves:
      - before_mask_bw.png
      - before_overlay.png
      - after_mask_bw.png
      - after_overlay.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- BEFORE: convert to numpy ---
    img_before = np.asarray(img_pil_before).astype(np.float32) / 255.0
    mask_before = np.asarray(mask_pil_before)
    if mask_before.ndim == 3:
        mask_before = mask_before[..., 0]
    mask_before = (mask_before > 0).astype(np.float32)

    # --- AFTER: de-normalize image tensor back to [0,1] for visualization ---
    img_after = img_t_after.detach().cpu().clone()
    for c in range(3):
        img_after[c] = img_after[c] * VGG16_STD[c] + VGG16_MEAN[c]
    img_after = img_after.clamp(0, 1).permute(1, 2, 0).numpy()

    if mask_t_after.ndim == 3:
        m_after = mask_t_after[0].detach().cpu().numpy().astype(np.float32)
    else:
        m_after = mask_t_after.detach().cpu().numpy().astype(np.float32)

    # --- Save BEFORE mask BW ---
    plt.figure()
    plt.imshow(mask_before, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_before_mask_bw.png", dpi=200)
    plt.close()

    # --- Save BEFORE overlay ---
    plt.figure()
    plt.imshow(img_before)
    plt.imshow(mask_before, cmap="gray", alpha=0.5, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_before_overlay.png", dpi=200)
    plt.close()

    # --- Save AFTER mask BW ---
    plt.figure()
    plt.imshow(m_after, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_after_mask_bw.png", dpi=200)
    plt.close()

    # --- Save AFTER overlay ---
    plt.figure()
    plt.imshow(img_after)
    plt.imshow(m_after, cmap="gray", alpha=0.5, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_after_overlay.png", dpi=200)
    plt.close()

def process_single_image(
    img_path: Path,
    predictor: SAM2ImagePredictor,
    cfg: Dict,
    get_label_fn: Callable[[Path], int],
    onnx_session: ort.InferenceSession,
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
    epsilons = list(cfg["verification"]["epsilon"])
    num_classes = int(cfg["verification"]["num_classes"])

    image = Image.open(img_path).convert("RGB")

    # resize + center crop of the img
    img_t_224 = vgg16_preprocess(image)
    image_np_224 = img_t_224.detach().cpu().numpy()  # (3,224,224)

    # Run model to verify the image is correctly classified before generating specs
    label = get_label_fn(img_path)
    print(f"True label index: {label} / {num_classes}")

    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_input = np.expand_dims(image_np_224, axis=0).astype(np.float32, copy=False)
    logits = onnx_session.run(
        None, {onnx_input_name: onnx_input}
    )[0]
    pred_label = int(np.argmax(logits, axis=1)[0])
    print(f"Predicted label index: {pred_label} / {num_classes}")
    if pred_label != label:
        print(
            f"[SKIP] Model top-1 ({pred_label}) != ground truth ({label}); skipping image."
        )
        return []

    # Second-best class for targeted misclassification (matches benchmark style)
    logits_flat = logits[0].reshape(-1)
    order = np.argsort(logits_flat)
    top1_label = int(order[-1])
    runner_up_label = int(order[-2]) if order.size >= 2 else label
    if top1_label != label:
        print(f"[WARN] Top-1 from logits ({top1_label}) != label ({label}); using label as top-1.")
        top1_label = label
    
    img_t_224 = vgg16_preprocess(image, normalize=False)       # CHW in [0,1]
    image_np_224 = img_t_224.permute(1, 2, 0).numpy()          # HWC in [0,1] for run_segmentation_model




    ver_cfg = cfg.get("verification", {})
    pixel_counts = ver_cfg.get("perturb_pixels", [None])
    # allow a single int as shorthand
    if isinstance(pixel_counts, int):
        pixel_counts = [pixel_counts]
    select = ver_cfg.get("perturb_select", "random")
    seed = int(ver_cfg.get("perturb_seed", 0))

    seg_cfg = cfg["segmentation"]

    point_coords = None
    point_labels = None

    if seg_cfg.get("prompt_mode", "grid") == "manual":
        prompts_path = Path(seg_cfg["prompts_path"])
        prompts_db = load_manual_prompts(prompts_path)

        # IMPORTANT: pick a consistent key. I suggest img basename.
        # e.g. "n01443537_12345.jpg" or whatever your img file name is.
        coords, labels = prompts_for_image(prompts_db, Path(img_path).name)

        if coords is None:
            print(f"[PROMPTS] No manual prompts found for {Path(img_path).name}; falling back to grid.")
        else:
            point_coords, point_labels = coords, labels


    # Always run segmentation (grid by default; manual prompts if provided above)
    segments = run_segmentation_model(
        predictor=predictor,
        image_np=image_np_224,
        grid_size=seg_cfg.get("grid_size", 6),
        iou_threshold=seg_cfg.get("iou_threshold", 0.8),
        score_threshold=seg_cfg.get("score_threshold", 0.0),
        max_segments=seg_cfg.get("max_segments", None),
        point_coords=point_coords,
        point_labels=point_labels,
    )

    img_basename = img_path.stem
    visualize_segments(
        image_np_224,
        segments,
        debug_dir,
        img_basename
        )
    
    csv_rows: List[Tuple[str, str, int]] = []

    # 1) VNNLIBs for original image (global perturbation), for every (eps, k)
    for eps in epsilons:
        for k in pixel_counts:
            global_vnnlib = vnnlib_dir / f"{img_basename}_global_k{k}_eps_{eps:.4f}.vnnlib"
            lb_global, ub_global = bounds_for_segment(
                image_np_224,
                eps=eps,
                mask=None,
                max_pixels=k,
                select=select,
                seed=seed,
            )

            write_vnnlib_for_segment(
                lb=lb_global,
                ub=ub_global,
                label=label,
                num_classes=num_classes,
                out_path=global_vnnlib,
                target_label=runner_up_label,
            )

            global_stats = report_changed_inputs(
                lb_flat=lb_global,
                ub_flat=ub_global,
                tag=f"{img_basename}_global",
                extra={
                    "image": img_basename,
                    "is_global": True,
                    "segment_index": -1,
                    "eps": eps,
                    "k": k,
                    "score": None,
                    "bbox": None,
                    "pattern": "global",
                },
            )
            if stats_rows is not None:
                stats_rows.append(global_stats)

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
        # numpy HxW (bool or 0/1)
        mask = np.asarray(seg["mask"]).astype(bool)
        # ------------------------------------------------
        # (A) FIX MASK: only NON-MASK region can move
        # ------------------------------------------------
        for eps in epsilons:
            for k in pixel_counts:
                # FIX MASK: object is frozen => allow changes only in NON-mask
                lb_fix_mask, ub_fix_mask = bounds_for_segment(
                    image_np=image_np_224,
                    mask=~mask,
                    eps=eps,
                    max_pixels=k,
                    select=select,
                    seed=seed + seg_idx,
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
                        "k": k,
                        "score": float(seg["score"]),
                        "bbox": seg["bbox"],
                        "pattern": "fix_mask",
                    },
                )
                if stats_rows is not None:
                    stats_rows.append(seg_stats_fix_mask)

                seg_vnnlib_fix_mask = (
                    vnnlib_dir
                    / f"{img_basename}_seg{seg_idx}_fixmask_k{k}_eps_{eps:.4f}.vnnlib"
                )
                write_vnnlib_for_segment(
                    lb=lb_fix_mask,
                    ub=ub_fix_mask,
                    label=label,
                    num_classes=num_classes,
                    out_path=seg_vnnlib_fix_mask,
                    target_label=runner_up_label,
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
        # ------------------------------------------------

        for eps in epsilons:
            for k in pixel_counts:
                # FIX NON-MASK: background is frozen => allow changes only in MASK
                lb_fix_nonmask, ub_fix_nonmask = bounds_for_segment(
                    image_np=image_np_224,
                    mask=mask,
                    eps=eps,
                    max_pixels=k,
                    select=select,
                    seed=seed + seg_idx,
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
                        "k": k,
                        "score": float(seg["score"]),
                        "bbox": seg["bbox"],
                        "pattern": "fix_nonmask",
                    },
                )
                if stats_rows is not None:
                    stats_rows.append(seg_stats_fix_nonmask)

                seg_vnnlib_fix_nonmask = (
                    vnnlib_dir
                    / f"{img_basename}_seg{seg_idx}_fixnonmask_k{k}_eps_{eps:.4f}.vnnlib"
                )
                write_vnnlib_for_segment(
                    lb=lb_fix_nonmask,
                    ub=ub_fix_nonmask,
                    label=label,
                    num_classes=num_classes,
                    out_path=seg_vnnlib_fix_nonmask,
                    target_label=runner_up_label,
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


def expand_image_entries(entries: Iterable[Union[str, Path]]) -> List[Path]:
    """
    Normalize a mixed list of files/dirs into a list of image files.
    Directories are expanded to common image formats; non-dirs are passed through.
    """
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    expanded: List[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            expanded.extend(
                sorted(
                    f
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in allowed_suffixes
                )
            )
        else:
            expanded.append(path)
    return expanded


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
    sys.argv += ["--config", "configs/xaiv/vggnet16_benchmark2022_segmented.yaml"]


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

    ort.set_default_logger_severity(3)

    onnx_rel_path = str(Path("onnx") / onnx_target.name)
    onnx_session = ort.InferenceSession(str(onnx_target))

    # -----------------------------
    # Dataset
    # -----------------------------

    all_images, img_to_label, num_classes = load_imagenet_vggnet16_metadata(
        cfg["dataset"]
    )

    def get_label(img_path: Path) -> int:
        return get_imagenet_label_for_image(img_path, img_to_label)

    print(f"[Dataset] Found {len(all_images)} images; num_classes = {num_classes}")
    cfg.setdefault("verification", {})
    cfg["verification"]["num_classes"] = num_classes

    # -----------------------------
    # Which images to process?
    # -----------------------------
    images_to_process: List[Path] = []


    cfg_indices = cfg.get("run", {}).get("indices", [])

    if cfg_indices is not None and len(cfg_indices) > 0:
        for idx in cfg_indices:
            images_to_process.append(all_images[idx])
    else:
        if args.images:
            images_to_process.extend(expand_image_entries(args.images))
        cfg_paths = cfg.get("run", {}).get("image_paths", [])
        if isinstance(cfg_paths, (str, Path)):
            cfg_paths = [cfg_paths]
        for idx in cfg_indices:
            images_to_process.append(all_images[idx])
        images_to_process.extend(expand_image_entries(cfg_paths))

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
            onnx_session=onnx_session,
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
    with instances_csv_path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(all_rows)

    if change_stats_rows:
        fieldnames = [
            "image",
            "tag",
            "is_global",
            "segment_index",
            "eps",
            "k",
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
