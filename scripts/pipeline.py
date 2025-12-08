#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import binary_dilation

from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch


# ============================================================
# 1) Config & basic helpers
# ============================================================

def load_config(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ============================================================
# 2b) ImageNet VGGNet16 benchmark helpers
# ============================================================
from pathlib import Path
from typing import Dict, List, Tuple

def load_imagenet_vggnet16_metadata(dataset_cfg: Dict) -> Tuple[List[Path],
                                                                Dict[str, int],
                                                                int]:
    """
    Expects a flat directory like:

        root/
          n01440764_tench.JPEG
          n01443537_goldfish.JPEG
          ...

    Filenames are of the form: <wnid>_<human-readable>.JPEG
    We map each distinct <wnid> to a label index 0..K-1.

    Returns:
      all_images: sorted list of image Paths
      img_to_label: dict[str(path)] -> int label
      num_classes: number of distinct wnids
    """
    root = Path(dataset_cfg["root"])

    # If you also have .jpg, you can widen this glob:
    # all_images = sorted(list(root.glob("*.JPEG")) + list(root.glob("*.jpg")))
    all_images: List[Path] = sorted(root.glob("*.JPEG"))

    if not all_images:
        raise RuntimeError(f"No .JPEG files found in {root}")

    # Collect all wnids from filenames
    wnids_set = set()
    for img_path in all_images:
        stem = img_path.stem  # e.g. "n01440764_tench"
        wnid = stem.split("_")[0]  # "n01440764"
        wnids_set.add(wnid)

    wnids_sorted = sorted(wnids_set)
    wnid_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(wnids_sorted)}

    img_to_label: Dict[str, int] = {}
    for img_path in all_images:
        stem = img_path.stem
        wnid = stem.split("_")[0]
        label = wnid_to_idx[wnid]
        img_to_label[str(img_path)] = label

    num_classes = len(wnids_sorted)

    print(f"[ImageNet] Found {len(all_images)} images in {num_classes} classes")
    return all_images, img_to_label, num_classes


def get_imagenet_label_for_image(
    img_path: Path,
    img_to_label: Dict[str, int],
) -> int:
    return img_to_label[str(img_path)]

def get_imagenet_label_for_image(
    img_path: Path,
    img_to_label: Dict[str, int],
) -> int:
    return img_to_label[str(img_path)]

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 2) TinyImageNet helpers: images + labels
# ============================================================

def load_tinyimagenet_metadata(dataset_cfg: Dict) -> Tuple[List[Path],
                                                           Dict[str, str],
                                                           Dict[str, int],
                                                           int]:
    """
    Returns:
      all_images: sorted list of validation image paths
      img_to_wnid: map filename -> wnid
      wnid_to_idx: map wnid -> class index
      num_classes: number of classes
    """
    root = Path(dataset_cfg["root"])
    val_images_dir = root / dataset_cfg["val_images_subdir"]
    val_ann_path = root / dataset_cfg["val_annotations"]
    wnids_path = root / dataset_cfg["wnids"]

    all_images = sorted(val_images_dir.glob("*.JPEG"))

    # map image name -> wnid
    img_to_wnid: Dict[str, str] = {}
    with val_ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_name, wnid = parts[0], parts[1]
            img_to_wnid[img_name] = wnid

    # wnid -> idx
    wnids: List[str] = []
    with wnids_path.open("r") as f:
        for line in f:
            wn = line.strip()
            if wn:
                wnids.append(wn)

    wnid_to_idx: Dict[str, int] = {wn: i for i, wn in enumerate(wnids)}
    num_classes = len(wnids)

    return all_images, img_to_wnid, wnid_to_idx, num_classes


def get_true_label_for_image(img_path: Path,
                             img_to_wnid: Dict[str, str],
                             wnid_to_idx: Dict[str, int]) -> int:
    name = img_path.name
    wnid = img_to_wnid[name]
    return wnid_to_idx[wnid]


# ============================================================
# 3) SAM2 loading + segmentation
# ============================================================

def load_sam2_predictor(seg_cfg: Dict) -> SAM2ImagePredictor:
    device_cfg = seg_cfg.get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    print(f"[SAM2] Using device: {device}")

    predictor = SAM2ImagePredictor.from_pretrained(
        seg_cfg["model_name"],
        device=device,
    )
    predictor.model.eval()
    return predictor

def run_segmentation_model(
    predictor: SAM2ImagePredictor,
    image_np: np.ndarray,
    grid_size: int = 6,
    iou_threshold: float = 0.8,
    score_threshold: float = 0.0,
    max_segments: Optional[int] = None,
):
    """
    image_np: H x W x 3 in [0,1]
    Returns a list of segments:
      {"bbox": [x_min, y_min, x_max, y_max],
       "mask": 2D bool (H,W),
       "score": float}
    """
    H, W, _ = image_np.shape

    # --- 1) set image for SAM2 ---
    image_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
    predictor.set_image(image_uint8)

    # --- 2) grid of positive points ---
    ys = np.linspace(0, H - 1, grid_size).astype(int)
    xs = np.linspace(0, W - 1, grid_size).astype(int)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    point_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    point_labels = np.ones(len(point_coords), dtype=np.int64)

    # --- 3) predict masks ---
    masks, scores, logits = predictor.predict(
        point_coords=point_coords[None, ...],
        point_labels=point_labels[None, ...],
        multimask_output=True,
    )

    # Convert to numpy and normalise shapes
    masks_np = np.array(masks)
    scores_np = np.array(scores)

    # masks: we want shape (K, H, W)
    if masks_np.ndim == 2:
        # single mask H x W
        masks_np = masks_np[None, ...]
    elif masks_np.ndim == 3 and masks_np.shape[0] != 3:
        # could already be (K, H, W) – then OK
        pass
    elif masks_np.ndim == 4:
        # sometimes comes as (1, K, H, W)
        masks_np = masks_np.reshape(-1, H, W)

    # scores: we want shape (K,)
    scores_np = np.atleast_1d(scores_np).reshape(-1)

    K = masks_np.shape[0]
    if scores_np.shape[0] != K:
        # if SAM2 gives a single score for all masks, broadcast it
        if scores_np.shape[0] == 1:
            scores_np = np.repeat(scores_np, K)
        else:
            raise RuntimeError(
                f"Shape mismatch: {K} masks but {scores_np.shape[0]} scores"
            )

    segments = []
    for k in range(K):
        score = float(scores_np[k])
        if score < score_threshold:
            continue

        mask = masks_np[k].astype(bool)
        ys_mask, xs_mask = np.where(mask)
        if len(ys_mask) == 0:
            continue

        x_min, x_max = xs_mask.min(), xs_mask.max()
        y_min, y_max = ys_mask.min(), ys_mask.max()

        segments.append(
            {
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "mask": mask,
                "score": score,
            }
        )

    # sort by score desc and maybe truncate
    segments.sort(key=lambda s: s["score"], reverse=True)
    if max_segments is not None:
        segments = segments[:max_segments]

    print(f"[SAM2] Found {len(segments)} segments after filtering")
    return segments

# ============================================================
# 4) Bounds and VNNLIB writing (from your notebook)
# ============================================================

def bounds_for_segment(
    image_np: np.ndarray,
    mask: np.ndarray,
    eps: float,
    dilation_radius: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    image_np: H x W x C in [0,1]
    mask: H x W boolean array; True = pixels where we allow perturbation
    eps: epsilon for those pixels
    dilation_radius: dilate the mask before using (0 = no dilation)
    """
    H, W, C = image_np.shape

    if dilation_radius > 0:
        struct = np.ones(
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
            dtype=bool,
        )
        mask = binary_dilation(mask, structure=struct)

    base = image_np.copy()
    lb = base.copy()
    ub = base.copy()

    # Allow +/- eps only inside the mask; outside the mask, lb=ub=base
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        for c in range(C):
            lb[y, x, c] = np.clip(base[y, x, c] - eps, 0.0, 1.0)
            ub[y, x, c] = np.clip(base[y, x, c] + eps, 0.0, 1.0)

    # Flatten as (C, H, W) → vector
    lb_flat = lb.transpose(2, 0, 1).reshape(-1)
    ub_flat = ub.transpose(2, 0, 1).reshape(-1)
    return lb_flat, ub_flat


def write_vnnlib_for_segment(
    lb_flat: np.ndarray,
    ub_flat: np.ndarray,
    label: int,
    num_classes: int,
    out_path: Path | str,
):
    """
    Simple VNNLIB writer:

      - Declares X_0..X_N-1 and Y_0..Y_{num_classes-1}
      - Constrains X_i between lb_flat[i] and ub_flat[i]
      - Property: ∃k != label : Y_k >= Y_label
    """
    out_path = Path(out_path)
    N = len(lb_flat)

    with out_path.open("w") as f:
        f.write(f"; TinyImageNet segmented property; true label {label}\n\n")

        # Declare inputs
        for i in range(N):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare outputs
        for k in range(num_classes):
            f.write(f"(declare-const Y_{k} Real)\n")
        f.write("\n")

        # Input bounds
        f.write("; Input bounds\n")
        for i in range(N):
            f.write(f"(assert (>= X_{i} {lb_flat[i]:.6f}))\n")
            f.write(f"(assert (<= X_{i} {ub_flat[i]:.6f}))\n")
        f.write("\n")

        # Misclassification property
        f.write("; Output constraints: there exists a k != label with Y_k >= Y_label\n")
        f.write("(assert (or\n")
        for k in range(num_classes):
            if k == label:
                continue
            f.write(f"    (and (>= Y_{k} Y_{label}))\n")
        f.write("))\n")

    print(f"[VNNLIB] Saved property to: {out_path.resolve()}")


# ============================================================
# 5) Per-image pipeline
# ============================================================

from typing import Callable

def process_single_image(
    img_path: Path,
    predictor: SAM2ImagePredictor,
    cfg: Dict,
    get_label_fn: Callable[[Path], int],
    onnx_rel_path: str,
    vnnlib_dir: Path,
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

    # true label
    label = get_label_fn(img_path)
    print(f"True label index: {label} / {num_classes}")

    # load image as [0,1]
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image) / 255.0

    # segment
    seg_cfg = cfg["segmentation"]
    segments = run_segmentation_model(
        predictor=predictor,
        image_np=image_np,
        grid_size=seg_cfg.get("grid_size", 6),
        iou_threshold=seg_cfg.get("iou_threshold", 0.8),
        score_threshold=seg_cfg.get("score_threshold", 0.0),
        max_segments=seg_cfg.get("max_segments", None),
    )

    img_basename = img_path.stem
    csv_rows: List[Tuple[str, str, int]] = []

    # 1) global epsilon VNNLIB
    lb_global = np.clip(image_np - eps, 0.0, 1.0)
    ub_global = np.clip(image_np + eps, 0.0, 1.0)
    lb_global_flat = lb_global.transpose(2, 0, 1).reshape(-1)
    ub_global_flat = ub_global.transpose(2, 0, 1).reshape(-1)

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

    # 2) segmented VNNLIBs
    for seg_idx, seg in enumerate(segments):
        mask = seg["mask"]
        lb_flat, ub_flat = bounds_for_segment(
            image_np=image_np,
            mask=mask,
            eps=eps,
            dilation_radius=dilation_radius,
        )

        seg_vnnlib = vnnlib_dir / f"{img_basename}_seg{seg_idx}_eps_{eps:.4f}.vnnlib"
        write_vnnlib_for_segment(
            lb_flat=lb_flat,
            ub_flat=ub_flat,
            label=label,
            num_classes=num_classes,
            out_path=seg_vnnlib,
        )

        csv_rows.append(
            (
                onnx_rel_path,
                str(Path("vnnlib") / seg_vnnlib.name),
                timeout,
            )
        )

    print(f"[PIPELINE] Created {len(csv_rows)} VNNLIBs for {img_path.name}")
    return csv_rows


# ============================================================
# 6) Main entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TinyImageNet → Segmentation → VNNLIB pipeline"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    # Optionally override images from CLI
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        help="TinyImageNet val indices (0-based) to process",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        help="Explicit image paths to process",
    )


    # TODO: delete later
    import sys
    sys.argv += ["--config", "configs/vggnet_16/imagenet_vggnet16_segmented.yaml"]

    args = parser.parse_args()

    cfg = load_config(args.config)

    # -----------------------------
    # Prepare benchmark directories
    # -----------------------------
    out_root = ensure_dir(Path(cfg["output"]["benchmark_dir"]))
    onnx_dir = ensure_dir(out_root / "onnx")
    vnnlib_dir = ensure_dir(out_root / "vnnlib")
    instances_csv_path = out_root / "instances.csv"

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

    # Path stored in instances.csv (relative to benchmark root)
    onnx_rel_path = str(Path("onnx") / onnx_target.name)

    # -----------------------------
    # Dataset metadata (TinyImageNet or ImageNet)
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
        # if CLI indices not given, fall back to config.run.indices
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
        )
        all_rows.extend(rows)

    # -----------------------------
    # Write instances.csv
    # -----------------------------
    print(f"\n[CSV] Writing {len(all_rows)} rows to {instances_csv_path}")
    with instances_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # no header – matches abCROWN style
        for row in all_rows:
            writer.writerow(row)

    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()