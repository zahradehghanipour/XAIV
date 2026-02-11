"""
sam2_utils.py
--------------
Functions for loading SAM2 and generating segmentation masks.

Inputs:
    - Segmentation config dict (model name, device, grid size, etc.)
    - Image tensors in numpy format (H × W × 3, float in [0,1])

Outputs:
    - A loaded SAM2ImagePredictor model ready for inference.
    - A list of segmentation dictionaries:
        {
            "mask": bool H×W array,
            "bbox": [x_min, y_min, x_max, y_max],
            "score": float
        }
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_dilation

def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask

    # circular-ish structuring element
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    selem = x*x + y*y <= radius*radius

    return binary_dilation(mask, structure=selem)

def run_segmentation_model(
    predictor: SAM2ImagePredictor,
    image_np: np.ndarray,
    grid_size: int = 6,
    iou_threshold: float = 0.8,
    score_threshold: float = 0.0,
    max_segments: Optional[int] = None,
    dilation_radius: int = 0,
    point_coords: Optional[np.ndarray] = None,   # (N,2) in (x,y)
    point_labels: Optional[np.ndarray] = None,   # (N,) 1=FG, 0=BG
):
    """
    image_np: H x W x 3 in [0,1]

    If point_coords/point_labels are provided, they are used as prompts.
    Otherwise a grid prompt is generated.
    """
    H, W = image_np.shape[:2]
    predictor.set_image(image_np)

    # --- choose prompts ---
    if point_coords is None or point_labels is None:
        # grid prompts (your current behavior)
        xs = np.linspace(0, W - 1, grid_size, dtype=np.float32)
        ys = np.linspace(0, H - 1, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        point_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        point_labels = np.ones(len(point_coords), dtype=np.int64)
    else:
        point_coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
        point_labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)

        if len(point_coords) != len(point_labels):
            raise ValueError("point_coords and point_labels must have same length")

    masks, scores, logits = predictor.predict(
        point_coords=point_coords[None, ...],
        point_labels=point_labels[None, ...],
        multimask_output=True,
    )

    masks_np = np.array(masks)
    scores_np = np.array(scores)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim == 4:
        masks_np = masks_np.reshape(-1, H, W)

    scores_np = np.atleast_1d(scores_np).reshape(-1)
    K = masks_np.shape[0]

    if scores_np.shape[0] != K:
        if scores_np.shape[0] == 1:
            scores_np = np.repeat(scores_np, K)
        else:
            raise RuntimeError(
                f"Shape mismatch: {K} masks but {scores_np.shape[0]} scores"
            )

    segments: List[Dict] = []
    for k in range(K):
        score = float(scores_np[k])
        if score < score_threshold:
            continue

        mask = masks_np[k].astype(bool)

        if dilation_radius > 0:
            mask = dilate_mask(mask, dilation_radius)

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

    segments.sort(key=lambda s: s["score"], reverse=True)
    if max_segments is not None:
        segments = segments[:max_segments]

    print(f"[SAM2] Found {len(segments)} segments after filtering")
    return segments


def load_sam2_predictor(seg_cfg: Dict) -> SAM2ImagePredictor:
    """Load a SAM2ImagePredictor with device selection."""
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