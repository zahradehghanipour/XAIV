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

from typing import Dict, List, Optional

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor


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

    image_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
    predictor.set_image(image_uint8)

    ys = np.linspace(0, H - 1, grid_size).astype(int)
    xs = np.linspace(0, W - 1, grid_size).astype(int)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    point_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    point_labels = np.ones(len(point_coords), dtype=np.int64)

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