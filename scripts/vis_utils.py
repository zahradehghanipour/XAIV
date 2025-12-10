"""
vis_utils.py
-------------
Utility functions for visualizing segmentation masks.

Inputs:
    - Original image (H×W×3, float or uint8)
    - A list of segment dicts containing masks and scores
    - Output directory for debug images

Outputs:
    - Saved PNG files:
        • original image
        • top-k segments overlaid in red
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def visualize_segments(
    image_np: np.ndarray,
    segments: List[Dict],
    out_dir: Path,
    img_basename: str,
    max_vis: int = 5,
):
    """
    Save:
      - original image
      - image with top-k segments overlaid as red regions
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    image_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
    orig_path = out_dir / f"{img_basename}_original.png"
    Image.fromarray(image_uint8).save(orig_path)
    print(f"[VIS] Saved original image to: {orig_path}")

    for i, seg in enumerate(segments[:max_vis]):
        mask = seg["mask"]
        score = seg["score"]
        bbox = seg["bbox"]
        num_pixels = int(mask.sum())

        print(
            f"[VIS] Segment {i}: score={score:.3f}, "
            f"bbox={bbox}, num_pixels={num_pixels}"
        )

        overlay = image_uint8.copy()
        red = np.array([255, 0, 0], dtype=np.uint8)
        alpha = 0.5

        overlay[mask] = (
            (1 - alpha) * overlay[mask].astype(np.float32)
            + alpha * red.astype(np.float32)
        ).astype(np.uint8)

        seg_img = Image.fromarray(overlay)
        seg_path = out_dir / f"{img_basename}_seg{i}_score_{score:.3f}.png"
        seg_img.save(seg_path)
        print(f"[VIS] Saved segment {i} overlay to: {seg_path}")