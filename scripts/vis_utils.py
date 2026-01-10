"""
vis_utils.py
-------------
Utility functions for visualizing segmentation masks.

Inputs:
    - Original image (H×W×3)
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
from PIL import Image, ImageDraw


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
      - black & white mask image for each segment
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save original image ---
    image_uint8 = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
    orig_path = out_dir / f"{img_basename}_original.png"
    Image.fromarray(image_uint8).save(orig_path)
    print(f"[VIS] Saved original image to: {orig_path}")

    for i, seg in enumerate(segments[:max_vis]):
        mask = seg["mask"]          # boolean or {0,1} mask
        score = seg["score"]
        bbox = seg["bbox"]
        num_pixels = int(mask.sum())

        print(
            f"[VIS] Segment {i}: score={score:.3f}, "
            f"bbox={bbox}, num_pixels={num_pixels}"
        )

        # --- Overlay visualization (red mask) ---
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

        # --- Black & white mask image ---
        bw_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        bw_mask[mask] = 255  # white = segment, black = background

        bw_img = Image.fromarray(bw_mask, mode="L")
        bw_path = out_dir / f"{img_basename}_seg{i}_mask_bw.png"
        bw_img.save(bw_path)


def visualize_selected_masks_on_image(
    image_np: np.ndarray,
    mask_in: np.ndarray,
    mask_out: np.ndarray,
    out_path: str,
    x_size: int = 6,
    x_width: int = 2,
):
    """
    Visualize selected pixels with:
      - RED X  : selected pixels inside the segmentation mask
      - BLUE X : selected pixels outside the segmentation mask
    """
    img = np.asarray(image_np)
    mi = np.asarray(mask_in).astype(bool)
    mo = np.asarray(mask_out).astype(bool)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image_np must be HxWx3, got {img.shape}")
    if mi.shape != img.shape[:2] or mo.shape != img.shape[:2]:
        raise ValueError("Mask/image shape mismatch")
    if np.any(mi & mo):
        raise ValueError("mask_in and mask_out must be disjoint")

    # Convert image to uint8
    if img.dtype != np.uint8:
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        img_uint8 = img

    im = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(im)

    # RED X → mask_in
    ys, xs = np.where(mi)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.line([(x - x_size, y - x_size), (x + x_size, y + x_size)],
                  fill=(255, 0, 0), width=x_width)
        draw.line([(x - x_size, y + x_size), (x + x_size, y - x_size)],
                  fill=(255, 0, 0), width=x_width)

    # BLUE X → mask_out
    ys, xs = np.where(mo)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.line([(x - x_size, y - x_size), (x + x_size, y + x_size)],
                  fill=(0, 0, 255), width=x_width)
        draw.line([(x - x_size, y + x_size), (x + x_size, y - x_size)],
                  fill=(0, 0, 255), width=x_width)

    im.save(out_path)
    print(
        f"[VIS] saved → {out_path} "
        f"(in-mask={mi.sum()}, out-mask={mo.sum()})"
    )