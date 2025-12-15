"""
vnnlib_utils.py
----------------
Functions for converting segments to perturbation bounds and writing VNNLIB files.

Inputs:
    - image_np : H×W×C float image, normalized in [0,1]
    - segmentation mask (H×W boolean)
    - eps : float perturbation radius
    - dilation radius : int
    - label : ground-truth class
    - num_classes : total number of classes

Outputs:
    - (lb_flat, ub_flat): flattened input bounds for each pixel
    - statistical dictionaries describing how many inputs changed
    - .vnnlib files encoding verification constraints

"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image


def to_224_rgb01(img: "Image.Image | np.ndarray") -> np.ndarray:
    # img can be PIL or HWC numpy
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        img_pil = Image.fromarray(img_u8).convert("RGB")
    else:
        img_pil = img.convert("RGB")

    img_pil = img_pil.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    assert arr.shape == (224, 224, 3), f"Resized image wrong shape: {arr.shape}"
    return arr

def mask_to_224(mask_np: np.ndarray) -> np.ndarray:
    # mask_np: HxW boolean or 0/1
    m = Image.fromarray(mask_np.astype(np.uint8) * 255)
    m = m.resize((224, 224), Image.NEAREST)  # IMPORTANT: keep mask crisp
    return (np.asarray(m) > 127)

def bounds_for_segment(
    image_np: np.ndarray,
    mask: np.ndarray,
    eps: float,
    dilation_radius: int = 0,
    freeze_mask: bool = False,
    expected_hw: tuple[int,int] | None = (224,224)
) -> Tuple[np.ndarray, np.ndarray]:

    # --- 1. Resize FIRST ---
    image_np = to_224_rgb01(image_np)   # <-- FIXED
    mask = mask_to_224(mask)            # <-- FIXED

    # --- 2. Now check shape ---
    assert image_np.ndim == 3, f"image_np must be HxWxC, got {image_np.shape}"
    H, W, C = image_np.shape
    assert C == 3, f"Expected 3 channels, got C={C}"

    if expected_hw is not None:
        eh, ew = expected_hw
        assert (H, W) == (eh, ew), f"Expected {(eh, ew)}, got {(H, W)}"

    # --- 3. Dilation on RESIZED mask ---
    if dilation_radius > 0:
        struct = np.ones(
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
            dtype=bool,
        )
        mask = binary_dilation(mask, structure=struct)

    base = image_np.copy()
    lb = base.copy()
    ub = base.copy()

    if freeze_mask:
        free_region = ~mask
    else:
        free_region = mask

    ys, xs = np.where(free_region)
    for y, x in zip(ys, xs):
        for c in range(C):
            lb[y, x, c] = np.clip(base[y, x, c] - eps, 0.0, 1.0)
            ub[y, x, c] = np.clip(base[y, x, c] + eps, 0.0, 1.0)

    lb_flat = lb.transpose(2, 0, 1).reshape(-1)
    ub_flat = ub.transpose(2, 0, 1).reshape(-1)

    # --- 4. Final sanity check ---
    assert lb_flat.size == 3 * 224 * 224, f"Wrong input size: {lb_flat.size}"

    return lb_flat, ub_flat

def report_changed_inputs(
    lb_flat: np.ndarray,
    ub_flat: np.ndarray,
    tag: str = "",
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Compute how many input dimensions are allowed to change (lb != ub)
    vs how many are fixed (lb == ub). Also returns percentage.
    """
    assert lb_flat.shape == ub_flat.shape
    total = lb_flat.size
    changed_mask = np.abs(ub_flat - lb_flat) > 1e-9
    num_changed = int(changed_mask.sum())
    num_fixed = total - num_changed
    percent_changed = 100.0 * num_changed / total if total > 0 else 0.0

    prefix = f"[VNNLIB][{tag}] " if tag else "[VNNLIB] "
    print(
        f"{prefix}Inputs changed: {num_changed}, "
        f"fixed: {num_fixed}, total: {total}, "
        f"percent_changed={percent_changed:.2f}%"
    )

    row = {
        "tag": tag,
        "num_changed": num_changed,
        "num_fixed": num_fixed,
        "total": total,
        "percent_changed": percent_changed,
    }
    if extra:
        row.update(extra)
    return row

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

    N = len(lb_flat)
    print(f"[VNNLIB] Writing with N={N} inputs (expect 150528 for 3x224x224)")

    out_path = Path(out_path)
    N = len(lb_flat)

    with out_path.open("w") as f:
        f.write(f"; TinyImageNet segmented property; true label {label}\n\n")

        for i in range(N):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        for k in range(num_classes):
            f.write(f"(declare-const Y_{k} Real)\n")
        f.write("\n")

        f.write("; Input bounds\n")
        for i in range(N):
            f.write(f"(assert (>= X_{i} {lb_flat[i]:.6f}))\n")
            f.write(f"(assert (<= X_{i} {ub_flat[i]:.6f}))\n")
        f.write("\n")

        f.write("; Output constraints: there exists a k != label with Y_k >= Y_label\n")
        f.write("(assert (or\n")
        for k in range(num_classes):
            if k == label:
                continue
            f.write(f"    (and (>= Y_{k} Y_{label}))\n")
        f.write("))\n")

    print(f"[VNNLIB] Saved property to: {out_path.resolve()}")