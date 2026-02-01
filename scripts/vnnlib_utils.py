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
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from collections import deque

def mask_to_224(mask_np: np.ndarray) -> np.ndarray:
    # mask_np: HxW boolean or 0/1
    m = Image.fromarray(mask_np.astype(np.uint8) * 255)
    m = m.resize((224, 224), Image.NEAREST)  # IMPORTANT: keep mask crisp
    return (np.asarray(m) > 127)

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

    # prefix = f"[VNNLIB][{tag}] " if tag else "[VNNLIB] "
    # print(
    #     f"{prefix}Inputs changed: {num_changed}, "
    #     f"fixed: {num_fixed}, total: {total}, "
    #     f"percent_changed={percent_changed:.2f}%"
    # )

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



def bounds_for_segment(
    image_np: np.ndarray,
    eps: float,
    mask: Optional[np.ndarray] = None,
    max_pixels: Optional[int] = None,
    select: str = "random",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create (lb, ub) bounds with an optional *pixel budget*.

    Semantics:
      - If ``mask`` is provided: pixels where mask==True are eligible to change.
        Else: all pixels are eligible.
      - If ``max_pixels`` is provided: at most that many *spatial* pixels (H×W)
        are allowed to change. (All 3 RGB channels for a chosen pixel change.)

    Notes:
      - ``image_np`` must be H×W×C in [0,1] (unnormalized).
      - ``select`` currently supports: "random".
    """

    img = np.asarray(image_np, dtype=np.float32)
    if img.ndim != 3:
        raise ValueError(f"image_np must be 3D (HWC); got shape {img.shape}")
    H, W, C = img.shape
    if C != 3:
        # Not strictly required, but this matches your VGG16 pipeline.
        raise ValueError(f"Expected 3 channels (RGB); got C={C}")

    mask_bool = np.ones((H, W), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool.shape != (H, W):
        raise ValueError(f"mask shape {mask_bool.shape} must match spatial dims {(H, W)}")

    if max_pixels is not None:
        max_pixels = int(max_pixels)
        if max_pixels < 0:
            raise ValueError("max_pixels must be >= 0")
        ys, xs = np.where(mask_bool)
        n = int(ys.size)

        if n > max_pixels:
            if select != "random":
                raise ValueError(f"Unknown select='{select}'. Only 'random' is implemented.")
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=max_pixels, replace=False)
            limited = np.zeros((H, W), dtype=bool)
            limited[ys[idx], xs[idx]] = True
            mask_bool = limited

    eps_tensor = np.where(mask_bool[:, :, None], float(eps), 0.0).astype(np.float32)
    lb = img - eps_tensor
    ub = img + eps_tensor
    return lb.reshape(-1), ub.reshape(-1)

def write_vnnlib_for_segment(
    lb: np.ndarray,
    ub: np.ndarray,
    label: int,
    num_classes: int,
    out_path: Path | str,
    target_label: int | None = None,
):
    """
    Simple VNNLIB writer:

      - Declares X_0..X_N-1 and Y_0..Y_{num_classes-1}
      - Constrains X_i between lb_flat[i] and ub_flat[i]
      - Property: targeted misclassification (Y_target >= Y_label) when
        target_label is provided, otherwise untargeted (∃k != label : Y_k >= Y_label)
    """

    out_path = Path(out_path)
    N = len(lb)

    with out_path.open("w") as f:
        f.write(f"; ImageNet segmented property; true label {label}\n\n")

        for i in range(N):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        for k in range(num_classes):
            f.write(f"(declare-const Y_{k} Real)\n")
        f.write("\n")

        f.write("; Input bounds\n")
        for i in range(N):
            f.write(f"(assert (>= X_{i} {lb[i]:.6f}))\n")
            f.write(f"(assert (<= X_{i} {ub[i]:.6f}))\n")
        f.write("\n")

        if target_label is not None:
            f.write("; Output constraint: targeted misclassification\n")
            f.write(f"(assert (>= Y_{target_label} Y_{label}))\n")
        else:
            f.write("; Output constraints: there exists a k != label with Y_k >= Y_label\n")
            f.write("(assert (or\n")
            for k in range(num_classes):
                if k == label:
                    continue
                f.write(f"    (and (>= Y_{k} Y_{label}))\n")
            f.write("))\n")

    # print(f"[VNNLIB] Saved property to: {out_path.resolve()}")
