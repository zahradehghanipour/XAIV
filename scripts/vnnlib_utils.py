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

def _ensure_hwc01(image_np, size=224):
    """
    Returns float32 HWC in [0,1], shape (size,size,3).
    Accepts HWC or CHW.
    """
    if hasattr(image_np, "detach"):  # torch tensor
        image_np = image_np.detach().cpu().numpy()

    img = np.asarray(image_np)

    if img.ndim != 3:
        raise ValueError(f"image_np must be 3D, got {img.shape}")

    # CHW -> HWC
    if img.shape[0] == 3 and img.shape[1] == size and img.shape[2] == size:
        img = np.transpose(img, (1, 2, 0))

    if not (img.shape[0] == size and img.shape[1] == size and img.shape[2] == 3):
        raise ValueError(f"Expected image shape ({size},{size},3) or (3,{size},{size}), got {img.shape}")

    img = img.astype(np.float32)

    # If it came as uint8
    if img.max() > 1.5:
        img = img / 255.0

    # Safety clip
    img = np.clip(img, 0.0, 1.0)
    return img

def _ensure_hw_bool(mask, size=224):
    """
    Returns bool mask shape (size,size).
    Accepts:
      - (H,W) bool/int
      - (1,H,W)
      - (H,W,1)
      - (3,H,W) / (H,W,3) (will collapse via any channel)
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    m = np.asarray(mask)

    # Collapse common shapes
    if m.ndim == 3:
        # (1,H,W) -> (H,W)
        if m.shape[0] == 1 and m.shape[1] == size and m.shape[2] == size:
            m = m[0]
        # (H,W,1) -> (H,W)
        elif m.shape[0] == size and m.shape[1] == size and m.shape[2] == 1:
            m = m[..., 0]
        # (H,W,3) -> (H,W)
        elif m.shape[0] == size and m.shape[1] == size and m.shape[2] == 3:
            m = np.any(m > 0, axis=2)
        # (3,H,W) -> (H,W)
        elif m.shape[0] == 3 and m.shape[1] == size and m.shape[2] == size:
            m = np.any(m > 0, axis=0)
        else:
            raise ValueError(f"Unrecognized mask shape: {m.shape}")

    if m.ndim != 2:
        raise ValueError(f"mask must become 2D, got {m.shape}")

    if not (m.shape[0] == size and m.shape[1] == size):
        raise ValueError(f"Expected mask shape ({size},{size}), got {m.shape}")

    # Make boolean
    m = (m > 0)
    return m

def _debug_mask_stats(m_bool, tag=""):
    total = m_bool.size
    ones = int(m_bool.sum())
    zeros = total - ones
    ones_pct = 100.0 * ones / total if total else 0.0
    zeros_pct = 100.0 * zeros / total if total else 0.0
    uniq = [0, 1] if (ones > 0 and zeros > 0) else ([1] if ones == total else [0])
    print(f"[DEBUG][MASK]{'['+tag+']' if tag else ''} unique={uniq} "
          f"ones={ones_pct:.2f}% zeros={zeros_pct:.2f}% (H={m_bool.shape[0]}, W={m_bool.shape[1]})")


def bounds_for_segment(
    image_np,
    mask,
    eps=0.0039,
    clip_min=0.0,
    clip_max=1.0,
    flatten_order="NCHW",  # "NCHW" or "NHWC"
    debug_tag="",
):
    """
    Create per-variable bounds for VNNLIB.

    Inputs:
      image_np: resized/cropped image (224x224) as HWC or CHW, in [0,1] (or uint8).
      mask:     segment mask aligned to image, various shapes accepted; treated as foreground=True where >0.

    Returns:
      lb_flat, ub_flat: 1D float32 arrays of length 224*224*3
                        flattened in specified order.
    """
    img = _ensure_hwc01(image_np, size=224)         # (224,224,3) float32
    m   = _ensure_hw_bool(mask, size=224)           # (224,224) bool
    if debug_tag:
        _debug_mask_stats(m, tag=debug_tag)

    # Expand mask to channels
    m3 = np.repeat(m[:, :, None], 3, axis=2)        # (224,224,3) bool

    eps = np.where(m3, eps, eps).astype(np.float32)  # (224,224,3)

    lb = np.clip(img - eps, clip_min, clip_max).astype(np.float32)
    ub = np.clip(img + eps, clip_min, clip_max).astype(np.float32)

    # Flatten in the same order as your VNNLIB variable indexing expects
    if flatten_order.upper() == "NHWC":
        lb_flat = lb.reshape(-1)
        ub_flat = ub.reshape(-1)
    elif flatten_order.upper() == "NCHW":
        lb_flat = np.transpose(lb, (2, 0, 1)).reshape(-1)  # C,H,W
        ub_flat = np.transpose(ub, (2, 0, 1)).reshape(-1)
    else:
        raise ValueError("flatten_order must be 'NCHW' or 'NHWC'")

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