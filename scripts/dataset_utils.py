"""
dataset_utils.py
-----------------
Helpers for loading dataset metadata.

Supported dataset types:
  - imagenet_vgg16: flat directory of *.JPEG named like <wnid>_<name>.JPEG
  - cifar100_decoded: decoded images named like label_<int>__idx_<int>.png (or .jpg/.jpeg)

Outputs:
  - all_images : list[Path]
  - img_to_label : dict[str(path)] -> int label
  - num_classes : int
"""

from pathlib import Path
from typing import Dict, List, Tuple
import re


# -------------------------
# ImageNet-VGG16
# -------------------------
def load_imagenet_vggnet16_metadata(
    dataset_cfg: Dict,
) -> Tuple[List[Path], Dict[str, int], int]:
    """
    Expects a flat directory like:

        root/
          n01440764_tench.JPEG
          n01443537_goldfish.JPEG
          ...

    Filenames are of the form: <wnid>_<human-readable>.JPEG
    """
    root = Path(dataset_cfg["root"])

    all_images: List[Path] = sorted(root.glob("*.JPEG"))
    if not all_images:
        raise RuntimeError(f"No .JPEG files found in {root}")

    wnids_set = set()
    for img_path in all_images:
        stem = img_path.stem  # e.g. "n01440764_tench"
        wnid = stem.split("_")[0]
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


# -------------------------
# CIFAR-100 decoded
# -------------------------

_CIFAR_LABEL_RE = re.compile(r"label_(?P<label>\d+)", re.IGNORECASE)

def load_cifar100_decoded_metadata(
    dataset_cfg: Dict,
) -> Tuple[List[Path], Dict[str, int], int]:
    """
    Expects decoded images in a directory (recursively), e.g.

        root/
          label_0__idx_2404.png
          label_10__idx_7516.png
          subdir/label_3__idx_123.png
          ...

    Label is parsed from the filename: label_<int>...
    """
    root = Path(dataset_cfg["root"])
    if not root.exists():
        raise RuntimeError(f"[CIFAR100] root does not exist: {root}")

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    all_images = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not all_images:
        raise RuntimeError(f"[CIFAR100] No image files found under {root}")

    img_to_label: Dict[str, int] = {}
    labels = set()

    bad = []
    for p in all_images:
        m = _CIFAR_LABEL_RE.search(p.name)
        if not m:
            bad.append(p)
            continue
        lab = int(m.group("label"))
        img_to_label[str(p)] = lab
        labels.add(lab)

    if bad:
        sample = "\n".join(str(x) for x in bad[:5])
        raise RuntimeError(
            f"[CIFAR100] Could not parse label_### from {len(bad)} files. Sample:\n{sample}"
        )

    # CIFAR-100 should be 100 classes; but we infer safely
    num_classes = max(labels) + 1 if labels else 100

    print(f"[CIFAR100] Found {len(all_images)} images; inferred num_classes={num_classes}")
    return all_images, img_to_label, num_classes


def get_cifar_label_for_image(
    img_path: Path,
    img_to_label: Dict[str, int],
) -> int:
    return img_to_label[str(img_path)]