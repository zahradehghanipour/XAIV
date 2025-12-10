"""
dataset_utils.py
-----------------
Helpers for loading dataset metadata (TinyImageNet, ImageNet-VGG16).

Inputs:
    - Dataset configuration dictionary from YAML.
    - Paths to dataset folders:
        • TinyImageNet: val images, val_annotations.txt, wnids.txt
        • ImageNet-VGG16: flat directory of *.JPEG files

Outputs:
    - all_images : list[Path]
    - label mapping:
        • TinyImageNet: (img_name → wnid), (wnid → class_index)
        • ImageNet: (image_path → class_index)
    - num_classes : int
"""


from pathlib import Path
from typing import Dict, List, Tuple

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

    Returns:
      all_images: sorted list of image Paths
      img_to_label: dict[str(path)] -> int label
      num_classes: number of distinct wnids
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