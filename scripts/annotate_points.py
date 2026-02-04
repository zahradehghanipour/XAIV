import argparse
import json
from pathlib import Path

import cv2

from config_utils import load_config, ensure_dir
from dataset_utils import load_imagenet_vggnet16_metadata

# ===============================
# CONFIG
# ===============================

# TODO: delete later
# import sys
# sys.argv += ["--config", "configs/xaiv/vggnet16_benchmark2022_segmented_manual.yaml"]

IMG_SIZE = 224

# ===============================
# State
# ===============================
fg_points = []
bg_points = []

def mouse_callback(event, x, y, flags, param):
    global fg_points, bg_points

    if event == cv2.EVENT_LBUTTONDOWN:
        fg_points.append([x, y])
        print(f"FG point: ({x},{y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        bg_points.append([x, y])
        print(f"BG point: ({x},{y})")

# ===============================
# Resolve image + output path
# ===============================
parser = argparse.ArgumentParser(description="Annotate manual prompt points")
parser.add_argument(
    "--config",
    "-c",
    type=str,
    required=True,
    help="Path to YAML config file",
)
args = parser.parse_args()
cfg = load_config(args.config)

run_cfg = cfg.get("run", {})
image_paths = run_cfg.get("image_paths") or []
indices = run_cfg.get("indices") or []

image_path: Path | None = None
if image_paths:
    first = Path(image_paths[0])
    if first.is_dir():
        candidates = sorted(
            p
            for p in first.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jpeg"}
        )
        if not candidates:
            raise RuntimeError(f"No images found in directory {first}")
        image_path = candidates[0]
    else:
        image_path = first
else:
    all_images, _, _ = load_imagenet_vggnet16_metadata(cfg["dataset"])
    if not indices:
        raise RuntimeError("No images specified. Use run.image_paths or run.indices in config.")
    idx = int(indices[0])
    if idx < 0 or idx >= len(all_images):
        raise RuntimeError(f"Index {idx} out of range (0..{len(all_images) - 1})")
    image_path = all_images[idx]

if image_path is None:
    raise RuntimeError("Could not resolve image path from config.")

prompts_path = Path(cfg["segmentation"]["prompts_path"])
out_dir = prompts_path if prompts_path.suffix == "" else prompts_path.parent
ensure_dir(out_dir)
out_path = out_dir / image_path.with_suffix(".json").name

# ===============================
# Load + resize image
# ===============================
img = cv2.imread(str(image_path))
if img is None:
    raise RuntimeError(f"Cannot load image {image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# ===============================
# UI loop
# ===============================
cv2.namedWindow("Annotate (L=FG, R=BG, Q=quit)")
cv2.setMouseCallback("Annotate (L=FG, R=BG, Q=quit)", mouse_callback)

while True:
    vis = img.copy()

    # draw FG points (green)
    for x, y in fg_points:
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)

    # draw BG points (red)
    for x, y in bg_points:
        cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)

    cv2.imshow("Annotate (L=FG, R=BG, Q=quit)", vis)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

# ===============================
# Save JSON
# ===============================
image_key = image_path.name
prompts = {
    image_key: {
        "foreground": fg_points,
        "background": bg_points,
    }
}

out_path.write_text(json.dumps(prompts, indent=2))
print(f"\nSaved prompts to {out_path}")
