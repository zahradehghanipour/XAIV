# XAIV

Segmentation-guided semantic verification benchmarks for image classifiers.

## Abstract

XAIV studies semantic robustness by replacing unconstrained, image-wide perturbations with segment-aware threat models. Starting from a correctly classified image, the pipeline uses SAM2 to extract a foreground mask, converts that mask into verification bounds, and emits VNNLIB properties that can be solved by `alpha-beta-CROWN`. The repository contains the full benchmark-generation workflow, bundled benchmark assets for VGG16/ImageNet and CIFAR-100, verification configs, checked-in result tables, and the notebooks used to produce paper figures.

The central question is simple: how does the verification outcome change when perturbations are restricted to semantically meaningful regions such as the object itself or its background? XAIV answers that question with three comparable property families per image: global perturbations, object-only perturbations, and background-only perturbations.

## Verification Setting

Given an image `x`, a perturbation radius `eps`, a pixel budget `k`, and a segmentation mask `M`, XAIV builds three supports:

- `global`: any pixel may change
- `fix_nonmask`: only pixels inside `M` may change
- `fix_mask`: only pixels outside `M` may change

For allowed pixels, the pipeline writes interval bounds of the form `x' in [x - eps, x + eps]`; all other pixels are fixed. The resulting property is serialized as VNNLIB and paired with an ONNX classifier.

Two dataset-specific output specifications are used:

- ImageNet/VGG16: targeted runner-up property
- CIFAR-100: untargeted VNN-COMP-style property

The naming is slightly counterintuitive but important:

- `fix_nonmask` corresponds to object-only perturbations
- `fix_mask` corresponds to background-only perturbations

## Method Overview

The main pipeline lives in `scripts/pipeline.py` and performs the following steps:

1. Load a dataset image and keep only inputs that the ONNX model classifies correctly.
2. Run SAM2 to obtain one or more segmentation masks.
3. Construct perturbation bounds for the global, object, and background settings.
4. Normalize and reorder bounds as required by the target ONNX model.
5. Write VNNLIB files, `instances.csv`, change-statistics tables, and debug visualizations.
6. Run `alpha-beta-CROWN` on the generated benchmark and analyze the results in the notebooks under `plots/` and `analysis/`.

In addition to the raw VNNLIBs, XAIV records how much of the input space was actually released for each property through `input_change_stats.csv`, and when visualization is enabled it saves the exact post-processed property masks used for verification.


## Environment Setup

Two separate environments are provided:

- `xaiv.yml` for segmentation, benchmark generation, and analysis
- `abcrown.yml` for running `alpha-beta-CROWN`

Typical setup:

```bash
conda env create -f xaiv.yml
conda env create -f abcrown.yml

conda activate xaiv
pip install -e .
```

Notes:

- The supplied environment files are cluster-oriented and assume a Linux/CUDA stack.
- SAM2 checkpoints are not checked into the repository. The first SAM2 run may require downloading weights, or you can prefetch them using the vendor tree under `sam2/`.

## Generating Semantic Benchmarks

Run the pipeline from the repository root.

Example: CIFAR-100 semantic benchmark

```bash
conda activate xaiv
python scripts/pipeline.py --config configs/cifar100/cifar100_all.yaml
```

Example: CIFAR-100 dilation

```bash
python scripts/pipeline.py --config configs/cifar100/cifar100_all_d_1.yaml
python scripts/pipeline.py --config configs/cifar100/cifar100_all_d_2.yaml
python scripts/pipeline.py --config configs/cifar100/cifar100_all_d_3.yaml
python scripts/pipeline.py --config configs/cifar100/cifar100_all_d_4.yaml
```

Example: ImageNet/VGG16 semantic benchmark

```bash
python scripts/pipeline.py --config configs/xaiv/vggnet16_benchmark2022_segmented_all_imgs.yaml
```

Each run writes a benchmark directory containing:

- `onnx/`
- `vnnlib/`
- `instances.csv`
- `input_change_stats.csv`
- `debug_vis/`
- `debug_vis/property_masks.csv` when mask visualization is enabled

## Running alpha-beta-CROWN

The repository vendors `ab-crown/complete_verifier/` and provides starter configs in `configs/abcrown/`.

The intended workflow is:

1. Generate a benchmark folder with `scripts/pipeline.py`.
2. Point `general.root_path` in the relevant `configs/abcrown/*.yaml` file to that benchmark directory.
3. Run `alpha-beta-CROWN` on the generated `instances.csv`.

Typical invocation from the repository root:

```bash
python ab-crown/complete_verifier/abcrown.py --config configs/abcrown/cifar100.yaml
```

The provided configs cover the two main settings:

- `configs/abcrown/cifar100.yaml`
- `configs/abcrown/vggnet16.yaml`

Cluster execution scripts used in the experiments are available under `scripts/slurm/`.

## Practical Notes

- The pipeline skips any image that is not correctly classified before property generation.
- For CIFAR-100, the model is inferred from the filename prefix such as `resnet_large__...` or `resnet_medium__...`.
- Grid-prompted SAM2 is the default segmentation mode, but manual prompts are also supported through JSON prompt files.
