# Verifier-guided active-region search

Run:

```bash
python pipeline.py --config example_config.yaml
```

Optional subset:

```bash
python pipeline.py --config example_config.yaml --indices 0 1 2
```

The default greedy-verifier loop is:

1. load image and label,
2. build an initial patch-grid mask,
3. write a fresh VNNLIB for the active pixels,
4. call αβ-CROWN using `verifier.command`,
5. parse status/lower bound/runtime,
6. expand the mask if the current mask is safe, otherwise shrink it,
7. keep the largest mask that remains verifiable.

Alternative search modes:

- `search.method: greedy_verifier`: deterministic expand/shrink search.
- `search.method: q_learning`: patch-toggle reinforcement learning.
- `search.method: random`: random patch-toggle baseline.

Main files:

- `pipeline.py`: entrypoint and orchestration.
- `active_region_search.py`: environment and simple random/Q-learning agents.
- `mask_utils.py`: patch grid and mask construction.
- `vnnlib_utils.py`: bounds and VNNLIB writer.
- `verifier_runner.py`: subprocess call to αβ-CROWN and output parsing.
- `dataset_utils.py`: generic image discovery and label loading.
- `image_utils.py`: preprocessing.
- `config_utils.py`: config helpers.
# MyXAIV
