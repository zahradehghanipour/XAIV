import torch
import pandas as pd
import os
import pickle

from collections import OrderedDict
from typing import Any

import arguments
from .fsb import FsbBranching
from .RL import _netinfo


def _copy_tensor(t):
    assert isinstance(t, torch.Tensor)
    # Copy tensor and move to CPU memory
    return t.detach().clone().cpu()


# TODO not compatible with input splitting script, as ab-crown is instantiated again when checking a split property.
#      need to somehow provide some unique name to the buffer.
class GenerateDataFsbBranching(FsbBranching):
    def __init__(self, net, args):
        super().__init__(net)
        self.layers, self.nodes, self.relus = _netinfo(net)
        
        self.onnx_path = args["model"]["onnx_path"]
        self.vnnlib_path = args["specification"]["vnnlib_path"]
        self.benchmark = os.path.basename(args["general"]["root_path"])
        
        # -- make output directory
        os.makedirs(args["general"]['dataset_output_dir'], exist_ok=True)

        # -- create buffer path and buffer
        self.index = arguments.Globals['example_idx']
        self.buffer_path = os.path.join(args["general"]['dataset_output_dir'], f"{self.benchmark}_buffer_{self.index:03d}.pkl")
        self.buffer = []
        
    def extractstate(self, domains) -> dict[str, Any]:
        """
        Extracts mask, lower bounds, upper bounds, and global lower bound from ab-crown state.
        """
        msk = OrderedDict()
        lbs = OrderedDict()
        ubs = OrderedDict()
        glb = None

        for _, layer in enumerate(self.layers):
            msk[layer] = _copy_tensor(domains["mask"][layer])
            lbs[layer] = _copy_tensor(domains["lower_bounds"][layer])
            ubs[layer] = _copy_tensor(domains["upper_bounds"][layer])

        if domains["global_lb"].shape != domains["thresholds"].shape:
            raise ValueError(
                f"Shape mismatch between global lower bounds and thresholds ({domains['global_lb'].shape} vs {domains['thresholds'].shape})"
            )

        # Let's also squeeze the tensor to go from shape [1, K] to [K]
        glb = _copy_tensor(domains["global_lb"] - domains["thresholds"]).squeeze(0)

        # NOTE at the moment, this only works for batch_size=1
        depth = domains["depths"][0] 

        return {"msk": msk, "lbs": lbs, "ubs": ubs, "glb": glb, "depth": depth}

    def add_to_buffer(self, state, action) -> None:
        self.buffer.append({"episode_id": self.index,
                            "model": self.onnx_path, 
                            "property": self.vnnlib_path, 
                            "state": state, 
                            "action": action})

    def save_buffer(self) -> None:
        with open(self.buffer_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def get_branching_decisions(self, domains, split_depth=1, **kwargs): # pyright: ignore[reportIncompatibleMethodOverride]
        decisions = super().get_branching_decisions(domains, split_depth, **kwargs)

        state = self.extractstate(domains)
        for action in decisions[0]:
            self.add_to_buffer(state, action)

        return decisions