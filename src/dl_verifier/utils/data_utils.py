from collections import defaultdict
from math import prod
from typing import Any

import torch
from auto_LiRPA.operators.relu import BoundRelu
from onnx2pytorch import ConvertModel


def get_biases(
    model: ConvertModel, layers: list[str], sizes: list[int]
) -> torch.Tensor:
    # get all children that *have* a bias attribute (may be None)
    children_with_bias_attr = [
        child
        for name, child in model.named_children()
        if hasattr(child, "bias")
    ]

    biases_per_layer = {}

    # we rely on the same ordering assumption as before
    for layer, size, child in zip(layers, sizes, children_with_bias_attr):
        bias = child.bias  # may be a tensor, may be None

        if bias is None:
            print(f"[DEBUG] Layer {layer} ({child.__class__.__name__}) has no bias; using zeros.")
            # Layer has no learned bias (e.g. bias=False).
            # We treat it as zero bias.
            if hasattr(child, "weight") and child.weight is not None:
                channel_size = child.weight.size(0)  # out_channels / out_features
            else:
                # fallback: assume one bias per "size"
                channel_size = size

            bias = torch.zeros(channel_size)
        else:
            channel_size = bias.size(0)

        if channel_size != size:
            # convolutional case: one bias per channel, but 'size' is flattened
            # e.g. channel_size = 8, size = 2048, need to repeat each bias
            bias = copy_tensor(bias)[:, None].expand(-1, (size // channel_size))
            biases_per_layer[layer] = torch.flatten(bias)
        else:
            biases_per_layer[layer] = copy_tensor(bias)

    return biases_per_layer


def copy_tensor(t):
    assert isinstance(t, torch.Tensor)
    return t.detach().clone().cpu()


def extract_state(
    net, domains: dict[str, Any], feature_keys: list[str]
) -> dict[str, list[list[float] | float]]:
    """Extracts αβ-CROWN state"""

    state = defaultdict(list)

    for key in feature_keys:
        # first check for global features, such as global_lb and depth.
        if key == "global_lb":
            threshold = domains["thresholds"]
            global_lb = domains[key]
            assert global_lb.size() == threshold.size()
            state[key] = torch.flatten(global_lb - threshold).tolist()
        elif key == "thresholds":
            state[key] = torch.flatten(domains[key]).tolist()
        elif key == "depths":
            state[key] = domains[key]  # already a list
        elif key == "alphas":
            bound_activations: dict[str, BoundRelu] = {
                bound_activation.name: bound_activation
                for bound_activation in net.net.splittable_activations
            }

            for layer, values in domains[key].items():
                if layer == net.final_name:
                    continue  # skip last layer which is not a splittable layer.

                # this is based off `auto_LiRPA/operatores/relu.py:621:637`
                # the values are dictionary that should contain only 1 set of
                # alphas with the shape is (2, 1, ...)
                # the lower bound are first item on the first dimension
                sparse_alpha = values[net.final_name]
                lb_sparse_alpha = sparse_alpha[0]

                bound_relu = bound_activations[layer]

                full_shape = lb_sparse_alpha.shape[:-1] + bound_relu.shape
                full_alpha = bound_relu.reconstruct_full_alpha(
                    lb_sparse_alpha, full_shape, bound_relu.alpha_indices
                )
                state[key].append(torch.flatten(full_alpha).tolist())
        elif key == "beta":
            if domains[key][0] is not None:
                nodes_with_betas = {node.name: node for node in net.net.nodes_with_beta}
                for layer, values in domains[key][0].items():
                    # NOTE collection beta values doesn't work yet..

                    # Half the information regarding the betas is missing, namely
                    # the indices I guess we can still infer the beta position from
                    # the split history?

                    # another issue is we go up the tree, but then the betas from
                    # domains and the indices from the bounded module go out of sync

                    sparse_betas = nodes_with_betas[layer].sparse_betas[0]
                    size = prod(nodes_with_betas[layer].output_shape)

                    # indices has shape (2, ...), where 2 is the max_number of
                    # splits per layer. I guess we can assume we will only every
                    # have two, the tricky bit is how we match one index with our
                    # one beta value we get from domains.
                    # I think this 2 is for the postive and negative split, with
                    # index 0 being the negative split (due to the sign).

                    _values = sparse_betas.val.detach().clone().cpu()
                    indices = sparse_betas.loc.detach().clone().cpu()
                    state[key][layer] = torch.sparse_coo_tensor(
                        indices, _values, size=(size, size)
                    )
                else:
                    state[key] = []
        else:
            # this clause handles lower_bounds, upper_bounds, and mask
            for layer, values in domains[key].items():
                if layer == net.final_name:
                    continue  # skip last layer which is not a splittable layer.
                state[key].append(torch.flatten(copy_tensor(values)).tolist())

    return state
