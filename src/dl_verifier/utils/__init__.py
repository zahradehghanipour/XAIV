import itertools
from collections import OrderedDict

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

# TODO refactor into seperate _utils.py files


def compute_adj_matrix(
    sizes: list[int],
    bidirectional: bool = False,
    add_self_loops: bool = True,
    return_sparse_tensor: bool = True,
) -> Adj:
    # compute the cumulative number of nodes per layer, we use this to
    # offset the indices to create the adjacency matrix or edge index
    *offsets, total_nodes = itertools.accumulate([0] + sizes)

    edges = []
    for i, (source_offset, target_offset) in enumerate(itertools.pairwise(offsets)):
        source_index = torch.arange(sizes[i]) + source_offset
        target_index = torch.arange(sizes[i + 1]) + target_offset

        source_grid, target_grid = torch.meshgrid(
            source_index, target_index, indexing="ij"
        )

        forward = torch.stack([source_grid.flatten(), target_grid.flatten()], dim=1)
        edges.append(forward)

        if bidirectional:
            backward = torch.stack(
                [target_grid.flatten(), source_grid.flatten()], dim=1
            )
            edges.append(backward)

    if add_self_loops:
        self_loops = torch.stack(
            [torch.arange(total_nodes), torch.arange(total_nodes)], dim=1
        )
        edges.append(self_loops)

    edge_index = torch.cat(edges, dim=0).t().contiguous()

    if return_sparse_tensor:
        return SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(total_nodes, total_nodes),
        )

    return edge_index


def state_to_features(
    state: dict[str, OrderedDict[str, Tensor]], feature_keys: list[str]
) -> Tensor:
    selected_features = [v.values() for k, v in state.items() if k in feature_keys]
    features = []

    for layer_values in selected_features:
        layer_features = [torch.flatten(v) for v in layer_values]
        features.append(torch.cat(layer_features, dim=0))

    x = torch.stack(features, dim=0).t()
    return x


def flatten_features(
    state: dict[str, OrderedDict[str, Tensor]], feature_keys: list[str]
) -> dict[str, Tensor]:
    selected_features = {k: v.values() for k, v in state.items() if k in feature_keys}
    features = {}

    for feature_key, layer_values in selected_features.items():
        layer_features = [torch.flatten(v) for v in layer_values]
        features[feature_key] = torch.cat(layer_features, dim=0)

    return features


def action_index(action: list[int], sizes: list[int]) -> int:
    assert len(action) == 2, (
        f"Not a valid action. Got {action}"
    )  # FIXME could also be fixed by changing type to tuple[int, int]

    layer_index, node_index = action

    assert layer_index >= 0 and layer_index < len(sizes), (
        f"Layer index is out of bounds. Got {layer_index}"
    )
    assert node_index >= 0 and node_index < sizes[layer_index], (
        f"Node index is out of bounds. Got {node_index}"
    )

    index = sum(sizes[:layer_index]) + node_index
    return index


def index_action(index: int, sizes: list[int]) -> list[int]:
    assert index >= 0, f"Index must be postive. Got {index}."

    pairwise_accumulated_sizes = itertools.pairwise(itertools.accumulate([0] + sizes))
    for i, (curr_size, next_size) in enumerate(pairwise_accumulated_sizes):
        if index >= curr_size and index < next_size:
            layer_index = i
            node_index = index % curr_size if curr_size > 0 else index
            return [layer_index, node_index]

    raise ValueError(f"Index is out of bounds. Got {index}.")
