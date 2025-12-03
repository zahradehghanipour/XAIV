import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.typing import Adj

from dl_verifier.utils.debug_utils import warn


class NeuralHeuristic(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def load_for_inference(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()

    def load(self, path: str, optimizer: Optimizer) -> Tuple[int, float]:
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # !! side effect
        return checkpoint["epoch"], checkpoint["loss"]

    def save(
        self,
        path: str,
        optimizer: Optional[Optimizer] = None,
        last_epoch_stats: Dict[str, int | float] = {},
    ):
        optimizer_state_dict = (
            {"optimizer_state_dict": optimizer.state_dict()}
            if optimizer is not None
            else {}
        )
        state = {
            "model_state_dict": self.state_dict(),
            **optimizer_state_dict,
            **last_epoch_stats,
        }
        torch.save(state, path)


class SimpleGNN(NeuralHeuristic):
    def __init__(self, d_inputs: int, d_features: int = 64, d_outputs: int = 1) -> None:
        super().__init__()
        self.conv1 = GCNConv(d_inputs, d_features, add_self_loops=False)
        self.conv2 = GCNConv(d_features, d_features, add_self_loops=False)
        self.conv3 = GCNConv(d_features, d_outputs, add_self_loops=False)

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        x = self.conv1(x, edge_index, edge_weight)  # (B, N, input) -> (B, N, 64)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)  # (B, N, 64) -> (B, N, 64)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)  # (B, N, 64) -> (B, N, 1)
        return x


class SimpleGNNLinear(NeuralHeuristic):
    def __init__(self, d_inputs: int, d_features: int = 64, d_outputs: int = 1) -> None:
        super().__init__()
        self.conv1 = GCNConv(d_inputs, d_features, add_self_loops=False)
        self.conv2 = GCNConv(d_features, d_features, add_self_loops=False)
        self.fc1 = nn.Linear(d_features, d_features)
        self.fc2 = nn.Linear(d_features, d_outputs)

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        x = self.conv1(x, edge_index, edge_weight)  # (B, N, input) -> (B, N, 64)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)  # (B, N, 64) -> (B, N, 64)

        x = self.fc1(x)  # (B, N, 64) -> (B, N, 64)
        x = F.relu(x)
        x = self.fc2(x)  # (B, N, 64) -> (B, N, 1)
        return x


class SimpleMLPMultiple(NeuralHeuristic):
    def __init__(self, d_inputs: int, n_nodes: int, d_features: int = 16) -> None:
        super().__init__()
        self.conv1 = GCNConv(d_inputs, d_features, add_self_loops=False)
        self.conv2 = GCNConv(d_features, d_features, add_self_loops=False)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_nodes * d_features, (n_nodes * d_features) // 4)
        self.fc2 = nn.Linear((n_nodes * d_features) // 4, n_nodes)

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        x = self.conv1(x, edge_index, edge_weight)  # (B, N, input) -> (B, N, 16)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)  # (B, N, 16) -> (B, N, 16)

        x = self.flatten(x)  # (B, N, 16) -> (B, N * 16)

        x = self.fc1(x)  # (B, N * 16) -> (B, (N * 16) / 4)
        x = F.relu(x)
        x = self.fc2(x)  # (B, (N * 16) / 4) -> (B, N)
        return x


class GATGraphEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, heads=4):
        super(GATGraphEncoder, self).__init__()
        # NOTE self loops are add manually in adjacency matrix.
        self.head = GATConv(
            d_input, d_hidden, heads=heads, concat=True, add_self_loops=False
        )
        self.tail = GATConv(
            d_hidden * heads, d_output, heads=1, concat=False, add_self_loops=False
        )

    def forward(self, x, a):
        x = F.elu(self.head(x, a))
        x = F.elu(self.tail(x, a))
        return x


class GCNGraphEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_output):
        super(GCNGraphEncoder, self).__init__()
        # NOTE self loops are add manually in adjacency matrix.
        self.head = GCNConv(d_input, d_hidden, add_self_loops=False)
        self.tail = GCNConv(d_hidden, d_output, add_self_loops=False)

    def forward(self, x, a):
        x = F.relu(self.head(x, a))
        x = F.relu(self.tail(x, a))
        return x


class PolicyNet(nn.Module):
    # MAYBE inherit NeuralHeuristic class above?

    def __init__(
        self,
        d_inputs,
        d_graph_features=64,
        d_transformer_features=128,
        n_heads=4,
        n_transformer_layers=2,
        GAT=False,
    ):
        super(PolicyNet, self).__init__()

        self.config = {
            "d_inputs": d_inputs,
            "d_graph_features": d_graph_features,
            "d_transformer_features": d_transformer_features,
            "n_heads": n_heads,
            "n_transformer_layers": n_transformer_layers,
            "GAT": GAT,
        }
        self.graph_encoder = (
            GATGraphEncoder(d_inputs, d_graph_features, d_graph_features)
            if GAT
            else GCNGraphEncoder(d_inputs, d_graph_features, d_graph_features)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_graph_features,
                nhead=n_heads,
                dim_feedforward=d_transformer_features,
            ),
            num_layers=n_transformer_layers,
        )
        self.glb_proj = nn.Linear(1, d_graph_features)
        self.output = nn.Linear(d_graph_features, 1)

    def forward(self, x, adj):
        # NOTE one idea is to also concatenate the gnn inputs to the node embeddings
        # providing aditional information for the transformer

        # NOTE perhaps add global_lb as an additional input feature. There can be
        # multiple global_lb values so in that case we need to project these and prepend
        # them all to the gnn outputs. And prune them after the transformer.

        # print(f"[ZRL][graph_encoder] in:  x={tuple(x.shape)}, adj={tuple(adj.shape) if hasattr(adj,'shape') else type(adj)}")
        # x = self.graph_encoder(x, adj)
        # print(f"[ZRL][graph_encoder] out: x={tuple(x.shape)}")

        # print(f"[ZRL][transformer]   in:  x={tuple(x.shape)}")
        # x = self.transformer(x)
        # print(f"[ZRL][transformer]   out: x={tuple(x.shape)}")

        # print(f"[ZRL][output]        in:  x={tuple(x.shape)}")
        # x = self.output(x)
        # print(f"[ZRL][output]        out: x={tuple(x.shape)}")
        # return x

        x = self.graph_encoder(x, adj)  # (1, N, d_graph_features)
        x = self.transformer(x)  # (1, N, d_graph_features)
        x = self.output(x)  # (1, N, 1)
        return x

    def clone(self, eval=True):
        net = PolicyNet(**self.config)
        net.load_state_dict(self.state_dict())
        if eval:
            net.eval()
        return net

    def mirror(self, other):
        assert isinstance(other, PolicyNet)
        self.load_state_dict(other.state_dict())
        # NOTE we could here check if cuda is available instead
        self.to(next(other.parameters()).device)

    def store(self, opt=None, step=None, file_name="checkpoint.pt", prefix=None):
        directory, filename = os.path.split(file_name)
        name, ext = os.path.splitext(filename)

        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(name)
        # NOTE Don't store step to make training loop easier across instances
        # parts.append(f"{step:04d}")

        filename = "-".join(parts) + ext
        file_path = os.path.join(directory, filename)

        checkpoint = {"theModel": self.state_dict(), "config": self.config}

        if opt:
            checkpoint["theOptimizer"] = opt.state_dict()

        if step:
            checkpoint["step"] = step

        torch.save(checkpoint, file_path)

    def restore(self, path, opt=None):
        if not os.path.exists(path):
            warn(f"File {path} not found. Starting with a new net.")
            return 0

        checkpoint = torch.load(path)
        # Restore all learnable parameters
        self.load_state_dict(checkpoint["theModel"])

        self.config = checkpoint["config"]
        # Restore optimizer state
        if opt:
            opt.load_state_dict(checkpoint["theOptimizer"])
        # Return training step
        return checkpoint.get("step", 0)
