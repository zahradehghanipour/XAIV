from typing import Optional

from torch import Tensor
from torch.utils.data import Dataset
from torch_sparse.typing import Adj


class LearningHeuristicDataset(Dataset):
    """WIP"""

    def __init__(
        self,
        features: Tensor,
        labels: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

        # NOTE In the future need to store the adj here. For example the OVAL21 benchmark
        # has three different architectures, so the dataset class should return a different
        # edge index.
        # For now, we assume one network so one adjacency matrix / edge_index
        self.edge_index = edge_index

        # NOTE at some point it would be good to convert network weights into an edge
        # weight matrix that can be fed through the GNN layers.
        self.edge_weight = edge_weight

    def __len__(self) -> int:
        return self.labels.size(dim=0)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        features = self.features[index]
        label = self.labels[index]

        return features, label
