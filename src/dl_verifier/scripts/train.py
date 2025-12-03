import argparse
import os
import pickle
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torch_geometric.typing import Adj

from dl_verifier.dataset import LearningHeuristicDataset
from dl_verifier.models import SimpleGNN as SimpleLearningHeuristic
from dl_verifier.utils import (
    action_index,
    compute_adj_matrix,
    state_to_features,
)


@torch.no_grad()
def evaluate(data_loader: DataLoader, model: Module, adj: Adj) -> tuple[float, float]:
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in data_loader:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        logits = model(x, adj)
        loss = F.cross_entropy(logits, y)

        running_loss += loss.item()

        y_hat = torch.argmax(logits, 1)
        correct += (y_hat == y).sum().item()
        total += y.size(0)

    return running_loss / len(data_loader), correct / total


def train(
    data_loader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    adj: Adj,
    desc: Optional[str],
) -> tuple[float, float]:
    running_loss = 0
    correct = 0
    total = 0

    for x, y in (pbar := tqdm.tqdm(data_loader, total=len(data_loader))):
        pbar.set_description(desc=desc)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # -- reset gradients
        optimizer.zero_grad()

        # -- process one batch
        logits = model(x, adj)

        # -- compute error and optimise
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_hat = torch.argmax(logits, 1)
        correct += (y_hat == y).sum().item()
        total += y.size(0)

    return running_loss / len(data_loader), correct / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--epochs", type=int, default=100, dest="n_epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("Loading dataset...", end=" ")
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    print("Done.")

    # -- adjacency matrix setup -------------------------------

    # -- hacky way to get layer names and layer sizes
    layers, sizes = [], []
    for layer, mask in data[0]["state"]["msk"].items():
        layers.append(layer)
        sizes.append(mask.size(1))

    # -- this should move to the dataset class
    adj = compute_adj_matrix(sizes, add_self_loops=True, return_sparse_tensor=True)
    if torch.cuda.is_available():
        adj = adj.cuda()

    # -- data setup ------------------------------------------

    feature_keys = ["lbs", "ubs", "msk"]
    features = torch.stack(
        [state_to_features(row["state"], feature_keys) for row in data]
    )
    assert (len(data), sum(sizes), 3) == features.size()

    labels = torch.tensor([action_index(row["action"], sizes) for row in data])
    labels = labels.unsqueeze(
        -1
    )  # is necessary because the output of the SimpleGNN is (B, N, 1)
    assert (len(data), 1) == labels.size()

    dataset = LearningHeuristicDataset(features, labels, adj)

    # -- reproducible dataset split
    generator = torch.Generator().manual_seed(42)
    train_data, validation_data, test_data = random_split(
        dataset, lengths=[0.7, 0.2, 0.1], generator=generator
    )

    # -- create loaders for each dataset for training and evaluation
    train_loader = DataLoader(  # noqa
        dataset=train_data, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        dataset=validation_data, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(  # noqa
        dataset=test_data, batch_size=1, shuffle=True
    )

    # -- model and optimizer setup ----------------------------

    model = SimpleLearningHeuristic(3, 64, 1)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # -- init a list to keep track of training stats
    history = []

    for epoch in range(1, args.n_epochs + 1):
        # -- training -----------------------------------------

        model.train()
        train_loss, train_accuracy = train(
            train_loader, model, optimizer, adj, desc=f"Epoch {epoch}/{args.n_epochs}"
        )

        # -- validation -----------------------------------------

        model.eval()
        validation_loss, validation_accuracy = evaluate(validation_loader, model, adj)

        stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        print(stats)
        history.append(stats)

    # -- evaluation -----------------------------------------

    model.eval()
    test_loss, test_accuracy = evaluate(test_loader, model, adj)

    print(f"test loss: {test_loss} test accuracy: {test_accuracy}")

    # -- save results ---------------------------------------

    output_path = os.path.join(os.getcwd(), "results")
    os.makedirs(output_path, exist_ok=True)

    pd.DataFrame(history).to_csv(
        os.path.join(output_path, f"history-{args.experiment_name}-{args.n_epochs}.csv")
    )

    pd.DataFrame(
        [{"epoch": args.n_epochs, "loss": test_loss, "accuracy": test_accuracy}]
    ).to_csv(
        os.path.join(output_path, f"test-{args.experiment_name}-{args.n_epochs}.csv")
    )

    model.save(
        os.path.join(
            output_path, f"heuristic-{args.experiment_name}-{args.n_epochs}.pth"
        ),
        optimizer,
        history[-1],
    )

    return 0


if __name__ == "__main__":
    main()
