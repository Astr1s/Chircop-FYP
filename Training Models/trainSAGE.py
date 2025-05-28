import glob
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

def GraphSAGE(in_channels, hidden_channels, num_layers, num_classes):
    class _GraphSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            # Input layer
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            # Final conv
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            # Classification head
            self.classifier = Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1: 
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return self.classifier(x)

    return _GraphSAGE()


def prepare_labels_and_masks(data, bug_info, train_ratio=0.8, val_ratio=0.1):
    num_nodes = data.num_nodes
    num_bugs = data.num_bugs

    # Initialize all labels to -1
    y = torch.full((num_nodes,), -1, dtype=torch.long)

    # Assign label for each bug as (global user idx) - num_bugs
    for bug_id, bug_idx in data.bug_idx_map.items():
        assignee = bug_info.get(bug_id, {}).get('assignee')
        if assignee and assignee in data.user_idx_map:
            user_global = data.user_idx_map[assignee]
            y[bug_idx] = user_global - num_bugs

    # Identify labeled bug nodes
    labeled = (y[:num_bugs] >= 0).nonzero(as_tuple=False).view(-1)
    n = labeled.size(0)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio   * n)

    perm = torch.randperm(n)
    train_idx = labeled[perm[:n_train]]
    val_idx   = labeled[perm[n_train:n_train+n_val]]
    test_idx  = labeled[perm[n_train+n_val:]]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.y = y
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    return data

def to_device(data, device):
    data.x           = data.x.to(device)
    data.edge_index  = data.edge_index.to(device)
    data.y           = data.y.to(device)
    data.train_mask  = data.train_mask.to(device)
    data.val_mask    = data.val_mask.to(device)
    data.test_mask   = data.test_mask.to(device)
    return data


def train_sage(model, data,criterion, lr=1e-2, weight_decay=5e-4, epochs=100, patience=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            patience_counter = 0  # Reset counter on improvement
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch:03d} | Best Val Accuracy: {best_val:.4f}")
                break

        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    return model

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha.to(device)  # Tensor of shape [num_classes]
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        at = self.alpha[targets]
        focal_term = at * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term


def test_sage(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


def predict_top_k(model, data, bug_idx, k=5):
    """
    Returns global node indices of the top-k developer predictions for a given bug node index.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        bug_logits = logits[bug_idx]
        topk = bug_logits.topk(k).indices
        return (topk + data.num_bugs).tolist()

def evaluate_top_k(model, data, ks=[1, 3, 5]):
    """
    Computes top-k accuracy for each k in ks on the test set of bug nodes.
    Returns a dict {k: accuracy}.
    """
    model.eval()
    results = {}
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_bugs = data.test_mask.nonzero(as_tuple=False).view(-1)
        for k in ks:
            correct = 0
            for bug_idx in test_bugs:
                true_label = data.y[bug_idx].item()
                topk = logits[bug_idx].topk(k).indices.tolist()
                if true_label in topk:
                    correct += 1
            results[k] = correct / test_bugs.size(0)
    return results

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = torch.load('graph_BERT_FULL.pt')
    from datasetupBERT import load_buglist_data

    bug_info = load_buglist_data(r"F:\Thesis\RAW DATA\buglist")

    train_data = prepare_labels_and_masks(train_data, bug_info)
    train_data = to_device(train_data, device)

    model = GraphSAGE(
        in_channels=train_data.x.size(1),
        hidden_channels=256,
        num_layers=3,
        num_classes=train_data.num_users
    ).to(device)

    train_labels = train_data.y[train_data.train_mask]
    class_counts = torch.bincount(train_labels, minlength=train_data.num_users)

    eps = 1e-6
    class_weights = 1.0 / (class_counts.float() + eps)
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    model = train_sage(model, train_data, epochs=350, patience= 30, criterion =criterion
)
    test_sage(model, train_data)
    torch.save(model.state_dict(), 'graphsage_bugtriage.pth')