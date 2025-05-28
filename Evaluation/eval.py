
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, HANConv
import pandas as pd
import pickle
import random
import numpy as np
import matplotlib
from sklearn.metrics import balanced_accuracy_score
matplotlib.use('Agg')
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


# ── CONFIG ────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = "Training Models/graph/MozillaGraphB.pt"
BUG_INFO_PATH = "Mozilla_preprocessed_bug_info.pkl"

MODELS = [
    {"name": "GraphSAGE‑FT", "type": "sage", "ckpt": r"Evaluation/reTrain/reTrainedModels/retrain2/best_sage_finetuned.pt"},
    {"name": "GAT‑FT",     "type": "gat",  "ckpt": r"Evaluation\reTrain\reTrainedModels\retrain2\best_gat_finetuned.pth"},
    {"name": "GCN‑FT",     "type": "gcn",  "ckpt": r"Evaluation/reTrain/reTrainedModels/retrain2/best_gcn_finetuned.pt"},
    {"name": "HAN‑FT",     "type": "han",  "ckpt": r"Evaluation\reTrain\reTrainedModels\HAN_OPTUNA_FINETUNED.pt"},
] 
KS = [1, 3, 5]
HIDDEN_DIM = 256
NUM_LAYERS = 3
# ── CONFIG ────────────────────────────────────────────────────────────────────────

def load_graph(path):
    data: Data = torch.load(path)
    data.x = data.x.to(DEVICE)
    data.edge_index = data.edge_index.to(DEVICE)
    return data


def build_labels_and_masks(data, bug_info, seed=42):
    assignees = sorted({info["assignee"] for info in bug_info.values()})
    a2i = {a: i for i, a in enumerate(assignees)}
    num_bugs = data.num_bugs

    y = torch.full((num_bugs,), -1, dtype=torch.long)
    for bid, idx in data.bug_idx_map.items():
        y[idx] = a2i.get(bug_info[bid]["assignee"], -1)

    valid = torch.where(y >= 0)[0]
    perm = valid[torch.randperm(valid.size(0), generator=torch.Generator().manual_seed(seed))]
    n = perm.size(0)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def mask(idx):
        m = torch.zeros(num_bugs, dtype=torch.bool)
        m[idx] = True
        return m.to(DEVICE)

    return y.to(DEVICE), mask(train_idx), mask(val_idx), mask(test_idx), assignees

def compute_mrr(y_pred, y_true):
    rankings = torch.argsort(y_pred, dim=1, descending=True)  # (num_samples, num_classes)
    ranks = []
    for i in range(y_true.shape[0]):
        true_class = y_true[i].item()
        rank_idx = (rankings[i] == true_class).nonzero(as_tuple=True)[0].item()
        reciprocal_rank = 1.0 / (rank_idx + 1)  # +1 because rank starts from 1
        ranks.append(reciprocal_rank)

    return sum(ranks) / len(ranks)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super().__init__()
        self.convs = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        )
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def topk_accuracy(logits, labels, mask, ks):
    lm, lv = logits[mask], labels[mask]
    return {k: (lm.topk(k, dim=1)[1].eq(lv.unsqueeze(1)).any(1).float().mean().item()) for k in ks}


class HAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, metadata, heads, num_classes):
        super().__init__()
        self.han1 = HANConv(
            in_channels=in_dim, out_channels=hidden_dim,
            metadata=metadata, heads=heads, dropout=0.2
        )
        self.han2 = HANConv(
            in_channels=hidden_dim, out_channels=out_dim,
            metadata=metadata, heads=1, dropout=0.2
        )
        self.classifier = torch.nn.Linear(out_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        h = self.han1(x_dict, edge_index_dict)
        h = {k: F.elu(v) for k, v in h.items()}
        out = self.han2(h, edge_index_dict)
        return self.classifier(out['bug'])

import torch

def balanced_topk_accuracy(logits, labels, mask, k):
    lm = logits[mask]         
    lv = labels[mask]  
    topk = lm.topk(k, dim=1).indices 
    
    C = logits.size(1)
    recalls = []
    for c in range(C):
        idx_c = (lv == c).nonzero(as_tuple=True)[0]
        if idx_c.numel() == 0:
            continue  # skip classes with no test samples
        hits = (topk[idx_c] == c).any(dim=1).float().sum().item()
        recalls.append(hits / idx_c.numel())
    
    return float(sum(recalls) / len(recalls)) if recalls else 0.0

def prepare_hetero(data, y, train_mask, val_mask, test_mask):
    hetero = HeteroData()
    hetero['bug'].x = data.x[: data.num_bugs]
    hetero['user'].x = data.x[data.num_bugs:]
    hetero['bug'].y = y
    hetero['bug'].train_mask = train_mask
    hetero['bug'].val_mask = val_mask
    hetero['bug'].test_mask = test_mask

    bu, ub = [], []
    for s, t in data.edge_index.t().tolist():
        if s < data.num_bugs <= t:
            bu.append([s, t - data.num_bugs])
        elif t < data.num_bugs <= s:
            ub.append([s - data.num_bugs, t])
    hetero['bug','interacts','user'].edge_index = torch.tensor(bu).t().contiguous()
    hetero['user','interacts','bug'].edge_index = torch.tensor(ub).t().contiguous()
    return hetero.to(DEVICE)


def main():
    data = load_graph(GRAPH_PATH)
    bug_info = pickle.load(open(BUG_INFO_PATH, 'rb'))
    y, train_mask, val_mask, test_mask, assignees = build_labels_and_masks(data, bug_info)
    num_classes = len(assignees)
    in_dim = data.x.size(1)

    hetero = prepare_hetero(data, y, train_mask, val_mask, test_mask)
    metadata = hetero.metadata()
    valid = torch.where(y >= 0)[0].tolist()
    sampled = random.sample(valid, k=min(5, len(valid)))

    models = {}
    for cfg in MODELS:
        mtype, name, ckpt = cfg['type'], cfg['name'], cfg['ckpt']
        if mtype == 'sage':
            m = GraphSAGE(in_dim, HIDDEN_DIM, NUM_LAYERS, num_classes)
            fn = lambda mdl: mdl(data.x, data.edge_index)[: data.num_bugs]
        elif mtype == 'gat':
            m = GAT(in_dim, 128, num_classes, num_layers=2, heads=4)
            fn = lambda mdl: mdl(data.x, data.edge_index)[: data.num_bugs]
        elif mtype == 'gcn':
            m = GCN(in_dim, 128, num_classes)
            fn = lambda mdl: mdl(data.x, data.edge_index)[: data.num_bugs]
        else:
            m = HAN(in_dim, 128, 64, metadata, heads=4, num_classes=num_classes)
            fn = lambda mdl: mdl(hetero.x_dict, hetero.edge_index_dict)

        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.to(DEVICE).eval()
        models[name] = (m, fn)

    mrr_scores = {}
    # Evaluate
    records = []
    for name, (m, fn) in models.items():
        with torch.no_grad():
            logits = fn(m)
            preds = logits.argmax(dim=1)
            val_scores = topk_accuracy(logits, y, val_mask, KS)
            test_scores = topk_accuracy(logits, y, test_mask, KS)
            mrr = compute_mrr(logits[test_mask], y[test_mask])
            mrr_scores[name] = mrr
            y_t = y[test_mask].cpu().numpy()
            p_t = preds[test_mask].cpu().numpy()
            bal_acc = balanced_accuracy_score(y_t, p_t)
            bal_topk = { k: balanced_topk_accuracy(logits, y, test_mask, k) for k in KS }
            print(f"\n{name} sample predictions:")
            print(f"{'BugID':<15}{'True':<20}" + ''.join(f"Top{k:<8}" for k in KS))
            for idx in sampled:
                true_lbl = assignees[y[idx].item()]
                topk_inds = logits[idx].topk(max(KS), dim=0).indices.tolist()
                preds_str = [','.join(assignees[i] for i in topk_inds[:k]) for k in KS]
                # retrieve bug ID from mapping
                bug_id = next(b for b, i in data.bug_idx_map.items() if i == idx)
                print(f"{bug_id:<15}{true_lbl:<20}" + ''.join(f"{p:<8}" for p in preds_str))

            records.append({
                'Model': name,
                **{f'Val@{k}': val_scores[k] for k in KS},
                **{f'Test@{k}': test_scores[k] for k in KS},
                **{f'Balanced@{k}': bal_topk[k] for k in KS},
            })
            y_t  = y[test_mask].cpu().numpy()
            p_t  = preds[test_mask].cpu().numpy()
            bal = balanced_accuracy_score(y_t, p_t)

            print(f"{name} — Test@1: {test_scores[1]:.4f}, Balanced Acc: {bal_acc:.4f}")
            print(f"        Balanced@1={bal_topk[1]:.4f}, Balanced@3={bal_topk[3]:.4f}, Balanced@5={bal_topk[5]:.4f}")
            print(f"\n{name} results:")
            print(f"Balanced Accuracy: {bal_acc:.4f}")
            print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
            for k in KS:
                print(f"Top-{k} Accuracy: {test_scores[k]:.4f}, Balanced Top-{k}: {bal_topk[k]:.4f}")

    df = pd.DataFrame(records).set_index('Model')
    print(df.to_markdown(floatfmt='.4f'))

    plt.figure(figsize=(8, 5))
    model_names = list(mrr_scores.keys())
    mrr_values = [mrr_scores[name] for name in model_names]

    plt.bar(model_names, mrr_values, color='skyblue')
    plt.ylabel("Mean Reciprocal Rank (MRR)")
    plt.title("MRR Scores for Fine-Tuned GNN Models")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig("Evaluation/MRR_scores.png") 
    print("MRR plot saved to Evaluation/MRR_scores.png")

if __name__ == '__main__':
    main()
