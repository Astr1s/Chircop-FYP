import os
import pickle
import torch
import torch.nn.functional as F
import optuna
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HANConv

# Load graph and bug info
graph_path = r"Training Models\graph\graph_BERT_FULL.pt"
data: Data = torch.load(graph_path)
bug_info = pickle.load(open("Eclipse_preprocessed_bug_info.pkl", "rb"))

assignees = sorted({info["assignee"] for info in bug_info.values() if info["assignee"]})
assignee2idx = {a: i for i, a in enumerate(assignees)}

num_bugs = data.num_bugs
y = torch.full((num_bugs,), -1, dtype=torch.long)
for bid, idx in data.bug_idx_map.items():
    if bid in bug_info and "assignee" in bug_info[bid]:
        assignee = bug_info[bid]["assignee"]
        if assignee in assignee2idx:
            y[idx] = assignee2idx[assignee]

valid = y >= 0
indices = torch.arange(num_bugs)[valid]
indices = indices[torch.randperm(indices.size(0))]
n = indices.size(0)
train_idx = indices[: int(0.8 * n)]
val_idx   = indices[int(0.8 * n): int(0.9 * n)]
test_idx  = indices[int(0.9 * n):]

train_mask = torch.zeros(num_bugs, dtype=torch.bool)
val_mask   = train_mask.clone()
test_mask  = train_mask.clone()
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

def to_hetero(data: Data):
    hetero = HeteroData()
    hetero['bug'].x  = data.x[: data.num_bugs]
    hetero['user'].x = data.x[data.num_bugs :]
    hetero['bug'].y = y
    hetero['bug'].train_mask = train_mask
    hetero['bug'].val_mask   = val_mask
    hetero['bug'].test_mask  = test_mask

    edges_bug_user, edges_user_bug = [], []
    for src, dst in data.edge_index.t().tolist():
        if src < data.num_bugs <= dst:
            edges_bug_user.append([src, dst - data.num_bugs])
        elif dst < data.num_bugs <= src:
            edges_user_bug.append([src - data.num_bugs, dst])
    hetero['bug','interacts','user'].edge_index  = torch.tensor(edges_bug_user).t().contiguous()
    hetero['user','interacts','bug'].edge_index  = torch.tensor(edges_user_bug).t().contiguous()
    return hetero

hetero_data = to_hetero(data)
metadata = hetero_data.metadata()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_dict = {k: v.to(device) for k, v in hetero_data.x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in hetero_data.edge_index_dict.items()}
y_true = hetero_data['bug'].y.to(device)
train_mask = hetero_data['bug'].train_mask.to(device)
val_mask = hetero_data['bug'].val_mask.to(device)

class HAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, metadata, heads=2):
        super().__init__()
        self.han1 = HANConv(in_dim, hidden_dim, metadata, heads=heads, dropout=0.2)
        self.han2 = HANConv(hidden_dim, out_dim, metadata, heads=1, dropout=0.2)
        self.classifier = torch.nn.Linear(out_dim, len(assignees))

    def forward(self, x_dict, edge_index_dict):
        x_hidden = self.han1(x_dict, edge_index_dict)
        x_hidden = {k: F.elu(v) for k, v in x_hidden.items()}
        x_out = self.han2(x_hidden, edge_index_dict)
        return self.classifier(x_out['bug'])

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 150, 350)

    model = HAN(
        in_dim=hetero_data['bug'].x.size(1),
        hidden_dim=128,
        out_dim=64,
        metadata=metadata,
        heads=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        logits = model(x_dict, edge_index_dict)
        loss = criterion(logits[train_mask], y_true[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_dict, edge_index_dict)
        pred = logits.argmax(dim=1)
        val_acc = (pred[val_mask] == y_true[val_mask]).float().mean().item()
        trial.set_user_attr("val_acc", val_acc)
        trial.set_user_attr("model_state_dict", model.state_dict())

    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Save best model and params
best_trial = study.best_trial
print("Best Validation Accuracy: {:.4f}".format(best_trial.value))
print("Best Hyperparameters:", best_trial.params)

torch.save(best_trial.user_attrs["model_state_dict"], "HAN_best_optuna.pt")
with open("HAN_best_optuna_params.pkl", "wb") as f:
    pickle.dump(best_trial.params, f)
