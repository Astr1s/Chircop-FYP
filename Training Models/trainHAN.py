import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HANConv
import random
import os

graph_path = r"Training Models\graph\graph_BERT_FULL.pt"
BUG_INFO_PATH = r"Eclipse_preprocessed_bug_info.pkl"
data: Data = torch.load(graph_path)
print(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges.")


bug_info    = pickle.load(open(BUG_INFO_PATH, "rb"))
assignees   = sorted({info["assignee"] for info in bug_info.values() if info["assignee"]})
assignee2idx= {a: i for i, a in enumerate(assignees)}

num_bugs = data.num_bugs
y = torch.full((num_bugs,), -1, dtype=torch.long)
for bid, idx in data.bug_idx_map.items():
    if bid in bug_info and "assignee" in bug_info[bid]:
        assignee = bug_info[bid]["assignee"]
        if assignee in assignee2idx:
            y[idx] = assignee2idx[assignee]
valid   = y >= 0

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
val_mask[ val_idx] = True
test_mask[test_idx] = True

def to_hetero(data: Data):
    hetero = HeteroData()
    hetero['bug'].x  = data.x[: data.num_bugs]
    hetero['user'].x = data.x[data.num_bugs :]
    hetero['bug'].y = y
    hetero['bug'].train_mask = train_mask
    hetero['bug'].val_mask   = val_mask
    hetero['bug'].test_mask  = test_mask
    edges_bug_user = []
    edges_user_bug = []
    for src, dst in data.edge_index.t().tolist():
        if src < data.num_bugs <= dst:
            edges_bug_user.append([src, dst - data.num_bugs])
        elif dst < data.num_bugs <= src:
            edges_user_bug.append([src - data.num_bugs, dst])
    hetero['bug','interacts','user'].edge_index  = torch.tensor(edges_bug_user, dtype=torch.long).t().contiguous()
    hetero['user','interacts','bug'].edge_index  = torch.tensor(edges_user_bug, dtype=torch.long).t().contiguous()
    return hetero

hetero_data = to_hetero(data)
print(hetero_data)

class HAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, metadata, heads=2):
        super().__init__()
        self.han1 = HANConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=heads,
            dropout=0.2,
            negative_slope=0.2,
        )
        self.han2 = HANConv(
            in_channels=hidden_dim,
            out_channels=out_dim,
            metadata=metadata,
            heads=1,  
            dropout=0.2,
            negative_slope=0.2,
        )
        self.classifier = torch.nn.Linear(out_dim, len(assignees))

    def forward(self, x_dict, edge_index_dict):
        x_hidden = self.han1(x_dict, edge_index_dict)    
        x_hidden = {k: F.elu(v) for k, v in x_hidden.items()}

        x_out = self.han2(x_hidden, edge_index_dict)    
        
        return self.classifier(x_out['bug'])

node_types, edge_types = hetero_data.metadata()
metadata = (node_types, edge_types)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metadata = hetero_data.metadata() 
model = HAN(
    in_dim     = hetero_data['bug'].x.size(1),
    hidden_dim = 128,
    out_dim    = 64,
    metadata   = metadata,
    heads      = 4,
).to(device)

opt       = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
x_dict          = {k: v.to(device) for k, v in hetero_data.x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in hetero_data.edge_index_dict.items()}
y_true          = hetero_data['bug'].y.to(device)
train_mask      = hetero_data['bug'].train_mask.to(device)
val_mask        = hetero_data['bug'].val_mask.to(device)
test_mask       = hetero_data['bug'].test_mask.to(device)

best_val_acc = 0.0
for epoch in range(1, 301):
    model.train()
    logits = model(x_dict, edge_index_dict)
    loss   = criterion(logits[train_mask], y_true[train_mask])
    loss.backward()
    opt.step()
    opt.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        pred     = logits.argmax(dim=1)
        val_acc  = (pred[val_mask] == y_true[val_mask]).float().mean().item()
        test_acc = (pred[test_mask] == y_true[test_mask]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_han.pt")
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"Training complete. Best val acc: {best_val_acc:.4f}")
torch.save(model.state_dict(), 'HYHAN.pth')