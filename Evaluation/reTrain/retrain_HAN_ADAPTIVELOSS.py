import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HANConv
import pickle

# ── CONFIG ────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH     = "Training Models/graph/MozillaGraphB.pt"
BUG_INFO_PATH  = "Mozilla_preprocessed_bug_info.pkl"
CHECKPOINT_IN  = "Training Models\models\HAN.pth"
CHECKPOINT_OUT = "best_han_finetuned.pt"
EPOCHS         = 400
LR             = 1e-3
WEIGHT_DECAY   = 1e-4


HIDDEN_DIM = 128    # hidden channels in han1
OUT_DIM    = 64     # out channels in han2
HEADS      = 4      # number of heads in han1
# ──────────────────────────────────────────────────────────────────────────────────

# 1) Load original Data graph
data: Data = torch.load(GRAPH_PATH)

# 2) Build labels & masks on bug nodes
bug_info    = pickle.load(open(BUG_INFO_PATH, "rb"))
assignees   = sorted({info["assignee"] for info in bug_info.values() if info["assignee"]})
assignee2idx= {a: i for i, a in enumerate(assignees)}

num_bugs = data.num_bugs
y = torch.full((num_bugs,), -1, dtype=torch.long)
for bid, idx in data.bug_idx_map.items():
    a = bug_info[bid]["assignee"]
    if a in assignee2idx:
        y[idx] = assignee2idx[a]
valid   = y >= 0

# 80/10/10 split (fixed seed)
indices   = torch.arange(num_bugs)[valid]
indices   = indices[torch.randperm(indices.size(0), generator=torch.Generator().manual_seed(42))]
n         = indices.size(0)
train_idx = indices[: int(0.8 * n)]
val_idx   = indices[int(0.8 * n): int(0.9 * n)]
test_idx  = indices[int(0.9 * n):]

train_mask = torch.zeros(num_bugs, dtype=torch.bool)
val_mask   = torch.zeros(num_bugs, dtype=torch.bool)
test_mask  = torch.zeros(num_bugs, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

# Move masks & labels to DEVICE
y          = y.to(DEVICE)
train_mask = train_mask.to(DEVICE)
val_mask   = val_mask.to(DEVICE)
test_mask  = test_mask.to(DEVICE)

torch.save(data.cpu(), "Training Models/graph/MozillaGraphB_with_masks.pt")
# 3) Convert to HeteroData
def to_hetero(data: Data):
    hetero = HeteroData()
    hetero['bug'].x  = data.x[: data.num_bugs]
    hetero['user'].x = data.x[data.num_bugs : ]
    hetero['bug'].y           = y
    hetero['bug'].train_mask  = train_mask
    hetero['bug'].val_mask    = val_mask
    hetero['bug'].test_mask   = test_mask

    edges_bug_user, edges_user_bug = [], []
    for src, dst in data.edge_index.t().tolist():
        if src < data.num_bugs <= dst:
            edges_bug_user.append([src, dst - data.num_bugs])
        elif dst < data.num_bugs <= src:
            edges_user_bug.append([src - data.num_bugs, dst])
    hetero['bug','interacts','user'].edge_index = torch.tensor(edges_bug_user, dtype=torch.long).t().contiguous()
    hetero['user','interacts','bug'].edge_index = torch.tensor(edges_user_bug, dtype=torch.long).t().contiguous()
    return hetero

hetero_data = to_hetero(data).to(DEVICE)
metadata   = hetero_data.metadata()
in_dim     = hetero_data['bug'].x.size(1)
new_num_classes = len(assignees)

# 4) Original HAN class (unchanged)
class HAN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, metadata, heads=2):
        super().__init__()
        self.han1 = HANConv(in_channels=in_dim,
                             out_channels=hidden_dim,
                             metadata=metadata,
                             heads=heads,
                             dropout=0.2,
                             negative_slope=0.2)
        self.han2 = HANConv(in_channels=hidden_dim,
                             out_channels=out_dim,
                             metadata=metadata,
                             heads=1,
                             dropout=0.2,
                             negative_slope=0.2)
        self.classifier = torch.nn.Linear(out_dim, new_num_classes)

    def forward(self, x_dict, edge_index_dict):
        x_hidden = self.han1(x_dict, edge_index_dict)
        x_hidden = {k: F.elu(v) for k, v in x_hidden.items()}
        x_out    = self.han2(x_hidden, edge_index_dict)
        return self.classifier(x_out['bug'])

# 5) Instantiate & warm‑start
model = HAN(in_dim, HIDDEN_DIM, OUT_DIM, metadata, heads=HEADS).to(DEVICE)
old_state = torch.load(CHECKPOINT_IN, map_location=DEVICE)
state = model.state_dict()
for k, v in old_state.items():
    if k in state and v.size() == state[k].size():
        state[k] = v
model.load_state_dict(state)

# Freeze han1/han2, train only classifier
for name, param in model.named_parameters():
    if not name.startswith('classifier'):
        param.requires_grad = False

# 6) Optimizer & loss
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
criterion = torch.nn.CrossEntropyLoss()

# 7) Training loop (full heterogeneous graph)
best_val = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()
    logits = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    loss   = criterion(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        pred     = logits.argmax(dim=1)
        val_acc  = (pred[val_mask] == y[val_mask]).float().mean().item()
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CHECKPOINT_OUT)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"\nFine‑tuning complete. Best val acc: {best_val:.4f}")
print(f"Saved to {CHECKPOINT_OUT}")
