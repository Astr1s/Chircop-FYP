import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH     = r"Training Models/graph/MozillaGraphB.pt"
BUG_INFO_PATH  = r"Mozilla_preprocessed_bug_info.pkl"
CHECKPOINT_IN  = r"Training Models\models\best_gcn.pt"
CHECKPOINT_OUT = r"best_gcn_finetuned.pt"
EPOCHS         = 400
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
# ──────────────────────────────────────────────────────────────────────────────────

# 1) Load graph
data: Data = torch.load(GRAPH_PATH)
x           = data.x.to(DEVICE)
edge_index  = data.edge_index.to(DEVICE)

# 2) Build labels & masks (same as before)
bug_info  = pickle.load(open(BUG_INFO_PATH, "rb"))
assignees = sorted({info["assignee"] for info in bug_info.values() if info["assignee"]})
assignee2idx = {a: i for i, a in enumerate(assignees)}

num_bugs = data.num_bugs
y        = torch.full((num_bugs,), -1, dtype=torch.long)
for bid, idx in data.bug_idx_map.items():
    assignee = bug_info[bid]["assignee"]
    if assignee in assignee2idx:
        y[idx] = assignee2idx[assignee]
valid     = y >= 0

# 80/10/10 split reproducibly
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
val_mask[  val_idx]   = True
test_mask[ test_idx]  = True

y          = y.to(DEVICE)
train_mask = train_mask.to(DEVICE)
val_mask   = val_mask.to(DEVICE)
test_mask  = test_mask.to(DEVICE)

# 3) Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# 4) Instantiate with NEW num_classes, load pretrained conv1 only
old_state = torch.load(CHECKPOINT_IN, map_location=DEVICE)
old_num_classes = old_state['conv2.bias'].shape[0]
new_num_classes = len(assignees)

model = GCN(x.size(1), 128, new_num_classes).to(DEVICE)

# 4a) Warm-start conv1 weights; skip conv2 entirely
state = model.state_dict()
# copy matching tensors
for k, v in old_state.items():
    if k in state and v.size() == state[k].size():
        state[k] = v
model.load_state_dict(state)

# 5) Freeze all but conv2
for name, param in model.named_parameters():
    if 'conv2' not in name:
        param.requires_grad = False

# 6) Setup optimizer (only conv2 params)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
criterion = torch.nn.CrossEntropyLoss()

# 7) Training loop (fine‑tune head)
best_val = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    logits_bug = logits[:num_bugs]
    loss = criterion(logits_bug[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        pred = logits_bug.argmax(dim=1)
        val_acc   = (pred[val_mask] == y[val_mask]).float().mean().item()
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CHECKPOINT_OUT)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"\nFine‑tuning complete. Best val acc: {best_val:.4f}")
print(f"Fine‑tuned model saved to: {CHECKPOINT_OUT}")
