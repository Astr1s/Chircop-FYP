import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pickle
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH     = "Training Models/graph/MozillaGraphB.pt"
BUG_INFO_PATH  = "Mozilla_preprocessed_bug_info.pkl"
CHECKPOINT_IN  = "Training Models/models/graphsage_bugtriage.pth"
CHECKPOINT_OUT = "best_sage_finetuned.pt"
EPOCHS         = 400
LR             = 1e-3      # set learning rate to 0.001
WEIGHT_DECAY   = 1e-4
# Learning rate decay removed
# STEP_SIZE      = 50        # LR decay every 50 epochs (removed)
# GAMMA          = 0.5       # LR multiplied by 0.5 each step (removed)
HIDDEN_DIM     = 256        # must match pretraining
NUM_LAYERS     = 3
# ──────────────────────────────────────────────────────────────────────────────────

# 1) Load graph
data: Data = torch.load(GRAPH_PATH)
x           = data.x.to(DEVICE)
edge_index  = data.edge_index.to(DEVICE)

# 2) Build labels & masks
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

indices   = torch.arange(num_bugs)[valid]
indices   = indices[torch.randperm(indices.size(0), 
                                   generator=torch.Generator().manual_seed(42))]
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

y          = y.to(DEVICE)
train_mask = train_mask.to(DEVICE)
val_mask   = val_mask.to(DEVICE)
test_mask  = test_mask.to(DEVICE)
new_num_classes = len(assignees)
in_dim = x.size(1)

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
            self.classifier = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return self.classifier(x)
    return _GraphSAGE()

# 4) Instantiate and warm‑start
model = GraphSAGE(in_dim, HIDDEN_DIM, NUM_LAYERS, new_num_classes).to(DEVICE)
old_state = torch.load(CHECKPOINT_IN, map_location=DEVICE)

state = model.state_dict()
for k, v in old_state.items():
    if k in state and v.size() == state[k].size():
        state[k] = v
model.load_state_dict(state)

# 5) Freeze only the first convolution (convs[0])
for name, param in model.named_parameters():
    if not name.startswith('classifier'):
        param.requires_grad = False

# 6) Optimizer & loss (no scheduler)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
criterion = torch.nn.CrossEntropyLoss()

# 7) Training loop
best_val = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    optimizer.zero_grad()

    logits = model(x, edge_index)[:num_bugs]
    loss   = criterion(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred     = logits.argmax(dim=1)
        val_acc  = (pred[val_mask] == y[val_mask]).float().mean().item()
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), CHECKPOINT_OUT)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"\nFine‑tuning complete. Best val acc: {best_val:.4f}")
print(f"Model saved to {CHECKPOINT_OUT}")
