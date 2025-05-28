import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv
from torch.nn import Linear
import pickle

# ── CONFIG ────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH     = "Training Models/graph/MozillaGraphB.pt"
BUG_INFO_PATH  = "Mozilla_preprocessed_bug_info.pkl"
CHECKPOINT_IN  = "Training Models/models/gat_bugtriage.pth"
CHECKPOINT_OUT = "best_gat_finetuned.pth"
EPOCHS         = 50          # fewer epochs per batch‑training
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 1024        # number of target nodes per batch
NUM_NEIGHBORS  = [10, 10]    # neighbors sampled at each of 2 layers
# ──────────────────────────────────────────────────────────────────────────────────

# 1) Load full graph
data: Data = torch.load(GRAPH_PATH)
data = data.to('cpu')  # loader will move subgraphs to GPU
num_bugs = data.num_bugs

# 2) Build labels & masks (only on bug nodes)
bug_info    = pickle.load(open(BUG_INFO_PATH, "rb"))
assignees   = sorted({info["assignee"] for info in bug_info.values() if info["assignee"]})
assignee2idx= {a: i for i, a in enumerate(assignees)}

y = torch.full((num_bugs,), -1, dtype=torch.long)
for bid, idx in data.bug_idx_map.items():
    a = bug_info[bid]["assignee"]
    if a in assignee2idx:
        y[idx] = assignee2idx[a]
valid   = y >= 0

indices = torch.arange(num_bugs)[valid]
# fixed split
indices = indices[torch.randperm(indices.size(0), generator=torch.Generator().manual_seed(42))]
n       = indices.size(0)
train_idx = indices[: int(0.8 * n)]
val_idx   = indices[int(0.8 * n): int(0.9 * n)]
test_idx  = indices[int(0.9 * n):]

data.y = torch.cat([y, torch.full((data.num_nodes - num_bugs,), -1, dtype=torch.long)])
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask   = data.train_mask.clone()
data.test_mask  = data.train_mask.clone()
data.train_mask[train_idx] = True
data.val_mask[val_idx]     = True
data.test_mask[test_idx]   = True

# 3) Define GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, heads=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # Input attention layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads,
                    dropout=dropout)
        )
        # Hidden attention layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels,
                        heads=heads, dropout=dropout)
            )
        # Final attention layer (single head, no concat)
        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels,
                    heads=1, concat=False, dropout=dropout)
        )
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)



# 4) Instantiate & warm‑start
old_state    = torch.load(CHECKPOINT_IN, map_location='cpu')
in_dim       = data.x.size(1)
hidden_dim   = 128
num_classes  = len(assignees)

model = GAT(in_dim, hidden_dim, num_classes, heads=4).to(DEVICE)
state = model.state_dict()
for k, v in old_state.items():
    if k in state and v.size() == state[k].size():
        state[k] = v
model.load_state_dict(state)

# freeze all but final layer
for name, param in model.named_parameters():
    if name.startswith('convs.0.'):
        param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
criterion = torch.nn.CrossEntropyLoss()

# 5) Prepare neighbor loader on bug nodes only
loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    input_nodes=data.train_mask,   # only sample from training bug nodes
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# 6) Training loop with mini‑batches
best_val = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # only compute loss on root nodes (the batch of bug nodes)
        mask = batch.train_mask[:batch.batch_size]
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate on full graph (still cheap since just inference)
    model.eval()
    with torch.no_grad():
        out_full = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))[:num_bugs]
        pred     = out_full.argmax(dim=1)
        val_acc  = (pred[val_idx.to(DEVICE)] == data.y[val_idx].to(DEVICE)).float().mean().item()
        test_acc = (pred[test_idx.to(DEVICE)] == data.y[test_idx].to(DEVICE)).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CHECKPOINT_OUT)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(loader):.4f} "
              f"| Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print(f"\nDone. Best val: {best_val:.4f}. Model saved to {CHECKPOINT_OUT}")
