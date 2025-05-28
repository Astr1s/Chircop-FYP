import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle

data = torch.load("Training Models\graph\graph_BERT_FULL.pt")
bug_info = pickle.load(open("preprocessed_bug_info.pkl", "rb"))

assignees = sorted({info["assignee"] for info in bug_info.values() if info["assignee"]})
assignee2idx = {a: i for i, a in enumerate(assignees)}

num_bugs = data.num_bugs
y = torch.full((num_bugs,), -1, dtype=torch.long)
for bid, idx in data.bug_idx_map.items():
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
val_mask   = torch.zeros(num_bugs, dtype=torch.bool)
test_mask  = torch.zeros(num_bugs, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def accuracy(logits, labels, mask, topk=(1,)):
    accs = []
    _, pred = logits[mask].topk(max(topk), dim=1, largest=True, sorted=True)
    correct = pred.eq(labels[mask].unsqueeze(1).expand_as(pred))
    for k in topk:
        acc = correct[:, :k].any(dim=1).float().mean().item()
        accs.append(acc)
    return accs

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = data.x.to(device)
edge_index = data.edge_index.to(device)
y = y.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

model = GCN(
    in_channels=x.size(1),
    hidden_channels=128,
    out_channels=len(assignees)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training
best_val_acc = 0
for epoch in range(1, 251):
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    logits_bug = logits[:num_bugs]  # restrict to bug nodes
    loss = criterion(logits_bug[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_top1, val_top3, val_top5 = accuracy(logits_bug, y, val_mask, topk=(1,3,5))
        test_top1, test_top3, test_top5 = accuracy(logits_bug, y, test_mask, topk=(1,3,5))

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(model.state_dict(), "best_gcn.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val@1: {val_top1:.4f} | Test@1: {test_top1:.4f} | Val@5: {val_top5:.4f} | Test@5: {test_top5:.4f}")

print("Training complete.")
