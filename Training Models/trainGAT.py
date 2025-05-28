import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from  GraphSAGEFULLGRAPH import evaluate_top_k

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
        self.classifier = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def train_gat(model, data, lr=1e-3, weight_decay=5e-4,
              epochs=100, patience=10, batch_size=1024, device=torch.device('cpu')):
    model.to(device)
    data = data.to(device)

    train_loader = NeighborLoader(
        data,
        num_neighbors=[10] * len(model.convs),
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, patience_counter = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            y = batch.y.to(device)
            mask = batch.train_mask.to(device)

            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out[mask], y[mask])
            loss.backward() 
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()

        avg_loss = total_loss / data.train_mask.sum().item()

        # Validation on full graph
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val, best_state, patience_counter = val_acc, model.state_dict(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch:03d} | Best Val Acc: {best_val:.4f}")
                break

    print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | Val {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model


if __name__ == '__main__':
    from datasetupBERT import load_buglist_data
    # Load data
    data = torch.load('graph_BERT_FULL.pt')
    bug_info = load_buglist_data(r"F:\Thesis\RAW DATA\buglist")
    from GraphSAGEFULLGRAPH import prepare_labels_and_masks
    data = prepare_labels_and_masks(data, bug_info)

    # Instantiate GAT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(
        in_channels=data.x.size(1),
        hidden_channels=128,
        out_channels=data.num_users,
        num_layers=3,
        heads=8,
        dropout=0.6,
    )

    # Train
    model = train_gat(model, data,
                      lr=1e-3, weight_decay=5e-4,
                      epochs=200, patience=20,
                      device=device)

    # Evaluate
    res = evaluate_top_k(model, data, ks=[1,3,5])
    for k, acc in res.items():
        print(f"Top-{k} Accuracy: {acc:.4f}")

    # Save
    torch.save(model.state_dict(), 'gat_bugtriage.pth')