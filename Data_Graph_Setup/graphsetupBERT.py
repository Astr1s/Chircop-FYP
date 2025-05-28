import pickle
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import random

def build_pyg_graph(bug_info, bug_history, bug_descriptions, subset_size=None, device=None):
    bug_texts = {bid: text for bid, text in bug_descriptions.items() if text.strip()}
    non_empty = [bid for bid in bug_texts if bid in bug_info]
    if not non_empty:
        raise ValueError("No bug descriptions available after preprocessing.")

    if subset_size and subset_size < len(non_empty):
        sampled = random.sample(non_empty, subset_size)
        print(f"Selected {subset_size} of {len(non_empty)} bugs.")
    else:
        sampled = non_empty
        print(f"Using all {len(non_empty)} bugs.")
    bug_info = {bid: bug_info[bid] for bid in sampled}
    bug_texts = {bid: bug_texts[bid] for bid in sampled}

    sorted_bugs = sorted(sampled)
    bug_idx = {bid: i for i, bid in enumerate(sorted_bugs)}
    n_bugs = len(sorted_bugs)

    # BERT setup
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    embeds = []
    texts = [bug_texts[bid][:512] for bid in sorted_bugs]
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
            out = model(**inputs)
            embeds.append(out.last_hidden_state[:,0,:].cpu())
    bug_embeddings = torch.cat(embeds, dim=0)

    users = set()
    for bid, recs in bug_history.items():
        for rec in recs:
            who = rec.get('who','').strip()
            if who: users.add(who)
    for bid, info in bug_info.items():
        a = info.get('assignee','').strip()
        if a: users.add(a)
    users = sorted(users)
    user_idx = {u: idx + n_bugs for idx, u in enumerate(users)}
    n_users = len(users)
    print(f"Found {n_users} users.")

    edges, attrs = [], []
    for bid, recs in bug_history.items():
        if bid not in bug_idx: continue
        b_i = bug_idx[bid]
        for rec in recs:
            who = rec.get('who','').strip()
            if who and who in user_idx:
                u_i = user_idx[who]
                edges += [[b_i, u_i], [u_i, b_i]]
                attrs += [1.0, 1.0]

    for bid, info in bug_info.items():
        b_i = bug_idx[bid]
        a = info.get('assignee','').strip()
        if a and a in user_idx:
            u_i = user_idx[a]
            edges += [[b_i, u_i], [u_i, b_i]]
            attrs += [1.0, 1.0]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.float)

    x = torch.cat([bug_embeddings, torch.zeros(n_users, bug_embeddings.size(1))], dim=0)
    node_type = torch.zeros(n_bugs + n_users, dtype=torch.long)
    node_type[n_bugs:] = 2

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_type=node_type)
    data.num_bugs = n_bugs
    data.num_users = n_users
    data.bug_idx_map = bug_idx
    data.user_idx_map = user_idx
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges.")
    return data

if __name__ == '__main__':
    # Load preprocessed data from our pkl files
    with open('Mozilla_preprocessed_bug_info.pkl', 'rb') as f:
        bug_info = pickle.load(f)
    with open('Mozilla_preprocessed_bug_history.pkl', 'rb') as f:
        bug_history = pickle.load(f)
    with open('Mozilla_preprocessed_bug_desc.pkl', 'rb') as f:
        bug_desc = pickle.load(f)

    data = build_pyg_graph(bug_info, bug_history, bug_desc, subset_size=400000)
    torch.save(data, 'MozillaGraphB.pt')
    print('Done. Graph saved as graph_from_preprocessed.pt')
