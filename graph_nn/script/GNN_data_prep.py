# %% [markdown]
# # Data Prep

# %% [markdown]
# ## Parse the json

# %%
import json
with open('../solve/data/Json/graph_out.json', 'r') as f:
    adjacency_list = json.load(f)

# %% [markdown]
# ## Convert to `edge_index` for PyTorch

# %%
import torch
from torch_geometric.data import Data

# Map node names (strings) to indices (integers)
node_mapping = {node: idx for idx, node in enumerate(adjacency_list.keys())}

# Create edge indices with numerical node IDs
edges = []
for node, neighbors in adjacency_list.items():
    for neighbor in neighbors:
        edges.append((node_mapping[node], node_mapping[neighbor]))

# Convert to tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index


# %% [markdown]
# ## Construct default `node_features` matrix

# %%
num_nodes = len(adjacency_list)
node_features = torch.eye(num_nodes)

# %% [markdown]
# ## Extract labels `y`

# %%
# Extract labels for each node based on the subject
node_labels = {}
for node in adjacency_list.keys():
    # Extract the subject (second component after the first dot)
    subject = node.split('.')[1] if '.' in node else "Unknown"
    node_labels[node] = subject

label_mapping = {label: idx for idx, label in enumerate(set(node_labels.values()))}
y = [label_mapping[label] for label in node_labels.values()]
y = torch.tensor(y, dtype=torch.long)

# %% [markdown]
# ## Import to torch_geometric Data

# %%
data = Data(x=node_features, edge_index=edge_index, y=y)

# %% [markdown]
# ## Do a train-val-test split

# %%
# Define the number of nodes
num_nodes = data.num_nodes

# Train-test split proportions
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Randomly shuffle node indices
indices = torch.randperm(num_nodes)

# Split indices
train_size = int(train_ratio * num_nodes)
val_size = int(val_ratio * num_nodes)

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:]

# Create masks
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True
