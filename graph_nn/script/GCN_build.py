# %% [markdown]
# # Build GCN and train it

# %%
# Build a one-layer GCN
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels = 16):
        super().__init__()
        self.gcn = GCNConv(data.num_features, hidden_channels)
        self.out = Linear(hidden_channels, len(label_mapping))

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z

import matplotlib.pyplot as plt

x = data.x
train_y = data.y[data.train_mask]
val_y = data.y[data.val_mask]

learning_rate = 0.01
epochs = 100
hidden_channels = 512
model = GCN(hidden_channels)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

# the training loop
accs = []
losses = []
val_accs = []
val_losses = []
test_accs = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    h, z = model(x, edge_index)
    train_h, train_z = h[data.train_mask], z[data.train_mask]
    loss = criterion(train_z, train_y)
    loss.backward()
    optimizer.step()

    acc = accuracy(train_z.argmax(dim=1), train_y)
    accs.append(acc)
    losses.append(loss.detach().numpy())

    model.eval()
    with torch.no_grad():
        val_h, val_z = h[data.val_mask], z[data.val_mask]
        val_acc = accuracy(val_z.argmax(dim=1), val_y)
        val_loss = criterion(val_z, val_y)
        val_accs.append(val_acc)
        val_losses.append(val_loss.detach().numpy())
        test_acc = accuracy(z[data.test_mask].argmax(dim=1), data.y[data.test_mask])
        test_accs.append(test_acc)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:>3}, Loss: {loss:.4f}, Accuracy: {acc*100:>6.2f}%; Val Accuracy: {val_acc*100:>6.2f}%; Test Accuracy: {test_acc*100:>6.2f}%")
    

plt.figure(figsize=(10, 6))
plt.plot(losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("fig/loss.pdf", dpi = 300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(accs, label="Train")
plt.plot(val_accs, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("fig/acc.pdf", dpi = 300)
plt.show()

# %%
# Build a two-layer GCN
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels1 = 16, hidden_channels2 = 16):
        super().__init__()
        self.gcn1 = GCNConv(data.num_features, hidden_channels1)
        self.gcn2 = GCNConv(hidden_channels1, hidden_channels2)
        self.out = Linear(hidden_channels2, len(label_mapping))

    def forward(self, x, edge_index):
        h1 = self.gcn1(x, edge_index).relu()
        h = self.gcn2(h1, edge_index).relu()
        z = self.out(h)
        return h, z
    
import matplotlib.pyplot as plt

x = data.x
train_y = data.y[data.train_mask]
val_y = data.y[data.val_mask]

learning_rate = 0.01
epochs = 100
hidden_channels1 = 512
hidden_channels2 = 16
model = GCN(hidden_channels1, hidden_channels2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

# the training loop
accs = []
losses = []
val_accs = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    h, z = model(x, edge_index)
    train_h, train_z = h[data.train_mask], z[data.train_mask]
    loss = criterion(train_z, train_y)
    loss.backward()
    optimizer.step()

    acc = accuracy(train_z.argmax(dim=1), train_y)
    accs.append(acc)
    losses.append(loss.detach().numpy())

    model.eval()
    with torch.no_grad():
        val_h, val_z = h[data.val_mask], z[data.val_mask]
        val_acc = accuracy(val_z.argmax(dim=1), val_y)
        val_loss = criterion(val_z, val_y)
        val_accs.append(val_acc)
        val_losses.append(val_loss.detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:>3}, Loss: {loss:.4f}, Accuracy: {acc*100:>6.2f}%; Val Accuracy: {val_acc*100:>6.2f}%")
    

plt.figure(figsize=(12, 6))
plt.plot(losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("fig/loss.pdf", dpi = 300)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(accs, label="Train")
plt.plot(val_accs, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("fig/acc.pdf", dpi = 300)
plt.show()

# %%
# baseline model
import torch
from collections import Counter

def compute_baseline(edge_index, y, test_mask):
    # Number of nodes
    num_nodes = y.size(0)
    
    # Create adjacency list
    adjacency_list = {i: [] for i in range(num_nodes)}
    for src, dst in edge_index.t().tolist():
        adjacency_list[src].append(dst)
    
    # Predict labels for test nodes
    y_pred = torch.zeros_like(y)
    for node in range(num_nodes):
        if test_mask[node]:
            # Get labels of the node's neighbors
            neighbor_labels = [y[neighbor].item() for neighbor in adjacency_list[node]]
            
            if neighbor_labels:
                # Most common label among neighbors
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                y_pred[node] = most_common_label
            else:
                # If no neighbors, fallback to a default label (e.g., 0)
                y_pred[node] = 0

    return y_pred

y_pred = compute_baseline(edge_index, y, data.test_mask)
# Calculate accuracy on the test set
correct = (y_pred[data.test_mask] == y[data.test_mask]).sum().item()
total = data.test_mask.sum().item()
accuracy = correct / total

print(f"Baseline Accuracy: {accuracy:.4f}")


# %%
# Build a one-layer GCN
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels = 16):
        super().__init__()
        self.gcn = GCNConv(data.num_features, hidden_channels)
        self.out = Linear(hidden_channels, len(label_mapping))

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z

import matplotlib.pyplot as plt

x = data.x
train_y = data.y[data.train_mask]
val_y = data.y[data.val_mask]

learning_rate = 0.01
epochs = 30
hidden_channels = 512
model = GCN(hidden_channels)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

# the training loop
accs = []
losses = []
val_accs = []
val_losses = []
test_accs = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    h, z = model(x, edge_index)
    train_h, train_z = h[data.train_mask], z[data.train_mask]
    loss = criterion(train_z, train_y)
    loss.backward()
    optimizer.step()

    acc = accuracy(train_z.argmax(dim=1), train_y)
    accs.append(acc)
    losses.append(loss.detach().numpy())

    model.eval()
    with torch.no_grad():
        val_h, val_z = h[data.val_mask], z[data.val_mask]
        val_acc = accuracy(val_z.argmax(dim=1), val_y)
        val_loss = criterion(val_z, val_y)
        val_accs.append(val_acc)
        val_losses.append(val_loss.detach().numpy())
        test_acc = accuracy(z[data.test_mask].argmax(dim=1), data.y[data.test_mask])
        test_accs.append(test_acc)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:>3}, Loss: {loss:.4f}, Accuracy: {acc*100:>6.2f}%; Val Accuracy: {val_acc*100:>6.2f}%; Test Accuracy: {test_acc*100:>6.2f}%")

# %%
# get y_pred
y_pred_gnn = z.argmax(dim=1)

# plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cm = confusion_matrix(data.y[data.test_mask], y_pred_gnn[data.test_mask])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, fmt="d", cmap="Blues")
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.savefig("fig/confusion_matrix_gnn.pdf", dpi = 300)
plt.show()


cm = confusion_matrix(data.y[data.test_mask], y_pred[data.test_mask])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, fmt="d", cmap="Blues")
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.savefig("fig/confusion_matrix_baseline.pdf", dpi = 300)
plt.show()
