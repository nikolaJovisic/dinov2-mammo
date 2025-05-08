import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_npz(path):
    data = np.load(path)
    X = torch.tensor(data['embeddings'], dtype=torch.float32)
    y = torch.tensor(data['labels'], dtype=torch.float32)
    return X, y

# X, y = load_npz('inbreast.npz')
# y = torch.squeeze(y)
# y = (y == 1.0).float()

# dataset = TensorDataset(X, y)

# n_total = len(dataset)
# n_train = int(n_total * 0.7)
# n_val = int(n_total * 0.15)
# n_test = n_total - n_train - n_val

# train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

# train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

X_train, y_train = load_npz('train_test.npz')
X_val, y_val = load_npz('valid_test.npz')
X_test, y_test = load_npz('test_test.npz')

y_train = np.squeeze(y_train)
y_val = np.squeeze(y_val)
y_test = np.squeeze(y_test)

y_train = (y_train == 1.0).float()
y_val = (y_val == 1.0).float()
y_test = (y_test == 1.0).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

class SimpleLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleLinear(X_train.shape[1]).to('cuda')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to('cuda'), yb.to('cuda')
            logits = model(xb).squeeze()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())
    preds_bin = np.array(all_preds) > 0.5
    return {
        'acc': accuracy_score(all_labels, preds_bin),
        'prec': precision_score(all_labels, preds_bin),
        'rec': recall_score(all_labels, preds_bin),
        'f1': f1_score(all_labels, preds_bin),
        'auroc': roc_auc_score(all_labels, all_preds)
    }

best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(1000):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to('cuda'), yb.to('cuda')
        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    val_metrics = evaluate(model, val_loader)
    train_metrics = evaluate(model, train_loader)

    print(f"Epoch {epoch + 1}")
    print(f"Train: {train_metrics}")
    print(f"Val:   {val_metrics}")

    val_loss = 1 - val_metrics['auroc']
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_model_state)
test_metrics = evaluate(model, test_loader)
print("\nFinal test metrics:")
print(test_metrics)
