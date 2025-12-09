import numpy as np
from nflreadpy_dataset_embed import NFLDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- Config -------------------------
WINDOW = 5
BATCH_SIZE = 128
EPOCHS = 25
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(DEVICE))

dataset = NFLDataset(["QB"], 0.0, np.arange(2015, 2024), min_window = WINDOW, max_window = WINDOW, sequential = False)
#  dataset = NFLDataset(["QB"], 0.0, [2022, 2023, 2024], min_window = WINDOW, max_window = WINDOW, sequential = False)
val_dataset = NFLDataset(["QB"], 0.0, [2025], min_window = WINDOW, max_window = WINDOW, sequential = False)
#  val_dataset = dataset.valset()
print("Training Dataset: x {}, y {}".format(dataset.x.shape, dataset.y.shape))
print("Validation Dataset: x {}, y {}".format(val_dataset.x.shape, val_dataset.y.shape))

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------ Model ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)   # regression
        )
    def forward(self, x):
        return self.net(x)

model = MLP(dataset.x.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

def eval_loader(dl):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            b = xb.size(0)
            tot += loss.item() * b
            n += b
    return tot / max(1, n)

# ----------------------- Training -------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        #  print(xb.shape)
        #  print(yb.shape)
        #  input()
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
    tr = eval_loader(train_loader)
    va = eval_loader(val_loader)
    print(f"Epoch {epoch:02d} | train MSE: {tr:.3f} | val MSE: {va:.3f}")
# ----------------------------------------------------------

# ------------------ Quick Sanity Check --------------------

with torch.no_grad():
    if len(val_dataset) > 10:
        for i in np.random.randint(0, len(val_dataset), size =10):
            xb, yb = val_dataset[i]
            pred = model(xb.unsqueeze(0).to(DEVICE)).item()
            print(f"\nExample {i}— true label: {float(yb.item()):.2f} | prediction: {pred:.2f}")
# ----------------------------------------------------------

