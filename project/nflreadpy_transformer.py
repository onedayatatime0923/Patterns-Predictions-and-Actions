import math
import numpy as np
from nflreadpy_dataset_embed import NFLDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- Config -------------------------
BATCH_SIZE = 32
EPOCHS = 200
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(DEVICE))
# ----------------------------------------------------------

#  dataset = NFLDataset(["QB"], 0.0, [2021, 2022, 2023, 2024])
dataset = NFLDataset(["QB"], 0.0, np.arange(2015, 2024))
val_dataset = NFLDataset(["QB"], 0.0, [2025], max_window = dataset.max_window)
#  val_dataset = dataset.valset()
print("Training Dataset: x {}, y {}".format(dataset.x.shape, dataset.y.shape))
print("Validation Dataset: x {}, y {}".format(val_dataset.x.shape, val_dataset.y.shape))

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------ Model ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #  self.pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class NFLTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, max_sequence, d_model = 512):
        super().__init__()
        self.d_model = d_model
        self.pe = PositionalEncoding(self.d_model)

        self.in_proj = nn.Linear(in_dim, self.d_model)
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            #  nhead=8,
            #  num_encoder_layers=2,
            #  num_decoder_layers=2,
            #  dropout = 0.3,
            batch_first=True
        )
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Flatten(),
            nn.Linear(max_sequence, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, src):
        tgt = torch.roll(src, shifts=1, dims=1).clone()
        tgt[:, 0] = 0  # first output is zero
        src = self.pe(self.in_proj(src))
        tgt = self.pe(self.in_proj(tgt))
        out = self.transformer(src, tgt)
        return self.out_proj(out)
# ----------------------------------------------------------

model = NFLTransformer(dataset.x.shape[2], dataset.y.shape[1], dataset.x.shape[1]).to(DEVICE)
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
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        #  print(xb.shape)
        #  print(yb.shape)
        #  print(pred.shape)
        #  print(loss)
        #  input()
    model.eval()
    tr = eval_loader(train_loader)
    va = eval_loader(val_loader)
    print(f"Epoch {epoch:02d} | train MSE: {tr:.3f} | val MSE: {va:.3f}")

    with torch.no_grad():
        if len(val_dataset) > 3:
            for i in range(3):
                xb, yb = val_dataset[i]
                pred = model(xb.unsqueeze(0).to(DEVICE)).item()
                print(f"\tExample {i}— true label: {float(yb.item()):.2f} | prediction: {pred:.2f}")
# ----------------------------------------------------------

# ------------------ Quick Sanity Check --------------------
with torch.no_grad():
    if len(val_dataset) > 10:
        for i in np.random.randint(0, len(val_dataset), size =10):
            xb, yb = val_dataset[i]
            pred = model(xb.unsqueeze(0).to("cpu")).item()
            print(f"\nExample {i}— true label: {float(yb.item()):.2f} | prediction: {pred:.2f}")
# ----------------------------------------------------------

