import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------- Config -------------------------
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(DEVICE))
EMBED = False
# ----------------------------------------------------------
NFLDataset.set_seed()

if EMBED:
    from nflreadpy_dataset import NFLDataset
    dataset = NFLDatasetEmbed(["QB"], 0.0, np.arange(2015, 2024))
    val_dataset = NFLDatasetEmbed(["QB"], 0.0, [2025], max_window = dataset.max_window)
else:
    from nflreadpy_dataset_embed import NFLDataset as NFLDatasetEmbed
    dataset = NFLDataset(["QB"], 0.0, np.arange(2015, 2024))
    val_dataset = NFLDataset(["QB"], 0.0, [2025], max_window = dataset.max_window, mu = dataset.mu, sigma = dataset.sigma)
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
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, src, mask = None):
        tgt = torch.roll(src, shifts=1, dims=1).clone()
        tgt[:, 0] = -1  # first output is zero
        if mask is not None:
            tgt_mask = torch.roll(mask, shifts=1, dims=1).clone()
            tgt_mask[:, 0] = -1
        src = self.pe(self.in_proj(src) * math.sqrt(self.d_model))
        tgt = self.pe(self.in_proj(tgt) * math.sqrt(self.d_model))

        if mask is None:
            out = self.transformer(src, tgt)
            pooled = (out).mean(dim=1)
        else:
            out = self.transformer(src, tgt, src_key_padding_mask=mask, tgt_key_padding_mask=tgt_mask)
            valid = ~mask  # False->True
            lengths = valid.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (out * valid.unsqueeze(-1)).sum(dim=1) / lengths
        return self.out_proj(out)
# ----------------------------------------------------------

model = NFLTransformer(dataset.x.shape[2], dataset.y.shape[1], dataset.x.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

def eval_loader(dl):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, mask in dl:
            xb, yb, mask = xb.to(DEVICE), yb.to(DEVICE), mask.to(DEVICE)
            pred = model(xb, mask)
            loss = loss_fn(pred, yb)
            b = xb.size(0)
            tot += loss.item() * b
            n += b
    return tot / max(1, n)

# ----------------------- Training -------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb, mask in train_loader:
        xb, yb, mask = xb.to(DEVICE), yb.to(DEVICE), mask.to(DEVICE)
        opt.zero_grad()
        pred = model(xb, mask)
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
                xb, yb, mask = val_dataset[i]
                pred = model(xb.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)).item()
                print(f"\tExample {i}— true label: {float(yb.item()):.2f} | prediction: {pred:.2f}")
# ----------------------------------------------------------

# ------------------ Quick Sanity Check --------------------
model = model.to("cpu")
with torch.no_grad():
    if len(val_dataset) > 10:
        for i in np.random.randint(0, len(val_dataset), size =10):
            xb, yb, mask = val_dataset[i]
            pred = model(xb.unsqueeze(0).to("cpu"), mask.unsqueeze(0).to("cpu")).item()
            print(f"\nExample {i}— true label: {float(yb.item()):.2f} | prediction: {pred:.2f}")
# ----------------------------------------------------------

