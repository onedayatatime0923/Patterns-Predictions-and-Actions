import nflgame
from collections import defaultdict, deque
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------- Config -----------------
YEARS = [2018, 2019]          # You can extend this list (e.g., [2015, 2016, 2017, 2018, 2019])
WEEKS = list(range(1, 18))    # Regular season weeks
WINDOW = 10                   # Past weeks to aggregate as features
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stats to aggregate from the past WINDOW weeks to form features.
# You can add/remove any nflgame stat keys you care about:
STAT_KEYS = [
    "rushing_att",
    "rushing_yds",
    "rushing_tds",
    "receiving_rec",
    "receiving_yds",
    "receiving_tds",
    "passing_att",
    "passing_cmp",
    "passing_yds",
    "passing_tds",
    "passing_int"
]

# The label for the CURRENT week (prediction target).
# Change to, e.g., "rushing_tds" or a custom fantasy formula below.
LABEL_KEY = "rushing_yds"
# ------------------------------------------


# ----------------- Utils ------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def player_key(p):
    """Stable player identifier."""
    pid = getattr(p, "gsis_id", None) or getattr(p, "playerid", None)
    if not pid:
        # Fallback
        pid = f"{getattr(p, 'name', str(p))}|{getattr(p, 'team', '')}"
    return pid

def stat_line_from_player(p, keys):
    """Extract a dict of stats with zero default."""
    return {k: int(getattr(p, k, 0) or 0) for k in keys}

def agg_stats(deq, keys):
    """Sum stats over the last WINDOW weeks."""
    totals = {k: 0 for k in keys}
    for s in deq:
        for k in keys:
            totals[k] += s.get(k, 0)
    return totals

def to_feature_vec(agg_dict, keys):
    """Flatten the totals dictionary into a stable feature vector."""
    return np.array([agg_dict[k] for k in keys], dtype=np.float32)

def current_label_from_player(p):
    """The label is the current week's LABEL_KEY (e.g., rushing_yds)."""
    return float(getattr(p, LABEL_KEY, 0) or 0.0)
# ------------------------------------------


# -------------- Dataset Build --------------
set_seed(SEED)

# rolling[pid] -> deque of last WINDOW per-week stat dicts (for features)
rolling = defaultdict(lambda: deque(maxlen=WINDOW))

# We’ll accumulate (X, y) pairs here
X_rows = []
y_rows = []

# Iterate in chronological order (year, week)
for y in YEARS:
    for w in WEEKS:
        games = nflgame.games(y, week=w)
        players = nflgame.combine_game_stats(games)

        # 1) For each player who appears this week, if we have a full WINDOW
        #    of prior weeks, create a training example BEFORE ingesting current week.
        for p in players:
            pid = player_key(p)
            prior_deq = rolling.get(pid, None)
            if prior_deq is not None and len(prior_deq) == WINDOW:
                # Build features from past WINDOW weeks
                padded = list(prior_deq)
                flat = []
                for week_dict in padded:
                    flat.extend([week_dict.get(k, 0) for k in STAT_KEYS])
                x = np.array(flat, dtype=np.float32)

                # Label is the current-week stat
                y_label = current_label_from_player(p)

                X_rows.append(x)
                y_rows.append(y_label)

        # 2) Now ingest current week stats so they count for future weeks
        for p in players:
            pid = player_key(p)
            line = stat_line_from_player(p, STAT_KEYS)
            rolling[pid].append(line)

X = np.stack(X_rows) if len(X_rows) else np.zeros((0, len(STAT_KEYS)), dtype=np.float32)
y = np.array(y_rows, dtype=np.float32)

print(f"Built dataset with {len(X)} samples, feature_dim={X.shape[1]}")

# Simple guard in case early-season years don't yield enough samples
if len(X) == 0:
    raise RuntimeError("No samples built. Try expanding YEARS, adjusting WINDOW, or STAT_KEYS.")
# ------------------------------------------


# -------------- Torch Dataset --------------
class PlayerWeekDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Train/val split
n = len(X)
idxs = np.arange(n)
np.random.shuffle(idxs)
val_n = int(VAL_SPLIT * n)
val_idx = idxs[:val_n]
train_idx = idxs[val_n:]

train_ds = PlayerWeekDataset(X[train_idx], y[train_idx])
val_ds   = PlayerWeekDataset(X[val_idx],   y[val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# ------------------------------------------


# ----------------- Model -------------------
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # regression to yards (or whatever LABEL_KEY)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            bsz = xb.size(0)
            total_loss += loss.item() * bsz
            total_n += bsz
    return total_loss / max(1, total_n)
# ------------------------------------------


# --------------- Training ------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    tr_loss = evaluate(train_loader)
    va_loss = evaluate(val_loader)
    print(f"Epoch {epoch:02d} | train MSE: {tr_loss:.3f} | val MSE: {va_loss:.3f}")
# ------------------------------------------


# ------------- Example Inference -----------
# Take a few validation samples and print predictions
model.eval()
with torch.no_grad():
    if len(val_ds) > 0:
        xb, yb = val_ds[0]
        pred = model(xb.unsqueeze(0).to(DEVICE)).item()
        print(f"\nExample — true {LABEL_KEY}: {yb.item():.1f} | predicted: {pred:.1f}")

