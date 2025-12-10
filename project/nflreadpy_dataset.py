from collections import defaultdict, deque
import pandas as pd
import nflreadpy as nfl
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SEED = 0
#  YEARS = [2021, 2022, 2023, 2024]      # extend if you want more history (e.g., [2021, 2022, 2023])
YEARS = [2023, 2024]      # extend if you want more history (e.g., [2021, 2022, 2023])
VAL_SPLIT = 0.2

#  LOAD_YEARS = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
#  LOAD_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
LOAD_YEARS = [2021, 2022, 2023, 2024, 2025]

class NFLDataset(Dataset):
    def __init__(self, position_filter= None, valsplit = VAL_SPLIT, year = YEARS, min_window = 1, max_window = None, sequential = True, mu = None, sigma = None):
        super().__init__()
        self.position_filter = position_filter
        self.val_split = valsplit

        player_stats = nfl.load_player_stats(LOAD_YEARS)
        ps = player_stats.to_pandas()
        assert("player_id" in ps.columns)
        assert("player_name" in ps.columns)
        assert("team" in ps.columns)
        assert("season" in ps.columns)
        assert("week" in ps.columns)
        assert("fantasy_points_ppr" in ps.columns)
        assert("position" in ps.columns)
        #  ps = ps[ps["position"] == "QB"]
        if ps.empty:
            raise RuntimeError("player_stats is empty. Check seasons or data availability.")
        #  print(ps)
        #  input()

        cols = set(ps.columns)
        for i in ["player_id", "player_name", "team", "season", "week", "position_group", "headshot_url", "season_type"]:
            cols.discard(i)

        cols = [c for c in cols if np.issubdtype(ps[c].dtype, np.number)]
        #  print("Feature: {}".format(" ".join(cols)))
        #  input()
        ps.sort_values(by=["player_id", "season", "week"], inplace=True)
        ps.reset_index(drop=True, inplace=True)

        if max_window is None:
            player_length = defaultdict(int)
            for _, row in ps.iterrows():
                pid = row["player_id"]
                player_length[pid] += 1
            self.max_window = max(player_length.values())
            print("Max player window: {}".format(self.max_window))
        else:
            self.max_window = max_window
        #  input()

        self.x, self.mask, self.y = [], [], []
        self.x_val, self.mask_val, self.y_val = [], [], []

        # We'll require a full min_window-week history before creating a sample
        rolling = defaultdict(lambda: deque(maxlen=self.max_window))

        # Iterate row-by-row chronologically
        for _, row in ps.iterrows():
            pid = row["player_id"]
            position = row["position"]
            #  print(rolling[pid])
            #  print(position)
            #  input()
            if len(rolling[pid]) >= min_window and not pd.isna(row["fantasy_points_ppr"]) and row["season"] in year:
                # Flatten oldest->most recent
                value = []
                for r in rolling[pid]:
                    value.append([])
                    for c in cols:
                        v = r.get(c, 0.0)
                        if pd.isna(v): v = 0.0
                        value[-1].append(float(v))

                value = torch.tensor(value)
                feature = torch.zeros(self.max_window, len(cols))
                feature[-1 * value.shape[0]:] = value
                # mask: True means padded (to be ignored by Transformer)
                mask = torch.ones(self.max_window).bool()
                mask[-1 * value.shape[0]:] = False
                if not sequential:
                    feature = feature.view((-1))
                #  print(value.shape)
                #  print(feature)
                #  print(mask)
                #  input()

                # Label = current week's chosen metric (or fallback)
                y_val = float(row["fantasy_points_ppr"])
                #  print(feature)
                #  print(y_val)
                #  input()

                if self.position_filter is None or position in self.position_filter:
                    self.x.append(feature)
                    self.mask.append(mask)
                    self.y.append(y_val)
            # Ingest current week into the rolling window (store only the
            # feature cols)
            # rolling[pid].append()
            rolling[pid].append({c: row[c] for c in cols})
            
        #  print(self.x)
        #  print(self.y)
        self.x = torch.stack(self.x, dim = 0)
        self.mask = torch.stack(self.mask, dim = 0)
        self.y = torch.tensor(self.y).float().view(-1, 1)
        #  print(self.x.shape)
        #  print(self.y.shape)
        #  input()

        if mu is None and sigma is None:
            self.mu = self.x.mean(axis=0, keepdims=True)
            self.sigma = self.x.std(axis=0, keepdims=True)
            self.sigma[self.sigma < 1e-6] = 1.0  # avoid div-by-zero
        else:
            self.mu = mu
            self.sigma = sigma
            assert(mu is not None and sigma is not None)
        self.x = (self.x - self.mu) / self.sigma

        self.y_mu = self.y.mean()
        self.y_sigma = self.y.std().clamp(min=1e-6)
        self.y = (self.y - self.y_mu) / self.y_sigma

        idx = torch.randperm(len(self.x))
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.mask = self.mask[idx]

        val_n = int(self.val_split * len(self.x))

        self.x_val = self.x[:val_n]
        self.y_val = self.y[:val_n]
        self.mask_val = self.mask[:val_n]
        assert(len(self.x_val) == len(self.y_val))
        assert(len(self.x_val) == len(self.mask_val))

        self.x = self.x[val_n:]
        self.y = self.y[val_n:]
        self.mask = self.mask[val_n:]

        #  print("Train: x {}, y {}".format(self.x.shape, self.y.shape))
        #  if self.x_val.shape[0] > 0:
        #      print("Val: x {}, y {}".format(self.x_val.shape, self.y_val.shape))
        #  input()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]

    def valset(self):
        return NFLValset(self.x_val, self.mask_val, self.y_val)

    @classmethod
    def set_seed(cls, s=SEED):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

class NFLValset(Dataset):
    def __init__(self, x, mask, y):
        super().__init__()
        self.x = torch.tensor(x)
        self.mask = torch.tensor(mask)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]

if __name__ == "__main__":
    WINDOW = 5
    # Create dataset & dataloader
    #  dataset = NFLDataset(["QB"], min_window = WINDOW, max_window = WINDOW, sequential = False)
    dataset = NFLDataset(["QB"], 0.0, [2021, 2022, 2023, 2024])

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #  val_loader = DataLoader(dataset.valset(), batch_size=32, shuffle=True)

    # Example: iterate
    for x, y, _ in train_loader:
        print(x.shape, y.shape)
        break

    #  for x, y, _ in val_loader:
    #      print(x.shape, y.shape)
    #      break
