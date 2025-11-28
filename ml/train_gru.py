from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model_gru import PriceGenGRU


class PriceSeqDataset(Dataset):
    """
    Wraps the NPZ dataset with X, Y arrays into a PyTorch Dataset.
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # [N, seq_len, feat_dim]
        self.Y = data["Y"].astype(np.float32)  # [N, horizon, 5]
        self.feature_cols = data["feature_cols"]
        self.target_cols = data["target_cols"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        count += X.size(0)

    return total_loss / max(count, 1)


def eval_one_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = loss_fn(pred, Y)

            total_loss += loss.item() * X.size(0)
            count += X.size(0)

    return total_loss / max(count, 1)


def main():
    symbol = "MESM5"
    date_str = "20250318"
    window_in = 64
    horizon = 1

    out_dir = Path("out")
    ds_path = out_dir / f"ds_{symbol}_{date_str}_win{window_in}_h{horizon}.npz"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")

    print(f"Loading dataset from: {ds_path}")
    full_dataset = PriceSeqDataset(ds_path)

    # Basic info
    N, seq_len, feat_dim = full_dataset.X.shape
    _, h, tgt_dim = full_dataset.Y.shape
    print(f"N={N}, seq_len={seq_len}, feat_dim={feat_dim}, horizon={h}, tgt_dim={tgt_dim}")

    # Train/val split (e.g. 90/10)
    val_frac = 0.1
    val_size = max(1, int(N * val_frac))
    train_size = N - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = PriceGenGRU(
        input_dim=feat_dim,
        hidden_dim=64,
        num_layers=2,
        horizon=horizon,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val_loss = float("inf")
    best_path = out_dir / f"gru_pricegen_{symbol}_{date_str}.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": feat_dim,
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "horizon": horizon,
                    "feature_cols": full_dataset.feature_cols,
                    "target_cols": full_dataset.target_cols,
                },
                best_path,
            )
            print(f"  → New best model saved to {best_path}")

    print("Training complete.")
    print(f"Best val_loss = {best_val_loss:.4f}")
    print(f"Best model path: {best_path}")


if __name__ == "__main__":
    main()
