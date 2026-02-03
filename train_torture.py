"""Fine-tune v4 model on torture test dataset."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import XTransUNet
from losses import CombinedLoss


class TortureDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files in {data_dir}")
        print(f"Found {len(self.files)} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        inp = torch.from_numpy(data["input"].astype(np.float32))
        target = torch.from_numpy(data["target"].astype(np.float32))
        return inp, target


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 100.0 if mse < 1e-10 else -10 * np.log10(mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_v4/best.pt")
    parser.add_argument("--data-dir", default="/Volumes/External/torture_dataset")
    parser.add_argument("--output-dir", default="checkpoints_torture")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    full_ds = TortureDataset(args.data_dir)
    val_size = int(len(full_ds) * 0.1)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = XTransUNet()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    start_psnr = ckpt.get("best_val_psnr", 0)
    print(f"Loaded: {args.checkpoint} ({start_psnr:.2f} dB)")

    criterion = CombinedLoss(gradient_weight=args.gradient_weight, chroma_weight=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    best_psnr = 0.0
    n_batches = len(train_loader)
    
    print(f"\nFine-tuning for {args.epochs} epochs, {n_batches} batches/epoch")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_psnr_sum = 0.0, 0.0
        
        for i, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss, _ = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                train_psnr_sum += psnr(out, tgt)
            
            if (i + 1) % 500 == 0:
                print(f"  Epoch {epoch} batch {i+1}/{n_batches}")
        
        # Validation
        model.eval()
        val_loss, val_psnr_sum = 0.0, 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                loss, _ = criterion(out, tgt)
                val_loss += loss.item()
                val_psnr_sum += psnr(out, tgt)
        
        scheduler.step()
        
        train_psnr = train_psnr_sum / n_batches
        val_psnr = val_psnr_sum / len(val_loader)
        elapsed = time.time() - t0
        
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            torch.save({"epoch": epoch, "model": model.state_dict(), 
                       "best_val_psnr": best_psnr}, output_dir / "best.pt")
        
        torch.save({"epoch": epoch, "model": model.state_dict(),
                   "best_val_psnr": best_psnr}, output_dir / "latest.pt")
        
        marker = " *" if is_best else ""
        print(f"Epoch {epoch:2d} | Train: {train_psnr:.2f} dB | Val: {val_psnr:.2f} dB | Best: {best_psnr:.2f} dB | {elapsed:.0f}s{marker}")

    print("-" * 60)
    print(f"Done. Best: {best_psnr:.2f} dB saved to {output_dir}")


if __name__ == "__main__":
    main()
