"""
Evaluate a trained ATCNetRegressor checkpoint: MAE, RMSE, R^2, Pearson r, plus a
predicted-vs-actual scatter plot and a Bland-Altman-style residual plot.

Usage:
    python evaluate.py --checkpoint runs/atcnet_raw/best.pt --data data/processed
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.atcnet import ATCNetRegressor
from train import load_pooled


def evaluate(model, X, y, device, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)

    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)
    r, p = pearsonr(y, preds)
    return preds, {"MAE": mae, "RMSE": rmse, "R2": r2, "pearson_r": r, "pearson_p": p}


def make_plots(y, preds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.scatter(y, preds, alpha=0.3, s=10)
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True relative intensity")
    plt.ylabel("Predicted relative intensity")
    plt.title("Predicted vs. actual stimulus intensity")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_pred_vs_actual.png", dpi=150)
    plt.close()

    residuals = preds - y
    mean_val = (preds + y) / 2
    plt.figure(figsize=(5, 5))
    plt.scatter(mean_val, residuals, alpha=0.3, s=10)
    plt.axhline(residuals.mean(), color="k", linestyle="--")
    plt.xlabel("Mean of prediction and truth")
    plt.ylabel("Prediction - truth")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/atcnet_raw/eval")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, groups = load_pooled(Path(args.data))
    n_channels, n_samples = X.shape[1], X.shape[2]

    model = ATCNetRegressor(n_channels=n_channels, n_samples=n_samples).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    preds, metrics = evaluate(model, X, y, device)
    print("Metrics:", metrics)
    make_plots(y, preds, Path(args.out))
    print(f"Plots written to {args.out}")


if __name__ == "__main__":
    main()
