"""Stage 5: can raw baseline EEG explain any of the residual left over after
Stage 1-4 (population intensity trend + each subject's own intercept/slope)?

Stage 4 found real between-subject slope heterogeneity, and that 8/20
hand-crafted baseline features already predict a subject's sensitivity slope.
This asks the sharper, harder question: after removing population trend AND
each subject's own average level/slope (statsmodels MixedLM's residual e_ij),
does the trial's own raw baseline EEG waveform predict anything about that
trial-to-trial leftover -- something a fixed set of ~20 hand-picked spectral/
entropy features might miss (e.g. spatial patterns across all 28 channels,
or fine-grained spectral shape)?

Two models are trained on the SAME target (stage4's per-trial `residual`
column) and compared on the SAME subject-grouped folds, so the comparison is
apples-to-apples:

  1. Ridge regression on the ~20 hand-crafted BASELINE features (already
     computed in Stage 1) -- the "have we already captured this?" baseline.
  2. A compact EEGNet-style CNN (reusing ../CNN/train_eegnet.py's
     architecture and training loop directly) reading the raw 28-channel x
     1s baseline window.

If (2) beats (1) out-of-fold, raw EEG carries information about individual
trial-level responsiveness that the hand-crafted features don't. If not, the
hand-crafted features already extracted what's extractable and the remainder
looks like irreducible trial-to-trial noise.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CNN"))
import train_eegnet as te  # noqa: E402


CNN_ARGS = SimpleNamespace(
    temporal_filters=8, depth_multiplier=2, dropout=0.35,
    extra_temporal_block=False, multi_scale_temporal=False,
    signal_feature_branch=False, ordinal_auxiliary=False,
    clip=5.0, epoch_normalize=False, grad_clip=1.0,
    input_noise=0.0, channel_dropout=0.0, time_jitter=0,
    loss="huber", margin=1.0, lr=1e-3, weight_decay=1e-3,
    epochs=15, batch_size=64,
)


def load_aligned(epochs_path, residuals_path, features_path):
    z = np.load(epochs_path, allow_pickle=True)
    n_t = z["X"].shape[-1]
    split = n_t // 2  # tmin=-1.0, tmax=1.0 -> first half is baseline
    meta = pd.DataFrame({
        "dataset": z["dataset"].astype(str), "global_subject": z["global_subject"].astype(str),
        "epoch": z["epoch"], "_row": np.arange(len(z["dataset"])),
    })

    residuals = pd.read_csv(residuals_path)
    features = pd.read_csv(features_path)
    baseline_cols = [c for c in features.columns if c.endswith("_baseline")]

    merged = meta.merge(residuals[["dataset", "global_subject", "epoch", "residual"]],
                         on=["dataset", "global_subject", "epoch"], how="inner")
    merged = merged.merge(features[["dataset", "global_subject", "epoch", *baseline_cols]],
                           on=["dataset", "global_subject", "epoch"], how="inner")
    print(f"{len(merged)} / {len(meta)} trials matched between epoch tensor, residuals, and features")

    rows = merged["_row"].to_numpy()
    X_baseline = z["X"][rows][:, :, :split].astype(np.float32)
    y = merged["residual"].to_numpy(np.float32)
    groups = merged["global_subject"].to_numpy()
    hand_features = merged[baseline_cols].to_numpy(np.float32)
    return X_baseline, y, groups, hand_features, baseline_cols


def evaluate_ridge(X, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(y), np.nan)
    for tr, te_idx in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[tr]), y[tr])
        oof_pred[te_idx] = model.predict(scaler.transform(X[te_idx]))
    r, p = pearsonr(oof_pred, y)
    return r, p, oof_pred


def evaluate_cnn(X, y, groups, n_splits, device):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(y), np.nan)
    for fold, (tr, te_idx) in enumerate(gkf.split(X, y, groups)):
        model, mean, std, history, best_epoch = te.train_one_fold(
            X[tr], y[tr], X[te_idx], y[te_idx], None, None, CNN_ARGS, device
        )
        pred = te.predict(model, X[te_idx], mean, std, CNN_ARGS.clip, CNN_ARGS.batch_size, device)
        oof_pred[te_idx] = pred
        fold_r, _ = pearsonr(pred, y[te_idx])
        print(f"  fold {fold}: best_epoch={best_epoch}, val r={fold_r:.3f}")
    r, p = pearsonr(oof_pred, y)
    return r, p, oof_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-npz", required=True, help="nurovo_epochs_baseline.npz")
    ap.add_argument("--residuals", required=True, help="stage4_trial_residuals.csv")
    ap.add_argument("--features", required=True, help="features_normalized.csv (has *_baseline columns)")
    ap.add_argument("--output-prefix", default="stage5")
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    X_baseline, y, groups, hand_features, baseline_cols = load_aligned(
        a.epochs_npz, a.residuals, a.features
    )
    print(f"X_baseline: {X_baseline.shape}, target residual std={y.std():.4g}, "
          f"{len(np.unique(groups))} subjects")

    print("\n=== Ridge on hand-crafted baseline features (apples-to-apples floor) ===")
    ridge_r, ridge_p, ridge_pred = evaluate_ridge(hand_features, y, groups, a.n_splits)
    print(f"out-of-fold r={ridge_r:.4f}, p={ridge_p:.3g}")

    print("\n=== Compact EEGNet-style CNN on raw baseline EEG ===")
    cnn_r, cnn_p, cnn_pred = evaluate_cnn(X_baseline, y, groups, a.n_splits, device)
    print(f"out-of-fold r={cnn_r:.4f}, p={cnn_p:.3g}")

    verdict = ("raw EEG explains meaningfully more" if cnn_r - ridge_r > 0.05 else
               "no meaningful improvement over hand-crafted features" if abs(cnn_r - ridge_r) <= 0.05 else
               "hand-crafted features actually did better (CNN likely undertrained/overfit for this N)")
    print(f"\n=== Verdict: ridge r={ridge_r:.3f} vs CNN r={cnn_r:.3f} -> {verdict} ===")

    pd.DataFrame({
        "global_subject": groups, "residual": y,
        "ridge_pred": ridge_pred, "cnn_pred": cnn_pred,
    }).to_csv(f"{a.output_prefix}_predictions.csv", index=False)
    pd.DataFrame([
        {"model": "ridge_baseline_features", "r": ridge_r, "p": ridge_p, "n_features": len(baseline_cols)},
        {"model": "cnn_raw_baseline_eeg", "r": cnn_r, "p": cnn_p, "n_features": np.prod(X_baseline.shape[1:])},
    ]).to_csv(f"{a.output_prefix}_summary.csv", index=False)
    print(f"\nsaved {a.output_prefix}_predictions.csv, {a.output_prefix}_summary.csv")


if __name__ == "__main__":
    main()
