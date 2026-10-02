"""Stage 16: the one lever nothing so far has tried -- a raw-EEG deep model
using ALL 28 channels, on the FULL 678-subject / 29,515-trial pooled dataset,
predicting laser_power directly (not a residual).

Stage 5 tried raw-EEG CNN vs. hand-crafted ridge and ridge won, but that was
on a much smaller target (the trial-level residual left over after Stage 4's
population+subject model) and a much smaller cohort (11k trials, the 124
within-subject-only subjects). This repeats the same honest apples-to-apples
comparison -- same trials, same subject-grouped folds, same architecture --
but at full scale and against the actual deployable target (laser_power),
so a fair verdict can be reached on whether more data changes the Stage 5
conclusion.

Also builds a simple two-feature stacked ensemble (hand-crafted composite +
CNN raw-EEG prediction) since the two model families could be picking up
complementary information even if neither wins outright alone.
"""
import argparse
import sys
import time
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
    # Stage 16's first pass reused Stage 5's small architecture (tuned for 11k
    # trials, deliberately below EEGNet's own class default of
    # temporal_filters=12/depth_multiplier=3, to avoid overfitting that
    # smaller cohort). At 29.5k trials that headroom concern doesn't apply --
    # this reverts to the class default capacity. Two of five Stage-10 folds
    # also hit the 20-epoch cap without early-stopping on their own, so epochs
    # is raised to 30 to let those actually converge.
    temporal_filters=12, depth_multiplier=3, dropout=0.4,
    extra_temporal_block=False, multi_scale_temporal=False,
    signal_feature_branch=False, ordinal_auxiliary=False,
    clip=5.0, epoch_normalize=False, grad_clip=1.0,
    input_noise=0.0, channel_dropout=0.0, time_jitter=0,
    loss="huber", margin=1.0, lr=1e-3, weight_decay=1e-3,
    epochs=30, batch_size=64,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-npz", default="nurovo_epochs_baseline_all.npz")
    ap.add_argument("--hand-crafted-scored", default="stage15_scored_trials.csv",
                     help="has global_subject/epoch/dataset + nurovo_pain_raw (Stage 15 ridge composite)")
    ap.add_argument("--output-prefix", default="stage16")
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    t0 = time.time()
    z = np.load(a.epochs_npz, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["power"].astype(np.float32)
    groups = z["global_subject"].astype(str)
    meta = pd.DataFrame({"dataset": z["dataset"].astype(str), "global_subject": groups,
                          "epoch": z["epoch"], "_row": np.arange(len(groups))})
    print(f"loaded {a.epochs_npz} in {time.time()-t0:.1f}s: X={X.shape}, {len(np.unique(groups))} subjects")

    hand = pd.read_csv(a.hand_crafted_scored)[["dataset", "global_subject", "epoch", "nurovo_pain_raw"]]
    merged = meta.merge(hand, on=["dataset", "global_subject", "epoch"], how="inner").dropna(subset=["nurovo_pain_raw"])
    print(f"{len(merged)} / {len(meta)} trials matched with Stage 15's hand-crafted composite (for the ensemble step)")

    gkf = GroupKFold(n_splits=a.n_splits)
    cnn_oof = np.full(len(y), np.nan)
    for fold, (tr, te_idx) in enumerate(gkf.split(X, y, groups)):
        fold_t0 = time.time()
        model, mean, std, history, best_epoch = te.train_one_fold(
            X[tr], y[tr], X[te_idx], y[te_idx], None, None, CNN_ARGS, device)
        pred = te.predict(model, X[te_idx], mean, std, CNN_ARGS.clip, CNN_ARGS.batch_size, device)
        cnn_oof[te_idx] = pred
        fold_r, _ = pearsonr(pred, y[te_idx])
        print(f"  fold {fold}: n_train={len(tr)}, n_test={len(te_idx)}, best_epoch={best_epoch}, "
              f"val r={fold_r:.4f}, took {time.time()-fold_t0:.0f}s")

    cnn_r, cnn_p = pearsonr(cnn_oof, y)
    print(f"\n=== Stage 16 CNN (28-channel raw EEG, full 678-subject pool) vs laser_power ===")
    print(f"out-of-fold r={cnn_r:.4f}, p={cnn_p:.3g}")
    print(f"Comparison points: Stage 15 ridge (hand-crafted, same pool) r=0.314; "
          f"Stage 5 CNN (11k trials, residual target) r=0.268")

    pd.DataFrame({"dataset": meta["dataset"], "global_subject": groups, "epoch": meta["epoch"],
                  "laser_power": y, "cnn_pred": cnn_oof}).to_csv(f"{a.output_prefix}_cnn_predictions.csv", index=False)

    # Stacked ensemble: combine CNN OOF pred + hand-crafted composite, evaluated
    # with the SAME group folds so the meta-model never sees a subject's own
    # trials during its own fit either.
    ens = merged.merge(
        pd.DataFrame({"_row": np.arange(len(groups)), "cnn_pred": cnn_oof}), on="_row")
    ens = ens.dropna(subset=["cnn_pred", "nurovo_pain_raw"])
    Xe = ens[["nurovo_pain_raw", "cnn_pred"]].to_numpy()
    ye = y[ens["_row"].to_numpy()]
    ge = ens["global_subject"].to_numpy()

    ens_oof = np.full(len(ye), np.nan)
    for tr, te_idx in gkf.split(Xe, ye, ge):
        scaler = StandardScaler().fit(Xe[tr])
        meta_model = Ridge(alpha=1.0).fit(scaler.transform(Xe[tr]), ye[tr])
        ens_oof[te_idx] = meta_model.predict(scaler.transform(Xe[te_idx]))
    ens_r, ens_p = pearsonr(ens_oof, ye)
    hand_r, _ = pearsonr(ens["nurovo_pain_raw"], ye)
    print(f"\n=== Stacked ensemble (hand-crafted composite + CNN, same folds) ===")
    print(f"hand-crafted alone (this subset): r={hand_r:.4f}")
    print(f"CNN alone (this subset): r={pearsonr(ens['cnn_pred'], ye)[0]:.4f}")
    print(f"ensemble: r={ens_r:.4f}, p={ens_p:.3g}  "
          f"({'ensemble beats both alone' if ens_r > max(hand_r, cnn_r) + 0.01 else 'no meaningful gain from combining'})")

    pd.DataFrame([
        {"model": "cnn_raw_eeg_28ch_full_pool", "r": cnn_r, "p": cnn_p, "n_trials": len(y)},
        {"model": "hand_crafted_ridge_stage15", "r": hand_r, "p": np.nan, "n_trials": len(ye)},
        {"model": "stacked_ensemble", "r": ens_r, "p": ens_p, "n_trials": len(ye)},
    ]).to_csv(f"{a.output_prefix}_summary.csv", index=False)
    print(f"\nsaved {a.output_prefix}_cnn_predictions.csv, {a.output_prefix}_summary.csv")


if __name__ == "__main__":
    main()
