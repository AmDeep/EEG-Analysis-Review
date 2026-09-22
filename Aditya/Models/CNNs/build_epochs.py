"""Build a fixed-size epoch tensor (trials x channels x time) for EEGNet.

Assumptions (edit COLS / CHANNELS below if they don't hold):
  * one epoched EEGLAB .set per subject under <root>/**/derivatives/rerefer/**, with the dataset id
    (dsNNNNNN) and sub-NNN somewhere in the path
  * a per-trial label CSV (e.g. your fixed extraction output) with dataset, subject, epoch, laser_power,
    where trials for a subject appear in the same order as the epochs in that subject's .set
"""
import argparse
import glob
import os
import re
from collections import Counter

import mne
import numpy as np
import pandas as pd

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
            "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
COLS = dict(dataset="dataset", subject="subject", epoch="epoch", power="laser_power")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--datasets", nargs="+", default=["ds005293", "ds005473"])
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=1.0)
    ap.add_argument("--out", default="nurovo_epochs.npz")
    a = ap.parse_args()

    lab = pd.read_csv(a.labels)
    lab["_ds"] = lab[COLS["dataset"]].astype(str).str.extract(r"(ds\d{6})")[0]
    lab["_sub"] = lab[COLS["subject"]].astype(str).str.extract(r"(\d+)")[0].astype(int)
    groups = {k: v for k, v in lab.groupby(["_ds", "_sub"])}

    n_t = int(round((a.tmax - a.tmin) * a.sfreq))
    pattern = os.path.join(a.root, "**", "derivatives", "rerefer", "**", "*.set")
    Xs, power, dsets, subjs, epochs = [], [], [], [], []
    skipped = Counter()

    for f in sorted(glob.glob(pattern, recursive=True)):
        m_ds, m_sub = re.search(r"(ds\d{6})", f), re.search(r"sub-(\d+)", f)
        if not (m_ds and m_sub) or m_ds.group(1) not in a.datasets:
            continue
        ds, sub = m_ds.group(1), int(m_sub.group(1))
        try:
            ep = mne.read_epochs_eeglab(f, verbose="ERROR")
        except Exception as e:  # e.g. file is continuous, not epoched
            skipped["read_failed"] += 1
            print(f"skip {f}: {e}")
            continue
        lut = {c.lower(): c for c in ep.ch_names}
        if any(c.lower() not in lut for c in CHANNELS):
            skipped["missing_channels"] += 1
            continue
        g = groups.get((ds, sub))
        if g is None or len(g) != len(ep):
            skipped["label_mismatch"] += 1
            print(f"skip {ds} sub-{sub}: {0 if g is None else len(g)} labels vs {len(ep)} epochs")
            continue
        if COLS["epoch"] in g:
            g = g.sort_values(COLS["epoch"])
        ep.rename_channels({lut[c.lower()]: c for c in CHANNELS})
        ep.pick(CHANNELS).reorder_channels(CHANNELS)
        ep.load_data()
        if ep.tmin < 0:
            ep.apply_baseline((None, 0))
        ep.resample(a.sfreq)
        ep.crop(a.tmin, a.tmax, include_tmax=False)
        X = ep.get_data() * 1e6  # volts -> microvolts
        if X.shape[-1] < n_t:
            skipped["too_short"] += 1
            continue
        Xs.append(X[..., :n_t].astype(np.float32))
        power.append(g[COLS["power"]].to_numpy(float))
        dsets += [ds] * len(g)
        subjs += [f"{ds}_sub-{sub:03d}"] * len(g)
        epochs += list(range(len(g)))

    X = np.concatenate(Xs)
    np.savez_compressed(a.out, X=X, power=np.concatenate(power), dataset=np.array(dsets),
                        global_subject=np.array(subjs), epoch=np.array(epochs),
                        ch_names=np.array(CHANNELS), sfreq=a.sfreq, tmin=a.tmin)
    print(f"saved {a.out}: X{X.shape}, {len(set(subjs))} subjects; skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()