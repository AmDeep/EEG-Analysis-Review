"""
MNE/MNE-BIDS preprocessing pipeline: raw EEG -> filtered, resampled, epoched trials +
per-trial stimulus-intensity labels, harmonized to a common channel set across datasets.

Output per dataset: data/processed/<dataset_id>/
    X.npy            float32, shape (n_trials, n_channels, n_samples)
    y_raw.npy         float32, raw label as given by the dataset (Joules, or 1/2/3 level)
    y_relative.npy    float32, label min-max normalized to [0, 1] within this dataset
    subjects.npy      int, subject index per trial (for LOSO splits)
    meta.csv          trial-level metadata (dataset id, subject, run, raw label, etc.)

This is written against the BIDS layout OpenNeuro publishes for the Zhao mega-study datasets and
against a best-guess layout for the Tiemann OSF release. Column/event names marked VERIFY in
configs/default.yaml should be checked against the first subject's events.tsv (the script prints
them) before trusting label extraction across the whole dataset.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def harmonize_channels(raw, common_channels: list[str]):
    """Rename dataset-native channel labels to standard 10-20 names, then pick the common subset."""
    import mne

    # Try MNE's built-in standard_1020 montage names as the canonical target; datasets that ship
    # a channels.tsv with non-standard labels (e.g. Biosemi 'A1', 'A2', ...) need a per-dataset
    # mapping -- see the dataset's channels.tsv for the vendor->10-20 lookup and fill in `mapping`.
    mapping = {}  # e.g. {"A1": "Fp1", "A2": "AF7", ...} -- VERIFY per dataset from channels.tsv
    if mapping:
        raw.rename_channels(mapping)

    available = [ch for ch in common_channels if ch in raw.ch_names]
    missing = [ch for ch in common_channels if ch not in raw.ch_names]
    if missing:
        print(f"  [warn] missing channels {missing} for {raw.info.get('subject_info')}; "
              f"using only {available}")
    raw.pick_channels(available, ordered=True)
    return raw, available


def preprocess_raw(raw, cfg):
    raw.load_data()
    raw.notch_filter(cfg["notch_hz"])
    raw.filter(l_freq=cfg["bandpass_hz"][0], h_freq=cfg["bandpass_hz"][1])
    raw.resample(cfg["sample_rate_hz"])
    raw.set_eeg_reference("average")
    return raw


def epoch_and_label(raw, events_df, cfg, label_column: str):
    import mne

    if label_column not in events_df.columns:
        print(f"  [warn] label column '{label_column}' not found; available columns: "
              f"{list(events_df.columns)}")
        return None, None

    sfreq = raw.info["sfreq"]
    tmin, tmax = cfg["epoch_window_s"]
    bmin, bmax = cfg["baseline_window_s"]

    onsets_samples = (events_df["onset"].values * sfreq).astype(int)
    events_arr = np.column_stack([onsets_samples, np.zeros_like(onsets_samples), np.arange(len(onsets_samples))])

    epochs = mne.Epochs(
        raw, events_arr, event_id=None, tmin=tmin, tmax=tmax,
        baseline=(bmin, bmax), reject=dict(eeg=cfg["reject_uv"] * 1e-6),
        preload=True, verbose=False,
    )
    kept_idx = epochs.selection  # indices into events_arr that survived rejection
    labels = events_df[label_column].values[kept_idx]
    return epochs.get_data(), labels


def process_dataset(dataset_id: str, ds_cfg: dict, cfg: dict, bids_root: Path, out_root: Path):
    import mne
    from mne_bids import BIDSPath, read_raw_bids, find_matching_paths

    out_dir = out_root / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_matching_paths(bids_root, datatypes="eeg", extensions=[".bdf", ".vhdr", ".edf", ".set"])
    if not paths:
        print(f"No BIDS EEG files found under {bids_root} for {dataset_id} -- check the download path.")
        return

    all_X, all_y_raw, all_subj = [], [], []
    meta_rows = []

    for i, bp in enumerate(paths):
        try:
            raw = read_raw_bids(bp, verbose=False)
        except Exception as e:
            print(f"  [skip] {bp}: {e}")
            continue

        raw, kept_channels = harmonize_channels(raw, cfg["common_channels"])
        if len(kept_channels) < 3:
            print(f"  [skip] {bp}: too few common channels found ({kept_channels})")
            continue
        raw = preprocess_raw(raw, cfg)

        events_path = bp.copy().update(suffix="events", extension=".tsv")
        if not Path(events_path.fpath).exists():
            print(f"  [skip] no events.tsv for {bp}")
            continue
        events_df = pd.read_csv(events_path.fpath, sep="\t")

        if i == 0:
            print(f"  [{dataset_id}] events.tsv columns for first subject: {list(events_df.columns)}")

        X, y = epoch_and_label(raw, events_df, cfg, ds_cfg["label_column"])
        if X is None:
            continue

        all_X.append(X.astype("float32"))
        all_y_raw.append(np.asarray(y, dtype="float32"))
        subj_id = bp.subject
        all_subj.append(np.full(len(y), int(subj_id) if subj_id.isdigit() else i, dtype="int32"))
        meta_rows.append(pd.DataFrame({
            "dataset": dataset_id, "subject": subj_id, "trial": np.arange(len(y)), "label_raw": y,
        }))

    if not all_X:
        print(f"No usable trials extracted for {dataset_id}.")
        return

    X = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y_raw, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    y_relative = (y_raw - y_raw.min()) / max(1e-8, (y_raw.max() - y_raw.min()))

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y_raw.npy", y_raw)
    np.save(out_dir / "y_relative.npy", y_relative)
    np.save(out_dir / "subjects.npy", subjects)
    pd.concat(meta_rows, ignore_index=True).to_csv(out_dir / "meta.csv", index=False)
    print(f"[{dataset_id}] wrote {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-root", default="data/raw/openneuro")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of dataset ids to process")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds_ids = args.datasets or [d for d in cfg["datasets"] if cfg["datasets"][d]["source"] == "openneuro"]

    out_root = Path(args.out)
    for ds_id in ds_ids:
        ds_cfg = cfg["datasets"][ds_id]
        bids_root = Path(args.raw_root) / ds_id
        if not bids_root.exists():
            print(f"[skip] {ds_id}: {bids_root} not found -- run data/download_openneuro.py first")
            continue
        process_dataset(ds_id, ds_cfg, cfg, bids_root, out_root)


if __name__ == "__main__":
    main()
