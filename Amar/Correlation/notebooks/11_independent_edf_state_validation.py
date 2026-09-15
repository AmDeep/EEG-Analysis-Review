"""Independent EDF validation for portable multi-dataset EEG states.

The script uses task annotations only after unsupervised discovery and
leave-one-dataset-out (LODO) assignment. Run from the notebooks directory:

    python 11_independent_edf_state_validation.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import chi2_contingency
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import RobustScaler

mne.set_log_level("ERROR")

DATA = Path("data")
OUT = DATA / "preprocessed"
ZHAO_ROOT = DATA / "ds005284"
EDF_ROOT = DATA / "eegmmidb"
WINDOW_SECONDS = (2, 4, 8)
TARGET_SFREQ = 160.0
RANDOM_STATE = 284
MAX_WINDOWS_PER_RECORDING = 48
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}
CLUSTER_FEATURES = [
    *(f"rel_{band}_{summary}" for band in BANDS for summary in ("median", "iqr")),
    "spectral_entropy_median",
    "spectral_entropy_iqr",
    "hjorth_mobility_median",
    "hjorth_mobility_iqr",
    "hjorth_complexity_median",
    "hjorth_complexity_iqr",
    "log_rms_median",
    "log_rms_iqr",
    "log_ptp_median",
    "log_ptp_iqr",
    "gfp_median",
    "gfp_iqr",
]
QC_COLUMNS = [
    "flat_channel_fraction",
    "extreme_amplitude_fraction",
    "line_noise_ratio",
    "missing_channel_fraction",
    "retained_duration_fraction",
]


def iqr(values: np.ndarray, axis=None) -> np.ndarray:
    return np.subtract(*np.percentile(values, [75, 25], axis=axis))


def normalize_channel_name(name: str) -> str:
    return name.strip().strip(".").upper()


def prepare_raw(path: Path, dataset: str) -> mne.io.BaseRaw:
    if dataset == "ds005284":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        stem = path.name.replace("_eeg.bdf", "")
        channels = pd.read_csv(path.with_name(stem + "_channels.tsv"), sep="\t")
        mapped = channels["name"].astype(str).tolist()
        raw.rename_channels(dict(zip(raw.ch_names[: len(mapped)], mapped)))
        task_label = "pain_protocol"
    else:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        task_label = edf_run_condition(path)
    raw.rename_channels({name: normalize_channel_name(name) for name in raw.ch_names})
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(eeg_picks)
    raw.filter(1.0, 40.0, method="iir", verbose="ERROR")
    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.info["description"] = task_label
    return raw


def edf_run_condition(path: Path) -> str:
    run = int(re.search(r"R(\d+)", path.stem).group(1))
    return {
        1: "eyes_open_rest",
        2: "eyes_closed_rest",
        3: "motor_execution",
        4: "motor_imagery",
        7: "motor_execution",
        8: "motor_imagery",
        11: "motor_execution",
        12: "motor_imagery",
    }[run]


def annotation_at(raw: mne.io.BaseRaw, midpoint: float) -> str:
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if onset <= midpoint < onset + max(duration, 1e-9):
            return str(description)
    return "unannotated"


def extract_window_features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    eps = np.finfo(float).eps
    freqs, psd = welch(data, fs=sfreq, nperseg=min(data.shape[1], int(2 * sfreq)), axis=1)
    analysis = (freqs >= 1.0) & (freqs <= 40.0)
    total = np.trapezoid(psd[:, analysis], freqs[analysis], axis=1) + eps
    features: dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        rel = np.trapezoid(psd[:, mask], freqs[mask], axis=1) / total
        features[f"rel_{band}_median"] = float(np.median(rel))
        features[f"rel_{band}_iqr"] = float(iqr(rel))

    probability = psd[:, analysis] / (psd[:, analysis].sum(axis=1, keepdims=True) + eps)
    entropy = -(probability * np.log(probability + eps)).sum(axis=1) / np.log(probability.shape[1])
    first = np.diff(data, axis=1)
    second = np.diff(first, axis=1)
    var0, var1, var2 = np.var(data, axis=1), np.var(first, axis=1), np.var(second, axis=1)
    mobility = np.sqrt(var1 / (var0 + eps))
    complexity = np.sqrt(var2 / (var1 + eps)) / (mobility + eps)
    rms = np.sqrt(np.mean(data**2, axis=1))
    ptp = np.ptp(data, axis=1)
    gfp = np.std(data, axis=0)
    for name, values in (
        ("spectral_entropy", entropy),
        ("hjorth_mobility", mobility),
        ("hjorth_complexity", complexity),
        ("log_rms", np.log10(rms + eps)),
        ("log_ptp", np.log10(ptp + eps)),
    ):
        features[f"{name}_median"] = float(np.median(values))
        features[f"{name}_iqr"] = float(iqr(values))
    features["gfp_median"] = float(np.median(gfp))
    features["gfp_iqr"] = float(iqr(gfp))

    channel_std = np.std(data, axis=1)
    features["flat_channel_fraction"] = float(np.mean(channel_std < 1e-8))
    features["extreme_amplitude_fraction"] = float(np.mean(np.max(np.abs(data), axis=1) > 500e-6))
    line_mask = (freqs >= 38.0) & (freqs <= 40.0)
    features["line_noise_ratio"] = float(
        np.median(np.trapezoid(psd[:, line_mask], freqs[line_mask], axis=1) / total)
    )
    return features


def recording_inventory(paths: list[Path], dataset: str) -> list[dict]:
    rows = []
    for path in paths:
        raw = (
            mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
            if dataset == "ds005284"
            else mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        )
        rows.append(
            {
                "dataset_id": dataset,
                "subject": path.parts[-3] if dataset == "ds005284" else path.parent.name,
                "raw_file": str(path),
                "format": path.suffix[1:].upper(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "file_bytes": path.stat().st_size,
                "sampling_rate_hz": raw.info["sfreq"],
                "duration_seconds": raw.n_times / raw.info["sfreq"],
                "raw_eeg_channel_count": len(mne.pick_types(raw.info, eeg=True)),
                "annotation_count": len(raw.annotations),
                "device": "BioSemi ActiveTwo" if dataset == "ds005284" else "BCI2000 64-channel",
            }
        )
        raw.close()
    return rows


def extract_dataset_all_windows(paths: list[Path], dataset: str) -> dict[int, pd.DataFrame]:
    rows = {window_seconds: [] for window_seconds in WINDOW_SECONDS}
    for path in paths:
        subject = path.parts[-3] if dataset == "ds005284" else path.parent.name
        raw = prepare_raw(path, dataset)
        duration = raw.n_times / raw.info["sfreq"]
        for window_seconds in WINDOW_SECONDS:
            starts = np.arange(0.0, duration - window_seconds + 1e-6, window_seconds)
            if len(starts) > MAX_WINDOWS_PER_RECORDING:
                starts = starts[
                    np.linspace(0, len(starts) - 1, MAX_WINDOWS_PER_RECORDING).astype(int)
                ]
            for index, start in enumerate(starts):
                stop = start + window_seconds
                data = raw.get_data(start=int(start * TARGET_SFREQ), stop=int(stop * TARGET_SFREQ))
                features = extract_window_features(data, TARGET_SFREQ)
                features.update(
                    {
                        "dataset_id": dataset,
                        "subject": subject,
                        "device": (
                            "BioSemi ActiveTwo"
                            if dataset == "ds005284"
                            else "BCI2000 64-channel"
                        ),
                        "raw_file": str(path),
                        "window_seconds": window_seconds,
                        "window_index": index,
                        "start_seconds": start,
                        "task_condition": raw.info["description"],
                        "task_annotation": annotation_at(raw, start + window_seconds / 2),
                        "missing_channel_fraction": 0.0,
                        "retained_duration_fraction": 1.0,
                    }
                )
                rows[window_seconds].append(features)
        raw.close()
    return {
        window_seconds: pd.DataFrame(values)
        for window_seconds, values in rows.items()
    }


def balanced_fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    minimum = frame.groupby(["dataset_id", "subject"]).size().min()
    return (
        frame.groupby(["dataset_id", "subject"], group_keys=False)
        .sample(n=int(minimum), random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left, right)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt((chi2 / n) / max(1, min(table.shape) - 1)))


def confound_audit(frame: pd.DataFrame, labels: np.ndarray) -> dict[str, float | bool]:
    audit = frame.copy()
    audit["state"] = labels
    state_subject_share = (
        audit.groupby("state").subject.value_counts(normalize=True).groupby(level=0).max().max()
    )
    state_dataset_share = (
        audit.groupby("state").dataset_id.value_counts(normalize=True).groupby(level=0).max().max()
    )
    qc_nmi = max(
        normalized_mutual_info_score(
            pd.qcut(audit[column].rank(method="first"), q=4, labels=False, duplicates="drop"),
            labels,
        )
        for column in QC_COLUMNS
    )
    return {
        "max_subject_share": float(state_subject_share),
        "max_dataset_share": float(state_dataset_share),
        "dataset_cramers_v": cramers_v(audit.dataset_id, audit.state),
        "device_cramers_v": cramers_v(audit.device, audit.state),
        "max_qc_nmi": float(qc_nmi),
        "reject_subject_dominated": bool(state_subject_share > 0.30),
        "reject_dataset_or_device_dominated": bool(
            state_dataset_share > 0.80 or cramers_v(audit.dataset_id, audit.state) > 0.50
        ),
        "reject_qc_dominated": bool(qc_nmi > 0.20),
    }


def grouped_gmm_selection(train: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Select state count using training subjects and temporal blocks only."""
    rows = []
    subjects = sorted(train.subject.unique())
    rng = np.random.default_rng(RANDOM_STATE)
    for state_count in (2, 3, 4):
        heldout_log_likelihood, heldout_coverage = [], []
        for heldout_subject in subjects:
            fit = train[train.subject != heldout_subject]
            validation = train[train.subject == heldout_subject]
            scaler = RobustScaler().fit(fit[CLUSTER_FEATURES])
            model = GaussianMixture(
                n_components=state_count,
                covariance_type="diag",
                n_init=3,
                random_state=RANDOM_STATE,
            ).fit(scaler.transform(fit[CLUSTER_FEATURES]))
            validation_x = scaler.transform(validation[CLUSTER_FEATURES])
            heldout_log_likelihood.append(model.score(validation_x))
            heldout_coverage.append(float((model.predict_proba(validation_x).max(axis=1) >= 0.60).mean()))

        reference_scaler = RobustScaler().fit(train[CLUSTER_FEATURES])
        reference_x = reference_scaler.transform(train[CLUSTER_FEATURES])
        reference = GaussianMixture(
            n_components=state_count,
            covariance_type="diag",
            n_init=5,
            random_state=RANDOM_STATE,
        ).fit(reference_x)
        reference_labels = reference.predict(reference_x)
        bootstrap_ari = []
        train_with_blocks = train.copy()
        train_with_blocks["temporal_block"] = np.floor(
            train_with_blocks.start_seconds / 24.0
        ).astype(int)
        for _ in range(12):
            sampled_subjects = rng.choice(subjects, len(subjects), replace=True)
            pieces = []
            for subject in sampled_subjects:
                subject_rows = train_with_blocks[train_with_blocks.subject == subject]
                blocks = subject_rows[["raw_file", "temporal_block"]].drop_duplicates()
                chosen = blocks.sample(
                    n=max(1, int(np.ceil(len(blocks) * 0.70))),
                    replace=True,
                    random_state=int(rng.integers(1_000_000)),
                )
                for raw_file, block in chosen.itertuples(index=False):
                    pieces.append(
                        subject_rows[
                            (subject_rows.raw_file == raw_file)
                            & (subject_rows.temporal_block == block)
                        ]
                    )
            sample = pd.concat(pieces, ignore_index=True)
            scaler = RobustScaler().fit(sample[CLUSTER_FEATURES])
            model = GaussianMixture(
                n_components=state_count,
                covariance_type="diag",
                n_init=2,
                random_state=int(rng.integers(1_000_000)),
            ).fit(scaler.transform(sample[CLUSTER_FEATURES]))
            bootstrap_ari.append(
                adjusted_rand_score(
                    reference_labels,
                    model.predict(scaler.transform(train[CLUSTER_FEATURES])),
                )
            )

        kmeans_labels = KMeans(
            n_clusters=state_count, n_init=30, random_state=RANDOM_STATE
        ).fit_predict(reference_x)
        hdbscan_labels = HDBSCAN(
            min_cluster_size=max(20, len(train) // 30), copy=True
        ).fit_predict(reference_x)
        valid_hdbscan = hdbscan_labels >= 0
        rows.append(
            {
                "state_count": state_count,
                "subject_cv_log_likelihood": float(np.mean(heldout_log_likelihood)),
                "subject_cv_coverage": float(np.mean(heldout_coverage)),
                "subject_temporal_block_bootstrap_ari": float(np.mean(bootstrap_ari)),
                "kmeans_silhouette": float(
                    silhouette_score(
                        reference_x,
                        kmeans_labels,
                        sample_size=min(2000, len(train)),
                        random_state=RANDOM_STATE,
                    )
                ),
                "hdbscan_cluster_count": int(len(set(hdbscan_labels[valid_hdbscan]))),
                "hdbscan_noise_fraction": float((~valid_hdbscan).mean()),
                "hmm_evaluated": False,
                "hmm_reason": "hmmlearn unavailable; temporal metrics reported without fitting an HMM",
            }
        )
    scores = pd.DataFrame(rows)
    eligible = scores[
        (scores.subject_temporal_block_bootstrap_ari >= 0.60)
        & (scores.subject_cv_coverage >= 0.50)
    ]
    ranked = eligible if len(eligible) else scores
    chosen = int(
        ranked.sort_values(
            ["subject_cv_log_likelihood", "subject_temporal_block_bootstrap_ari"],
            ascending=False,
        ).iloc[0].state_count
    )
    scores["selected"] = scores.state_count == chosen
    return chosen, scores


def nested_lodo_assignment(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments, summaries, model_rows, audit_rows = [], [], [], []
    for train_dataset, test_dataset in (("ds005284", "eegmmidb"), ("eegmmidb", "ds005284")):
        train = balanced_fit_rows(
            frame[frame.dataset_id == train_dataset].reset_index(drop=True)
        )
        test = frame[frame.dataset_id == test_dataset].reset_index(drop=True)
        state_count, selection = grouped_gmm_selection(train)
        selection.insert(0, "trained_on_dataset", train_dataset)
        selection.insert(1, "held_out_dataset", test_dataset)
        model_rows.append(selection)

        scaler = RobustScaler().fit(train[CLUSTER_FEATURES])
        train_x = scaler.transform(train[CLUSTER_FEATURES])
        test_x = scaler.transform(test[CLUSTER_FEATURES])
        model = GaussianMixture(
            n_components=state_count,
            covariance_type="diag",
            n_init=20,
            random_state=RANDOM_STATE,
        ).fit(train_x)
        train_likelihood = model.score_samples(train_x)
        posterior = model.predict_proba(test_x)
        predicted = posterior.argmax(axis=1)
        confidence = posterior.max(axis=1)
        likelihood = model.score_samples(test_x)
        unknown = (confidence < 0.60) | (likelihood < np.quantile(train_likelihood, 0.05))

        out = test[
            [
                "dataset_id", "subject", "raw_file", "window_seconds", "window_index",
                "start_seconds", "task_condition", "task_annotation",
            ]
        ].copy()
        out["trained_on_dataset"] = train_dataset
        out["assigned_state"] = predicted
        out["posterior_confidence"] = confidence
        out["log_likelihood"] = likelihood
        out["low_confidence_unknown"] = unknown
        assignments.append(out)

        combined = balanced_fit_rows(pd.concat([train, test], ignore_index=True))
        combined_x = scaler.transform(combined[CLUSTER_FEATURES])
        combined_labels = model.predict(combined_x)
        audit = confound_audit(combined, combined_labels)
        audit.update(
            {
                "trained_on_dataset": train_dataset,
                "held_out_dataset": test_dataset,
                "state_count": state_count,
            }
        )
        audit["accepted"] = not any(
            audit[key]
            for key in (
                "reject_subject_dominated",
                "reject_dataset_or_device_dominated",
                "reject_qc_dominated",
            )
        )
        audit_rows.append(audit)

        counts = pd.Series(predicted[~unknown]).value_counts(normalize=True)
        summaries.append(
            {
                "trained_on_dataset": train_dataset,
                "held_out_dataset": test_dataset,
                "state_count": state_count,
                "coverage": float(1 - unknown.mean()),
                "assignment_entropy": (
                    float(-(counts * np.log(counts + np.finfo(float).eps)).sum() / np.log(state_count))
                    if len(counts) and state_count > 1
                    else np.nan
                ),
                "mean_heldout_log_likelihood": float(np.mean(likelihood)),
                "mean_posterior_confidence": float(np.mean(confidence)),
            }
        )
    return (
        pd.concat(assignments, ignore_index=True),
        pd.DataFrame(summaries),
        pd.concat(model_rows, ignore_index=True),
        pd.DataFrame(audit_rows),
    )


def windows_are_contiguous(
    previous_start: float | None, current_start: float, window_seconds: int
) -> bool:
    return (
        previous_start is not None
        and np.isclose(current_start - previous_start, window_seconds, atol=1e-6)
    )


def temporal_metrics(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in assignments.groupby(
        ["window_seconds", "trained_on_dataset", "dataset_id", "subject", "raw_file"]
    ):
        group = group.sort_values("start_seconds")
        sequence = [
            None if unknown else int(state)
            for state, unknown in zip(group.assigned_state, group.low_confidence_unknown)
        ]
        known = [state for state in sequence if state is not None]
        state_counts = Counter(known)
        runs = []
        current, length = None, 0
        transitions = Counter()
        previous = None
        previous_start = None
        gap_breaks = 0
        for state, start_seconds in zip(sequence, group.start_seconds):
            if previous_start is not None and not windows_are_contiguous(
                previous_start, start_seconds, keys[0]
            ):
                if current is not None:
                    runs.append((current, length))
                current, length, previous = None, 0, None
                gap_breaks += 1
            if state is None:
                if current is not None:
                    runs.append((current, length))
                current, length, previous = None, 0, None
                previous_start = start_seconds
                continue
            if current == state:
                length += 1
            else:
                if current is not None:
                    runs.append((current, length))
                current, length = state, 1
            if previous is not None and previous != state:
                transitions[(previous, state)] += 1
            previous = state
            previous_start = start_seconds
        if current is not None:
            runs.append((current, length))
        for state in sorted(set(known)):
            dwell = [length * keys[0] for run_state, length in runs if run_state == state]
            rows.append(
                {
                    "window_seconds": keys[0],
                    "trained_on_dataset": keys[1],
                    "dataset_id": keys[2],
                    "subject": keys[3],
                    "raw_file": keys[4],
                    "state": state,
                    "occupancy": state_counts[state] / len(sequence),
                    "median_dwell_seconds": float(np.median(dwell)) if dwell else 0.0,
                    "transition_count_from_state": sum(
                        count for (source, _), count in transitions.items() if source == state
                    ),
                    "unknown_fraction": 1 - len(known) / len(sequence),
                    "sampling_gap_break_count": gap_breaks,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zhao_inventory_paths = sorted(ZHAO_ROOT.glob("sub-*/eeg/*_eeg.bdf"))
    edf_paths = sorted(EDF_ROOT.glob("S*/*.edf"))
    if len(zhao_inventory_paths) != 26 or len(edf_paths) != 80:
        raise RuntimeError(
            "Expected 26 BDF and 80 EDF files; found "
            f"{len(zhao_inventory_paths)} and {len(edf_paths)}"
        )
    zhao_paths = zhao_inventory_paths[:10]

    inventory = pd.DataFrame(
        recording_inventory(zhao_inventory_paths, "ds005284")
        + recording_inventory(edf_paths, "eegmmidb")
    )
    inventory.to_csv(OUT / "multidataset_raw_eeg_inventory.csv", index=False)

    acquisition = {
        "dataset_id": "eegmmidb",
        "version": "1.0.0",
        "checked_on": "2026-09-15",
        "official_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "access_policy": "Anyone can access files subject to the stated license.",
        "license": "Open Data Commons Attribution License v1.0",
        "doi": "10.13026/C28G6P",
        "full_uncompressed_size_gb": 3.4,
        "full_zip_size_gb": 1.9,
        "local_subset": "S001-S010; runs 01,02,03,04,07,08,11,12",
        "local_file_count": len(edf_paths),
        "local_bytes": int(sum(path.stat().st_size for path in edf_paths)),
        "selection_rationale": "First 10 subjects, with two baseline, three execution, and three imagery runs.",
        "comparison_subset": "ds005284 sub-001 through sub-010 for balanced subject counts",
    }
    (OUT / "eegmmidb_acquisition_audit.json").write_text(json.dumps(acquisition, indent=2) + "\n")

    registry_path = OUT / "multidataset_eeg_registry.csv"
    registry = pd.read_csv(registry_path)
    mask = registry.dataset_id == "eegmmidb"
    registry.loc[mask, "local_status"] = "available_subset_audited"
    registry.loc[mask, "local_root"] = str(EDF_ROOT)
    registry.to_csv(registry_path, index=False)

    cached = all(
        (OUT / f"multidataset_tier1_features_{window_seconds}s.csv").exists()
        for window_seconds in WINDOW_SECONDS
    )
    if not cached:
        extracted_zhao = extract_dataset_all_windows(zhao_paths, "ds005284")
        extracted_edf = extract_dataset_all_windows(edf_paths, "eegmmidb")
        for window_seconds in WINDOW_SECONDS:
            pd.concat(
                [extracted_zhao[window_seconds], extracted_edf[window_seconds]],
                ignore_index=True,
            ).to_csv(
                OUT / f"multidataset_tier1_features_{window_seconds}s.csv",
                index=False,
            )

    all_model_summaries, all_confound, all_lodo, all_assignments = [], [], [], []
    for window_seconds in WINDOW_SECONDS:
        cache = OUT / f"multidataset_tier1_features_{window_seconds}s.csv"
        features = pd.read_csv(cache)
        assigned, lodo, model_selection, audits = nested_lodo_assignment(features)
        assigned.to_csv(OUT / f"multidataset_lodo_assignments_{window_seconds}s.csv", index=False)
        lodo.insert(0, "window_seconds", window_seconds)
        model_selection.insert(0, "window_seconds", window_seconds)
        audits.insert(0, "window_seconds", window_seconds)
        all_assignments.append(assigned)
        all_lodo.append(lodo)
        all_model_summaries.append(model_selection)
        all_confound.append(audits)

    pd.concat(all_model_summaries, ignore_index=True).to_csv(
        OUT / "multidataset_state_model_selection.csv", index=False
    )
    confounds = pd.concat(all_confound, ignore_index=True)
    confounds.to_csv(OUT / "multidataset_state_confound_audit.csv", index=False)
    lodo_summary = pd.concat(all_lodo, ignore_index=True)
    lodo_summary.to_csv(OUT / "multidataset_lodo_summary.csv", index=False)
    assignment_frame = pd.concat(all_assignments, ignore_index=True)
    temporal_metrics(assignment_frame).to_csv(
        OUT / "multidataset_lodo_temporal_metrics.csv", index=False
    )

    interpretation = assignment_frame[
        (assignment_frame.dataset_id == "eegmmidb")
        & (assignment_frame.trained_on_dataset == "ds005284")
        & (~assignment_frame.low_confidence_unknown)
    ]
    interpretation = (
        interpretation.groupby(
            [
                "window_seconds", "trained_on_dataset", "assigned_state",
                "task_condition", "task_annotation",
            ]
        )
        .size()
        .rename("window_count")
        .reset_index()
    )
    interpretation["annotation_used_during_fit"] = False
    interpretation.to_csv(OUT / "eegmmidb_posthoc_state_interpretation.csv", index=False)

    confound_gate_passed = [
        int(window_seconds)
        for window_seconds, group in confounds.groupby("window_seconds")
        if group.accepted.all()
    ]
    lodo_gate_passed = [
        int(window_seconds)
        for window_seconds, group in lodo_summary.groupby("window_seconds")
        if group.coverage.min() >= 0.50
    ]
    reproducible = sorted(set(confound_gate_passed) & set(lodo_gate_passed))
    confirmed = bool(reproducible)
    conclusion = {
        "confound_gate_passed_window_lengths": confound_gate_passed,
        "lodo_coverage_gate_passed_window_lengths": lodo_gate_passed,
        "reproducible_window_lengths": reproducible,
        "rejected_window_lengths": sorted(set(WINDOW_SECONDS) - set(reproducible)),
        "confirmed_on_one_independent_edf_dataset": confirmed,
        "general_cross_dataset_state_claim_supported": False,
        "result": (
            "At least one portable state solution was confirmed on the independent "
            "EDF dataset." if confirmed else
            "No portable state solution passed both bidirectional LODO coverage and "
            "nuisance-confound gates on the independent EDF dataset."
        ),
        "generalization_note": (
            "Confirmation in this two-dataset task is distinct from a general EEG "
            "state claim, which remains gated on recurrence in a third dataset."
        ),
        "decision_rule": {
            "subject_dominated": "maximum subject share within any state > 0.30",
            "dataset_or_device_dominated": "maximum dataset share > 0.80 or Cramer's V > 0.50",
            "qc_dominated": "maximum normalized mutual information with a QC variable > 0.20",
            "lodo_transportable": "coverage >= 0.50 in both assignment directions",
            "independent_edf_confirmation": "confound and LODO gates pass in both directions",
            "general_cross_dataset_claim": "later recurrence in at least three independent datasets",
        },
        "interpretation_policy": "Task/run annotations were excluded from all feature, scaling, clustering, model-selection, and LODO fitting steps and joined only afterward.",
    }
    (OUT / "multidataset_state_validation_conclusion.json").write_text(
        json.dumps(conclusion, indent=2) + "\n"
    )
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()