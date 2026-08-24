"""
Fast smoke test for CI: generates synthetic epoched EEG data (no network,
no real dataset needed), runs it through extract_features.process_file(),
and checks that every expected feature column is present and the row count
matches the number of epochs.

This is intentionally NOT a correctness test of the neuroscience (e.g. it
doesn't assert N2 amplitude is in some physiologically plausible range,
since synthetic noise has no real ERP) -- it exists to catch pipeline
breakage: import errors, shape mismatches, a loader silently returning
zero rows, an exception in one feature function taking down the whole run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_features as ef  # noqa: E402

mne = pytest.importorskip("mne")

EXPECTED_COLUMNS = {
    "dataset", "subject", "epoch", "source_file", "vertex_channel", "sfreq",
    "pseudo_epoched", "n2_amp", "n2_lat", "p2_amp", "p2_lat", "n2p2_amp",
    "gamma_power", "gamma_band_hz", "alpha_erd_pct", "beta_erd_pct",
    "psd_delta", "psd_theta", "psd_alpha", "psd_beta", "psd_gamma",
    "plv_Fz-Cz", "plv_Cz-Pz",
}


@pytest.fixture(scope="module")
def synthetic_fif(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("synthetic")
    sfreq = 500.0
    ch_names = ["Fz", "Cz", "CPz", "Pz", "C3", "C4"]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    n_epochs = 6
    tmin = -0.3
    n_times = int((0.6 - tmin) * sfreq) + 1
    rng = np.random.default_rng(0)
    data = rng.normal(0, 5e-6, size=(n_epochs, len(ch_names), n_times))

    events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, verbose=False)
    path = tmp_dir / "sub-99_task-synthetic-epo.fif"
    epochs.save(str(path), overwrite=True)
    return path, n_epochs


def test_process_file_produces_expected_rows_and_columns(synthetic_fif):
    path, n_epochs = synthetic_fif
    rows = ef.process_file(path, print_epoch_info=False)
    assert len(rows) == n_epochs, "one feature row expected per epoch"

    df = pd.DataFrame(rows)
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"missing expected feature columns: {missing}"
    assert df["pseudo_epoched"].eq(False).all(), "true .fif epochs should not be marked pseudo-epoched"
    assert df["vertex_channel"].eq("Cz").all()


def test_discover_input_files_finds_supported_extensions(tmp_path):
    (tmp_path / "a.set").touch()
    (tmp_path / "b.fif").touch()
    (tmp_path / "c.txt").touch()
    found = ef.discover_input_files(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.set", "b.fif"]


def test_discover_input_files_rejects_unsupported_single_file(tmp_path):
    bad = tmp_path / "not_eeg.txt"
    bad.touch()
    with pytest.raises(ValueError):
        ef.discover_input_files(bad)
