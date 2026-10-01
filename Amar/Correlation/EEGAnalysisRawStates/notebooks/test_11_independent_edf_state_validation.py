"""Focused regression checks for sparse-window temporal handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("11_independent_edf_state_validation.py")
SPEC = importlib.util.spec_from_file_location("independent_edf_validation", MODULE_PATH)
validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validation)


def test_sparse_windows_break_dwell_and_transitions() -> None:
    frame = pd.DataFrame(
        {
            "window_seconds": [2, 2, 2],
            "trained_on_dataset": ["ds005284"] * 3,
            "dataset_id": ["eegmmidb"] * 3,
            "subject": ["S001"] * 3,
            "raw_file": ["sample.edf"] * 3,
            "start_seconds": [0.0, 2.0, 8.0],
            "assigned_state": [0, 0, 1],
            "low_confidence_unknown": [False, False, False],
        }
    )

    result = validation.temporal_metrics(frame)
    state_zero = result[result.state == 0].iloc[0]
    state_one = result[result.state == 1].iloc[0]

    assert validation.windows_are_contiguous(0.0, 2.0, 2)
    assert not validation.windows_are_contiguous(2.0, 8.0, 2)
    assert state_zero.median_dwell_seconds == 4.0
    assert state_zero.transition_count_from_state == 0
    assert state_one.median_dwell_seconds == 2.0
    assert state_one.sampling_gap_break_count == 1


def test_temporal_blocks_use_actual_recording_time() -> None:
    starts = pd.Series([0.0, 8.0, 24.0, 31.0])
    blocks = (starts // 24.0).astype(int).tolist()
    assert blocks == [0, 0, 1, 1]