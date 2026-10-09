"""Explicit target mappings from author documentation, not EEG/label correlations."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfilt

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
CHANNELS = ["F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz"]


def channels(labels):
    lookup = {str(v).strip().upper(): i for i, v in enumerate(labels)}
    missing = [c for c in CHANNELS if c.upper() not in lookup]
    if missing:
        raise ValueError(f"Missing prespecified channels: {missing}")
    return [lookup[c.upper()] for c in CHANNELS]


def prepare(x, time, fs, onset=0.0):
    """Causal filter: our prestimulus features cannot use future samples.

    Deposited shock EEG has prior author zero-phase processing, which cannot
    be undone. Thus its prestimulus control is not a clean causal negative control.
    """
    x = np.asarray(x, float)
    time = np.asarray(time, float) - onset
    if not np.isfinite(x).all():
        return None, "nonfinite"
    x -= x.mean(axis=0, keepdims=True)
    # Same masking for delivered and omitted stimuli and both modalities.
    # Suppress immediate trigger/current artifacts; do not choose by target label.
    mask = (time >= -0.010) & (time <= 0.030)
    if mask.any():
        for row in x:
            row[mask] = np.interp(time[mask], time[~mask], row[~mask])
    x = sosfilt(butter(4, [1, 40], btype="bandpass", fs=fs, output="sos"), x, axis=-1)
    if fs != 500:
        raise ValueError(f"Unreviewed sampling rate {fs}; no silent resampling assumptions")
    x = resample_poly(x, 1, 2, axis=-1)
    time = time[::2][:x.shape[-1]]
    pre_mask = (time >= -0.8) & (time < -0.05)
    post_mask = (time >= 0.05) & (time < 1.25)
    if pre_mask.sum() < 185 or post_mask.sum() < 298:
        return None, "incomplete_window"
    pre, post = x[:, pre_mask], x[:, post_mask]
    if np.max(np.abs(post)) > 400 or np.max(np.abs(pre)) > 200:
        return None, "amplitude_threshold"
    if np.median(np.std(post, axis=1)) < 0.05:
        return None, "flat_signal"
    # Epoch-only centering uses no target labels, fitted cohort statistics or metadata.
    baseline = pre.mean(axis=1, keepdims=True)
    return (pre - baseline, post - baseline), None


def heat(path):
    d = loadmat(path, simplify_cells=True)["data"]
    info = np.asarray(d["trialinfo"])
    if info.ndim != 2 or info.shape[1] != 6:
        raise ValueError("Expected deposited six-column trialinfo; wiki's old four-column description is not used")
    idx = channels(d["label"])
    fs = float(d["fsample"])
    for i, code in enumerate(info[:, 1].astype(int)):
        # Author OSF wiki: 1-9 = pain cue / heat, 19-27 = picture cue / heat.
        # Within each group three cue intensities cycle inside each physical heat level.
        # DO NOT use column 4 (codes 37-42): it describes cues, not actual stimulus.
        if not (1 <= code <= 9 or 19 <= code <= 27):
            continue
        local = code if code <= 9 else code - 18
        temperature = [42.0, 46.0, 48.0][(local - 1) // 3]
        windows, error = prepare(np.asarray(d["trial"][i])[idx].copy(),
                                 d["time"][i], fs)
        yield {"source_trial": i, "temperature_c": temperature,
               "binary_target": int(temperature >= 46), "target_native": temperature,
               "target_units": "degrees Celsius", "condition_code": code,
               "block_context": None, "windows": windows, "error": error}


def as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(np.asarray(value, dtype=object).reshape(-1))
    return [value]


def shock(path):
    d = loadmat(path, simplify_cells=True)
    idx = channels([c["labels"] for c in d["chanlocs"]])
    fs = float(d["srate"])
    time = np.asarray(d["times"], float) / 1000.0
    epochs = as_list(d["epoch"])
    for i, epoch in enumerate(epochs):
        types = [str(v) for v in as_list(epoch["eventtype"])]
        latencies = [float(v) / 1000 for v in as_list(epoch["eventlatency"])]
        if len(types) != len(latencies):
            raise ValueError("Unmatched event type / latency")
        # Restrict to the same pain-associated cue population.
        current_cues = [v for v, t in zip(types, latencies)
                        if abs(t) < 0.003 and re.fullmatch(r"(left|right)[1-4]", v)]
        if len(current_cues) != 1:
            continue
        events = [(v, t) for v, t in zip(types, latencies)
                  if re.fullmatch(r"PColl(Shock|NoShock)[1-4]", v) and 0 < t < 3.5]
        if len(events) != 1:
            yield {"source_trial": i, "windows": None, "error": "ambiguous_collision"}
            continue
        name, onset = events[0]
        target = int(name.startswith("PCollShock"))
        windows, error = prepare(np.asarray(d["data"])[idx, :, i].copy(), time, fs, onset)
        yield {"source_trial": i, "temperature_c": np.nan, "binary_target": target,
               "target_native": target, "target_units": "delivered/omitted fixed shock 0/1",
               "condition_code": name, "block_context": int(name[-1]),
               "windows": windows, "error": error}


def describe_targets():
    return {
        "heat": {
            "mapping_authority": "https://api.osf.io/v2/wikis/xc3vs/content/",
            "column": "trialinfo[:,1] (MATLAB column 2), stimulus-cue combination",
            "heat_conditions": "1-9 and 19-27",
            "temperature_mapping": "1-3,19-21 =42 C; 4-6,22-24 =46 C; 7-9,25-27 =48 C",
            "actual_schema": "Six columns; old wiki mentions four. Column 4 contains cue markers 37-42, NOT physical temperature. No rating column is used.",
            "alignment": "Deposited time vector around stimulus trigger; no intensity-dependent ramp-up shift.",
            "units_caveat": "MAT has no channel-unit field; voltage values treated as microvolts in line with BrainAmp export convention. This assumption affects amplitude QC and is explicitly disclosed."
        },
        "shock": {
            "mapping_authority": "https://raw.githubusercontent.com/dhewitt16/PainAnticipationVRStudy/main/EEGAnalysis/PPS_2_EpochFiles.m",
            "column": "epoch.eventtype / epoch.eventlatency",
            "positive": "PCollShock1-4: pain-related collision resulting in electrical shock",
            "negative": "PCollNoShock1-4: same pain-associated cues, omitted shock",
            "excluded": "middle/nonpain-associated cues and nPCollNoShock events",
            "alignment": "Collision-event latency in both delivered and omitted cases; not cue onset.",
            "source_processing": "Author average reference, 1-70 Hz filtering, 48-52 Hz stop filter, ICA/manual cleaning; already epoched. Not an equivalent raw preprocessing replication."
        }
    }
