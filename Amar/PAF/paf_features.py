"""Paper-style 9–11 Hz center-of-gravity PAF, without PCA.

Reference: Furman et al. 2019, doi:10.1152/jn.00279.2019.
python paf_features.py [recording-key]
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, periodogram

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
# Shared EEG preprocessing remains in the parent analysis folder.
sys.path.insert(0, str(ROOT))
from de_features import located_record, verify_binary
from nonlinear_eeg import COMMON, FREQ, get_epochs

CHANNELS = ("C3", "C4", "CZ")
SECONDS = 5


def cog(frequencies: np.ndarray, psd: np.ndarray, low=9., high=11.) -> np.ndarray:
    """Power-weighted centroid; zero-power inputs are unavailable, never 10 Hz."""
    mask = (frequencies >= low - 1e-9) & (frequencies <= high + 1e-9)
    power = psd[..., mask]
    total = power.sum(axis=-1)
    return np.divide((power * frequencies[mask]).sum(axis=-1), total,
                     out=np.full(total.shape, np.nan), where=total > 0)


def peak(frequencies: np.ndarray, psd: np.ndarray) -> tuple[float | None, float | None]:
    """Exploratory local 8–12 Hz peak with >=1 dB prominence, not a paper rule."""
    mask = (frequencies >= 7) & (frequencies <= 13)
    f, p = frequencies[mask], psd[mask]
    if not np.all(p > 0):
        return None, None
    ids, properties = find_peaks(10 * np.log10(p), prominence=1.)
    eligible = [i for i, idx in enumerate(ids) if 8 <= f[idx] <= 12]
    if not eligible:
        return None, None
    i = max(eligible, key=lambda i: p[ids[i]])
    return float(f[ids[i]]), float(properties["prominences"][i])


def extract(key: str) -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "sample_manifest.json").read_text())
    record = next(r for r in manifest["recordings"] if r["key"] == key)
    if record["status"] != "downloaded":
        raise ValueError(f"{key}: unavailable raw data")
    resolved = located_record(record)
    verify_binary(resolved)
    epochs, indices, info = get_epochs(resolved, window_seconds=SECONDS, min_windows=1)
    data = epochs[:, [COMMON.index(c) for c in CHANNELS], :]
    f, psd = periodogram(data, fs=FREQ, window=np.hanning(data.shape[-1]),
                         detrend="constant", scaling="density", axis=-1)
    assert np.isclose(f[1] - f[0], .2)
    narrow = cog(f, psd)
    broad = cog(f, psd, 8., 12.)
    if not np.isfinite(narrow).all():
        raise ValueError("Undefined centroid: refusing to fill missing spectral power")
    rows = []
    for i, index in enumerate(indices):
        row = {"recording": key, "epoch_index": int(index),
               "time_start_s": info["segment_start_s"] + SECONDS * int(index),
               "roi_cog_9_11_hz": float(narrow[i].mean())}
        for j, ch in enumerate(CHANNELS):
            row[f"{ch}_cog_9_11_hz"] = float(narrow[i, j])
            row[f"{ch}_cog_8_12_hz"] = float(broad[i, j])
        rows.append(row)
    with (OUT / f"{key}_epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    channel_results = {}
    for j, ch in enumerate(CHANNELS):
        mean_spectrum = psd[:, j].mean(axis=0)
        local_peak, prominence = peak(f, mean_spectrum)
        mask = (f >= 8) & (f <= 12)
        channel_results[ch] = {
            "cog_9_11_hz": float(narrow[:, j].mean()),
            "cog_8_12_hz": float(broad[:, j].mean()),
            "argmax_8_12_hz": float(f[mask][np.argmax(mean_spectrum[mask])]),
            "qualified_local_peak_hz": local_peak,
            "local_peak_prominence_db": prominence,
            "epoch_sd_hz": float(narrow[:, j].std(ddof=1)) if len(narrow) > 1 else None,
        }
    half = len(narrow) // 2
    result = {
        "key": key, "source": record["source"], "status": "analyzed", **info,
        "epoch_seconds": SECONDS, "frequency_spacing_hz": .2,
        "raw_binary_sha256": next(x["sha256"] for x in record["files"]
                                 if x["path"].endswith((".eeg", ".bdf", ".edf"))),
        "roi_cog_9_11_hz": float(narrow.mean()),
        "roi_first_half_hz": float(narrow[:half].mean()) if half else None,
        "roi_second_half_hz": float(narrow[half:].mean()) if half else None,
        "channels": channel_results,
        "qualification": "Excerpt-level spectral feature, not confirmed pain-free baseline or pain prediction.",
    }
    (OUT / f"{key}_result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez_compressed(OUT / f"{key}_spectra.npz", frequency_hz=f,
                        mean_psd_uv2_per_hz=psd.mean(axis=0), epoch_cog_hz=narrow,
                        epoch_index=indices, channels=np.array(CHANNELS))
    print(key, len(indices), result["roi_cog_9_11_hz"], flush=True)


def aggregate() -> list[dict]:
    manifest = json.loads((ROOT / "sample_manifest.json").read_text())
    results, rows = [], []
    for record in manifest["recordings"]:
        key = record["key"]
        if record["status"] != "downloaded":
            result = {"key": key, "status": "unavailable", "reason": "Raw download unavailable; no estimate"}
        else:
            result = json.loads((OUT / f"{key}_result.json").read_text())
            rows.append({
                "recording": key, "epochs_used": result["windows_used"],
                "epochs_rejected": result["windows_rejected"],
                "roi_cog_9_11_hz": result["roi_cog_9_11_hz"],
                **{f"{ch}_cog_hz": result["channels"][ch]["cog_9_11_hz"] for ch in CHANNELS},
                "second_minus_first_half_hz": result["roi_second_half_hz"] - result["roi_first_half_hz"],
                "channels_with_qualified_peak": sum(
                    result["channels"][ch]["qualified_local_peak_hz"] is not None for ch in CHANNELS),
            })
        results.append(result)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return results


if __name__ == "__main__":
    if len(sys.argv) == 2:
        extract(sys.argv[1])
    else:
        manifest = json.loads((ROOT / "sample_manifest.json").read_text())
        env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
        for record in manifest["recordings"]:
            if record["status"] == "downloaded":
                # Never use an old result after a failed extraction.
                (OUT / f"{record['key']}_result.json").unlink(missing_ok=True)
                subprocess.run([sys.executable, __file__, record["key"]],
                               cwd=ROOT, env=env, check=True)
        aggregate()