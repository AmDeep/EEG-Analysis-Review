"""
Generalized EEG feature extraction CLI.

This is a refactor of the Zhao et al. (2025) LEP-mega-dataset-specific
lep_feature_extraction.py into a standalone tool that runs on ANY local
epoched EEG file(s), so it can be dropped into a CI/CD pipeline and pointed
at whatever data lands in an input directory.

WHAT STAYED THE SAME (verbatim, from the original script):
  - Every feature function: N2/P2 peak-picking, gamma band power, alpha/beta
    ERD, PSD band powers (Welch), single-trial inter-site PLV, antropy
    time-series features. These already only depend on generic
    (n_epochs, n_channels, n_times) arrays + channel names + a time vector +
    sampling rate, so nothing about them was OpenNeuro/LEP-specific to begin
    with.
  - The hand-rolled EEGLAB .set/.fdt loader (scipy path + h5py v7.3
    fallback), including the "twofiles" .fdt reshape fix and the same
    caveats about the h5py path being best-effort.

WHAT CHANGED:
  - Input is now "any local file or directory," not "9 hardcoded OpenNeuro
    S3 dataset accessions." The S3 pull is still available (--s3) for
    reproducing the original LEP workflow, but it is no longer required.
  - Added loaders for .fif (MNE) and .edf/.bdf (MNE) so the tool isn't
    locked to EEGLAB's .set format. These both go through `mne`, which is
    an optional dependency (guarded import, same pattern as antropy in the
    original script) -- if mne isn't installed, .set files still work fine,
    but .fif/.edf/.bdf files will raise a clear error telling you to
    `pip install mne`.
  - Label extraction (EEG.event -> rating/laser_power) only applies to
    .set files, since it depends on LEP-specific event fields that have no
    equivalent convention in generic EEG data. For .fif/.edf/.bdf inputs,
    the rating/laser_power columns are simply omitted -- if you have your
    own trial labels for those, join them onto the output CSV yourself
    using the (dataset, subject, epoch) key columns.
  - CAVEAT if you point this at continuous (non-epoched) .fif/.edf/.bdf
    data: it will be cut into fixed-length pseudo-epochs (see
    FALLBACK_EPOCH_LENGTH_SEC below) with NO real stimulus onset at t=0.
    N2/P2/gamma/ERD/PLV windows are all defined relative to a stimulus at
    t=0, so on fixed-length pseudo-epochs those features are not
    meaningful -- only the PSD/antropy features (which don't assume an
    event) are reliable in that case. This is printed as a warning at
    runtime; don't silently trust N2/P2 amplitudes from continuous input.

Requirements:
  pip install -r requirements.txt

Usage:
  # single file
  python extract_features.py --input path/to/sub-01_task-lep_eeg.set --output-dir ./out

  # directory of files (recurses, picks up .set/.fif/.edf/.bdf)
  python extract_features.py --input ./data --output-dir ./out

  # original OpenNeuro S3 workflow (LEP mega-dataset), unchanged behavior
  python extract_features.py --s3 ds005280 --output-dir ./out
"""

import argparse
import io
import re
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, hilbert, butter, filtfilt

warnings.filterwarnings("ignore")

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

try:
    import antropy as ant
    HAVE_ANTROPY = True
except ImportError:
    HAVE_ANTROPY = False
    print("[!] antropy not installed (pip install antropy) -- antropy features will be skipped.")

try:
    import mne
    mne.set_log_level("ERROR")
    HAVE_MNE = True
except ImportError:
    HAVE_MNE = False
    # only fatal if the caller actually hands us a .fif/.edf/.bdf file

# --------------------------------------------------------------------------
# Config (feature windows/bands unchanged from the original script)
# --------------------------------------------------------------------------

VERTEX_CHANNEL_PRIORITY = ["Cz", "CPz", "FCz", "Fz", "Pz"]
PLV_CHANNEL_PAIRS = [("Fz", "Cz"), ("Cz", "Pz"), ("C3", "C4"), ("FCz", "CPz")]

BASELINE_WINDOW = (-0.30, -0.05)
N2_WINDOW = (0.15, 0.35)
P2_WINDOW = (0.30, 0.55)
GAMMA_WINDOW = (0.05, 0.35)
ERD_TASK_WINDOW = (0.0, 0.60)
PLV_WINDOW = (0.0, 0.50)
PSD_WINDOW = (-0.30, 0.60)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}

# only used when a continuous (unepoched) .fif/.edf/.bdf file is supplied
FALLBACK_EPOCH_LENGTH_SEC = 1.0


# --------------------------------------------------------------------------
# .set / .fdt loading -- unchanged from the original script
# --------------------------------------------------------------------------

def _mat_str(val):
    if isinstance(val, np.ndarray):
        val = val.item() if val.size == 1 else val
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _get_chan_names_scipy(eeg):
    chanlocs = getattr(eeg, "chanlocs", None)
    if chanlocs is None:
        return None
    has_len = hasattr(chanlocs, "__len__") and not np.isscalar(chanlocs)
    items = list(chanlocs) if has_len else [chanlocs]
    names = []
    for c in items:
        label = getattr(c, "labels", None)
        names.append(_mat_str(label) if label is not None else "unknown")
    return names


def _get_epoch_times_scipy(eeg):
    times = getattr(eeg, "times", None)
    if times is not None:
        times = np.asarray(times, dtype=float).ravel()
        return times / 1000.0
    srate = float(np.asarray(getattr(eeg, "srate", 1000)).ravel()[0])
    pnts = int(np.asarray(getattr(eeg, "pnts", 0)).ravel()[0])
    xmin = getattr(eeg, "xmin", None)
    if xmin is not None and pnts > 0:
        xmin = float(np.asarray(xmin).ravel()[0])
        return xmin + np.arange(pnts) / srate
    raise ValueError("Could not determine epoch time vector from this .set file.")


def _scalar_int(eeg, field, default=None):
    val = getattr(eeg, field, default)
    if val is None:
        return default
    return int(np.asarray(val).ravel()[0])


def _read_fdt_binary(fdt_path, nbchan, pnts, trials):
    raw = np.fromfile(str(fdt_path), dtype="<f4")
    expected = nbchan * pnts * trials
    if raw.size != expected:
        raise ValueError(
            f".fdt size mismatch: file has {raw.size} float32 values, "
            f"expected nbchan*pnts*trials = {nbchan}*{pnts}*{trials} = {expected}"
        )
    return raw.reshape((nbchan, pnts, trials), order="F")


def load_set_scipy(local_set):
    from scipy.io import loadmat

    mat = loadmat(str(local_set), struct_as_record=False, squeeze_me=True)
    if "EEG" in mat:
        eeg = mat["EEG"]
    elif "epoch" in mat or "data" in mat:
        class FlatStruct:
            pass
        eeg = FlatStruct()
        for k, v in mat.items():
            if not k.startswith("__"):
                setattr(eeg, k, v)
    else:
        raise ValueError(f"Could not find 'EEG' or 'data' among top-level keys: {list(mat.keys())}")

    raw_data = getattr(eeg, "data")
    srate = float(np.asarray(getattr(eeg, "srate", 1000)).ravel()[0])
    ch_names = _get_chan_names_scipy(eeg)
    times = _get_epoch_times_scipy(eeg)

    is_numeric_array = isinstance(raw_data, np.ndarray) and raw_data.dtype.kind in "fciu" and raw_data.ndim >= 2

    if is_numeric_array:
        data = np.asarray(raw_data)
    else:
        fdt_path = local_set.with_suffix(".fdt")
        if not fdt_path.exists():
            raise ValueError(
                f"EEG.data is external ({raw_data!r}) but no matching .fdt file sits "
                f"next to {local_set.name} -- make sure the .fdt is in the same "
                "directory as the .set."
            )
        nbchan = _scalar_int(eeg, "nbchan")
        pnts = _scalar_int(eeg, "pnts")
        trials = _scalar_int(eeg, "trials", default=1)
        if nbchan is None or pnts is None:
            raise ValueError("EEG.data is external but EEG.nbchan/EEG.pnts are missing -- can't reshape .fdt.")
        data = _read_fdt_binary(fdt_path, nbchan, pnts, trials)

    if data.ndim == 3:
        data = np.moveaxis(data, 2, 0)
    elif data.ndim == 2:
        raise ValueError(
            "EEG.data is 2D (continuous, not epoched) -- convert to epochs in "
            "EEGLAB first, or pass a .fif/.edf/.bdf file instead (this tool will "
            "auto fixed-length-epoch those, see FALLBACK_EPOCH_LENGTH_SEC caveat)."
        )
    else:
        raise ValueError(f"Unexpected EEG.data shape: {data.shape}")

    if data.shape[1] != len(ch_names):
        if data.shape[2] == len(ch_names):
            data = np.moveaxis(data, 2, 1)
        else:
            raise ValueError(
                f"data shape {data.shape} doesn't match n_channels={len(ch_names)} "
                "on any axis -- inspect this file manually."
            )

    if data.shape[2] != len(times):
        raise ValueError(
            f"data n_times={data.shape[2]} doesn't match reconstructed times "
            f"vector length={len(times)} -- inspect this file manually."
        )

    return data, ch_names, times, srate


def load_set_h5py(local_set):
    import h5py

    with h5py.File(local_set, "r") as f:
        if "EEG" not in f:
            raise ValueError(f"No 'EEG' group in HDF5 file; top-level keys: {list(f.keys())}")
        g = f["EEG"]

        data = np.asarray(g["data"])
        srate = float(np.asarray(g["srate"]).ravel()[0])

        ch_names = []
        if "chanlocs" in g:
            chanlocs = g["chanlocs"]
            labels_ref = chanlocs["labels"]
            for i in range(labels_ref.shape[0] if labels_ref.ndim else labels_ref.shape[1]):
                try:
                    ref = labels_ref[i][0] if labels_ref.ndim == 2 else labels_ref[0][i]
                    arr = f[ref][()]
                    ch_names.append("".join(chr(c) for c in arr.ravel()))
                except Exception:
                    ch_names.append(f"ch{i}")

        if "times" in g:
            times = np.asarray(g["times"]).ravel() / 1000.0
        else:
            pnts = int(np.asarray(g["pnts"]).ravel()[0])
            xmin = float(np.asarray(g["xmin"]).ravel()[0]) if "xmin" in g else 0.0
            times = xmin + np.arange(pnts) / srate

        if data.ndim == 3 and ch_names and len(times):
            shape = data.shape
            ch_axis = next((i for i, s in enumerate(shape) if s == len(ch_names)), None)
            t_axis = next((i for i, s in enumerate(shape) if s == len(times) and i != ch_axis), None)
            if ch_axis is not None and t_axis is not None:
                ep_axis = [i for i in range(3) if i not in (ch_axis, t_axis)][0]
                data = np.transpose(data, (ep_axis, ch_axis, t_axis))

        return data, ch_names, times, srate


def load_epoched_set(local_set):
    try:
        return load_set_scipy(local_set)
    except ValueError:
        raise
    except Exception:
        pass  # likely v7.3, fall through
    return load_set_h5py(local_set)


def extract_epoch_labels_scipy(local_set, n_epochs):
    from scipy.io import loadmat

    mat = loadmat(str(local_set), struct_as_record=False, squeeze_me=True)
    eeg = mat.get("EEG", None)
    if eeg is None:
        return pd.DataFrame(index=range(n_epochs))

    event = getattr(eeg, "event", None)
    if event is None:
        return pd.DataFrame(index=range(n_epochs))

    has_len = hasattr(event, "__len__") and not np.isscalar(event)
    events_list = list(event) if has_len else [event]

    rows = {}
    for ev in events_list:
        epoch_idx = getattr(ev, "epoch", None)
        if epoch_idx is None:
            continue
        epoch_idx = int(np.asarray(epoch_idx).ravel()[0]) - 1
        if epoch_idx in rows:
            continue
        rows[epoch_idx] = {
            "rating": getattr(ev, "rating", None),
            "laser_power": getattr(ev, "laser_power", None),
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reindex(range(n_epochs))


# --------------------------------------------------------------------------
# .fif / .edf / .bdf loading -- new, via MNE
# --------------------------------------------------------------------------

def load_mne_file(local_path):
    """Load any MNE-readable file. Returns
    (data[n_epochs, n_channels, n_times], ch_names, times_sec, srate, is_pseudo_epoched).
    True epochs (.fif saved as Epochs, or a .set opened via MNE-BIDS-style
    naming) keep their real stimulus-locked time axis. Continuous data
    (Raw) is cut into fixed-length, NON-stimulus-locked pseudo-epochs --
    see the FALLBACK_EPOCH_LENGTH_SEC caveat in the module docstring."""
    if not HAVE_MNE:
        raise ImportError(
            "mne is required to read .fif/.edf/.bdf files. Install it with "
            "`pip install mne` (see requirements.txt)."
        )

    suffix = local_path.suffix.lower()
    is_pseudo_epoched = False

    if suffix == ".fif":
        try:
            epochs = mne.read_epochs(str(local_path), preload=True, verbose=False)
        except Exception:
            raw = mne.io.read_raw_fif(str(local_path), preload=True, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw, duration=FALLBACK_EPOCH_LENGTH_SEC, preload=True, verbose=False
            )
            is_pseudo_epoched = True
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(local_path), preload=True, verbose=False)
        epochs = mne.make_fixed_length_epochs(
            raw, duration=FALLBACK_EPOCH_LENGTH_SEC, preload=True, verbose=False
        )
        is_pseudo_epoched = True
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(str(local_path), preload=True, verbose=False)
        epochs = mne.make_fixed_length_epochs(
            raw, duration=FALLBACK_EPOCH_LENGTH_SEC, preload=True, verbose=False
        )
        is_pseudo_epoched = True
    else:
        raise ValueError(f"Unsupported MNE format: {suffix}")

    data = epochs.get_data()  # (n_epochs, n_channels, n_times), volts
    data = data * 1e6  # convert to microvolts to match EEGLAB-style units used elsewhere
    ch_names = epochs.info["ch_names"]
    times = epochs.times.astype(float)  # already seconds, relative to epoch/event onset (or 0 if pseudo)
    srate = float(epochs.info["sfreq"])

    return data, ch_names, times, srate, is_pseudo_epoched


# --------------------------------------------------------------------------
# Dispatch: any input file -> (data, ch_names, times, sfreq, labels_df, is_pseudo_epoched)
# --------------------------------------------------------------------------

def load_any_eeg_file(local_path):
    local_path = Path(local_path)
    suffix = local_path.suffix.lower()

    if suffix == ".set":
        data, ch_names, times, srate = load_epoched_set(local_path)
        labels_df = extract_epoch_labels_scipy(local_path, data.shape[0])
        return data, ch_names, times, srate, labels_df, False

    if suffix in (".fif", ".edf", ".bdf"):
        data, ch_names, times, srate, is_pseudo = load_mne_file(local_path)
        labels_df = pd.DataFrame(index=range(data.shape[0]))  # no label convention for these formats
        return data, ch_names, times, srate, labels_df, is_pseudo

    raise ValueError(
        f"Unsupported file extension '{suffix}' for {local_path.name}. "
        "Supported: .set (+ companion .fdt), .fif, .edf, .bdf"
    )


# --------------------------------------------------------------------------
# Channel selection -- unchanged
# --------------------------------------------------------------------------

def pick_channel(ch_names, priority_list):
    lower_map = {c.lower(): i for i, c in enumerate(ch_names)}
    for name in priority_list:
        if name.lower() in lower_map:
            return lower_map[name.lower()], ch_names[lower_map[name.lower()]]
    return None, None


def resolve_plv_pairs(ch_names):
    lower_map = {c.lower(): i for i, c in enumerate(ch_names)}
    resolved = []
    for a, b in PLV_CHANNEL_PAIRS:
        if a.lower() in lower_map and b.lower() in lower_map:
            resolved.append((lower_map[a.lower()], lower_map[b.lower()], f"{a}-{b}"))
    return resolved


# --------------------------------------------------------------------------
# Feature functions -- unchanged from the original script
# --------------------------------------------------------------------------

def _window_mask(times, window):
    return (times >= window[0]) & (times <= window[1])


def bandpass(sig, sfreq, band, order=4):
    nyq = sfreq / 2.0
    low = max(band[0] / nyq, 1e-4)
    high = min(band[1] / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def bandpower_welch(sig, sfreq, band, nperseg=None):
    nperseg = min(nperseg or len(sig), len(sig))
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    n_bins = int(mask.sum())
    if n_bins == 0:
        return np.nan
    if n_bins == 1:
        bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else band[1] - band[0]
        return float(psd[mask][0] * bin_width)
    return _trapz(psd[mask], freqs[mask])


def n2_p2_features(sig, times):
    baseline = sig[_window_mask(times, BASELINE_WINDOW)]
    baseline_mean = np.nanmean(baseline) if baseline.size else 0.0
    sig_bc = sig - baseline_mean

    n2_mask = _window_mask(times, N2_WINDOW)
    p2_mask = _window_mask(times, P2_WINDOW)

    if not n2_mask.any() or not p2_mask.any():
        return dict(n2_amp=np.nan, n2_lat=np.nan, p2_amp=np.nan, p2_lat=np.nan, n2p2_amp=np.nan)

    n2_seg, n2_t = sig_bc[n2_mask], times[n2_mask]
    p2_seg, p2_t = sig_bc[p2_mask], times[p2_mask]

    n2_idx = np.argmin(n2_seg)
    p2_idx = np.argmax(p2_seg)

    n2_amp, n2_lat = float(n2_seg[n2_idx]), float(n2_t[n2_idx])
    p2_amp, p2_lat = float(p2_seg[p2_idx]), float(p2_t[p2_idx])

    return dict(
        n2_amp=n2_amp, n2_lat=n2_lat,
        p2_amp=p2_amp, p2_lat=p2_lat,
        n2p2_amp=p2_amp - n2_amp,
    )


def gamma_power_feature(sig, times, sfreq):
    seg = sig[_window_mask(times, GAMMA_WINDOW)]
    if seg.size < 8:
        return np.nan, BANDS["gamma"]
    band = (BANDS["gamma"][0], min(BANDS["gamma"][1], sfreq / 2 - 5))
    return bandpower_welch(seg, sfreq, band), band


def erd_features(sig, times, sfreq):
    baseline = sig[_window_mask(times, BASELINE_WINDOW)]
    task = sig[_window_mask(times, ERD_TASK_WINDOW)]
    out = {}
    for band_name in ("alpha", "beta"):
        band = BANDS[band_name]
        if baseline.size < 8 or task.size < 8:
            out[f"{band_name}_erd_pct"] = np.nan
            continue
        p_base = bandpower_welch(baseline, sfreq, band)
        p_task = bandpower_welch(task, sfreq, band)
        if not p_base or np.isnan(p_base) or p_base == 0:
            out[f"{band_name}_erd_pct"] = np.nan
        else:
            out[f"{band_name}_erd_pct"] = float((p_base - p_task) / p_base * 100.0)
    return out


def psd_band_features(sig, times, sfreq):
    seg = sig[_window_mask(times, PSD_WINDOW)]
    out = {}
    if seg.size < 8:
        for band_name in BANDS:
            out[f"psd_{band_name}"] = np.nan
        return out
    for band_name, band in BANDS.items():
        band = (band[0], min(band[1], sfreq / 2 - 5))
        out[f"psd_{band_name}"] = bandpower_welch(seg, sfreq, band)
    return out


def plv_single_trial(data, times, sfreq, ch_pairs):
    out = {}
    mask = _window_mask(times, PLV_WINDOW)
    if mask.sum() < 8:
        for _, _, label in ch_pairs:
            out[f"plv_{label}"] = np.nan
        return out

    for i, j, label in ch_pairs:
        try:
            sig_i = bandpass(data[i, :], sfreq, (4, 30))[mask]
            sig_j = bandpass(data[j, :], sfreq, (4, 30))[mask]
            phase_i = np.angle(hilbert(sig_i))
            phase_j = np.angle(hilbert(sig_j))
            phase_diff = phase_i - phase_j
            out[f"plv_{label}"] = float(np.abs(np.mean(np.exp(1j * phase_diff))))
        except Exception:
            out[f"plv_{label}"] = np.nan
    return out


def antropy_features(sig):
    out = {}
    if not HAVE_ANTROPY:
        return out
    try:
        out["perm_entropy"] = ant.perm_entropy(sig, normalize=True)
    except Exception:
        out["perm_entropy"] = np.nan
    try:
        out["spectral_entropy"] = ant.spectral_entropy(sig, sf=1.0, method="welch", normalize=True)
    except Exception:
        out["spectral_entropy"] = np.nan
    try:
        out["sample_entropy"] = ant.sample_entropy(sig)
    except Exception:
        out["sample_entropy"] = np.nan
    try:
        out["higuchi_fd"] = ant.higuchi_fd(sig)
    except Exception:
        out["higuchi_fd"] = np.nan
    try:
        out["dfa"] = ant.detrended_fluctuation(sig)
    except Exception:
        out["dfa"] = np.nan
    try:
        mobility, complexity = ant.hjorth_params(sig)
        out["hjorth_mobility"] = mobility
        out["hjorth_complexity"] = complexity
    except Exception:
        out["hjorth_mobility"] = np.nan
        out["hjorth_complexity"] = np.nan
    return out


# --------------------------------------------------------------------------
# Per-file pipeline
# --------------------------------------------------------------------------

def process_file(local_path, dataset_label=None, print_epoch_info=True):
    """Extract one feature row per epoch from a single local EEG file.
    Returns a list of dicts (one per epoch)."""
    local_path = Path(local_path)
    m = re.search(r"(sub-[A-Za-z0-9]+)", local_path.stem)
    subj = m.group(1) if m else local_path.stem
    dataset_label = dataset_label or local_path.parent.name or "dataset"

    rows = []
    try:
        data, ch_names, times, sfreq, labels_df, is_pseudo_epoched = load_any_eeg_file(local_path)
        n_epochs = data.shape[0]

        if print_epoch_info:
            print(f"    epoch window: {times.min():.3f}s to {times.max():.3f}s "
                  f"({times.max()-times.min():.3f}s total, {len(times)} samples @ {sfreq:.0f} Hz)"
                  + ("  [PSEUDO-EPOCHED: no real stimulus onset, N2/P2/gamma/ERD/PLV are not "
                     "meaningful here, only PSD/antropy features are]" if is_pseudo_epoched else ""))

        vertex_idx, vertex_name = pick_channel(ch_names, VERTEX_CHANNEL_PRIORITY)
        if vertex_idx is None:
            print(f"    [!] {subj}: no vertex-priority channel found among {ch_names[:10]}..., skipping")
            return []

        plv_pairs = resolve_plv_pairs(ch_names)

        for ep in range(n_epochs):
            sig = data[ep, vertex_idx, :]

            row = {"dataset": dataset_label, "subject": subj, "epoch": ep,
                   "source_file": local_path.name, "vertex_channel": vertex_name,
                   "sfreq": sfreq, "pseudo_epoched": is_pseudo_epoched}

            row.update(n2_p2_features(sig, times))

            gamma_pow, gamma_band_used = gamma_power_feature(sig, times, sfreq)
            row["gamma_power"] = gamma_pow
            row["gamma_band_hz"] = f"{gamma_band_used[0]:.1f}-{gamma_band_used[1]:.1f}"

            row.update(erd_features(sig, times, sfreq))
            row.update(psd_band_features(sig, times, sfreq))
            row.update(plv_single_trial(data[ep], times, sfreq, plv_pairs))
            row.update(antropy_features(sig[_window_mask(times, PSD_WINDOW)]))

            if ep in labels_df.index:
                if "rating" in labels_df.columns:
                    row["rating"] = labels_df.loc[ep, "rating"]
                if "laser_power" in labels_df.columns:
                    row["laser_power"] = labels_df.loc[ep, "laser_power"]

            rows.append(row)

        print(f"    {subj}: {n_epochs} epochs -> features extracted (vertex={vertex_name}, "
              f"{len(plv_pairs)} PLV pairs)")

    except Exception as e:
        print(f"    [!] {local_path.name}: failed ({e})")

    return rows


def discover_input_files(input_path):
    """Return a sorted list of supported EEG files under input_path (a file
    or a directory, searched recursively)."""
    input_path = Path(input_path)
    supported_ext = {".set", ".fif", ".edf", ".bdf"}
    if input_path.is_file():
        if input_path.suffix.lower() not in supported_ext:
            raise ValueError(f"{input_path} has unsupported extension {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in supported_ext)
        return files
    raise FileNotFoundError(f"--input path does not exist: {input_path}")


# --------------------------------------------------------------------------
# Optional: original OpenNeuro S3 workflow, kept for backward compatibility
# --------------------------------------------------------------------------

def run_s3_workflow(accession, output_dir, label_subfolder="rerefer", max_subjects=None):
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = "openneuro.org"

    def list_keys(prefix):
        paginator = s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def fetch_bytes(key):
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    prefix = f"{accession}/"
    all_keys = list_keys(prefix)
    set_keys = sorted(k for k in all_keys if f"/derivatives/{label_subfolder}/" in k and k.endswith(".set"))
    if not set_keys:
        print(f"  [!] No .set files found under derivatives/{label_subfolder}/ for {accession}")
        return pd.DataFrame()
    if max_subjects:
        set_keys = set_keys[:max_subjects]

    local_dir = output_dir / "raw_downloaded" / accession
    local_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, key in enumerate(set_keys, 1):
        print(f"  [{i}/{len(set_keys)}] {key}")
        local_set = local_dir / Path(key).name
        local_fdt = None
        try:
            local_set.write_bytes(fetch_bytes(key))
            fdt_key = key.rsplit(".", 1)[0] + ".fdt"
            if fdt_key in all_keys:
                local_fdt = local_dir / Path(fdt_key).name
                local_fdt.write_bytes(fetch_bytes(fdt_key))
            all_rows.extend(process_file(local_set, dataset_label=accession, print_epoch_info=(i == 1)))
        except Exception:
            print(f"    [!] Unhandled error on {key}, skipping.")
            traceback.print_exc()
        finally:
            for p in (local_set, local_fdt):
                try:
                    if p is not None and p.exists():
                        p.unlink()
                except Exception:
                    pass

    return pd.DataFrame(all_rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract EEG features from any local .set/.fif/.edf/.bdf file(s).")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to a single EEG file, or a directory to search recursively.")
    parser.add_argument("--s3", type=str, default=None,
                         help="OpenNeuro accession (e.g. ds005280) to pull via the original S3 workflow, "
                              "instead of --input.")
    parser.add_argument("--label-subfolder", type=str, default="rerefer",
                         help="Only used with --s3: derivatives subfolder to pull .set files from.")
    parser.add_argument("--max-subjects", type=int, default=None,
                         help="Only used with --s3: cap number of subjects, for a quick test run.")
    parser.add_argument("--output-dir", type=str, default="./feature_output",
                         help="Where to write per-file and combined feature CSVs.")
    parser.add_argument("--max-files", type=int, default=None,
                         help="Cap number of local files processed, for a quick test run.")
    args = parser.parse_args()

    if not args.input and not args.s3:
        parser.error("Provide either --input <file-or-dir> or --s3 <openneuro-accession>.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.s3:
        print(f"\n=== Feature extraction (S3): {args.s3} ===")
        df = run_s3_workflow(args.s3, output_dir, args.label_subfolder, args.max_subjects)
        if not df.empty:
            out_path = output_dir / f"features_{args.s3}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved -> {out_path}")
        else:
            print("  No features extracted.")
        return 0 if not df.empty else 1

    files = discover_input_files(args.input)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print(f"[!] No supported EEG files (.set/.fif/.edf/.bdf) found under {args.input}")
        return 1

    print(f"\n=== Feature extraction: {len(files)} file(s) from {args.input} ===")
    all_rows = []
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f}")
        try:
            all_rows.extend(process_file(f, print_epoch_info=(i == 1)))
        except Exception:
            print(f"    [!] Unhandled error on {f}, skipping. Full traceback:")
            traceback.print_exc()

    if not all_rows:
        print("\nNo features extracted -- check input files / formats.")
        return 1

    combined = pd.DataFrame(all_rows)
    out_path = output_dir / "features_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined feature table -> {out_path} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
