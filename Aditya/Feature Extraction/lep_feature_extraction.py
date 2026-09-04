"""
Feature extraction for the Zhao et al. (2025) LEP mega-dataset (same 9
experiments as lep_eda.py).

lep_eda.py's Stage 1.5 pulls per-trial LABELS (rating, laser_power) from
EEG.event in derivatives/rerefer/*.set. This script is the companion piece:
it pulls the actual epoched SIGNAL (EEG.data) from the same .set files and
computes a feature table, one row per epoch/trial, with:

  - N2 amplitude + latency         (negative peak, vertex channel)
  - P2 amplitude + latency         (positive peak, vertex channel)
  - N2-P2 peak-to-peak amplitude
  - Gamma band power               (poststimulus window, vertex channel)
  - Alpha/beta ERD                 (% power change, task vs. baseline window)
  - Inter-site phase locking (PLV) (single-trial PLV between electrode pairs)
  - PSD band powers                (delta/theta/alpha/beta/gamma, Welch)
  - Antropy time-series features   (permutation/spectral/sample entropy,
                                     Higuchi FD, DFA, Hjorth mobility/complexity)

FIX (this version): label extraction previously re-parsed the .set file from
scratch via its own loadmat() call that only checked for a top-level "EEG"
key. Per lep_eda.py's Stage 0.5 diagnostics, most of these 9 datasets load
via a FLATTENED top-level struct (no "EEG" key at all), which load_set_scipy
already handles correctly for signal data -- but the old label extractor
didn't, so it silently returned all-NaN labels for every flattened-struct
dataset. Labels are now pulled from the SAME already-normalized `eeg` object
that load_set_scipy produces, so both loaders agree on struct shape, and each
subject file is now only loadmat()'d once instead of twice.

IMPORTANT CAVEATS (read before trusting the output):
  - N2/P2 windows below (150-350 ms / 300-550 ms) are reasonable literature
    defaults for laser-evoked potentials at the vertex, but you should sanity
    check them against a grand-average ERP for each dataset before treating
    the amplitudes/latencies as final -- stimulus site (left vs. right hand)
    and interstimulus jitter can shift these slightly. See PEAK WINDOWS below.
  - Gamma band power is capped at min(GAMMA_BAND[1], sfreq/2 - 5) per dataset,
    since not every dataset's sampling rate comfortably supports 30-100 Hz
    (1000/1024 Hz datasets are fine; if you point this at a lower-sfreq
    dataset the effective gamma band will silently narrow -- check the
    printed "effective gamma band" line per subject if that matters to you).
  - The h5py (MATLAB v7.3/HDF5) fallback path is best-effort, same as in
    lep_eda.py. It has NOT been validated against every dataset in DATASETS,
    and it does NOT currently extract labels (no eeg struct is available in
    that branch) -- if any subject anywhere ends up going through this path,
    its rows will have signal features but NaN rating/laser_power. Per
    Stage 0.5 diagnostics none of the 9 datasets currently need this path,
    so this is a latent gap, not an active one.
  - Single-trial PLV (see plv_single_trial below) measures phase-locking
    ACROSS TIME WITHIN one trial between two electrodes, not the more common
    across-TRIALS PLV at one electrode pair/timepoint. This is a legitimate
    but less standard variant -- say so explicitly if you write this up.

Requirements:
  pip install boto3 pandas numpy scipy mne antropy h5py

Run:
  python lep_feature_extraction.py
"""

import re
import traceback
import warnings
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config
from scipy.signal import welch, hilbert, butter, filtfilt

warnings.filterwarnings("ignore")

# numpy 2.0 renamed trapz -> trapezoid and dropped the old name entirely
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

try:
    import antropy as ant
    HAVE_ANTROPY = True
except ImportError:
    HAVE_ANTROPY = False
    print("[!] antropy not installed (pip install antropy) -- antropy features will be skipped.")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATASETS = {
    "exp1_ds005284": "ds005284",
    "exp2_ds005285": "ds005285",
    "exp3_ds005289": "ds005289",
    "exp4_ds005286": "ds005286",
    "exp5_ds005291": "ds005291",
    "exp6_ds005293": "ds005293",
    "exp7_ds005292": "ds005292",
    "exp8_ds005280": "ds005280",
    "exp9_ds005473": "ds005473",
}

BUCKET = "openneuro.org"
OUT_DIR = Path("./lep_feature_output")
OUT_DIR.mkdir(exist_ok=True, parents=True)

LABEL_SUBFOLDER = "rerefer"     # same derivatives subfolder lep_eda.py pulls labels from
MAX_SUBJECTS_PER_DATASET = None  # set an int to test quickly; None = all subjects

# vertex-channel priority list for N2/P2/gamma/ERD (first match wins per dataset montage)
VERTEX_CHANNEL_PRIORITY = ["Cz", "CPz", "FCz", "Fz", "Pz"]

# electrode pairs to compute inter-site PLV over (first match per side wins; skipped if absent)
PLV_CHANNEL_PAIRS = [("Fz", "Cz"), ("Cz", "Pz"), ("C3", "C4"), ("FCz", "CPz")]

# time windows, all in SECONDS relative to stimulus onset (t=0)
BASELINE_WINDOW = (-0.30, -0.05)
N2_WINDOW = (0.15, 0.35)
P2_WINDOW = (0.30, 0.55)
GAMMA_WINDOW = (0.05, 0.35)      # poststimulus window for gamma power
ERD_TASK_WINDOW = (0.0, 0.60)    # poststimulus window for alpha/beta ERD
PLV_WINDOW = (0.0, 0.50)         # poststimulus window for inter-site PLV
PSD_WINDOW = (-0.30, 0.60)       # window PSD/antropy features are computed over

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),  # capped per-dataset at sfreq/2 - 5, see GAMMA_BAND note above
}

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


# --------------------------------------------------------------------------
# S3 / dataset access helpers (same pattern as lep_eda.py)
# --------------------------------------------------------------------------

def list_keys(prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def fetch_bytes(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def find_latest_version_prefix(accession):
    root_keys = list_keys(f"{accession}/")
    if root_keys:
        return f"{accession}/"
    raise RuntimeError(f"Could not find any objects under prefix {accession}/")


# --------------------------------------------------------------------------
# .set loading -- pulls the actual epoched signal (EEG.data) AND returns the
# normalized eeg struct so label extraction can reuse it (see fix note above).
# --------------------------------------------------------------------------

def _mat_str(val):
    """scipy sometimes returns 0-d arrays / bytes for what should be plain
    strings (e.g. channel labels, dataset field names) -- normalize to str."""
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
    """Reconstruct the per-epoch time vector (seconds) from whatever fields
    are present. EEGLAB epoched sets normally carry EEG.times in ms."""
    times = getattr(eeg, "times", None)
    if times is not None:
        times = np.asarray(times, dtype=float).ravel()
        # EEGLAB stores this in milliseconds
        return times / 1000.0

    # fallback: reconstruct from xmin/xmax/pnts/srate
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
    """EEGLAB's 'twofiles' save mode writes EEG.data to disk as raw
    single-precision floats and leaves EEG.data in the .set as just a
    filename string (which is why np.asarray(...) on it collapses to shape
    ()). The .fdt is written from a MATLAB array of shape
    [nbchan x (pnts*trials)] via A(:), i.e. column-major with channel as the
    fastest-varying index -- reshape with order='F' to recover that."""
    raw = np.fromfile(str(fdt_path), dtype="<f4")
    expected = nbchan * pnts * trials
    if raw.size != expected:
        raise ValueError(
            f".fdt size mismatch: file has {raw.size} float32 values, "
            f"expected nbchan*pnts*trials = {nbchan}*{pnts}*{trials} = {expected}"
        )
    return raw.reshape((nbchan, pnts, trials), order="F")  # (n_channels, n_times, n_epochs)


def load_set_scipy(local_set):
    """Load a .set file via scipy.io.loadmat and return
    (data[n_epochs, n_channels, n_times], ch_names, times_sec, srate, eeg)
    where eeg is the normalized struct (works whether the file stored a
    top-level "EEG" key or a flattened struct) -- callers should reuse eeg
    for label extraction instead of re-loading the file. Raises for v7.3/
    HDF5 files (caller should fall back to load_set_h5py)."""
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

    # Two possible storage modes:
    #  (a) EEG.data is the actual numeric array, embedded in the .set
    #  (b) EEG.data is just a filename string, with the real values written
    #      externally to a .fdt file ('twofiles' save mode) -- this is what
    #      produced "Unexpected EEG.data shape: ()" before this fix, since
    #      np.asarray() on a string collapses to a 0-d array.
    is_numeric_array = isinstance(raw_data, np.ndarray) and raw_data.dtype.kind in "fciu" and raw_data.ndim >= 2

    if is_numeric_array:
        data = np.asarray(raw_data)
    else:
        fdt_path = local_set.with_suffix(".fdt")
        if not fdt_path.exists():
            raise ValueError(
                f"EEG.data is external ({raw_data!r}) but no matching .fdt file was "
                f"downloaded alongside {local_set.name} -- check that the caller also "
                "fetches the .fdt key."
            )
        nbchan = _scalar_int(eeg, "nbchan")
        pnts = _scalar_int(eeg, "pnts")
        trials = _scalar_int(eeg, "trials", default=1)
        if nbchan is None or pnts is None:
            raise ValueError("EEG.data is external but EEG.nbchan/EEG.pnts are missing -- can't reshape .fdt.")
        data = _read_fdt_binary(fdt_path, nbchan, pnts, trials)

    # EEGLAB stores epoched data as (n_channels, n_times, n_epochs) -- move
    # epochs to axis 0 for convenience. Continuous (unepoched) 2D data is
    # NOT handled here since derivatives/rerefer files in this dataset are
    # already epoched per lep_eda.py's Stage 0.5 findings.
    if data.ndim == 3:
        data = np.moveaxis(data, 2, 0)  # -> (n_epochs, n_channels, n_times)
    elif data.ndim == 2:
        raise ValueError(
            "EEG.data is 2D (continuous, not epoched) -- this script expects "
            "already-epoched derivative .set files. Skipping this file."
        )
    else:
        raise ValueError(f"Unexpected EEG.data shape: {data.shape}")

    if data.shape[1] != len(ch_names):
        # some datasets store data as (n_epochs, n_times, n_channels); guard
        # against the axis order being flipped relative to what we assumed
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

    return data, ch_names, times, srate, eeg


def load_set_h5py(local_set):
    """Best-effort MATLAB v7.3/HDF5 fallback. NOT validated against every
    dataset -- see the module docstring caveat. Does NOT return an eeg
    struct usable for label extraction (h5py groups don't map cleanly onto
    the same attribute-access pattern) -- rows from this path will have
    signal features but NaN rating/laser_power. If output looks wrong, open
    the file in EEGLAB instead of trusting it."""
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

        times = None
        if "times" in g:
            times = np.asarray(g["times"]).ravel() / 1000.0
        else:
            pnts = int(np.asarray(g["pnts"]).ravel()[0])
            xmin = float(np.asarray(g["xmin"]).ravel()[0]) if "xmin" in g else 0.0
            times = xmin + np.arange(pnts) / srate

        # Reorder data to (n_epochs, n_channels, n_times) -- infer axis order
        # from which dimension matches len(ch_names) and len(times).
        if data.ndim == 3 and ch_names and len(times):
            shape = data.shape
            ch_axis = next((i for i, s in enumerate(shape) if s == len(ch_names)), None)
            t_axis = next((i for i, s in enumerate(shape) if s == len(times) and i != ch_axis), None)
            if ch_axis is not None and t_axis is not None:
                ep_axis = [i for i in range(3) if i not in (ch_axis, t_axis)][0]
                data = np.transpose(data, (ep_axis, ch_axis, t_axis))

        return data, ch_names, times, srate, None  # no eeg struct -> no labels from this path


def load_epoched_set(local_set):
    try:
        return load_set_scipy(local_set)
    except NotImplementedError:
        pass
    except ValueError:
        raise
    except Exception:
        pass  # likely v7.3, fall through
    return load_set_h5py(local_set)


def dump_event_fields_sample(local_set, accession):
    """Diagnostic: print the actual field names present on EEG.event for one
    file, so label extraction can be matched to what's really there instead
    of guessed. Run once per dataset when labels aren't showing up."""
    from scipy.io import loadmat

    mat = loadmat(str(local_set), struct_as_record=False, squeeze_me=True)
    eeg = mat.get("EEG", None)
    if eeg is None:
        print(f"    [diag] {accession}: no EEG struct found")
        return
    event = getattr(eeg, "event", None)
    if event is None:
        print(f"    [diag] {accession}: EEG.event is None")
        return
    has_len = hasattr(event, "__len__") and not np.isscalar(event)
    ev0 = event[0] if has_len else event
    fields = [f for f in dir(ev0) if not f.startswith("_")]
    print(f"    [diag] {accession}: EEG.event[0] fields = {fields}")
    for f in fields:
        try:
            print(f"        {f} = {getattr(ev0, f)!r}")
        except Exception:
            pass


def extract_epoch_labels_from_eeg(eeg, n_epochs):
    """Pull rating/laser_power per epoch from an already-loaded, normalized
    eeg struct (the object load_set_scipy returns -- works for both the
    nested-EEG and flattened-struct cases, since eeg is already resolved by
    the time this is called; this is the fix for the label-loss bug: the old
    version re-parsed the file with its own loadmat() call that only checked
    for a top-level "EEG" key and silently produced all-NaN labels for every
    flattened-struct dataset). Returns a DataFrame indexed 0..n_epochs-1.

    If more than one event in an epoch carries values, prefers the one that
    actually has a non-None laser_power over the one encountered first."""
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
        epoch_idx = int(np.asarray(epoch_idx).ravel()[0]) - 1  # MATLAB 1-indexed
        rating = getattr(ev, "rating", None)
        laser_power = getattr(ev, "laser_power", None)
        existing = rows.get(epoch_idx)
        if existing is None or (existing.get("laser_power") is None and laser_power is not None):
            rows[epoch_idx] = {"rating": rating, "laser_power": laser_power}

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reindex(range(n_epochs))


# --------------------------------------------------------------------------
# Channel selection
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
# Feature functions -- all operate on a single epoch's data
#   sig: 1D array (n_times,) for single-channel features
#   data: 2D array (n_channels, n_times) for multi-channel features (PLV)
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
    """Welch PSD band power. Uses the full available segment as one window
    by default (rather than capping nperseg low) since these epochs are
    short (a few hundred to ~1000 samples) and narrow low-frequency bands
    (delta 1-4 Hz, theta 4-8 Hz, alpha 8-13 Hz) need fine frequency
    resolution to land more than one FFT bin inside the band -- a coarse
    nperseg was previously causing delta/theta/alpha to silently compute as
    exactly 0.0 (a single-point "area" is 0 under the trapezoidal rule) and
    causing alpha ERD to come out NaN every time. This trades PSD estimate
    smoothness (single window, no Welch averaging) for actually having
    enough bins to integrate over -- reasonable for per-trial exploratory
    features, but worth knowing if you later want publication-grade PSDs
    (those would want longer segments / lower minimum band width)."""
    nperseg = min(nperseg or len(sig), len(sig))
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    n_bins = int(mask.sum())
    if n_bins == 0:
        return np.nan
    if n_bins == 1:
        # trapz needs >=2 points to give a nonzero area; approximate the
        # single bin's contribution as psd * bin width instead
        bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else band[1] - band[0]
        return float(psd[mask][0] * bin_width)
    return _trapz(psd[mask], freqs[mask])


def n2_p2_features(sig, times):
    """Peak-picking for the N2 (most negative) and P2 (most positive)
    deflections in their respective windows, baseline-corrected."""
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
    """Classic Pfurtscheller ERD%: (baseline_power - task_power) / baseline_power * 100.
    Positive = desynchronization (power drop), negative = synchronization."""
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
    """Single-trial inter-site phase locking: for each electrode pair, band-
    pass (broadband 4-30 Hz by default -- covers theta/alpha/beta where LEP
    phase coupling is usually reported), take the Hilbert phase, compute the
    phase difference across TIME SAMPLES within the PLV_WINDOW of this one
    trial, and take the resultant vector length. This is a single-trial
    variant, not the more standard across-trials PLV -- see module docstring."""
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
    """Time-series complexity/entropy features via the antropy package.
    Applied to the full epoch (PSD_WINDOW), on the vertex channel."""
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
# Per-subject / per-dataset pipeline
# --------------------------------------------------------------------------

def process_subject_file(key, all_keys, local_dir, accession, print_epoch_info=False):
    m = re.search(r"(sub-[A-Za-z0-9]+)", key)
    subj = m.group(1) if m else "unknown"

    local_set = local_dir / Path(key).name
    local_fdt = None
    rows = []
    try:
        local_set.write_bytes(fetch_bytes(key))
        fdt_key = key.rsplit(".", 1)[0] + ".fdt"
        if fdt_key in all_keys:
            local_fdt = local_dir / Path(fdt_key).name
            local_fdt.write_bytes(fetch_bytes(fdt_key))

        data, ch_names, times, sfreq, eeg = load_epoched_set(local_set)
        n_epochs = data.shape[0]

        if print_epoch_info:
            print(f"    epoch window: {times.min():.3f}s to {times.max():.3f}s "
                  f"({times.max()-times.min():.3f}s total, {len(times)} samples @ {sfreq:.0f} Hz)")

        vertex_idx, vertex_name = pick_channel(ch_names, VERTEX_CHANNEL_PRIORITY)
        if vertex_idx is None:
            print(f"    [!] {subj}: no vertex-priority channel found among {ch_names[:10]}..., skipping")
            return []

        plv_pairs = resolve_plv_pairs(ch_names)
        labels_df = extract_epoch_labels_from_eeg(eeg, n_epochs)
        if eeg is None:
            print(f"    [!] {subj}: loaded via h5py fallback -- labels unavailable for this file (see caveats)")

        for ep in range(n_epochs):
            sig = data[ep, vertex_idx, :]

            row = {"dataset": accession, "subject": subj, "epoch": ep,
                   "vertex_channel": vertex_name, "sfreq": sfreq}

            row.update(n2_p2_features(sig, times))

            gamma_pow, gamma_band_used = gamma_power_feature(sig, times, sfreq)
            row["gamma_power"] = gamma_pow
            row["gamma_band_hz"] = f"{gamma_band_used[0]:.1f}-{gamma_band_used[1]:.1f}"

            row.update(erd_features(sig, times, sfreq))
            row.update(psd_band_features(sig, times, sfreq))
            row.update(plv_single_trial(data[ep], times, sfreq, plv_pairs))
            row.update(antropy_features(sig[_window_mask(times, PSD_WINDOW)]))

            if ep in labels_df.index:
                row["rating"] = labels_df.loc[ep, "rating"] if "rating" in labels_df.columns else None
                row["laser_power"] = labels_df.loc[ep, "laser_power"] if "laser_power" in labels_df.columns else None
            else:
                row["rating"] = None
                row["laser_power"] = None

            rows.append(row)

        n_labeled = int(pd.notna([r["laser_power"] for r in rows]).sum())
        print(f"    {subj}: {n_epochs} epochs -> features extracted (vertex={vertex_name}, "
              f"{len(plv_pairs)} PLV pairs, {n_labeled}/{n_epochs} epochs with laser_power)")

    except Exception as e:
        print(f"    [!] {subj}: failed ({e})")
    finally:
        for p in (local_set, local_fdt):
            try:
                if p is not None and p.exists():
                    p.unlink()
            except Exception:
                pass

    return rows


def process_dataset(label, accession):
    print(f"\n=== Feature extraction: {label} ({accession}) ===")
    prefix = find_latest_version_prefix(accession)
    all_keys = list_keys(prefix)

    set_keys = sorted(k for k in all_keys if f"/derivatives/{LABEL_SUBFOLDER}/" in k and k.endswith(".set"))
    if not set_keys:
        print(f"  [!] No .set files found under derivatives/{LABEL_SUBFOLDER}/ for {accession}")
        return pd.DataFrame()

    if MAX_SUBJECTS_PER_DATASET:
        set_keys = set_keys[:MAX_SUBJECTS_PER_DATASET]

    local_dir = OUT_DIR / "raw_epoched" / accession
    local_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, key in enumerate(set_keys, 1):
        print(f"  [{i}/{len(set_keys)}] {key}")
        try:
            all_rows.extend(process_subject_file(key, all_keys, local_dir, accession, print_epoch_info=(i == 1)))
        except Exception:
            # process_subject_file already catches per-subject errors internally;
            # this is a belt-and-suspenders catch for anything that slips past it
            # (e.g. a network hiccup mid-download) so one bad file doesn't kill
            # the whole dataset's run.
            print(f"    [!] Unhandled error on {key}, skipping. Full traceback:")
            traceback.print_exc()

    return pd.DataFrame(all_rows)


def print_label_coverage_summary(combined):
    """Post-run sanity check: per-dataset epoch count and % with a non-null
    laser_power. Catches a regression like the flattened-struct label bug
    immediately instead of silently shipping a smaller labeled dataset."""
    if combined.empty:
        return
    print("\n" + "=" * 72)
    print("LABEL COVERAGE SUMMARY (per dataset)")
    print("=" * 72)
    summary = combined.groupby("dataset").agg(
        n_epochs=("laser_power", "size"),
        n_labeled=("laser_power", lambda s: s.notna().sum()),
    )
    summary["pct_labeled"] = (summary["n_labeled"] / summary["n_epochs"] * 100).round(1)
    print(summary.to_string())
    zero_label_datasets = summary.index[summary["n_labeled"] == 0].tolist()
    if zero_label_datasets:
        print(f"\n[!] WARNING: these datasets have ZERO labeled epochs -- "
              f"investigate before using this output: {zero_label_datasets}")


def main():
    frames = []
    for label, accession in DATASETS.items():
        try:
            df = process_dataset(label, accession)
        except Exception:
            print(f"  [!] Dataset {label} ({accession}) failed entirely, skipping. Full traceback:")
            traceback.print_exc()
            continue
        if not df.empty:
            out_path = OUT_DIR / f"features_{accession}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved -> {out_path}")
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(OUT_DIR / "features_combined.csv", index=False)
        print(f"\nSaved combined feature table -> {OUT_DIR / 'features_combined.csv'} "
              f"({len(combined)} rows)")
        print_label_coverage_summary(combined)
    else:
        print("\nNo features extracted -- check dataset access / derivatives subfolder name.")


if __name__ == "__main__":
    main()