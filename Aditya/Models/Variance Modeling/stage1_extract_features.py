"""Stage 1 feature extraction: baseline vs. response window, single vertex channel.

Reads an epoch tensor built with build_epochs.py --tmin -1.0 --tmax 1.0 (baseline
= [-1, 0), response = [0, 1) relative to stimulus onset) and computes a small,
interpretable feature set separately on each half, for one vertex channel.

Output is one row per trial with `<feature>_baseline` and `<feature>_response`
columns; baseline_normalization.py turns those into deltas/Z-scores.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import welch

try:
    import antropy as ant
    HAVE_ANTROPY = True
except ImportError:
    HAVE_ANTROPY = False
    print("[!] antropy not installed (pip install antropy) -- entropy/Hjorth features will be NaN.")

# numpy 2.0 renamed trapz -> trapezoid and dropped the old name entirely
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

VERTEX_PRIORITY = ["Cz", "CPz", "FCz", "Fz", "Pz"]

BASE_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}
RATIO_PAIRS = [("beta", "alpha"), ("gamma", "alpha"), ("theta", "alpha")]


def pick_channel(ch_names, priority):
    lower = {c.lower(): i for i, c in enumerate(ch_names)}
    for name in priority:
        if name.lower() in lower:
            i = lower[name.lower()]
            return i, ch_names[i]
    raise ValueError(f"None of {priority} found in {ch_names}")


def window_features(sig, sfreq):
    """All Stage 1 features for one 1D time window."""
    out = {}
    gamma_hi = min(100.0, sfreq / 2 - 5)
    bands = dict(BASE_BANDS, gamma=(30.0, gamma_hi))

    nperseg = min(len(sig), 256)
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    broadband = (freqs >= 1) & (freqs <= gamma_hi)
    total_power = float(_trapz(psd[broadband], freqs[broadband])) if broadband.sum() > 1 else np.nan

    band_power = {}
    for name, band in bands.items():
        mask = (freqs >= band[0]) & (freqs <= band[1])
        p = float(_trapz(psd[mask], freqs[mask])) if mask.sum() > 1 else np.nan
        band_power[name] = p
        out[f"{name}_power_abs"] = p
        out[f"{name}_power_rel"] = p / total_power if total_power else np.nan

    for num, den in RATIO_PAIRS:
        d = band_power.get(den)
        out[f"{num}_{den}_ratio"] = band_power[num] / d if d else np.nan

    out["peak_freq"] = float(freqs[broadband][np.argmax(psd[broadband])]) if broadband.any() else np.nan

    out["rms"] = float(np.sqrt(np.mean(sig ** 2)))
    out["variance"] = float(np.var(sig))
    out["std"] = float(np.std(sig))
    out["hjorth_activity"] = out["variance"]

    if HAVE_ANTROPY:
        try:
            out["spectral_entropy"] = float(ant.spectral_entropy(sig, sf=sfreq, method="welch", normalize=True))
        except Exception:
            out["spectral_entropy"] = np.nan
        try:
            mobility, complexity = ant.hjorth_params(sig)
            out["hjorth_mobility"] = float(mobility)
            out["hjorth_complexity"] = float(complexity)
        except Exception:
            out["hjorth_mobility"] = np.nan
            out["hjorth_complexity"] = np.nan
        try:
            out["sample_entropy"] = float(ant.sample_entropy(sig))
        except Exception:
            out["sample_entropy"] = np.nan
    else:
        out["spectral_entropy"] = out["hjorth_mobility"] = out["hjorth_complexity"] = out["sample_entropy"] = np.nan

    return out


def baseline_drift(sig, n_quarters=4):
    """QC only (not a filter): RMS in n_quarters equal sub-windows of the
    baseline, plus the linear trend across them. A steadily rising/falling RMS
    means the "baseline" isn't a stable pre-stimulus resting state."""
    n = len(sig)
    edges = np.linspace(0, n, n_quarters + 1).astype(int)
    rms_vals = [float(np.sqrt(np.mean(sig[edges[i]:edges[i + 1]] ** 2))) for i in range(n_quarters)]
    slope = float(np.polyfit(np.arange(n_quarters), rms_vals, 1)[0])
    out = {f"baseline_rms_q{i + 1}": v for i, v in enumerate(rms_vals)}
    out["baseline_drift_slope"] = slope
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="epoch tensor from build_epochs.py --tmin -1.0 --tmax 1.0")
    ap.add_argument("--output", default="features_baseline_response.csv")
    ap.add_argument("--channels", nargs="+", default=None,
                     help="Vertex-channel priority list (default: Cz, CPz, FCz, Fz, Pz); first match wins")
    a = ap.parse_args()

    z = np.load(a.input, allow_pickle=True)
    X = z["X"]  # trials x channels x time
    ch_names = z["ch_names"].tolist()
    sfreq = float(z["sfreq"])
    tmin = float(z["tmin"])
    power, dataset, global_subject, epoch = z["power"], z["dataset"], z["global_subject"], z["epoch"]

    n_t = X.shape[-1]
    split = int(round(-tmin * sfreq))  # sample index of stimulus onset (t=0)
    if not 0 < split < n_t:
        raise ValueError(f"tmin={tmin}, sfreq={sfreq} implies t=0 at sample {split}, outside [0, {n_t})")

    ch_idx, ch_used = pick_channel(ch_names, a.channels or VERTEX_PRIORITY)
    print(f"Using vertex channel: {ch_used} ({X.shape[0]} trials, baseline={split} samples, "
          f"response={n_t - split} samples @ {sfreq:.0f} Hz)")

    rows = []
    for i in range(X.shape[0]):
        sig = X[i, ch_idx, :]
        baseline_sig, response_sig = sig[:split], sig[split:]

        row = {
            "dataset": dataset[i], "global_subject": global_subject[i], "epoch": epoch[i],
            "laser_power": power[i], "vertex_channel": ch_used,
        }
        for k, v in window_features(baseline_sig, sfreq).items():
            row[f"{k}_baseline"] = v
        for k, v in window_features(response_sig, sfreq).items():
            row[f"{k}_response"] = v
        row.update(baseline_drift(baseline_sig))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(a.output, index=False)

    thresh = 3 * df["baseline_drift_slope"].abs().median()
    drift_flag = df["baseline_drift_slope"].abs() > thresh
    print(f"saved {a.output}: {df.shape}")
    print(f"trials with |baseline drift slope| > 3x median ({thresh:.4g}): "
          f"{drift_flag.sum()} / {len(df)} ({100 * drift_flag.mean():.1f}%)")


if __name__ == "__main__":
    main()
