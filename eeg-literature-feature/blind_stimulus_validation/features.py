"""Candidate feature-family implementations; no labels enter this module."""
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import signal, stats
from sklearn.neighbors import KDTree

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from fooof import FOOOF

FS = 250
ROI = {"frontal": [0, 1, 2], "central": [3, 4, 5], "posterior": [6, 7, 8]}
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "lowgamma": (30, 40)}
PAIRS = {"F3_F4": (0, 1), "C3_C4": (3, 4), "Fz_Cz": (2, 5), "Cz_Pz": (5, 8)}
FAMILIES = {
    "erp": "Evoked N1/N2/P2 amplitudes",
    "morph": "Evoked latencies/morphology",
    "erd": "Alpha/beta ERD",
    "pre": "Pre-stimulus power/phase",
    "power": "Absolute/relative spectral power",
    "paf": "Peak alpha frequency",
    "aper": "Aperiodic slope/offset",
    "asym": "Hemispheric power asymmetry",
    "ratio": "Alpha/theta power ratio",
    "coh": "Magnitude-squared coherence",
    "plv": "PLV/phase synchrony",
    "lag": "PLI/wPLI/imaginary coherency",
    "pac": "Phase–amplitude coupling",
    "env": "Envelope correlation",
    "variability": "Within-window amplitude variability",
    "hfd": "Higuchi fractal dimension",
    "pe": "Permutation entropy",
    "sampen": "Sample/multiscale entropy",
    "hjorth": "Hjorth parameters",
    "quality": "Signal quality/SNR and variance controls",
    "csp": "CSP/spatial log-variance",
}


def permutation_entropy(x, order=3):
    a = np.lib.stride_tricks.sliding_window_view(x, order)
    ranks = np.argsort(a, axis=1, kind="stable")
    _, count = np.unique(ranks, axis=0, return_counts=True)
    p = count / count.sum()
    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))


def higuchi(x, kmax=8):
    n = len(x)
    lengths = []
    for k in range(1, kmax + 1):
        values = []
        for m in range(k):
            seq = x[m::k]
            count = len(seq) - 1
            values.append(np.sum(np.abs(np.diff(seq))) * (n - 1) /
                          (max(count, 1) * k * k))
        lengths.append(np.mean(values))
    return float(np.polyfit(np.log(1 / np.arange(1, kmax + 1)),
                            np.log(np.maximum(lengths, 1e-15)), 1)[0])


def sample_entropy(x, scale=1):
    if scale > 1:
        n = len(x) // scale
        x = x[:n * scale].reshape(n, scale).mean(axis=1)
    sd = np.std(x)
    if sd < 1e-12 or len(x) < 12:
        return np.nan
    patterns = np.lib.stride_tricks.sliding_window_view(x, 3)
    radius = 0.2 * sd
    counts = []
    for dimensions in (2, 3):
        a = np.ascontiguousarray(patterns[:, :dimensions])
        matches = KDTree(a, metric="chebyshev").query_radius(a, radius, count_only=True) - 1
        counts.append(matches.sum())
    return float(-np.log(counts[1] / counts[0])) if min(counts) > 0 else np.nan


def modulation_index(phase, amplitude):
    bins = np.clip(((phase + np.pi) * 18 / (2 * np.pi)).astype(int), 0, 17)
    values = np.array([amplitude[bins == i].mean() if np.any(bins == i) else 0
                       for i in range(18)])
    p = np.maximum(values, 1e-15)
    p /= p.sum()
    return float((np.log(18) + np.sum(p * np.log(p))) / np.log(18))


def single_window(x, prefix):
    out = {}

    def put(family, name, value):
        out[f"{prefix}__{family}__{name}"] = float(value)

    f, p = signal.welch(x, FS, nperseg=min(250, x.shape[-1]),
                        noverlap=0, axis=-1)
    band_power = {}
    total = np.trapezoid(p[:, (f >= 1) & (f <= 40)], f[(f >= 1) & (f <= 40)], axis=-1)
    for name, (low, high) in BANDS.items():
        keep = (f >= low) & (f <= high)
        bp = np.trapezoid(p[:, keep], f[keep], axis=-1)
        band_power[name] = bp
        for roi, idx in ROI.items():
            put("power", f"{roi}_{name}_logabs", np.log10(np.mean(bp[idx]) + 1e-12))
            put("power", f"{roi}_{name}_relative", np.mean(bp[idx] / (total[idx] + 1e-12)))
    for roi, idx in ROI.items():
        put("ratio", roi, np.log((np.mean(band_power["alpha"][idx]) + 1e-12) /
                                (np.mean(band_power["theta"][idx]) + 1e-12)))
        a = x[idx].mean(axis=0)
        sd = np.std(a)
        d1, d2 = np.diff(a), np.diff(a, n=2)
        mobility = np.std(d1) / max(sd, 1e-12)
        put("hjorth", f"{roi}_activity", np.var(a))
        put("hjorth", f"{roi}_mobility", mobility)
        put("hjorth", f"{roi}_complexity",
            (np.std(d2) / max(np.std(d1), 1e-12)) / max(mobility, 1e-12))
        put("hfd", roi, higuchi(a))
        for order in (3, 5):
            put("pe", f"{roi}_order{order}", permutation_entropy(a, order))
        for scale in (1, 2):
            put("sampen", f"{roi}_scale{scale}", sample_entropy(a, scale))
        for name, value in {
                "sd": sd, "ptp": np.ptp(a), "skew": stats.skew(a), "kurtosis": stats.kurtosis(a),
                "subwindow_sd_cv": np.std([np.std(v) for v in np.array_split(a, 3)]) /
                                   max(np.mean([np.std(v) for v in np.array_split(a, 3)]), 1e-12)
        }.items():
            put("variability", f"{roi}_{name}", value)
    for name in BANDS:
        for pair, (a, b) in {"F3_F4": (0, 1), "C3_C4": (3, 4), "P3_P4": (6, 7)}.items():
            put("asym", f"{pair}_{name}",
                np.log((band_power[name][a] + 1e-12) / (band_power[name][b] + 1e-12)))
    # FOOOF: fit periodic peaks separately, unlike a raw log-log linear slope.
    central_psd = p[ROI["central"]].mean(axis=0)
    fm = FOOOF(peak_width_limits=(3, 12), max_n_peaks=3, min_peak_height=0.1,
               peak_threshold=2.0, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, np.maximum(central_psd, 1e-15), [2, 40])
    if fm.has_model:
        put("aper", "central_offset", fm.aperiodic_params_[0])
        put("aper", "central_exponent", fm.aperiodic_params_[1])
        peaks = fm.peak_params_
        eligible = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)]
        peak = eligible[np.argmax(eligible[:, 1])] if len(eligible) else [np.nan] * 3
        put("paf", "central_peak_hz", peak[0])
        put("paf", "central_peak_height", peak[1])
        put("quality", "spectral_fit_r2", fm.r_squared_)
    else:
        for family, name in [("aper", "central_offset"), ("aper", "central_exponent"),
                             ("paf", "central_peak_hz"), ("paf", "central_peak_height"),
                             ("quality", "spectral_fit_r2")]:
            put(family, name, np.nan)
    analytic = {}
    for name in ("theta", "alpha", "beta", "lowgamma"):
        low, high = BANDS[name]
        filtered = signal.sosfiltfilt(
            signal.butter(3, [low, high], btype="bandpass", fs=FS, output="sos"), x, axis=-1)
        z = signal.hilbert(filtered, axis=-1)
        analytic[name] = z
        for roi, idx in ROI.items():
            amp = np.abs(z[idx])
            put("variability", f"{roi}_{name}_envelope_cv",
                np.mean(np.std(amp, axis=-1) / np.maximum(np.mean(amp, axis=-1), 1e-12)))
        for pair, (a, b) in PAIRS.items():
            phase_difference = np.angle(z[a]) - np.angle(z[b])
            imaginary = np.imag(z[a] * np.conjugate(z[b]))
            put("plv", f"{pair}_{name}", np.abs(np.mean(np.exp(1j * phase_difference))))
            put("lag", f"{pair}_{name}_pli", abs(np.mean(np.sign(np.sin(phase_difference)))))
            put("lag", f"{pair}_{name}_wpli",
                abs(np.mean(imaginary)) / max(np.mean(abs(imaginary)), 1e-12))
            put("env", f"{pair}_{name}", np.corrcoef(np.abs(z[a]), np.abs(z[b]))[0, 1])
            fw, pxy = signal.csd(x[a], x[b], FS, nperseg=125, noverlap=62)
            _, pxx = signal.welch(x[a], FS, nperseg=125, noverlap=62)
            _, pyy = signal.welch(x[b], FS, nperseg=125, noverlap=62)
            keep = (fw >= low) & (fw <= high)
            coherency = pxy / np.sqrt(np.maximum(pxx * pyy, 1e-24))
            put("coh", f"{pair}_{name}", np.mean(np.abs(coherency[keep]) ** 2))
            put("lag", f"{pair}_{name}_imaginary_coherency",
                np.mean(np.abs(np.imag(coherency[keep]))))
        if prefix == "pre":
            for roi, idx in ROI.items():
                zmean = z[idx].mean(axis=0)
                put("pre", f"{roi}_{name}_end_phase_sin", np.sin(np.angle(zmean[-1])))
                put("pre", f"{roi}_{name}_end_phase_cos", np.cos(np.angle(zmean[-1])))
                put("pre", f"{roi}_{name}_power", np.mean(band_power[name][idx]))
    for phase_name, amp_name in [("theta", "beta"), ("alpha", "lowgamma")]:
        for roi, idx in ROI.items():
            phase = np.angle(analytic[phase_name][idx].mean(axis=0))
            amplitude = np.abs(analytic[amp_name][idx]).mean(axis=0)
            put("pac", f"{roi}_{phase_name}_{amp_name}", modulation_index(phase, amplitude))
    for roi, idx in ROI.items():
        a = x[idx].mean(axis=0)
        put("quality", f"{roi}_log_total_power", np.log10(np.mean(total[idx]) + 1e-12))
        put("quality", f"{roi}_line_length", np.mean(np.abs(np.diff(a))))
    return out, band_power


def extract(pre, post):
    before, pre_power = single_window(pre, "pre")
    after, post_power = single_window(post, "post")
    out = {**before, **after}
    for roi, idx in ROI.items():
        for band in ("alpha", "beta"):
            out[f"contrast__erd__{roi}_{band}"] = (
                np.mean(post_power[band][idx]) - np.mean(pre_power[band][idx])) / (
                    np.mean(pre_power[band][idx]) + 1e-12)
        a = post[idx].mean(axis=0)
        pre_noise = np.std(pre[idx].mean(axis=0))
        time = 0.05 + np.arange(len(a)) / FS
        for component, (start, end, direction) in {
                "N1_like": (0.08, 0.18, -1), "N2_like": (0.18, 0.40, -1),
                "P2_like": (0.40, 0.75, 1)}.items():
            keep = (time >= start) & (time <= end)
            b = a[keep]
            loc = np.argmax(b * direction)
            out[f"post__erp__{roi}_{component}_amplitude_uv"] = b[loc]
            out[f"post__morph__{roi}_{component}_latency_s"] = time[keep][loc]
        for j, values in enumerate(np.array_split(a, 6)):
            out[f"post__morph__{roi}_bin{j}_mean_uv"] = values.mean()
        out[f"contrast__quality__{roi}_peak_baseline_sd_ratio"] = np.max(abs(a)) / max(pre_noise, 1e-12)
    central = post[ROI["central"]]
    covariance = np.cov(central)
    return out, covariance.astype(np.float32)


def family(name):
    return name.split("__")[1]


def feature_set(names, which, omit=None):
    if which == "power_only":
        mask = np.array([family(n) == "power" for n in names])
    elif which == "prestimulus_only":
        mask = np.array([n.startswith("pre__") for n in names])
    else:
        mask = np.ones(len(names), bool)
    if omit:
        mask &= np.array([family(n) != omit for n in names])
    return mask
