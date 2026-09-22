"""DWT (db4, level 5) statistical features for epoched EEG."""
import argparse
import numpy as np
import pandas as pd
import pywt

DEFAULT_ROI = ("Fz", "Cz", "Pz", "C3", "C4")


def band_table(sfreq, level=5):
    return [
        (f"D{k}", sfreq / 2 ** (k + 1), sfreq / 2 ** k)
        for k in range(1, level + 1)
    ] + [(f"A{level}", 0.0, sfreq / 2 ** (level + 1))]


def dwt_features(X, ch_names, roi=DEFAULT_ROI, wavelet="db4", level=5):
    """X: trials x channels x time. Returns float32 features and names."""
    ch_names = list(ch_names)
    idx = [ch_names.index(c) for c in roi if c in ch_names]
    if not idx:
        raise ValueError(f"None of {roi} found in ch_names")

    max_level = pywt.dwt_max_level(X.shape[-1], pywt.Wavelet(wavelet).dec_len)
    if level > max_level:
        raise ValueError(
            f"level={level} is too deep for n_times={X.shape[-1]} "
            f"with {wavelet}; max_level={max_level}"
        )

    coeffs = pywt.wavedec(X[:, idx, :], wavelet, level=level, axis=-1)
    bands = [f"A{level}"] + [f"D{k}" for k in range(level, 0, -1)]
    total = sum((c ** 2).sum(-1) for c in coeffs) + 1e-12

    feats, names = [], []
    for band, c in zip(bands, coeffs):
        energy = (c ** 2).sum(-1)
        stats = {
            "logE": np.log(energy + 1e-12),
            "relE": energy / total,
            "std": c.std(-1),
            "mabs": np.abs(c).mean(-1),
            "zcr": (np.diff(np.signbit(c), axis=-1) != 0).mean(-1),
        }
        for stat, values in stats.items():
            for j, ci in enumerate(idx):
                feats.append(values[:, j])
                names.append(f"{ch_names[ci]}_{band}_{stat}")

    return np.stack(feats, axis=1).astype(np.float32), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="nurovo_epochs.npz")
    ap.add_argument("--output", required=True, help="nurovo_dwt.npz")
    ap.add_argument("--roi", nargs="+", default=list(DEFAULT_ROI))
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=5)
    a = ap.parse_args()

    z = np.load(a.input, allow_pickle=True)
    X = z["X"]
    ch_names = z["ch_names"].tolist()
    F, names = dwt_features(X, ch_names, tuple(a.roi), a.wavelet, a.level)

    np.savez_compressed(
        a.output,
        F=F,
        feature_names=np.array(names),
        dataset=z["dataset"],
        global_subject=z["global_subject"],
        epoch=z["epoch"],
        power=z["power"],
        sfreq=z["sfreq"],
    )
    print(f"saved {a.output}: F{F.shape}")


if __name__ == "__main__":
    main()
