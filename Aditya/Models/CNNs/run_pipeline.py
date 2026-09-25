"""Convenience launcher.

1) build_epochs.py must first produce nurovo_epochs.npz.
2) This launcher builds DWT features and then runs EEGNet CV.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True, help="Existing nurovo_epochs.npz")
    ap.add_argument("--outdir", default="nurovo_run")
    ap.add_argument("--skip-dwt", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dwt = outdir / "nurovo_dwt.npz"
    if not a.skip_dwt:
        run([
            sys.executable, "dwt_features.py",
            "--input", a.epochs,
            "--output", str(dwt),
        ])

    run([
        sys.executable, "train_eegnet.py",
        "--input", a.epochs,
        "--outdir", str(outdir / "eegnet"),
    ])

    print(f"\nFinished. Outputs are under: {outdir}")


if __name__ == "__main__":
    main()
