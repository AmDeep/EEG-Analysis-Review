"""
Download the Tiemann et al. (2018) laser-pain EEG dataset from its OSF project.
https://osf.io/bsv86/  ("Distinct patterns of brain activity mediate perceptual and
motor and autonomic responses to noxious stimuli", Nature Communications)

Uses `osfclient`. OSF project storage for public projects doesn't require login,
but osfclient still needs a project id (the part after osf.io/).

Usage:
    python download_tiemann.py --out data/raw/tiemann
"""
import argparse
import subprocess
import sys
from pathlib import Path

OSF_PROJECT_ID = "bsv86"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw/tiemann")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # osfclient's CLI clones the whole project tree into --project
    cmd = [sys.executable, "-m", "osfclient", "clone", "-p", OSF_PROJECT_ID, str(out_dir)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            "osfclient failed. Fallback: browse https://osf.io/bsv86/files/ manually and download "
            "the EEG + behavioral files into", out_dir,
            "\nOr try the OSF API directly: https://api.osf.io/v2/nodes/bsv86/files/osfstorage/",
        )


if __name__ == "__main__":
    main()
