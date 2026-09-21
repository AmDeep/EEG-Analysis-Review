"""
Download raw BIDS EEG data for one or more of the Zhao et al. laser-evoked-potential
(LEP) experiments hosted on OpenNeuro (Hu Li lab, Scientific Data 2025 mega-study).

Uses the `openneuro-py` package, which pulls directly from OpenNeuro's S3 mirror
(no account/API key needed for public datasets).

Usage:
    python download_openneuro.py --datasets ds005284 ds005285 ds005473 --out data/raw/openneuro

Each dataset ID maps to one experiment of the mega-study:
    ds005284 (26 subj, Biosemi)  ds005285 (29 subj, ANT)     ds005289 (39 subj, BrainProducts)
    ds005286 (30 subj, ANT)      ds005291 (65 subj, ANT)     ds005293 (95 subj, BrainProducts)
    ds005292 (142 subj, Biosemi) ds005280 (223 subj, BrainProducts) ds005473 (29 subj, BrainProducts)
"""
import argparse
from pathlib import Path

KNOWN_DATASETS = {
    "ds005284": "Experiment 1, 26 subjects, Biosemi",
    "ds005285": "Experiment 2, 29 subjects, ANT",
    "ds005289": "Experiment 3, 39 subjects, BrainProducts",
    "ds005286": "Experiment 4, 30 subjects, ANT",
    "ds005291": "Experiment 5, 65 subjects, ANT",
    "ds005293": "Experiment 6, 95 subjects, BrainProducts",
    "ds005292": "Experiment 7, 142 subjects, Biosemi",
    "ds005280": "Experiment 8, 223 subjects, BrainProducts",
    "ds005473": "Experiment 9, 29 subjects, BrainProducts",
}


def download(dataset_id: str, out_dir: Path, tag: str = "1.0.0") -> Path:
    import openneuro

    if dataset_id not in KNOWN_DATASETS:
        raise ValueError(
            f"{dataset_id} isn't one of the known Zhao mega-study IDs: {list(KNOWN_DATASETS)}. "
            "If it's a different dataset entirely that's fine, just double check the accession number."
        )
    target = out_dir / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_id} ({KNOWN_DATASETS[dataset_id]}) -> {target}")
    openneuro.download(dataset=dataset_id, target_dir=str(target), tag=tag)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, help="OpenNeuro accession numbers, e.g. ds005284 ds005285")
    ap.add_argument("--out", type=str, default="data/raw/openneuro")
    args = ap.parse_args()

    out_dir = Path(args.out)
    for ds in args.datasets:
        try:
            download(ds, out_dir)
        except Exception as e:
            print(f"FAILED to download {ds}: {e}")
            print("If your environment can't reach OpenNeuro's S3 bucket directly, try instead:")
            print(f"  aws s3 sync --no-sign-request s3://openneuro.org/{ds} {out_dir/ds}")


if __name__ == "__main__":
    main()
