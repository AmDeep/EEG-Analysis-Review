# EEG Feature Extraction Pipeline

Turns any epoched EEG file(s) into a per-epoch feature table (N2/P2,
gamma power, alpha/beta ERD, PSD band powers, inter-site PLV, and
antropy entropy/complexity measures). Originally built around the
Zhao et al. (2025) laser-evoked-potential mega-dataset; generalized here to
run on any local `.set` (+ `.fdt`), `.fif`, `.edf`, or `.bdf` file, and
wired into a GitHub Actions CI/CD pipeline.

## Repo layout

```
extract_features.py              # the pipeline itself (CLI)
requirements.txt
Dockerfile
data/                             # drop EEG files here to trigger auto-extraction on push
tests/test_smoke.py               # CI smoke test (synthetic data, no network)
.github/workflows/extract-features.yml
```

## Run it locally

```bash
pip install -r requirements.txt

# single file
python extract_features.py --input path/to/sub-01_eeg.set --output-dir ./out

# a whole directory, searched recursively for .set/.fif/.edf/.bdf
python extract_features.py --input ./some_data_dir --output-dir ./out

# original OpenNeuro S3 workflow (LEP mega-dataset), unchanged
python extract_features.py --s3 ds005280 --output-dir ./out
```

Output: `./out/features_combined.csv`, one row per epoch.

## Run it in Docker

```bash
docker build -t eeg-features .
docker run -v /path/to/your/eeg/files:/data -v /path/to/output:/out eeg-features
```
(Dockerfile is written but not build-tested in this environment - no Docker
daemon was available here. Sanity-check the build once before relying on it.)

## CI/CD (GitHub Actions)

`.github/workflows/extract-features.yml` does two things:

1. **CI** - every push and PR runs `tests/test_smoke.py`, which generates
   synthetic epoched data and checks the pipeline produces the expected
   feature columns with no exceptions. This catches breakage (bad imports,
   shape mismatches, a feature function silently failing) before anything
   gets published.
2. **CD** - once tests pass, feature extraction runs and the resulting CSV
   is uploaded as a downloadable workflow artifact (Actions tab → run →
   Artifacts). Three ways to trigger it:
   - **Automatic**: commit new files under `data/` on `main` - extraction
     runs against `data/` and publishes the CSV.
   - **Manual, local path**: Actions tab → "EEG Feature Extraction" → "Run
     workflow" → set `input_path` to any path in the repo (defaults to
     `data/`).
   - **Manual, OpenNeuro S3**: same manual trigger, set `s3_accession` to an
     accession like `ds005280` instead (reproduces the original LEP
     workflow, ignores `input_path`).

To actually use the automatic trigger, `.set` files pushed into `data/`
need their companion `.fdt` alongside them if the EEGLAB file was saved in
"twofiles" mode - the loader looks for `<name>.fdt` next to `<name>.set`.
Large binary EEG files in a git repo get expensive fast; consider Git LFS
or point `input_path`/`s3_accession` at external storage instead of
committing raw data if files are more than a few MB each.

## What each feature means / caveats that matter

See the docstrings and inline comments in `extract_features.py` - they're
carried over from the original script and still apply, most importantly:

- N2/P2 windows (150–350 ms / 300–550 ms) are literature defaults for
  laser-evoked potentials at the vertex; sanity-check against a
  grand-average ERP before trusting amplitudes/latencies as final for a
  new dataset.
- Gamma band power is capped at `min(100 Hz, sfreq/2 - 5)` per file.
- Single-trial PLV measures phase-locking **across time within one trial**
  between two electrodes - not the more standard across-trials PLV.
- **New in this version**: if you feed in continuous (non-epoched)
  `.fif`/`.edf`/`.bdf` data, it gets cut into 1-second fixed-length
  pseudo-epochs with no real stimulus onset. In that case only the
  PSD/antropy features are meaningful - N2/P2/gamma/ERD/PLV assume a
  stimulus at t=0 and will not mean what their names imply. The output CSV
  flags this per-row via the `pseudo_epoched` column; check it before using
  those columns.
- `.fif`/`.edf`/`.bdf` inputs don't carry `rating`/`laser_power` labels
  (that convention is specific to the LEP `.set` files' `EEG.event`
  struct) - those columns are simply absent for those rows. Join your own
  labels onto the output using the `dataset`/`subject`/`epoch` key columns.
