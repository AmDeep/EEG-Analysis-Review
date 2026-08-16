"""
EDA script for 4 experiments from the Zhao et al. (2025) 678-subject LEP mega-dataset.

Covers the non-intervention nociception experiments (largest, cleanest subset):

  - Experiment 1 -> ds005284  (N=26,  Biosemi (62-ch, 1024 Hz), left hand)   - music analgesia
  - Experiment 2 -> ds005285  (N=29,  ANT Neuro (32-ch, 1000 Hz), left hand) - VR+TENS analgesia
  - Experiment 3 -> ds005289  (N=39,  Brain Products (64-ch, 1000 Hz), left hand) - movement-induced hypoalgesia (hand-shaking)
  - Experiment 4 -> ds005286  (N=30,  ANT Neuro (32-ch, 1000 Hz), right hand) - sustained drumming-rhythm analgesia
  - Experiment 5 -> ds005291  (N=65,  ANT Neuro (32-ch, 1000 Hz), right hand) - acute drumming-rhythm analgesia
  - Experiment 6 -> ds005293  (N=95,  Brain Products (64-ch, 1000 Hz), both hands) - LEP encodes intensity
  - Experiment 7 -> ds005292  (N=142, Biosemi (62-ch, 1024 Hz), left hand) - pain discriminability
  - Experiment 8 -> ds005280  (N=223, Brain Products (64-ch, 1000 Hz), left hand) - pain discriminability, largest cohort
  - Experiment 9 -> ds005473  (N=29,  Brain Products (64-ch, 1000 Hz), left hand except sub-013=right) - Aδ/C-fiber LEPs

Design:
  Stage 0.5 (always runs, cheap): diagnostic pass. Raw BIDS events.tsv in these datasets
    only contains onset/duration/sample/value -- NOT the pain rating or stimulus intensity.
    Per the paper's methods, those live inside the preprocessed derivative .set files'
    epoch metadata (EEG.epoch struct in EEGLAB). This stage fetches events.json sidecars
    (to decode what 'value' actually means), checks for any stray *beh.tsv files, and
    downloads ONE example derivative .set file per dataset to print what fields exist in
    its epoch metadata -- so you know exactly where/how to pull ratings before building
    the full extraction pipeline.

  Stage 1 (always runs, cheap): pulls participants.tsv + *_events.tsv directly from
    OpenNeuro's public S3 bucket (no auth needed) for every subject in every dataset.
    Demographic EDA always works. Behavioral (rating/intensity) EDA will currently show
    up empty/None until Stage 0.5's findings are used to point it at the right source
    (see run_stage1 -> eda_behavior).

  Stage 1.5 (test-scale by default via MAX_SUBJECTS_FOR_LABELS): extracts the real
    per-trial labels (rating, laser_power) from derivatives/rerefer/*.set via EEG.event.
    Falls back to h5py for MATLAB v7.3/HDF5-format .set files that scipy.io.loadmat
    can't read. Followed by diagnose_zero_ratings(), which checks whether rating==0
    trials look like real "no pain" reports or an init/placeholder artifact.

  Stage 2 (optional, heavier): downloads raw EEG + channels.tsv for a configurable number
    of example subjects per dataset and does signal-level sanity checks (channel counts,
    sampling rate, a quick Cz grand-average ERP). Off by default -- set RUN_SIGNAL_EDA=True.

Requirements:
  pip install boto3 pandas matplotlib numpy scipy
  # needed for MATLAB v7.3/HDF5 .set files (fallback path in label extraction):
  pip install h5py
  # only needed for Stage 2:
  pip install mne mne-bids

Run:
  python lep_eda.py
"""

import io
import json
import re
import warnings
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

warnings.filterwarnings("ignore")

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
OUT_DIR = Path("./lep_eda_output")
OUT_DIR.mkdir(exist_ok=True, parents=True)

RUN_SIGNAL_EDA = False          # set True to also pull raw EEG for a few subjects
N_SUBJECTS_FOR_SIGNAL_EDA = 3   # per dataset, only used if RUN_SIGNAL_EDA=True

LABEL_SUBFOLDER = "rerefer"      # derivatives subfolder that carries rating/laser_power per event
MAX_SUBJECTS_FOR_LABELS = None      # set an int (e.g. 5) to test quickly; None = all subjects

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


# --------------------------------------------------------------------------
# S3 helpers (OpenNeuro datasets are public, unsigned requests work)
# --------------------------------------------------------------------------

def list_keys(prefix):
    """List every object key under a prefix in the OpenNeuro bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def fetch_bytes(key):
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def fetch_text(key):
    return fetch_bytes(key).decode("utf-8", errors="replace")


def find_latest_version_prefix(accession):
    """
    OpenNeuro stores versions as ds00XXXX/ (root usually points at latest) or
    ds00XXXX/versions/X.Y.Z/. We try the simple root prefix first since that's
    what OpenNeuro's bucket layout uses for the current snapshot.
    """
    root_keys = list_keys(f"{accession}/")
    if root_keys:
        return f"{accession}/"
    raise RuntimeError(f"Could not find any objects under prefix {accession}/")


# --------------------------------------------------------------------------
# Stage 0.5: figure out WHERE ratings/intensity actually live
#   (raw events.tsv only has onset/duration/sample/value -- ratings and
#   stimulus intensity are documented in the paper as living inside the
#   derivative .set files' epoch metadata, not in plain BIDS events.tsv)
# --------------------------------------------------------------------------

def inspect_sidecars_and_beh(accession, prefix, all_keys):
    """Fetch task-level *_events.json sidecars and look for any beh.tsv files,
    which sometimes hold per-trial behavioral data outside of events.tsv."""
    print(f"\n  -- Sidecar / behavioral file check for {accession} --")

    json_keys = sorted(set(k for k in all_keys if k.endswith("_events.json")))
    if json_keys:
        # events.json is often identical across subjects (one task-level file),
        # so just show a couple of distinct ones.
        seen_content = set()
        for jk in json_keys:
            try:
                content = fetch_text(jk)
            except Exception as e:
                print(f"    [!] Could not fetch {jk}: {e}")
                continue
            if content in seen_content:
                continue
            seen_content.add(content)
            print(f"    events.json -> {jk}")
            try:
                parsed = json.loads(content)
                print(json.dumps(parsed, indent=2)[:1500])
            except Exception:
                print(content[:1500])
            if len(seen_content) >= 2:
                break
    else:
        print("    No *_events.json sidecar found.")

    beh_keys = [k for k in all_keys if "beh" in Path(k).name.lower()]
    if beh_keys:
        print(f"    Found {len(beh_keys)} file(s) with 'beh' in the name, e.g.:")
        for k in beh_keys[:5]:
            print(f"      {k}")
    else:
        print("    No files with 'beh' in the name found.")


def _print_struct_fields(struct_obj, label, keyword_filter=None):
    """Print every field on a scipy-loaded MATLAB struct (mat_struct object),
    optionally only those matching keywords (e.g. 'rating','intensity','level')."""
    fields = [f for f in dir(struct_obj) if not f.startswith("_")]
    if keyword_filter:
        hits = [f for f in fields if any(kw in f.lower() for kw in keyword_filter)]
        print(f"    {label} fields matching {keyword_filter}: {hits if hits else 'NONE'}")
    print(f"    {label} all fields: {fields}")
    for f in fields:
        try:
            val = getattr(struct_obj, f)
            print(f"      {f}: {val!r}"[:200])
        except Exception:
            pass


def _load_set_flexibly(local_set):
    """Load a .set file whether it was saved with a nested 'EEG' struct or
    with fields flattened directly at the top level (both occur across these
    datasets). Returns a dict-like/struct-like object to inspect, or None.

    Raises NotImplementedError if the file is MATLAB v7.3/HDF5 format, which
    scipy.io.loadmat cannot read -- callers should fall back to
    _load_set_with_h5py_fallback() in that case."""
    from scipy.io import loadmat

    mat = loadmat(str(local_set), struct_as_record=False, squeeze_me=True)
    if "EEG" in mat:
        return mat["EEG"], "EEG (nested)"
    # flattened case: fields like 'epoch', 'event', 'nbchan' sit at top level
    if "epoch" in mat or "event" in mat:
        class FlatStruct:
            pass
        flat = FlatStruct()
        for k, v in mat.items():
            if not k.startswith("__"):
                setattr(flat, k, v)
        return flat, "top-level (flattened) struct"
    print(f"    Could not find 'EEG' or 'epoch'/'event' among keys: {list(mat.keys())}")
    return None, None


def _load_set_with_h5py_fallback(local_set):
    """Try scipy.io.loadmat first; fall back to h5py for MATLAB v7.3/HDF5 files.
    Returns (eeg_struct_or_none, how_str_or_none).

    NOTE: when the h5py fallback fires, EEG.event/EEG.epoch come back as raw
    h5py Groups/Datasets of object references, not the mat_struct objects the
    scipy path produces. getattr(ev, field, None) further down the pipeline
    will NOT work against these -- they need to be dereferenced one field/one
    event at a time. This fallback exists so a v7.3 file is at least *loaded*
    (and reported) instead of silently skipped; if you see "h5py (v7.3/HDF5)"
    show up in practice, the event-field extraction loop needs a dedicated
    h5py-aware branch before that dataset's labels can actually be pulled."""
    try:
        eeg, how = _load_set_flexibly(local_set)
        if eeg is not None:
            return eeg, how
    except NotImplementedError:
        pass  # v7.3 format, fall through to h5py
    except Exception:
        raise  # genuine load error, let caller's except handle it

    try:
        import h5py
        f = h5py.File(local_set, "r")
        if "EEG" in f:
            class H5Struct:
                pass
            eeg = H5Struct()
            eeg._h5file = f
            eeg._h5group = f["EEG"]
            # exposed as raw h5py groups/datasets -- see NOTE above, these are
            # NOT yet compatible with the scipy-style getattr(ev, field) reads
            eeg.event = f["EEG"].get("event", None)
            eeg.epoch = f["EEG"].get("epoch", None)
            return eeg, "h5py (v7.3/HDF5)"
        return None, None
    except Exception:
        return None, None


def inspect_one_set_file(target_key, local_dir, all_keys):
    print(f"    Using: {target_key}")
    local_set = local_dir / Path(target_key).name
    local_set.write_bytes(fetch_bytes(target_key))

    fdt_key = target_key.rsplit(".", 1)[0] + ".fdt"
    if fdt_key in all_keys:
        (local_dir / Path(fdt_key).name).write_bytes(fetch_bytes(fdt_key))

    try:
        eeg, how = _load_set_flexibly(local_set)
    except NotImplementedError:
        eeg, how = None, None
    except Exception as e:
        print(f"    scipy.io.loadmat failed ({e}); trying h5py (MATLAB v7.3 format)...")
        eeg, how = None, None

    if eeg is not None:
        print(f"    Loaded via: {how}")
        keywords = ["rating", "intens", "level", "nrs", "pain", "joule", "power"]

        epoch = getattr(eeg, "epoch", None)
        if epoch is not None:
            has_len = hasattr(epoch, "__len__") and not np.isscalar(epoch)
            if has_len and len(epoch) == 0:
                print("    'epoch' field present but empty (likely pre-epoching / continuous data stage).")
            else:
                epoch0 = epoch[0] if has_len else epoch
                _print_struct_fields(epoch0, "EEG.epoch[0]", keyword_filter=keywords)
        else:
            print("    No 'epoch' field found on this struct.")

        event = getattr(eeg, "event", None)
        if event is not None:
            has_len = hasattr(event, "__len__") and not np.isscalar(event)
            if has_len and len(event) == 0:
                print("    'event' field present but empty.")
            else:
                event0 = event[0] if has_len else event
                _print_struct_fields(event0, "EEG.event[0]", keyword_filter=keywords)
        else:
            print("    No 'event' field found on this struct.")
        return

    # v7.3 / HDF5 fallback
    try:
        import h5py

        with h5py.File(local_set, "r") as f:
            print(f"    Opened as HDF5/v7.3. Top-level keys: {list(f.keys())}")
            if "EEG" in f:
                eeg_group = f["EEG"]
                print(f"    EEG group keys: {list(eeg_group.keys())}")
    except Exception as e:
        print(f"    [!] Could not open as HDF5 either: {e}")
        print("    Recommend opening this .set file manually in EEGLAB (MATLAB) to inspect fields.")


def inspect_derivative_epochs(accession, prefix, all_keys):
    """Check every derivatives/<subfolder>/ present (mark_ica, rerefer, etc.)
    since rating info may only appear at a later preprocessing stage than
    mark_ica."""
    print(f"\n  -- Derivative .set inspection for {accession} --")

    deriv_set_keys = [k for k in all_keys if k.endswith(".set") and "derivatives" in k.lower()]
    if not deriv_set_keys:
        deriv_set_keys = [k for k in all_keys if k.endswith(".set")]
    if not deriv_set_keys:
        print("    No .set files found anywhere in this dataset.")
        return

    # group by the derivatives subfolder name, e.g. derivatives/mark_ica/..., derivatives/rerefer/...
    subfolders = {}
    for k in deriv_set_keys:
        parts = k.split("/")
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            sub = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        else:
            sub = "unknown"
        subfolders.setdefault(sub, []).append(k)

    local_dir = OUT_DIR / "derivative_check" / accession
    local_dir.mkdir(parents=True, exist_ok=True)

    for sub, keys in subfolders.items():
        print(f"\n    [derivatives/{sub}] ({len(keys)} .set files)")
        inspect_one_set_file(sorted(keys)[0], local_dir, all_keys)


def run_stage0_5():
    """Diagnostic pass: find out where ratings/intensity actually live before
    building the full feature-extraction pipeline."""
    for label, accession in DATASETS.items():
        print(f"\n=== Stage 0.5 diagnostics: {label} ({accession}) ===")
        prefix = find_latest_version_prefix(accession)
        all_keys = list_keys(prefix)

        inspect_sidecars_and_beh(accession, prefix, all_keys)
        inspect_derivative_epochs(accession, prefix, all_keys)


def extract_labels_for_dataset(accession, prefix, all_keys, subfolder=LABEL_SUBFOLDER,
                                max_subjects=MAX_SUBJECTS_FOR_LABELS):
    """Pull EEG.event (rating, laser_power, type, latency, etc.) for every
    subject's derivative .set file. This is the real per-trial label source
    -- raw BIDS events.tsv does not carry rating/laser_power for these datasets."""
    set_keys = sorted(k for k in all_keys if f"/derivatives/{subfolder}/" in k and k.endswith(".set"))
    if not set_keys:
        print(f"  [!] No .set files found under derivatives/{subfolder}/ for {accession}")
        return pd.DataFrame()

    if max_subjects:
        set_keys = set_keys[:max_subjects]

    local_dir = OUT_DIR / "labels_raw" / accession
    local_dir.mkdir(parents=True, exist_ok=True)

    label_fields = ["type", "laser_power", "rating", "latency", "epoch", "urevent", "code", "duration"]
    rows = []
    failed_subjects = []

    for i, key in enumerate(set_keys, 1):
        m = re.search(r"(sub-[A-Za-z0-9]+)", key)
        subj = m.group(1) if m else "unknown"
        print(f"  [{i}/{len(set_keys)}] {accession} {subj} ...", end=" ", flush=True)

        local_set = local_dir / Path(key).name
        local_fdt = None
        try:
            local_set.write_bytes(fetch_bytes(key))
            fdt_key = key.rsplit(".", 1)[0] + ".fdt"
            if fdt_key in all_keys:
                local_fdt = local_dir / Path(fdt_key).name
                local_fdt.write_bytes(fetch_bytes(fdt_key))

            eeg, how = _load_set_with_h5py_fallback(local_set)
            if eeg is None:
                print("could not load (including h5py fallback), skipping")
                failed_subjects.append((subj, "load failed"))
                continue

            event = getattr(eeg, "event", None)
            if event is None:
                print("no 'event' field, skipping")
                failed_subjects.append((subj, "no event field"))
                continue

            has_len = hasattr(event, "__len__") and not np.isscalar(event)
            events_list = list(event) if has_len else [event]

            n_added = 0
            for ev in events_list:
                row = {"subject": subj, "dataset": accession}
                for f in label_fields:
                    row[f] = getattr(ev, f, None)
                rows.append(row)
                n_added += 1
            if n_added == 0:
                failed_subjects.append((subj, "0 events extracted"))
            print(f"{n_added} events (via {how})")

        except Exception as e:
            print(f"failed: {e}")
            failed_subjects.append((subj, str(e)))
        finally:
            # clean up raw files as we go -- these datasets are large in aggregate
            try:
                if local_set.exists():
                    local_set.unlink()
                if local_fdt is not None and local_fdt.exists():
                    local_fdt.unlink()
            except Exception:
                pass

    if failed_subjects:
        print(f"\n  [!] {len(failed_subjects)}/{len(set_keys)} subjects failed for {accession}:")
        for subj, reason in failed_subjects:
            print(f"      {subj}: {reason}")

    return pd.DataFrame(rows)


def run_stage1_5_labels():
    """Build the real per-trial label table (rating + laser_power) across all
    datasets in DATASETS. This replaces reliance on raw events.tsv for
    behavioral EDA."""
    frames = []
    for label, accession in DATASETS.items():
        print(f"\n=== Extracting labels: {label} ({accession}) ===")
        prefix = find_latest_version_prefix(accession)
        all_keys = list_keys(prefix)
        df = extract_labels_for_dataset(accession, prefix, all_keys)
        if not df.empty:
            frames.append(df)

    labels_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    labels_df.to_csv(OUT_DIR / "labels_combined.csv", index=False)
    print(f"\nSaved combined labels -> {OUT_DIR/'labels_combined.csv'}")
    return labels_df


# --------------------------------------------------------------------------
# Stage 1: lightweight participants + events EDA
# --------------------------------------------------------------------------

def load_participants(accession, prefix):
    key = f"{prefix}participants.tsv"
    try:
        df = pd.read_csv(io.StringIO(fetch_text(key)), sep="\t")
    except Exception as e:
        print(f"  [!] Could not load participants.tsv for {accession}: {e}")
        return pd.DataFrame()
    df["dataset"] = accession
    return df


def load_all_events(accession, prefix):
    """Pull every *_events.tsv for every subject/session in the dataset."""
    all_keys = list_keys(prefix)
    events_keys = [k for k in all_keys if k.endswith("_events.tsv")]
    frames = []
    for key in events_keys:
        try:
            df = pd.read_csv(io.StringIO(fetch_text(key)), sep="\t")
        except Exception as e:
            print(f"  [!] Skipping {key}: {e}")
            continue
        m = re.search(r"(sub-[A-Za-z0-9]+)", key)
        df["subject"] = m.group(1) if m else "unknown"
        df["dataset"] = accession
        df["source_file"] = key
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def guess_column(columns, keywords):
    """Find the first column whose name contains any of the given keywords."""
    for col in columns:
        low = col.lower()
        if any(kw in low for kw in keywords):
            return col
    return None


def run_stage1():
    all_participants = []
    all_events = []

    for label, accession in DATASETS.items():
        print(f"\n=== {label} ({accession}) ===")
        prefix = find_latest_version_prefix(accession)

        participants = load_participants(accession, prefix)
        if not participants.empty:
            print(f"  participants.tsv: {len(participants)} rows")
            all_participants.append(participants)

        events = load_all_events(accession, prefix)
        if not events.empty:
            n_subj = events["subject"].nunique()
            print(f"  events: {len(events)} rows across {n_subj} subjects")
            all_events.append(events)
        else:
            print("  [!] No events files found")

    participants_df = (
        pd.concat(all_participants, ignore_index=True) if all_participants else pd.DataFrame()
    )
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    participants_df.to_csv(OUT_DIR / "participants_combined.csv", index=False)
    events_df.to_csv(OUT_DIR / "events_combined.csv", index=False)

    print(f"\nSaved combined participants -> {OUT_DIR/'participants_combined.csv'}")
    print(f"Saved combined events       -> {OUT_DIR/'events_combined.csv'}")

    return participants_df, events_df


def eda_demographics(participants_df):
    if participants_df.empty:
        print("No participant data to summarize.")
        return

    print("\n--- Demographics ---")
    print(participants_df.groupby("dataset").size().rename("n_subjects"))

    age_col = guess_column(participants_df.columns, ["age"])
    sex_col = guess_column(participants_df.columns, ["sex", "gender"])

    if age_col:
        print("\nAge summary by dataset:")
        print(participants_df.groupby("dataset")[age_col].describe()[["count", "mean", "std", "min", "max"]])

        fig, ax = plt.subplots(figsize=(8, 5))
        for name, grp in participants_df.groupby("dataset"):
            ax.hist(grp[age_col].dropna(), bins=15, alpha=0.5, label=name)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Age distribution by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "age_distribution.png", dpi=150)
        plt.close(fig)

    if sex_col:
        print("\nSex/gender counts by dataset:")
        print(participants_df.groupby(["dataset", sex_col]).size().unstack(fill_value=0))

        fig, ax = plt.subplots(figsize=(8, 5))
        ct = pd.crosstab(participants_df["dataset"], participants_df[sex_col])
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Sex/gender by dataset")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "sex_by_dataset.png", dpi=150)
        plt.close(fig)


def eda_behavior(labels_df):
    """Now takes the labels_df built from EEG.event (rating + laser_power),
    not the raw events.tsv (which doesn't carry these fields for this dataset)."""
    if labels_df.empty:
        print("No label data to summarize.")
        return

    print("\n--- Trial counts (from extracted labels) ---")
    trial_counts = labels_df.groupby(["dataset", "subject"]).size().reset_index(name="n_trials")
    print(trial_counts.groupby("dataset")["n_trials"].describe()[["count", "mean", "std", "min", "max"]])

    fig, ax = plt.subplots(figsize=(8, 5))
    trial_counts.boxplot(column="n_trials", by="dataset", ax=ax)
    ax.set_title("Trials per subject by dataset")
    ax.set_ylabel("Trial count")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trials_per_subject.png", dpi=150)
    plt.close(fig)

    rating_col = "rating" if "rating" in labels_df.columns else guess_column(labels_df.columns, ["rating", "pain", "nrs"])
    intensity_col = "laser_power" if "laser_power" in labels_df.columns else guess_column(labels_df.columns, ["intensity", "energy", "power", "joule"])

    print(f"\nUsing rating column: {rating_col}")
    print(f"Using intensity column: {intensity_col}")

    if rating_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, grp in labels_df.groupby("dataset"):
            vals = pd.to_numeric(grp[rating_col], errors="coerce").dropna()
            if len(vals):
                ax.hist(vals, bins=range(0, 12), alpha=0.5, label=name, density=True)
        ax.set_xlabel("Pain rating (NRS)")
        ax.set_ylabel("Density")
        ax.set_title("Pain rating distribution by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "pain_rating_distribution.png", dpi=150)
        plt.close(fig)

    if intensity_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, grp in labels_df.groupby("dataset"):
            vals = pd.to_numeric(grp[intensity_col], errors="coerce").dropna()
            if len(vals):
                ax.hist(vals, bins=20, alpha=0.5, label=name, density=True)
        ax.set_xlabel("Laser power (J)")
        ax.set_ylabel("Density")
        ax.set_title("Stimulus intensity distribution by dataset")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "intensity_distribution.png", dpi=150)
        plt.close(fig)

    if rating_col and intensity_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, grp in labels_df.groupby("dataset"):
            x = pd.to_numeric(grp[intensity_col], errors="coerce")
            y = pd.to_numeric(grp[rating_col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum():
                ax.scatter(x[mask], y[mask], s=4, alpha=0.2, label=name)
        ax.set_xlabel("Stimulus intensity (J)")
        ax.set_ylabel("Pain rating")
        ax.set_title("Stimulus intensity vs. pain rating")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "intensity_vs_rating.png", dpi=150)
        plt.close(fig)

        print("\nCorrelation (intensity vs rating) by dataset:")
        for name, grp in labels_df.groupby("dataset"):
            x = pd.to_numeric(grp[intensity_col], errors="coerce")
            y = pd.to_numeric(grp[rating_col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() > 2:
                r = np.corrcoef(x[mask], y[mask])[0, 1]
                print(f"  {name}: r = {r:.3f}  (n={mask.sum()})")


def diagnose_zero_ratings(labels_df):
    """Check whether rating==0 trials look like real data or an init/placeholder
    artifact. Prompted by a preview sighting of rating:0 on a 3.5J "Low intensity"
    trial -- checks whether rating==0 clusters at a specific (e.g. first) epoch
    index per subject, and whether it's uncorrelated with stimulus intensity
    (both would point to a placeholder rather than a genuine zero-pain report)."""
    if labels_df.empty or "rating" not in labels_df.columns:
        print("No label data to check.")
        return

    df = labels_df.copy()
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["power_num"] = pd.to_numeric(df["laser_power"], errors="coerce")

    zeros = df[df["rating_num"] == 0]
    print(f"\n--- rating==0 diagnostic ---")
    print(f"Total trials: {len(df)}, rating==0 trials: {len(zeros)} ({100*len(zeros)/max(len(df),1):.1f}%)")

    if zeros.empty:
        print("No zero ratings found.")
        return

    # does rating==0 cluster at a specific epoch/trial index (e.g. epoch 1)?
    if "epoch" in zeros.columns:
        epoch_counts = zeros["epoch"].value_counts().sort_index()
        print("\nEpoch index distribution of rating==0 trials (top 10):")
        print(epoch_counts.head(10))
        frac_at_epoch1 = (zeros["epoch"] == zeros["epoch"].min()).mean()
        print(f"Fraction of zero-ratings at the minimum epoch index: {frac_at_epoch1:.2f}")

    # is rating==0 correlated with low intensity, or spread across the full range?
    print("\nlaser_power distribution for rating==0 trials:")
    print(zeros["power_num"].describe())
    print("\nlaser_power distribution for all trials (for comparison):")
    print(df["power_num"].describe())

    # per-subject: does every subject have exactly one rating==0 (init artifact)
    # or is it scattered/variable (possibly real)?
    per_subj = zeros.groupby(["dataset", "subject"]).size()
    print(f"\nPer-subject rating==0 count -- mean: {per_subj.mean():.2f}, "
          f"subjects with exactly 1: {(per_subj == 1).sum()}/{len(per_subj)}")


# --------------------------------------------------------------------------
# Stage 2 (optional): signal-level EDA on a handful of subjects
# --------------------------------------------------------------------------

def run_stage2():
    try:
        import mne
        from mne.io import read_raw_brainvision
    except ImportError:
        print("\n[Stage 2 skipped] Install mne to run signal-level EDA: pip install mne mne-bids")
        return

    for label, accession in DATASETS.items():
        print(f"\n=== Signal EDA: {label} ({accession}) ===")
        prefix = find_latest_version_prefix(accession)
        all_keys = list_keys(prefix)

        subj_dirs = sorted(set(
            re.search(r"(sub-[A-Za-z0-9]+)/", k).group(1)
            for k in all_keys if re.search(r"(sub-[A-Za-z0-9]+)/", k)
        ))[:N_SUBJECTS_FOR_SIGNAL_EDA]

        cz_epochs = []

        for subj in subj_dirs:
            vhdr_keys = [k for k in all_keys if subj in k and k.endswith(".vhdr")]
            if not vhdr_keys:
                print(f"  [!] No .vhdr found for {subj} (may not be BrainVision format)")
                continue

            local_dir = OUT_DIR / "raw" / accession / subj
            local_dir.mkdir(parents=True, exist_ok=True)

            vhdr_key = vhdr_keys[0]
            base = vhdr_key.rsplit(".", 1)[0]
            needed_ext = [".vhdr", ".eeg", ".vmrk"]
            local_vhdr = None

            for ext in needed_ext:
                k = base + ext
                if k in all_keys:
                    local_path = local_dir / Path(k).name
                    local_path.write_bytes(fetch_bytes(k))
                    if ext == ".vhdr":
                        local_vhdr = local_path

            if local_vhdr is None:
                continue

            try:
                raw = read_raw_brainvision(local_vhdr, preload=True, verbose=False)
                print(f"  {subj}: {len(raw.ch_names)} channels, sfreq={raw.info['sfreq']} Hz")
                if "Cz" in raw.ch_names:
                    cz_epochs.append((subj, raw))
            except Exception as e:
                print(f"  [!] Could not read {subj}: {e}")

        if cz_epochs:
            fig, ax = plt.subplots(figsize=(8, 5))
            for subj, raw in cz_epochs:
                data = raw.get_data(picks="Cz")[0]
                t = raw.times
                window = min(len(t), int(30 * raw.info["sfreq"]))  # first 30s preview
                ax.plot(t[:window], data[:window] * 1e6, alpha=0.6, label=subj)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (uV)")
            ax.set_title(f"Cz raw trace preview - {accession}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"cz_preview_{accession}.png", dpi=150)
            plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    run_stage0_5()

    participants_df, _raw_events_df = run_stage1()
    eda_demographics(participants_df)

    # Real per-trial labels (rating + laser_power) come from the derivative
    # .set files' EEG.event struct, not raw BIDS events.tsv.
    labels_df = run_stage1_5_labels()
    eda_behavior(labels_df)
    diagnose_zero_ratings(labels_df)

    if RUN_SIGNAL_EDA:
        run_stage2()

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")