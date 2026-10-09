"""Audited public download; no example/synthetic signals substituted for EEG."""
import concurrent.futures
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT.parent / "generated-artifacts/blind-stimulus-eeg"
SEED = 20261009


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "Independent-EEG-research/1.0"}), timeout=90) as response:
        return response.read()


def listing(url):
    rows = []
    while url:
        body = json.loads(fetch(url))
        rows.extend(body["data"])
        url = body.get("links", {}).get("next")
    return rows


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exclusion_audit():
    import csv
    features = list(csv.DictReader((ROOT / "feature_prediction_table.csv").open(encoding="utf-8-sig")))
    ids = {i.strip() for r in features for i in r["evidence_ids"].split(";") if i.strip()}
    records = json.loads((ROOT / "research/all_records.json").read_text())
    excluded = [r for r in records if r["source_id"] in ids]
    body = json.dumps(excluded).lower()
    protocol = json.loads((HERE / "protocol.json").read_text())
    for key, d in protocol["datasets"].items():
        assert d["project"] not in body, f"Excluded source accession: {key}"
        if d.get("paper_doi"):
            assert d["paper_doi"].lower() not in body, f"Excluded source paper: {key}"
    result = {
        "feature_catalogue_sha256": sha(ROOT / "feature_prediction_table.csv"),
        "feature_source_ids": sorted(ids),
        "excluded_original_source_records": excluded,
        "chosen_cohorts": protocol["datasets"],
        "overlap_check": "No selected OSF accession or identified heat paper DOI occurs in feature-cited records. Heat collected in Hamburg; VR study from Hewitt/Bath. No documented shared cohort. Reviews themselves supply no newly collected EEG cohort.",
        "limitations": "No globally identifying participant linkage; no claim of independently proving every historical cohort identity.",
    }
    (HERE / "dataset_exclusion_audit.json").write_text(json.dumps(result, indent=2))


def download(row):
    directory = RAW / row["dataset"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / row["name"]
    try:
        if not path.exists() or path.stat().st_size != row["expected_bytes"]:
            temp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(urllib.request.Request(
                    row["url"], headers={"User-Agent": "Independent-EEG-research/1.0"}),
                    timeout=240) as src, temp.open("wb") as dest:
                shutil.copyfileobj(src, dest, length=4 * 1024 * 1024)
            if temp.stat().st_size != row["expected_bytes"]:
                raise ValueError("File size mismatch; incomplete signal not used")
            temp.replace(path)
        return {**row, "status": "downloaded", "path": str(path.relative_to(ROOT.parent)),
                "sha256": sha(path)}
    except Exception as exc:
        return {**row, "status": "failed", "error": repr(exc)}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    exclusion_audit()
    frozen = HERE / "protocol_frozen.json"
    current_sha = sha(HERE / "protocol.json")
    if frozen.exists():
        assert json.loads(frozen.read_text())["protocol_sha256"] == current_sha, "Protocol changed after freeze"
    else:
        frozen.write_text(json.dumps({"protocol_sha256": current_sha,
                                     "frozen_before_prediction": True,
                                     "seed": SEED}, indent=2))
    urls = {
        "heat_f2mua": "https://api.osf.io/v2/nodes/f2mua/files/osfstorage/?page[size]=100",
        "shock_w25k4": "https://api.osf.io/v2/nodes/w25k4/files/osfstorage/67c1e610fa9a53b2b1556a99/?page[size]=100",
    }
    selected = []
    for key, url in urls.items():
        files = listing(url)
        candidates = []
        for v in files:
            a = v["attributes"]
            name = a["name"]
            pattern = r"RawData_PART\d+\.mat$" if key.startswith("heat") else r"P\d+_cleaned_2702\.set$"
            if a["kind"] == "file" and re.fullmatch(pattern, name):
                candidates.append({"dataset": key, "name": name, "url": v["links"]["download"],
                                   "expected_bytes": a["size"],
                                   "selection_hash": hashlib.sha256(f"{SEED}:{key}:{name}".encode()).hexdigest()})
        chosen = sorted(candidates, key=lambda r: r["selection_hash"])[:10]
        if len(chosen) < 8:
            raise ValueError(f"Not enough independently sampled participants in {key}")
        selected.extend(chosen)
        (HERE / f"{key}_source_listing.json").write_text(json.dumps(candidates, indent=2))
    (HERE / "selected_files.json").write_text(json.dumps(selected, indent=2))
    manifest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(download, selected):
            manifest.append(row)
            (HERE / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(row["dataset"], row["name"], row["status"], flush=True)
    print("Downloaded participants:", sum(r["status"] == "downloaded" for r in manifest), flush=True)


if __name__ == "__main__":
    main()
