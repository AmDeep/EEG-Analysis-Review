"""Supplement the reviewed catalogue with transparent, bibliographic-only screening.

These rows NEVER qualify as verified public EEG data merely because an article
mentions a public dataset. They retain actual Europe PMC metadata/XML evidence.
"""
import concurrent.futures
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research/sources/bibliographic-screen"


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "EEG-literature-access-review/1.0"}), timeout=45) as response:
        return response.read()


def text(node):
    return " ".join(" ".join(node.itertext()).split())


def candidate(row):
    pmcid = row.get("pmcid")
    if not pmcid:
        return None
    api = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        raw = fetch(api)
        tree = ET.fromstring(raw)
        actual_title = tree.find(".//article-title")
        if actual_title is None:
            return None
        title = text(actual_title)
        if not re.search(r"\bEEG\b|electroencephalo", title, flags=re.I):
            return None
        body = tree.find("body")
        abstract_node = tree.find(".//abstract")
        abstract = text(abstract_node) if abstract_node is not None else ""
        paragraphs = [text(p) for p in tree.findall(".//p")]
        data_statements = [p for p in paragraphs if re.search(
            r"data availab|data.*publicly availab|public.*data|dataset.*availab|"
            r"EEGDenoiseNet|PhysioNet|OpenNeuro|BNCI|BCI Competition", p, flags=re.I)]
        relevant = [p for p in paragraphs if re.search(
            r"signal.to.noise|SNR|variance|Higuchi|permutation entropy|phase.lag|"
            r"phase.lock|spectral|band.?power|denois|test.retest", p, flags=re.I)]
        # Store limited source evidence, not an unsupported synthetic summary.
        evidence = {
            "source_api": api, "metadata": row, "full_text_api_returned_xml": True,
            "article_title": title, "abstract": abstract,
            "data_related_source_paragraphs": data_statements[:8],
            "method_related_source_paragraphs": relevant[:6],
            "review_level": "Automated bibliographic/full-text access screen; dataset access NOT independently verified",
        }
        path = OUT / f"{pmcid}.json"
        path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
        doi = row.get("doi", "")
        tokens = title + " " + abstract
        families = []
        matches = [
            ("SNR/denoising", r"denois|signal.to.noise|\bSNR\b|noise|artifact"),
            ("Reliability/variance", r"reliab|test.retest|variance|stability"),
            ("Entropy/complexity", r"entropy|complexity|Higuchi|fractal|Lempel"),
            ("Spectral/aperiodic", r"spectral|band.?power|aperiodic|alpha|beta"),
            ("Phase/connectivity", r"phase|connectivity|coherence|synchron"),
        ]
        for name, pattern in matches:
            if re.search(pattern, tokens, flags=re.I):
                families.append(name)
        if not families:
            return None
        public_link = ""
        for sec in tree.findall(".//sec"):
            heading = sec.find("title")
            if heading is not None and re.search("data availab|data record", text(heading), re.I):
                for link in sec.findall(".//ext-link"):
                    url = link.get("{http://www.w3.org/1999/xlink}href", "")
                    if re.search(r"osf\.io|physionet|openneuro|figshare|zenodo|mendeley|github", url):
                        public_link = url
                        break
        source_method = relevant[0] if relevant else abstract
        # The quoted fragment is intentionally short and source-specific.
        fragment = " ".join(source_method.split()[:23])
        types = " ".join(row.get("pubTypeList", {}).get("pubType", []))
        kind = "review" if "review" in types.lower() else "paper"
        if "preprint" in types.lower():
            kind = "preprint"
        return {
            "title": title, "authors": row.get("authorString", ""),
            "year": row.get("pubYear", ""), "doi": doi, "item_type": kind,
            "evidence_scope": "direct pain" if re.search(r"\bpain\b", title, re.I) else "transferable EEG",
            "feature_family": "; ".join(families),
            "feature_definition": "Screening-level source excerpt: “" + fragment + "”. Detailed estimator not independently extracted.",
            "prediction_rationale": "Transfer hypothesis only: assess the source-defined signal/feature method for reliability or incremental prediction before any pain application; no demonstrated pain improvement inferred.",
            "verified_result_metric": "Not extracted in the automated bibliographic/access screen.",
            "prediction_target": "Source study target; not independently extracted. Pain benefit not assumed.",
            "validation_design": "Not independently reviewed in this supplementary screen.",
            "article_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            "open_access_status": "Open full text verified via Europe PMC XML API; OA metadata recorded.",
            "data_url": public_link, "data_access_status": "unverified",
            "data_evidence": ("Source data-related text: " + " ".join(data_statements[0].split()[:23])
                              if data_statements else "No specific data-access statement extracted; do not count as public data."),
            "relevance_limitations": "AUTOMATED SUPPLEMENTARY SCREEN, not a fully reviewed eligible public-data paper. Dataset URLs/access, validation and quantitative outcomes need independent review. General EEG results are not pain evidence.",
            "source_evidence_paths": [str(path.relative_to(ROOT.parent))],
            "source_urls": [api, f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"],
        }
    except Exception as exc:
        print(pmcid, "excluded:", type(exc).__name__, str(exc)[:100], flush=True)
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    inputs = [Path("/tmp/eeg_literature_signal_candidates.json"),
              Path("/tmp/eeg_literature_feature_candidates.json")]
    if all(source.exists() for source in inputs):
        for source in inputs:
            candidates.extend(json.loads(source.read_text())["resultList"]["result"])
    else:
        # The downloadable packet includes the actual retrieved metadata,
        # making the source pool usable outside the original /tmp environment.
        for source in sorted(OUT.glob("*.json")):
            candidates.append(json.loads(source.read_text())["metadata"])
    seen = set()
    reviewed = set()
    for file in (ROOT / "research").glob("*.json"):
        if file.name == "bibliographic-screen.json":
            continue
        report = json.loads(file.read_text())
        if isinstance(report, dict):
            for r in report.get("items", []):
                reviewed.add(str(r.get("doi", "")).lower())
    selected = []
    for row in candidates:
        key = row.get("pmcid")
        if not key or key in seen or row.get("doi", "").lower() in reviewed:
            continue
        seen.add(key)
        selected.append(row)
    # A finite candidate pool; no endless quota search or repeated failed fetches.
    selected = selected[:105]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        items = [r for r in pool.map(candidate, selected) if r is not None]
    output = {"items": items,
              "conflicts": ["Public-dataset mentions do not establish accessible study data."],
              "gaps": ["These supplementary rows are bibliographically/access screened only; no public-data qualification is claimed."],
              "sources": [u for r in items for u in r["source_urls"]]}
    (ROOT / "research/bibliographic-screen.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print("Supplementary screen:", len(items), "actual open-text records; public data remains unverified.", flush=True)


if __name__ == "__main__":
    main()
