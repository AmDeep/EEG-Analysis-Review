"""Assemble evidence-backed literature records; never fabricate quota fillers."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import xlsxwriter

ROOT = Path(__file__).resolve().parent
AREAS = ("pain-evidence", "phase-coupling", "spectral-features",
         "complexity-envelope", "signal-data", "gap-public-methods", "gap-pain-data",
         "bibliographic-screen")
FIELDS = (
    "source_id", "title", "authors", "year", "doi", "item_type", "evidence_scope", "review_level",
    "feature_family", "feature_definition", "prediction_rationale",
    "verified_result_metric", "prediction_target", "validation_design",
    "article_url", "open_access_status", "data_url", "data_access_status",
    "data_evidence", "relevance_limitations", "access_assessment",
    "source_evidence_paths", "source_urls",
)


def canonical_doi(value):
    match = re.search(r"10\.\d{4,9}/[^\s]+", str(value or ""), flags=re.I)
    return match.group(0).lower().rstrip(" .;,") if match else ""


def access(row):
    text = str(row.get("open_access_status", "")).lower()
    public_text = (bool(row.get("article_url")) and
                   any(w in text for w in ("open", "accessible", "free", "verified", "public"))
                   and not any(w in text for w in ("unverified", "paywall", "abstract only",
                                                  "unknown", "not verified", "not open",
                                                  "no full text")))
    data = str(row.get("data_access_status", "")).lower()
    has_data = str(row.get("data_url", "")).startswith(("https://", "http://"))
    if public_text and has_data and data == "public eeg verified":
        return "Open text + public EEG verified"
    if public_text and has_data and data == "public derived data only":
        return "Open text + public derived data only"
    if not public_text:
        return "Publication access not verified"
    if data in ("not public", "on request"):
        return "Does not meet public-data requirement"
    return "Public data not verified"


def read_items():
    records = []
    reports = {}
    for area in AREAS:
        path = ROOT / "research" / f"{area}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        reports[area] = report
        for item in report["items"]:
            item = dict(item)
            item["research_area"] = area
            item["doi"] = canonical_doi(item.get("doi"))
            # Preserve human-review detail while canonicalizing equivalent
            # labels. Never infer availability from an automated dataset mention.
            status = str(item.get("data_access_status", ""))
            lower_status = status.lower()
            if lower_status == "public raw eeg verified" or lower_status.startswith((
                    "public eeg recordings verified on openneuro;",
                    "public openneuro eeg datasets verified by")):
                item["data_access_status"] = "public EEG verified"
                item["data_evidence"] = str(item.get("data_evidence", "")) + " Access-review detail: " + status
            elif lower_status.startswith("public derived eeg verified:"):
                item["data_access_status"] = "public derived data only"
                item["data_evidence"] = str(item.get("data_evidence", "")) + " Access-review detail: " + status
            item["access_assessment"] = access(item)
            records.append(item)
    unique = {}
    for row in records:
        # Conflicting raw/derived descriptions of the consumer PhysioPain EEG
        # must not promote 1-Hz exported band features into raw voltage EEG.
        if "physiopain" in row["title"].lower():
            row["data_access_status"] = "public derived data only"
            row["data_evidence"] = (
                str(row.get("data_evidence", "")) +
                " Consolidated assessment: accessible EEG exports include 1-Hz band/derived values; raw voltage EEG was not verified.")
            row["access_assessment"] = access(row)
        key = "doi:" + row["doi"] if row["doi"] else "title:" + re.sub(
            r"[^a-z0-9]", "", row["title"].lower())
        if key not in unique:
            unique[key] = row
        else:
            current = unique[key]
            for field in ("source_urls", "source_evidence_paths"):
                current[field] = sorted(set(current.get(field, []) + row.get(field, [])))
            if ("public EEG verified" in row["access_assessment"]
                    and "public EEG verified" not in current["access_assessment"]):
                unique[key] = {**row, "source_urls": current["source_urls"],
                               "source_evidence_paths": current["source_evidence_paths"]}
    items = sorted(unique.values(), key=lambda r: (
        r.get("evidence_scope") != "direct pain",
        r["access_assessment"] != "Open text + public EEG verified",
        str(r.get("year", "")), r["title"].lower()))
    for i, row in enumerate(items, 1):
        row["source_id"] = f"R{i:03}"
        row["review_level"] = (
            "Supplementary automated bibliographic/access screen; study/data not independently reviewed"
            if row.get("research_area") == "bibliographic-screen"
            else "Study/access evidence reviewed; see individual limitations")
    return items, reports


def flat(row, fields):
    values = []
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, list):
            value = "; ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        values.append("" if value is None else str(value))
    return values


def write_csv(name, fields, rows):
    with (ROOT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(flat(row, fields) for row in rows)


def worksheet(workbook, title, fields, rows):
    ws = workbook.add_worksheet(title)
    heading = workbook.add_format({"bold": True, "bg_color": "#153A56",
                                   "font_color": "white", "text_wrap": True})
    body = workbook.add_format({"valign": "top", "text_wrap": True, "font_size": 10})
    link = workbook.add_format({"valign": "top", "text_wrap": True, "font_color": "#1263A0",
                               "underline": True, "font_size": 10})
    ws.write_row(0, 0, fields, heading)
    ws.set_row(0, 42)
    for i, row in enumerate(rows, 1):
        values = flat(row, fields)
        for j, value in enumerate(values):
            if fields[j] in ("article_url", "data_url") and value.startswith(("http://", "https://")):
                # Multiple data links remain plain text rather than a misleading single hyperlink.
                if " " not in value and ";" not in value:
                    ws.write_url(i, j, value, link, value)
                else:
                    ws.write_string(i, j, value, body)
            elif fields[j] == "doi" and value.startswith("10."):
                ws.write_url(i, j, "https://doi.org/" + value, link, value)
            else:
                ws.write_string(i, j, value, body)
        ws.set_row(i, 120 if len(fields) > 10 else 92)
    for j, field in enumerate(fields):
        width = 12 if field in ("source_id", "year") else 24
        if field in ("title", "feature_definition", "prediction_rationale", "data_evidence",
                     "relevance_limitations", "finding", "decision", "limitation"):
            width = 48
        if field in ("article_url", "data_url", "source_evidence_paths", "source_urls"):
            width = 38
        ws.set_column(j, j, width)
    ws.freeze_panes(1, min(2, len(fields)))
    if rows:
        ws.add_table(0, 0, len(rows), len(fields) - 1,
                     {"name": re.sub("[^A-Za-z0-9]", "", title) + "Table",
                      "columns": [{"header": f} for f in fields],
                      "style": "Table Style Medium 2"})
    ws.hide_gridlines(2)
    ws.set_landscape()
    ws.fit_to_pages(1, 0)
    ws.repeat_rows(0)
    return ws


def feature_rows(items):
    plans = [
        ("Evoked N1/N2/P2 amplitudes", "N1|N2|P2|evoked|single-trial ERP",
         "Single-trial component amplitudes and N2–P2 peak-to-peak after verified event alignment.",
         "Test whether trial-specific evoked response amplitude predicts reported pain, separating stimulus energy and habituation.",
         "Needs events, pre-stimulus baseline and suitable montage; five-second resting summaries cannot recover trial components."),
        ("Evoked latencies/morphology", "latency|morphology|wavelet|dispersion",
         "Component latencies and template distortion/dispersion estimates.",
         "Reduce amplitude-estimation errors due to latency jitter; evaluate incremental held-out pain prediction.",
         "Requires aligned trials; waveform estimation must not use held-out target information."),
        ("Alpha/beta ERD", "ERD|event-related desynchron",
         "Baseline-normalized event-related alpha/beta power change, with stimulation-side contrasts.",
         "May distinguish stimulus-related sensorimotor modulation from subjective-pain-related activity.",
         "Requires pre/post stimulus timing; not interchangeable with resting alpha/beta power."),
        ("High-gamma response", "high.gamma|60.?100|70.?90|Gamma.Band Oscillations in the Primary|pain. and touch.related neuronal gamma",
         "Event-related high-gamma power, peak power and peak latency in a source-verified band/window.",
         "Candidate trial-intensity feature; assess muscle artifacts, stimulus energy and habituation.",
         "Cannot recover 60–100 or 70–90 Hz from 128 Hz, 1–40 Hz processed EEG; raw data and higher-rate processing required."),
        ("Pre-stimulus power/phase", "prestimulus|pre-stimulus|before stimulus",
         "Pre-stimulus bandpower, power variability and phase encoded as sine/cosine.",
         "Model baseline-dependent trial-response variation rather than treating it as unstructured noise.",
         "Needs pre-stimulus windows and accurate event timing; association need not be incremental out-of-sample prediction."),
        ("Absolute/relative spectral power", "spectral|band.?power|power spectrum",
         "Integrated band PSD or normalized power, with explicit bandwidth and units.",
         "Represent amplitude and spectral redistribution; compare absolute and relative forms under identical validation.",
         "Artifact/reference sensitivity and compositional dependence; low-gamma 30–40 Hz is not high gamma."),
        ("Peak alpha frequency", "peak alpha|alpha frequency|PAF|alpha slowing",
         "Distinguish actual spectral-peak estimation from PSD-weighted alpha centroid.",
         "Test baseline sensitivity or pain associations; do not assume universal alpha slowing.",
         "A 9–11 Hz centroid is not a full individual alpha peak; account for no-peak cases and aperiodic slope."),
        ("Aperiodic slope/offset", "aperiodic|1/f|log.?log|spectral parameter",
         "Separate periodic peaks from aperiodic spectral slope and offset using a source-defined estimator.",
         "Prevent broadband/aperiodic changes being mislabeled as oscillatory bandpower changes.",
         "Current descriptive log-log regression is not a peak-corrected estimator; age/arousal/artifacts can confound."),
        ("Hemispheric power asymmetry", "asymmetr|contralateral|ipsilateral",
         "Explicit left-minus-right or log-ratio power, with region and stimulation side recorded.",
         "Candidate lateralized feature; assess whether it adds information beyond total power.",
         "Frontal asymmetry requires frontal electrodes; C3/C4 asymmetry is not a substitute for Fp1/Fp2."),
        ("Alpha/theta power ratio", "alpha.?theta|theta.?alpha|power ratio",
         "Prespecified alpha/theta ratio or log-ratio with explicit band definitions.",
         "Test a relative inhibition/arousal hypothesis, not an established universal nociceptive score.",
         "Ratio instability and correlated/compositional predictors; determine any benefit inside training folds."),
        ("Magnitude-squared coherence", "coherence|coherency",
         "Frequency-specific magnitude-squared cross-spectrum normalization.",
         "Candidate network feature alongside local power; assess added held-out information.",
         "Volume conduction/common reference can inflate coupling; depends on window length and spectral averaging."),
        ("PLV/phase synchrony", "phase-lock|phase locking|PLV|synchron",
         "Absolute circular mean of inter-channel phase differences.",
         "Represent phase coordination separately from power; superiority must be shown for the actual task.",
         "Can reflect zero-lag mixing and finite-sample bias; central pairs do not recover frontal/occipital networks."),
        ("PLI/wPLI/imaginary coherency", "wPLI|phase lag|imaginary",
         "Lag-sensitive cross-channel phase/coherency measures with explicit estimator.",
         "Reduce some instantaneous-mixing effects when testing connectivity predictors.",
         "Not proof of causal connectivity; short windows and low signal strength reduce reliability."),
        ("Phase–amplitude coupling", "phase.amplitude|PAC|cross.frequency",
         "Coupling of slow-rhythm phase to faster-rhythm amplitude, with frequency grid and surrogate controls.",
         "Candidate cross-frequency feature; separately test raw coupling and between-region asymmetry.",
         "Non-sinusoidal waveforms/transients can create spurious PAC; high-frequency components may exceed current preprocessing."),
        ("Envelope correlation", "envelope|amplitude coupling",
         "Correlation of band-limited Hilbert amplitude envelopes; state whether orthogonalization is used.",
         "Model slow cofluctuation of regional amplitudes beyond average power.",
         "Mixing/reference effects and autocorrelation; correlation alone is not explained pain variance."),
        ("CVUE/Piqβ", "CVUE|Piq|upper envelope",
         "Source-specific upper-envelope coefficient-of-variation pipeline, including its beta band, windows and aggregation.",
         "Pilot hypothesis for pain-related amplitude variability; requires independent validation before thresholds are adopted.",
         "The 2024 pilot raw data are not public; generic Hilbert-envelope CV need not reproduce the published Piqβ algorithm."),
        ("Within-window amplitude variability", "variability|coefficient of variation|amplitude distribution",
         "Amplitude dispersion, envelope variation and subwindow-power variation at specified scales.",
         "Capture temporal variation discarded by a mean power statistic.",
         "Window dependence, artifacts and vigilance can explain variation; temporal dependence must be respected."),
        ("Higuchi fractal dimension", "Higuchi|fractal dimension|HFD",
         "Log-length versus inverse-scale slope with specified kmax and sampling rate.",
         "Candidate complexity feature to compare against spectral/amplitude baselines.",
         "Filtering, sample rate and window length affect estimates; classification accuracy is not variance explained."),
        ("Permutation entropy", "permutation entropy|ordinal",
         "Ordinal-pattern entropy with prespecified order/delay, normalization and tie rule.",
         "Represent temporal predictability; assess incremental information beyond bandpower.",
         "No universal theta-band importance; feature weights depend on model, correlated features and validation."),
        ("Sample/multiscale entropy", "sample entropy|multiscale|approximate entropy",
         "Pattern recurrence/entropy across explicitly specified scales and tolerance.",
         "Candidate irregularity feature, with estimation feasibility checked for short windows.",
         "Short-record bias, nonstationarity and tolerance dependence; no assumed clinical pain validation."),
        ("Hjorth parameters", "Hjorth",
         "Activity, mobility and complexity defined from variance and successive differences.",
         "Low-cost amplitude/frequency-distribution summary for baseline prediction models.",
         "Correlated with spectral measures; per-sample mobility changes with sampling rate."),
        ("CSP/spatial log-variance", "CSP|common spatial|spatial filter",
         "Training-fitted spatial filters and component log-variance.",
         "Potential supervised trial discrimination method; its performance depends on label and validation design.",
         "CSP is label-informed, not an unsupervised state-discovery representation; fit only on training subjects."),
        ("Signal quality/SNR and variance controls", "signal.to.noise|SNR|denois|artifact|reliability|variance",
         "Source-defined noise/artifact or reliability estimates, including estimator and reference signal.",
         "Determine whether apparent predictive information is neural, stable and recoverable rather than noise/artifact.",
         "SNR improvement or signal variance preserved is not R² for pain; compare matched held-out targets."),
    ]
    rows = []
    for family, pattern, definition, rationale, limitation in plans:
        matched = [r for r in items if r.get("research_area") != "bibliographic-screen" and re.search(pattern, " ".join(
            str(r.get(f, "")) for f in ("title", "feature_family", "feature_definition")), flags=re.I)]
        direct = [r for r in matched if r.get("evidence_scope") == "direct pain"]
        selected = (direct or matched)[:6]
        rows.append({"feature": family, "definition": definition,
                     "potential_prediction_role": rationale,
                     "evidence_ids": "; ".join(r["source_id"] for r in selected),
                     "evidence_basis": "Source-linked candidate; rationale is proposed use, not a promised effect"
                         if selected else "No matching source verified in this catalogue; hypothesis only",
                     "limitation": limitation})
    return rows


def main():
    items, reports = read_items()
    if not items:
        raise ValueError("No actual research records; refuse empty/fabricated table")
    strict = [r for r in items if r["access_assessment"] == "Open text + public EEG verified"]
    derived = [r for r in items if r["access_assessment"] == "Open text + public derived data only"]
    papers = [r for r in items if r.get("item_type") == "paper"]
    strict_papers = [r for r in strict if r.get("item_type") == "paper"]
    features = feature_rows(items)
    audit_path = ROOT / "research/claim_audit.json"
    claims = json.loads(audit_path.read_text()) if audit_path.exists() else []
    write_csv("literature_table.csv", FIELDS, items)
    write_csv("verified_public_eeg_items.csv", FIELDS, strict)
    write_csv("feature_prediction_table.csv", tuple(features[0]), features)
    if claims:
        write_csv("supplied_claims_audit.csv", tuple(claims[0]), claims)
    summary = {
        "research_date": "2026-10-09", "unique_items_screened": len(items),
        "paper_rows_screened": len(papers),
        "open_text_public_eeg_items": len(strict),
        "open_text_public_eeg_original_papers": len(strict_papers),
        "open_text_public_derived_only_items": len(derived),
        "direct_pain_public_eeg_original_papers": sum(
            r.get("evidence_scope") == "direct pain" for r in strict_papers),
        "type_counts": dict(collections.Counter(r.get("item_type", "") for r in items)),
        "scope_counts": dict(collections.Counter(r.get("evidence_scope", "") for r in items)),
        "access_counts": dict(collections.Counter(r["access_assessment"] for r in items)),
        "requested_100_open_public_eeg_papers_achieved": len(strict_papers) >= 100,
        "feature_rows": len(features),
        "supplementary_automated_screen_items": sum(
            r.get("research_area") == "bibliographic-screen" for r in items),
        "fully_reviewed_screen_items": sum(
            r.get("research_area") != "bibliographic-screen" for r in items),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    sources = {}
    for row in items:
        key = row["source_id"]
        sources[key] = {"title": row["title"], "url": row["article_url"],
                        "year": row.get("year", ""), "doi": row["doi"], "tier": 1,
                        "review_level": row["review_level"],
                        "evidence_paths": row.get("source_evidence_paths", []),
                        "data_url": row.get("data_url", ""),
                        "access_assessment": row["access_assessment"]}
    (ROOT / "research/sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    (ROOT / "research/all_records.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    workbook = xlsxwriter.Workbook(str(ROOT / "eeg_research_tables.xlsx"),
                                   {"strings_to_formulas": False, "strings_to_urls": False})
    summary_rows = [{"measure": key, "value": json.dumps(v) if isinstance(v, dict) else str(v)}
                    for key, v in summary.items()]
    summary_rows += [
        {"measure": "Public EEG criterion", "value": "Open publication/resource and a verified public EEG dataset link with source evidence; not a data-on-request statement."},
        {"measure": "Metric distinction", "value": "Held-out R², adjusted/in-sample R², correlation, classification accuracy and signal variance are different quantities. No conversion or cross-study ranking."},
        {"measure": "Prediction role", "value": "Feature/prediction rationale is a proposed use unless a source-verified result explicitly supports it; no promised improvement."},
        {"measure": "Access date", "value": "2026-10-09; public landing-page/access evidence checked, but entire datasets were not downloaded."},
        {"measure": "Quota", "value": "The catalogue includes screened items that fail access requirements, clearly marked. Only eligible paper counts can satisfy the requested 100-paper quota."},
    ]
    worksheet(workbook, "Summary", ("measure", "value"), summary_rows)
    worksheet(workbook, "LiteratureCatalogue", FIELDS, items)
    worksheet(workbook, "VerifiedPublicEEG", FIELDS, strict)
    worksheet(workbook, "PublicDerivedOnly", FIELDS, derived)
    worksheet(workbook, "FeaturesForPrediction", tuple(features[0]), features)
    if claims:
        worksheet(workbook, "SuppliedClaimsAudit", tuple(claims[0]), claims)
    worksheet(workbook, "AccessNotQualified", FIELDS, [
        r for r in items if r not in strict and r not in derived])
    worksheet(workbook, "BibliographicScreen", FIELDS, [
        r for r in items if r.get("research_area") == "bibliographic-screen"])
    workbook.close()
    quota = ("The requested minimum of 100 open-access original papers with public EEG data was met."
             if len(strict_papers) >= 100 else
             f"The requested minimum of 100 open-access original papers with public EEG data was NOT established: {len(strict_papers)} were verified under that strict criterion. No inaccessible/unverified paper or dataset/software resource is counted as an eligible original paper.")
    report = f"""# EEG variance, signal strength and pain: research tables

Research/access-check date: **2026-10-09**.

The catalogue contains **{len(items)} distinct screened research items**, including **{len(papers)} paper entries**. **{len(strict)} items** have both open publication/resource access and verified public EEG data; **{len(strict_papers)} of those are research papers**, including **{summary['direct_pain_public_eeg_original_papers']} direct pain papers**. A further **{len(derived)} items** have public derived data only. {quota}

Of the catalogue entries, **{summary['fully_reviewed_screen_items']} were reviewed for study/access evidence**, while **{summary['supplementary_automated_screen_items']} are explicitly supplementary bibliographic/full-text-access screens**. The latter provide real open-paper metadata and saved source excerpts, but their datasets, metrics and validation designs have not been independently reviewed. They do not satisfy the public-data requirement.

The literature table distinguishes direct pain evidence from transferable EEG methodology. General EEG decoding, noise-reduction or reliability results are not demonstrated pain-prediction gains. Reusing an existing dataset does not create an independent cohort. Prediction rationale is a proposed use, not a guaranteed effect, and literature results do not establish an unsupervised physiological-state count.

## Files

| File | Contents |
|---|---|
| eeg_research_tables.xlsx | Filterable workbook with summary, catalogue, public-EEG subset, derived-only subset, feature table, claim audit and nonqualifying access records |
| literature_table.csv | Complete screened table: citation, paper/data links, feature, potential use, verified metric, target, validation design, access and limitations |
| verified_public_eeg_items.csv | Only rows assessed as open text/resource + verified public EEG |
| feature_prediction_table.csv | {len(features)} feature families, definitions, qualified prediction rationale, source IDs and current-pipeline limitations |
| supplied_claims_audit.csv | Checks of the supplied examples and strong claims, with primary sources and decisions |
| summary.json | Machine-readable counts; separates papers from other items and qualifying from unverified access |
| research/sources.json | Source registry keyed by the table's mechanically assigned source IDs |
| research/sources/ | Saved page/statement/snippet evidence; consult the path in each row |
| research/all_records.json | Deduplicated structured research records |
| build_tables.py | Rebuilds tables from the saved research records without inventing rows |
| screen_open_methods.py | Documents the automated supplementary screen; no dataset-access eligibility is inferred from mentions |

## Interpretation and safeguards

Explained variation in pain ratings requires a stated outcome, evaluation set and R² definition. Correlation, binary accuracy, explained variance of EEG signals and variance retained by a representation are not interchangeable. Reported source-specific figures must be interpreted in their original design; no cross-study benchmark equivalence is inferred.

The current C3/C4/Cz pipeline at 128 Hz with 1–40 Hz filtering cannot extract frontal/occipital network effects or 60–100 Hz gamma from its retained representation. Event-related amplitudes, latencies and ERD require correctly aligned events and pre-stimulus baselines. The feature table documents these requirements; this research does not modify the existing analysis.

Publication access and data access were assessed separately from saved evidence. “On request”, publicly readable results, supplementary summary tables and open code are not silently promoted to public raw EEG. Public derived data are labelled separately. A verified public dataset link documents accessibility, not independent verification of every signal file, pain rating or license condition.

## Limitations

This is an evidence-screened research catalogue, not an exhaustive systematic review or a clinical recommendation. Any strict quota shortfall is disclosed above. Entries with unverified access remain visible for follow-up but do not satisfy the requested public-data criterion. Metrics not verified from available source text are left explicitly unverified rather than fabricated.
"""
    (ROOT / "README.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
