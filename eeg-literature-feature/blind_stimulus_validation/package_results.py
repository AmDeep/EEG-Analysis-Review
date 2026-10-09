"""Build tables/report/bundle from measured notebook outputs only."""
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    required = ["accuracy_results.csv", "heldout_predictions.csv", "feature_set_gains.csv",
                "family_ablation_results.csv", "permutation_summary.csv", "verdict.json"]
    for name in required:
        assert (HERE / name).exists(), f"Missing actual result: {name}"
    nb = nbformat.read(HERE / "blind_stimulus_validation.ipynb", as_version=4)
    assert not any(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == "code")
    result = pd.read_csv(HERE / "accuracy_results.csv")
    preds = pd.read_csv(HERE / "heldout_predictions.csv")
    # Every trial is held out once per configuration; no duplicate-fold predictions.
    keys = ["evaluation", "task", "feature_set", "model", "subject", "source_trial"]
    assert not preds.duplicated(keys).any()
    audits = json.loads((HERE / "split_leakage_audit.json").read_text())
    assert all(not set(r["train_subjects"]).intersection(r["test_subjects"]) for r in audits)
    assert not (result.evaluation == "cross_dataset").any(), "Unexpected claimed independent second-cohort results"
    z = np.load(HERE / "features.npz")
    assert len(pd.read_csv(HERE / "trial_metadata.csv")) == len(z["X"])
    shutil.copyfile(HERE.parent / "feature_prediction_table.csv", HERE / "original_feature_catalogue.csv")
    summary = json.loads((HERE / "feature_summary.json").read_text())
    regression = result[result.task == "regression"].copy()
    classification = result[result.task == "classification"].copy()
    gains = pd.read_csv(HERE / "feature_set_gains.csv")
    ablations = pd.read_csv(HERE / "family_ablation_results.csv")
    workbook = {
        "Readme": pd.DataFrame([
            {"item": "Scored cohort", "value": "Independent heat OSF f2mua;10 participants,2520 trials"},
            {"item": "Target", "value": "Measured physical temperature42/46/48C;NOT subjective pain"},
            {"item": "Validation", "value": "Four-fold participant-held-out. Cross-dataset stage blocked."},
            {"item": "Features", "value": "505 cached EEG columns+2 training-only CSP=507;21/23 candidate families"},
            {"item": "Not tested", "value": "High gamma, source-specific CVUE/Piqbeta"},
            {"item": "Sampling", "value": "File SHA256(seed:dataset:name) rank before model scoring"},
            {"item": "Intervals", "value": "1000 whole-participant bootstrap resamples;conditional on fitted models"},
            {"item": "Native accuracy", "value": "Regressor output rounded to nearest protocol42/46/48C, no fitted threshold"},
            {"item": "Caveat", "value": "Retrospective,only10 people;cue/arousal/muscle effects not ruled out"},
            {"item": "Great accuracy", "value": "NOT VALIDATED: original criterion requires2 independent scored cohorts"}]),
        "Temperature regression": regression,
        "Higher-lower classification": classification,
        "Full versus power": gains,
        "Family ablations": ablations,
        "Permutation controls": pd.read_csv(HERE / "permutation_summary.csv"),
        "Dataset eligibility": pd.read_csv(HERE / "dataset_eligibility.csv"),
        "Family coverage": pd.read_csv(HERE / "feature_family_coverage.csv"),
        "Generated columns": pd.read_csv(HERE / "generated_feature_catalogue.csv"),
        "Participant results": pd.read_csv(HERE / "participant_accuracy_results.csv"),
        "Participant target counts": pd.read_csv(HERE / "participant_target_counts.csv"),
    }
    with pd.ExcelWriter(HERE / "blind_stimulus_results.xlsx", engine="xlsxwriter") as writer:
        for name, table in workbook.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(table), len(table.columns) - 1)
            ws.set_column(0, len(table.columns) - 1, 22)
    native_cols = ["feature_set", "model", "R2", "R2_CI_low", "R2_CI_high",
                   "MAE", "RMSE", "native_level_accuracy"]
    class_cols = ["feature_set", "model", "accuracy", "balanced_accuracy",
                  "balanced_accuracy_CI_low", "balanced_accuracy_CI_high", "ROC_AUC"]
    def table(frame):
        return frame.to_html(index=False, float_format=lambda v: f"{v:.4f}", escape=True)
    verdict = json.loads((HERE / "verdict.json").read_text())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Independent EEG physical-stimulus test — measured results</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;color:#18222e;max-width:1200px;margin:40px auto;padding:0 20px}}
h1,h2{{line-height:1.2}}.notice{{background:#fff2d6;padding:16px;border-left:5px solid #a46b00}}
table{{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}}td,th{{padding:8px;text-align:left;border:1px solid #d9dfe5}}
th{{background:#edf2f6}}.scroll{{overflow:auto}}code{{background:#edf2f6;padding:2px 5px}}a{{color:#155db1}}
</style></head><body><h1>Independent EEG feature test: physical stimulus intensity</h1>
<p class="notice"><strong>Partial result—not complete cross-dataset validation.</strong>
{html.escape(verdict["summary"])} This study does not establish prediction of perceived or clinical pain.</p>
<p><strong>{summary["trials"]:,} real trials;10 held-out participants;21 candidate families;507 full-model predictors.</strong>
Target:42,46 or48°C. Four-fold participant-held-out splits, training-only imputation/scaling/CSP,
no PCA and no test-set tuning. The original feature-source cohorts were excluded.</p>
<h2>Native temperature prediction</h2><p>MAE and RMSE are degrees Celsius. R²=0 is the mean-prediction reference;
negative R² is worse. Exact-level accuracy rounds predictions to the nearest of42/46/48°C.
Intervals resample entire people1000 times, conditional on the fitted models.</p>
<div class="scroll">{table(regression[native_cols])}</div>
<h2>Secondary experimental contrast</h2><p>42°C versus46/48°C is not a pain diagnosis.
Majority ordinary accuracy is66.7%; balanced chance accuracy is50%.
These temperatures include a low condition often perceived as nonpainful.</p>
<div class="scroll">{table(classification[class_cols])}</div>
<h2>Do all the collected families help?</h2><p>Compare the fixed full logistic model with power-only on exactly the same held-out trials.
Positive gains favor the full feature set; an interval containing zero is inconclusive.</p>{table(gains)}
<h2>What was generated</h2><p>505 cached columns plus2 CSP columns fitted within training folds.
High-gamma and CVUE/Piqβ were not tested. Published algorithms are not all identically replicated:
short-window connectivity/entropy/PAF and candidate component-like extrema need additional validation.</p>
<div class="scroll">{table(pd.read_csv(HERE / "feature_family_coverage.csv")[["feature","generated_columns","status"]])}</div>
<h2>Dataset audit</h2><div class="scroll">{table(pd.read_csv(HERE / "dataset_eligibility.csv"))}</div>
<h2>Limits on interpretation</h2><ul>
<li>One independent cohort is scored; multiple participants are not multiple datasets.</li>
<li>VR deposited cue epochs end about100ms after shock, too short for the prespecified features; no fake transfer result is supplied.</li>
<li>Only10 people; no globally identifying linkage to prove every historical cohort disjointness.</li>
<li>Cue expectations, arousal and peripheral artifacts can explain prediction; no causal/pain-specificity claim.</li>
<li>MAT voltage units use the stated BrainAmp microvolt assumption; short spectra limit peak-frequency estimation.</li>
<li>21 family ablations and20 permutation controls are exploratory, not independently validated selection.</li>
</ul><h2>Reproduce and inspect</h2><p>Open <code>blind_stimulus_validation.ipynb</code> for executable model code and actual outputs.
All tables and predictions, feature cache, source audits and code are in the same folder and bundle.
Raw EEG is not redistributed. Sources:
<a href="https://osf.io/f2mua/">OSF f2mua</a>,
<a href="https://elifesciences.org/articles/62809">eLife62809</a>,
<a href="https://api.osf.io/v2/wikis/xc3vs/content/">author codebook</a>.
Earlier literature tables and unsupervised results remain unchanged.</p></body></html>"""
    (HERE / "measured_results.html").write_text(page)
    file_list = sorted(p.name for p in HERE.iterdir()
                       if p.is_file() and p.suffix not in (".zip", ".pyc") and p.name != "checksums.csv")
    pd.DataFrame([{"file": n, "sha256": hashlib.sha256((HERE / n).read_bytes()).hexdigest(),
                   "bytes": (HERE / n).stat().st_size} for n in file_list]).to_csv(HERE / "checksums.csv", index=False)
    packet = HERE / "blind_stimulus_notebook_and_results.zip"
    with zipfile.ZipFile(packet, "w", zipfile.ZIP_DEFLATED) as bundle:
        for n in file_list + ["checksums.csv"]:
            bundle.write(HERE / n, arcname="blind_stimulus_validation/" + n)
    print("VERIFIED: executed notebook, no participant overlap, no fabricated cross-dataset score")
    print("Bundle bytes:", packet.stat().st_size)
    print(regression[native_cols].to_string(index=False))
    print(classification[class_cols].to_string(index=False))


if __name__ == "__main__":
    main()
