"""Generate an executable notebook with actual model code, not invented outputs."""
import ast
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent


def main():
    nb = nbformat.v4.new_notebook()
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Independently sourced EEG: physical stimulus prediction\n\n"
            "**Partial validation, not proof of great pain accuracy.** The feature-source cohorts "
            "are excluded. Ten independently sampled participants in OSF f2mua provide usable "
            "heat trials at **42/46/48°C**. Four-fold participant-held-out scores measure generalization "
            "to people, **not** independent cross-dataset transfer.\n\n"
            "The second downloaded cohort (OSF w25k4) ends about 100 ms after shock; the planned "
            "poststimulus features cannot be recovered. It is excluded, not silently converted into "
            "a successful blind dataset. A pre-score availability amendment records this limitation.\n\n"
            "Stimulus temperature is **not reported pain intensity**. These results cannot establish "
            "clinical pain prediction, causality or usefulness of every published feature estimator."),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport sys, json\nimport pandas as pd\n"
            "from IPython.display import display, Markdown, Code\n"
            "NOTEBOOK_DIR = Path.cwd()\nsys.path.insert(0, str(NOTEBOOK_DIR))\n"
            "__file__ = str(NOTEBOOK_DIR / 'evaluate.py')\n"
            "pd.set_option('display.max_rows', 100)\n"
            "pd.set_option('display.max_columns', 20)\n"
            "assert (NOTEBOOK_DIR / 'features.npz').exists(), 'Extract/download EEG first: see README.md'\n"
            "display(pd.read_csv('dataset_eligibility.csv'))"),
        nbformat.v4.new_markdown_cell(
            "## Frozen design and target provenance\n\n"
            "No model selection uses held-out performance. Source condition codes—not EEG correlations "
            "or subjective ratings—supply the target. Heat source MATLAB column 4 contains **cue codes**, "
            "not temperature. Temperature comes from the author's condition-combination codebook. "
            "Zero, 42, 46 and 48 are never introduced as fabricated observations."),
        nbformat.v4.new_code_cell(
            "display(Markdown('```json\\n' + Path('protocol_availability_amendment.json').read_text() + '\\n```'))\n"
            "display(Markdown('```json\\n' + Path('target_mapping.json').read_text() + '\\n```'))\n"
            "display(pd.read_csv('participant_target_counts.csv'))"),
        nbformat.v4.new_markdown_cell(
            "## Feature implementations and extraction code\n\n"
            "The cache has 505 numeric EEG columns. Two supervised CSP columns are generated only "
            "inside training folds, giving 507 predictors for the full candidate model. "
            "Twenty-one of 23 families have candidate implementations. High gamma and CVUE/Piqβ "
            "are **not tested**, not approximated under their original names. FOOOF separates periodic "
            "peaks from the aperiodic component; an absent alpha peak remains missing.\n\n"
            "Short-window PAF, connectivity, entropy and coupling estimates are exploratory. "
            "Fixed component-like extrema are not replicated WF+MLRd. Causal common filtering and "
            "fixed trigger interpolation use no target labels. Heat voltage units assume the "
            "BrainAmp microvolt convention because the deposited MAT lacks a unit field."),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv('feature_family_coverage.csv')[['feature', 'generated_columns', 'status', 'implementation_note']])\n"
            "display(pd.read_csv('generated_feature_catalogue.csv').head(20))\n"
            "print('Full column catalogue is generated_feature_catalogue.csv')"),
        nbformat.v4.new_code_cell(
            "for filename in ['readers.py', 'features.py', 'build_features.py']:\n"
            "    display(Markdown('### ' + filename))\n"
            "    display(Code(filename=filename, language='python'))"),
        nbformat.v4.new_markdown_cell(
            "## Actual fitted-model code\n\n"
            "The cells below contain the executable source of evaluate.py. Imputation, scaling and CSP "
            "see training subjects only. There is no PCA, clustering, threshold tuning or feature "
            "selection on held-out scores. Dummy, fixed linear and fixed Extra Trees models are all "
            "reported; none is declared best to fit the desired outcome."),
    ]
    source = (HERE / "evaluate.py").read_text()
    lines = source.splitlines()
    parsed = ast.parse(source)
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef)]
    first = functions[0].lineno - 1
    cells.append(nbformat.v4.new_code_cell("\n".join(lines[:first])))
    for node in functions:
        cells.append(nbformat.v4.new_markdown_cell(f"### `{node.name}`"))
        cells.append(nbformat.v4.new_code_cell("\n".join(lines[node.lineno - 1:node.end_lineno])))
    cells.extend([
        nbformat.v4.new_markdown_cell(
            "## Execute the prespecified empirical test\n\n"
            "This cell performs real fits on cached features extracted from the downloaded EEG. "
            "It also produces 1000 participant-cluster bootstrap intervals, 21 family ablations "
            "and 20 grouped training-label permutation controls. The cache is included in the bundle; "
            "raw EEG is downloaded separately rather than redistributed."),
        nbformat.v4.new_code_cell("main()"),
        nbformat.v4.new_markdown_cell(
            "## Held-out temperature prediction, in native degrees Celsius\n\n"
            "**R², MAE and RMSE are the continuous-prediction metrics.** Exact-level accuracy rounds "
            "a regressor's prediction to the nearest protocol-defined 42/46/48°C, without using "
            "test labels to choose cutoffs."),
        nbformat.v4.new_code_cell(
            "results = pd.read_csv('accuracy_results.csv')\n"
            "display(results[results.task == 'regression'][['evaluation','feature_set','model','test_subjects','test_trials',"
            "'R2','R2_CI_low','R2_CI_high','MAE','RMSE','native_level_accuracy']])"),
        nbformat.v4.new_markdown_cell(
            "## Secondary lower/higher-temperature classification\n\n"
            "42°C versus 46/48°C is an experimental contrast, **not a validated pain diagnosis**. "
            "A majority predictor reaches 66.7% ordinary accuracy but only 50% balanced accuracy. "
            "Always compare with that baseline; chance ordinary accuracy is not 50% here."),
        nbformat.v4.new_code_cell(
            "display(results[results.task == 'classification'][['evaluation','feature_set','model',"
            "'accuracy','balanced_accuracy','balanced_accuracy_CI_low','balanced_accuracy_CI_high','ROC_AUC','macro_F1']])\n"
            "display(pd.read_csv('feature_set_gains.csv'))"),
        nbformat.v4.new_markdown_cell(
            "## Candidate-family contribution and negative controls\n\n"
            "A positive full-minus-without-family score means the family helped this fixed model in "
            "these folds; a negative value means removal helped. Correlated families and 21 comparisons "
            "prevent a causal or independently validated ranking. Do **not** select families here "
            "and call the resulting score a fresh blind test.\n\n"
            "Only 20 permutations: the smallest possible empirical p-value is 1/21. These controls "
            "are exploratory. Bootstrap intervals resample whole participants but do not capture "
            "all uncertainty from retraining on a new cohort."),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv('family_ablation_results.csv').sort_values('full_minus_without_family', ascending=False))\n"
            "display(pd.read_csv('permutation_summary.csv'))\n"
            "display(Markdown('```json\\n' + Path('verdict.json').read_text() + '\\n```'))"),
        nbformat.v4.new_markdown_cell(
            "## What remains blocked\n\n"
            "No usable second independently graded pain EEG cohort has been scored. Thus no "
            "cross-dataset accuracy or universal physical-dose calibration is reported. "
            "The original two-cohort 'great accuracy' criterion cannot be satisfied by these "
            "single-cohort results. Missing subjective ratings were never filled in.\n\n"
            "Reproduce acquisition/extraction with acquire.py and build_features.py. Original "
            "literature tables and earlier unsupervised analyses remain unchanged. See README.md "
            "for dependencies, the full file catalogue, provenance and instructions.")
    ])
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "research_scope": "Actual participant-held-out physical temperature results; independent second-cohort validation blocked."
    }
    nbformat.write(nb, HERE / "blind_stimulus_validation.ipynb")
    print("Notebook created with", len(cells), "cells")


if __name__ == "__main__":
    main()
