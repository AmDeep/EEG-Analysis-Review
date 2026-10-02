"""Generate the PAF inspection notebook and its actual-data spectral figure."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT


def main():
    results = [r for r in json.loads((OUT / "results.json").read_text())
               if r["status"] == "analyzed"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), layout="constrained")
    for ax, record in zip(axes.flat, results):
        with np.load(OUT / f"{record['key']}_spectra.npz") as data:
            f = data["frequency_hz"]
            mask = (f >= 7) & (f <= 13)
            for i, ch in enumerate(data["channels"]):
                ax.semilogy(f[mask], data["mean_psd_uv2_per_hz"][i, mask],
                            label=ch, lw=1.4)
        ax.axvspan(9, 11, color="#888888", alpha=.15)
        ax.set(title=f"{record['key']} · ROI CoG {record['roi_cog_9_11_hz']:.3f} Hz",
               xlabel="Frequency (Hz)", ylabel="Mean PSD (µV²/Hz)", xlim=(7, 13))
        ax.legend(fontsize=8)
    fig.suptitle("Measured C3/C4/Cz alpha spectra\nShading: paper's 9–11 Hz centroid band", fontsize=15)
    path = OUT / "alpha_spectra.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    nb = nbformat.v4.new_notebook(metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    })
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Peak alpha frequency: measured C3/C4/Cz features\n\n"
            "[Furman et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6843105/) "
            "uses **9–11 Hz power-weighted center of gravity**, not an argmax. "
            "We use five-second symmetric-Hann epochs and 0.2-Hz bins, average epoch "
            "centroids per channel, then average C3/C4/Cz. Existing PCA-free "
            "preprocessing is retained. These are selected excerpts, not verified "
            "pain-free baselines. Read `paf_report.md` before interpreting the values."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\n"
            "candidates = (Path.cwd(), Path.cwd() / 'paf_results', "
            "Path.cwd() / 'eeg-analysis' / 'paf_results')\n"
            "BASE = next(p for p in candidates if (p / 'summary.csv').is_file())\n"
            "summary = pd.read_csv(BASE / 'summary.csv')\n"
            "print('Measured recordings:', len(summary))\n"
            "print('Retained five-second epochs:', summary.epochs_used.sum())\n"
            "display(summary.round(3))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Spectral inspection\n\nThe shaded band may not contain a recording's "
            "strongest alpha peak. A centroid still exists in a sloped spectrum; "
            "it is not proof of an oscillation.\n\n"
            "![Actual alpha spectra](attachment:alpha_spectra.png)",
            attachments={"alpha_spectra.png": {
                "image/png": base64.b64encode(path.read_bytes()).decode()}},
        ),
        nbformat.v4.new_code_cell(
            "import json\nresults = json.loads((BASE / 'results.json').read_text())\n"
            "rows = [dict(recording=r['key'], channel=ch, **values)\n"
            "        for r in results if r['status']=='analyzed'\n"
            "        for ch, values in r['channels'].items()]\n"
            "display(pd.DataFrame(rows).round(3))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Reproduce\n\nRun `python paf_features.py`, then `python paf_outputs.py` "
            "from `eeg-analysis/paf_results/`. Raw binaries are SHA-256 verified first. "
            "The per-epoch CSV files retain original epoch indices and onset times, "
            "including gaps after rejection. `qualified_local_peak_hz` is a secondary "
            "8–12 Hz local peak with a heuristic ≥1 dB prominence in the mean spectrum; "
            "it is neither the paper's feature nor a validated alpha-quality classifier. "
            "Missing peaks are left empty, not filled at 10 Hz. No pain correlation is fitted."
        ),
    ]
    NotebookClient(nb, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / "paf_results.ipynb")
    print("Saved spectral figure and executed PAF notebook")


if __name__ == "__main__":
    main()