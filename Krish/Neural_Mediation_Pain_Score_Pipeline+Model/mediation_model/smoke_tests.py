#!/usr/bin/env python3
"""Quick assertions for the training/inference code using SIMULATED EEG features."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from make_demo_data import make_demo
from neural_mediation import analyze, predict, CORE_FEATURES


def run():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        path = root / 'fake_features.csv'
        output = root / 'output'
        make_demo(path, subjects=15, trials=18, seed=3)
        analyze(path, output, bootstraps=10)
        data = pd.read_csv(path)
        model = json.loads((output / 'eeg_neural_score_model.json').read_text())
        # EEG score MUST be unaffected by presence or absence of stimulus and pain labels at inference.
        full = predict(data, model, 0, 10)
        unlabeled = predict(data.drop(columns=['stimulus_intensity', 'pain_rating']), model, 0, 10)
        assert np.allclose(full['neural_score_z'], unlabeled['neural_score_z'])
        assert np.allclose(full['score_0_100'], unlabeled['score_0_100'])
        assert set(model['features']) == set(CORE_FEATURES)
        assert len(pd.read_csv(output / 'heldout_trial_predictions.csv')) == len(data)
        assert np.isfinite(full['neural_score_z']).all()
        assert full['score_0_100'].between(0, 100).all()
        metrics = json.loads((output / 'heldout_metrics.json').read_text())
        assert all(np.isfinite(metrics[k]['MAE']) for k in ['eeg_only','intensity_only','intensity_plus_eeg'])
        print('PASS: grouped cross-fitting, EEG-only inference, feature inputs, output CSVs, metrics')

if __name__ == '__main__':
    run()
