#!/usr/bin/env python3
"""Code-path smoke tests only; no real physiology or validated pain markers."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from make_demo_data import make_demo
from neural_mediation import analyze, predict, ALL_13_FEATURES, FULL_ADULT_FEATURES
from eeg_pain_features import Config, row_features, load_neonatal_template


def run():
    with tempfile.TemporaryDirectory() as tmp:
        r = Path(tmp)
        adult = r/'adult.csv'
        out = r/'fit_adult'
        make_demo(adult, subjects=12, trials=20, seed=12)
        analyze(adult, out, feature_set='all-adult', bootstraps=10, folds=3)
        model = json.loads((out/'eeg_neural_score_model.json').read_text())
        assert model['features'] == FULL_ADULT_FEATURES
        df = pd.read_csv(adult)
        labeled = predict(df, model, 0, 10)
        unlabeled = predict(df.drop(columns=['stimulus_intensity','pain_rating']), model, 0, 10)
        assert np.allclose(labeled['neural_score_z'], unlabeled['neural_score_z'])
        assert len(pd.read_csv(out/'heldout_trial_predictions.csv')) == len(df)
        assert np.isfinite(labeled['score_0_100']).all()
        try:
            analyze(adult, r/'should_fail', feature_set='all-13', fit_only=True)
        except ValueError as exc:
            assert 'neonatal' in str(exc)
        else:
            raise AssertionError('Adult 13-feature model must fail')
        neonatal = r/'neonatal.csv'
        make_demo(neonatal, subjects=12, trials=20, seed=22, cohort='neonatal')
        analyze(neonatal, r/'fit_neonatal', feature_set='all-13', folds=3, bootstraps=0)
        neonatal_model = json.loads((r/'fit_neonatal'/'eeg_neural_score_model.json').read_text())
        assert neonatal_model['features'] == ALL_13_FEATURES
        # Synthetic external waveform used to exercise optional extraction.
        tt = np.linspace(-0.2, 0.9, 250)
        reference = 2*np.exp(-0.5*((tt-.36)/.075)**2)
        template_file = r/'synthetic_template.csv'
        pd.DataFrame({'time_s':tt, 'amplitude_uv':reference}).to_csv(template_file, index=False)
        template = load_neonatal_template(template_file, 'Cz')
        epoch_times = np.arange(-2.0, 2.0, 1/256)
        wave = np.interp(epoch_times, tt, reference, left=0, right=0)
        feat = row_features(np.tile(wave,(7,1)), epoch_times,
                            ['Cz','F3','F4','C3','C4','Fz','FCz'], 256, Config(), template)
        assert np.isfinite(feat['neonatal_noxious_template_corr'])
        assert feat['neonatal_noxious_template_corr'] > 0.99
        assert np.isfinite(feat['delta_cz_response_temporal_sd_uv'])
        assert np.isnan(row_features(np.tile(wave,(7,1)),epoch_times,
                           ['Cz','F3','F4','C3','C4','Fz','FCz'],256,Config())['neonatal_noxious_template_corr'])
        print('PASS: 12 adult features, 13 neonatal features, guarded infant pathway, EEG-only inference, trial temporal metrics')

if __name__=='__main__':
    run()
