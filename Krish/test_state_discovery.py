"""Checks for dataset scope, feature leakage, and elbow edge cases."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("state_discovery", Path(__file__).with_name("07_state_discovery.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class StateDiscoveryTests(unittest.TestCase):
    def test_filter_and_exclude_outcomes(self):
        records = []
        for dataset in [*experiment.DATASETS, "ds005284"]:
            for epoch in range(3):
                row = dict.fromkeys(experiment.EEG_FEATURES, float(epoch))
                row.update(dataset=dataset, subject="sub-001", epoch=epoch, rating=999, laser_power=888)
                records.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            data, X, dropped = experiment.load_features(path)
        self.assertEqual(set(data.dataset), set(experiment.DATASETS))
        self.assertEqual(len(data), 9)
        self.assertEqual(list(X), experiment.EEG_FEATURES)
        self.assertNotIn("rating", X)
        self.assertNotIn("laser_power", X)
        self.assertEqual(dropped, 0)

    def test_no_elbow_on_linear_or_flat_curve(self):
        for inertia in ([10, 8, 6, 4, 2], [0, 0, 0, 0, 0]):
            self.assertIsNone(experiment.elbow(pd.DataFrame({"k": range(1, 6), "inertia": inertia}))[0])

    def test_obvious_elbow(self):
        curve = pd.DataFrame({"k": range(1, 6), "inertia": [100, 50, 10, 9, 8]})
        self.assertEqual(experiment.elbow(curve)[0], 3)

    def test_preprocessing_keeps_training_statistics(self):
        train = pd.DataFrame({"a": [1., 2., 3., np.nan], "b": [np.nan] * 4})
        prep = experiment.preprocessing().fit(train)
        before = prep[0].statistics_.copy()
        result = prep.transform(pd.DataFrame({"a": [1e6], "b": [1e6]}))
        np.testing.assert_array_equal(before, prep[0].statistics_)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
