"""Analytical/synthetic unit tests only; not research data or study findings."""
import unittest

import numpy as np
from scipy.signal import periodogram

from paf_features import cog, peak


class PAFTests(unittest.TestCase):
    def test_known_frequency(self):
        t = np.arange(640) / 128
        for frequency in (9.6, 10., 10.6):
            f, p = periodogram(np.sin(2 * np.pi * frequency * t), fs=128,
                               window=np.hanning(640), scaling="density")
            self.assertAlmostEqual(f[1] - f[0], .2)
            self.assertAlmostEqual(float(cog(f, p)), frequency, places=3)

    def test_power_scaling_invariance(self):
        f = np.arange(0, 64.1, .2)
        p = np.exp(-((f - 9.7) / .4) ** 2) + .01
        np.testing.assert_allclose(cog(f, p), cog(f, p * 9), atol=1e-12)

    def test_zero_power_is_missing(self):
        f = np.arange(0, 64.1, .2)
        self.assertTrue(np.isnan(cog(f, np.zeros_like(f))))

    def test_no_local_peak_on_monotonic_spectrum(self):
        f = np.arange(0, 64.1, .2)
        self.assertEqual(peak(f, 1 / (f + 1)), (None, None))

    def test_uniform_power_centroid_is_not_peak_evidence(self):
        f = np.arange(0, 64.1, .2)
        self.assertAlmostEqual(float(cog(f, np.ones_like(f))), 10.)
        self.assertEqual(peak(f, np.ones_like(f)), (None, None))


if __name__ == "__main__":
    unittest.main()