"""Quick checks that run in a few seconds: data shapes, the SNR helper and one training step."""
import unittest

import numpy as np

from src.data_loader import generate_synthetic_data
from analyze import calculate_snr


class DataTests(unittest.TestCase):
    def test_shapes_and_range(self):
        noisy, clean = generate_synthetic_data(num_samples=4, img_shape=(32, 32), seed=1)
        self.assertEqual(noisy.shape, (4, 32, 32, 1))
        self.assertEqual(clean.shape, (4, 32, 32, 1))
        self.assertTrue(((noisy >= 0) & (noisy <= 1)).all())

    def test_seed_is_reproducible(self):
        a, _ = generate_synthetic_data(num_samples=2, img_shape=(16, 16), seed=3)
        b, _ = generate_synthetic_data(num_samples=2, img_shape=(16, 16), seed=3)
        np.testing.assert_array_equal(a, b)


class SnrTests(unittest.TestCase):
    def test_identical_signal_is_capped(self):
        x = np.ones((8, 8))
        self.assertEqual(calculate_snr(x, x), 100)

    def test_known_value(self):
        clean = np.ones(100)
        noisy = clean + 0.1  # noise power is 1% of signal power -> 20 dB
        self.assertAlmostEqual(calculate_snr(clean, noisy), 20.0, places=6)


class ModelTests(unittest.TestCase):
    def test_model_round_trip_shape(self):
        from src.model import build_denoising_autoencoder
        model = build_denoising_autoencoder((32, 32, 1))
        noisy, clean = generate_synthetic_data(num_samples=4, img_shape=(32, 32), seed=2)
        model.fit(noisy, clean, epochs=1, batch_size=2, verbose=0)
        self.assertEqual(model.predict(noisy, verbose=0).shape, noisy.shape)


if __name__ == '__main__':
    unittest.main()
