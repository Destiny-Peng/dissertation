"""CPU-only exact-PCA reuse and training-data isolation tests."""
import sys
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_localization_head import latent


class PCACacheTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.dataset = {name: {"sequence": rng.normal(size=(12, 2560)).astype(np.float32)}
                        for name in ("a", "b", "val")}

    def test_exact_basis_reused_across_dimensions_and_fused_inputs(self):
        cache = latent.PCACache(8)
        messages = []
        exact = latent.fit_pca(self.dataset, ["a", "b"], 4)
        with mock.patch.object(latent, "fit_pca", wraps=latent.fit_pca) as fit:
            eight = cache.get(self.dataset, ["b", "a"], 8, logger=messages.append)
            four = cache.get(self.dataset, ["a", "b"], 4, logger=messages.append)
            fused = {key: {"sequence": np.concatenate([row["sequence"], np.ones((12, 2), dtype=np.float32)], axis=1)}
                     for key, row in self.dataset.items()}
            another = cache.get(fused, ["a", "b"], 4)
            self.assertEqual(fit.call_count, 1)
        np.testing.assert_array_equal(four["components"], eight["components"][:4])
        np.testing.assert_allclose(four["components"], exact["components"])
        np.testing.assert_array_equal(four["components"], another["components"])
        self.assertTrue(any("pca_fit_done" in row for row in messages))
        self.assertTrue(any("pca_cache_hit" in row for row in messages))

    def test_validation_data_ignored_training_changes_and_split_invalidate(self):
        cache = latent.PCACache(4)
        with mock.patch.object(latent, "fit_pca", wraps=latent.fit_pca) as fit:
            cache.get(self.dataset, ["a"], 4)
            self.dataset["val"]["sequence"][:] = 999999
            cache.get(self.dataset, ["a"], 4)
            self.assertEqual(fit.call_count, 1)
            cache.get(self.dataset, ["b"], 4)
            self.assertEqual(fit.call_count, 2)
            self.dataset["a"]["sequence"][0, 0] += 1
            cache.get(self.dataset, ["a"], 4)
            self.assertEqual(fit.call_count, 3)

    def test_concurrent_requests_share_one_inflight_fit(self):
        cache = latent.PCACache(4)
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
        original = latent.fit_pca
        def slow_fit(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test release timed out")
            return original(*args)
        def logger(message):
            if "pca_cache_wait" in message:
                waiting.set()
        with mock.patch.object(latent, "fit_pca", side_effect=slow_fit) as fit, ThreadPoolExecutor(2) as pool:
            first = pool.submit(cache.get, self.dataset, ["a", "b"], 4)
            self.assertTrue(entered.wait(5))
            second = pool.submit(cache.get, self.dataset, ["a", "b"], 3, logger)
            try:
                self.assertTrue(waiting.wait(5))
            finally:
                release.set()
            one, two = first.result(timeout=5), second.result(timeout=5)
            self.assertEqual(fit.call_count, 1)
        np.testing.assert_array_equal(one["components"][:3], two["components"])

    def test_failed_fit_does_not_poison_cache_and_sample_limits_still_apply(self):
        cache = latent.PCACache(4)
        with mock.patch.object(latent, "fit_pca", side_effect=RuntimeError("fit failed")):
            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                cache.get(self.dataset, ["a"], 4)
        result = cache.get(self.dataset, ["a"], 4)
        self.assertEqual(result["components"].shape, (4, 2560))
        for ids, dimensions in (([], 4), (["a", "a"], 4), (["a"], 128)):
            with self.assertRaises(ValueError):
                cache.get(self.dataset, ids, dimensions)


if __name__ == "__main__":
    unittest.main()
