import unittest
from unittest.mock import patch

import numpy as np

from concreteness_knn_core.knn import cross_validate_knn, evaluate_candidate, predict_with_subset


class TestModelKNN(unittest.TestCase):
    def test_predict_with_subset_restricts_neighbors(self) -> None:
        train_x = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        train_y = np.asarray([5.0, 4.8, 1.0], dtype=np.float32)
        target_x = np.asarray([[1.0, 0.0]], dtype=np.float32)

        pred_full = predict_with_subset(
            train_vectors=train_x,
            train_scores=train_y,
            target_vectors=target_x,
            subset_indices=None,
            k=1,
            weighting="uniform",
        )
        pred_subset = predict_with_subset(
            train_vectors=train_x,
            train_scores=train_y,
            target_vectors=target_x,
            subset_indices=np.asarray([2], dtype=np.int32),
            k=1,
            weighting="uniform",
        )
        self.assertGreater(pred_full[0], pred_subset[0])
        self.assertAlmostEqual(float(pred_subset[0]), 1.0, places=6)

    def test_evaluate_candidate_returns_metrics(self) -> None:
        train_x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        train_y = np.asarray([5.0, 1.0], dtype=np.float32)
        target_x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        target_y = np.asarray([5.0, 1.0], dtype=np.float32)

        out = evaluate_candidate(
            train_vectors=train_x,
            train_scores=train_y,
            target_vectors=target_x,
            target_scores=target_y,
            subset_indices=np.asarray([0, 1], dtype=np.int32),
            k=1,
            weighting="uniform",
        )
        self.assertIn("spearman", out)
        self.assertIn("rmse", out)
        self.assertLess(out["rmse"], 1e-8)

    def test_cross_validate_knn_passes_n_jobs(self) -> None:
        x = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            dtype=np.float32,
        )
        y = np.asarray([5.0, 4.7, 1.0, 1.3], dtype=np.float32)
        seen_n_jobs = []

        class DummyKNN:
            def __init__(self, *, n_jobs: int, **kwargs):
                seen_n_jobs.append(int(n_jobs))
                self._mean = 0.0

            def fit(self, x_train, y_train):
                self._mean = float(np.mean(y_train))
                return self

            def predict(self, x_val):
                return np.full(len(x_val), self._mean, dtype=np.float32)

        with patch("concreteness_knn_core.knn.knn_backend.KNeighborsRegressor", DummyKNN):
            out = cross_validate_knn(
                x=x,
                y=y,
                k_values=[1, 2],
                weights_options=["uniform"],
                n_splits=2,
                seed=13,
                n_jobs=3,
            )

        self.assertTrue(out)
        self.assertTrue(seen_n_jobs)
        self.assertTrue(all(v == 3 for v in seen_n_jobs))


if __name__ == "__main__":
    unittest.main()
