import unittest

import numpy as np

from concreteness_knn_core.knn import predict_with_subset, topk_indices_and_distances
from concreteness_knn_core.knn.knn_subset_eval import (
    _topk_indices_and_distances_batched,
    normalize_rows,
)


def _legacy_topk(
    train_vectors: np.ndarray,
    target_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Provide a whole-matrix reference for exact equivalence assertions."""
    train_norm = normalize_rows(train_vectors)
    target_norm = normalize_rows(target_vectors)
    sim = target_norm @ train_norm.T

    if k >= sim.shape[1]:
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
    else:
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        part_sim = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        topk_idx = np.take_along_axis(part, order, axis=1)

    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
    return topk_idx, 1.0 - topk_sim


class TestBatchedTopK(unittest.TestCase):
    def assert_matches_legacy(
        self,
        train: np.ndarray,
        target: np.ndarray,
        k: int,
        batch_size: int,
    ) -> None:
        expected_idx, expected_dist = _legacy_topk(train, target, k)
        actual_idx, actual_dist = _topk_indices_and_distances_batched(
            train,
            target,
            k,
            batch_size=batch_size,
        )
        np.testing.assert_array_equal(actual_idx, expected_idx)
        np.testing.assert_allclose(actual_dist, expected_dist, rtol=1e-6, atol=1e-7)
        self.assertEqual(actual_idx.dtype, expected_idx.dtype)
        self.assertEqual(actual_dist.dtype, expected_dist.dtype)

    def test_batch_sizes_and_k_edges_match_whole_matrix(self) -> None:
        rng = np.random.default_rng(13)
        train_values = rng.normal(size=(11, 7))
        target_values = rng.normal(size=(8, 7))

        for dtype in (np.float32, np.float64):
            train = train_values.astype(dtype)
            target = target_values.astype(dtype)
            for batch_size in (1, 3, 100):
                for k in (1, 4, len(train)):
                    with self.subTest(dtype=dtype, batch_size=batch_size, k=k):
                        self.assert_matches_legacy(train, target, k, batch_size)

    def test_duplicate_tied_and_zero_vectors_match(self) -> None:
        train = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        target = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        # Full sorting defines deterministic tie ordering for this oracle.
        self.assert_matches_legacy(train, target, len(train), batch_size=2)

    def test_self_overlap_targets_match(self) -> None:
        rng = np.random.default_rng(41)
        train = rng.normal(size=(17, 5)).astype(np.float32)
        target = train[[0, 5, 16]].copy()
        self.assert_matches_legacy(train, target, k=6, batch_size=2)

        idx, distances = topk_indices_and_distances(train, target, k=1)
        np.testing.assert_array_equal(idx[:, 0], np.asarray([0, 5, 16]))
        self.assertTrue(
            np.all(np.abs(distances[:, 0]) <= np.finfo(np.float32).eps)
        )

    def test_public_helper_crosses_internal_512_row_boundary(self) -> None:
        rng = np.random.default_rng(83)
        train = rng.normal(size=(19, 6)).astype(np.float32)
        target = rng.normal(size=(517, 6)).astype(np.float32)
        expected_idx, expected_dist = _legacy_topk(train, target, k=5)
        actual_idx, actual_dist = topk_indices_and_distances(train, target, k=5)
        np.testing.assert_array_equal(actual_idx, expected_idx)
        np.testing.assert_allclose(actual_dist, expected_dist, rtol=1e-6, atol=1e-7)

    def test_predictions_remain_numerically_equal(self) -> None:
        rng = np.random.default_rng(107)
        train = rng.normal(size=(23, 8)).astype(np.float32)
        target = rng.normal(size=(7, 8)).astype(np.float32)
        scores = rng.uniform(1.0, 5.0, size=len(train)).astype(np.float32)

        expected_idx, expected_dist = _legacy_topk(train, target, k=7)
        expected_weights = 1.0 / np.clip(expected_dist, 1e-12, None)
        expected = (expected_weights * scores[expected_idx]).sum(axis=1) / np.clip(
            expected_weights.sum(axis=1), 1e-12, None
        )
        actual = predict_with_subset(
            train_vectors=train,
            train_scores=scores,
            target_vectors=target,
            k=7,
            weighting="distance",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_empty_targets_preserve_shape_and_dtype(self) -> None:
        train = np.eye(3, dtype=np.float32)
        target = np.empty((0, 3), dtype=np.float32)
        self.assert_matches_legacy(train, target, k=2, batch_size=1)

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            _topk_indices_and_distances_batched(
                np.eye(2, dtype=np.float32),
                np.eye(2, dtype=np.float32),
                k=1,
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
