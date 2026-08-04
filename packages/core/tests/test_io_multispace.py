import unittest

import numpy as np

from concreteness_knn_core.data import build_joint_vectors, vectorize_words_intersection


class _DummyEmbeddings:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_vector(self, word: str):
        vec = self._mapping.get(word)
        if vec is None:
            return None
        return np.asarray(vec, dtype=np.float32)


class TestIOMultiSpace(unittest.TestCase):
    def test_vectorize_words_intersection_keeps_only_common_words(self) -> None:
        words = ["alpha", "beta", "gamma"]
        spaces = {
            "s1": _DummyEmbeddings({"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}),
            "s2": _DummyEmbeddings({"beta": [2.0, 0.0], "gamma": [0.0, 2.0]}),
        }

        vectors_by_space, kept_idx = vectorize_words_intersection(words, spaces)

        self.assertEqual(kept_idx, [1])
        self.assertEqual(set(vectors_by_space.keys()), {"s1", "s2"})
        np.testing.assert_allclose(vectors_by_space["s1"], np.asarray([[0.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(vectors_by_space["s2"], np.asarray([[2.0, 0.0]], dtype=np.float32))

    def test_build_joint_vectors_normalizes_per_space_before_concat(self) -> None:
        vectors_by_space = {
            "s1": np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
            "s2": np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        }
        joint = build_joint_vectors(vectors_by_space)

        expected = np.asarray(
            [
                [0.6, 0.8, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(joint, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
