"""Prediction data preparation: gold data, embeddings, splits, and vocab features."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from concreteness_knn_core.config import active_space_ids
from concreteness_knn_core.data import (
    build_joint_vectors,
    load_embeddings_by_space,
    load_gold_arrays,
    load_vocab_words,
    oov_words,
    vectorize_words,
    vectorize_words_intersection,
)

from .prediction_types import PredictionPreparedData, PredictionSplitData


class PredictionDataPreparer:
    """Load lexical inputs and materialize active embedding feature views."""

    def prepare_data(self, config: dict) -> PredictionPreparedData:
        """Vectorize covered gold words in the configured embedding mode.

        Single mode loads only its active space. Joint and multi-space modes
        retain words covered by every configured space. Embedding adapters stay
        attached to the returned object until optional target vectorization is
        complete.
        """
        dataset_cfg = config["dataset"]
        embeddings_cfg = config["embeddings"]

        embedding_mode = str(embeddings_cfg["mode"])
        configured_space_ids = active_space_ids(embeddings_cfg)

        words_all, y_all = load_gold_arrays(dataset_cfg)
        embeddings_by_space = load_embeddings_by_space(embeddings_cfg)

        if embedding_mode == "single":
            space_id = configured_space_ids[0]
            x_all, kept_idx = vectorize_words(words_all, embeddings_by_space[space_id])
            x_all_by_view: dict[str, np.ndarray] = {space_id: x_all}
        elif embedding_mode == "joint":
            selected = {space_id: embeddings_by_space[space_id] for space_id in configured_space_ids}
            x_by_space, kept_idx = vectorize_words_intersection(words_all, selected)
            x_all_by_view = {"joint": build_joint_vectors(x_by_space)}
        elif embedding_mode == "multi_space":
            selected = {space_id: embeddings_by_space[space_id] for space_id in configured_space_ids}
            x_all_by_view, kept_idx = vectorize_words_intersection(words_all, selected)
        else:  # pragma: no cover (guarded by config validation)
            raise ValueError(f"Unknown embeddings.mode: {embedding_mode}")

        words_kept = [words_all[i] for i in kept_idx]
        y_kept = y_all[kept_idx]
        dropped_gold_words = oov_words(words_all, kept_idx)
        coverage = float(len(words_kept) / max(1, len(words_all)))

        return PredictionPreparedData(
            embedding_mode=embedding_mode,
            configured_space_ids=configured_space_ids,
            view_ids=list(x_all_by_view.keys()),
            words_all=words_all,
            y_all=y_all,
            words_kept=words_kept,
            y_kept=y_kept,
            dropped_gold_words=dropped_gold_words,
            coverage=coverage,
            embeddings_by_space=embeddings_by_space,
            x_all_by_view=x_all_by_view,
        )

    def split_data(self, prepared: PredictionPreparedData, config: dict) -> PredictionSplitData:
        """Create the outer train/test split used for prediction reporting."""
        prediction_cfg = config["prediction"]
        runtime_cfg = config["runtime"]

        all_idx = np.arange(len(prepared.y_kept), dtype=np.int32)
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=float(prediction_cfg["test_size"]),
            random_state=int(runtime_cfg["seed"]),
            shuffle=True,
        )

        x_train_by_view = {
            view_id: x_all[train_idx] for view_id, x_all in prepared.x_all_by_view.items()
        }
        x_test_by_view = {view_id: x_all[test_idx] for view_id, x_all in prepared.x_all_by_view.items()}

        return PredictionSplitData(
            train_idx=train_idx,
            test_idx=test_idx,
            y_train=prepared.y_kept[train_idx],
            y_test=prepared.y_kept[test_idx],
            w_train=[prepared.words_kept[int(i)] for i in train_idx],
            w_test=[prepared.words_kept[int(i)] for i in test_idx],
            x_train_by_view=x_train_by_view,
            x_test_by_view=x_test_by_view,
            view_ids=list(x_train_by_view.keys()),
        )

    def prepare_vocab_features(
        self,
        prepared: PredictionPreparedData,
        config: dict,
    ) -> tuple[list[str], dict[str, np.ndarray], list[str]] | None:
        """Vectorize the optional target vocabulary and report uncovered words.

        Returns ``None`` when no target is configured; otherwise returns target
        words with coverage, their feature matrices by active view, and OOV
        target words.
        """
        target = config["prediction"].get("target")
        if not target:
            return None

        vocab_words = load_vocab_words(str(target), bool(config["dataset"]["lowercase"]))

        if prepared.embedding_mode == "single":
            space_id = prepared.configured_space_ids[0]
            x_vocab, kept_idx = vectorize_words(vocab_words, prepared.embeddings_by_space[space_id])
            x_vocab_by_view: dict[str, np.ndarray] = {space_id: x_vocab}
        elif prepared.embedding_mode == "joint":
            selected = {
                space_id: prepared.embeddings_by_space[space_id]
                for space_id in prepared.configured_space_ids
            }
            x_by_space, kept_idx = vectorize_words_intersection(vocab_words, selected)
            x_vocab_by_view = {"joint": build_joint_vectors(x_by_space)}
        elif prepared.embedding_mode == "multi_space":
            selected = {
                space_id: prepared.embeddings_by_space[space_id]
                for space_id in prepared.configured_space_ids
            }
            x_vocab_by_view, kept_idx = vectorize_words_intersection(vocab_words, selected)
        else:  # pragma: no cover (guarded by validation)
            raise ValueError(f"Unknown embeddings.mode: {prepared.embedding_mode}")

        kept_words = [vocab_words[i] for i in kept_idx]
        dropped_words = oov_words(vocab_words, kept_idx)
        return kept_words, x_vocab_by_view, dropped_words
