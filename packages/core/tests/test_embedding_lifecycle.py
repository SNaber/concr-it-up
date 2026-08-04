import unittest
from types import SimpleNamespace
from unittest.mock import patch

from concreteness_knn_core.config import default_config
from concreteness_knn_core.data import data_embeddings_registry
from concreteness_knn_core.prediction import PredictionPipeline


class _FailingDataPreparer:
    def __init__(self) -> None:
        self.prepared = SimpleNamespace(embeddings_by_space={"active": object()})

    def prepare_data(self, config: dict):
        return self.prepared

    def split_data(self, prepared, config: dict):
        raise RuntimeError("synthetic split failure")


class _SuccessfulDataPreparer:
    def __init__(self) -> None:
        self.prepared = SimpleNamespace(embeddings_by_space={"active": object()})
        self.split = object()

    def prepare_data(self, config: dict):
        return self.prepared

    def split_data(self, prepared, config: dict):
        return self.split


class _LifecycleCheckingTrainer:
    def __init__(self) -> None:
        self.called = False

    def train(self, *, prepared, split, config, vocab_features):
        if prepared.embeddings_by_space:
            raise AssertionError("embedding objects reached model training")
        self.called = True
        return object()


class _NoOpReportWriter:
    def write_prediction_outputs(self, **kwargs):
        return {}


def _valid_config() -> dict:
    config = default_config()
    config["dataset"]["gold"] = "synthetic.csv"
    config["embeddings"]["active_space"] = "active"
    config["embeddings"]["spaces"] = [
        {"id": "active", "kind": "vec", "path": "active.vec", "label": "active"}
    ]
    config["runtime"]["n_jobs"] = 1
    return config


class TestEmbeddingLifecycle(unittest.TestCase):
    def test_pipeline_releases_embeddings_before_model_training(self) -> None:
        data_prep = _SuccessfulDataPreparer()
        trainer = _LifecycleCheckingTrainer()
        pipeline = PredictionPipeline(
            data_prep=data_prep,
            trainer=trainer,
            report_writer=_NoOpReportWriter(),
        )

        with patch(
            "concreteness_knn_core.prediction.prediction_pipeline.gc.collect"
        ) as collect:
            self.assertEqual(
                pipeline.run_prediction(_valid_config(), config_path="synthetic.json"),
                {},
            )

        self.assertTrue(trainer.called)
        self.assertEqual(data_prep.prepared.embeddings_by_space, {})
        collect.assert_called_once_with()

    def test_pipeline_releases_embeddings_even_if_vector_stage_fails(self) -> None:
        data_prep = _FailingDataPreparer()
        pipeline = PredictionPipeline(data_prep=data_prep)

        with patch(
            "concreteness_knn_core.prediction.prediction_pipeline.gc.collect"
        ) as collect:
            with self.assertRaisesRegex(RuntimeError, "synthetic split failure"):
                pipeline.run_prediction(_valid_config(), config_path="synthetic.json")

        self.assertEqual(data_prep.prepared.embeddings_by_space, {})
        collect.assert_called_once_with()

    def test_single_mode_loads_only_active_embedding(self) -> None:
        config = {
            "mode": "single",
            "active_space": "selected",
            "spaces": [
                {"id": "unused", "kind": "vec", "path": "unused.vec"},
                {"id": "selected", "kind": "vec", "path": "selected.vec"},
            ],
        }
        sentinel = object()
        with patch.object(
            data_embeddings_registry,
            "load_embeddings",
            return_value=sentinel,
        ) as loader:
            loaded = data_embeddings_registry.load_embeddings_by_space(config)

        self.assertEqual(loaded, {"selected": sentinel})
        loader.assert_called_once_with("vec", "selected.vec")

    def test_joint_and_multi_space_modes_load_every_configured_space(self) -> None:
        spaces = [
            {"id": "first", "kind": "vec", "path": "first.vec"},
            {"id": "second", "kind": "ft_bin", "path": "second.bin"},
        ]
        for mode in ("joint", "multi_space"):
            with self.subTest(mode=mode):
                with patch.object(
                    data_embeddings_registry,
                    "load_embeddings",
                    side_effect=lambda kind, path: (kind, path),
                ) as loader:
                    loaded = data_embeddings_registry.load_embeddings_by_space(
                        {"mode": mode, "spaces": spaces}
                    )

                self.assertEqual(list(loaded), ["first", "second"])
                self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
