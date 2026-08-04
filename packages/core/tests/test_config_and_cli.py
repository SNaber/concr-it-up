import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from concreteness_knn_core.cli import build_parser, main as cli_main
from concreteness_knn_core.config import default_config, load_config, validate_config


class TestConfigAndCLI(unittest.TestCase):
    def test_prediction_config_init_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "prediction.json"
            with patch(
                "sys.argv",
                [
                    "concreteness-knn-core",
                    "config",
                    "init",
                    "--task",
                    "prediction",
                    "--out",
                    str(cfg_path),
                ],
            ):
                cli_main()

            self.assertTrue(cfg_path.exists())
            cfg = load_config(str(cfg_path))
            validate_config(cfg, task="prediction_run")

    def test_cli_prediction_run_routes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "dummy.json"
            cfg_path.write_text("{}", encoding="utf-8")

            with patch("concreteness_knn_core.cli.load_config") as load_mock:
                with patch("concreteness_knn_core.cli.PredictionPipeline.run_prediction") as run_mock:
                    with patch(
                        "sys.argv",
                        ["concreteness-knn-core", "prediction", "run", "--config", str(cfg_path)],
                    ):
                        cli_main()

            self.assertTrue(load_mock.called)
            self.assertTrue(run_mock.called)

    def test_cli_prediction_holdout_routes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "dummy.json"
            cfg_path.write_text("{}", encoding="utf-8")

            with patch("concreteness_knn_core.cli.load_config") as load_mock:
                with patch("concreteness_knn_core.cli.PredictionPipeline.run_holdout") as run_mock:
                    with patch(
                        "sys.argv",
                        ["concreteness-knn-core", "prediction", "holdout", "--config", str(cfg_path)],
                    ):
                        cli_main()

            self.assertTrue(load_mock.called)
            self.assertTrue(run_mock.called)

    def test_validate_single_mode_requires_matching_space_id(self) -> None:
        cfg = default_config()
        cfg["dataset"]["gold"] = "dummy.csv"
        cfg["embeddings"]["spaces"][0]["path"] = "dummy.vec"
        cfg["embeddings"]["mode"] = "single"
        cfg["embeddings"]["active_space"] = "missing"
        with self.assertRaisesRegex(ValueError, "active_space"):
            validate_config(cfg, task="prediction_run")

    def test_validate_joint_mode_requires_two_spaces(self) -> None:
        cfg = default_config()
        cfg["dataset"]["gold"] = "dummy.csv"
        cfg["embeddings"]["mode"] = "joint"
        cfg["embeddings"]["spaces"] = [{"id": "a", "kind": "vec", "path": "a.vec", "label": "a"}]
        with self.assertRaisesRegex(ValueError, "requires at least 2 spaces"):
            validate_config(cfg, task="prediction_run")

    def test_validate_rejects_legacy_config_keys(self) -> None:
        cfg = default_config()
        cfg["dataset"]["gold_csv"] = "dummy.csv"
        cfg["embeddings"]["spaces"][0]["path"] = "dummy.vec"
        with self.assertRaisesRegex(ValueError, "Legacy config keys are not supported"):
            validate_config(cfg, task="prediction_run")

    def test_parser_rejects_search_group(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "run", "--config", "cfg.json"])


if __name__ == "__main__":
    unittest.main()
