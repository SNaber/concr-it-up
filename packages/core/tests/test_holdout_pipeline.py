import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from concreteness_knn_core.prediction import PredictionPipeline
from concreteness_knn_core.config import load_config


class TestHoldoutPipeline(unittest.TestCase):
    def test_holdout_outputs_summary_and_unmatched(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            holdout = tmp / "holdout.csv"
            predictions_csv = tmp / "pred.csv"
            unmatched_csv = tmp / "unmatched.csv"

            pd.DataFrame({"Word": ["alpha", "beta", "gamma"], "Conc.M": [1.0, 2.0, 3.0]}).to_csv(
                holdout, index=False
            )
            pd.DataFrame({"word": ["alpha", "gamma", "delta"], "pred": [1.1, 2.9, 4.0]}).to_csv(
                predictions_csv, index=False
            )

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(holdout),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "prediction": {
                    "holdout": str(holdout),
                    "predictions_csv": str(predictions_csv),
                    "prediction_column": "pred",
                    "unmatched_csv": str(unmatched_csv),
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))

            outputs = PredictionPipeline().run_holdout(cfg, config_path=str(cfg_path))
            self.assertTrue(Path(outputs["holdout_summary"]).exists())
            self.assertTrue(Path(outputs["holdout_unmatched"]).exists())
            self.assertEqual(Path(outputs["holdout_summary"]).name, "holdout_summary.json")
            self.assertEqual(Path(outputs["holdout_unmatched"]).name, "unmatched.csv")

            summary = json.loads(Path(outputs["holdout_summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["n_matched"], 2)

            unmatched = pd.read_csv(outputs["holdout_unmatched"])
            self.assertEqual(list(unmatched.columns), ["source", "word"])

    def test_holdout_rejects_duplicate_words_in_holdout_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            holdout = tmp / "holdout.csv"
            predictions_csv = tmp / "pred.csv"

            pd.DataFrame({"Word": ["alpha", "alpha", "gamma"], "Conc.M": [1.0, 1.2, 3.0]}).to_csv(
                holdout, index=False
            )
            pd.DataFrame({"word": ["alpha", "gamma"], "pred": [1.1, 2.9]}).to_csv(
                predictions_csv, index=False
            )

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(holdout),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "prediction": {
                    "holdout": str(holdout),
                    "predictions_csv": str(predictions_csv),
                    "prediction_column": "pred",
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))

            with self.assertRaisesRegex(ValueError, "Duplicate words found in holdout"):
                PredictionPipeline().run_holdout(cfg, config_path=str(cfg_path))

    def test_holdout_rejects_duplicate_words_in_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            holdout = tmp / "holdout.csv"
            predictions_csv = tmp / "pred.csv"

            pd.DataFrame({"Word": ["alpha", "beta", "gamma"], "Conc.M": [1.0, 2.0, 3.0]}).to_csv(
                holdout, index=False
            )
            pd.DataFrame({"word": ["alpha", "alpha", "gamma"], "pred": [1.1, 1.2, 2.9]}).to_csv(
                predictions_csv, index=False
            )

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(holdout),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "prediction": {
                    "holdout": str(holdout),
                    "predictions_csv": str(predictions_csv),
                    "prediction_column": "pred",
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))

            with self.assertRaisesRegex(ValueError, "Duplicate words found in predictions_csv"):
                PredictionPipeline().run_holdout(cfg, config_path=str(cfg_path))


if __name__ == "__main__":
    unittest.main()
