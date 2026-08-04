import tempfile
import unittest
from pathlib import Path

import pandas as pd

from concreteness_knn_core.data.data_gold_loader import load_gold_df


class TestGoldLoader(unittest.TestCase):
    def test_loads_tsv_gold_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            gold_tsv = tmp / "gold.tsv"
            pd.DataFrame(
                {
                    "word": ["Haus", "Idee", "Tisch"],
                    "score": [7.2, 2.1, 6.9],
                }
            ).to_csv(gold_tsv, sep="\t", index=False)

            cfg = {
                "gold": str(gold_tsv),
                "word_column": "word",
                "score_column": "score",
                "lowercase": True,
                "pos_filter": {"enabled": False, "pos_column": "pos", "tags": ["Noun"]},
            }

            df = load_gold_df(cfg)
            self.assertEqual(3, len(df))
            self.assertIn("haus", set(df["word"].tolist()))

    def test_token_contains_pos_filter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            gold_tsv = tmp / "gold.tsv"
            pd.DataFrame(
                {
                    "word": ["casa", "bonito", "andar", "profundo"],
                    "score": [6.0, 4.0, 3.0, 5.0],
                    "pos": ["N", "ADJ", "V", "ADJ,N"],
                }
            ).to_csv(gold_tsv, sep="\t", index=False)

            cfg = {
                "gold": str(gold_tsv),
                "word_column": "word",
                "score_column": "score",
                "lowercase": True,
                "pos_filter": {
                    "enabled": True,
                    "pos_column": "pos",
                    "tags": ["N"],
                    "match_mode": "token_contains",
                    "token_pattern": r"[,;/| ]+",
                },
            }

            df = load_gold_df(cfg)
            self.assertEqual({"casa", "profundo"}, set(df["word"].tolist()))


if __name__ == "__main__":
    unittest.main()
