"""Gold dataset loading, normalization, and optional POS filtering."""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd


def normalize_word(word: str, lowercase: bool) -> str:
    """Normalize a token using strip + optional lowercasing."""
    token = str(word).strip()
    return token.lower() if lowercase else token


def _read_gold_table(path_text: str) -> pd.DataFrame:
    """Read gold table with extension-aware delimiter fallback."""
    path = str(path_text).strip().lower()
    primary_sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    secondary_sep = "," if primary_sep == "\t" else "\t"

    first = pd.read_csv(path_text, sep=primary_sep)
    if len(first.columns) > 1:
        return first

    # Fallback for mislabeled extensions.
    second = pd.read_csv(path_text, sep=secondary_sep)
    return second if len(second.columns) > 1 else first


def _build_pos_mask(series: pd.Series, tags: set[str], match_mode: str, token_pattern: str) -> pd.Series:
    """Match normalized POS values exactly or by separator-delimited tokens."""
    normalized = series.astype(str).str.strip().str.lower()
    if match_mode == "token_contains":
        return normalized.map(
            lambda value: bool(tags.intersection({tok for tok in re.split(token_pattern, value) if tok}))
        )
    return normalized.isin(tags)


def load_gold_df(dataset_cfg: dict) -> pd.DataFrame:
    """Load, normalize, and optionally POS-filter gold ratings."""
    gold = str(dataset_cfg["gold"])
    word_column = str(dataset_cfg["word_column"])
    score_column = str(dataset_cfg["score_column"])
    lowercase = bool(dataset_cfg.get("lowercase", True))
    pos_filter = dict(dataset_cfg.get("pos_filter", {}))
    match_mode = str(pos_filter.get("match_mode", "exact")).strip().lower()
    token_pattern = str(pos_filter.get("token_pattern", r"[,;/| ]+"))

    df = _read_gold_table(gold)
    required = [word_column, score_column]
    if bool(pos_filter.get("enabled", False)):
        required.append(str(pos_filter.get("pos_column", "Dom_Pos")))
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {gold}.")

    if bool(pos_filter.get("enabled", False)):
        pos_column = str(pos_filter.get("pos_column", "Dom_Pos"))
        tags = {str(tag).strip().lower() for tag in pos_filter.get("tags", ["Noun"])}
        df = df[[word_column, score_column, pos_column]].dropna()
        if match_mode not in {"exact", "token_contains"}:
            raise ValueError("dataset.pos_filter.match_mode must be 'exact' or 'token_contains'.")
        mask = _build_pos_mask(df[pos_column], tags=tags, match_mode=match_mode, token_pattern=token_pattern)
        df = df[mask]
    else:
        df = df[[word_column, score_column]].dropna()

    if df.empty:
        raise ValueError("No rows left after loading/filtering gold data.")

    df[word_column] = df[word_column].astype(str).map(lambda value: normalize_word(value, lowercase))
    df[score_column] = df[score_column].astype(float)
    return df.groupby(word_column, as_index=False)[score_column].mean()


def load_gold_arrays(dataset_cfg: dict) -> Tuple[List[str], np.ndarray]:
    """Return normalized words and scores arrays from gold data."""
    df = load_gold_df(dataset_cfg)
    word_column = str(dataset_cfg["word_column"])
    score_column = str(dataset_cfg["score_column"])
    words = df[word_column].tolist()
    scores = df[score_column].to_numpy(dtype=np.float32)
    return words, scores
