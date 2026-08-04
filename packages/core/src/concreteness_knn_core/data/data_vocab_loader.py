"""Vocabulary and OOV utility helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .data_gold_loader import normalize_word


def load_vocab_words(vocab_path: str, lowercase: bool) -> List[str]:
    """Load one token per line vocabulary file."""
    words: List[str] = []
    with open(vocab_path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                words.append(normalize_word(token, lowercase))
    return words


def oov_words(all_words: Iterable[str], kept_indices: Sequence[int]) -> List[str]:
    """Return words excluded by embedding coverage."""
    kept = {int(i) for i in kept_indices}
    return [word for i, word in enumerate(all_words) if i not in kept]
