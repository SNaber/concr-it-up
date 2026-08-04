"""Embedding loader dispatch and multi-space construction."""

from __future__ import annotations

from typing import Dict

from .data_embeddings_protocol import Embeddings
from .data_embeddings_fasttext import FastTextBinEmbeddings
from .data_embeddings_word2vec import Word2VecTextEmbeddings


def load_embeddings(kind: str, path: str) -> Embeddings:
    """Factory for embedding adapters."""
    if kind == "ft_bin":
        return FastTextBinEmbeddings(path)
    if kind == "vec":
        return Word2VecTextEmbeddings(path)
    raise ValueError("embeddings.spaces[*].kind must be one of: ft_bin, vec")


def load_embeddings_by_space(embeddings_cfg: dict) -> Dict[str, Embeddings]:
    """Load only the embedding spaces active for the configured mode."""
    mode = str(embeddings_cfg.get("mode", "single"))
    active_space = str(embeddings_cfg.get("active_space", ""))
    out: Dict[str, Embeddings] = {}
    for space in embeddings_cfg.get("spaces", []):
        space_id = str(space["id"])
        if mode == "single" and space_id != active_space:
            continue
        out[space_id] = load_embeddings(str(space["kind"]), str(space["path"]))
    return out
