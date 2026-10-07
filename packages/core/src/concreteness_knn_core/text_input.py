"""Lossless Unicode text decoding and shared lexical normalization."""

from __future__ import annotations

import codecs
import io
import unicodedata
from pathlib import Path
from typing import TextIO


_ENCODING_ERROR = (
    "File is not valid Unicode text. Save or export it as UTF-8 "
    "(for spreadsheets, choose CSV UTF-8). UTF-16 and UTF-32 require a byte-order marker."
)


def _encoding(prefix: bytes) -> str:
    # UTF-32 LE starts with the UTF-16 LE marker, so check it first.
    if prefix.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return "utf-32"
    if prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    return "utf-8-sig"


def _validate_text(text: str) -> None:
    if "\x00" in text:
        raise ValueError("File contains NUL characters. Save or export it as plain UTF-8 text.")


def decode_text_input(payload: bytes) -> str:
    """Decode Unicode without guessing legacy encodings or replacing characters."""
    try:
        text = payload.decode(_encoding(payload[:4]))
    except UnicodeDecodeError as exc:
        raise ValueError(_ENCODING_ERROR) from exc
    _validate_text(text)
    return text


def open_text_input(path: str | Path, *, newline: str | None = None) -> TextIO:
    """Return a validated text stream; callers must close it (usually with `with`).

    Scan in bounded chunks before parsing: some CSV parsers silently truncate
    fields containing NULs. Rewind the same handle after validation.
    """
    raw = open(path, "rb")
    try:
        encoding = _encoding(raw.read(4))
        raw.seek(0)
        handle = io.TextIOWrapper(raw, encoding=encoding, newline=newline)
    except Exception:
        raw.close()
        raise
    try:
        while chunk := handle.read(64 * 1024):
            _validate_text(chunk)
        handle.seek(0)
    except UnicodeDecodeError as exc:
        handle.close()
        raise ValueError(_ENCODING_ERROR) from exc
    except Exception:
        handle.close()
        raise
    return handle


def normalize_word(word: str, lowercase: bool) -> str:
    """Match canonical Unicode spellings while preserving lexical distinctions."""
    token = unicodedata.normalize("NFC", str(word).strip())
    return unicodedata.normalize("NFC", token.lower()) if lowercase else token
