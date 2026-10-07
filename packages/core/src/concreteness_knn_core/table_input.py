"""Shared, strict CSV/TSV parsing for gold ratings."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .text_input import open_text_input


@contextmanager
def open_gold_table(path: str | Path) -> Iterator[tuple[list[str], Iterator[list[str]]]]:
    """Read a table without inferring an index or shifting malformed columns.

    Keep fields as text. Blank lines are ignored; every other record must match
    the header, including explicit empty fields for missing values.
    """
    primary = "\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ","
    with open_text_input(path, newline="") as handle:
        reader = csv.reader(handle, delimiter=primary, strict=True)
        try:
            header = next((row for row in reader if row and any(cell.strip() for cell in row)), [])
            if len(header) <= 1:
                handle.seek(0)
                reader = csv.reader(handle, delimiter="," if primary == "\t" else "\t", strict=True)
                header = next((row for row in reader if row and any(cell.strip() for cell in row)), [])
            if not header:
                raise ValueError("Gold file is empty.")
            names = [name.strip() for name in header]
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("Gold column names must be non-empty and unique.")

            def rows() -> Iterator[list[str]]:
                for row in reader:
                    if not row or (len(row) == 1 and not row[0].strip()):
                        continue
                    if len(row) != len(header):
                        raise ValueError(
                            f"Gold table line {reader.line_num}: expected {len(header)} columns, "
                            f"found {len(row)}. Check delimiters and quote fields containing them."
                        )
                    yield row

            yield header, rows()
        except csv.Error as exc:
            raise ValueError(f"Invalid gold table near line {reader.line_num}: {exc}") from exc
