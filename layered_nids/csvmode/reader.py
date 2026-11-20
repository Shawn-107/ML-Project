"""Tolerant CSV reading. Real flow files are messy: comma, tab or semicolon separated, a byte-order mark before the first column,
padding spaces in headers and cells, blank or malformed lines, empty files. None of that should crash the tool or be silently ignored.

Everything is read as text; each schema decides how to turn columns into numbers (and reports what could not be converted).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


class CsvError(ValueError):
    """A problem with the input file that the user can fix (empty, no rows, unknown format, too large)."""


def detect_delimiter(path, probe_bytes: int = 65536) -> str:
    """Pick the separator from the header line: the candidate that occurs most often. Defaults to a comma."""
    with open(path, "rb") as f:
        head = f.read(probe_bytes).decode("utf-8-sig", errors="replace")
    line = next((ln for ln in head.splitlines() if ln.strip()), "")
    counts = {d: line.count(d) for d in (",", "\t", ";", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def count_data_lines(path) -> int:
    """Non-empty lines after the header (to report how many lines were skipped as malformed)."""
    n = 0
    with open(path, "rb") as f:
        next(f, None)
        for line in f:
            if line.strip():
                n += 1
    return n


def read_chunks(path, chunksize: int = 200_000, delimiter: str | None = None) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of text cells with lower-cased, stripped column names. Malformed lines are skipped (the caller can compare
    the rows it received with count_data_lines to report them). Raises CsvError for an empty file or one with a header only."""
    path = Path(path)
    if not path.exists():
        raise CsvError(f"file not found: {path}")
    if path.stat().st_size == 0:
        raise CsvError("the file is empty")
    sep = delimiter or detect_delimiter(path)
    try:
        reader = pd.read_csv(path, sep=sep, encoding="utf-8-sig", encoding_errors="replace", skipinitialspace=True, dtype=str,
                             keep_default_na=False, on_bad_lines="skip", chunksize=chunksize)
        got_rows = False
        for chunk in reader:
            if len(chunk) == 0:                              # a header-only file yields one empty chunk: it is not data
                continue
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            for c in chunk.columns:
                chunk[c] = chunk[c].str.strip()
            got_rows = True
            yield chunk
        if not got_rows:
            raise CsvError("the file has a header but no data rows")
    except pd.errors.EmptyDataError as e:
        raise CsvError("the file has no readable header") from e
    except pd.errors.ParserError as e:
        raise CsvError(f"the file could not be parsed as CSV: {e}") from e
