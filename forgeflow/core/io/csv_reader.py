import csv
from pathlib import Path
from typing import Any, Iterable, Union

from forgeflow.interfaces import CsvStats, Record

PathLike = Union[str, Path]


def read_csv_rows(path: PathLike) -> Iterable[list[str]]:
    csv_path = Path(path)
    with csv_path.open(mode="r", encoding="utf-8", newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            yield row


def _coerce_value(value: str) -> Any:
    cell = value.strip()
    if cell == "":
        return ""

    try:
        return int(cell)
    except ValueError:
        pass

    try:
        return float(cell)
    except ValueError:
        return cell


def _new_csv_stats() -> CsvStats:
    return {
        "total_data_rows": 0,
        "valid_rows": 0,
        "skipped_bad_rows": 0,
    }


def _row_to_record(header: list[str], row: list[str]) -> Record:
    return {key: _coerce_value(value) for key, value in zip(header, row)}


def read_csv_records(path: PathLike) -> tuple[list[Record], CsvStats]:
    rows = read_csv_rows(path)
    header = next(rows, None)
    stats = _new_csv_stats()

    if header is None:
        return [], stats

    records: list[Record] = []
    for row in rows:
        stats["total_data_rows"] += 1
        if len(row) != len(header):
            stats["skipped_bad_rows"] += 1
            continue

        record = _row_to_record(header, row)
        records.append(record)
        stats["valid_rows"] += 1

    return records, stats


def read_csv_records_in_chunks(
    path: PathLike, chunk_size: int
) -> Iterable[tuple[list[Record], CsvStats]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    rows = read_csv_rows(path)
    header = next(rows, None)
    if header is None:
        return

    stats = _new_csv_stats()
    chunk: list[Record] = []

    for row in rows:
        stats["total_data_rows"] += 1
        if len(row) != len(header):
            stats["skipped_bad_rows"] += 1
            continue

        chunk.append(_row_to_record(header, row))
        stats["valid_rows"] += 1

        if len(chunk) >= chunk_size:
            yield chunk, dict(stats)
            chunk = []

    if chunk:
        yield chunk, dict(stats)
