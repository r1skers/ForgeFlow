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


def read_csv_records(path: PathLike) -> tuple[list[Record], CsvStats]:
    rows = read_csv_rows(path)
    header = next(rows, None)
    stats: CsvStats = {
        "total_data_rows": 0,
        "valid_rows": 0,
        "skipped_bad_rows": 0,
    }

    if header is None:
        return [], stats

    records: list[Record] = []
    for row in rows:
        stats["total_data_rows"] += 1
        if len(row) != len(header):
            stats["skipped_bad_rows"] += 1
            continue

        record = {key: _coerce_value(value) for key, value in zip(header, row)}
        records.append(record)
        stats["valid_rows"] += 1

    return records, stats
