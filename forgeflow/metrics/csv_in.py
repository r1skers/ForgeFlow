import csv
from pathlib import Path
from typing import Any, Iterable, Union

PathLike = Union[str, Path]
CsvRecord = dict[str, Any]
CsvStats = dict[str, int]

def read_csv_rows(path: PathLike) -> Iterable[list[str]]:
    csv_path = Path(path)
    with csv_path.open(mode="r", encoding="utf-8", newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Keep parser output clean by dropping blank lines early.
            if not row:
                continue
            yield row

def _coerce_value(value: str) -> Any:
    cell = value.strip()
    if cell == "":
        return ""

    # Prefer integers before floats so "20" stays int instead of 20.0.
    try:
        return int(cell)
    except ValueError:
        pass

    try:
        return float(cell)
    except ValueError:
        return cell

def read_csv_records(path: PathLike) -> tuple[list[CsvRecord], CsvStats]:
    rows = read_csv_rows(path)
    # First non-empty row is treated as schema.
    header = next(rows, None)
    stats: CsvStats = {
        "total_data_rows": 0,
        "valid_rows": 0,
        "skipped_bad_rows": 0,
    }

    if header is None:
        return [], stats

    records: list[CsvRecord] = []
    for row in rows:
        stats["total_data_rows"] += 1
        # Drop malformed rows so downstream modeling sees aligned columns only.
        if len(row) != len(header):
            stats["skipped_bad_rows"] += 1
            continue

        # Build typed record keyed by header name.
        record = {key: _coerce_value(value) for key, value in zip(header, row)}
        records.append(record)
        stats["valid_rows"] += 1

    return records, stats
