from pathlib import Path

from forgeflow.metrics.csv_in import read_csv_records
from processors import show

if __name__ == "__main__":
    # Default to repository sample data for quick local runs.
    default_csv = Path(__file__).parent / "forgeflow" / "Input" / "test" / "test.csv"
    records, stats = read_csv_records(default_csv)

    # Stage 1 output: structured records.
    for record in records:
        show(record)

    # Stage 1 quality gate: row-level ingestion stats.
    print(f"[csv] total_data_rows={stats['total_data_rows']}")
    print(f"[csv] valid_rows={stats['valid_rows']}")
    print(f"[csv] skipped_bad_rows={stats['skipped_bad_rows']}")
