from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_ROOT = "https://api.waterdata.usgs.gov/ogcapi/v0"
DAILY_COLLECTION = "daily"


@dataclass(frozen=True)
class DailyValue:
    day: date
    value: float
    approval_status: str
    qualifier: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download USGS daily water temperature and build ForgeFlow train/infer CSV."
    )
    parser.add_argument("--site-id", default="USGS-01491000", help="USGS monitoring_location_id.")
    parser.add_argument(
        "--parameter-code",
        default="00010",
        help="USGS parameter code (00010 = Temperature, water).",
    )
    parser.add_argument(
        "--statistic-id",
        default="00003",
        help="USGS statistic id (00003 = Daily mean).",
    )
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--train-until",
        default="2024-12-31",
        help="Rows with date_t <= train-until go to train.csv (YYYY-MM-DD).",
    )
    parser.add_argument("--limit", type=int, default=2000, help="Page size for API calls.")
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Include non-approved rows. Default keeps Approved rows only.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="USGS API key. Empty means read from USGS_API_KEY env var or anonymous access.",
    )
    parser.add_argument(
        "--train-out",
        default="",
        help="Optional output path for train CSV. Relative paths are resolved from app root.",
    )
    parser.add_argument(
        "--infer-out",
        default="",
        help="Optional output path for infer CSV. Relative paths are resolved from app root.",
    )
    return parser


def build_request_url(
    site_id: str,
    parameter_code: str,
    statistic_id: str,
    start_date: str,
    end_date: str,
    limit: int,
) -> str:
    params = {
        "monitoring_location_id": site_id,
        "parameter_code": parameter_code,
        "statistic_id": statistic_id,
        "datetime": f"{start_date}/{end_date}",
        "limit": str(limit),
        "f": "json",
    }
    return f"{API_ROOT}/collections/{DAILY_COLLECTION}/items?{urlencode(params)}"


def fetch_json(url: str, api_key: str) -> dict[str, Any]:
    headers = {"Accept": "application/geo+json"}
    if api_key:
        headers["X-API-KEY"] = api_key
    request = Request(url=url, headers=headers, method="GET")
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("USGS response must be a JSON object")
    return payload


def fetch_all_features(initial_url: str, api_key: str) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    next_url: str | None = initial_url
    visited_urls: set[str] = set()

    while next_url is not None:
        if next_url in visited_urls:
            raise RuntimeError("pagination loop detected while fetching USGS data")
        visited_urls.add(next_url)

        payload = fetch_json(next_url, api_key)
        page_features = payload.get("features", [])
        if not isinstance(page_features, list):
            raise ValueError("USGS payload field 'features' must be a list")
        for item in page_features:
            if isinstance(item, dict):
                features.append(item)

        next_url = None
        links = payload.get("links", [])
        if isinstance(links, list):
            for link in links:
                if not isinstance(link, dict):
                    continue
                if str(link.get("rel", "")).strip().lower() == "next":
                    href = link.get("href")
                    if isinstance(href, str) and href:
                        next_url = href
                    break

    return features


def extract_daily_values(features: list[dict[str, Any]], approved_only: bool) -> list[DailyValue]:
    values: list[DailyValue] = []
    for feature in features:
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue

        raw_day = str(properties.get("time", "")).strip()
        raw_value = properties.get("value")
        if raw_day == "" or raw_value in ("", None):
            continue

        try:
            day = date.fromisoformat(raw_day[:10])
            value = float(raw_value)
        except ValueError:
            continue

        approval_status = str(properties.get("approval_status") or "").strip()
        if approved_only and approval_status.lower() != "approved":
            continue

        qualifier_raw = properties.get("qualifier")
        qualifier = "" if qualifier_raw is None else str(qualifier_raw).strip()
        values.append(
            DailyValue(
                day=day,
                value=value,
                approval_status=approval_status,
                qualifier=qualifier,
            )
        )

    # Keep one value per day to avoid duplicate records breaking lag pairing.
    by_day: dict[date, DailyValue] = {}
    for item in values:
        by_day[item.day] = item

    return [by_day[day] for day in sorted(by_day)]


def build_supervised_rows(values: list[DailyValue]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx in range(len(values) - 1):
        current = values[idx]
        nxt = values[idx + 1]

        dt_days = (nxt.day - current.day).days
        if dt_days <= 0:
            continue

        day_of_year = current.day.timetuple().tm_yday
        angle = 2.0 * math.pi * float(day_of_year) / 365.2425
        rows.append(
            {
                "date_t": current.day.isoformat(),
                "date_t1": nxt.day.isoformat(),
                "temp_t": f"{current.value:.6f}",
                "doy_sin": f"{math.sin(angle):.9f}",
                "doy_cos": f"{math.cos(angle):.9f}",
                "dt_days": f"{float(dt_days):.1f}",
                "y": f"{nxt.value:.6f}",
                "approval_t": current.approval_status,
                "approval_t1": nxt.approval_status,
                "qualifier_t": current.qualifier,
                "qualifier_t1": nxt.qualifier,
            }
        )
    return rows


def split_rows(
    rows: list[dict[str, str]],
    train_until: date,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    infer_rows: list[dict[str, str]] = []
    for row in rows:
        row_day = date.fromisoformat(row["date_t"])
        if row_day <= train_until:
            train_rows.append(row)
        else:
            infer_rows.append(row)
    return train_rows, infer_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "date_t",
        "date_t1",
        "temp_t",
        "doy_sin",
        "doy_cos",
        "dt_days",
        "y",
        "approval_t",
        "approval_t1",
        "qualifier_t",
        "qualifier_t1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_path(app_root: Path, default_path: Path, raw_path: str) -> Path:
    candidate = raw_path.strip()
    if candidate == "":
        return default_path
    path = Path(candidate)
    if path.is_absolute():
        return path
    return app_root / path


def main() -> None:
    args = build_parser().parse_args()

    date.fromisoformat(args.start_date)
    date.fromisoformat(args.end_date)
    train_until = date.fromisoformat(args.train_until)

    approved_only = not bool(args.include_provisional)
    api_key = args.api_key.strip() or os.getenv("USGS_API_KEY", "").strip()

    app_root = Path(__file__).resolve().parents[1]
    processed_dir = app_root / "data" / "processed"
    infer_dir = app_root / "data" / "infer"

    request_url = build_request_url(
        site_id=args.site_id,
        parameter_code=args.parameter_code,
        statistic_id=args.statistic_id,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )

    print(f"[usgs] request={request_url}")
    features = fetch_all_features(request_url, api_key)
    print(f"[usgs] fetched_features={len(features)}")

    values = extract_daily_values(features, approved_only=approved_only)
    print(f"[usgs] usable_daily_values={len(values)}")
    if len(values) < 3:
        raise RuntimeError("not enough usable daily values; adjust site/date filters")

    supervised_rows = build_supervised_rows(values)
    if len(supervised_rows) < 2:
        raise RuntimeError("not enough lag pairs to create train/infer data")

    train_rows, infer_rows = split_rows(supervised_rows, train_until=train_until)
    if not train_rows:
        raise RuntimeError("train.csv would be empty; move --train-until later")
    if not infer_rows:
        raise RuntimeError("infer.csv would be empty; move --train-until earlier")

    train_csv = resolve_output_path(app_root, processed_dir / "train.csv", args.train_out)
    infer_csv = resolve_output_path(app_root, infer_dir / "infer.csv", args.infer_out)
    write_csv(train_csv, train_rows)
    write_csv(infer_csv, infer_rows)

    print(f"[usgs] train_rows={len(train_rows)}")
    print(f"[usgs] infer_rows={len(infer_rows)}")
    print(f"[usgs] train_csv={train_csv}")
    print(f"[usgs] infer_csv={infer_csv}")


if __name__ == "__main__":
    main()
