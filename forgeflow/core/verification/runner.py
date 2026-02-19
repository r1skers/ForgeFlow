import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VerificationStepConfig:
    name: str
    command: list[str]
    report_csv: Path
    status_field: str
    pass_value: str


@dataclass(frozen=True)
class VerificationConfig:
    task: str
    summary_csv: Path
    steps: list[VerificationStepConfig]


def _resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_config(config_path: Path) -> VerificationConfig:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    task = str(payload.get("task", "")).strip()
    if not task:
        raise ValueError("verification.task is required")

    summary_raw = payload.get("summary_csv")
    if summary_raw is None:
        raise ValueError("verification.summary_csv is required")

    base_dir = config_path.parent
    summary_csv = _resolve_path(str(summary_raw), base_dir)

    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("verification.steps must be a non-empty array")

    steps: list[VerificationStepConfig] = []
    for idx, raw_step in enumerate(steps_payload):
        if not isinstance(raw_step, dict):
            raise ValueError(f"verification.steps[{idx}] must be an object")

        name = str(raw_step.get("name", "")).strip()
        if not name:
            raise ValueError(f"verification.steps[{idx}].name is required")

        raw_command = raw_step.get("command")
        if not isinstance(raw_command, list) or not raw_command:
            raise ValueError(f"verification.steps[{idx}].command must be a non-empty array")
        command = [str(item) for item in raw_command]

        raw_report_csv = raw_step.get("report_csv")
        if raw_report_csv is None:
            raise ValueError(f"verification.steps[{idx}].report_csv is required")
        report_csv = _resolve_path(str(raw_report_csv), base_dir)

        status_field = str(raw_step.get("status_field", "status")).strip()
        pass_value = str(raw_step.get("pass_value", "PASS")).strip()
        if not status_field:
            raise ValueError(f"verification.steps[{idx}].status_field cannot be empty")
        if not pass_value:
            raise ValueError(f"verification.steps[{idx}].pass_value cannot be empty")

        steps.append(
            VerificationStepConfig(
                name=name,
                command=command,
                report_csv=report_csv,
                status_field=status_field,
                pass_value=pass_value,
            )
        )

    return VerificationConfig(
        task=task,
        summary_csv=summary_csv,
        steps=steps,
    )


def _evaluate_statuses(
    report_csv: Path, status_field: str, pass_value: str
) -> tuple[str, int, int, str]:
    if not report_csv.exists():
        return "FAIL", 0, 0, "report_missing"

    with report_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return "FAIL", 0, 0, "report_empty"

    if any(status_field not in row for row in rows):
        return "FAIL", len(rows), len(rows), f"missing_status_field:{status_field}"

    failed_rows = sum(1 for row in rows if str(row[status_field]).strip() != pass_value)
    status = "PASS" if failed_rows == 0 else "FAIL"
    return status, len(rows), failed_rows, ""


def _resolve_command(command: list[str]) -> list[str]:
    resolved: list[str] = []
    for token in command:
        if token == "{python}":
            resolved.append(sys.executable)
        else:
            resolved.append(token)
    return resolved


def _run_step(step: VerificationStepConfig) -> dict[str, Any]:
    command = _resolve_command(step.command)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    command_ok = completed.returncode == 0
    if command_ok:
        status, row_count, failed_rows, note = _evaluate_statuses(
            report_csv=step.report_csv,
            status_field=step.status_field,
            pass_value=step.pass_value,
        )
    else:
        status = "FAIL"
        row_count = 0
        failed_rows = 0
        note = f"command_failed:returncode={completed.returncode}"

    stdout_preview = completed.stdout.strip().splitlines()
    stderr_preview = completed.stderr.strip().splitlines()
    stdout_tail = " | ".join(stdout_preview[-2:]) if stdout_preview else ""
    stderr_tail = " | ".join(stderr_preview[-2:]) if stderr_preview else ""

    return {
        "step": step.name,
        "status": status,
        "return_code": completed.returncode,
        "command": " ".join(command),
        "report_csv": str(step.report_csv),
        "status_field": step.status_field,
        "pass_value": step.pass_value,
        "row_count": row_count,
        "failed_rows": failed_rows,
        "note": note,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def run_verification(config_path: Path) -> Path:
    resolved_config = config_path if config_path.is_absolute() else (PROJECT_ROOT / config_path)
    config = _load_config(resolved_config)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for step in config.steps:
        result = _run_step(step)
        rows.append(
            {
                "timestamp_utc": timestamp_utc,
                "task": config.task,
                **result,
            }
        )

    overall_status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    summary_row = {
        "timestamp_utc": timestamp_utc,
        "task": config.task,
        "step": "_overall_",
        "status": overall_status,
        "return_code": 0,
        "command": "-",
        "report_csv": "-",
        "status_field": "-",
        "pass_value": "-",
        "row_count": sum(int(row["row_count"]) for row in rows),
        "failed_rows": sum(int(row["failed_rows"]) for row in rows),
        "note": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    rows.append(summary_row)

    config.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.summary_csv.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "timestamp_utc",
                "task",
                "step",
                "status",
                "return_code",
                "command",
                "report_csv",
                "status_field",
                "pass_value",
                "row_count",
                "failed_rows",
                "note",
                "stdout_tail",
                "stderr_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[verify] config={resolved_config}")
    for row in rows:
        print(
            f"[verify:{row['step']}] status={row['status']} "
            f"failed_rows={row['failed_rows']} note={row['note']}"
        )
    print(f"[verify] summary_file={config.summary_csv}")
    return config.summary_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configured verification steps and summarize results.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Verification config JSON file.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_verification(args.config)


if __name__ == "__main__":
    main()
