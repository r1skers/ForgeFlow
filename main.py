import argparse
import logging
from pathlib import Path

from forgeflow.core.runner import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ForgeFlow framework runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/linear_xy/config/run.json"),
        help="Path to runtime configuration file (JSON).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for console output.",
    )
    return parser


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    run_pipeline(config_path, project_root)
