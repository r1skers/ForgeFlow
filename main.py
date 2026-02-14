import argparse
from pathlib import Path

from forgeflow.core.runner import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ForgeFlow framework runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/linear_xy/config.json"),
        help="Path to runtime configuration file (JSON).",
    )
    return parser


if __name__ == "__main__":
    project_root = Path(__file__).parent
    args = build_arg_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    run_pipeline(config_path, project_root)
