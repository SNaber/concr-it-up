"""Command-line interface for core prediction workflows."""

from __future__ import annotations

import argparse

from .config import load_config, write_template
from .prediction import PredictionPipeline


def _require_config(path: str) -> str:
    if not path:
        raise ValueError("--config is required.")
    return path


def _cmd_prediction_run(args: argparse.Namespace) -> None:
    config_path = _require_config(args.config)
    config = load_config(config_path)
    outputs = PredictionPipeline().run_prediction(config, config_path=config_path)
    print("[INFO] Prediction run complete. Artifacts:")
    for key, value in outputs.items():
        print(f"  - {key}: {value}")


def _cmd_prediction_holdout(args: argparse.Namespace) -> None:
    config_path = _require_config(args.config)
    config = load_config(config_path)
    outputs = PredictionPipeline().run_holdout(config, config_path=config_path)
    print("[INFO] Holdout evaluation complete. Artifacts:")
    for key, value in outputs.items():
        print(f"  - {key}: {value}")


def _cmd_config_init(args: argparse.Namespace) -> None:
    out = write_template(task=args.task, out_path=args.out)
    print(f"[INFO] Wrote {args.task} template config to {out}")


def build_parser() -> argparse.ArgumentParser:
    """Build argparse command tree for the core toolkit."""
    parser = argparse.ArgumentParser(prog="concreteness-knn-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prediction = subparsers.add_parser("prediction", help="Prediction kNN workflows")
    prediction_sub = prediction.add_subparsers(dest="prediction_cmd", required=True)

    prediction_run = prediction_sub.add_parser("run", help="Train/evaluate prediction model")
    prediction_run.add_argument("--config", required=True, help="Path to JSON config")
    prediction_run.set_defaults(func=_cmd_prediction_run)

    prediction_holdout = prediction_sub.add_parser("holdout", help="Score predictions against holdout")
    prediction_holdout.add_argument("--config", required=True, help="Path to JSON config")
    prediction_holdout.set_defaults(func=_cmd_prediction_holdout)

    config_cmd = subparsers.add_parser("config", help="Config utilities")
    config_sub = config_cmd.add_subparsers(dest="config_cmd", required=True)
    config_init = config_sub.add_parser("init", help="Write starter config")
    config_init.add_argument("--task", required=True, choices=["prediction"])
    config_init.add_argument("--out", required=True, help="Output JSON path")
    config_init.set_defaults(func=_cmd_config_init)

    return parser


def main() -> None:
    """Run the core CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
