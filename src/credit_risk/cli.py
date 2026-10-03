from __future__ import annotations

import argparse

from credit_risk.data.processing import build_labeled_dataset
from credit_risk.logging_utils import configure_logging
from credit_risk.training.trainer import train


def process_main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build labeled training dataset")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    build_labeled_dataset(use_synthetic=args.synthetic)


def train_main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train credit-risk models")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    train(use_synthetic=args.synthetic, quick=args.quick)


def serve_main() -> None:
    configure_logging()
    import uvicorn

    uvicorn.run("credit_risk.api.app:app", host="0.0.0.0", port=8000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit-risk production CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    process_parser = sub.add_parser("process", help="Build labeled dataset")
    process_parser.add_argument("--synthetic", action="store_true")

    train_parser = sub.add_parser("train", help="Train and persist champion model")
    train_parser.add_argument("--synthetic", action="store_true")
    train_parser.add_argument("--quick", action="store_true")

    sub.add_parser("serve", help="Start the scoring API")
    args = parser.parse_args()

    if args.command == "process":
        configure_logging()
        build_labeled_dataset(use_synthetic=args.synthetic)
    elif args.command == "train":
        configure_logging()
        train(use_synthetic=args.synthetic, quick=args.quick)
    elif args.command == "serve":
        serve_main()


if __name__ == "__main__":
    main()

