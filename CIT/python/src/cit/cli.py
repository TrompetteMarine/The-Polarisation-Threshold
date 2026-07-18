from __future__ import annotations

import argparse
from pathlib import Path
from .pipeline import run_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CIT numerical replication pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--publication", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    report = run_pipeline(args.output, publication=args.publication)
    for key, value in report.items():
        print(f"{key}: {value:.8g}")

if __name__ == "__main__":
    main()
