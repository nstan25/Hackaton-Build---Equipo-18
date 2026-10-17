"""`make eval MODEL=<name> STAGE=<m2|m3|m4>` (SPEC §3.2). Stub de T00: T02 lo implementa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

THRESHOLDS = Path(__file__).with_name("thresholds.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", required=True, choices=["m2", "m3", "m4"])
    args = parser.parse_args(argv)
    thresholds = yaml.safe_load(THRESHOLDS.read_text())[args.stage]
    print(f"umbrales {args.stage}: {thresholds}", file=sys.stderr)
    print("eval/evaluate.py: no implementado todavía (lo hace T02)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
