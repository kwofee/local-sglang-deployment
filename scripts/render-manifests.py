#!/usr/bin/env python3
"""Inject machine-local values into rendered Kubernetes manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODEL_PATH_MARKER = "__SGLANG_MODEL_PATH__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="absolute host path containing the downloaded model",
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser()
    if not model_path.is_absolute():
        parser.error("--model-path must be absolute")

    rendered = sys.stdin.read()
    occurrences = rendered.count(MODEL_PATH_MARKER)
    if occurrences != 1:
        raise SystemExit(
            f"expected one {MODEL_PATH_MARKER} marker, found {occurrences}"
        )

    # JSON strings are valid YAML scalars and safely preserve spaces and symbols.
    sys.stdout.write(rendered.replace(MODEL_PATH_MARKER, json.dumps(str(model_path))))


if __name__ == "__main__":
    main()
