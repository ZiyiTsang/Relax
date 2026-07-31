# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Normalize SciKnowEval and ToolAlpaca rows for the Relax rollout loader.

The command intentionally performs no implicit train/test split.  The input
file is the split selected by the experiment owner, and the output records
retain ``metadata['source_split']`` for leakage checks.  This is important for
SciKnowEval because a locally mounted copy may contain only benchmark test
records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from examples.sdpo.data_normalizers import normalize_rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".jsonl":
            return _read_jsonl(path)
        text = path.read_text(encoding="utf-8").strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return _read_jsonl(path)
        return value if isinstance(value, list) else [value]
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as parquet

        return parquet.read_table(path).to_pylist()
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sciknoweval", "toolalpaca", "tooluse"), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-split", required=True, help="Metadata label such as train or test.")
    parser.add_argument(
        "--domain",
        default=None,
        help="SciKnowEval domain for the reference flat format; defaults to the input parent directory name.",
    )
    args = parser.parse_args()

    rows = _read_rows(args.input)
    domain = args.domain or args.input.parent.name
    normalized = normalize_rows(args.dataset, rows, source_split=args.source_split, domain=domain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
