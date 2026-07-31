# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys

from examples.sdpo.prepare_data import _read_rows, main


def test_read_rows_accepts_jsonl_with_json_suffix(tmp_path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        "\n".join(json.dumps(row) for row in ({"idx": 1}, {"idx": 2})) + "\n",
        encoding="utf-8",
    )

    assert _read_rows(path) == [{"idx": 1}, {"idx": 2}]


def test_read_rows_accepts_json_array(tmp_path) -> None:
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"idx": 1}, {"idx": 2}]), encoding="utf-8")

    assert _read_rows(path) == [{"idx": 1}, {"idx": 2}]


def test_prepare_data_main_writes_normalized_jsonl(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "chemistry" / "train.json"
    output_path = tmp_path / "prepared" / "chemistry.jsonl"
    input_path.parent.mkdir()
    input_path.write_text(
        json.dumps(
            {
                "idx": 4,
                "dataset": "sciknoweval",
                "kind": "mcq",
                "answer": "B",
                "prompt": "Choose B.",
                "system": "Return only the answer.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--dataset",
            "sciknoweval",
            "--input",
            str(input_path),
            "--domain",
            "chemistry",
            "--source-split",
            "train",
            "--output",
            str(output_path),
        ],
    )

    main()

    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["label"] == "B"
    assert row["metadata"]["source_split"] == "train"
    assert row["metadata"]["domain"] == "Chemistry"
