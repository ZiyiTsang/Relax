# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from examples.sdpo.prepare_data import normalize_rows
from examples.sdpo.reward import score


def _sample(metadata: dict, response: str) -> SimpleNamespace:
    return SimpleNamespace(metadata=metadata, response=response, label="")


def test_normalize_sciknoweval_filters_l3_domains_and_preserves_split() -> None:
    rows = [
        {
            "question": "Which option?",
            "choices": {"text": ["one", "two"], "label": ["A", "B"]},
            "answerKey": "B",
            "type": "mcq-2-choices",
            "domain": "Physics",
            "details": {"level": "L3"},
        },
        {
            "question": "filtered",
            "choices": {"text": [], "label": []},
            "answerKey": "A",
            "type": "mcq-2-choices",
            "domain": "Physics",
            "details": {"level": "L2"},
        },
    ]

    normalized = normalize_rows("sciknoweval", rows, source_split="train")

    assert len(normalized) == 1
    assert normalized[0]["metadata"]["source_split"] == "train"
    assert normalized[0]["metadata"]["domain"] == "Physics"


def test_normalize_sciknoweval_accepts_reference_flat_domain_format() -> None:
    rows = [
        {
            "idx": 7,
            "dataset": "sciknoweval",
            "kind": "mcq",
            "answer": "C",
            "prompt": "Question\n\nA: one\nB: two\nC: three\nD: four",
            "system": "Return only the answer tag.",
        }
    ]

    normalized = normalize_rows("sciknoweval", rows, source_split="train", domain="material")

    assert len(normalized) == 1
    assert normalized[0]["label"] == "C"
    assert normalized[0]["metadata"]["domain"] == "Materials"
    assert "Return only the answer tag." in normalized[0]["prompt"]


def test_toolalpaca_reward_accepts_canonical_action_input() -> None:
    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [{"Action": "search", "Action_Input": '{"query": "relax"}'}],
        },
        'Action: search\nAction Input: {"query": "relax"}',
    )

    result = score(None, sample)

    assert result["score"] == 1.0
    assert result["feedback"] == ""


def test_reference_tooluse_row_is_normalized_for_the_same_reward() -> None:
    rows = [
        {
            "idx": 3,
            "dataset": "tooluse",
            "kind": "tooluse",
            "prompt": "Action: <tool>\nAction Input: <JSON>",
            "answer": '[{"Action": "search", "Action_Input": "{\\"query\\": \\"relax\\"}"}]',
        }
    ]

    normalized = normalize_rows("tooluse", rows, source_split="train")

    assert len(normalized) == 1
    assert normalized[0]["metadata"]["data_source"] == "tooluse"
    assert normalized[0]["metadata"]["golden_answer"][0]["Action"] == "search"


def test_reward_returns_feedback_without_exposing_gold_answer() -> None:
    sample = _sample(
        {
            "data_source": "toolalpaca",
            "golden_answer": [{"Action": "search", "Action_Input": '{"query": "relax"}'}],
        },
        'Action: lookup\nAction Input: {"query": "relax"}',
    )

    result = score(None, sample)

    assert result["score"] == 0.0
    assert "does not match" in result["feedback"]
    assert "search" not in result["feedback"]
