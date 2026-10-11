import json

import pytest

from bench.tasks import base
from bench.tasks.intent import INTENTS, TASK


def decision(choice, probabilities):
    return {
        "answers": [
            {
                "type": "choice",
                "name": "intent",
                "choice": choice,
                "probabilities": [{"value": v, "probability": p} for v, p in probabilities],
                "confidence": 0.7,
            }
        ]
    }


def response(payload):
    return {"text": json.dumps(payload), "refused": False}


GOOD_RESPONSE = {
    "first_choice": "card_arrival",
    "second_choice": "card_delivery_estimate",
    "third_choice": "lost_or_stolen_card",
    "confidence": 0.8,
}


def test_there_are_77_intents_and_one_choice_question():
    assert len(INTENTS) == 77 == len(set(INTENTS))
    assert "reverted_card_payment?" in INTENTS and "Refund_not_showing_up" in INTENTS
    (question,) = TASK.questions
    assert question["type"] == "choice"
    assert [c["value"] for c in question["choices"]] == INTENTS


def test_decisions_answer_is_ranked_by_probability():
    raw = decision(
        "card_arrival",
        [
            ("age_limit", 0.01),
            ("lost_or_stolen_card", 0.04),
            ("card_arrival", 0.81),
            ("card_delivery_estimate", 0.12),
        ],
    )
    assert TASK.to_prediction("decisions", raw) == {
        "status": "ok",
        "label": "card_arrival",
        "ranked": ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card"],
        "probability": 0.81,
        "expected": None,
    }


def test_responses_answer_is_parsed_and_duplicates_collapse():
    prediction = TASK.to_prediction("responses_default", response(GOOD_RESPONSE))
    assert prediction["status"] == "ok"
    assert prediction["label"] == "card_arrival"
    assert prediction["ranked"] == ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card"]
    assert prediction["probability"] == 0.8
    repeated = {**GOOD_RESPONSE, "third_choice": "card_arrival"}
    assert TASK.to_prediction("responses_tuned", response(repeated))["ranked"] == [
        "card_arrival",
        "card_delivery_estimate",
    ]


def test_refusals_are_recognised_on_both_endpoints():
    refusal = {"answers": [{"type": "refusal", "name": "intent"}]}
    assert TASK.to_prediction("decisions", refusal) == base.blank("refused")
    assert TASK.to_prediction("responses_default", {"text": None, "refused": True}) == base.blank("refused")


@pytest.mark.parametrize(
    "endpoint, raw",
    [
        ("decisions", {"answers": []}),
        ("decisions", {}),
        ("decisions", None),
        ("decisions", decision("not_an_intent", [("not_an_intent", 0.9)])),
        ("decisions", decision("card_arrival", [])),
        ("decisions", decision("card_arrival", [("card_arrival", 81)])),
        ("decisions", decision(["card_arrival"], [("card_arrival", 0.8)])),
        ("decisions", {"answers": [{"type": "predicate", "name": "intent", "probability": 0.5}]}),
        ("responses_default", {"text": "not json", "refused": False}),
        ("responses_default", {"text": None, "refused": False}),
        ("responses_default", {"text": "[1, 2]", "refused": False}),
        ("responses_default", response({**GOOD_RESPONSE, "confidence": 85})),
        ("responses_default", response({**GOOD_RESPONSE, "confidence": True})),
        ("responses_default", response({**GOOD_RESPONSE, "first_choice": "made_up"})),
        ("responses_default", response({**GOOD_RESPONSE, "second_choice": ["card_arrival"]})),
        ("responses_default", response({"first_choice": "card_arrival"})),
    ],
)
def test_answers_outside_the_contract_are_malformed(endpoint, raw):
    assert TASK.to_prediction(endpoint, raw) == base.blank("malformed")


def test_score_counts_malformed_as_wrong_and_leaves_refusals_out():
    items = [{"id": str(n), "label": "card_arrival"} for n in range(4)]
    right = {**base.blank("ok"), "label": "card_arrival", "ranked": ["card_arrival"], "probability": 0.9}
    near = {
        **base.blank("ok"),
        "label": "card_delivery_estimate",
        "ranked": ["card_delivery_estimate", "card_arrival"],
        "probability": 0.6,
    }
    scored = TASK.score(items, [right, near, base.blank("malformed"), base.blank("refused")])
    top1 = scored["metrics"]["top1_accuracy"]
    top3 = scored["metrics"]["top3_accuracy"]
    assert (top1["value"], top1["n"], top1["format"]) == (pytest.approx(1 / 3), 3, "percent")
    assert (top3["value"], top3["n"]) == (pytest.approx(2 / 3), 3)
    assert scored["calibration"]["x_label"] == "Stated confidence"
    assert sum(b["n"] for b in scored["calibration"]["bins"]) == 2
    assert scored["breakdown"] == [{"group": "card_arrival", "n": 3, "correct": 1}]
    assert TASK.headline in scored["metrics"]


def test_score_with_no_answers_returns_empty_metrics():
    scored = TASK.score([], [])
    assert scored["metrics"]["top1_accuracy"]["value"] is None
    assert scored["calibration"]["error"] is None
    assert scored["breakdown"] == []


def test_responses_prompt_reuses_the_question_text_and_ends_with_the_input():
    item = {"id": "x", "text": "Where is my card?", "label": "card_arrival"}
    prompt = base.responses_prompt(TASK, item)
    assert prompt.startswith(TASK.questions[0]["instructions"])
    assert "- card_arrival: Card arrival." in prompt
    assert TASK.format_note in prompt
    assert prompt.endswith(TASK.render_input(item))


def test_load_items_names_the_fetch_command_when_data_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="tools.fetch"):
        base.load_items("intent")
    (tmp_path / "intent.jsonl").write_text('{"id": "a", "text": "hi", "label": "age_limit"}\n\n')
    assert base.load_items("intent") == [{"id": "a", "text": "hi", "label": "age_limit"}]
