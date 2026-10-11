import json

import pytest

from bench.tasks import TASKS, base
from bench.tasks.relevance import LEVELS, TASK


def decision(score, probabilities):
    return {
        "answers": [
            {
                "type": "score",
                "name": "relevance",
                "score": score,
                "probabilities": [
                    {"value": n, "label": LEVELS[n]["label"], "probability": p}
                    for n, p in enumerate(probabilities)
                ],
                "confidence": 0.5,
            }
        ]
    }


def response(payload):
    return {"text": json.dumps(payload), "refused": False}


def ok(label, expected=None):
    return {**base.blank("ok"), "label": label, "probability": 0.5, "expected": float(label) if expected is None else expected}


def test_registry_lists_the_three_tasks_in_order():
    assert list(TASKS) == ["intent", "injection", "relevance"]
    assert all(len(task.questions) == 1 for task in TASKS.values())
    assert all(TASKS[name].name == name for name in TASKS)


def test_one_score_question_with_four_levels():
    (question,) = TASK.questions
    assert question["type"] == "score"
    assert [level["label"] for level in question["levels"]] == [
        "Irrelevant",
        "Related",
        "Highly relevant",
        "Perfectly relevant",
    ]


def test_decisions_answer_takes_the_most_likely_level_and_the_reported_score():
    prediction = TASK.to_prediction("decisions", decision(1.9, [0.05, 0.2, 0.55, 0.2]))
    assert prediction == {"status": "ok", "label": 2, "ranked": [], "probability": 0.55, "expected": 1.9}


def test_decisions_tie_goes_to_the_lower_level():
    assert TASK.to_prediction("decisions", decision(1.5, [0.0, 0.5, 0.5, 0.0]))["label"] == 1


def test_responses_answer_computes_the_expected_score_from_normalised_probabilities():
    payload = {"grade": 3, "p0": 0.0, "p1": 0.1, "p2": 0.3, "p3": 0.4}
    prediction = TASK.to_prediction("responses_default", response(payload))
    assert prediction["status"] == "ok" and prediction["label"] == 3
    assert prediction["expected"] == pytest.approx((0.1 + 0.6 + 1.2) / 0.8)
    assert prediction["probability"] == pytest.approx(0.5)


def test_refusals_are_recognised():
    assert TASK.to_prediction("decisions", {"answers": [{"type": "refusal", "name": "relevance"}]}) == base.blank("refused")
    assert TASK.to_prediction("responses_tuned", {"text": None, "refused": True}) == base.blank("refused")


@pytest.mark.parametrize(
    "endpoint, raw",
    [
        ("decisions", decision(1.9, [0.1, 0.9])),
        ("decisions", decision(4.2, [0.1, 0.2, 0.3, 0.4])),
        ("decisions", decision(None, [0.1, 0.2, 0.3, 0.4])),
        ("decisions", decision(1.0, [0.1, 20, 0.3, 0.4])),
        ("decisions", {"answers": [{"type": "score", "name": "relevance", "score": 1.0}]}),
        ("responses_default", response({"grade": 5, "p0": 0.1, "p1": 0.2, "p2": 0.3, "p3": 0.4})),
        ("responses_default", response({"grade": True, "p0": 0.1, "p1": 0.2, "p2": 0.3, "p3": 0.4})),
        ("responses_default", response({"grade": 1, "p0": 0, "p1": 0, "p2": 0, "p3": 0})),
        ("responses_default", response({"grade": 1, "p0": 10, "p1": 70, "p2": 15, "p3": 5})),
        ("responses_default", response({"grade": 1})),
    ],
)
def test_answers_outside_the_contract_are_malformed(endpoint, raw):
    assert TASK.to_prediction(endpoint, raw) == base.blank("malformed")


def test_score_reports_kappa_on_parsed_answers_and_accuracy_with_malformed_wrong():
    truth = [0, 0, 1, 1, 2, 2, 3, 3]
    predicted = [0, 1, 1, 1, 2, 3, 3, 1]
    items = [{"id": str(n), "label": label} for n, label in enumerate(truth)] + [
        {"id": "m", "label": 2},
        {"id": "r", "label": 2},
    ]
    predictions = [ok(p) for p in predicted] + [base.blank("malformed"), base.blank("refused")]
    scored = TASK.score(items, predictions)
    kappa = scored["metrics"]["kappa"]
    assert (kappa["value"], kappa["n"]) == (pytest.approx(2 / 3), 8)
    assert kappa["ci"][0] <= 2 / 3 <= kappa["ci"][1]
    exact = scored["metrics"]["exact_accuracy"]
    assert (exact["value"], exact["n"]) == (pytest.approx(5 / 9), 9)
    assert (scored["calibration"]["low"], scored["calibration"]["high"]) == (0.0, 3.0)
    assert len(scored["calibration"]["bins"]) == 6
    groups = {g["group"]: (g["n"], g["correct"]) for g in scored["breakdown"]}
    assert groups["Grade 2: Highly relevant"] == (3, 1)
    assert TASK.headline == "kappa"


def test_score_with_no_answers_returns_empty_metrics():
    scored = TASK.score([], [])
    assert scored["metrics"]["kappa"]["value"] is None
    assert scored["metrics"]["exact_accuracy"]["value"] is None


def test_input_shows_the_query_then_the_passage():
    assert TASK.render_input({"query": "q?", "passage": "p."}) == "Query: q?\n\nPassage: p."
