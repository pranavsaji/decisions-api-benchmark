"""How each task reads an answer from a trained endpoint (19th or the local baseline)."""
import pytest

from bench.tasks import TASKS
from bench.tasks.intent import INTENTS


def table(winner: str, top: float = 0.6) -> dict:
    rest = (1 - top) / (len(INTENTS) - 1)
    return {intent: (top if intent == winner else rest) for intent in INTENTS}


def test_intent_baseline_ranks_by_probability():
    probabilities = table("card_arrival")
    probabilities["card_delivery_estimate"] = 0.2
    prediction = TASKS["intent"].to_prediction("local_baseline", {"probabilities": probabilities})
    assert (prediction["status"], prediction["label"], prediction["probability"]) == ("ok", "card_arrival", 0.6)
    assert prediction["ranked"][:2] == ["card_arrival", "card_delivery_estimate"] and len(prediction["ranked"]) == 3


def test_intent_19th_keeps_its_own_label():
    output = {"label": "atm_support", "probabilities": table("card_arrival")}
    prediction = TASKS["intent"].to_prediction("nineteenth", {"task": "intent", "output": output})
    assert prediction["label"] == "atm_support" and prediction["ranked"][1] == "card_arrival"


@pytest.mark.parametrize(
    "raw",
    [
        {"probabilities": {"card_arrival": 1.0}},  # most intents missing
        {"probabilities": {**table("card_arrival"), "atm_support": 1.5}},
        {"nothing": True},
        "text",
    ],
)
def test_intent_incomplete_or_invalid_tables_are_malformed(raw):
    assert TASKS["intent"].to_prediction("local_baseline", raw)["status"] == "malformed"


def test_injection_reads_the_attack_probability():
    baseline = TASKS["injection"].to_prediction("local_baseline", {"probabilities": {"attack": 0.7, "benign": 0.3}})
    assert (baseline["label"], baseline["probability"]) == ("attack", 0.7)
    ours = TASKS["injection"].to_prediction("nineteenth", {"task": "injection", "output": {"probability": 0.2}})
    assert (ours["label"], ours["probability"]) == ("benign", 0.2)


def test_relevance_baseline_takes_the_likeliest_grade_and_the_expected_grade():
    raw = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}}
    prediction = TASKS["relevance"].to_prediction("local_baseline", raw)
    assert (prediction["label"], prediction["probability"]) == (2, 0.6)
    assert prediction["expected"] == pytest.approx(1.7)


def test_relevance_19th_grade_can_differ_from_the_likeliest():
    output = {"grade": 1, "expected_grade": 1.4, "probabilities": {"0": 0.45, "1": 0.3, "2": 0.15, "3": 0.1}}
    prediction = TASKS["relevance"].to_prediction("nineteenth", {"task": "relevance", "output": output})
    assert (prediction["label"], prediction["expected"]) == (1, 1.4)
