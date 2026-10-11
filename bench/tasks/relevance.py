"""Search relevance grading on TREC Deep Learning 2019: one `score` question with four levels."""
from bench import metrics
from bench.tasks.base import (
    Task,
    blank,
    breakdown,
    correct,
    is_probability,
    number_metric,
    rate_metric,
    unwrap,
)

# The TREC Deep Learning passage-judgement scale.
LEVELS = [
    {"label": "Irrelevant", "description": "The passage has nothing to do with the query."},
    {"label": "Related", "description": "The passage is on the topic of the query but does not answer it."},
    {
        "label": "Highly relevant",
        "description": "The passage has some answer to the query, but the answer may be unclear or buried in other material.",
    },
    {
        "label": "Perfectly relevant",
        "description": "The passage is dedicated to the query and contains the exact answer.",
    },
]
GRADES = range(len(LEVELS))

QUESTION = {
    "type": "score",
    "name": "relevance",
    "instructions": "How well does the passage answer the query?",
    "levels": LEVELS,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "integer", "enum": list(GRADES)},
        **{f"p{n}": {"type": "number"} for n in GRADES},
    },
    "required": ["grade", *(f"p{n}" for n in GRADES)],
    "additionalProperties": False,
}

FORMAT_NOTE = (
    "Reply with JSON. grade is the number of the best-matching level. p0, p1, p2 and p3 are the "
    "probabilities, between 0 and 1, of levels 0, 1, 2 and 3."
)


def render_input(item: dict) -> str:
    return f"Query: {item['query']}\n\nPassage: {item['passage']}"


def _is_grade(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in GRADES


def to_prediction(endpoint: str, raw) -> dict:
    status, payload = unwrap(endpoint, raw)
    if status != "ok":
        return blank(status)
    if endpoint == "decisions":
        if payload.get("type") != "score":
            return blank("malformed")
        table = {}
        for entry in payload.get("probabilities") or []:
            if isinstance(entry, dict) and _is_grade(entry.get("value")):
                table[entry["value"]] = entry.get("probability")
        probabilities = [table.get(n) for n in GRADES]
        expected = payload.get("score")
        if not all(is_probability(p) for p in probabilities):
            return blank("malformed")
        if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not 0 <= expected <= GRADES[-1]:
            return blank("malformed")
        label = max(GRADES, key=lambda n: (probabilities[n], -n))
        return {**blank("ok"), "label": label, "probability": probabilities[label], "expected": expected}
    grade = payload.get("grade")
    probabilities = [payload.get(f"p{n}") for n in GRADES]
    if not _is_grade(grade) or not all(is_probability(p) for p in probabilities):
        return blank("malformed")
    total = sum(probabilities)
    if total <= 0:
        return blank("malformed")
    expected = sum(n * p for n, p in zip(GRADES, probabilities)) / total
    return {**blank("ok"), "label": grade, "probability": probabilities[grade] / total, "expected": expected}


def _kappa(rows: list[tuple[int, int]]) -> float | None:
    return metrics.quadratic_weighted_kappa([t for t, _ in rows], [p for _, p in rows], len(LEVELS))


def score(items: list[dict], predictions: list[dict]) -> dict:
    pairs = [(i, p) for i, p in zip(items, predictions) if p["status"] != "refused"]
    parsed = [(i, p) for i, p in pairs if p["status"] == "ok"]
    rows = [(i["label"], p["label"]) for i, p in parsed]
    calibration = metrics.reliability(
        [p["expected"] for _, p in parsed], [i["label"] for i, _ in parsed], 0.0, float(GRADES[-1]), 6
    )
    return {
        "metrics": {
            "kappa": number_metric(
                "Agreement with human grades (weighted kappa)", _kappa(rows), metrics.bootstrap_ci(_kappa, rows), len(rows)
            ),
            "exact_accuracy": rate_metric("Exact grade matched", sum(correct(i, p) for i, p in pairs), len(pairs)),
        },
        "calibration": {**calibration, "x_label": "Expected grade from the model", "y_label": "Mean human grade"},
        "breakdown": breakdown(pairs, lambda item: f"Grade {item['label']}: {LEVELS[item['label']]['label']}"),
    }


TASK = Task(
    name="relevance",
    title="Search relevance",
    fields=[("query", "Search query"), ("passage", "Passage")],
    questions=[QUESTION],
    responses_schema=SCHEMA,
    format_note=FORMAT_NOTE,
    render_input=render_input,
    to_prediction=to_prediction,
    score=score,
    headline="kappa",
    examples=[
        {
            "query": "how long does it take to boil an egg",
            "passage": "For a hard-boiled egg, lower it into boiling water and cook for 9 to 12 minutes. For a soft yolk, 6 minutes is enough.",
        },
        {
            "query": "how long does it take to boil an egg",
            "passage": "Eggs are a good source of protein and are eaten at breakfast in many countries.",
        },
    ],
)
