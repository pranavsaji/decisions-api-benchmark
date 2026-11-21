"""The Task interface and helpers shared by every task module."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench import metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".tmp" / "data"


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    fields: list[tuple[str, str]]  # (item key, label shown in the UI)
    questions: list[dict]  # sent to the Decisions API as-is; exactly one question
    responses_schema: dict  # strict JSON schema for the Responses API
    format_note: str  # tells the Responses API how to fill the schema
    render_input: Callable[[dict], str]
    to_prediction: Callable[[str, object], dict]
    score: Callable[[list[dict], list[dict]], dict]
    headline: str  # key of the headline metric in score()["metrics"]
    examples: list[dict]
    # headline metric from (item, prediction) pairs; used to compare endpoints on the same items
    headline_value: Callable[[list[tuple[dict, dict]]], float | None]


# Endpoints that answer from a model trained on labelled examples (see bench/supervised.py).
TRAINED = ("nineteenth", "nineteenth_ultra", "local_baseline")


def blank(status: str) -> dict:
    return {"status": status, "label": None, "ranked": [], "probability": None, "expected": None}


def is_probability(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def unwrap(endpoint: str, raw) -> tuple[str, dict | None]:
    """Return ("ok", payload), ("refused", None) or ("malformed", None) for a raw endpoint answer."""
    if not isinstance(raw, dict):
        return "malformed", None
    if endpoint in TRAINED:
        from bench.supervised import reduced  # imported here: bench.supervised imports this module

        payload = reduced(endpoint, raw)
        return ("ok", payload) if payload else ("malformed", None)
    if endpoint == "decisions":
        answers = raw.get("answers")
        if not isinstance(answers, list) or not answers or not isinstance(answers[0], dict):
            return "malformed", None
        if answers[0].get("type") == "refusal":
            return "refused", None
        return "ok", answers[0]
    if raw.get("refused"):
        return "refused", None
    try:
        payload = json.loads(raw.get("text"))
    except (TypeError, ValueError):
        return "malformed", None
    return ("ok", payload) if isinstance(payload, dict) else ("malformed", None)


def probability_table(payload: dict, labels) -> dict | None:
    """A trained endpoint's probabilities keyed by each task label, or None if any is missing or invalid."""
    table = payload.get("probabilities") or {}
    values = {label: table.get(str(label)) for label in labels}
    return values if all(is_probability(v) for v in values.values()) else None


def correct(item: dict, prediction: dict) -> bool:
    return prediction["status"] == "ok" and prediction["label"] == item["label"]


def rate_metric(label: str, numerator: int, denominator: int) -> dict:
    return {"label": label, **metrics.rate(numerator, denominator), "format": "percent"}


def number_metric(label: str, value: float | None, ci: list[float] | None, n: int) -> dict:
    return {"label": label, "value": value, "ci": ci, "n": n, "format": "number"}


def breakdown(pairs: list[tuple[dict, dict]], key: Callable[[dict], str]) -> list[dict]:
    groups: dict[str, dict] = {}
    for item, prediction in pairs:
        group = groups.setdefault(key(item), {"group": key(item), "n": 0, "correct": 0})
        group["n"] += 1
        group["correct"] += correct(item, prediction)
    return sorted(groups.values(), key=lambda g: (g["correct"] / g["n"], g["group"]))


def instruction_block(task: Task) -> str:
    question = task.questions[0]
    lines = [question["instructions"]]
    if question["type"] == "choice":
        lines.append("Options:")
        lines += [f"- {c['value']}: {c['description']}" for c in question["choices"]]
    if question["type"] == "score":
        lines.append("Levels:")
        lines += [f"- {n}: {level['label']}. {level['description']}" for n, level in enumerate(question["levels"])]
    lines.append(task.format_note)
    return "\n".join(lines)


def responses_prompt(task: Task, item: dict) -> str:
    # Fixed instructions come first so that prompt caching can apply across items.
    return instruction_block(task) + "\n\n" + task.render_input(item)


def load_items(name: str) -> list[dict]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No data for task '{name}'. Run: uv run python -m tools.fetch")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
