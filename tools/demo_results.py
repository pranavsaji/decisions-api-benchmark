"""Write made-up reports to .tmp/demo-results so the dashboard can be checked without API calls.

Usage: uv run python -m tools.demo_results
Then: RESULTS_DIR=.tmp/demo-results uv run uvicorn app.server:app --port 8643
The numbers are random. Never copy them into results/ or the README.
"""
import json
import random

from bench import report
from bench.endpoints import ENDPOINTS
from bench.tasks import TASKS
from bench.tasks.base import ROOT, load_items
from bench.tasks.intent import INTENTS

OUT = ROOT / ".tmp" / "demo-results"
SPEED = {"decisions": 180, "responses_default": 3500, "responses_tuned": 700}


def fake_raw(task_name: str, endpoint: str, item: dict, rng: random.Random) -> dict:
    right = rng.random() < 0.8
    if task_name == "intent":
        choice = item["label"] if right else rng.choice(INTENTS)
        others = rng.sample([i for i in INTENTS if i != choice], 2)
        confidence = round(rng.uniform(0.5, 0.99), 2)
        if endpoint == "decisions":
            table = [(choice, confidence), (others[0], 0.05), (others[1], 0.02)]
            answer = {"type": "choice", "name": "intent", "choice": choice, "probabilities": [{"value": v, "probability": p} for v, p in table]}
            return {"answers": [answer]}
        payload = {"first_choice": choice, "second_choice": others[0], "third_choice": others[1], "confidence": confidence}
    elif task_name == "injection":
        attack = (item["label"] == "attack") == right
        probability = round(rng.uniform(0.55, 0.98) if attack else rng.uniform(0.02, 0.45), 2)
        if endpoint == "decisions":
            return {"answers": [{"type": "predicate", "name": "injection", "probability": probability}]}
        payload = {"probability": probability}
    else:
        grade = item["label"] if right else max(0, min(3, item["label"] + rng.choice([-1, 1])))
        spread = [0.1, 0.1, 0.1, 0.1]
        spread[grade] = 0.7
        if endpoint == "decisions":
            score = sum(n * p for n, p in enumerate(spread))
            answer = {"type": "score", "name": "relevance", "score": score, "probabilities": [{"value": n, "probability": p} for n, p in enumerate(spread)]}
            return {"answers": [answer]}
        payload = {"grade": grade, **{f"p{n}": p for n, p in enumerate(spread)}}
    return {"text": json.dumps(payload), "refused": False}


def main() -> None:
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, task in TASKS.items():
        items = load_items(name)[:120]
        cache = {}
        for item in items:
            for endpoint in ENDPOINTS:
                cache[(item["id"], endpoint)] = {
                    "item_id": item["id"],
                    "endpoint": endpoint,
                    "status": "ok",
                    "raw": fake_raw(name, endpoint, item, rng),
                    "latency_ms": round(rng.lognormvariate(0, 0.4) * SPEED[endpoint]),
                    "input_tokens": 400,
                    "cached_tokens": 0 if endpoint == "decisions" else 200,
                    "output_tokens": 0 if endpoint == "decisions" else 250,
                }
        # One endpoint with nothing at all, to check the "no data" display.
        if name == "relevance":
            cache = {key: row for key, row in cache.items() if key[1] != "responses_tuned"}
        built = report.build_task_report(task, items, cache, {"name": "DEMO DATA, NOT REAL RESULTS"})
        (OUT / f"{name}.json").write_text(json.dumps(built))
        reports.append(built)
    (OUT / "summary.json").write_text(json.dumps(report.build_summary(reports)))
    print(f"Wrote demo reports to {OUT}")


if __name__ == "__main__":
    main()
