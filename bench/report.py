"""Turn the cached answers into results/<task>.json and results/summary.json.

Usage: uv run python -m bench.report [--markdown]
Makes no API calls. --markdown also prints each task's comparison table for the README.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from bench import metrics
from bench.endpoints import ENDPOINTS, MODEL
from bench.runner import cache_path, load_cache
from bench.tasks import TASKS
from bench.tasks.base import DATA, ROOT, correct, load_items

RESULTS = ROOT / "results"
ENDPOINT_TITLES = {
    "decisions": "Decisions API",
    "responses_default": "Responses API, default",
    "responses_tuned": "Responses API, tuned",
}


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise_endpoint(task, items: list[dict], cache: dict, endpoint: str) -> tuple[dict, dict]:
    ran = [(item, cache[(item["id"], endpoint)]) for item in items if (item["id"], endpoint) in cache]
    answered = [(item, row) for item, row in ran if row["status"] == "ok"]
    predictions = [task.to_prediction(endpoint, row["raw"]) for _, row in answered]
    statuses = Counter(prediction["status"] for prediction in predictions)
    scored = task.score([item for item, _ in answered], predictions)
    latencies = [row["latency_ms"] for _, row in answered]
    tokens = {
        "input": _mean([row.get("input_tokens") for _, row in answered]),
        "cached": _mean([row.get("cached_tokens") for _, row in answered]),
        "output": _mean([row.get("output_tokens") for _, row in answered]),
    }
    summary = {
        "counts": {
            "planned": len(items),
            "answered": statuses["ok"],
            "refused": statuses["refused"],
            "malformed": statuses["malformed"],
            "errored": len(ran) - len(answered),
            "missing": len(items) - len(ran),
        },
        **scored,
        "latency_ms": {
            "p50": metrics.percentile(latencies, 0.5),
            "p95": metrics.percentile(latencies, 0.95),
            "mean": _mean(latencies),
            "histogram": metrics.histogram(latencies, metrics.LATENCY_EDGES_MS),
        },
        "tokens": tokens,
        "cost_per_1000": metrics.cost_per_1000(tokens["input"], tokens["cached"], tokens["output"]),
    }
    answers = {
        item["id"]: {
            "status": prediction["status"],
            "label": prediction["label"],
            "probability": prediction["probability"],
            "expected": prediction["expected"],
            "correct": correct(item, prediction),
            "latency_ms": row["latency_ms"],
        }
        for (item, row), prediction in zip(answered, predictions)
    }
    for item, row in ran:
        if row["status"] != "ok":
            answers[item["id"]] = {
                "status": "error", "label": None, "probability": None, "expected": None, "correct": False, "latency_ms": None,
            }
    return summary, answers


def build_task_report(task, items: list[dict], cache: dict, dataset: dict | None) -> dict:
    endpoints = {}
    answers = {}
    for endpoint in ENDPOINTS:
        endpoints[endpoint], answers[endpoint] = summarise_endpoint(task, items, cache, endpoint)
    cases = [
        {
            "id": item["id"],
            "label": item["label"],
            "fields": {key: item[key] for key, _ in task.fields},
            "answers": {endpoint: answers[endpoint][item["id"]] for endpoint in ENDPOINTS if item["id"] in answers[endpoint]},
        }
        for item in items
    ]
    return {
        "task": task.name,
        "title": task.title,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "dataset": dataset,
        "questions": task.questions,
        "fields": [list(field) for field in task.fields],
        "headline": task.headline,
        "endpoints": endpoints,
        "cases": cases,
    }


def build_summary(reports: list[dict]) -> dict:
    rows = []
    for built in reports:
        for endpoint, summary in built["endpoints"].items():
            rows.append(
                {
                    "task": built["task"],
                    "title": built["title"],
                    "endpoint": endpoint,
                    "headline": summary["metrics"][built["headline"]],
                    "p50_ms": summary["latency_ms"]["p50"],
                    "p95_ms": summary["latency_ms"]["p95"],
                    "cost_per_1000": summary["cost_per_1000"],
                    "answered": summary["counts"]["answered"],
                    "planned": summary["counts"]["planned"],
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "prices": {"per_million": metrics.PRICES_PER_MILLION, "read_on": metrics.PRICES_READ_ON},
        "rows": rows,
    }


def _value(value: float | None, fmt: str) -> str:
    if value is None:
        return "no data"
    return f"{value * 100:.1f}%" if fmt == "percent" else f"{value:.3f}"


def _metric(metric: dict) -> str:
    if metric["value"] is None:
        return "no data"
    text = _value(metric["value"], metric["format"])
    if metric["ci"]:
        low, high = (_value(bound, metric["format"]).rstrip("%") for bound in metric["ci"])
        text += f" ({low} to {high})"
    return text


def _answered(counts: dict) -> str:
    text = f"{counts['answered']} of {counts['planned']}"
    if counts["refused"] or counts["malformed"] or counts["errored"]:
        text += f" ({counts['refused']} refused, {counts['malformed']} malformed, {counts['errored']} errors)"
    return text


def markdown(built: dict) -> str:
    summaries = built["endpoints"]
    names = list(summaries)

    def row(label: str, cell) -> str:
        return f"| {label} | " + " | ".join(cell(summaries[name]) for name in names) + " |"

    def ms(value: float | None) -> str:
        return "no data" if value is None else f"{value:,.0f} ms"

    lines = [
        "| | " + " | ".join(ENDPOINT_TITLES[name] for name in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for key, metric in summaries[names[0]]["metrics"].items():
        lines.append(row(metric["label"], lambda s, key=key: _metric(s["metrics"][key])))
    lines.append(row("Median response time", lambda s: ms(s["latency_ms"]["p50"])))
    lines.append(row("95th-percentile response time", lambda s: ms(s["latency_ms"]["p95"])))
    lines.append(
        row("Cost per 1,000 decisions", lambda s: "no data" if s["cost_per_1000"] is None else f"${s['cost_per_1000']:.3f}")
    )
    lines.append(row("Answered", lambda s: _answered(s["counts"])))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true", help="print each task's comparison table")
    args = parser.parse_args()

    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    RESULTS.mkdir(exist_ok=True)
    reports = []
    for name, task in TASKS.items():
        cache = load_cache(cache_path(name))
        if not cache:
            print(f"{name}: no answers cached yet, skipped")
            continue
        # Report on the items that were run, so a --limit smoke run gives a sensible report.
        ran = {item_id for item_id, _ in cache}
        items = [item for item in load_items(name) if item["id"] in ran]
        built = build_task_report(task, items, cache, manifest.get(name))
        (RESULTS / f"{name}.json").write_text(json.dumps(built))
        reports.append(built)
        print(f"{name}: wrote results/{name}.json ({len(items)} items)")
        if args.markdown:
            print("\n" + markdown(built) + "\n")
    if reports:
        (RESULTS / "summary.json").write_text(json.dumps(build_summary(reports), indent=1))
        print("wrote results/summary.json")


if __name__ == "__main__":
    main()
