"""Turn the cached answers into results/<task>.json and results/summary.json.

Usage: uv run python -m bench.report [--markdown]
Makes no API calls. --markdown also prints each task's comparison table for the README.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone

from bench import embeddings, metrics, supervised
from bench.endpoints import ENDPOINTS, MODEL
from bench.runner import cache_path, load_cache
from bench.tasks import TASKS
from bench.tasks.base import DATA, ROOT, TRAINED, correct, load_items

RESULTS = ROOT / "results"
ENDPOINT_TITLES = {
    "decisions": "Decisions API",
    "responses_default": "Responses API, default",
    "responses_tuned": "Responses API, tuned",
    "nineteenth": "19th",
    "local_baseline": "Local baseline",
}
# Model details safe to publish: no prediction URL or request key.
TRAINING_FIELDS = ("rows", "training_mode", "model_variant", "route", "wall_seconds", "model_bytes", "C", "train_seconds")


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
    if endpoint in TRAINED:
        # Only the embedding call is billed per decision that we can price; 19th publishes no prices in its alpha.
        cost = None if tokens["input"] is None else tokens["input"] * embeddings.PRICE_PER_MILLION / 1000
    else:
        cost = metrics.cost_per_1000(tokens["input"], tokens["cached"], tokens["output"])
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
        "cost_per_1000": cost,
    }
    if endpoint in TRAINED:
        summary["latency_ms"]["embed_p50"] = metrics.percentile([row.get("embed_ms") for _, row in answered if row.get("embed_ms") is not None], 0.5)
        summary["latency_ms"]["predict_p50"] = metrics.percentile([row.get("predict_ms") for _, row in answered if row.get("predict_ms") is not None], 0.5)
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


def paired_comparison(task, items: list[dict], answers: dict, endpoint: str, baseline: str = "decisions") -> dict | None:
    """Headline metric of endpoint minus baseline, on the items both of them answered."""
    usable = ("ok", "malformed")
    shared = [
        item
        for item in items
        if answers[endpoint].get(item["id"], {}).get("status") in usable
        and answers[baseline].get(item["id"], {}).get("status") in usable
    ]
    compared = metrics.paired_difference(
        task.headline_value,
        [(item, answers[endpoint][item["id"]]) for item in shared],
        [(item, answers[baseline][item["id"]]) for item in shared],
    )
    return {**compared, "baseline": baseline, "n": len(shared)} if compared else None


def reported_endpoints(cache: dict) -> list[str]:
    """The no-examples endpoints always; a trained one once it has answers cached."""
    ran = {endpoint for _, endpoint in cache}
    return [name for name in ENDPOINTS if name not in TRAINED or name in ran]


def training_details(task_name: str, endpoints: list[str]) -> dict:
    details = {}
    for endpoint in endpoints:
        if endpoint in TRAINED and supervised.model_path(task_name, endpoint).exists():
            info = supervised.load_model(task_name, endpoint)
            details[endpoint] = {key: info[key] for key in TRAINING_FIELDS if key in info}
    return details


def build_task_report(task, items: list[dict], cache: dict, dataset: dict | None) -> dict:
    names = reported_endpoints(cache)
    endpoints = {}
    answers = {}
    for endpoint in names:
        endpoints[endpoint], answers[endpoint] = summarise_endpoint(task, items, cache, endpoint)
    cases = [
        {
            "id": item["id"],
            "label": item["label"],
            "fields": {key: item[key] for key, _ in task.fields},
            "answers": {endpoint: answers[endpoint][item["id"]] for endpoint in names if item["id"] in answers[endpoint]},
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
        "paired": {name: paired_comparison(task, items, answers, name) for name in names if name != "decisions"},
        # 19th against the plain model trained on the same rows: what 19th adds beyond having labels.
        "paired_trained": (
            paired_comparison(task, items, answers, "nineteenth", baseline="local_baseline")
            if all(name in names for name in TRAINED)
            else None
        ),
        "training": training_details(task.name, names),
        "embedding_model": embeddings.MODEL if any(name in TRAINED for name in names) else None,
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
                    "paired": built["paired"].get(endpoint),
                    "paired_trained": built.get("paired_trained") if endpoint == "nineteenth" else None,
                    "p50_ms": summary["latency_ms"]["p50"],
                    "embed_p50_ms": summary["latency_ms"].get("embed_p50"),
                    "p95_ms": summary["latency_ms"]["p95"],
                    "cost_per_1000": summary["cost_per_1000"],
                    "answered": summary["counts"]["answered"],
                    "planned": summary["counts"]["planned"],
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "prices": {
            "per_million": metrics.PRICES_PER_MILLION,
            "read_on": metrics.PRICES_READ_ON,
            "embedding_per_million": embeddings.PRICE_PER_MILLION,
            "embedding_model": embeddings.MODEL,
        },
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


def _cost(value: float) -> str:
    # The embedding-only cost of the trained endpoints is a fraction of a tenth of a cent.
    return f"${value:.4f}" if value < 0.001 else f"${value:.3f}"


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
    headline = summaries[names[0]]["metrics"][built["headline"]]

    def difference(compared: dict | None) -> str:
        if not compared or not compared["ci"]:
            return "no data"
        scale, digits, unit = (100, 1, " points") if headline["format"] == "percent" else (1, 3, "")
        low, high = (f"{bound * scale:+.{digits}f}" for bound in compared["ci"])
        return f"{compared['difference'] * scale:+.{digits}f}{unit} ({low} to {high})"

    def paired(name: str) -> str:
        return "baseline" if name == "decisions" else difference(built["paired"].get(name))

    lines.append(
        f"| {headline['label']}, difference from Decisions API on the same items | "
        + " | ".join(paired(name) for name in names)
        + " |"
    )
    if built.get("paired_trained"):
        cells = {"local_baseline": "baseline", "nineteenth": difference(built["paired_trained"])}
        lines.append(
            f"| {headline['label']}, difference from the local baseline on the same items | "
            + " | ".join(cells.get(name, "") for name in names)
            + " |"
        )
    lines.append(row("Median response time", lambda s: ms(s["latency_ms"]["p50"])))
    lines.append(row("95th-percentile response time", lambda s: ms(s["latency_ms"]["p95"])))
    lines.append(
        row("Cost per 1,000 decisions", lambda s: "no data" if s["cost_per_1000"] is None else _cost(s["cost_per_1000"]))
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
