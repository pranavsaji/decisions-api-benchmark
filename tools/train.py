"""Train the 19th model and the local baseline for each task from <task>-train.jsonl.

Usage: uv run python -m tools.train <intent|injection|relevance|all>
       [--only local_baseline|nineteenth|nineteenth_ultra] [--dry-run] [--fresh]

The endpoint sets 19th's training_mode: nineteenth is trained in high mode, nineteenth_ultra in ultra.
Without --only, the local baseline and 19th in high mode are trained.

Needs OPENAI_API_KEY (embeddings, cached under .tmp/embeddings/) and, for 19th, NINETEENTH_API_KEY.
Writes .tmp/models/<task>-<endpoint>.json, which the benchmark endpoints read. Rerunning with the
same data and settings recovers the same 19th run instead of training again.
--dry-run embeds and writes the 19th training CSV to .tmp/models/ without uploading it.
--fresh starts a new 19th run even if one with the same data and settings exists (for a run whose
hosted model 19th failed to serve).
"""
import argparse
import json
import os
import sys
import time
import uuid

import numpy as np
from dotenv import load_dotenv

from bench import embeddings, nineteenth, supervised
from bench.endpoints import make_client
from bench.tasks import TASKS
from bench.tasks.base import DATA, ROOT


def load_training(task_name: str) -> list[dict]:
    path = DATA / f"{task_name}-train.jsonl"
    if not path.exists():
        sys.exit(f"No training data for '{task_name}'. Run: uv run python -m tools.fetch")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def embed_rows(client, task_name: str, rows: list[dict]) -> tuple[list[np.ndarray], int]:
    vectors, paid = [], 0
    for key in supervised.spec(task_name).texts:
        texts = [row[key] for row in rows]
        done = 0

        def progress(count: int) -> None:
            nonlocal done
            done += count
            print(f"\r  embedding {key}: {done} new", end="", flush=True)

        matrix, tokens = embeddings.embed_cached(client, texts, on_batch=progress)
        if done:
            print()
        vectors.append(matrix)
        paid += tokens
    return vectors, paid


def train_baseline(task_name: str, vectors: list[np.ndarray], labels: list) -> None:
    print("  local baseline: fitting logistic regression (3-fold CV over C)")
    model, stats = supervised.train_baseline(task_name, supervised.features(task_name, vectors), labels)
    filename = f"{task_name}-local_baseline.npz"
    supervised.MODELS.mkdir(parents=True, exist_ok=True)
    model.save(supervised.MODELS / filename)
    supervised.save_model(task_name, "local_baseline", {"file": filename, "rows": len(labels), **stats})
    print(f"  local baseline: C={stats['C']}, {stats['train_seconds']}s")


def find_project(http) -> str:
    project = os.environ.get(nineteenth.PROJECT_KEY_NAME, "").strip()
    if project:
        return project
    for path in sorted(supervised.MODELS.glob("*-nineteenth.json")):
        run_id = json.loads(path.read_text()).get("run_id")
        if run_id:
            return nineteenth.project_of_run(http, run_id)
    sys.exit(f"Train a single-text task first, or set {nineteenth.PROJECT_KEY_NAME} in .env, to find the 19th project.")


def train_19th_pairs(task_name: str, endpoint: str, csv_bytes: bytes, labels: list, fresh: bool) -> None:
    """Two-text tasks go through the project route, the only one that accepts the embedding pair."""
    mode = supervised.NINETEENTH_MODES[endpoint]
    spec = supervised.project_spec(task_name, mode, csv_bytes)
    key = str(uuid.uuid4()) if fresh else supervised.idempotency_key(task_name, csv_bytes, spec)
    with nineteenth.make_http() as http:
        project = find_project(http)
        started = time.perf_counter()
        submitted = nineteenth.upload_and_train(http, project, spec, csv_bytes, key)
        run = nineteenth.wait_for_run(http, submitted["id"], on_poll=lambda phase: print(f"  19th: {phase}", flush=True))
        seconds = time.perf_counter() - started
    supervised.MODELS.mkdir(parents=True, exist_ok=True)
    (supervised.MODELS / f"{task_name}-{endpoint}-fit.json").write_text(json.dumps(run, indent=1))
    phase = (run.get("status") or {}).get("phase")
    url = nineteenth.prediction_url(run)
    if phase != "Succeeded" or not url:
        sys.exit(f"  19th training ended {phase} with prediction URL {'present' if url else 'missing'}")
    target = next(t for t in run["inference_targets"] if t.get("candidate_id") == "fast")
    supervised.save_model(
        task_name,
        endpoint,
        {
            "run_id": run.get("id"),
            "model_id": target.get("model_id"),
            "model_version_id": target.get("model_version_id"),
            "prediction_url": url,
            "training_mode": mode,
            "model_variant": "fast",
            "route": "project upload with pairs",
            "rows": len(labels),
            "csv_bytes": len(csv_bytes),
            "wall_seconds": round(seconds, 1),
            "idempotency_key": key,
        },
    )
    print(f"  19th: ready in {seconds:.0f}s (project route, embedding pair declared)")


def train_19th(task_name: str, endpoint: str, vectors: list[np.ndarray], labels: list, dry_run: bool, fresh: bool = False) -> None:
    mode = supervised.NINETEENTH_MODES[endpoint]
    csv_bytes = supervised.training_csv(task_name, vectors, labels)
    request = nineteenth.fit_request(f"Benchmark {task_name} classifier", supervised.fit_spec(task_name, mode), csv_bytes)
    key = supervised.idempotency_key(task_name, csv_bytes, request)
    if fresh:
        key = str(uuid.uuid4())
    print(f"  19th: {len(labels):,} rows, {len(csv_bytes) / 1e6:.1f} MB CSV, training_mode={mode}")
    if supervised.uses_pairs(task_name) and not dry_run:
        train_19th_pairs(task_name, endpoint, csv_bytes, labels, fresh)
        return
    if dry_run:
        path = supervised.MODELS / f"{task_name}-nineteenth-train.csv"
        path.write_bytes(csv_bytes)
        (supervised.MODELS / f"{task_name}-nineteenth-request.json").write_text(json.dumps(request, indent=1))
        print(f"  19th: dry run, wrote {path.relative_to(ROOT)}")
        return
    with nineteenth.make_http() as http:
        started = time.perf_counter()
        result = nineteenth.fit(http, request, csv_bytes, key)
        result = nineteenth.wait_until_done(http, result, on_poll=lambda r: print(f"  19th: {r.get('state')}", flush=True))
        seconds = time.perf_counter() - started
    supervised.MODELS.mkdir(parents=True, exist_ok=True)
    (supervised.MODELS / f"{task_name}-{endpoint}-fit.json").write_text(json.dumps(result, indent=1))
    if result.get("state") != "ready":
        sys.exit(f"  19th training ended {result.get('state')}: {json.dumps(result.get('error'))}")
    model = result.get("model") or {}
    prediction = model.get("prediction") or {}
    if not prediction.get("url"):
        sys.exit(f"  19th returned no hosted prediction URL (prediction state {prediction.get('state')})")
    download = model.get("download") or {}
    supervised.save_model(
        task_name,
        endpoint,
        {
            "run_id": result.get("run_id"),
            "model_id": model.get("model_id"),
            "model_version_id": model.get("model_version_id"),
            "prediction_url": prediction["url"],
            "training_mode": mode,
            "model_variant": request["model_variant"],
            "rows": len(labels),
            "csv_bytes": len(csv_bytes),
            "wall_seconds": round(seconds, 1),
            "model_bytes": download.get("size_bytes"),
            "idempotency_key": key,
        },
    )
    print(f"  19th: ready in {seconds:.0f}s, model {download.get('size_bytes')} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[*TASKS, "all"])
    parser.add_argument("--only", choices=["local_baseline", *supervised.NINETEENTH_MODES])
    parser.add_argument("--dry-run", action="store_true", help="write the 19th CSV and request instead of uploading")
    parser.add_argument("--fresh", action="store_true", help="start a new 19th run instead of recovering a matching one")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = make_client()
    for name in TASKS if args.task == "all" else [args.task]:
        rows = load_training(name)
        print(f"{name}: {len(rows):,} training rows")
        vectors, paid = embed_rows(client, name, rows)
        if paid:
            print(f"  embeddings: {paid:,} new tokens, ${paid * embeddings.PRICE_PER_MILLION / 1e6:.4f}")
        labels = [row["label"] for row in rows]
        if args.only in (None, "local_baseline"):
            train_baseline(name, vectors, labels)
        for endpoint in supervised.NINETEENTH_MODES:
            if args.only == endpoint or (args.only is None and endpoint == "nineteenth"):
                train_19th(name, endpoint, vectors, labels, args.dry_run, args.fresh)


if __name__ == "__main__":
    main()
