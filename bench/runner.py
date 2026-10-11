"""Run a task's items through every endpoint, interleaved, and cache each raw answer.

Usage: uv run python -m bench.runner <intent|injection|relevance|all> [--limit N] [--workers 8] [--retry-errors]
Answers are cached in .tmp/runs/<task>.jsonl, so an interrupted run resumes without paying twice.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from dotenv import load_dotenv

from bench.endpoints import ENDPOINTS, make_client
from bench.tasks import TASKS
from bench.tasks.base import ROOT, load_items

# A rejected key or missing access will fail every call, so the run stops instead of recording errors.
FATAL = (openai.AuthenticationError, openai.PermissionDeniedError)


def cache_path(task_name: str) -> Path:
    return ROOT / ".tmp" / "runs" / f"{task_name}.jsonl"


def load_cache(path: Path) -> dict[tuple[str, str], dict]:
    cache = {}
    if not path.exists():
        return cache
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            cache[(row["item_id"], row["endpoint"])] = row
        except (ValueError, KeyError, TypeError):
            continue  # a blank line, or a row cut short when a run was killed
    return cache


def plan_work(items: list[dict], endpoint_names: list[str], cache: dict, retry_errors: bool) -> list[tuple[dict, str]]:
    work = []
    for item in items:
        for name in endpoint_names:
            row = cache.get((item["id"], name))
            if row is None or (retry_errors and row["status"] == "error"):
                work.append((item, name))
    return work


def attempt(call, client, task, item: dict, retries: int, pause: float) -> dict:
    for tries in range(retries + 1):
        try:
            return {"status": "ok", **call(client, task, item)}
        except FATAL:
            raise
        except openai.BadRequestError as error:
            return {"status": "error", "error": str(error)}  # the same request would fail again
        except openai.OpenAIError as error:
            if tries == retries:
                return {"status": "error", "error": str(error)}
            time.sleep(pause * (tries + 1))


def run(
    task,
    items: list[dict],
    endpoints: dict,
    path: Path,
    workers: int = 8,
    client=None,
    retry_errors: bool = False,
    retries: int = 2,
    pause: float = 1.0,
    on_row=None,
) -> dict:
    todo = plan_work(items, list(endpoints), load_cache(path), retry_errors)
    counts = {"cached": len(items) * len(endpoints) - len(todo), "ok": 0, "error": 0}
    if not todo:
        return counts
    path.parent.mkdir(parents=True, exist_ok=True)
    ends_cleanly = not path.exists() or path.stat().st_size == 0 or path.read_bytes().endswith(b"\n")
    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as out:
        if not ends_cleanly:
            out.write("\n")
        futures = {
            pool.submit(attempt, endpoints[name], client, task, item, retries, pause): (item, name)
            for item, name in todo
        }
        for future in as_completed(futures):
            item, name = futures[future]
            try:
                row = future.result()
            except FATAL as error:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(f"OpenAI rejected the request, stopping: {error}")
            row = {"item_id": item["id"], "endpoint": name, **row}
            out.write(json.dumps(row) + "\n")
            out.flush()
            counts[row["status"]] += 1
            if on_row:
                on_row(row)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[*TASKS, "all"])
    parser.add_argument("--limit", type=int, help="only run the first N items")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-errors", action="store_true", help="rerun pairs cached as errors")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    try:
        client = make_client()
    except openai.OpenAIError as error:
        sys.exit(f"Cannot create the OpenAI client. Check OPENAI_API_KEY in .env: {error}")

    for name in TASKS if args.task == "all" else [args.task]:
        try:
            items = load_items(name)
        except FileNotFoundError as error:
            sys.exit(str(error))
        if args.limit:
            items = items[: args.limit]
        done = 0

        def progress(row: dict) -> None:
            nonlocal done
            done += 1
            print(f"\r{name}: {done} new answers", end="", flush=True)

        counts = run(TASKS[name], items, ENDPOINTS, cache_path(name), args.workers, client, args.retry_errors, on_row=progress)
        print(f"\r{name}: {counts['cached']} cached, {counts['ok']} answered, {counts['error']} errors")
        if counts["error"]:
            print(f"  Rerun with --retry-errors to try the {counts['error']} failed calls again.")


if __name__ == "__main__":
    main()
