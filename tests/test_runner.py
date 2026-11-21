import json

import openai
import pytest

from bench import runner
from bench.tasks import TASKS
from tests.helpers import api_error, connection_error

TASK = TASKS["injection"]
ITEMS = [{"id": "a", "text": "one", "label": "attack"}, {"id": "b", "text": "two", "label": "benign"}]


def answer(probability=0.5):
    return {
        "raw": {"answers": [{"type": "predicate", "name": "injection", "probability": probability}]},
        "latency_ms": 12,
        "input_tokens": 30,
        "cached_tokens": None,
        "output_tokens": 0,
    }


class Recorder:
    """A fake endpoint that records calls and can fail a set number of times per item."""

    def __init__(self, failures=None):
        self.calls = []
        self.failures = dict(failures or {})

    def __call__(self, client, task, item):
        self.calls.append(item["id"])
        pending = self.failures.get(item["id"])
        if pending:
            error, remaining = pending
            self.failures[item["id"]] = (error, remaining - 1) if remaining > 1 else None
            raise error
        return answer()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_plan_work_keeps_the_endpoints_for_one_item_together():
    work = runner.plan_work(ITEMS, ["decisions", "responses_default"], {}, False)
    assert [(item["id"], name) for item, name in work] == [
        ("a", "decisions"),
        ("a", "responses_default"),
        ("b", "decisions"),
        ("b", "responses_default"),
    ]


def test_run_writes_one_row_per_pair_and_a_second_run_calls_nothing(tmp_path):
    path = tmp_path / "runs" / "injection.jsonl"
    first, second = Recorder(), Recorder()
    counts = runner.run(TASK, ITEMS, {"decisions": first, "responses_default": second}, path, workers=2)
    assert counts == {"cached": 0, "ok": 4, "error": 0}
    written = rows(path)
    assert {(r["item_id"], r["endpoint"]) for r in written} == {
        ("a", "decisions"), ("a", "responses_default"), ("b", "decisions"), ("b", "responses_default"),
    }
    assert all(r["status"] == "ok" and r["latency_ms"] == 12 and r["input_tokens"] == 30 for r in written)

    again = Recorder()
    counts = runner.run(TASK, ITEMS, {"decisions": again, "responses_default": again}, path, workers=2)
    assert counts == {"cached": 4, "ok": 0, "error": 0}
    assert again.calls == []


def test_a_transient_error_is_retried_and_then_succeeds(tmp_path):
    endpoint = Recorder({"a": (connection_error(), 2)})
    counts = runner.run(TASK, ITEMS[:1], {"decisions": endpoint}, tmp_path / "c.jsonl", pause=0)
    assert endpoint.calls == ["a", "a", "a"]
    assert counts == {"cached": 0, "ok": 1, "error": 0}


def test_an_error_that_outlasts_the_retries_is_recorded_and_not_rerun_by_default(tmp_path):
    path = tmp_path / "c.jsonl"
    endpoint = Recorder({"a": (connection_error(), 99)})
    counts = runner.run(TASK, ITEMS, {"decisions": endpoint}, path, workers=1, pause=0)
    assert endpoint.calls.count("a") == 3
    assert counts == {"cached": 0, "ok": 1, "error": 1}
    failed = next(r for r in rows(path) if r["item_id"] == "a")
    assert failed["status"] == "error" and "Connection error" in failed["error"]

    later = Recorder()
    assert runner.run(TASK, ITEMS, {"decisions": later}, path, pause=0) == {"cached": 2, "ok": 0, "error": 0}
    assert later.calls == []


def test_retry_errors_reruns_failed_pairs_and_the_newest_row_wins(tmp_path):
    path = tmp_path / "c.jsonl"
    runner.run(TASK, ITEMS, {"decisions": Recorder({"a": (connection_error(), 99)})}, path, workers=1, pause=0)
    retry = Recorder()
    counts = runner.run(TASK, ITEMS, {"decisions": retry}, path, retry_errors=True, pause=0)
    assert retry.calls == ["a"]
    assert counts == {"cached": 1, "ok": 1, "error": 0}
    assert runner.load_cache(path)[("a", "decisions")]["status"] == "ok"


def test_a_bad_request_is_recorded_without_retrying(tmp_path):
    endpoint = Recorder({"a": (api_error(openai.BadRequestError, 400), 99)})
    counts = runner.run(TASK, ITEMS[:1], {"decisions": endpoint}, tmp_path / "c.jsonl", pause=0)
    assert endpoint.calls == ["a"]
    assert counts["error"] == 1


@pytest.mark.parametrize("error_class, status", [(openai.AuthenticationError, 401), (openai.PermissionDeniedError, 403)])
def test_a_rejected_key_or_missing_access_stops_the_run(tmp_path, error_class, status):
    endpoint = Recorder({"a": (api_error(error_class, status), 99)})
    with pytest.raises(SystemExit, match="every call would fail"):
        runner.run(TASK, ITEMS, {"decisions": endpoint}, tmp_path / "c.jsonl", workers=1, pause=0)
    assert endpoint.calls.count("a") == 1


def test_a_half_written_last_line_is_ignored_and_the_next_row_is_clean(tmp_path):
    path = tmp_path / "c.jsonl"
    good = {"item_id": "b", "endpoint": "decisions", "status": "ok", **answer()}
    path.write_text(json.dumps(good) + "\n" + '{"item_id": "a", "endpoint": "decis')
    assert list(runner.load_cache(path)) == [("b", "decisions")]

    endpoint = Recorder()
    counts = runner.run(TASK, ITEMS, {"decisions": endpoint}, path, pause=0)
    assert endpoint.calls == ["a"]
    assert counts == {"cached": 1, "ok": 1, "error": 0}
    cache = runner.load_cache(path)
    assert set(cache) == {("a", "decisions"), ("b", "decisions")}
    assert cache[("a", "decisions")]["status"] == "ok"


def test_load_cache_of_a_missing_file_is_empty(tmp_path):
    assert runner.load_cache(tmp_path / "absent.jsonl") == {}


def test_on_row_is_called_for_each_new_row(tmp_path):
    seen = []
    runner.run(TASK, ITEMS, {"decisions": Recorder()}, tmp_path / "c.jsonl", on_row=lambda row: seen.append(row["item_id"]))
    assert sorted(seen) == ["a", "b"]


def test_cache_path_is_under_tmp_runs():
    assert runner.cache_path("intent").parts[-3:] == (".tmp", "runs", "intent.jsonl")


def test_an_interrupted_run_stops_calling_and_keeps_every_answer_it_paid_for(tmp_path):
    import time

    path = tmp_path / "runs" / "injection.jsonl"
    items = [{"id": str(n), "text": "x", "label": "benign"} for n in range(40)]
    calls = []

    def slow(client, task, item):
        calls.append(item["id"])
        time.sleep(0.02)
        return answer()

    seen = []

    def interrupt_after_three(row):
        seen.append(row)
        if len(seen) == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.run(TASK, items, {"decisions": slow}, path, workers=2, on_row=interrupt_after_three)
    assert len(calls) < 10
    assert sorted(r["item_id"] for r in rows(path)) == sorted(calls)


def test_an_untrained_model_stops_the_run_naming_the_command(tmp_path):
    from bench.supervised import NotTrained

    endpoint = Recorder({"a": (NotTrained("Run: uv run python -m tools.train injection"), 99)})
    with pytest.raises(SystemExit, match="tools.train injection"):
        runner.run(TASK, ITEMS, {"nineteenth": endpoint}, tmp_path / "c.jsonl", workers=1, pause=0)


@pytest.mark.parametrize("status, calls, outcome", [(401, 1, "stop"), (429, 3, "error"), (400, 1, "error")])
def test_19th_errors_stop_retry_or_record(tmp_path, status, calls, outcome):
    from bench.nineteenth import NineteenthError

    error = NineteenthError("19th answered", status=status, retryable=status == 429)
    endpoint = Recorder({"a": (error, 99)})
    path = tmp_path / "c.jsonl"
    if outcome == "stop":
        with pytest.raises(SystemExit):
            runner.run(TASK, ITEMS[:1], {"nineteenth": endpoint}, path, workers=1, pause=0)
    else:
        assert runner.run(TASK, ITEMS[:1], {"nineteenth": endpoint}, path, workers=1, pause=0)["error"] == 1
    assert endpoint.calls.count("a") == calls
