"""A small client for the 19th Training and Inference APIs (private alpha, read on 2026-11-20).

Docs: https://www.19th.com/docs/training and https://www.19th.com/docs/inference
The key goes in .env as NINETEENTH_API_KEY: a Training API key with training:write,
training:read and inference:invoke, so that one key can train and call the hosted model.
"""
import hashlib
import json
import os
import time

import httpx

TRAINING_ORIGIN = "https://training.19th.com"
KEY_NAME = "NINETEENTH_API_KEY"
TERMINAL = {"ready", "failed", "cancelled"}
PROJECT_KEY_NAME = "NINETEENTH_PROJECT_ID"
RUN_DONE = {"Succeeded", "Failed", "Cancelled", "Canceled"}


class NineteenthError(Exception):
    """An error answer from 19th. fatal: every later call would fail the same way."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.fatal = status in (401, 403)


def api_key() -> str:
    key = os.environ.get(KEY_NAME, "").strip()
    if not key:
        raise NineteenthError(f"{KEY_NAME} is missing. Add a 19th Training API key to .env.", status=401)
    return key


def make_http(timeout: float = 120) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key()}"})


def _error(response: httpx.Response) -> NineteenthError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    detail = body.get("error") if isinstance(body.get("error"), dict) else body
    message = detail.get("message") if isinstance(detail, dict) else None
    retryable = detail.get("retryable") if isinstance(detail, dict) else None
    if retryable is None:
        retryable = response.status_code in (429, 500, 502, 503, 504)
    return NineteenthError(
        f"19th answered {response.status_code}: {message or response.text[:300]}",
        status=response.status_code,
        retryable=bool(retryable),
    )


def _json(response: httpx.Response) -> dict:
    if response.status_code >= 400:
        raise _error(response)
    return response.json()


def fit_request(name: str, spec: dict, csv_bytes: bytes, model_variant: str = "fast") -> dict:
    return {
        "name": name,
        "file": {"content_length": len(csv_bytes), "sha256": hashlib.sha256(csv_bytes).hexdigest()},
        "spec": spec,
        "model_variant": model_variant,
        "wait": True,
        "inline": False,
    }


def fit(http: httpx.Client, request: dict, csv_bytes: bytes, idempotency_key: str) -> dict:
    """Submit a training run. Replaying the same key and bytes recovers the same run instead of retraining."""
    response = http.post(
        f"{TRAINING_ORIGIN}/v1/fit",
        headers={"Idempotency-Key": idempotency_key},
        # The request JSON must come before the CSV.
        files=[
            ("request", (None, json.dumps(request), "application/json")),
            ("file", ("train.csv", csv_bytes, "text/csv")),
        ],
        timeout=300,
    )
    return _json(response)


def wait_until_done(http: httpx.Client, result: dict, on_poll=None, limit_seconds: float = 3600) -> dict:
    started = time.monotonic()
    while result.get("state") not in TERMINAL:
        if time.monotonic() - started > limit_seconds:
            raise NineteenthError(f"training run {result.get('run_id')} still {result.get('state')} after {limit_seconds}s")
        if on_poll:
            on_poll(result)
        response = http.get(f"{TRAINING_ORIGIN}{result['status_url']}&wait_seconds=60", timeout=120)
        result = _json(response)
    return result


def project_of_run(http: httpx.Client, run_id: str) -> str:
    """The key's project UUID, read from one of its training runs (19th lists no projects over REST)."""
    data = _json(http.get(f"{TRAINING_ORIGIN}/v1/training-runs/{run_id}"))
    data = data.get("data", data)
    project = data.get("project_ref") or (data.get("spec") or {}).get("project_ref")
    if not project:
        raise NineteenthError(f"training run {run_id} names no project; set {PROJECT_KEY_NAME} in .env")
    return project


def upload_and_train(http: httpx.Client, project: str, spec: dict, csv_bytes: bytes, idempotency_key: str) -> dict:
    """The project route. Unlike /v1/fit it accepts column_types and pairs for embedding columns."""
    response = http.post(
        f"{TRAINING_ORIGIN}/v1/projects/{project}/training-runs/upload",
        headers={"Idempotency-Key": idempotency_key},
        files=[
            ("spec", (None, json.dumps(spec), "application/json")),
            ("file", ("train.csv", csv_bytes, "text/csv")),
        ],
        timeout=300,
    )
    body = _json(response)
    return body.get("data", body)


def wait_for_run(http: httpx.Client, run_id: str, on_poll=None, limit_seconds: float = 3600, pause: float = 10) -> dict:
    started = time.monotonic()
    while True:
        data = _json(http.get(f"{TRAINING_ORIGIN}/v1/training-runs/{run_id}"))
        data = data.get("data", data)
        phase = (data.get("status") or {}).get("phase")
        if phase in RUN_DONE:
            return data
        if time.monotonic() - started > limit_seconds:
            raise NineteenthError(f"training run {run_id} still {phase} after {limit_seconds}s")
        if on_poll:
            on_poll(phase)
        time.sleep(pause)


def prediction_url(run: dict, variant: str = "fast") -> str | None:
    for target in run.get("inference_targets") or []:
        if target.get("candidate_id") == variant:
            return target.get("predict_url")
    return None


def predict(http: httpx.Client, url: str, record: dict) -> dict:
    """One record to a hosted model. Returns that record's answer from `output`."""
    response = http.post(url, json={"input": [record]})
    body = _json(response)
    output = body.get("output")
    if not isinstance(output, list) or len(output) != 1:
        raise NineteenthError(f"19th returned {len(output) if isinstance(output, list) else 'no'} answers for one record")
    return output[0]
