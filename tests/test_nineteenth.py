import hashlib
import json

import httpx
import pytest

from bench import nineteenth


def http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer test"})


def test_fit_request_fingerprints_the_csv():
    data = b"a,label\n1,x\n"
    request = nineteenth.fit_request("Benchmark task", {"target_column": "label"}, data)
    assert request["file"] == {"content_length": 12, "sha256": hashlib.sha256(data).hexdigest()}
    assert (request["wait"], request["inline"], request["model_variant"]) == (True, False, "fast")


def test_fit_sends_the_request_json_before_the_csv_with_the_idempotency_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers["Idempotency-Key"]
        seen["url"] = str(request.url)
        body = request.read()
        seen["order"] = (body.index(b'name="request"'), body.index(b'name="file"'))
        return httpx.Response(200, json={"state": "ready", "run_id": "r1"})

    result = nineteenth.fit(http(handler), {"name": "x"}, b"a\n1\n", "key-1")
    assert result["run_id"] == "r1"
    assert seen["key"] == "key-1" and seen["url"] == "https://training.19th.com/v1/fit"
    assert seen["order"][0] < seen["order"][1]


def test_wait_until_done_polls_the_status_url_until_a_terminal_state():
    states = iter(["training", "packaging", "ready"])
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"state": next(states), "status_url": "/v1/fit/r1?inline=false"})

    start = {"state": "preparing", "status_url": "/v1/fit/r1?inline=false"}
    assert nineteenth.wait_until_done(http(handler), start)["state"] == "ready"
    assert urls[0] == "https://training.19th.com/v1/fit/r1?inline=false&wait_seconds=60"
    assert len(urls) == 3


def test_predict_returns_the_single_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.read()) == {"input": [{"x": [0.5]}]}
        return httpx.Response(200, json={"output": [{"probability": 0.31}]})

    assert nineteenth.predict(http(handler), "https://inference.get19th.com/p", {"x": [0.5]}) == {"probability": 0.31}


@pytest.mark.parametrize(
    "status, body, retryable, fatal",
    [
        (401, {"error": {"message": "bad key"}}, False, True),
        (403, {"error": {"message": "no scope"}}, False, True),
        (400, {"error": {"message": "input is required", "retryable": False}}, False, False),
        (429, {}, True, False),
        (503, {"error": {"retryable": True}}, True, False),
    ],
)
def test_errors_carry_status_retryable_and_fatal(status, body, retryable, fatal):
    client = http(lambda request: httpx.Response(status, json=body))
    with pytest.raises(nineteenth.NineteenthError) as caught:
        nineteenth.predict(client, "https://inference.get19th.com/p", {})
    assert (caught.value.status, caught.value.retryable, caught.value.fatal) == (status, retryable, fatal)
    assert str(status) in str(caught.value)


def test_a_missing_key_is_a_fatal_error_naming_the_variable(monkeypatch):
    monkeypatch.delenv("NINETEENTH_API_KEY", raising=False)
    with pytest.raises(nineteenth.NineteenthError, match="NINETEENTH_API_KEY") as caught:
        nineteenth.api_key()
    assert caught.value.fatal


def test_project_of_run_reads_the_project_ref():
    client = http(lambda request: httpx.Response(200, json={"data": {"id": "r1", "project_ref": "p1"}}))
    assert nineteenth.project_of_run(client, "r1") == "p1"


def test_upload_and_train_posts_spec_then_csv_to_the_project_route():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen.update(url=str(request.url), key=request.headers["Idempotency-Key"])
        seen["order"] = (body.index(b'name="spec"'), body.index(b'name="file"'))
        return httpx.Response(202, json={"data": {"id": "job1", "status": {"phase": "Pending"}}})

    result = nineteenth.upload_and_train(http(handler), "p1", {"run_name": "x"}, b"a\n1\n", "k1")
    assert result["id"] == "job1" and seen["key"] == "k1"
    assert seen["url"] == "https://training.19th.com/v1/projects/p1/training-runs/upload"
    assert seen["order"][0] < seen["order"][1]


def test_wait_for_run_returns_the_finished_run_and_its_fast_prediction_url():
    phases = iter(["Pending", "Running", "Succeeded"])
    targets = [{"candidate_id": "turbo", "predict_url": "t"}, {"candidate_id": "fast", "predict_url": "f"}]
    client = http(lambda request: httpx.Response(200, json={"data": {"status": {"phase": next(phases)}, "inference_targets": targets}}))
    run = nineteenth.wait_for_run(client, "job1", pause=0)
    assert run["status"]["phase"] == "Succeeded"
    assert nineteenth.prediction_url(run) == "f" and nineteenth.prediction_url({}) is None
