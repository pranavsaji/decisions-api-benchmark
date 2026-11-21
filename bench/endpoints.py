"""Every endpoint in the benchmark. Each returns the raw answer, timing and tokens.

Three ask gpt-6-luna with no examples. Two answer from a model trained on labelled examples
(19th and a local baseline, see bench/supervised.py).
"""
import time

import openai

from bench import supervised
from bench.tasks.base import Task, responses_prompt

MODEL = "gpt-6-luna"


def make_client() -> openai.OpenAI:
    # Retries are handled by the runner so that latency covers one attempt only.
    return openai.OpenAI(max_retries=0, timeout=120)


def _usage(usage) -> dict:
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details")
    return {
        "input_tokens": usage.get("input_tokens"),
        "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "output_tokens": usage.get("output_tokens"),
    }


def _refused(data: dict) -> bool:
    for output in data.get("output") or []:
        for part in (output.get("content") if isinstance(output, dict) else None) or []:
            if isinstance(part, dict) and part.get("type") == "refusal":
                return True
    return False


def decisions(client, task: Task, item: dict) -> dict:
    started = time.perf_counter()
    decision = client.decisions.create(model=MODEL, input=task.render_input(item), questions=task.questions)
    latency_ms = round((time.perf_counter() - started) * 1000)
    data = decision.model_dump()
    return {"raw": {"answers": data.get("answers")}, "latency_ms": latency_ms, **_usage(data.get("usage"))}


def _responses(client, task: Task, item: dict, **settings) -> dict:
    started = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        input=responses_prompt(task, item),
        text={"format": {"type": "json_schema", "name": task.name, "schema": task.responses_schema, "strict": True}},
        **settings,
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    data = response.model_dump()
    raw = {"text": response.output_text, "refused": _refused(data)}
    return {"raw": raw, "latency_ms": latency_ms, **_usage(data.get("usage"))}


def responses_default(client, task: Task, item: dict) -> dict:
    return _responses(client, task, item)


def responses_tuned(client, task: Task, item: dict) -> dict:
    return _responses(client, task, item, reasoning={"effort": "none"})


ENDPOINTS = {
    "decisions": decisions,
    "responses_default": responses_default,
    "responses_tuned": responses_tuned,
    "nineteenth": supervised.nineteenth_call,
    "local_baseline": supervised.baseline_call,
}
