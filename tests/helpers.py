"""Fakes shared by the endpoint, runner and server tests."""
from types import SimpleNamespace

import httpx
import openai

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/decisions")


class FakeResult:
    def __init__(self, data: dict, output_text: str | None = None):
        self._data = data
        self.output_text = output_text

    def model_dump(self) -> dict:
        return self._data


class FakeClient:
    def __init__(self, decision: FakeResult | None = None, response: FakeResult | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.decisions = SimpleNamespace(create=lambda **kwargs: self._record("decisions", kwargs, decision))
        self.responses = SimpleNamespace(create=lambda **kwargs: self._record("responses", kwargs, response))

    def _record(self, kind: str, kwargs: dict, result):
        self.calls.append((kind, kwargs))
        return result


def api_error(cls, status: int) -> openai.APIStatusError:
    return cls("nope", response=httpx.Response(status, request=REQUEST), body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=REQUEST)
