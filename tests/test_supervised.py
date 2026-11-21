import hashlib
import json
import csv
import io
from types import SimpleNamespace

import numpy as np
import pytest

from bench import embeddings, supervised
from bench.tasks import TASKS
from bench.tasks.intent import INTENTS


class FakeEmbeddings:
    """An OpenAI client stand-in whose embeddings.create returns a fixed vector per text."""

    def __init__(self):
        self.calls = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, model, input):
        self.calls.append(list(input))
        data = [
            SimpleNamespace(index=n, embedding=[float(len(text) % 7)] * embeddings.DIMENSIONS)
            for n, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=5 * len(input)))


def test_embed_orders_by_index_and_reports_tokens():
    vectors, tokens = embeddings.embed(FakeEmbeddings(), ["a", "abc"])
    assert vectors.shape == (2, embeddings.DIMENSIONS) and vectors.dtype == np.float32
    assert vectors[0, 0] == 1 and vectors[1, 0] == 3
    assert tokens == 10


def test_embed_cached_pays_once_per_text(tmp_path):
    client = FakeEmbeddings()
    first, paid = embeddings.embed_cached(client, ["a", "bb", "a"], cache=tmp_path)
    assert paid == 10 and client.calls == [["a", "bb"]]
    second, paid = embeddings.embed_cached(client, ["bb", "a"], cache=tmp_path)
    assert paid == 0 and len(client.calls) == 1
    assert np.array_equal(second, first[[1, 0]])


def test_batches_split_on_characters_and_count():
    texts = ["x" * 300_000, "y" * 300_000, "z"]
    assert [len(group) for group in embeddings.batches(texts)] == [1, 2]
    assert [len(group) for group in embeddings.batches(["a"] * 1100)] == [512, 512, 76]


def test_fit_specs_match_each_task():
    assert supervised.fit_spec("intent", "high") == {
        "target_column": "label",
        "input_columns": ["text_embedding"],
        "output_kind": "multiclass_label",
        "training_mode": "high",
    }
    assert supervised.fit_spec("injection", "high")["positive_label"] == "attack"
    relevance = supervised.fit_spec("relevance", "standard")
    assert relevance["input_columns"] == ["query_embedding", "passage_embedding"]
    assert (relevance["output_kind"], relevance["order"]) == ("ordered_class", [0, 1, 2, 3])
    assert supervised.spec("intent").classes == INTENTS


def test_training_csv_holds_json_number_arrays_and_the_label():
    vectors = [np.array([[0.5, -1.0], [2.0, 0.25]], dtype=np.float32)] * 2
    text = supervised.training_csv("relevance", vectors, [3, 0]).decode()
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["query_embedding", "passage_embedding", "label"]
    assert rows[1][2] == "3" and rows[2][2] == "0"
    assert json.loads(rows[2][0]) == [2.0, 0.25]
    assert json.loads(supervised.encode_vector(np.array([0.123456789], dtype=np.float32))) == [0.123457]


def test_prediction_record_sends_each_embedding_as_numbers():
    record = supervised.record("relevance", np.array([[0.5, 1.0], [0.0, 2.0]], dtype=np.float32))
    assert record == {"query_embedding": [0.5, 1.0], "passage_embedding": [0.0, 2.0]}
    assert set(supervised.record("intent", np.array([[0.5, 1.0]]))) == {"text_embedding"}


def test_relevance_features_combine_query_and_passage():
    query, passage = np.array([[1.0, 2.0]]), np.array([[3.0, 1.0]])
    assert supervised.features("relevance", [query, passage]).tolist() == [[3.0, 2.0, 2.0, 1.0]]
    assert supervised.features("intent", [query]) is query


def test_linear_model_matches_scikit_learn_and_round_trips(tmp_path):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4))
    for labels in (np.array(["a", "b"] * 30), np.array(["a", "b", "c"] * 20)):
        fitted = LogisticRegression().fit(x, labels)
        linear = supervised.Linear(list(fitted.classes_), fitted.coef_, fitted.intercept_)
        assert np.allclose(linear.probabilities(x[0]), fitted.predict_proba(x[:1])[0])
        linear.save(tmp_path / "m.npz")
        loaded = supervised.Linear.load(tmp_path / "m.npz")
        assert loaded.classes == linear.classes and np.allclose(loaded.probabilities(x[1]), linear.probabilities(x[1]))


def test_train_baseline_chooses_c_by_cross_validation():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(-1, 1, size=(30, 3)), rng.normal(1, 1, size=(30, 3))])
    model, stats = supervised.train_baseline("injection", x, ["benign"] * 30 + ["attack"] * 30)
    assert stats["C"] in (0.1, 1, 10, 100, 1000)
    answer = supervised.baseline_answer(model, np.array([2.0, 2.0, 2.0]))
    assert set(answer["probabilities"]) == {"attack", "benign"} and answer["probabilities"]["attack"] > 0.9


@pytest.mark.parametrize(
    "task, output, expected",
    [
        ("injection", {"probability": 0.31}, {"attack": 0.31, "benign": pytest.approx(0.69)}),
        # The shape 19th really returned on 2026-11-20: unnamed probabilities, positive class first.
        ("injection", {"event_probability": 0.03, "label": "benign", "probabilities": [0.03, 0.97]},
         {"attack": 0.03, "benign": pytest.approx(0.97)}),
        ("intent", {"label": "a", "classes": ["a", "b"], "probabilities": [0.8, 0.2]}, {"a": 0.8, "b": 0.2}),
        ("intent", {"label": "a", "probabilities": {"a": 0.8, "b": 0.2}}, {"a": 0.8, "b": 0.2}),
        ("intent", {"probabilities": [{"label": "a", "probability": 0.8}]}, {"a": 0.8}),
        ("relevance", {"grade": 2, "classes": [0, 1, 2, 3], "probabilities": [0.1, 0.2, 0.3, 0.4]},
         {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}),
    ],
)
def test_19th_answers_reduce_to_a_probability_table(task, output, expected):
    assert supervised.reduce_19th(task, output)["probabilities"] == expected


def test_19th_ordered_answer_as_really_returned_reads_probabilities_in_declared_order():
    output = {"expected_grade": 0.526, "grade": 1, "label": 0, "probabilities": [0.695, 0.158, 0.071, 0.076]}
    reduced = supervised.reduce_19th("relevance", output)
    assert reduced["probabilities"] == {"0": 0.695, "1": 0.158, "2": 0.071, "3": 0.076}
    assert (reduced["label"], reduced["expected"]) == (1, 0.526)


def test_19th_multiclass_answer_reads_unnamed_probabilities_in_sorted_label_order():
    probabilities = [0.001] * len(INTENTS)
    probabilities[sorted(INTENTS).index("exchange_charge")] = 0.9
    reduced = supervised.reduce_19th("intent", {"label": "exchange_charge", "probabilities": probabilities})
    assert reduced["probabilities"]["exchange_charge"] == 0.9 and len(reduced["probabilities"]) == 77
    prediction = TASKS["intent"].to_prediction("nineteenth", {"task": "intent", "output": {"label": "exchange_charge", "probabilities": probabilities}})
    assert (prediction["status"], prediction["label"], prediction["probability"]) == ("ok", "exchange_charge", 0.9)


def test_19th_ordered_answer_keeps_its_grade_and_expected_grade():
    reduced = supervised.reduce_19th("relevance", {"grade": 2, "label": 3, "expected_grade": 2.2, "probabilities": {}})
    assert (reduced["label"], reduced["expected"]) == (2, 2.2)
    assert supervised.reduce_19th("relevance", "nope") is None


def test_the_baseline_endpoint_embeds_live_and_times_both_steps(monkeypatch):
    model = supervised.Linear(["attack", "benign"], np.zeros((1, embeddings.DIMENSIONS)), np.array([1.0]))
    monkeypatch.setattr(supervised, "baseline_model", lambda name: model)
    result = supervised.baseline_call(FakeEmbeddings(), TASKS["injection"], {"id": "x", "text": "hello"})
    assert result["raw"]["probabilities"]["benign"] == pytest.approx(1 / (1 + np.exp(-1)))
    assert result["latency_ms"] == result["embed_ms"] + result["predict_ms"]
    assert result["input_tokens"] == 5


def test_the_19th_endpoint_sends_the_live_embedding(monkeypatch):
    sent = {}
    monkeypatch.setattr(supervised, "load_model", lambda task, endpoint: {"prediction_url": "https://p"})
    monkeypatch.setattr(supervised, "_http_client", lambda: "http")

    def predict(http, url, record):
        sent.update(url=url, record=record)
        return {"grade": 1, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.2, "3": 0.1}}

    monkeypatch.setattr(supervised.nineteenth, "predict", predict)
    item = {"id": "x", "query": "q", "passage": "pp"}
    result = supervised.nineteenth_call(FakeEmbeddings(), TASKS["relevance"], item)
    assert sent["url"] == "https://p" and set(sent["record"]) == {"query_embedding", "passage_embedding"}
    assert result["raw"] == {"task": "relevance", "output": {"grade": 1, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.2, "3": 0.1}}}
    prediction = TASKS["relevance"].to_prediction("nineteenth", result["raw"])
    assert (prediction["status"], prediction["label"], prediction["probability"]) == ("ok", 1, 0.5)


def test_an_untrained_task_says_which_command_trains_it(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "MODELS", tmp_path)
    with pytest.raises(supervised.NotTrained, match="tools.train intent"):
        supervised.load_model("intent", "nineteenth")


def test_the_idempotency_key_ignores_wait_and_inline_but_not_the_data():
    request = {"name": "x", "file": {"sha256": "a"}, "wait": True, "inline": False}
    key = supervised.idempotency_key("intent", b"", request)
    assert key == supervised.idempotency_key("intent", b"", {**request, "wait": False})
    assert key != supervised.idempotency_key("intent", b"", {**request, "file": {"sha256": "b"}})


def test_only_two_text_tasks_use_the_pair_route():
    assert supervised.uses_pairs("relevance")
    assert not supervised.uses_pairs("intent") and not supervised.uses_pairs("injection")


def test_project_spec_declares_vectors_the_pair_and_a_typed_order():
    spec = supervised.project_spec("relevance", "high", b"abc")
    assert spec["file"] == {"content_length": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert (spec["training_mode"], spec["run_name"]) == ("high", "Benchmark relevance with pairs")
    prediction = spec["prediction"]
    assert (prediction["task"], prediction["output_kind"]) == ("ordinal", "ordered_class")
    assert prediction["column_types"] == {"query_embedding": "vector", "passage_embedding": "vector"}
    assert prediction["pairs"] == [["query_embedding", "passage_embedding"]]
    assert prediction["order"][3] == {"type": "integer", "value": 3}
    injection = supervised.project_spec("injection", "high", b"abc")["prediction"]
    assert injection["positive_label"] == "attack" and "pairs" not in injection


def test_the_ultra_endpoint_uses_its_own_model(monkeypatch):
    asked = []
    monkeypatch.setattr(supervised, "load_model", lambda task, endpoint: asked.append(endpoint) or {"prediction_url": "u"})
    monkeypatch.setattr(supervised, "_http_client", lambda: "http")
    monkeypatch.setattr(supervised.nineteenth, "predict", lambda http, url, record: {"event_probability": 0.4})
    result = supervised.nineteenth_ultra_call(FakeEmbeddings(), TASKS["injection"], {"id": "x", "text": "hi"})
    assert asked == ["nineteenth_ultra"]
    assert TASKS["injection"].to_prediction("nineteenth_ultra", result["raw"])["probability"] == 0.4
    assert supervised.NINETEENTH_MODES == {"nineteenth": "high", "nineteenth_ultra": "ultra"}
