"""The two trained models: 19th, and a local logistic-regression baseline on the same inputs.

Both learn from <task>-train.jsonl, never from the scored items, and both see the same
OpenAI embeddings of the text. For each scored item, the endpoint embeds the text live and then
predicts, so the response time covers both steps.

Every answer is reduced to one shape that the task parsers read:
    {"label": <class or None>, "probabilities": {<str(class)>: p, ...}, "expected": <number or None>}
For 19th, the cached raw answer keeps 19th's own `output` object and is reduced at report time.
"""
import csv
import hashlib
import io
import json
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np

from bench import embeddings, nineteenth
from bench.tasks.base import ROOT, Task

MODELS = ROOT / ".tmp" / "models"
# The 19th endpoints and the training_mode each one is trained with.
NINETEENTH_MODES = {"nineteenth": "high", "nineteenth_ultra": "ultra"}
NAMESPACE = uuid.UUID("6f1c9a52-3b0e-4d43-9a57-19a0c3f2d7e1")


class NotTrained(Exception):
    """A trained endpoint was called before tools.train made its model."""


@dataclass(frozen=True)
class Spec:
    texts: list[str]  # item keys that are embedded, in order
    output_kind: str  # 19th spec.output_kind
    classes: list  # every label, in the task's order
    extra: dict  # other 19th spec fields


SPECS = {
    "intent": Spec(texts=["text"], output_kind="multiclass_label", classes=[], extra={}),
    "injection": Spec(
        texts=["text"], output_kind="event_probability", classes=["benign", "attack"], extra={"positive_label": "attack"}
    ),
    "relevance": Spec(
        texts=["query", "passage"], output_kind="ordered_class", classes=[0, 1, 2, 3], extra={"order": [0, 1, 2, 3]}
    ),
}


def spec(task_name: str) -> Spec:
    if task_name == "intent" and not SPECS["intent"].classes:
        from bench.tasks.intent import INTENTS

        SPECS["intent"] = Spec(texts=["text"], output_kind="multiclass_label", classes=list(INTENTS), extra={})
    return SPECS[task_name]


def column(text_key: str) -> str:
    return f"{text_key}_embedding"


# ---- inputs -------------------------------------------------------------------------------------


def features(task_name: str, vectors: list[np.ndarray]) -> np.ndarray:
    """Baseline features from the embeddings of one or more rows: one array per text key, each (rows, dims)."""
    if len(vectors) == 1:
        return vectors[0]
    query, passage = vectors
    return np.hstack([query * passage, np.abs(query - passage)])


def encode_vector(vector: np.ndarray) -> str:
    """A JSON array of numbers, which 19th detects as an embedding column.

    The docs also name base64 float32 cells, but 19th's fit read those as free text and refused
    them, and /v1/fit does not accept column_types to declare them. Six significant digits keep
    the file small; float32 embeddings carry about seven.
    """
    return "[" + ",".join(f"{x:.6g}" for x in np.asarray(vector, dtype=np.float32).tolist()) + "]"


def training_csv(task_name: str, vectors: list[np.ndarray], labels: list) -> bytes:
    keys = spec(task_name).texts
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([column(key) for key in keys] + ["label"])
    for row in range(len(labels)):
        writer.writerow([encode_vector(vectors[k][row]) for k in range(len(keys))] + [labels[row]])
    return out.getvalue().encode("utf-8")


def fit_spec(task_name: str, mode: str) -> dict:
    chosen = spec(task_name)
    columns = [column(key) for key in chosen.texts]
    result = {
        "target_column": "label",
        "input_columns": columns,
        "output_kind": chosen.output_kind,
        "training_mode": mode,
        **chosen.extra,
    }
    return result


TASK_KINDS = {"multiclass_label": ("multiclass", "class_label"), "event_probability": ("binary", "event_probability"),
              "ordered_class": ("ordinal", "ordered_class")}


def uses_pairs(task_name: str) -> bool:
    """Tasks with two texts are trained with a declared embedding pair, which only the project route accepts.

    On /v1/fit, which refuses `pairs`, 19th learned almost nothing about relevance (weighted kappa 0.080
    on 2026-11-20), and adding a similarity column made 19th fail to serve the model.
    """
    return len(spec(task_name).texts) == 2


def project_spec(task_name: str, mode: str, csv_bytes: bytes) -> dict:
    """The upload spec for 19th's project route: explicit task, vector column types and the pair."""
    chosen = spec(task_name)
    columns = [column(key) for key in chosen.texts]
    task, output_kind = TASK_KINDS[chosen.output_kind]
    prediction = {
        "task": task,
        "target_column": "label",
        "input_columns": columns,
        "output_kind": output_kind,
        "column_types": {name: "vector" for name in columns},
    }
    if len(columns) == 2:
        prediction["pairs"] = [columns]
    if "order" in chosen.extra:
        prediction["order"] = [{"type": "integer", "value": value} for value in chosen.extra["order"]]
    if "positive_label" in chosen.extra:
        prediction["positive_label"] = chosen.extra["positive_label"]
    return {
        "file": {"content_length": len(csv_bytes), "sha256": hashlib.sha256(csv_bytes).hexdigest()},
        "prediction": prediction,
        "training_mode": mode,
        "run_name": f"Benchmark {task_name} with pairs"[:80],
    }


def record(task_name: str, vectors: np.ndarray) -> dict:
    """One prediction record for 19th: each embedding column as a list of numbers."""
    return {column(key): [float(x) for x in vector] for key, vector in zip(spec(task_name).texts, vectors)}


def item_texts(task_name: str, item: dict) -> list[str]:
    return [item[key] for key in spec(task_name).texts]


# ---- the model store ----------------------------------------------------------------------------


def model_path(task_name: str, endpoint: str):
    return MODELS / f"{task_name}-{endpoint}.json"


def save_model(task_name: str, endpoint: str, info: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path(task_name, endpoint).write_text(json.dumps(info, indent=1))


def load_model(task_name: str, endpoint: str) -> dict:
    path = model_path(task_name, endpoint)
    if not path.exists():
        raise NotTrained(f"No {endpoint} model for '{task_name}'. Run: uv run python -m tools.train {task_name}")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Linear:
    """A fitted logistic regression as plain arrays: coef (k, d) or (1, d) for two classes."""

    classes: list[str]
    coef: np.ndarray
    intercept: np.ndarray

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        scores = self.coef @ x + self.intercept
        if len(self.classes) == 2:
            positive = 1 / (1 + np.exp(-scores[0]))
            return np.array([1 - positive, positive])
        scores = np.exp(scores - scores.max())
        return scores / scores.sum()

    def save(self, path) -> None:
        np.savez(path, classes=np.array(self.classes), coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path) -> "Linear":
        with np.load(path, allow_pickle=False) as data:
            return cls([str(c) for c in data["classes"]], data["coef"], data["intercept"])


_baselines: dict[str, Linear] = {}
_lock = threading.Lock()


def baseline_model(task_name: str) -> Linear:
    with _lock:
        if task_name not in _baselines:
            info = load_model(task_name, "local_baseline")
            _baselines[task_name] = Linear.load(MODELS / info["file"])
        return _baselines[task_name]


# ---- the local baseline -------------------------------------------------------------------------


def train_baseline(task_name: str, x: np.ndarray, labels: list, seed: int = 0) -> tuple[Linear, dict]:
    """Logistic regression with its strength chosen by 3-fold cross-validation on the training rows only."""
    from sklearn.linear_model import LogisticRegressionCV

    model = LogisticRegressionCV(
        Cs=[0.1, 1, 10, 100, 1000], cv=3, max_iter=3000, scoring="neg_log_loss", random_state=seed,
        l1_ratios=(0.0,), use_legacy_attributes=True,
    )
    started = time.perf_counter()
    model.fit(x, np.array([str(label) for label in labels]))
    seconds = time.perf_counter() - started
    linear = Linear([str(c) for c in model.classes_], model.coef_.astype(np.float64), model.intercept_.astype(np.float64))
    return linear, {"C": float(np.atleast_1d(model.C_)[0]), "train_seconds": round(seconds, 1)}


def baseline_answer(model: Linear, x: np.ndarray) -> dict:
    table = {label: float(p) for label, p in zip(model.classes, model.probabilities(x))}
    return {"label": None, "probabilities": table, "expected": None}


# ---- reading 19th's answer ----------------------------------------------------------------------


def _probability_table(output: dict, declared: list | None = None) -> dict[str, float]:
    """declared: the class order to assume when 19th lists probabilities without class names."""
    probabilities = output.get("probabilities")
    names = output.get("classes") or output.get("labels") or declared
    if isinstance(probabilities, dict):
        return {str(k): v for k, v in probabilities.items()}
    if isinstance(probabilities, list) and isinstance(names, list) and len(names) == len(probabilities):
        if all(not isinstance(p, dict) for p in probabilities):
            return {str(name): p for name, p in zip(names, probabilities)}
    table = {}
    for entry in probabilities if isinstance(probabilities, list) else []:
        if isinstance(entry, dict):
            name = next((entry[k] for k in ("label", "class", "value") if k in entry), None)
            if name is not None:
                table[str(name)] = entry.get("probability")
    return table


def reduce_19th(task_name: str, output) -> dict | None:
    """19th's `output` object for one record, reduced to the shared answer shape. None if unreadable."""
    if not isinstance(output, dict):
        return None
    chosen = spec(task_name)
    # 19th lists probabilities without class names. Checked on 2026-11-20: ordered classes come in the
    # declared order (its expected_grade recomputes from them), and multiclass labels in sorted order
    # (the highest probability is 19th's own label on every intent answer).
    declared = {"ordered_class": chosen.classes, "multiclass_label": sorted(chosen.classes, key=str)}
    table = _probability_table(output, declared.get(chosen.output_kind))
    if chosen.output_kind == "event_probability":
        # 19th answers {"event_probability": p, "label": ..., "probabilities": [p, 1 - p]} with no class
        # names, so the table is built from event_probability, the chance of positive_label.
        p = output.get("event_probability", output.get("probability"))
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            positive = chosen.extra["positive_label"]
            negative = next(c for c in chosen.classes if c != positive)
            table = {positive: p, negative: 1 - p}
        else:
            table = {}
    label = output.get("grade", output.get("label", output.get("prediction")))
    expected = next((output[k] for k in ("expected_grade", "expected", "expected_value") if k in output), None)
    return {"label": label, "probabilities": table, "expected": expected}


def reduced(endpoint: str, raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    if endpoint in NINETEENTH_MODES:
        return reduce_19th(raw.get("task"), raw.get("output"))
    return raw if isinstance(raw.get("probabilities"), dict) else None


# ---- the endpoint calls -------------------------------------------------------------------------


_http = threading.local()


def _http_client():
    if getattr(_http, "client", None) is None:
        _http.client = nineteenth.make_http(timeout=60)
    return _http.client


def _embed_item(client, task: Task, item: dict) -> tuple[np.ndarray, int, int | None]:
    return embeddings.embed_timed(client, item_texts(task.name, item))


def nineteenth_call(client, task: Task, item: dict, endpoint: str = "nineteenth") -> dict:
    url = load_model(task.name, endpoint)["prediction_url"]
    vectors, embed_ms, tokens = _embed_item(client, task, item)
    started = time.perf_counter()
    output = nineteenth.predict(_http_client(), url, record(task.name, vectors))
    predict_ms = round((time.perf_counter() - started) * 1000)
    return {
        "raw": {"task": task.name, "output": output},
        "latency_ms": embed_ms + predict_ms,
        "embed_ms": embed_ms,
        "predict_ms": predict_ms,
        "input_tokens": tokens,
        "cached_tokens": None,
        "output_tokens": 0,
    }


def nineteenth_ultra_call(client, task: Task, item: dict) -> dict:
    return nineteenth_call(client, task, item, endpoint="nineteenth_ultra")


def baseline_call(client, task: Task, item: dict) -> dict:
    model = baseline_model(task.name)
    vectors, embed_ms, tokens = _embed_item(client, task, item)
    started = time.perf_counter()
    answer = baseline_answer(model, features(task.name, [v[None, :] for v in vectors])[0])
    predict_ms = round((time.perf_counter() - started) * 1000)
    return {
        "raw": answer,
        "latency_ms": embed_ms + predict_ms,
        "embed_ms": embed_ms,
        "predict_ms": predict_ms,
        "input_tokens": tokens,
        "cached_tokens": None,
        "output_tokens": 0,
    }


def idempotency_key(task_name: str, csv_bytes: bytes, request: dict) -> str:
    """The same data and settings give the same key, so rerunning tools.train recovers the run."""
    fingerprint = json.dumps({k: v for k, v in request.items() if k not in ("wait", "inline")}, sort_keys=True)
    return str(uuid.uuid5(NAMESPACE, f"{task_name}:{fingerprint}"))
