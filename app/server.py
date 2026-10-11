"""Web app: serves the dashboard, the saved reports and the live playground.

Usage: uv run uvicorn app.server:app --port 8643
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictStr

from bench.endpoints import ENDPOINTS, make_client
from bench.tasks import TASKS
from bench.tasks.base import ROOT, Task

RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
MAX_INPUT_CHARS = 20_000

load_dotenv(ROOT / ".env")
app = FastAPI(title="Decisions API benchmark")


class TryBody(BaseModel):
    fields: dict[str, StrictStr]


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise HTTPException(404, f"Unknown task '{name}'.")
    return TASKS[name]


def read_report(filename: str) -> dict:
    path = RESULTS / filename
    if not path.exists():
        raise HTTPException(404, "No results yet. Run: uv run python -m bench.report")
    return json.loads(path.read_text())


@app.get("/api/tasks")
def tasks() -> list[dict]:
    return [
        {
            "name": task.name,
            "title": task.title,
            "fields": [list(field) for field in task.fields],
            "questions": task.questions,
            "examples": task.examples,
        }
        for task in TASKS.values()
    ]


@app.get("/api/summary")
def summary() -> dict:
    return read_report("summary.json")


@app.get("/api/results/{task}")
def results(task: str) -> dict:
    return read_report(f"{get_task(task).name}.json")


@app.post("/api/try/{task}")
def try_task(task: str, body: TryBody) -> dict:
    chosen = get_task(task)
    item = {"id": "playground"}
    for key, label in chosen.fields:
        item[key] = body.fields.get(key, "").strip()
        if not item[key]:
            raise HTTPException(400, f"{label} is empty.")
    if sum(len(item[key]) for key, _ in chosen.fields) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"Input is longer than {MAX_INPUT_CHARS:,} characters.")

    try:
        client = make_client()
    except openai.OpenAIError:
        raise HTTPException(503, "OPENAI_API_KEY is missing. Add it to .env and restart the server.")

    def ask(name: str):
        try:
            return ENDPOINTS[name](client, chosen, item), None
        except openai.OpenAIError as error:
            return None, error

    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
        outcomes = dict(zip(ENDPOINTS, pool.map(ask, ENDPOINTS)))

    errors = {name: error for name, (_, error) in outcomes.items() if error is not None}
    if any(isinstance(error, openai.AuthenticationError) for error in errors.values()):
        raise HTTPException(503, "OpenAI rejected the API key. Check OPENAI_API_KEY in .env.")
    if isinstance(errors.get("decisions"), openai.PermissionDeniedError):
        raise HTTPException(503, f"This OpenAI account cannot use the Decisions API: {errors['decisions']}")
    if len(errors) == len(outcomes):
        raise HTTPException(502, f"OpenAI request failed: {next(iter(errors.values()))}")

    answers = {}
    for name, (row, error) in outcomes.items():
        if error is not None:
            answers[name] = {"error": str(error)}
            continue
        answers[name] = {
            **chosen.to_prediction(name, row["raw"]),
            "latency_ms": row["latency_ms"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        }
    return {"answers": answers}


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
