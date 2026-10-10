# Decisions API benchmark: design

Date: 2026-10-10
Status: awaiting review

## Purpose

Find out how good OpenAI's Decisions API (`POST /v1/decisions`, model `gpt-6-luna`) is on the
kind of work it should be best at, and publish the evidence.

An earlier project tested it on skin-cancer images. That showed the speed (170 ms against
3,814 ms median for the Responses API) but the task was too hard for either endpoint
(AUC 0.65 to 0.70), so it could not show whether the answers are trustworthy. It also compared
against the Responses API at default settings only, which leaves open how much of the speed gap
is reasoning overhead.

This project tests three best-case tasks, one per Decisions question type, against two
Responses API baselines.

## Success criteria

- For each task, a published comparison of three endpoints on quality, calibration, latency,
  cost and reliability, with confidence intervals.
- A reader can reproduce every number from a seeded sample with one command per step.
- A web UI shows the results and lets a visitor try each task live on all three endpoints.
- The comparison is fair enough that the speed-tuned Responses baseline is a credible opponent.

## Out of scope

- Prompt tuning. Each task uses one prompt, written before the run.
- Non-LLM baselines (embeddings, BM25).
- Models other than `gpt-6-luna`.
- Image inputs. All three tasks are text.
- Deployment. The app runs locally.

## Endpoints compared

| Name | Call |
|---|---|
| `decisions` | `client.decisions.create` with typed questions |
| `responses_default` | `client.responses.create` with a strict JSON schema, default settings |
| `responses_tuned` | the same call with the lowest reasoning effort the model accepts |

The accepted reasoning-effort values for `gpt-6-luna` are checked against the API during
implementation. If the model exposes no such setting, `responses_tuned` is dropped and the
README says so.

## Tasks

| | Intent routing | Injection guardrail | Search relevance |
|---|---|---|---|
| Module | `tasks/intent.py` | `tasks/injection.py` | `tasks/relevance.py` |
| Input | one customer message | one prompt | a query and a passage |
| Question type | `choice`, 77 options | `predicate` | `score`, levels 0 to 3 |
| Dataset | Banking77 | deepset/prompt-injections, topped up from a second public set if needed | TREC Deep Learning passage judgements |
| Sample | 770 (10 per intent) | 600 (300 attack, 300 benign) | 600 (150 per grade) |
| Headline metrics | top-1 accuracy, top-3 accuracy | AUC, recall at 1% false-positive rate | quadratic-weighted kappa, exact-grade accuracy |
| Calibration | confidence against correctness | probability against attack rate | expected score against human grade |

All samples are seeded and written to `.tmp/data/<task>.jsonl` with a manifest recording the
seed, source and counts.

The Responses baselines must return what the Decisions API returns, so the metrics apply to all
three: a ranked top three with probabilities for intent, a probability for injection, and a
grade with a probability per level for relevance.

Relevance data: the full MS MARCO passage collection is about 3 GB. Implementation fetches only
the judged passages if a light source exists. If not, the task falls back to BEIR TREC-COVID
(grades 0 to 2, three levels, 200 per grade) and the spec's sample and metrics adjust to match.

## Architecture

```
decisions-api-benchmark/
  bench/
    endpoints.py     # the three callers
    runner.py        # interleaved run, concurrency, JSONL cache, resume
    metrics.py       # accuracy, Wilson intervals, calibration, AUC, kappa, latency, cost
    report.py        # builds results/<task>.json and results/summary.json
    tasks/
      base.py        # the Task interface
      intent.py
      injection.py
      relevance.py
  tools/fetch.py     # seeded dataset samples
  app/
    server.py
    static/          # index.html, app.js, styles.css
  tests/
  results/           # committed reports
  docs/
```

Stack: Python, uv, the `openai` SDK, FastAPI, plain HTML, CSS and JavaScript. Same as the
skin-lesion project.

### Task interface

Each task module provides:

- `name`, `title`: identifiers for the CLI and the UI.
- `load() -> list[Item]`: sampled items, each with an `id`, the input text and the gold label.
- `questions`: the Decisions API questions.
- `responses_schema` and `responses_prompt(item)`: the same request for the Responses API. The
  prompt text is generated from `questions`, so instructions are word-for-word identical.
- `to_prediction(endpoint, raw) -> Prediction`: normalises any endpoint's answer to one shape,
  or marks it as a refusal or as malformed.
- `score(items, predictions) -> dict`: the task's metrics.

`runner.py` and `endpoints.py` contain no task-specific code.

### Endpoints

`endpoints.py` exposes one function per endpoint with the same signature: it takes a client, a
task and an item, and returns the raw answer, latency in milliseconds, and the input and output
token counts reported by the API.

### Runner

`python -m bench.runner <task> [--limit N] [--workers 8]`

- Builds the work list as every (item, endpoint) pair, ordered so the three endpoints for one
  item run next to each other. All three share one worker pool, so they see the same load.
- Appends each result to `.tmp/runs/<task>.jsonl`. On start it skips pairs already cached.
- Retries API errors twice. A pair that still fails is cached as an error with the message.

### Reports

`python -m bench.report` reads the caches and writes:

- `results/<task>.json`: per endpoint, the task metrics with confidence intervals, calibration
  bins, latency p50, p95 and mean, tokens and cost per 1,000 decisions, counts of answered,
  refused, malformed and errored, and the per-item cases.
- `results/summary.json`: one row per task and endpoint with the headline metric, median
  latency and cost.

Cost uses reported token counts and a price table in `bench/metrics.py` with the date the prices
were read.

## Fairness rules

- Same items, same model, same instructions on every endpoint.
- Interleaved order and a shared worker pool.
- Thresholds fixed before the run: 0.5 for the injection predicate.
- Refusals, malformed answers and errors are reported per endpoint and never dropped silently.
- No prompt changes after the full run starts. If a prompt must change, all three endpoints rerun.

## Web UI

One page with a summary tab and a tab per task.

- Summary: tasks as rows, endpoints as columns, showing headline metric, median latency and
  cost per 1,000 decisions.
- Task tab: comparison table with confidence intervals, latency distribution chart, calibration
  chart, and a case list that can be filtered to cases where the endpoints disagree.
- Playground on each task tab: enter an input, run it, and see all three endpoints' answers,
  probabilities, latency and token counts side by side.

### Server

| Route | Purpose |
|---|---|
| `GET /api/tasks` | tasks, their questions and example inputs |
| `GET /api/summary` | `results/summary.json` |
| `GET /api/results/{task}` | `results/<task>.json` |
| `POST /api/try/{task}` | run one input on all three endpoints in parallel |

The dashboard reads committed reports, so it works without an API key. Only the playground
needs `OPENAI_API_KEY`.

## Failure handling

| Situation | Behaviour |
|---|---|
| Refusal | Counted per endpoint, excluded from quality metrics, count shown beside them |
| API error after two retries | Recorded as an error for that pair and reported |
| Malformed Responses output (bad JSON, probability out of range, unknown label) | Counted as malformed for that endpoint and scored as wrong |
| Interrupted run | Resumes from the cache |
| Missing or rejected API key in the playground | 503 with a message naming `OPENAI_API_KEY` |
| Account without Decisions API access | 503 saying so |
| Other upstream failure in the playground | 502 with the upstream message |
| One endpoint fails in the playground | The other two answers still display |
| Results file missing | 404 naming the command that creates it |

## Testing

- `metrics.py`: hand-computed cases for Wilson intervals, AUC, recall at a fixed false-positive
  rate, quadratic-weighted kappa, top-k accuracy and calibration bins.
- Each task's `to_prediction`: recorded sample answers from all three endpoints, including a
  refusal and a malformed answer.
- `runner.py`: a fake endpoint, covering caching, resume, interleaving and error recording.
- `server.py`: routes with a fake endpoint, covering the error mapping above.
- Before the full run: `--limit 20` per task against the real API.

## Run order

1. `python -m tools.fetch` downloads and samples the three datasets.
2. Smoke run at `--limit 20` per task, then print a cost estimate for the full run from the
   observed token counts.
3. Confirm the estimate with the project owner, then run all three tasks in full.
4. `python -m bench.report`, then write the README with the results.

Expected total API cost is a few dollars.

## Open items resolved during implementation

- Which reasoning-effort values `gpt-6-luna` accepts on the Responses API.
- Whether a light source exists for TREC Deep Learning judged passages.
- Whether deepset/prompt-injections has 300 usable cases per class, or needs a top-up source.
