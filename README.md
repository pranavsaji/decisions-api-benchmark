# Decisions API benchmark

A test of OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
(`POST /v1/decisions`, model `gpt-6-luna`, in beta) on three tasks it should be good at, against
the Responses API on the same model. Since 20 November 2026 it also includes
[19th](https://www.19th.com/docs/training), which trains a small model from labelled examples,
and a plain logistic regression trained on the same examples as a control.

This follows an [earlier test on skin-cancer images](https://github.com/pranavsaji/cancer-detection),
where the Decisions API was about 22 times faster but neither endpoint was accurate enough to judge.

![Screenshot of the dashboard](docs/screenshot.png)

## What is compared

| Endpoint | Call |
|---|---|
| Decisions API | typed questions, typed answers with probabilities |
| Responses API, default | the same question, answered as strict JSON at default settings |
| Responses API, tuned | the same, with reasoning effort set to `none` |
| 19th | a model trained by 19th's Training API on labelled examples, called through its hosted prediction URL |
| Local baseline | scikit-learn logistic regression trained on the same examples and inputs, run in-process |

The three OpenAI endpoints see the same items, the same model and the same instructions, and no
examples. Calls are interleaved through one worker pool so that all three see the same load.

19th and the local baseline are different in kind: they learn from labelled examples, kept
apart from the scored items (see below). 19th does not model raw text, so every text is first
turned into a 1,536-number `text-embedding-3-small` vector, and both trained models see exactly
those vectors. For each scored item, each of the two embeds the text in a live call and then
predicts, so their response time includes the embedding. The two run interleaved with each other.

| Task | Question type | Data | Items |
|---|---|---|---|
| Intent routing | `choice`, 77 options | Banking77 | 770 |
| Injection guardrail | `predicate` | deepset/prompt-injections, jackhhao/jailbreak-classification | 600 |
| Search relevance | `score`, 4 levels | TREC Deep Learning 2019 | 600 |

| Task | Training examples for 19th and the local baseline | Rows |
|---|---|---|
| Intent routing | Banking77 train split, without 2 messages that repeat a scored item | 10,001 |
| Injection guardrail | every row of the same two datasets left after removing the scored items' text | 1,349 |
| Search relevance | TREC Deep Learning 2020 judgements: different queries, same passages and scale; up to 1,000 per grade | 2,277 |

19th was trained with `training_mode: high` and the default `fast` model file; training took
32 to 110 seconds per task. The baseline's strength (C) was chosen by 3-fold cross-validation on
the training rows only.

## Results

The OpenAI endpoints ran on 10 October 2026, 19th and the local baseline on 20 November 2026.
Ranges are 95% intervals. Cost uses the token counts each call reported; for 19th and the local
baseline it covers the embedding call only ($0.02 per million tokens), because 19th publishes no
prices during its alpha. The "difference" rows compare each endpoint with the Decisions API, and
19th with the local baseline, on the same items (paired bootstrap); a range that excludes zero is
a real difference.

### Intent routing

| | Decisions API | Responses API, default | Responses API, tuned | 19th | Local baseline |
|---|---|---|---|---|---|
| Top-1 accuracy | 78.0% (75.0 to 80.8) | 83.8% (81.0 to 86.2) | 81.3% (78.4 to 83.9) | 93.9% (92.0 to 95.4) | 94.8% (93.0 to 96.2) |
| Top-3 accuracy | 89.2% (86.8 to 91.2) | 93.9% (92.0 to 95.4) | 92.6% (90.5 to 94.2) | 99.4% (98.5 to 99.7) | 99.5% (98.7 to 99.8) |
| Top-1 accuracy, difference from Decisions API on the same items | baseline | +5.9 points (+3.8 to +7.8) | +3.4 points (+1.3 to +5.3) | +15.9 points (+13.1 to +18.7) | +16.8 points (+14.2 to +19.6) |
| Top-1 accuracy, difference from the local baseline on the same items |  |  |  | -0.9 points (-2.1 to +0.3) | baseline |
| Median response time | 132 ms | 1,966 ms | 1,454 ms | 267 ms | 184 ms |
| 95th-percentile response time | 212 ms | 3,010 ms | 2,071 ms | 377 ms | 261 ms |
| Cost per 1,000 decisions | $0.100 | $0.155 | $0.120 | $0.0003 | $0.0003 |
| Answered | 769 of 770 (1 refused, 0 malformed, 0 errors) | 770 of 770 | 770 of 770 | 770 of 770 | 770 of 770 |

### Injection guardrail

| | Decisions API | Responses API, default | Responses API, tuned | 19th | Local baseline |
|---|---|---|---|---|---|
| AUC | 0.890 (0.865 to 0.912) | 0.948 (0.931 to 0.963) | 0.947 (0.932 to 0.961) | 0.976 (0.966 to 0.985) | 0.971 (0.959 to 0.982) |
| Attacks caught at 1% false alarms | 75.0% (69.9 to 79.8) | 72.3% (67.5 to 77.4) | 71.0% (64.0 to 76.9) | 74.3% (68.8 to 82.3) | 77.7% (55.0 to 85.1) |
| Accuracy at 50% cut-off | 83.8% (80.7 to 86.6) | 83.0% (79.8 to 85.8) | 83.5% (80.3 to 86.3) | 89.3% (86.6 to 91.6) | 90.0% (87.3 to 92.2) |
| AUC, difference from Decisions API on the same items | baseline | +0.058 (+0.037 to +0.079) | +0.057 (+0.036 to +0.077) | +0.086 (+0.064 to +0.111) | +0.082 (+0.058 to +0.106) |
| AUC, difference from the local baseline on the same items |  |  |  | +0.005 (-0.002 to +0.012) | baseline |
| Median response time | 127 ms | 1,368 ms | 1,019 ms | 268 ms | 184 ms |
| 95th-percentile response time | 242 ms | 2,349 ms | 1,651 ms | 380 ms | 247 ms |
| Cost per 1,000 decisions | $0.032 | $0.051 | $0.030 | $0.002 | $0.002 |
| Answered | 600 of 600 | 600 of 600 | 600 of 600 | 600 of 600 | 600 of 600 |

### Search relevance

| | Decisions API | Responses API, default | Responses API, tuned | 19th | Local baseline |
|---|---|---|---|---|---|
| Agreement with human grades (weighted kappa) | 0.581 (0.534 to 0.625) | 0.580 (0.532 to 0.630) | 0.616 (0.565 to 0.666) | 0.080 (0.046 to 0.113) | 0.296 (0.238 to 0.359) |
| Exact grade matched | 43.3% (39.4 to 47.3) | 44.7% (40.7 to 48.7) | 46.3% (42.4 to 50.3) | 26.8% (23.4 to 30.5) | 34.7% (31.0 to 38.6) |
| Agreement with human grades (weighted kappa), difference from Decisions API on the same items | baseline | -0.001 (-0.031 to +0.030) | +0.034 (+0.002 to +0.067) | -0.502 (-0.559 to -0.445) | -0.285 (-0.350 to -0.217) |
| Agreement with human grades (weighted kappa), difference from the local baseline on the same items |  |  |  | -0.216 (-0.282 to -0.157) | baseline |
| Median response time | 127 ms | 1,692 ms | 1,332 ms | 281 ms | 188 ms |
| 95th-percentile response time | 286 ms | 2,527 ms | 2,232 ms | 440 ms | 251 ms |
| Cost per 1,000 decisions | $0.028 | $0.079 | $0.051 | $0.002 | $0.002 |
| Answered | 600 of 600 | 600 of 600 | 600 of 600 | 600 of 600 | 600 of 600 |

### What the results show

- **Speed is the clear win.** The Decisions API answered in a median of 127 to 132 ms on every
  task. That is 11 to 15 times faster than the default Responses API (1,368 to 1,966 ms) and 8 to
  11 times faster than the tuned one (1,019 to 1,454 ms). Turning reasoning off closes only about
  a quarter of the gap, so most of it is not reasoning overhead.
- **It lost on intent routing.** With 77 options, the Decisions API scored 78.0% top-1 against
  83.8% for the default Responses API and 81.3% for the tuned one. On the same items it trails by
  5.9 points (3.8 to 7.8) and 3.4 points (1.3 to 5.3), so it lost to both. Top-3 shows the same
  order (89.2% against 93.9% and 92.6%).
- **Injection detection is mixed.** The Decisions API has a lower AUC (0.890 against 0.948 and
  0.947; it trails both by about 0.06, range 0.04 to 0.08). At fixed cut-offs the three are
  close, with overlapping intervals: 75.0%, 72.3% and 71.0% of attacks caught at 1% false alarms,
  and 83.0% to 83.8% accuracy at the 50% cut-off. All three flag only about half of the deepset
  attacks (106 or 107 of 200) and no benign prompt.
- **Search relevance is a tie with the default Responses API, and slightly behind the tuned
  one.** Weighted kappa is 0.581 against 0.580 (difference -0.001, range -0.031 to +0.030) and
  0.616 (difference +0.034, range +0.002 to +0.067), at about a tenth of the response time.
- **Cost is lower on two tasks, equal on one.** The Decisions API is cheapest on relevance
  ($0.028 against $0.079 and $0.051 per 1,000) and intent ($0.100 against $0.155 and $0.120). On
  injection it matches the tuned Responses API ($0.032 against $0.030), because it reported more
  input tokens for the same short text (323 against 225 on average).
- **All three were reliable.** Out of 5,910 calls there was one refusal (Decisions API, intent),
  no malformed answer and no API error.

### What the trained models show

- **Labelled examples beat every no-examples endpoint on intent and injection.** 19th reached
  93.9% top-1 on intent (against 83.8% for the best OpenAI endpoint) and 0.976 AUC on injection
  (against 0.948).
- **19th did no better than a plain logistic regression on the same inputs.** On the same items
  it trails the local baseline by 0.9 points on intent (range -2.1 to +0.3) and leads it by 0.005
  AUC on injection (range -0.002 to +0.012): both ties.
- **On search relevance it lost to everything.** 19th's weighted kappa is 0.080 against 0.296
  for the local baseline and 0.58 to 0.62 for the OpenAI endpoints. Its expected grade barely
  moves with the human grade (0.76 for grade 0 against 0.93 for grade 3). Relevance depends on
  how query and passage relate; the baseline is given their element-wise product, but 19th gets
  the two embeddings as separate columns because `/v1/fit` refuses the `pairs` field meant for
  this (see limits). Both trained models also learned from 54 queries only.
- **Response time is about twice the Decisions API's, mostly the embedding call.** 19th's median
  was 267 to 281 ms: about 185 ms embedding plus about 80 ms for 19th's hosted prediction call.
  The local baseline answers in under 1 ms after the same embedding (184 to 188 ms in all). 19th's
  estimate for its downloaded model file is 0.03 ms per row on a server CPU; this test used the
  hosted call, not the file.
- **Every call answered.** 1,970 of 1,970 for each, with no errors.

### Limits of this test

- One prompt per task, written before the run, with no tuning.
- One model, one run, one day. Latency was measured from one machine with 8 parallel requests.
- The injection labels come from two community datasets with different ideas of what counts as
  an attack. Results are broken down by source in the dashboard.
- The relevance sample is drawn from judged passages that retrieval systems surfaced, not from
  all judged passages.
- Malformed answers count as wrong in accuracy figures. AUC, recall at 1% false alarms and kappa
  are computed on parsed answers only.
- The Decisions API is in beta and may change.
- 19th is in a private alpha and may change. Its answers list probabilities without class names;
  the order was checked on every answer (sorted labels for intent, the declared order 0 to 3 for
  relevance).
- 19th's `/v1/fit` refuses `column_types` and `pairs`, and read base64 float32 embedding cells as
  free text, so embeddings are sent as JSON arrays of numbers (six significant digits). A
  relevance model with an added query-passage similarity column trained but could not be served
  (404 "model not found", download unavailable), twice, so the reported relevance model has the
  two embeddings only.
- The trained models depend on the embedding model as much as on the learner; another
  embedding could change both.

## Run it yourself

Requires [uv](https://docs.astral.sh/uv/) and an OpenAI API key with Decisions API access. The
full run cost about $0.45 at the prices read on 10 October 2026; the embeddings for the trained
models add about $0.02. 19th needs a Training API key from app.19th.com/api-keys with
`training:write`, `training:read` and `inference:invoke`.

```bash
uv sync
cp .env.example .env                           # then add OPENAI_API_KEY and NINETEENTH_API_KEY

uv run python -m tools.fetch                   # download and sample the datasets (no key needed)
uv run python -m tools.train all               # embed the training rows, fit the baseline, train on 19th
uv run python -m bench.runner all --limit 20   # small paid smoke run
uv run python -m bench.runner all              # full run
uv run python -m bench.report                  # build results/*.json
uv run uvicorn app.server:app --port 8643      # open http://localhost:8643
uv run pytest                                  # unit tests
```

Every answer is cached under `.tmp/runs/`, so an interrupted run resumes without paying twice.
The dashboard reads the committed reports and works without an API key; only the playground
calls OpenAI and 19th.

## Layout

- `bench/tasks/`: one module per task with its question, Responses schema, parser and scorer
- `bench/endpoints.py`: every endpoint call
- `bench/supervised.py`, `bench/nineteenth.py`, `bench/embeddings.py`: the trained models, the
  19th client and the embeddings
- `tools/train.py`: trains 19th and the local baseline
- `bench/runner.py`: interleaved run with caching and retries
- `bench/metrics.py`, `bench/report.py`: statistics and report building
- `tools/fetch.py`: seeded dataset samples
- `app/`: FastAPI server and the dashboard

## Data and licence

Code is released under the [MIT licence](LICENSE).

- Banking77 (Casanueva et al., 2020), CC BY 4.0. Test split scored, train split for training.
- deepset/prompt-injections and jackhhao/jailbreak-classification, from Hugging Face.
- TREC Deep Learning 2019 (scored) and 2020 (training) judgements over MS MARCO passages, which
  are provided for non-commercial research use.

The reports in `results/` include the sampled text so that the dashboard can show each case.
Check each dataset's terms before reusing it.
