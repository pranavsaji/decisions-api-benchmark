"""Download the three public datasets and write seeded samples to .tmp/data/.

Usage: uv run python -m tools.fetch
No API key is needed. Writes <task>.jsonl (the scored items) and <task>-train.jsonl (labelled
examples for the trained models, kept apart from the scored items) for each task, and manifest.json.
"""
import csv
import io
import json
import random
from collections import Counter, defaultdict

import httpx
import pyarrow.parquet as pq

from bench.tasks.base import DATA
from bench.tasks.intent import INTENTS

SEED = 20261010
HF = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
INJECTION_PER_CLASS = {"deepset": 200, "jackhhao": 100}
# Each training row carries two 1,536-number embeddings, so the relevance training set is capped
# to keep the upload small. Grade 3 has fewer judgements than the cap, so it is taken whole.
RELEVANCE_TRAIN_PER_GRADE = 1000


def _take(rows: list, count: int, rng: random.Random, what: str, unit: str = "rows") -> list:
    if len(rows) < count:
        raise ValueError(f"{what} has {len(rows)} {unit}, need {count}")
    return rng.sample(rows, count)


def sample_intent(rows: list[dict], per_intent: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_intent = defaultdict(list)
    for index, row in enumerate(rows):
        by_intent[row["category"]].append({"id": f"b77-{index}", "text": row["text"], "label": row["category"]})
    items = []
    for intent in sorted(by_intent):
        items += _take(by_intent[intent], per_intent, rng, f"'{intent}'")
    rng.shuffle(items)
    return items


def injection_pool(deepset: list[dict], jackhhao: list[dict]) -> list[dict]:
    """Every usable row from both sources, blank and repeated text removed, in source order."""
    candidates = [
        {"id": r["id"], "text": r["text"], "label": "attack" if r["label"] == 1 else "benign", "source": "deepset"}
        for r in deepset
    ] + [
        {"id": r["id"], "text": r["prompt"], "label": "attack" if r["type"] == "jailbreak" else "benign", "source": "jackhhao"}
        for r in jackhhao
    ]
    seen = set()
    pool = []
    for candidate in candidates:
        text = (candidate["text"] or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        pool.append({**candidate, "text": text})
    return pool


def sample_injection(deepset: list[dict], jackhhao: list[dict], per_class: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools = defaultdict(list)
    for candidate in injection_pool(deepset, jackhhao):
        pools[(candidate["source"], candidate["label"])].append(candidate)
    items = []
    for source, count in per_class.items():
        for label in ("attack", "benign"):
            items += _take(pools[(source, label)], count, rng, f"{source} {label}")
    rng.shuffle(items)
    return items


def judged_pairs(corpus: list[dict], queries: list[dict], qrels: list[dict]) -> dict[int, list[dict]]:
    """Query-passage pairs with a human grade, grouped by grade, for passages present in the corpus."""
    passages = {str(row["_id"]): row["text"] for row in corpus}
    query_text = {str(row["_id"]): row["text"] for row in queries}
    by_grade = defaultdict(list)
    for row in qrels:
        query_id, passage_id = str(row["query-id"]), str(row["corpus-id"])
        if query_id in query_text and passage_id in passages:
            grade = int(float(row["score"]))
            by_grade[grade].append(
                {
                    "id": f"{query_id}-{passage_id}",
                    "query": query_text[query_id],
                    "passage": passages[passage_id],
                    "label": grade,
                }
            )
    return by_grade


def sample_relevance(corpus: list[dict], queries: list[dict], qrels: list[dict], per_grade: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_grade = judged_pairs(corpus, queries, qrels)
    items = []
    for grade in range(4):
        items += _take(by_grade[grade], per_grade, rng, f"grade {grade}", "pairs")
    rng.shuffle(items)
    return items


def _same_text(text: str) -> str:
    return " ".join(text.lower().split())


def training_intent(rows: list[dict], scored: list[dict]) -> list[dict]:
    """The train split, without any message that repeats a scored item (the official split has a few)."""
    taken = {_same_text(item["text"]) for item in scored}
    return [
        {"id": f"b77-train-{n}", "text": row["text"], "label": row["category"]}
        for n, row in enumerate(rows)
        if _same_text(row["text"]) not in taken
    ]


def training_injection(pool: list[dict], scored: list[dict], seed: int) -> list[dict]:
    """Every pooled row whose text is not one of the scored items."""
    taken = {_same_text(item["text"]) for item in scored}
    rows = [row for row in pool if _same_text(row["text"]) not in taken]
    random.Random(seed).shuffle(rows)
    return rows


def training_relevance(by_grade: dict[int, list[dict]], cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for grade in range(4):
        pairs = by_grade[grade]
        rows += rng.sample(pairs, min(cap, len(pairs)))
    rng.shuffle(rows)
    return rows


def download(url: str) -> bytes:
    response = httpx.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    return response.content


def read_parquet(repo: str, split: str) -> list[dict]:
    return pq.read_table(io.BytesIO(download(HF.format(repo=repo, split=split)))).to_pylist()


def banking77(split: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(download(BANKING77.format(split=split)).decode("utf-8"))))
    unknown = {row["category"] for row in rows} - set(INTENTS)
    if unknown:
        raise ValueError(f"Banking77 has intents this project does not list: {sorted(unknown)}")
    return rows


def fetch_intent() -> tuple[list[dict], dict, list[dict], dict]:
    items = sample_intent(banking77("test"), per_intent=10, seed=SEED)
    training = training_intent(banking77("train"), items)
    return items, {
        "name": "Banking77 (test split)",
        "source": BANKING77.format(split="test"),
        "licence": "CC BY 4.0",
        "sampling": "10 messages per intent, seeded random sample.",
    }, training, {
        "name": "Banking77 (train split)",
        "source": BANKING77.format(split="train"),
        "sampling": "Every message in the train split except the few that repeat a scored test message.",
    }


def fetch_injection() -> tuple[list[dict], dict, list[dict], dict]:
    deepset = [
        {"id": f"deepset-{split}-{n}", **row}
        for split in ("train", "test")
        for n, row in enumerate(read_parquet("deepset/prompt-injections", split))
    ]
    jackhhao = [
        {"id": f"jackhhao-{split}-{n}", **row}
        for split in ("train", "test")
        for n, row in enumerate(read_parquet("jackhhao/jailbreak-classification", split))
    ]
    items = sample_injection(deepset, jackhhao, INJECTION_PER_CLASS, seed=SEED)
    training = training_injection(injection_pool(deepset, jackhhao), items, seed=SEED)
    return items, {
        "name": "deepset/prompt-injections and jackhhao/jailbreak-classification",
        "source": "https://huggingface.co/datasets/deepset/prompt-injections, https://huggingface.co/datasets/jackhhao/jailbreak-classification",
        "licence": "Apache 2.0 (both, as tagged on Hugging Face)",
        "sampling": "Per class: 200 from deepset and 100 from jackhhao, seeded random sample, duplicates removed.",
    }, training, {
        "name": "The same two datasets, scored items removed",
        "source": "https://huggingface.co/datasets/deepset/prompt-injections, https://huggingface.co/datasets/jackhhao/jailbreak-classification",
        "sampling": "Every row left after removing blank and repeated text and every scored item's text.",
    }


def fetch_relevance() -> tuple[list[dict], dict, list[dict], dict]:
    corpus = read_parquet("dmrau/trec_dl19", "corpus")
    queries = read_parquet("dmrau/trec_dl19", "queries")
    qrels = read_parquet("dmrau/trec_dl19-qrels", "test")
    items = sample_relevance(corpus, queries, qrels, per_grade=150, seed=SEED)
    training = training_relevance(
        judged_pairs(read_parquet("dmrau/trec_dl20", "corpus"), read_parquet("dmrau/trec_dl20", "queries"),
                     read_parquet("dmrau/trec_dl20-qrels", "test")),
        RELEVANCE_TRAIN_PER_GRADE,
        seed=SEED,
    )
    scored_queries = {item["query"] for item in items}
    if any(row["query"] in scored_queries for row in training):
        raise ValueError("A TREC DL 2020 training query is also a scored DL 2019 query")
    return items, {
        "name": "TREC Deep Learning 2019 passage judgements (MS MARCO passages)",
        "source": "https://huggingface.co/datasets/dmrau/trec_dl19, https://huggingface.co/datasets/dmrau/trec_dl19-qrels",
        "licence": "MS MARCO terms: non-commercial research use",
        "sampling": "150 query-passage pairs per grade, seeded random sample from judged passages present in the corpus subset.",
    }, training, {
        "name": "TREC Deep Learning 2020 passage judgements (MS MARCO passages)",
        "source": "https://huggingface.co/datasets/dmrau/trec_dl20, https://huggingface.co/datasets/dmrau/trec_dl20-qrels",
        "sampling": f"Up to {RELEVANCE_TRAIN_PER_GRADE:,} pairs per grade, seeded random sample. Different queries from the scored DL 2019 items.",
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for task_name, fetch in (("intent", fetch_intent), ("injection", fetch_injection), ("relevance", fetch_relevance)):
        items, info, training, training_info = fetch()
        (DATA / f"{task_name}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
        (DATA / f"{task_name}-train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in training))
        counts = dict(sorted(Counter(str(item["label"]) for item in items).items()))
        training_counts = dict(sorted(Counter(str(row["label"]) for row in training).items()))
        if task_name == "intent":
            counts = {"intents": len(counts), "per_intent": 10}
            training_counts = {"intents": len(training_counts)}
        manifest[task_name] = {
            **info,
            "seed": SEED,
            "counts": {"total": len(items), **counts},
            "training": {**training_info, "counts": {"total": len(training), **training_counts}},
        }
        print(f"{task_name}: {len(items)} items, {len(training)} training rows")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {DATA}")


if __name__ == "__main__":
    main()
