from collections import Counter

import pytest

from tools import fetch


def intent_rows():
    return [{"text": f"{category} message {n}", "category": category} for category in ("a", "b") for n in range(5)]


def test_sample_intent_takes_the_same_number_per_intent_and_is_seeded():
    sample = fetch.sample_intent(intent_rows(), per_intent=3, seed=1)
    assert Counter(item["label"] for item in sample) == {"a": 3, "b": 3}
    assert len({item["id"] for item in sample}) == 6
    assert all(item["id"].startswith("b77-") and item["text"] for item in sample)
    assert sample == fetch.sample_intent(intent_rows(), per_intent=3, seed=1)
    assert sample != fetch.sample_intent(intent_rows(), per_intent=3, seed=2)
    # Shuffled, so a --limit run sees a mix of intents.
    assert [item["label"] for item in sample] != sorted(item["label"] for item in sample)


def test_sample_intent_fails_when_an_intent_is_short():
    with pytest.raises(ValueError, match="'a' has 5 rows, need 6"):
        fetch.sample_intent(intent_rows(), per_intent=6, seed=1)


def injection_sources():
    deepset = [{"id": f"deepset-{n}", "text": f"deepset text {n}", "label": n % 2} for n in range(20)]
    jackhhao = [
        {"id": f"jackhhao-{n}", "prompt": f"jackhhao prompt {n}", "type": "jailbreak" if n % 2 else "benign"}
        for n in range(20)
    ]
    return deepset, jackhhao


def test_sample_injection_balances_class_and_source():
    deepset, jackhhao = injection_sources()
    sample = fetch.sample_injection(deepset, jackhhao, {"deepset": 4, "jackhhao": 2}, seed=1)
    assert Counter((item["source"], item["label"]) for item in sample) == {
        ("deepset", "attack"): 4,
        ("deepset", "benign"): 4,
        ("jackhhao", "attack"): 2,
        ("jackhhao", "benign"): 2,
    }
    assert set(sample[0]) == {"id", "text", "label", "source"}
    assert sample == fetch.sample_injection(deepset, jackhhao, {"deepset": 4, "jackhhao": 2}, seed=1)


def test_sample_injection_drops_blank_and_repeated_text():
    deepset, jackhhao = injection_sources()
    deepset += [{"id": "blank", "text": "   ", "label": 1}, {"id": "repeat", "text": "deepset text 1", "label": 1}]
    sample = fetch.sample_injection(deepset, jackhhao, {"deepset": 10, "jackhhao": 2}, seed=1)
    ids = {item["id"] for item in sample}
    assert "blank" not in ids and "repeat" not in ids


def test_sample_injection_fails_when_a_class_is_short():
    deepset, jackhhao = injection_sources()
    with pytest.raises(ValueError, match="deepset attack has 10 rows, need 11"):
        fetch.sample_injection(deepset, jackhhao, {"deepset": 11, "jackhhao": 2}, seed=1)


def relevance_sources():
    corpus = [{"_id": str(n), "text": f"passage {n}", "title": ""} for n in range(40)]
    queries = [{"_id": "q1", "text": "first query", "title": ""}, {"_id": "q2", "text": "second query", "title": ""}]
    qrels = [
        {"query-id": "q1" if n % 2 else "q2", "corpus-id": str(n), "score": str(n % 4)} for n in range(40)
    ]
    return corpus, queries, qrels


def test_sample_relevance_takes_the_same_number_per_grade():
    sample = fetch.sample_relevance(*relevance_sources(), per_grade=5, seed=1)
    assert Counter(item["label"] for item in sample) == {0: 5, 1: 5, 2: 5, 3: 5}
    first = sample[0]
    assert set(first) == {"id", "query", "passage", "label"}
    assert first["query"] in ("first query", "second query") and first["passage"].startswith("passage ")
    assert isinstance(first["label"], int)
    assert sample == fetch.sample_relevance(*relevance_sources(), per_grade=5, seed=1)


def test_sample_relevance_skips_judgements_whose_passage_or_query_is_missing():
    corpus, queries, qrels = relevance_sources()
    qrels += [
        {"query-id": "q1", "corpus-id": "missing", "score": "3"},
        {"query-id": "missing", "corpus-id": "1", "score": "3"},
    ]
    sample = fetch.sample_relevance(corpus, queries, qrels, per_grade=10, seed=1)
    assert len(sample) == 40
    assert not any("missing" in item["id"] for item in sample)


def test_sample_relevance_accepts_float_scores_and_fails_when_a_grade_is_short():
    corpus, queries, qrels = relevance_sources()
    floats = [{**row, "score": float(row["score"])} for row in qrels]
    assert len(fetch.sample_relevance(corpus, queries, floats, per_grade=2, seed=1)) == 8
    with pytest.raises(ValueError, match="grade 0 has 10 pairs, need 11"):
        fetch.sample_relevance(corpus, queries, qrels, per_grade=11, seed=1)
