import pytest

from bench import metrics


def test_wilson_interval_known_value():
    low, high = metrics.wilson_interval(80, 100)
    assert low == pytest.approx(0.7112, abs=1e-3)
    assert high == pytest.approx(0.8666, abs=1e-3)
    assert metrics.wilson_interval(0, 0) is None


def test_rate_handles_empty_denominator():
    assert metrics.rate(3, 4) == {"value": 0.75, "ci": metrics.wilson_interval(3, 4), "n": 4}
    assert metrics.rate(0, 0) == {"value": None, "ci": None, "n": 0}


def test_auc_perfect_reversed_ties_and_single_class():
    assert metrics.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert metrics.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert metrics.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert metrics.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert metrics.roc_auc([1, 1], [0.2, 0.9]) is None
    assert metrics.roc_auc([], []) is None


def test_recall_at_fpr_allows_only_the_permitted_false_positives():
    labels = [0] * 10 + [1] * 4
    probabilities = [0.9, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05] + [0.95, 0.6, 0.5, 0.3]
    # 10% of 10 negatives allows one false positive, so the cut-off sits above 0.5.
    assert metrics.recall_at_fpr(labels, probabilities, 0.1) == 0.5
    # No false positives allowed: only the 0.95 attack clears the 0.9 negative.
    assert metrics.recall_at_fpr(labels, probabilities, 0.0) == 0.25
    assert metrics.recall_at_fpr(labels, probabilities, 1.0) == 1.0
    assert metrics.recall_at_fpr([1, 1], [0.5, 0.6], 0.01) is None
    assert metrics.recall_at_fpr([], [], 0.01) is None


def test_quadratic_weighted_kappa_known_values():
    assert metrics.quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 4) == pytest.approx(1.0)
    assert metrics.quadratic_weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0], 4) == pytest.approx(-1.0)
    assert metrics.quadratic_weighted_kappa(
        [0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 1, 2, 3, 3, 1], 4
    ) == pytest.approx(2 / 3)
    assert metrics.quadratic_weighted_kappa([0, 1, 2, 3], [1, 1, 1, 1], 4) == pytest.approx(0.0)
    assert metrics.quadratic_weighted_kappa([], [], 4) is None
    # Everything in one cell: there is no disagreement to weigh, so kappa is undefined.
    assert metrics.quadratic_weighted_kappa([2, 2], [2, 2], 4) is None


def test_reliability_bins_and_error():
    table = metrics.reliability([0.05, 0.15, 0.95, 0.85], [0, 1, 1, 1], 0.0, 1.0, 2)
    assert table["low"] == 0.0 and table["high"] == 1.0
    first, second = table["bins"]
    assert (first["low"], first["high"], first["n"]) == (0.0, 0.5, 2)
    assert first["mean_x"] == pytest.approx(0.1) and first["mean_y"] == pytest.approx(0.5)
    assert second["mean_x"] == pytest.approx(0.9) and second["mean_y"] == pytest.approx(1.0)
    assert table["error"] == pytest.approx(0.25)


def test_reliability_puts_the_top_value_in_the_last_bin_and_handles_empty_input():
    table = metrics.reliability([3.0], [3], 0.0, 3.0, 6)
    assert [b["n"] for b in table["bins"]] == [0, 0, 0, 0, 0, 1]
    assert table["bins"][0]["mean_x"] is None
    empty = metrics.reliability([], [], 0.0, 1.0, 10)
    assert empty["error"] is None and len(empty["bins"]) == 10


def test_percentile_interpolates():
    assert metrics.percentile([10, 20, 30, 40], 0.5) == 25
    assert metrics.percentile([5], 0.95) == 5
    assert metrics.percentile([], 0.5) is None


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    rows = [(1, 0.9), (1, 0.8), (0, 0.7), (1, 0.6), (0, 0.4), (0, 0.3), (1, 0.35), (0, 0.1)]

    def auc(sample):
        return metrics.roc_auc([y for y, _ in sample], [p for _, p in sample])

    first = metrics.bootstrap_ci(auc, rows, resamples=200, seed=0)
    assert first == metrics.bootstrap_ci(auc, rows, resamples=200, seed=0)
    assert first[0] <= auc(rows) <= first[1]
    assert metrics.bootstrap_ci(lambda sample: 0.4, rows, resamples=50) == [0.4, 0.4]
    assert metrics.bootstrap_ci(auc, [], resamples=50) is None
    assert metrics.bootstrap_ci(lambda sample: None, rows, resamples=50) is None


def test_histogram_counts_and_open_last_bucket():
    assert metrics.histogram([50, 150, 150, 12000], [0, 100, 200]) == [
        {"low": 0, "high": 100, "n": 1},
        {"low": 100, "high": 200, "n": 2},
        {"low": 200, "high": None, "n": 1},
    ]


def test_cost_per_1000_uses_the_price_table():
    assert metrics.cost_per_1000(400, 0, 0) == pytest.approx(0.04)
    assert metrics.cost_per_1000(2000, 1500, 300) == pytest.approx(0.215)
    assert metrics.cost_per_1000(400, None, None) == pytest.approx(0.04)
    assert metrics.cost_per_1000(None, None, None) is None
