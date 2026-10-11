"""Statistics for the benchmark reports. Pure functions, no I/O."""
import math
import random

# gpt-6-luna, US dollars per million tokens. The Decisions API bills input tokens only.
PRICES_PER_MILLION = {"input": 0.10, "cached_input": 0.01, "output": 0.50}
PRICES_READ_ON = "2026-10-10"
LATENCY_EDGES_MS = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(numerator: int, denominator: int) -> dict:
    return {
        "value": numerator / denominator if denominator else None,
        "ci": wilson_interval(numerator, denominator),
        "n": denominator,
    }


def roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(labels)), key=lambda i: probabilities[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probabilities[order[j + 1]] == probabilities[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        rank_sum += average_rank * sum(labels[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def recall_at_fpr(labels: list[int], probabilities: list[float], max_fpr: float) -> float | None:
    """Share of positives scoring above the strictest cut-off that keeps false positives within max_fpr."""
    negatives = sorted((p for y, p in zip(labels, probabilities) if not y), reverse=True)
    positives = [p for y, p in zip(labels, probabilities) if y]
    if not negatives or not positives:
        return None
    allowed = math.floor(max_fpr * len(negatives))
    if allowed >= len(negatives):
        return 1.0
    floor = negatives[allowed]
    return sum(p > floor for p in positives) / len(positives)


def quadratic_weighted_kappa(truth: list[int], predicted: list[int], levels: int) -> float | None:
    total = len(truth)
    if total == 0 or levels < 2:
        return None
    observed = [[0] * levels for _ in range(levels)]
    for t, p in zip(truth, predicted):
        observed[t][p] += 1
    row = [sum(r) for r in observed]
    column = [sum(observed[i][j] for i in range(levels)) for j in range(levels)]
    disagreement = expected = 0.0
    for i in range(levels):
        for j in range(levels):
            weight = (i - j) ** 2 / (levels - 1) ** 2
            disagreement += weight * observed[i][j]
            expected += weight * row[i] * column[j] / total
    return 1 - disagreement / expected if expected else None


def reliability(xs: list[float], ys: list[float], low: float, high: float, bins: int) -> dict:
    """Bin xs across [low, high] and compare each bin's mean x with its mean y."""
    members: list[list[int]] = [[] for _ in range(bins)]
    width = (high - low) / bins
    for index, x in enumerate(xs):
        members[min(bins - 1, max(0, int((x - low) / width)))].append(index)
    table = []
    error = 0.0
    for b, indices in enumerate(members):
        row = {"low": low + b * width, "high": low + (b + 1) * width, "n": len(indices), "mean_x": None, "mean_y": None}
        if indices:
            row["mean_x"] = sum(xs[i] for i in indices) / len(indices)
            row["mean_y"] = sum(ys[i] for i in indices) / len(indices)
            error += len(indices) / len(xs) * abs(row["mean_x"] - row["mean_y"])
        table.append(row)
    return {"low": low, "high": high, "bins": table, "error": error if xs else None}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(statistic, rows: list, resamples: int = 1000, seed: int = 0) -> list[float] | None:
    """95% percentile interval for statistic(rows), resampling rows with replacement."""
    if not rows:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        value = statistic([rows[rng.randrange(len(rows))] for _ in rows])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [percentile(values, 0.025), percentile(values, 0.975)]


def histogram(values: list[float], edges: list[float]) -> list[dict]:
    buckets = [
        {"low": low, "high": high, "n": 0}
        for low, high in zip(edges, [*edges[1:], None])
    ]
    for value in values:
        for bucket in reversed(buckets):
            if value >= bucket["low"]:
                bucket["n"] += 1
                break
    return buckets


def cost_per_1000(
    input_tokens: float | None, cached_tokens: float | None, output_tokens: float | None
) -> float | None:
    """Dollars per 1,000 calls, given mean token counts per call."""
    if input_tokens is None:
        return None
    cached = cached_tokens or 0
    per_call = (
        (input_tokens - cached) * PRICES_PER_MILLION["input"]
        + cached * PRICES_PER_MILLION["cached_input"]
        + (output_tokens or 0) * PRICES_PER_MILLION["output"]
    ) / 1_000_000
    return per_call * 1000
