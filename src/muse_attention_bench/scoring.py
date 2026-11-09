"""Documented competition-style aggregation, independent of Torch/CUDA.

This is a reconstruction of TASK.md, not the organizer's unpublished scorer.
Bootstrap details and repetition counts are local protocol choices.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and strictly positive")
    return value


def geometric_mean(values) -> float:
    numbers = [_positive(x, "geometric-mean input") for x in values]
    if not numbers:
        raise ValueError("geometric mean requires at least one value")
    return math.exp(statistics.fmean(math.log(x) for x in numbers))


def quantile(values, probability: float) -> float:
    ordered = sorted(values)
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("quantile needs values and a probability in [0, 1]")
    index = (len(ordered) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def paired_ratios(samples: list[dict], overhead_us: float) -> list[float]:
    """Combine adjacent AB/BA repetitions geometrically, cancelling order bias.

    Both event times have the same measured graph fixed cost subtracted. A
    nonpositive corrected time invalidates this measurement; it is never clamped.
    """
    if len(samples) < 4 or len(samples) % 2:
        raise ValueError("need an even number of at least four repetitions")
    if not math.isfinite(overhead_us) or overhead_us < 0:
        raise ValueError("overhead_us must be finite and nonnegative")
    ratios = []
    for start in range(0, len(samples), 2):
        first, second = samples[start:start + 2]
        if first["order"] != "AB" or second["order"] != "BA":
            raise ValueError("adjacent repetitions must be ordered AB then BA")
        values = []
        for sample in (first, second):
            baseline = _positive(sample["baseline_us"], "baseline event time") - overhead_us
            candidate = _positive(sample["candidate_us"], "candidate event time") - overhead_us
            values.append(_positive(baseline, "corrected baseline time") /
                          _positive(candidate, "corrected candidate time"))
        ratios.append(geometric_mean(values))
    return ratios


def summarize_shape(samples: list[dict], overhead_us: float) -> dict:
    ratios = paired_ratios(samples, overhead_us)
    speedup = statistics.median(ratios)
    baseline_us = statistics.median(float(s["baseline_us"]) - overhead_us for s in samples)
    return {
        "speedup": speedup,
        "baseline_us": baseline_us,
        # TASK.md defines candidate shape time through baseline/speedup.
        "candidate_us": baseline_us / speedup,
        "candidate_event_median_us": statistics.median(float(s["candidate_us"]) for s in samples),
        "baseline_event_median_us": statistics.median(float(s["baseline_us"]) for s in samples),
        "fixed_graph_cost_us": overhead_us,
        "paired_ratios": ratios,
        "samples": samples,
    }


def aggregate_cases(case_results: list[dict], case_definitions: list[dict]) -> dict:
    """Weight shapes within each load, then geometric-mean the six load ratios."""
    by_id = {r["id"]: r for r in case_results}
    if len(by_id) != len(case_results):
        raise ValueError("duplicate case result IDs")
    expected = {d["id"] for d in case_definitions}
    if len(expected) != len(case_definitions) or set(by_id) != expected:
        raise ValueError("case results must exactly match unique scored case definitions")
    loads = defaultdict(lambda: {"baseline_us": 0.0, "candidate_us": 0.0, "cases": []})
    for definition in case_definitions:
        result = by_id[definition["id"]]
        weight = _positive(definition["weight"], "case weight")
        load = loads[definition["load"]]
        load["baseline_us"] += weight * _positive(result["baseline_us"], "baseline shape time")
        load["candidate_us"] += weight * _positive(result["candidate_us"], "candidate shape time")
        load["cases"].append(definition["id"])
    for load in loads.values():
        load["speedup"] = load["baseline_us"] / load["candidate_us"]
    return {
        "speedup": geometric_mean(l["speedup"] for l in loads.values()),
        "loads": dict(loads),
        "aggregation": "weighted sum within load; geometric mean across loads",
    }


def bootstrap_score(case_results: list[dict], case_definitions: list[dict], *,
                    samples: int = 2000, seed: int = 0, interval_floor: float = 0.015) -> dict:
    """Independent per-shape pair bootstrap; explicit local uncertainty method."""
    if samples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not 0 <= interval_floor < 1:
        raise ValueError("interval_floor must be in [0, 1)")
    point = aggregate_cases(case_results, case_definitions)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = []
        for case in case_results:
            ratios = case["paired_ratios"]
            if len(ratios) < 2:
                raise ValueError("at least two swapped-order pairs are required")
            speedup = statistics.median(rng.choices(ratios, k=len(ratios)))
            resampled.append({"id": case["id"], "baseline_us": case["baseline_us"],
                              "candidate_us": case["baseline_us"] / speedup})
        estimates.append(aggregate_cases(resampled, case_definitions)["speedup"])
    point["confidence_interval_95"] = [
        min(quantile(estimates, .025), point["speedup"] * (1 - interval_floor)),
        max(quantile(estimates, .975), point["speedup"] * (1 + interval_floor)),
    ]
    point["uncertainty"] = {
        "method": "independent per-shape swapped-pair percentile bootstrap",
        "bootstrap_samples": samples,
        "relative_interval_floor": interval_floor,
        "official_bootstrap_algorithm_known": False,
        "overhead_calibration_uncertainty_included": False,
    }
    return point
