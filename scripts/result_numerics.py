"""Reconstruct saved timing evidence on CPU; never import a candidate or CUDA."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import statistics
import types

# This is the pure scoring source included in the immutable experiment image.
SCORING_SHA256 = '3e3f748300e8ba187bb755c1a7c9fc7bc1d5221b8cec9cfde55a855ab14c8a29'
SCORING_PATH = Path(__file__).resolve().parents[1] / 'src/muse_attention_bench/scoring.py'


def same(actual, expected, label):
    """Finite numeric comparison plus exact structural/identity checks."""
    if isinstance(expected, bool) or expected is None:
        if actual is not expected:
            raise ValueError(label + ' differs')
    elif isinstance(expected, (int, float)):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError(label + ' differs from reconstructed timing evidence')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + ' has different coverage')
        for i, value in enumerate(expected):
            same(actual[i], value, f'{label}[{i}]')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + ' has different fields')
        for key, value in expected.items():
            same(actual[key], value, label + '.' + str(key))
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(label + ' differs')


def _number(value, label, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(label + ' is not a finite valid timing value')
    return value


def trusted_scoring():
    """Execute only the hash-verified, Torch-free stdlib scoring definitions."""
    body = SCORING_PATH.read_bytes()
    if hashlib.sha256(body).hexdigest() != SCORING_SHA256:
        raise ValueError('Pure scoring source differs from the frozen experiment')
    module = types.ModuleType('_muse_frozen_report_scoring')
    exec(compile(body, str(SCORING_PATH), 'exec'), module.__dict__)
    return module


def reconstruct_report(result, profile):
    """Check calibration -> AB/BA pairs -> cases ->46/16loads -> score/CI.

    Reconstruction consumes only saved scalar data. It cannot establish that
    the events or oracle outcomes actually happened; source/runtime/controller
    receipts supply the separate evidence bindings for those claims.
    """
    if result.get('status') != 'passed':
        if result.get('speedup') is not None or result.get('score'):
            raise ValueError('A failed numerical result cannot retain a full timing score')
        return {'full_score_reconstructed': False, 'reason': 'failed numerical evaluation; no full score'}
    scoring = trusted_scoring()
    timing = profile['timing']
    expected_tests = [c['id'] for c in profile.get('test_cases', [])] or [f'test-{i:02d}' for i in range(1,17)]
    mandatory = result.get('correctness_cases', [])
    if (len(mandatory) != len(expected_tests) or {c.get('id') for c in mandatory} != set(expected_tests)
            or not all(c.get('passed') is True for c in mandatory)):
        raise ValueError('Passing report lacks complete mandatory correctness coverage')
    cases = result.get('scored_cases', [])
    definitions = profile['scored_cases']
    if [v.get('id') for v in cases] != [v['id'] for v in definitions]:
        raise ValueError('Saved shape order/coverage differs from the frozen scored profile')
    seed = result.get('evaluation_seed')
    if type(seed) is not int or (profile.get('seed') is not None and seed != profile['seed']):
        raise ValueError('Saved evaluation seed differs from the frozen score bootstrap')
    calibration = result.get('graph_cost_calibration', {})
    raw_calibration = calibration.get('samples_us')
    if not isinstance(raw_calibration, list) or len(raw_calibration) != timing.get('calibration_repeats', 50):
        raise ValueError('Saved graph-cost calibration sample count differs')
    for value in raw_calibration:
        _number(value, 'Graph-cost sample', zero=True)
    overhead = statistics.median(raw_calibration)
    same(calibration.get('median_us'), overhead, 'Graph-cost calibration median')
    rebuilt = []
    for case in cases:
        if case.get('passed') is not True:
            raise ValueError('A passing full score includes an unsuccessful shape')
        samples = case.get('samples')
        if not isinstance(samples, list) or len(samples) != timing.get('repeats', 20):
            raise ValueError('Saved AB/BA sample count differs from the frozen repetition policy')
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError('Saved replay sample is not an object')
            for key in ('baseline_us', 'candidate_us'):
                _number(sample.get(key), 'Raw replay ' + key)
            if not all(isinstance(sample.get(k), str) and sample[k] for k in ('baseline_check', 'candidate_check')):
                raise ValueError('Saved replay lacks baseline/candidate oracle-check observations')
        summary = scoring.summarize_shape(samples, overhead)
        for key in ('speedup', 'baseline_us', 'candidate_us', 'candidate_event_median_us',
                    'baseline_event_median_us', 'fixed_graph_cost_us', 'paired_ratios'):
            same(case.get(key), summary[key], case['id'] + '.' + key)
        rebuilt.append({'id': case['id'], **summary})
    # The frozen helper reconstructs each weighted load from case timings;
    # bootstrapping also independently re-aggregates every resampled shape.
    score = scoring.bootstrap_score(rebuilt, definitions,
        samples=timing.get('bootstrap_samples', 2000), seed=seed,
        interval_floor=timing.get('minimum_relative_interval', timing.get('interval_floor', .015)))
    same(result.get('score'), score, 'Saved weighted-load score/CI')
    same(result.get('speedup'), score['speedup'], 'Top-level score')
    return {'full_score_reconstructed': True, 'scoring_source_sha256': SCORING_SHA256,
            'calibration_samples': len(raw_calibration), 'scored_cases': len(rebuilt),
            'replays_per_case': timing.get('repeats', 20),
            'bootstrap_samples': timing.get('bootstrap_samples', 2000),
            'speedup': score['speedup'], 'confidence_interval_95': score['confidence_interval_95']}
