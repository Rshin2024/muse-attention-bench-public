import math
import unittest

from muse_attention_bench.scoring import aggregate_cases, bootstrap_score, paired_ratios, summarize_shape


class ScoringTests(unittest.TestCase):
    def test_swapped_order_cancels_multiplicative_order_effect(self):
        # A/B has ratio 4; B/A has ratio 1. True paired speedup is sqrt(4)=2.
        samples = [dict(order="AB", baseline_us=24, candidate_us=9),
                   dict(order="BA", baseline_us=14, candidate_us=14)] * 2
        self.assertEqual(paired_ratios(samples, 4), [2, 2])
        result = summarize_shape(samples, 4)
        self.assertEqual(result["speedup"], 2)
        self.assertEqual(result["baseline_us"], 15)
        self.assertEqual(result["candidate_us"], 7.5)

    def test_refuse_nonpositive_corrected_time(self):
        samples = [dict(order=o, baseline_us=8, candidate_us=4) for o in ["AB", "BA"] * 2]
        with self.assertRaises(ValueError):
            paired_ratios(samples, 4)

    def test_refuse_unbalanced_orders(self):
        samples = [dict(order="AB", baseline_us=8, candidate_us=4)] * 4
        with self.assertRaises(ValueError):
            paired_ratios(samples, 0)

    def test_geomean_loads_not_global_weighted_ratio(self):
        definitions = [dict(id="a", load="small", weight=46),
                       dict(id="b", load="small", weight=16),
                       dict(id="c", load="large", weight=46),
                       dict(id="d", load="large", weight=16)]
        results = [dict(id="a", baseline_us=1, candidate_us=.5),
                   dict(id="b", baseline_us=1, candidate_us=.5),
                   dict(id="c", baseline_us=100, candidate_us=100),
                   dict(id="d", baseline_us=100, candidate_us=100)]
        score = aggregate_cases(results, definitions)
        self.assertAlmostEqual(score["speedup"], math.sqrt(2))
        self.assertEqual(score["loads"]["small"]["baseline_us"], 62)
        self.assertEqual(score["loads"]["small"]["candidate_us"], 31)

    def test_missing_duplicate_or_extra_shape_invalidates_score(self):
        defs = [dict(id="a", load="load", weight=1)]
        for results in [[], [dict(id="b", baseline_us=1, candidate_us=1)],
                        [dict(id="a", baseline_us=1, candidate_us=1)] * 2]:
            with self.assertRaises(ValueError):
                aggregate_cases(results, defs)

    def test_bootstrap_floor_and_reproducibility(self):
        defs = [dict(id="a", load="load", weight=46)]
        results = [dict(id="a", baseline_us=2, candidate_us=1, paired_ratios=[2, 2, 2])]
        first = bootstrap_score(results, defs, samples=100, seed=3)
        self.assertEqual(first, bootstrap_score(results, defs, samples=100, seed=3))
        self.assertAlmostEqual(first["confidence_interval_95"][0], 1.97)
        self.assertAlmostEqual(first["confidence_interval_95"][1], 2.03)
        self.assertFalse(first["uncertainty"]["official_bootstrap_algorithm_known"])

    def test_nonfinite_and_negative_values_rejected(self):
        defs = [dict(id="a", load="load", weight=1)]
        for invalid in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                aggregate_cases([dict(id="a", baseline_us=invalid, candidate_us=1)], defs)


if __name__ == "__main__":
    unittest.main()
