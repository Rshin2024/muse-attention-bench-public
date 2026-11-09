"""Saved-data checks catch altered measurements, accounting and outcome credit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce_results


class PublicResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'data'
        shutil.copytree(ROOT / 'data', self.data)

    def change_results_and_refresh_outer_digest(self, change):
        path = self.data / 'results.json'
        value = json.loads(path.read_text())
        change(value)
        body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        path.write_bytes(body)
        manifest_path = self.data / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        record = next(row for row in manifest['files'] if row['path'] == 'results.json')
        record.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def test_all_observed_reports_reconstruct(self):
        result = reproduce_results.verify_dataset(self.data)
        self.assertEqual((result['model_results'], result['baseline_controls']), (7, 2))
        self.assertEqual((result['authored_optimization_artifacts'], result['unchanged_starter_outcomes']), (4, 3))
        self.assertEqual((result['replay_brackets'], result['swapped_pairs'], result['calibration_samples']), (2160, 1080, 900))
        self.assertFalse(result['candidate_or_task_code_executed'])
        self.assertFalse(result['private_provenance_gate_replayed'])

    def test_changed_raw_observation_fails_even_with_new_outer_hash(self):
        def change(data):
            data['models'][0]['result']['scored_cases'][0]['samples'][0]['candidate_us'] *= 2
        self.change_results_and_refresh_outer_digest(change)
        with self.assertRaisesRegex(ValueError, 'reconstructed timing evidence|differs'):
            reproduce_results.verify_dataset(self.data)

    def test_starter_cannot_receive_model_performance_credit(self):
        def change(data):
            row = next(x for x in data['models'] if x['delivery']['inherited_starter'])
            row['delivery']['performance_point_eligible'] = True
        self.change_results_and_refresh_outer_digest(change)
        with self.assertRaisesRegex(ValueError, 'credited'):
            reproduce_results.verify_dataset(self.data)

    def test_cost_total_must_reconcile_with_selected_attempts(self):
        def change(data):
            data['costs']['selected_attempt_known_API_including_readiness_usd'] += 1
        self.change_results_and_refresh_outer_digest(change)
        with self.assertRaisesRegex(ValueError, 'Selected API total'):
            reproduce_results.verify_dataset(self.data)

    def test_added_runtime_metadata_is_outside_projection(self):
        def change(data):
            data['models'][0]['result']['runtime'] = {'label': 'extra_metadata'}
        self.change_results_and_refresh_outer_digest(change)
        with self.assertRaisesRegex(ValueError, 'outside the public projection'):
            reproduce_results.verify_dataset(self.data)

    def test_file_reference_cannot_escape_dataset(self):
        path = self.data / 'manifest.json'
        value = json.loads(path.read_text())
        value['files'][0]['path'] = '../outside.json'
        path.write_text(json.dumps(value) + '\n')
        with self.assertRaisesRegex(ValueError, 'coverage|allowlisted'):
            reproduce_results.verify_dataset(self.data)

    def test_hash_change_is_detected_before_math_replay(self):
        path = self.data / 'results.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'digest or size'):
            reproduce_results.verify_dataset(self.data)


if __name__ == '__main__':
    unittest.main()
