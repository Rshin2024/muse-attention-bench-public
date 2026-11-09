#!/usr/bin/env python3
"""Verify the public projection and reconstruct its nine saved scalar reports.

This command needs only Python's standard library. It makes no network, model,
cloud or GPU call, and never imports candidate or third-party task code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from project_results import MODEL_NAMES, project_profile, project_result, require
from result_numerics import reconstruct_report


ROOT = Path(__file__).resolve().parents[1]
ROW_KEYS = {'id', 'model', 'delivery', 'source', 'source_review', 'attempt', 'api_cost',
            'compute_attribution', 'result'}
TOP_KEYS = {'schema_version', 'kind', 'study', 'models', 'controls', 'costs', 'limitations'}


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-12),
            label + ' does not reconcile')


def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + ' fields differ')


def verified_json(root, record):
    name = record.get('path')
    require(isinstance(name, str) and name in ('results.json', 'profile.json'),
            'Public dataset reference is not allowlisted')
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root),
            'Public dataset reference escapes its root')
    body = path.read_bytes()
    require(type(record.get('bytes')) is int and len(body) == record['bytes']
            and hashlib.sha256(body).hexdigest() == record.get('sha256'),
            'Public dataset digest or size differs')
    return json.loads(body)


def verify_dataset(root=None):
    root = Path(root or ROOT / 'data').resolve(strict=True)
    require(root.is_dir(), 'Dataset directory is required')
    manifest_path = root / 'manifest.json'
    require(not manifest_path.is_symlink(), 'Dataset manifest cannot be a symlink')
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get('schema_version') == 1
            and manifest.get('kind') == 'public-scientific-projection-manifest'
            and manifest.get('projection_version') == 1,
            'Unsupported public projection manifest')
    require(manifest.get('original_input_paths_or_identifiers_included') is False
            and manifest.get('raw_producer_bytes_preserved_claimed') is False
            and manifest.get('source_bodies_included') is False
            and manifest.get('private_provenance_gate_publicly_replayed') is False,
            'Projection qualification differs')
    require(re.fullmatch(r'[0-9a-f]{64}', manifest.get('input_document_sha256_commitment', '')),
            'Original input content commitment is invalid')
    records = manifest.get('files', [])
    require(len(records) == 2 and {r.get('path') for r in records} == {'results.json', 'profile.json'},
            'Public projection file coverage differs')
    values = {record['path']: verified_json(root, record) for record in records}
    data, profile = values['results.json'], values['profile.json']
    exact_keys(data, TOP_KEYS, 'Scientific dataset')
    require(data['schema_version'] == 1 and data['kind'] == 'metadata-free-scientific-projection',
            'Unsupported scientific dataset')
    require(project_profile(profile) == profile, 'Protocol fact projection has extra or changed fields')
    exact_keys(data['controls'], ('final', 'diagnostic'), 'Control roster')
    rows = data['models']
    require(len(rows) == 7 and {row.get('id') for row in rows} == set(MODEL_NAMES),
            'The seven model outcomes are not complete and unique')
    public_results = []
    authored, starters = [], []
    for row in rows:
        exact_keys(row, ROW_KEYS, 'Model row')
        require(row['model'] == MODEL_NAMES[row['id']], 'Public model label differs')
        exact_keys(row['delivery'], ('classification', 'model_submission_delivered', 'optimization_delivered',
                                    'inherited_starter', 'performance_point_eligible'), 'Delivery')
        delivery = row['delivery']
        is_authored = delivery['classification'] == 'authored_kernel'
        require(delivery['classification'] in ('authored_kernel', 'unchanged_starter')
                and delivery['model_submission_delivered'] is is_authored
                and delivery['optimization_delivered'] is is_authored
                and delivery['performance_point_eligible'] is is_authored
                and delivery['inherited_starter'] is (not is_authored),
                'Inherited starter is incorrectly credited as a model optimization')
        (authored if is_authored else starters).append(row)
        exact_keys(row['source'], ('candidate_sha256', 'body_included'), 'Source commitment')
        require(row['source']['body_included'] is False
                and row['source']['candidate_sha256'] == row['result']['candidate_sha256'],
                'Source commitment binding differs')
        exact_keys(row['source_review'], ('current_kernel_rules_approved',
                   'strict_frozen_TASK_precision_compliant', 'frozen_TASK_other_rules_approved',
                   'static_review_only', 'public_source_review_replayable'), 'Source review')
        review = row['source_review']
        require(review['current_kernel_rules_approved'] is True
                and review['strict_frozen_TASK_precision_compliant'] is False
                and review['frozen_TASK_other_rules_approved'] is True
                and review['static_review_only'] is True
                and review['public_source_review_replayable'] is False,
                'Reported source-review qualification differs')
        exact_keys(row['attempt'], ('status', 'stop_reason', 'research_seconds', 'budget_seconds',
                                   'actions', 'diagnostics', 'recorded_within_budget', 'authoring_group'), 'Attempt')
        attempt = row['attempt']
        require(attempt['status'] == ('completed_voluntarily' if is_authored else 'failed_provider')
                and attempt['stop_reason'] in ('completed_voluntarily', 'request_timeout', 'request_process_deadline')
                and type(attempt['research_seconds']) in (int, float)
                and math.isfinite(attempt['research_seconds']) and 0 < attempt['research_seconds'] <= 1800
                and attempt['budget_seconds'] == 1800 and attempt['recorded_within_budget'] is True
                and attempt['authoring_group'] == ('recovery' if row['id'] == 'deepseek-v4-pro' else 'initial')
                and type(attempt['actions']) is int and attempt['actions'] >= 0
                and type(attempt['diagnostics']) is int and 0 <= attempt['diagnostics'] <= 3,
                'Attempt budget or outcome qualification differs')
        exact_keys(row['api_cost'], ('known_total_including_readiness_usd', 'known_research_usd',
                                    'readiness_usd', 'unresolved_reserve_not_confirmed_charge_usd'), 'API cost')
        close(row['api_cost']['known_total_including_readiness_usd'],
              row['api_cost']['known_research_usd'] + row['api_cost']['readiness_usd'], 'Selected API cost')
        require(row['api_cost']['unresolved_reserve_not_confirmed_charge_usd'] == (0 if is_authored else 30),
                'Unresolved usage is missing or credited as a confirmed charge')
        exact_keys(row['compute_attribution'], ('research_estimate_usd', 'original_validation_estimate_usd',
                                              'common_final_validation_estimate_usd'), 'Compute attribution')
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                    for v in row['compute_attribution'].values()), 'Invalid compute estimate')
        require(project_result(row['result']) == row['result'],
                'Scalar result contains fields outside the public projection')
        reconstructed = reconstruct_report(row['result'], profile)
        public_results.append({'model': row['model'],
                               'authored_performance_point': is_authored,
                               **reconstructed})
    require(len(authored) == 4 and len(starters) == 3,
            'Four authored outcomes and three unchanged starters are required')
    require(len({row['source']['candidate_sha256'] for row in authored}) == 4
            and len({row['source']['candidate_sha256'] for row in starters}) == 1,
            'Authored/starter source commitment coverage differs')
    controls = {}
    for key, value in data['controls'].items():
        exact_keys(value, ('label', 'result'), 'Control')
        require(value['label'] == ('Common final GPU baseline control' if key == 'final'
                                  else 'Independent diagnostic GPU baseline control'), 'Control label differs')
        require(project_result(value['result']) == value['result'], 'Control projection fields differ')
        controls[key] = reconstruct_report(value['result'], profile)
    costs = data['costs']
    require(costs['currency'] == 'USD' and costs['invoice_verified'] is False
            and costs['unpriced_fees_assumed_zero'] is False
            and costs['phase_attribution_added_to_lifecycle_total_again'] is False,
            'Cost estimate qualification differs')
    components = costs['additive_known_components']
    require(len(components) == 8 and len({x['id'] for x in components}) == 8,
            'Cost components are not unique and complete')
    require(all(type(x['known_estimate_usd']) in (int, float)
                and math.isfinite(x['known_estimate_usd']) and x['known_estimate_usd'] >= 0
                for x in components), 'Invalid additive known estimate')
    close(costs['whole_project_known_public_tariff_estimate_usd'],
          sum(x['known_estimate_usd'] for x in components), 'Whole-project known estimate')
    close(costs['selected_attempt_known_API_including_readiness_usd'],
          sum(row['api_cost']['known_total_including_readiness_usd'] for row in rows), 'Selected API total')
    close(costs['selected_attempt_unresolved_reserve_not_confirmed_charge_usd'],
          sum(row['api_cost']['unresolved_reserve_not_confirmed_charge_usd'] for row in rows), 'Selected unresolved reserve')
    close(costs['selected_attempt_known_API_including_readiness_usd'], 14.228959724, 'Observed selected API total')
    close(costs['whole_project_known_public_tariff_estimate_usd'], 300.25769917850084, 'Observed whole-project known estimate')
    close(costs['whole_project_native_API_known_counted_once_usd'], 37.023377544, 'Observed historical-inclusive API estimate')
    require(costs['selected_attempt_unresolved_reserve_not_confirmed_charge_usd'] == 90
            and costs['whole_project_unresolved_reserve_not_confirmed_charge_usd'] == 392,
            'Captured unresolved usage reserves differ')
    expected_counts = {'model_results': 7, 'baseline_controls': 2, 'mandatory_observations': 144,
                       'scored_shapes': 108, 'replay_brackets': 2160, 'swapped_pairs': 1080,
                       'calibration_samples': 900}
    require(manifest['included_record_counts'] == expected_counts, 'Projection count declarations differ')
    return {'status': 'PASS', 'verified_data_files': 2, **expected_counts,
            'authored_optimization_artifacts': 4, 'unchanged_starter_outcomes': 3,
            'results': public_results, 'controls': controls,
            'selected_attempt_known_API_including_readiness_usd': costs['selected_attempt_known_API_including_readiness_usd'],
            'whole_project_known_public_tariff_estimate_usd': costs['whole_project_known_public_tariff_estimate_usd'],
            'whole_project_unresolved_reserve_not_confirmed_charge_usd': 392,
            'source_bodies_publicly_re_reviewed': False,
            'private_provenance_gate_replayed': False,
            'GPU_or_model_or_network_calls': False, 'candidate_or_task_code_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    try:
        print(json.dumps(verify_dataset(args.data), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, 'Public result verification failed: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
