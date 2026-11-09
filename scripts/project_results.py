#!/usr/bin/env python3
"""Project operator-normalized results into a metadata-free scientific dataset.

The input is deliberately not distributed. This allowlist never copies source
bodies, raw errors, runtime identities, requests, paths, or resource receipts.
It performs no model, cloud, GPU, or candidate-code operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


MODEL_NAMES = {
    'gpt-6-astra': 'GPT-6 Astra',
    'claude-opus-5-5': 'Claude Opus 5.5',
    'gpt-6.1-sol': 'GPT-6.1 Sol',
    'glm-5.3': 'GLM 5.3',
    'claude-fable-5-1': 'Claude Fable 5.1',
    'gemini-4-argon': 'Gemini 4 Argon',
    'deepseek-v4-pro': 'DeepSeek V4 Pro',
}
LOADS = ('decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32')
CHECK = re.compile(
    r"relative L2 error [0-9.eE+-]+ \(worst row [0-9.eE+-]+\) "
    r"\[bound [0-9.eE+-]+: 2\.5 x the reference's [0-9.eE+-]+ "
    r"with FP8 E4M3 queries \+ 0\.015\]"
)
CASE_METRICS = (
    'speedup', 'baseline_us', 'candidate_us', 'candidate_event_median_us',
    'baseline_event_median_us', 'fixed_graph_cost_us',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value, *, zero=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if zero else value > 0), 'Invalid numeric fact')
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value),
            'Invalid SHA256 commitment')
    return value


def oracle_check(value):
    require(isinstance(value, str) and CHECK.fullmatch(value),
            'Only numeric oracle observations may be projected')
    return value


def project_result(raw):
    """Keep exact scalar observations while excluding all run identities."""
    require(raw['schema_version'] == 1 and raw['benchmark'] == 'muse_attention_decode',
            'Unexpected result format')
    require(raw['evaluation_kind'] == 'documented-protocol-reconstruction'
            and raw['official_score'] is False and raw['gpu_validated'] is True
            and raw['status'] == 'passed', 'Only the completed observed results are supported')
    mandatory = []
    for i, case in enumerate(raw['correctness_cases'], 1):
        require(case['id'] == f'test-{i:02d}' and case['passed'] is True,
                'Mandatory correctness order or observation differs')
        mandatory.append({'id': case['id'], 'passed': True,
                          'eager': oracle_check(case['eager']),
                          'changed_input_graph': oracle_check(case['changed_input_graph'])})
    require(len(mandatory) == 16, 'Sixteen mandatory observations are required')
    scored = []
    for i, case in enumerate(raw['scored_cases'], 1):
        require(case['id'] == f'scored-{i:02d}' and case['passed'] is True,
                'Scored observation order differs')
        projected = {'id': case['id'], 'passed': True}
        projected.update({k: number(case[k], zero=k == 'fixed_graph_cost_us') for k in CASE_METRICS})
        projected['paired_ratios'] = [number(x) for x in case['paired_ratios']]
        samples = []
        for sample in case['samples']:
            require(sample['order'] in ('AB', 'BA'), 'Unknown replay order')
            samples.append({'order': sample['order'],
                            'baseline_us': number(sample['baseline_us']),
                            'candidate_us': number(sample['candidate_us']),
                            'candidate_check': oracle_check(sample['candidate_check']),
                            'baseline_check': oracle_check(sample['baseline_check'])})
        require(len(samples) == 20 and len(projected['paired_ratios']) == 10,
                'Twenty replay brackets and ten swapped pairs are required')
        projected['samples'] = samples
        scored.append(projected)
    require(len(scored) == 12, 'Twelve scored observations are required')
    calibration = raw['graph_cost_calibration']
    require(calibration['method'] == 'empty CUDA graph, identical cold-L2/event bracket'
            and calibration['official_calibration_algorithm_known'] is False,
            'Unknown calibration semantics')
    samples = [number(v, zero=True) for v in calibration['samples_us']]
    require(len(samples) == 100, 'One hundred calibration observations are required')
    score = raw['score']
    require(set(score['loads']) == set(LOADS), 'Six weighted loads are required')
    loads = {}
    for name in LOADS:
        value = score['loads'][name]
        require(all(re.fullmatch(r'scored-\d{2}', x) for x in value['cases']), 'Unknown case label')
        loads[name] = {k: number(value[k]) for k in ('baseline_us', 'candidate_us', 'speedup')}
        loads[name]['cases'] = list(value['cases'])
    uncertainty = score['uncertainty']
    require(score['aggregation'] == 'weighted sum within load; geometric mean across loads'
            and uncertainty['method'] == 'independent per-shape swapped-pair percentile bootstrap'
            and uncertainty['bootstrap_samples'] == 2000
            and uncertainty['relative_interval_floor'] == .015
            and uncertainty['official_bootstrap_algorithm_known'] is False
            and uncertainty['overhead_calibration_uncertainty_included'] is False,
            'Unknown score uncertainty semantics')
    require(type(raw['evaluation_seed']) is int, 'Evaluation seed must be an integer')
    return {
        'schema_version': 1, 'benchmark': 'muse_attention_decode',
        'evaluation_kind': 'documented-protocol-reconstruction',
        'official_score': False, 'gpu_validated': True, 'status': 'passed',
        'candidate_sha256': digest(raw['candidate_sha256']),
        'protocol_sha256': digest(raw['protocol_sha256']),
        'speedup': number(raw['speedup']), 'correctness_cases': mandatory,
        'scored_cases': scored, 'evaluation_seed': raw['evaluation_seed'],
        'graph_cost_calibration': {
            'method': 'empty CUDA graph, identical cold-L2/event bracket',
            'median_us': number(calibration['median_us'], zero=True),
            'samples_us': samples, 'official_calibration_algorithm_known': False,
        },
        'score': {
            'speedup': number(score['speedup']), 'loads': loads,
            'aggregation': 'weighted sum within load; geometric mean across loads',
            'confidence_interval_95': [number(v) for v in score['confidence_interval_95']],
            'uncertainty': {
                'method': 'independent per-shape swapped-pair percentile bootstrap',
                'bootstrap_samples': 2000, 'relative_interval_floor': .015,
                'official_bootstrap_algorithm_known': False,
                'overhead_calibration_uncertainty_included': False,
            },
        },
        'elapsed_seconds': number(raw['elapsed_seconds']),
    }


def project_profile(raw):
    """Publish numeric protocol facts, without copied task statement prose."""
    result = {'schema_version': 1, 'benchmark': 'muse_attention_decode',
              'seed': raw['seed'], 'timing': {}, 'test_cases': [], 'scored_cases': []}
    for key in ('repeats', 'warmup', 'flush_bytes', 'bootstrap_samples',
                'calibration_repeats', 'confidence_level', 'minimum_relative_interval',
                'cold_l2_each_replay', 'run_timeout_seconds'):
        value = raw['timing'][key]
        require(type(value) in (int, float, bool), 'Profile facts must be numeric or boolean')
        result['timing'][key] = value
    for source, destination, expected in (('test_cases', 'test_cases', 16),
                                           ('scored_cases', 'scored_cases', 12)):
        for i, case in enumerate(raw[source], 1):
            require(case['id'] == (f'test-{i:02d}' if source == 'test_cases' else f'scored-{i:02d}'),
                    'Shape labels must be generic ordered identifiers')
            kwargs = case['kwargs']
            require(set(kwargs) <= {'batch', 'ctx', 'heads', 'layer', 'page', 'dist', 'q_len', 'spread', 'kv_scale'},
                    'Unexpected shape fact')
            require(kwargs.get('heads') in ('tp8', 'dp') and kwargs.get('layer') in ('sliding', 'full'),
                    'Unexpected head/layer fact')
            require(kwargs.get('dist', 'soft') in ('soft', 'coherent', 'edges'), 'Unexpected input distribution')
            require(kwargs.get('spread', 'mixed') in ('mixed', 'agentic'), 'Unexpected length distribution')
            if 'kv_scale' in kwargs: number(kwargs['kv_scale'])
            for key in ('batch', 'ctx', 'page', 'q_len'):
                if key in kwargs: require(type(kwargs[key]) is int and kwargs[key] > 0, 'Invalid shape dimension')
            require(case['load'] is None or case['load'] in LOADS, 'Unknown load label')
            result[destination].append({'id': case['id'], 'load': case['load'],
                                        'weight': number(case['weight'], zero=True),
                                        'kwargs': dict(kwargs)})
        require(len(result[destination]) == expected, 'Shape fact coverage differs')
    return result


def project_document(source):
    require(source.get('completed') is True and len(source['rows']) == 7,
            'The private normalized study must be complete')
    profile = project_profile(source['profile'])
    models = []
    for row in source['rows']:
        raw, review = row['raw'], row['review']
        model = row['model']
        require(model in MODEL_NAMES, 'Unexpected model identifier')
        delivered = row['delivery']['model_submission_delivered'] is True
        require(row['artifact_eligible'] is delivered, 'Authored artifact classification differs')
        require(raw['research_status'] in ('completed_voluntarily', 'failed_provider'),
                'Unexpected research outcome')
        require(raw['recorded_within_budget'] is True, 'Selected attempt was not recorded within budget')
        if delivered:
            require(not raw['errors'], 'A voluntarily completed attempt has an unexpected error')
            stop = 'completed_voluntarily'
        else:
            require(len(raw['errors']) == 1, 'Failed attempt error coverage differs')
            stop = raw['errors'][0].get('kind')
            require(stop in ('request_timeout', 'request_process_deadline'),
                    'Only allowlisted error kinds may be projected')
        models.append({
            'id': model, 'model': MODEL_NAMES[model],
            'delivery': {'classification': 'authored_kernel' if delivered else 'unchanged_starter',
                         'model_submission_delivered': delivered,
                         'optimization_delivered': delivered,
                         'inherited_starter': not delivered,
                         'performance_point_eligible': delivered},
            'source': {'candidate_sha256': digest(row['seal']['candidate_sha256']),
                       'body_included': False},
            'source_review': {
                'current_kernel_rules_approved': review['current_rules_approved'] is True,
                'strict_frozen_TASK_precision_compliant': review['frozen_task_precision_compliant'] is True,
                'frozen_TASK_other_rules_approved': review['frozen_task_other_rules_approved'] is True,
                'static_review_only': True, 'public_source_review_replayable': False,
            },
            'attempt': {
                'status': raw['research_status'], 'stop_reason': stop,
                'research_seconds': number(raw['research_elapsed_seconds']),
                'budget_seconds': 1800, 'actions': raw['actions'], 'diagnostics': raw['diagnostics'],
                'recorded_within_budget': True,
                'authoring_group': 'recovery' if model == 'deepseek-v4-pro' else 'initial',
            },
            'api_cost': {
                'known_total_including_readiness_usd': number(raw['api_cost_estimate_usd'], zero=True),
                'known_research_usd': number(raw['research_api_cost_estimate_usd'], zero=True),
                'readiness_usd': number(raw['preflight_api_cost_estimate_usd'], zero=True),
                'unresolved_reserve_not_confirmed_charge_usd': number(raw['api_cost_unknown_reserve_usd'], zero=True),
            },
            'compute_attribution': {
                'research_estimate_usd': number(row['research_compute_estimate_usd'], zero=True),
                'original_validation_estimate_usd': number(row['original_validation_compute_estimate_usd'], zero=True),
                'common_final_validation_estimate_usd': number(row['shared_A_validation_compute_estimate_usd'], zero=True),
            },
            'result': project_result(row['primary_result']),
        })
    require({m['id'] for m in models} == set(MODEL_NAMES), 'Model coverage differs')
    runtime = source['boards']['a']['runtime']
    software = {key: runtime[key] for key in ('torch_version', 'cuda_version', 'driver_version')}
    require(all(isinstance(v, str) and re.fullmatch(r'[0-9.+A-Za-z_-]+', v)
                for v in software.values()), 'Software versions contain unexpected metadata')
    component_names = (
        'historical_accelerator_engineering', 'historical_cpu_storage_auxiliaries',
        'cache_storage', 'pre_study_runtime_setup', 'initial_research_runtime',
        'all_native_api_requests', 'common_verification_and_recovery_runtime',
        'trusted_controller_runtime_and_storage',
    )
    components = source['cost_accounting']['additive_known_components']
    require(len(components) == len(component_names), 'Whole-study cost component coverage differs')
    known = number(source['cost_accounting']['whole_known_estimate_usd'], zero=True)
    selected_api = sum(m['api_cost']['known_total_including_readiness_usd'] for m in models)
    costs = {
        'currency': 'USD', 'selected_attempt_known_API_including_readiness_usd': selected_api,
        'selected_attempt_unresolved_reserve_not_confirmed_charge_usd': sum(
            m['api_cost']['unresolved_reserve_not_confirmed_charge_usd'] for m in models),
        'whole_project_known_public_tariff_estimate_usd': known,
        'whole_project_native_API_known_counted_once_usd': number(
            source['cost_accounting']['global_native_API_known_usd'], zero=True),
        'whole_project_unresolved_reserve_not_confirmed_charge_usd': number(
            source['cost_accounting']['all_unknown_native_request_reserves_usd'], zero=True),
        'additive_known_components': [{'id': name, 'known_estimate_usd': number(row['known_estimate_usd'], zero=True)}
                                      for name, row in zip(component_names, components)],
        'invoice_verified': False, 'unpriced_fees_assumed_zero': False,
        'phase_attribution_added_to_lifecycle_total_again': False,
        'scope': 'Whole-project estimate includes earlier engineering; selected-attempt attribution is a subset.',
    }
    result = {
        'schema_version': 1, 'kind': 'metadata-free-scientific-projection',
        'study': {
            'benchmark': 'muse_attention_decode', 'research_cap_seconds_per_model': 1800,
            'selected_source_authoring_opportunities_per_model': 1,
            'primary_selection': 'First full measurement of each frozen artifact on one common final GPU.',
            'mandatory_cases_per_artifact': 16, 'scored_cases_per_artifact': 12,
            'hardware': {'gpu_model': 'NVIDIA H200 SXM', 'gpu_count_per_evaluator': 1},
            'software': software,
            'authoring_condition_delta': {
                'identical_full_source_authoring_conditions': False,
                'final_numerical_and_timing_conditions_matched': True,
                'retained_initial_sources': 6, 'fresh_recovery_opportunities': 1,
                'unchanged_native_research_contract': True,
                'recovery_cpu_outer_cleanup_margin_seconds': 20,
                'initial_cpu_late_tail_reserve_seconds': 45,
                'recovery_cpu_late_tail_reserve_seconds': 70,
                'maximum_late_cpu_allowance_reduction_seconds': 25,
            },
        },
        'models': models,
        'controls': {
            'final': {'label': 'Common final GPU baseline control', 'result': project_result(source['boards']['a']['control'])},
            'diagnostic': {'label': 'Independent diagnostic GPU baseline control', 'result': project_result(source['boards']['b']['control'])},
        },
        'costs': costs,
        'limitations': [
            'This is a derived scientific projection, not a byte-for-byte raw producer archive.',
            'Saved scalar replay checks arithmetic and coverage; it does not rerun a GPU, oracle, model, or cloud provenance gate.',
            'Four models delivered authored kernels; three retained the supplied starter and receive no optimization performance point.',
            'Static source-review decisions are reported facts; candidate and third-party task bodies are not distributed or publicly re-reviewed.',
            'Current kernel-rule source eligibility differs from the older frozen TASK precision wording and stricter serving requirements.',
            'One selected attempt per model and the disclosed authoring recovery do not establish a general ranking of model ability.',
            'Local calibration, interval estimates, isolation and anti-cheat are not certified equivalent to the hosted judge.',
            'Known tariff estimates are not invoices; unresolved reserves are not confirmed charges; unpriced fees and credits remain unobserved.',
        ],
    }
    return result, profile


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write_projection(source_path, destination):
    source_path, destination = Path(source_path), Path(destination)
    body = source_path.read_bytes()
    source = json.loads(body)
    results, profile = project_document(source)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for name, value in (('results.json', results), ('profile.json', profile)):
        content = canonical(value)
        (destination / name).write_bytes(content)
        files.append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    manifest = {
        'schema_version': 1, 'kind': 'public-scientific-projection-manifest',
        'projection_version': 1, 'input_document_sha256_commitment': hashlib.sha256(body).hexdigest(),
        'original_input_paths_or_identifiers_included': False,
        'raw_producer_bytes_preserved_claimed': False,
        'source_bodies_included': False, 'private_provenance_gate_publicly_replayed': False,
        'included_record_counts': {'model_results': 7, 'baseline_controls': 2,
                                   'mandatory_observations': 144, 'scored_shapes': 108,
                                   'replay_brackets': 2160, 'swapped_pairs': 1080,
                                   'calibration_samples': 900},
        'field_policy': {
            'included': ['numeric correctness observations', 'calibration samples', 'paired replay samples',
                         'shape/load statistics', 'scores and intervals', 'protocol shape facts',
                         'source hash commitments and factual review flags', 'attempt durations and counts',
                         'known cost estimates and unresolved reserves', 'nonunique hardware and software versions'],
            'excluded': ['candidate and task source bodies', 'native provider conversations and states',
                         'credentials and authentication material', 'personal identities and local paths',
                         'cloud account, resource, image, host and device identifiers',
                         'provider request identifiers', 'raw errors and command output',
                         'resource-operation and IAM records', 'exact resource chronology',
                         'original raw provenance graph and private publication gate inputs'],
        },
        'files': files,
    }
    (destination / 'manifest.json').write_bytes(canonical(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Operator-local normalized study JSON; never distributed')
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    manifest = write_projection(args.input, args.output)
    print(json.dumps({'projected_files': len(manifest['files']),
                      'record_counts': manifest['included_record_counts'],
                      'cloud_or_model_calls': False, 'candidate_execution': False}, indent=2))


if __name__ == '__main__':
    main()
