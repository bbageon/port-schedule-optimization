"""Contracts and saved evidence for the one-case vertical transfer diagnostic."""
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = ('NO_REALLOC', 'RL_TIME')
MAIN_SEED, SMOKE_SEED = 20_000_000, 99_194_001
CHECKPOINT_SHA = '50f2da2e8b3070dbba48273f0de5debdf5514fe96f0fa1f5a8f25b8a830cba99'


def digest(value):
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def validate_config(cfg, base_cfg, synthetic_spec):
    if cfg.get('schema') != 'yr317.vertical-comparison.v1':
        raise ValueError('Unknown vertical comparison schema')
    if type(cfg.get('seed')) is not int or cfg['seed'] != MAIN_SEED:
        raise ValueError('Only the original first seed 20000000 is authorized')
    if (cfg.get('arms') != list(ARMS) or not base_cfg.get('seeds')
            or base_cfg['seeds'][0] != MAIN_SEED):
        raise ValueError('Fixed policies or original first seed changed')
    if (base_cfg.get('checkpoint') != 'outputs/v3/month-02/ckpt_029.pt'
            or base_cfg.get('checkpoint_sha256') != CHECKPOINT_SHA):
        raise ValueError('The original frozen checkpoint is required')
    if digest(cfg.get('environment_spec')) != digest(synthetic_spec):
        raise ValueError('The current synthetic vertical specification must be preserved')
    for key, length in (('source_commit', 40), ('base_config_sha256', 64), ('prereg_sha256', 64)):
        if not isinstance(cfg.get(key), str) or not re.fullmatch('[0-9a-f]{%d}' % length, cfg[key]):
            raise ValueError(f'Missing or invalid frozen identity: {key}')
    if not all(isinstance(cfg.get(k), str) and cfg[k] for k in ('base_config', 'prereg')):
        raise ValueError('Explicit base configuration and new preregistration are required')


def claim_output(out, arm):
    if arm not in ARMS:
        raise ValueError('Only NO_REALLOC and RL_TIME are supported')
    folder = Path(out) / arm
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_rows(path, rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def recording_checks(result, *, expected_input, environment_spec_sha256,
                     weights_unchanged, checkpoint_unchanged, runtime_unchanged, rollout_count):
    """Incomplete work/deadline infeasibility are outcomes, never missing records."""
    req, vessel, flow = (result[k] for k in
                         ('request_summary', 'vessel_work_summary', 'container_flow_summary'))
    days = result['days']
    return dict(request_recording=req['recording_ok'] is True,
        announcer_counts=req['announcer_counts_match'] is True,
        no_truck_loss=req['all_requests_admitted'] is True,
        wait_cost_reconciles=math.isclose(req['accounted_wait_krw'],
            math.fsum(d['c_wait'] for d in days), rel_tol=1e-10, abs_tol=1e-4),
        vessel_recording=vessel['recording_ok'] is True,
        no_vessel_loss=vessel['unadmitted_moves'] == 0,
        container_identity_chain=flow['passed'] is True,
        physical_invariants_enabled=req['physical_invariants_enabled'] is True,
        no_policy_exceptions=result['policy_exceptions'] == 0,
        no_teacher=rollout_count == result['rollout_calls'] == 0,
        frozen_networks=weights_unchanged is True,
        checkpoint_unchanged=checkpoint_unchanged is True,
        runtime_unchanged=runtime_unchanged is True,
        no_spatial_actions=result['space'] == 0,
        canonical_input=result['environment_manifest']['canonical_input'] == expected_input
            and result['expected_input'] == expected_input,
        environment_identity=result['environment_spec_sha256'] == environment_spec_sha256,
        daily_observation_complete=len(days) == len(result['plan']) and bool(days) and all(
            d['operational'].get('day_index') == d['index'] and 'cohort' in d['operational']
            for d in days))


def save_result(folder, res, *, arm, seed, days, elapsed_s, identity, runtime,
                expected_input, stamp, rollout_count):
    """Preserve every MonthResult field without copying the large raw lists."""
    from independent_eval_checks import sha
    req_path, link_path = folder/'requests.jsonl.gz', folder/'container-links.jsonl.gz'
    write_rows(req_path, res.request_ledger)
    write_rows(link_path, res.container_links)
    finals = [d.as_dict() for d in res.days]
    write_rows(folder/'daily-final.jsonl', (dict(seed=seed, arm=arm, policy_label=arm, **d) for d in finals))
    request_identity = [{k: r[k] for k in ('job_id', 'requested_day', 'flow', 'requested_block',
        'requested_arrival_s', 'lead_s', 'requested_target', 'travel_s')} for r in res.request_ledger]
    result = dict(arm=arm, label=arm, seed=seed, elapsed_s=elapsed_s,
        plan=[asdict(d) for d in days], days=finals, live=[d.as_dict() for d in res.live],
        admitted=res.admitted, skipped=res.skipped, traded=res.traded_edges,
        space=res.n_space, time=res.n_time, txn_failed=res.txn_failed,
        decisions=res.decisions, retargeted=res.retargeted,
        policy_exceptions=res.policy_exceptions, rollout_calls=rollout_count,
        request_summary=res.request_summary, vessel_admissions=res.vessel_admissions,
        requested_identity_sha256=digest(request_identity), request_ledger_sha256=sha(req_path),
        container_links_sha256=sha(link_path), container_flow_summary=res.container_flow_summary,
        vessel_work_ledger=res.vessel_work_ledger, vessel_work_summary=res.vessel_work_summary,
        demand_bindings=res.demand_bindings, supply_plan_audit=res.supply_plan_audit,
        daily_observation=res.daily_observation, environment_manifest=res.environment_manifest,
        daily_artifacts={n: sha(folder/n) for n in ('operating-state.jsonl.gz', 'daily-final.jsonl')},
        expected_input=expected_input, runtime=runtime, repro=stamp, **identity,
        claim_eligible=False, new_training_runs=0,
        scope='Single predefined transfer case; no confidence intervals or layout causal claim')
    return result


def operational_outcomes(result):
    req, vessel = result['request_summary'], result['vessel_work_summary']
    deadline_rows = result['vessel_admissions']
    return dict(all_trucks_completed=req['states'].get('COMPLETED', 0) == req['requested'],
        truck_unfinished=req['requested'] - req['states'].get('COMPLETED', 0),
        unbound_at_cutoff=req['unbound_jobs_at_end'],
        vessel_unfinished=vessel['outstanding_yard_jobs'],
        all_vessels_completed=vessel['all_requested_work_completed'],
        structurally_late_vessel_streams=sum(r.get('structural_min_overrun_s', 0) > 0 for r in deadline_rows),
        maximum_structural_min_overrun_s=max((r.get('structural_min_overrun_s', 0) for r in deadline_rows), default=0),
        scope='Observed completion/capacity outcomes; separate from input and recording validity')


def pair_summary(results):
    """Validate one same-environment pair; report descriptive differences only."""
    if len(results) != 2 or {r.get('arm') for r in results} != set(ARMS):
        raise ValueError('Exactly one completed result per fixed policy is required')
    a, b = (next(r for r in results if r['arm'] == arm) for arm in ARMS)
    for key in ('seed', 'plan', 'expected_input', 'requested_identity_sha256',
                'environment_spec_sha256', 'pair_contract_sha256', 'runtime', 'source_commit',
                'config_sha256', 'base_config_sha256', 'prereg_sha256',
                'checkpoint_sha256', 'network_identities'):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'Unpaired vertical results: {key}')
    if a['seed'] not in (MAIN_SEED, SMOKE_SEED):
        raise ValueError('Unregistered main/diagnostic seed')
    if a['environment_manifest'] != b['environment_manifest']:
        raise ValueError('Runtime environment/input manifests differ')
    if any(not r.get('recording_checks') or not all(v is True for v in r['recording_checks'].values())
           or r['claim_eligible'] is not False or r['rollout_calls'] != 0
           or r.get('new_training_runs') != 0 for r in (a, b)):
        raise ValueError('Recording or frozen-policy guard failed')
    for r in (a, b):
        planned = [d['index'] for d in r['plan']]
        if (not planned or planned != list(range(len(planned)))
                or [d['index'] for d in r['days']] != planned):
            raise ValueError('Daily records are missing, duplicated or out of order')
        expected_count = 30 if r['seed'] == MAIN_SEED else 2
        if len(planned) != expected_count or any(
                d['train'] is not (0 < d['index'] < expected_count-1) for d in r['days']):
            raise ValueError('Run length or measurement-day selection changed')
        if any(not math.isfinite(d['phi_krw']) or d['phi_krw'] < 0 for d in r['days']):
            raise ValueError('Daily costs must be finite and nonnegative')
    scopes = {}
    for name, condition in (('measurement_days', lambda d: d['train']), ('all_days', lambda d: True)):
        ca, cb = (math.fsum(d['phi_krw'] for d in r['days'] if condition(d)) for r in (a, b))
        scopes[name] = dict(baseline_cost_krw=ca, time_only_cost_krw=cb, saving_krw=ca-cb,
                           saving_pct=(100*(ca-cb)/ca if ca > 0 else None))
    return dict(passed=True, seed=a['seed'], policies=list(ARMS), costs=scopes,
        operational_outcomes={r['arm']: operational_outcomes(r) for r in (a, b)},
        claim_eligible=False, statistical_inference=False,
        independent_sample_count=1 if a['seed'] == MAIN_SEED else 0,
        purpose='Single predefined transfer case; descriptive only, no H/V causal comparison')
