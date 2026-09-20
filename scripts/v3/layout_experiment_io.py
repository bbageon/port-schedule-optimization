"""Contracts and checks for layout-specific training + same-seed evaluation (YR-317-h2).

Two layout environments (legacy shared, vertical end-transfer) each train their own
time-only weights for the same short training seed, then each frozen model is
evaluated in its own environment on one shared evaluation seed against NO_REALLOC.
Descriptive only: one training seed, one evaluation seed, no inference.
"""
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'yr317.layout-training.v1'
ENVIRONMENTS = ('legacy', 'vertical')
ARMS = ('NO_REALLOC', 'RL_TIME')
TRAIN_ARM = 'RL_TIME'
DIAGNOSTIC_BAND = 9_900_000


def digest(value):
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def spec_digest(spec):
    """Environment identity: None for the legacy layout (matches the training manifest)."""
    return None if spec is None else digest(spec)


def band(seed):
    return (int(seed) // 100_000) * 100_000


def validate_config(cfg, synthetic_spec):
    from yard_rl.v3.eval.guards import check_bands
    if cfg.get('schema') != SCHEMA:
        raise ValueError('Unknown layout training schema')
    for key, length in (('source_commit', 40), ('prereg_sha256', 64)):
        if not isinstance(cfg.get(key), str) or not re.fullmatch('[0-9a-f]{%d}' % length, cfg[key]):
            raise ValueError(f'Missing or invalid frozen identity: {key}')
    if not isinstance(cfg.get('prereg'), str) or not cfg['prereg']:
        raise ValueError('Explicit preregistration path is required')
    contract = cfg.get('contract') or {}
    if (contract.get('admission_mode') != 'PRESERVE' or contract.get('supply_mode') != 'COUNT_BALANCED'
            or contract.get('train_arm') != TRAIN_ARM or contract.get('eval_arms') != list(ARMS)
            or contract.get('checkpoint_selection') != 'final_day'):
        raise ValueError('Contract must be PRESERVE/COUNT_BALANCED, train RL_TIME, eval both arms, final-day checkpoint')
    envs = cfg.get('environments') or {}
    if set(envs) != set(ENVIRONMENTS):
        raise ValueError('Exactly the legacy and vertical environments are required')
    if envs['legacy'].get('environment_spec') is not None:
        raise ValueError('The legacy environment has no layout specification')
    if digest(envs['vertical'].get('environment_spec')) != digest(synthetic_spec):
        raise ValueError('The current synthetic vertical specification must be preserved')
    for phase in ('smoke', 'main'):
        p = cfg.get(phase) or {}
        for key in ('train_seed', 'eval_seed', 'init_seed', 'labels_per_day'):
            if type(p.get(key)) is not int or p[key] <= 0:
                raise ValueError(f'{phase}.{key} must be a positive integer')
        if band(p['train_seed']) != DIAGNOSTIC_BAND:
            raise ValueError(f'{phase}: training seed must lie in the diagnostic band')
        if check_bands([p['eval_seed']]):
            raise ValueError(f'{phase}: evaluation seed reuses a judged or diagnostic band')
        if phase == 'main' and (type(p.get('n_days')) is not int or not 3 <= p['n_days'] <= 30):
            raise ValueError('main.n_days must be an integer between 3 and 30')
        if phase == 'smoke' and (not isinstance(p.get('loads'), list) or len(p['loads']) < 2
                                 or any(type(x) is not int or x <= 0 for x in p['loads'])):
            raise ValueError('smoke.loads must list at least two positive daily volumes')
    if cfg['main']['train_seed'] == cfg['smoke']['train_seed'] or cfg['main']['eval_seed'] == cfg['smoke']['eval_seed']:
        raise ValueError('Smoke and main seeds must differ')


def plan_for(cfg, phase, which):
    """Day plan for a phase ('smoke'|'main') and role ('train'|'eval')."""
    from yard_rl.v3.stage.month import plan_days, plan_month
    p = cfg[phase]
    seed = p['train_seed'] if which == 'train' else p['eval_seed']
    if phase == 'smoke':
        return seed, plan_days(seed, tuple(p['loads']))
    return seed, plan_month(seed, n_days=p['n_days'])


def build_eval_input(seed, days, folder):
    """Canonical month payload with the count-balanced vessel plan; fingerprints saved."""
    from yard_rl.v3.eval.seed_bank import create_month_payload, validate_payload, write_bundle
    from yard_rl.v3.stage.supply_plan import balance_vessel_supply
    payload = create_month_payload(seed, days=days)
    validation = validate_payload(payload, require_month=False)
    initial = {b: len(s['containers']) for b, s in payload['initial_scenarios'].items()}
    vessels, supply = balance_vessel_supply(payload['vessels'], payload['schedule'],
                                           initial, {b: 1440 for b in initial})
    expected = dict(schedule_sha256=digest(payload['schedule']),
                    initial_scenarios_sha256=digest(payload['initial_scenarios']),
                    vessels_sha256=digest(vessels))
    save(folder / 'canonical-input.json', dict(validation=validation, supply_plan_audit=supply,
        canonical_bundle_sha256=write_bundle(folder / 'canonical-input.json.gz', payload),
        balanced_vessels_sha256=write_bundle(folder / 'balanced-vessels.json.gz', vessels)))
    return expected


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def write_rows(path, rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def final_checkpoint(train_dir, n_days):
    return Path(train_dir) / f'ckpt_{n_days - 1:03d}.pt'


def training_checks(train_dir, *, env_spec_sha256, n_days, labels_per_day, init_seed, seed):
    """Completed training evidence: every day checkpointed, manifest matches the contract."""
    train_dir = Path(train_dir)
    manifest = read(train_dir / 'training_manifest.json')
    history = read(train_dir / 'history.json')
    month = read(train_dir / 'month.json')
    ckpts = [train_dir / f'ckpt_{i:03d}.pt' for i in range(n_days)]
    checks = dict(
        manifest_contract=(manifest.get('arm') == TRAIN_ARM and manifest.get('admission_mode') == 'PRESERVE'
                           and manifest.get('supply_mode') == 'COUNT_BALANCED'
                           and manifest.get('environment_spec_sha256') == env_spec_sha256
                           and manifest.get('seed') == seed and manifest.get('init_seed') == init_seed
                           and manifest.get('labels_per_day') == labels_per_day),
        every_day_checkpointed=all(c.is_file() for c in ckpts) and (train_dir / 'ckpt_init.pt').is_file(),
        history_complete=len(history) == n_days,
        month_complete=len(month.get('days', [])) == n_days,
        measurement_days_fitted=set(manifest.get('measurement_days', [])) <= set(manifest.get('fit_days', [])),
        optimizer_updated=manifest.get('optimizer_steps', 0) > 0,
        no_spatial_actions=all(d.get('n_space', 0) == 0 for d in month.get('days', [])),
        labels_present=sum(h.get('n_labels', 0) for h in history) > 0,
    )
    return dict(passed=all(checks.values()), checks=checks,
                final_checkpoint=str(final_checkpoint(train_dir, n_days)),
                final_checkpoint_sha256=sha(final_checkpoint(train_dir, n_days)) if checks['every_day_checkpointed'] else None,
                init_checkpoint_sha256=sha(train_dir / 'ckpt_init.pt') if checks['every_day_checkpointed'] else None,
                manifest_sha256=sha(train_dir / 'training_manifest.json'),
                total_labels=sum(h.get('n_labels', 0) for h in history),
                optimizer_steps=manifest.get('optimizer_steps'), fit_days=manifest.get('fit_days'))


def recording_checks(result, *, environment, expected_input, environment_spec_sha256,
                     weights_unchanged, checkpoint_unchanged, runtime_unchanged, rollout_count):
    """Record validity only; unfinished work is an outcome, never a missing record."""
    req, vessel, flow = (result[k] for k in
                         ('request_summary', 'vessel_work_summary', 'container_flow_summary'))
    days = result['days']
    manifest = result.get('environment_manifest') or {}
    canonical = (manifest.get('canonical_input') == expected_input if environment == 'vertical'
                 else manifest == {})
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
        canonical_input=canonical and result['expected_input'] == expected_input,
        environment_identity=result['environment_spec_sha256'] == environment_spec_sha256,
        daily_observation_complete=len(days) == len(result['plan']) and bool(days) and all(
            d['operational'].get('day_index') == d['index'] and 'cohort' in d['operational']
            for d in days))


def save_result(folder, res, *, environment, arm, seed, days, elapsed_s, identity, runtime,
                expected_input, stamp, rollout_count):
    req_path, link_path = folder / 'requests.jsonl.gz', folder / 'container-links.jsonl.gz'
    write_rows(req_path, res.request_ledger)
    write_rows(link_path, res.container_links)
    finals = [d.as_dict() for d in res.days]
    write_rows(folder / 'daily-final.jsonl', (dict(seed=seed, arm=arm, policy_label=f'{environment}/{arm}', **d) for d in finals))
    request_identity = [{k: r[k] for k in ('job_id', 'requested_day', 'flow', 'requested_block',
        'requested_arrival_s', 'lead_s', 'requested_target', 'travel_s')} for r in res.request_ledger]
    return dict(environment=environment, arm=arm, label=f'{environment}/{arm}', seed=seed, elapsed_s=elapsed_s,
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
        daily_artifacts={n: sha(folder / n) for n in ('operating-state.jsonl.gz', 'daily-final.jsonl')},
        expected_input=expected_input, runtime=runtime, repro=stamp, **identity,
        claim_eligible=False, new_training_runs=0,
        scope='Layout-specific training, one training seed and one evaluation seed; descriptive only')


def operational_outcomes(result):
    req, vessel = result['request_summary'], result['vessel_work_summary']
    rows = result['vessel_admissions']
    return dict(all_trucks_completed=req['states'].get('COMPLETED', 0) == req['requested'],
        truck_unfinished=req['requested'] - req['states'].get('COMPLETED', 0),
        unbound_at_cutoff=req['unbound_jobs_at_end'],
        vessel_unfinished=vessel['outstanding_yard_jobs'],
        all_vessels_completed=vessel['all_requested_work_completed'],
        structurally_late_vessel_streams=sum(r.get('structural_min_overrun_s', 0) > 0 for r in rows),
        maximum_structural_min_overrun_s=max((r.get('structural_min_overrun_s', 0) for r in rows), default=0),
        scope='Observed completion/capacity outcomes; separate from input and recording validity')


def pair_summary(environment, results, *, n_days):
    """Same-environment pair: baseline vs its own trained time-only model."""
    if len(results) != 2 or {r.get('arm') for r in results} != set(ARMS):
        raise ValueError('Exactly one completed result per policy is required')
    a, b = (next(r for r in results if r['arm'] == arm) for arm in ARMS)
    for key in ('environment', 'seed', 'plan', 'expected_input', 'requested_identity_sha256',
                'environment_spec_sha256', 'pair_contract_sha256', 'runtime', 'source_commit',
                'config_sha256', 'prereg_sha256', 'checkpoint_sha256', 'network_identities'):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'Unpaired results in {environment}: {key}')
    if a['environment'] != environment or a['environment_manifest'] != b['environment_manifest']:
        raise ValueError('Environment identity differs inside the pair')
    if any(not r.get('recording_checks') or not all(v is True for v in r['recording_checks'].values())
           or r['claim_eligible'] is not False or r['rollout_calls'] != 0
           or r.get('new_training_runs') != 0 for r in (a, b)):
        raise ValueError('Recording or frozen-policy guard failed')
    for r in (a, b):
        planned = [d['index'] for d in r['plan']]
        if planned != list(range(n_days)) or [d['index'] for d in r['days']] != planned:
            raise ValueError('Daily records are missing, duplicated or out of order')
        if any(d['train'] is not (0 < d['index'] < n_days - 1) for d in r['days']):
            raise ValueError('Measurement-day selection changed')
        if any(not math.isfinite(d['phi_krw']) or d['phi_krw'] < 0 for d in r['days']):
            raise ValueError('Daily costs must be finite and nonnegative')
    scopes = {}
    for name, cond in (('measurement_days', lambda d: d['train']), ('all_days', lambda d: True)):
        ca, cb = (math.fsum(d['phi_krw'] for d in r['days'] if cond(d)) for r in (a, b))
        scopes[name] = dict(baseline_cost_krw=ca, time_only_cost_krw=cb, saving_krw=ca - cb,
                           saving_pct=(100 * (ca - cb) / ca if ca > 0 else None),
                           days=[d['index'] for d in a['days'] if cond(d)])
    return dict(passed=True, environment=environment, seed=a['seed'], policies=list(ARMS), n_days=n_days,
        costs=scopes, per_day=[dict(index=x['index'], load=x['load'], train=x['train'],
                                    baseline_phi_krw=x['phi_krw'], time_only_phi_krw=y['phi_krw'])
                               for x, y in zip(a['days'], b['days'])],
        operational_outcomes={r['arm']: operational_outcomes(r) for r in (a, b)},
        actions={r['arm']: dict(time=r['time'], space=r['space'], traded=r['traded']) for r in (a, b)},
        claim_eligible=False, statistical_inference=False, independent_sample_count=1,
        purpose='Layout-specific weights evaluated in their own environment; descriptive, no H/V causal claim')
