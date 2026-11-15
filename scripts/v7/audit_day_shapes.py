"""Input-only calibration/audit for YR-332-ao. No policy outcomes select shapes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np

from yard_rl.v7.day_shape import basis_hour_masses, periodic_linear_basis, build_bank, demand_for_shape, digest

OUT = Path('outputs/reports/yr332_day_shapes')
PREREG = Path('docs/research/v7-discharge/day-shape-prereg.md')
BASE = Path('outputs/reports/yr332_visit_links/demand-profiles.json')
INFO = Path('outputs/reports/yr332_real_order_information/profile.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fit_simplex(target, basis):
    """Convex simplex least squares: active sets for four bases, checked SLSQP otherwise."""
    matrix = np.array(basis).T
    if len(basis) > 4:
        from scipy.optimize import minimize
        size = len(basis)
        solution = minimize(lambda w: np.square(matrix @ w - target).sum(), np.ones(size) / size,
            jac=lambda w: 2 * matrix.T @ (matrix @ w - target), bounds=[(0., 1.)] * size,
            constraints=[dict(type='eq', fun=lambda w: w.sum() - 1., jac=lambda w: np.ones(size))],
            method='SLSQP', options=dict(ftol=1e-13, maxiter=1000))
        if not solution.success:
            raise RuntimeError(solution.message)
        w = np.maximum(0., solution.x);w /= w.sum()
        gradient = 2 * matrix.T @ (matrix @ w - target)
        active = w > 1e-7
        level = float(gradient[active].mean())
        if max(abs(gradient[active] - level)) > 2e-6 or np.any(gradient[~active] < level - 2e-6):
            raise RuntimeError('simplex optimum stationarity check failed')
        return float(np.square(matrix @ w - target).sum()), w, matrix @ w
    best = None
    for size in range(1, len(basis) + 1):
        for ids in combinations(range(len(basis)), size):
            a = matrix[:, ids]
            kkt = np.block([[a.T @ a, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(kkt, np.r_[a.T @ target, 1.], rcond=None)[0][:size]
            if solution.min() < -1e-10:
                continue
            w = np.zeros(len(basis))
            w[list(ids)] = np.maximum(0, solution)
            w /= w.sum()
            error = float(np.square(matrix @ w - target).sum())
            if best is None or error < best[0]:
                best = (error, w, matrix @ w)
    if best is None:
        raise RuntimeError('no feasible simplex fit')
    return best


def tvd(a, b):
    return float(.5 * np.abs(np.array(a) - np.array(b)).sum())


def stamp():
    from six_step_learning import sha
    files = [Path(__file__), PREREG, Path('src/yard_rl/v7/day_shape.py'), BASE, INFO]
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                tracked_dirty=bool(subprocess.check_output(['git', 'status', '--short', '--untracked-files=no']).strip()),
                sources={str(p).replace('\\', '/'): sha(p) for p in files})


def calibrate(*, knots=None):
    from audit_visit_units import load_rows, stamp as timestamp
    rows, manifest = load_rows()
    hours = defaultdict(lambda: [0] * 24)
    for row in rows:
        if row['gate_in_time']:
            t = timestamp(row['gate_in_time'])
            hours[t.date().isoformat()][t.hour] += 1
    # Freeze the already-viewed dates. New private dates must not silently enter development.
    allowed = read(Path('outputs/reports/yr332_visit_arrivals/visit-arrival-templates.json'))['dates']
    if set(hours) != set(allowed):
        raise ValueError('development dates changed; revise explicit split before using new dates')
    fits = []
    basis = basis_hour_masses() if knots is None else periodic_linear_basis(knots)
    family = 'original_gaussians' if knots is None else f'periodic_linear_{knots}'
    for index, (date, counts) in enumerate(sorted(hours.items())):
        p = np.array(counts) / sum(counts)
        error, w, fitted = fit_simplex(p, basis)
        rng = np.random.default_rng(9934000 + index)
        noise = .5 * np.abs(rng.multinomial(sum(counts), fitted, size=1000) / sum(counts) - fitted).sum(axis=1)
        threshold = float(np.quantile(noise, .95))
        residual = tvd(p, fitted)
        fits.append(dict(date=date, orders=sum(counts), hour_counts=counts, weights=w.tolist(),
                         hour_masses=fitted.tolist(), squared_error=error, residual_tvd=residual,
                         noise_tvd_p95=threshold, within_noise=residual <= threshold,
                         zero_weight_components=np.flatnonzero(w < 1e-8).tolist()))
    pooled = np.sum(list(hours.values()), axis=0);pooled = pooled / pooled.sum()
    mean = np.mean([f['hour_masses'] for f in fits], axis=0)
    old_basis = basis_hour_masses()
    original = .38 * np.array(old_basis[0]) + .62 * np.array([.317, .633, .050]) @ np.array(old_basis[1:])
    leave_one_out = []
    for removed in fits:
        remain = [f for f in fits if f is not removed]
        leave_one_out.append(dict(removed=removed['date'],
            mean_tvd=tvd(mean, np.mean([f['hour_masses'] for f in remain], axis=0)),
            weight_min=np.min([f['weights'] for f in remain], axis=0).tolist(),
            weight_max=np.max([f['weights'] for f in remain], axis=0).tolist()))
    bank = build_bank(fits, seed=9934000, basis=basis, family=family)
    identities = [r['shape_sha256'] for values in bank['splits'].values() for r in values]
    assert len(set(identities)) == len(identities)
    calibration = dict(schema='day-shape-calibration-v1', family=family, unit='entered container orders (not physical visits)',
        development_dates=sorted(hours), source_sha256=manifest, fits=fits, leave_one_out=leave_one_out,
        representation=dict(original_vs_pooled_order_tvd=tvd(original, pooled),
            mean_fit_vs_pooled_order_tvd=tvd(mean, pooled),
            observed_day_tvd_mean=float(np.mean([tvd(np.array(f['hour_counts']) / f['orders'], pooled) for f in fits])),
            fitted_day_tvd_mean=float(np.mean([tvd(f['hour_masses'], mean) for f in fits])),
            synthetic_train_tvd_mean=float(np.mean([tvd(r['hour_masses'], bank['fixed_hour_masses']) for r in bank['splits']['train']]))),
        input_family_pass=all(f['within_noise'] for f in fits),
        passed_dates=sum(f['within_noise'] for f in fits), total_dates=len(fits), repro=stamp())
    save(f'calibration-{family}.json', calibration)
    save(f'bank-{family}.json', bank)
    print(json.dumps({k: calibration[k] for k in ('input_family_pass', 'passed_dates', 'total_dates', 'representation')}, indent=2))


def check_stage(built, total):
    from yard_rl.v7.world.domain.enums import ContainerSize
    schedule = built['schedule']
    assert len(schedule) == total and built['flow_fallbacks_total'] == 0
    assert all(row['flow'] == row['requested_flow'] for row in schedule)
    visits = defaultdict(list)
    for row in schedule:
        assert 0 <= row['arrival_s'] < 86400
        assert 0 <= row['lead_s'] <= row['arrival_s']
        assert 'reserved_s' in row and 'job_done_time' not in row
        assert row['gate_in_s'] == row['arrival_s']
        if 'visit' in row:
            visits[row['visit']['id']].append(row)
    for group in visits.values():
        assert len({r['arrival_s'] for r in group}) == 1
        for flow in ('GATE_IN', 'GATE_OUT'):
            same = [r for r in group if r['flow'] == flow]
            def teu(row):
                if flow == 'GATE_IN':
                    return 2 if row['size_ft40'] else 1
                return 1 if built['scenarios'][row['block']].containers[row['target']].size == ContainerSize.FT20 else 2
            assert sum(teu(r) for r in same) <= 2
        ins = [r['visit']['stop'] for r in group if r['flow'] == 'GATE_IN']
        outs = [r['visit']['stop'] for r in group if r['flow'] == 'GATE_OUT']
        assert not ins or not outs or max(ins) < min(outs)
    return dict(orders=total, visit_links=built['visit_links']['visits'],
                flow_counts=dict(Counter(r['flow'] for r in schedule)),
                hour_counts=[sum(int(r['arrival_s'] // 3600) == h for r in schedule) for h in range(24)],
                multi_order_share=built['visit_links']['multi_order_share'],
                schedule_sha256=digest(schedule))


def audit_inputs():
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    bank = read(OUT / 'bank-ready.json')
    base = read(BASE)['profiles']['observed_visit_links']
    info = read(INFO)
    results = []
    for load in (3500, 7500, 15000):
        for i, shape in enumerate(bank['splits']['train'][:3]):
            cases = []
            seed = shape['input_seed']
            for arm, mass in [('fixed', bank['fixed_hour_masses']), ('variable', shape['hour_masses'])]:
                profile = demand_for_shape(base, mass, source=dict(bank_sha256=digest(bank), shape_id=shape['id'], arm=arm))
                built = build_stage(load=load, seed=seed, profile=build_h21_profile(), information_profile=info, demand_profile=profile)
                stats = check_stage(built, load)
                cases.append(built)
                results.append(dict(load=load, seed=seed, shape_id=shape['id'], arm=arm, **stats))
                if load == 3500 and i == 0 and arm == 'variable':
                    repeated = build_stage(load=load, seed=seed, profile=build_h21_profile(), information_profile=info, demand_profile=profile)
                    assert repeated == built
            a, b = cases
            assert a['scenarios'] == b['scenarios'] and a['vessel_schedule'] == b['vessel_schedule']
            for x, y in zip(a['schedule'], b['schedule']):
                for field in ('job_id', 'flow', 'block', 'target', 'con_no', 'size_ft40', 'travel_s', 'exit_travel_s'):
                    assert x.get(field) == y.get(field), field
    save('input-audit.json', dict(cases=results, checks_pass=True, repeated_case_exact=True, repro=stamp(),
                                scope='input contracts only; no runtime/learning/performance claim'))
    print(json.dumps(dict(cases=len(results), checks_pass=True, repeated_case_exact=True)))


def prepare_scenarios():
    """Condition on requested flows being preserved, before any policy is run.

    No curve is dropped. Only the exogenous stage seed is redrawn, equally for
    the fixed/variable arms. Rejecting a busy/slow valid stage is forbidden.
    """
    from yard_rl.v7.stage.orders import build_stage
    from yard_rl.v7.world.integrated.profiles import build_h21_profile
    bank = read(OUT / 'bank-periodic_linear_24.json')
    profile = demand_for_shape(read(BASE)['profiles']['observed_visit_links'], bank['fixed_hour_masses'],
                               source=dict(purpose='pre-policy flow-preservation preflight'))
    info = read(INFO)
    records = []
    for split, rows in bank['splits'].items():
        for shape in rows:
            requested = shape['input_seed']
            attempts = []
            for attempt in range(20):
                seed = requested + 1000 * attempt
                cases = []
                for load in (3500, 7500, 15000):
                    built = build_stage(load=load, seed=seed, profile=build_h21_profile(), information_profile=info,
                                        demand_profile=profile)
                    cases.append(dict(load=load, fallback_count=built['flow_fallbacks_total'],
                        fallback_by_block={k: v for k, v in built['flow_fallbacks'].items() if v},
                        requested_out=sum(e['requested_flow'] == 'GATE_OUT' for e in built['schedule']),
                        actual_out=sum(e['flow'] == 'GATE_OUT' for e in built['schedule'])))
                    if not built['flow_fallbacks_total']:
                        check_stage(built, load)
                good = all(c['fallback_count'] == 0 for c in cases)
                attempts.append(dict(seed=seed, accepted=good, cases=cases))
                save('scenario-preflight-progress.json', dict(completed=records, current=dict(id=shape['id'], attempts=attempts)))
                if good:
                    shape['requested_input_seed'] = requested
                    shape['input_seed'] = seed
                    break
            else:
                raise RuntimeError(f"no valid input after 20 prespecified attempts for {shape['id']}")
            records.append(dict(id=shape['id'], split=split, requested_seed=requested, input_seed=seed, attempts=attempts))
    seeds = [r['input_seed'] for r in records]
    assert len(set(seeds)) == len(seeds)
    manifest = dict(records=records, accepted=len(records), attempts=sum(len(r['attempts']) for r in records),
        rejection_rule='requested outbound containers unavailable; requested flow would be silently changed',
        retry_rule='requested input seed + 1000 * attempt; maximum 20; same seed for all loads and both arms',
        selection_scope='stock-feasible conditional synthetic scenarios; not unconditioned terminal demand',
        no_policy_outcomes_used=True, repro=stamp())
    bank['scenario_preflight_sha256'] = digest(manifest)
    save('scenario-preflight.json', manifest)
    save('bank-ready.json', bank)
    print(json.dumps(dict(accepted=manifest['accepted'], attempts=manifest['attempts'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['calibrate', 'inputs', 'prepare-scenarios'])
    parser.add_argument('--knots', type=int, choices=[6, 8, 12, 24])
    args = parser.parse_args()
    if args.command == 'calibrate':
        calibrate(knots=args.knots)
    elif args.command == 'prepare-scenarios':
        prepare_scenarios()
    else:
        audit_inputs()
