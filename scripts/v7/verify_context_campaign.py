"""Verify the finished frozen campaign and bind its independent raw-ledger readout."""
from pathlib import Path

from audit_day_shapes import read
from context_learning import OUT, SEEDS, ARMS, check, evaluation_cases
from six_step_learning import gz_read, sha
from yard_rl.integrated.repro import write_result


def main():
    manifest = check()
    summary = read(OUT / 'summary.json')
    assert summary['cases'] == 72
    for path, expected in summary['artifact_hashes'].items():
        assert sha(Path(path)) == expected, path
    names = {p.name for p in (OUT / 'evaluation').glob('*.json.gz')}
    expected_names = {f'{i:03d}.json.gz' for i in range(72)} | {'006-repeat.json.gz', '018-repeat.json.gz'}
    assert names == expected_names
    for sub in ('orders', 'flow-orders'):
        assert {p.name for p in (OUT / sub).glob('*.json.gz')} == names
    total_orders = 0
    for i, case in enumerate(evaluation_cases()):
        record = gz_read(OUT / 'evaluation' / f'{i:03d}.json.gz')
        assert record['case'] == case
        assert record['probe']['ledger_audit']['skipped'] == 0
        assert record['visits']['late_releases'] == record['visits']['late_exits'] == 0
        assert record['result']['admitted'] == case['load']
        total_orders += case['load']
    for i in (6, 18):
        original = gz_read(OUT / 'evaluation' / f'{i:03d}.json.gz')
        repeated = gz_read(OUT / 'evaluation' / f'{i:03d}-repeat.json.gz')
        for field in ('candidates_sha256', 'choices_sha256', 'events_sha256', 'jobs_sha256', 'records_sha256', 'result_sha256'):
            assert original['probe'][field] == repeated['probe'][field], (i, field)
        assert original['orders_sha256'] == repeated['orders_sha256']
        assert original['schedule_sha256'] == repeated['schedule_sha256']
        assert original['completion'] == repeated['completion']
        assert gz_read(OUT / 'flow-orders' / f'{i:03d}.json.gz') == gz_read(OUT / 'flow-orders' / f'{i:03d}-repeat.json.gz')
    preflight = read(OUT / 'input-preflight.json')
    assert preflight['cases'] == 12 and preflight['all_passed']
    corpus = read(OUT / 'corpus-audit.json')
    assert len(corpus['rows']) == 9
    fitting = read(OUT / 'training-audit.json')
    assert fitting['finite_checkpoints'] == len(SEEDS) * len(ARMS) == 18
    assert fitting['independently_recomputed_fitting_tables'] == 36
    assert fitting['first_seed_six_retraining_weights_exact']
    sidecars = [p for p in OUT.rglob('*.sha256') if p.name != 'verification.json.sha256']
    for sidecar in sidecars:
        assert sha(Path(str(sidecar)[:-7])) == sidecar.read_text(encoding='utf-8').strip(), sidecar
    metadata = dict(
        input_cases=12, source_files_unchanged=len(manifest['sources']),
        corpus_days=9, training_pairs=211, selection_pairs=96, half_training_pairs=104,
        trained_models=18, exact_retraining_models=6, independent_fitting_tables=36,
        primary_evaluation_runs=72, exact_operating_replays=2,
        primary_truck_order_rows=total_orders, replay_truck_order_rows=7000,
        raw_ledgers_recomputed_by='scripts/v7/report_context_learning.py -> report_synthetic_pipeline.audit',
        raw_readout_sha256=sha(OUT / 'summary.json'), checked_artifact_hashes=len(summary['artifact_hashes']),
        sidecars_checked=len(sidecars), source_snapshot=manifest['repro'],
        plot_visually_checked=True,
        test_bundles=[
            dict(files=['tests/v7/test_context_crane.py', 'tests/v7/test_crane_policy.py',
                        'tests/v7/test_crane_teacher.py', 'tests/v7/test_v7_crane_teacher.py'],
                 passed=37, seconds=329.15, stage='initial context implementation'),
            dict(files=['tests/v7/test_context_crane.py', 'tests/v7/test_correlated_arrivals.py'],
                 passed=11, seconds=86.91, stage='expanded context capture and first arrival implementation'),
            dict(files=['tests/v7/test_correlated_arrivals.py', 'tests/v7/test_synthetic_day_mix.py',
                        'tests/v7/test_observed_demand_profile.py'], passed=21, seconds=1.88,
                 stage='final multiscale arrival implementation')],
        scope='Overlapping test bundles are not a sum of unique cases. All 72 primary cases retained; bounded synthetic diagnostic, no field-performance claim.')
    assert total_orders == 624000
    write_result(OUT / 'verification.json', metadata)
    print({k: metadata[k] for k in ('primary_evaluation_runs', 'exact_operating_replays', 'primary_truck_order_rows', 'checked_artifact_hashes', 'sidecars_checked')})


if __name__ == '__main__':
    main()
