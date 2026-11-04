"""Realized service/throughput audit; does not fit on policy evaluation data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(calibration_dir, validation_dir):
    frozen = read(calibration_dir / 'calibration.json')
    out = dict(frozen=frozen, runs={},
               calibration_sha256=hashlib.sha256((calibration_dir/'calibration.json').read_bytes()).hexdigest())
    for label, root in [('calibration', calibration_dir), ('validation', validation_dir)]:
        if not (root/'report.json').exists():
            failure = read(root/'failure.json') if (root/'failure.json').exists() else {'error': 'missing report'}
            out['runs'][label] = dict(status='failed', failure=failure)
            continue
        report, jobs = read(root/'report.json'), read(root/'completed-jobs.json')
        cfg = frozen['config']
        horizons = {}
        for horizon in (300, 900, 1800):
            samples = defaultdict(float)
            for row in jobs:
                if not 86400 <= row['end'] < 172800:
                    continue
                cell = int((row['end']-86400)//horizon)
                samples[(row['block'], row['crane'], cell)] += cfg['service_s'][row['flow']]
            ratios = [samples[(b, c, k)]/(horizon*cfg['capacity_fraction'][c])
                      for b in {j['block'] for j in jobs} for c in cfg['capacity_fraction']
                      for k in range(86400//horizon)]
            horizons[str(horizon)] = dict(mean_realized_output_fraction=float(np.mean(ratios)),
                p90_realized_output_fraction=float(np.quantile(ratios, .9)),
                fraction_windows_above_provisional_cap=float(np.mean(np.array(ratios)>cfg['safe_fraction'])))
        cohort = report['telemetry']['measured_cohorts'][0]
        out['runs'][label] = dict(seed=report['seed'], completed_jobs=len(jobs),
            horizons=horizons, measured_cohort=cohort, invariants=report['invariants'],
            admitted=report['admission']['admitted'], skipped=report['admission']['skipped'],
            mean_queue_budget_300s_met=cohort['block_queue_mean_s'] <= 300)
    out['safe_cap_validated'] = False
    out['interpretation'] = ('Observed output fractions are throughput, not input demand or a controlled capacity sweep. '
        'Measured mean service and reference capacity are frozen; the queue formula is provisional. '
        'Passing mean queue in one cohort would not establish a per-crane admission guarantee or optimum.')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.calibration, a.validation)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
