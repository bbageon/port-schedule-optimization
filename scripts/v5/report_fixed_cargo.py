"""Freeze small execution proof and a point-in-time long-run snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from yard_rl.experiments.gate_harness import (
    attach_common_gates, judge_claim_alignment, report_from_dict,
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--long-run', type=Path, required=True)
    args = parser.parse_args()
    root = args.report
    smoke, live = args.smoke, args.long_run
    s, sm = read(smoke/'report.json'), read(smoke/'manifest.json')
    lm, ls = read(live/'manifest.json'), read(live/'status.json')
    days = read(smoke/'days.json')
    lc, la = read(live/'cargo-status.json'), read(live/'admissions.json')
    if ls != read(live/'status.json') or ls.get('time_s') != lc['time_s']:
        raise RuntimeError('Live snapshot changed while reading; retry at a stable boundary')
    xml = ET.parse(root/'verified-tests.xml').getroot()
    suites = list(xml.iter('testsuite'))
    tests = {k:sum(int(t.get(k,'0')) for t in suites)
             for k in ('tests','failures','errors','skipped')}
    tests['seconds'] = sum(float(t.get('time','0')) for t in suites)
    checks = dict(
        all_selected_tests_passed=all(tests[k]==0 for k in ('failures','errors','skipped')),
        same_clean_code=sm['code']['git_head']==lm['code']['git_head']
            and sm['code']['source_sha256']==lm['code']['source_sha256']
            and sm['code']['git_dirty'] is lm['code']['git_dirty'] is False,
        fixed_30d_input=lm['fixed_seed']['sha256']==
            'efdb45e2ec8feafed6987cf48dfe022ca2959dad8c4f1d56bee680456081af7a',
        small_run_completed=s['state']=='completed' and s['admitted']==180 and s['skipped']==0,
        no_counterfactuals=s['counterfactual_worlds']==0,
        calendar_updates=[d['updates'] for d in days]==[0,24,0],
        exactly_one_learning_day=s['learning_intervals']==1440,
        real_dependent_routing=s['cargo']['dependent_reroutes']>0
            and s['cargo']['remote_vessel_handoffs']>0,
        cargo_conservation=s['cargo']['physical_inventory']==s['cargo']['expected_inventory']
            and lc['physical_inventory']==lc['expected_inventory'],
        long_run_started=ls['state']=='running' and ls['time_s']>0,
        long_run_no_admission_loss=ls['skipped']==ls['vessel_failed']==0,
        same_hyperparameters=sm['ppo']==lm['ppo'],
    )
    if not all(checks.values()):
        raise RuntimeError(checks)
    for label in ('initial_checkpoint','final_checkpoint'):
        c = s[label]
        if digest(smoke/c['path']) != c['sha256']:
            raise RuntimeError('Checkpoint checksum mismatch')
    snapshots = dict(smoke={}, long_run=dict(manifest=lm,status=ls,cargo=lc,admissions=la))
    for name in ('manifest','report','status','days','admissions','cargo-status',
                 'container_contract','month_result','cohort_reports'):
        snapshots['smoke'][name] = read(smoke/(name+'.json'))
    artifacts = []
    for group, values in snapshots.items():
        folder = root/group
        folder.mkdir(exist_ok=False)
        for name,value in values.items():
            path = folder/(name+'.json')
            write(path,value)
            artifacts.append(path)
    target = root/'smoke/events.jsonl'
    shutil.copyfile(smoke/'events.jsonl', target)
    artifacts.append(target)
    artifacts += list(root.glob('*.xml'))
    counts = dict(tests=tests['tests'], small_trucks=s['admitted'], small_updates=len(s['updates']),
                  long_time_s=ls['time_s'], long_admitted=ls['admitted'], long_updates=ls['updates'])
    alignment = judge_claim_alignment(reported_values=counts, raw_values=dict(
        tests=len(list(xml.iter('testcase'))), small_trucks=180, small_updates=24,
        long_time_s=lc['time_s'], long_admitted=la['admitted'], long_updates=ls['updates']))
    if alignment.status.value != 'PASS':
        raise RuntimeError(alignment.as_dict())
    value = dict(task='YR-306', claim_scope='NO_PERFORMANCE_CLAIM',
        scope='fixed input runtime acceptance and observed restart, not 30-day completion',
        preregistration_commit='db29b87',
        preregistration='.claude/docs/dashboard-task-specs/YR-306-fixed-cargo-runtime.md',
        source_code=lm['code'], absolute_seed=lm['seed'], tests=tests, checks=checks,
        reported_counts=counts, claim_alignment=alignment.as_dict(),
        cpu_affinity=[23], maximum_cpu_cores=1, long_run_directory=str(live),
        long_run_snapshot_only=True, long_run_completed=False,
        training_schedule='1 warmup + 28 training + 1 cooldown day, then 2h drain',
        full_v5_suite_run=False, all_cargo_completed_claim=False,
        known_limits=['fixed cross-block vessel transfer service assumption, not distance-calibrated',
                      'cargo pooled within block, not real import/export flow calibration',
                      'untrained policy; no baseline comparison or performance claim'],
        artifacts_sha256={p.relative_to(root).as_posix():digest(p) for p in artifacts})
    write(root/'verification.json', attach_common_gates(value,
        report_from_dict(read(root/'gate-after-runtime.json'))))
    print(json.dumps(dict(checks=checks, counts=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
