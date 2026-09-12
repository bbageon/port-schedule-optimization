"""Archive the old admission failure and bounded no-rejection regression proof."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from yard_rl.experiments.gate_harness import (
    attach_common_gates, judge_claim_alignment, report_from_dict,
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--failed-run',type=Path,required=True)
    args = ap.parse_args()
    root,old = args.report,args.failed_run
    status,manifest = read(old/'status.json'),read(old/'manifest.json')
    if status['state']!='failed' or status['time_s']!=1817220 or status['skipped']!=1:
        raise RuntimeError('Unexpected prior failure; do not archive as the known capacity failure')
    cp = status['failed_checkpoint']
    if sha(old/cp['path'])!=cp['sha256']:
        raise RuntimeError('Failed checkpoint checksum mismatch')
    xml = ET.parse(root/'final-tests.xml').getroot()
    suites = list(xml.iter('testsuite'))
    tests = {k:sum(int(s.get(k,'0')) for s in suites)
             for k in ('tests','failures','errors','skipped')}
    if any(tests[k] for k in ('failures','errors','skipped')):
        raise RuntimeError(tests)
    tests['seconds'] = sum(float(s.get('time','0')) for s in suites)
    case = next(c for c in xml.iter('testcase') if c.get('name')==
                'test_full_yard_retains_waiting_order_and_continues_charging_ppo_reward')
    measurements = {p.get('name'):float(p.get('value')) for p in case.iter('property')}
    measurements['extra_cost_for_600_s_wait'] = (
        measurements['wait_cost_at_1200_s']-measurements['wait_cost_at_600_s'])
    reported = dict(tests=tests['tests'],old_updates=480,old_admitted=125269,
                    old_skipped=1,old_time_s=1817220)
    raw = dict(tests=len(list(xml.iter('testcase'))),old_updates=status['updates'],
               old_admitted=status['admitted'],old_skipped=status['skipped'],old_time_s=status['time_s'])
    alignment = judge_claim_alignment(reported_values=reported,raw_values=raw)
    if alignment.status.value!='PASS':
        raise RuntimeError(alignment.as_dict())
    code_paths = ['src/yard_rl/v5/stage/cargo_runtime.py','tests/v5/test_cargo_capacity_wait.py']
    diff = subprocess.check_output(['git','diff','415e3e7','--',*code_paths],text=True)
    if diff:
        raise RuntimeError('Tested files differ from the recorded source commit')
    folder = root/'failed-run'
    folder.mkdir(exist_ok=False)
    for name in ('manifest.json','status.json','admissions.json','cargo-status.json',
                 'days.json','partial_report.json','container_contract.json','events.jsonl'):
        shutil.copyfile(old/name,folder/name)
    artifacts = list(folder.iterdir())+list(root.glob('*.xml'))+[root/'gate-after-wait.json']
    value = dict(task='YR-306',scope='bounded admission contract correction, not a full replay',
        claim_scope='NO_PERFORMANCE_CLAIM',preregistration_commit='4937302',
        preregistration='.claude/docs/dashboard-task-specs/YR-306-admission-wait.md',
        test_source_commit='415e3e7',tested_source_matches_commit=True,
        test_source_sha256_lf={p:hashlib.sha256(Path(p).read_text(encoding='utf-8').encode()).hexdigest()
                              for p in code_paths},
        old_source=manifest['code'],absolute_seed=9900306,tests=tests,
        reported_counts=reported,claim_alignment=alignment.as_dict(),wait_measurements=measurements,
        full_block_test='actual 1440-box block; unchanged stack constraints and PPO cost/reward path',
        balance_test='synthetic 934-box + 542 planned-discharge reconstruction; not a day-22 replay',
        integration_test='actual new-bundle three-day test; only code_stamp substituted in the test',
        maximum_test_cpu_cores=1,cpu_affinity=[23],long_run_restarted=False,
        full_v5_suite=False,old_world_resumed=False,
        unchanged=['seed bundle','PPO parameters/features','cost coefficients','frozen world','v3/v4'],
        limits=['30-day rerun not performed','old one-truck short-run residual not diagnosed',
                'no baseline performance comparison or real cargo-flow calibration'],
        artifacts_sha256={p.relative_to(root).as_posix():sha(p) for p in artifacts})
    write(root/'verification.json',attach_common_gates(value,
          report_from_dict(read(root/'gate-after-wait.json'))))
    print(json.dumps(dict(tests=tests,wait_measurements=measurements),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
