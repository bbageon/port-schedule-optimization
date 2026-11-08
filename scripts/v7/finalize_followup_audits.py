"""Check report numbers against raw files and pin final board evidence for m/w."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from six_step_learning import gz_read,sha
from benchmark_default_eval import save
from yard_rl.experiments.gate_harness import (
    audit_dashboard,judge_runtime_evidence,judge_claim_alignment,GateOutcome,GateStatus,
    report_from_dict,attach_common_gates)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def ledger_claims(root):
    from report_admission_ledger import LABELS
    names={v:k for k,v in LABELS.items()}
    raw={}
    for p in (root/'evaluation').glob('*.json.gz'):
        d=gz_read(p);arm,seed,load,repeat=d['case']
        if repeat:continue
        rows=gz_read(root/'orders'/p.name)
        admitted=[r for r in rows if r['times']['gate_in_s'] is not None]
        def exited(end):
            return sum(r['times']['gate_out_s'] is not None and r['times']['gate_out_s']<=end for r in admitted)
        values=[len(admitted),len(rows)-len(admitted),len(admitted)-exited(86400),len(admitted)-exited(93600),
            sum(r['times']['job_done_s'] is None or r['times']['job_done_s']>93600 for r in admitted)]
        raw.setdefault((arm,load),[]).append(values)
    reported={};actual={}
    for line in (root/'README.md').read_text(encoding='utf-8').splitlines():
        cells=[c.strip() for c in line.split('|')][1:-1]
        if len(cells)!=7 or cells[0] not in names:continue
        arm=names[cells[0]];load=int(cells[1].replace(',',''))
        for i,value in enumerate(cells[2:]):
            key=f'{arm}/{load}/{i}'
            reported[key]=float(value);actual[key]=statistics.mean(r[i] for r in raw[(arm,load)])
    assert len(reported)==len(raw)*5
    return reported,actual


def action_claims(root):
    raw={4:[],5:[]}
    for p in (root/'evaluation').glob('*.json.gz'):
        d=gz_read(p)
        if d['case'][2]:continue
        for row in d['pairs']:raw[row['stage']].append(row)
    reported={};actual={};names={'기존 표본':4,'확대 표본·최신 학습':5}
    for line in (root/'README.md').read_text(encoding='utf-8').splitlines():
        cells=[c.strip() for c in line.split('|')][1:-1]
        if len(cells)!=5 or cells[0] not in names:continue
        stage=names[cells[0]];rs=raw[stage];u=[r for r in rs if not r['proof']['reachable']]
        values=[len(rs),sum(r['mixed_service_tier'] for r in rs),len(u),
                sum(r['phi_alternative']<r['phi_picked'] for r in u)]
        for i,value in enumerate(cells[1:]):
            key=f'{stage}/{i}';reported[key]=float(value);actual[key]=values[i]
    assert len(reported)==8
    return reported,actual


def finalize(task,pin):
    slug,spec=(('admission_audit','daily-admission-accounting') if task=='m' else ('action_contract','training-action-contract'))
    root=Path('outputs/reports')/f'yr332_{slug}'
    manifest=read(root/'campaign.json');verification=read(root/'verification.json')
    assert verification['summary_sha256']==sha(root/'summary.json')
    assert verification['report_sha256']==sha(root/'README.md')
    hashes={**manifest['sources'],**manifest['input_sha256'],**verification['artifact_sha256']}
    if task=='w':
        # Validation also reads these historical metric reports. Their checkpoints
        # were frozen in the campaign; verify the reports against that same commit.
        execution_commit=manifest['repro']['code']['git_head']
        extra={}
        for seed in manifest['repro']['seeds']['model']:
            p=Path(f'outputs/reports/yr332_count_transform/training/{seed}.json')
            h=sha(p)
            original=subprocess.check_output(['git','show',f'{execution_commit}:{p.as_posix()}'])
            assert hashlib.sha256(original).hexdigest()==h,p
            extra[p.as_posix()]=h
        save(root/'supplemental-input-verification.json',dict(status='PASS',
            execution_commit=execution_commit,input_sha256=extra,
            scope='Historical validation metric references omitted from campaign input list; exact execution-commit blobs verified after replay. No simulation or training input changed.'))
        hashes.update(extra)
    runtime=asdict(judge_runtime_evidence(manifest['repro'],artifact_hashes=hashes,root='.'))
    assert runtime['status']=='PASS',runtime
    reported,actual=(ledger_claims(root) if task=='m' else action_claims(root))
    claims=asdict(judge_claim_alignment(reported,actual,absolute_tolerance=.00500001))
    assert claims['status']=='PASS',claims
    upstream=subprocess.check_output(['git','rev-parse','--abbrev-ref','@{upstream}'],encoding='utf-8').strip()
    board=asdict(audit_dashboard('.',task_id=f'YR-332-{task}',expected_state='done',
        spec_path=f'.claude/docs/dashboard-task-specs/YR-332-{task}-{spec}.md',
        evidence_paths=[str(root/p) for p in ('campaign.json','summary.json','verification.json','README.md')],
        evidence_commits=[pin],remote_ref=upstream,pin_commit=pin))
    assert board['status']=='PASS',board
    board['evidence']['report_claim_alignment']=claims
    board['evidence']['audit_source_sha256']=sha(__file__)
    save(root/'final-dashboard-audit.json',board)
    verdict=read(root/'final-verdict.json')
    verdict['dashboard_alignment']=dict(status='PASS',pin_commit=pin,evidence='final-dashboard-audit.json')
    verdict['common_gates']['reliability']=GateOutcome('reliability',GateStatus.PASS,
        '실행 코드·원자료·보고 수치·Dashboard·원격 커밋이 일치함',
        evidence=dict(runtime=runtime,dashboard=board,claim_alignment=claims)).as_dict()
    verdict=attach_common_gates(verdict,report_from_dict(verdict))
    save(root/'final-verdict.json',verdict)
    print(f'YR-332-{task}: runtime, {len(reported)} report values, Dashboard and remote commit PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=('m','w'))
    p.add_argument('--pin',required=True)
    args=p.parse_args();finalize(args.task,args.pin)
