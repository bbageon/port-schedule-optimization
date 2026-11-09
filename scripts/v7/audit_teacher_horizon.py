"""Real finite-world boundary check for both teacher contracts and worker modes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from benchmark_default_eval import save
from six_step_learning import sha, gz_save

OUT = Path('outputs/reports/yr332_teacher_horizon')
PREREG = Path('docs/research/v7-discharge/reachable-teacher-horizon-amendment.md')


def run():
    import torch
    from yard_rl.integrated.repro import repro_stamp
    from yard_rl.v7.crane.policy import CraneNet, add_store_input
    from yard_rl.v7.stage.episode import run_episode
    from yard_rl.v7.stage.rollout import RolloutBudget
    from yard_rl.v7.world.integrated.terminal_stream import ObservationContract
    torch.set_num_threads(1)
    assert not subprocess.check_output(['git','status','--short','--untracked-files=no']).strip()
    config = dict(load=100,seed=9914697,measure_s=1800.,world_end_s=9000.,
        pairs=4,candidate_cap=None,cases=[('legacy_force',10800.,1),('legacy_force',10800.,2),
            ('reachable',10800.,1),('reachable',10800.,2),('reachable',1800.,1)])
    paths = [Path(__file__),PREREG,*Path('src/yard_rl/v7').rglob('*.py')]
    save(OUT/'manifest.json',dict(configuration=config,sources={p.as_posix():sha(p) for p in paths},
        repro=repro_stamp(experiment='YR-332-ac',seeds=dict(input=[config['seed']]),
                          params=config,prereg=PREREG.as_posix())))
    results = []
    for contract,horizon,workers in config['cases']:
        torch.manual_seed(config['seed'])
        net = add_store_input(CraneNet(feature_mode='feasible'))
        ep = run_episode(load=config['load'],seed=config['seed'],dispatcher='RL_CRANE',crane_net=net,
            obs=ObservationContract(warmup_s=0.,measure_s=config['measure_s']),
            crane_budget=RolloutBudget(max_labels=config['pairs'],identity_checks=0),
            crane_sampling='coverage',crane_alternative_contract=contract,horizon_s=horizon,
            workers=workers,candidate_cap=None,reuse_rehandle_slots=True,decision_probe_shortcircuit=True)
        data = dict(contract=contract,horizon_s=horizon,workers=workers,result=ep.as_dict(),
                    samples=[asdict(s) for s in ep.crane_labels])
        gz_save(OUT/f'{contract}-{int(horizon)}-{workers}.json.gz',data)
        cs=ep.crane_stats
        assert ep.policy_exceptions==cs['hook_errors']==cs['factual_mismatch']==0,cs
        assert cs['labeled']>0 and ep.admitted==100,cs
        if contract=='reachable':
            assert cs['force_failed']==0
        for r in cs['comparisons']:
            assert r['cost_end_s']==min(r['t']+horizon,config['world_end_s'])
            assert r['effective_horizon_s']==r['cost_end_s']-r['t']>0
            if horizon==10800:
                assert r['requested_end_s']>r['cost_end_s']==config['world_end_s']
            else:
                assert r['cost_end_s']==r['requested_end_s']
        results.append(json.loads(json.dumps(data)))
        print(f'{contract}/{horizon}/{workers}: {cs["labeled"]} pairs; cost inside world',flush=True)
    for a,b in ((results[0],results[1]),(results[2],results[3])):
        assert a['result']==b['result'] and a['samples']==b['samples']
    for r in results[1:]:
        assert r['result']['crane']['main_choices_sha256']==results[0]['result']['crane']['main_choices_sha256']
        assert r['result']['phi_krw']==results[0]['result']['phi_krw']
    save(OUT/'verification.json',dict(status='PASS',runs=5,main_trajectory_equal=True,
        serial_parallel_equal=True,clipped_and_full_windows=True,
        comparisons=sum(len(d['result']['crane']['comparisons']) for d in results),
        artifact_sha256={p.as_posix():sha(p) for p in OUT.glob('*.gz')}))


def finalize(pin):
    from yard_rl.experiments.gate_harness import (
        GateOutcome,GateStatus,attach_common_gates,audit_dashboard,
        judge_runtime_evidence,report_from_dict,judge_claim_alignment)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    m,v=read(OUT/'manifest.json'),read(OUT/'verification.json')
    artifacts={**m['sources'],**v['artifact_sha256'],str(OUT/'verification.json'):sha(OUT/'verification.json')}
    runtime=asdict(judge_runtime_evidence(m['repro'],artifact_hashes=artifacts,root='.'))
    assert runtime['status']=='PASS',runtime
    board=asdict(audit_dashboard('.',task_id='YR-332-ac',expected_state='done',
        spec_path='.claude/docs/dashboard-task-specs/YR-332-ac-teacher-horizon-accounting.md',
        evidence_paths=[str(OUT/'verification.json'),str(OUT/'manifest.json')],
        evidence_commits=[pin],remote_ref='origin/강화학습-판매',pin_commit=pin))
    assert board['status']=='PASS',board
    claims=asdict(judge_claim_alignment({'runs':v['runs']},{'runs':len(m['configuration']['cases'])}))
    gates=dict(performance=GateOutcome('performance',GateStatus.FAIL,'새 학습 평가 전; 채택 없음').as_dict(),
        reliability=GateOutcome('reliability',GateStatus.PASS,'환경 끝·장부·직렬병렬·기록 일치',
            evidence=dict(runtime=runtime,dashboard=board,claim_alignment=claims)).as_dict(),
        scenario_validity=GateOutcome('scenario_validity',GateStatus.FAIL,
            '교사 비용 창 보정 완료; 운반·예약 실측 보정 미확정',
            reasons=['YR-332-i transport calibration','YR-332-f appointment realism']).as_dict())
    verdict=dict(task='YR-332-ac',horizon_correction='PASS',common_gates=gates)
    save(OUT/'final-verdict.json',attach_common_gates(verdict,report_from_dict(verdict)))
    print('finite-world accounting PASS; empirical calibration remains unresolved')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin')
    args=p.parse_args()
    finalize(args.pin) if args.pin else run()
