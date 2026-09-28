"""Recompute all registered paired results and audit immutable execution evidence."""
import argparse
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from yard_rl.v6.ppo.provenance import file_sha256 as sha
from yard_rl.v6.ppo.journal import write_json
from yard_rl.v6.reward.operational import KEYS, REFERENCE, reference_config, normalized_loss
from yard_rl.experiments.gate_harness import (
    GateOutcome, GateStatus, ResearchGateReport, attach_common_gates, audit_dashboard,
    combine_reliability, judge_claim_alignment, judge_runtime_evidence, judge_scenario_validity,
)


def read(path):
    if not path.exists() and path.with_suffix(path.suffix+'.gz').exists():
        return json.loads(gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()))
    return json.loads(path.read_text(encoding='utf-8'))


def guard_values(report):
    endpoint = report['evaluation_endpoint']
    jobs = endpoint['cargo']['active_unfinished_jobs']
    return dict(original_request_s=report['measured_original_request_s'],
                truck_remaining=endpoint['request']['unfinished'],
                vessel_remaining=sum(n for k,n in jobs.items() if k.startswith('VESSEL_')))


def render_report(root,result):
    means=result['mean_objectives']; reduction=result['mean_reduction_percent']
    verdict='진단의 수치 조건 충족, 통계적 확증 아님' if result['diagnostic_success'] else '성능 개선 채택 조건 미충족'
    lines=['# v6 원화 없는 정규화 — 학습·평가 결과', '',
        f'**구현·학습·평가 완료. {verdict}.**', '',
        '## 진행사항', '',
        '- 원화 환산 없이 트럭 체류초·본선 중단초·빈 이동초·재취급 횟수를 각각 정규화했다.',
        '- 보상은 `−Σ_i 0.25×Δx_i/s_i`. 분모는 별도 기준 운전의 할인 누적값 표준편차이며 학습·평가 중 고정이다.',
        '- 원화 단가·추가 원화 분모·평균 제거·절단·부하 보조 점수는 새 보상에 없다. 원화 장부는 진단용이다.',
        f"- CPU {result['trained_models']}모델 × 24회 = {result['updates_applied']}회 실제 갱신, 동결 평가 {result['frozen_evaluations']}회. 등록한 15실행 모두 완료했다.",
        '- 같은 초기값의 세 쌍을 학습했다. 쌍별 첫날 물리 기록·초기 가중치·행동 난수가 같고, 평가 중 가중치 갱신은 0이다.',
        '- 비교군은 d의 원화 합계/37,620,956.31원 방식으로 같은 횟수만큼 새로 학습했다. 과거의 다른 길이 실험 수치를 그대로 가져온 비교가 아니다.',
        '- 주문 누락·본선 투입 실패·물리 불변조건 위반 0. 신경망 저장·복원 및 학습 종료 뒤 가중치 동결을 확인했다.',
        '- CPU 검사 89건, 추가 배열·연속 실행 검사 16건, 배열 호환·음수 장부 보완 검사 3건 통과(일부 재검사 포함).',
        '', '## 해석', '',
        '점수는 네 지표의 가중 합으로 작을수록 좋다. 개선률은 `100×(1−새 보상 모델 점수/비교군 점수)`이며 음수는 악화다.', '',
        '| 평가 입력 | 새 운영 보상 | 이전 원화 보상 | 규칙 | 이전 대비 개선률 | 규칙 대비 개선률 |',
        '|---|---:|---:|---:|---:|---:|']
    for p in result['pairs']:
        v=p['objectives']; c=p['comparisons']
        lines.append(f"| {p['eval_seed']} | {v['operational']:.4f} | {v['legacy']:.4f} | {v['rule']:.4f} | {c['legacy']['reduction_percent']:+.2f}% | {c['rule']['reduction_percent']:+.2f}% |")
    lines.extend([f"| 평균 | {means['operational']:.4f} | {means['legacy']:.4f} | {means['rule']:.4f} | {reduction['legacy']:+.2f}% | {reduction['rule']:+.2f}% |", '',
        '평균 개선률은 세 점수의 평균끼리 비교한 값이며 개별 개선률의 단순 평균과 다를 수 있다.',
        '주 측정창은 둘째 날 24시간이다. 아래 원료 지표는 세 평가 입력의 평균이며 시간은 여러 작업의 시간을 합친 값이다.', '',
        '| 정책 | 트럭 체류 합계(시간) | 본선 중단 합계(시간) | 빈 이동 합계(시간) | 재취급(회) |',
        '|---|---:|---:|---:|---:|'])
    for arm,label in [('operational','새 운영 보상'),('legacy','이전 원화 보상'),('rule','규칙')]:
        raw={k:statistics.mean(p['physical'][arm][k] for p in result['pairs']) for k in KEYS}
        lines.append(f"| {label} | {raw['truck_s']/3600:.2f} | {raw['vessel_s']/3600:.2f} | {raw['empty_s']/3600:.2f} | {raw['rehandles']:.2f} |")
    lines.extend(['', '점수가 좋아도 작업을 미뤄 대기나 잔여가 늘면 진단 성공으로 인정하지 않는다. 아래는 새 모델−비교군이며 양수는 악화다.', '',
        '| 입력 | 비교군 | 원요청 누적시간 변화(트럭·시간) | 트럭 잔여 변화(대) | 본선 잔여 변화(건) | 보호 조건 |',
        '|---|---|---:|---:|---:|---|'])
    for p in result['pairs']:
        for arm,label in [('legacy','이전 원화'),('rule','규칙')]:
            row=p['comparisons'][arm]; d=row['guard_deltas']
            lines.append(f"| {p['eval_seed']} | {label} | {d['original_request_s']/3600:+.2f} | {d['truck_remaining']:+d} | {d['vessel_remaining']:+d} | {'통과' if row['guards_pass'] else '미충족'} |")
    lines.extend(['',
        '원요청 누적시간은 둘째 날 증가분이며, 진입이 미뤄진 트럭도 포함한다. 트럭·본선 잔여는 셋째 날 끝에서 측정했다.',
        '미완료 작업은 현재까지의 시간을 계속 부담한다. 잔여는 0이라고 가정하거나 기록에서 삭제하지 않았다.', '',
        '## 한계와 재현', '',
        '- 네 항의 동일 비중은 초기 설계 가정이다. 정규화만으로 최적 우선순위나 경제적 가치를 증명하지 않는다.',
        '- 이전 보상과는 목적 간 교환 비율도 달라졌다. 정책 변화 전체를 수치 크기 조절의 효과로만 설명할 수 없다.',
        '- 각 모델의 학습은 하루 24회 갱신이다. 이 결과를 수렴한 최종 모델의 우열로 일반화하지 않는다.',
        '- 학습하는 둘째 날은 5000건, 평가 둘째 날은 7500건이다. 두 보상 모두 같은 조건으로 더 높은 부하를 평가했다.',
        '- 평가 입력은 이번 학습과 분리했지만 기존 YR-331-b의 보존 자료를 재사용했다. 미노출 잠금평가가 아니다.',
        '- 원래 5일 입력을 3일 끝에서 끊은 학습이다. 사전 통지된 다음 날 작업도 등록되며 전체 입력을 완주했다고 표시하지 않았다.',
        '- 진입시각 기준 장부의 이연 한계·독립 보정 입력 공급 부족·실제 항만 운영자료 부재는 남아 있다.',
        '- GPU용 배열 보상 계산은 CPU에서 검사했다. 시장·고정 화물 이식이 미완료여서 이번 연구 학습은 CPU 경로로 실행했다.',
        f"- 실행 코드 `{result['execution_commit'][:8]}`, 배열 호환 보완 `b32c344c`. 초기 점검 실패와 수정 후 결과를 모두 보존했다.",
        '- manifest의 normalization은 공통 평가 눈금이다. 실제 학습 모드는 ppo.reward_mode, 실제 보상 계약은 report의 reward_normalization에 기록했다.',
        '- [수식·문헌 검토](../../../docs/research/v6-workload-reward/operational-normalization.md) · [사전등록 원문](prereg-executed.md) · [원값](result.json) · [파일 지문](artifacts.json)',
        '- 문헌: [Dong 등, 크레인 물리 시간 보상](https://link.springer.com/article/10.1007/s40747-025-02174-3), [Toure 등, 목적별 기준 통계](https://www.eurecom.fr/publication/8137/download/comsys-publi-8137.pdf), [SB3 공식 할인 누적값 정규화](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html). 우리 고정 기준·동일 비중은 이를 참고한 자체 설계다.', '',
        '## 예정사항', '',
        '- YR-331-c: 검증 입력에서 부족한 컨테이너 공급의 원인을 바로잡는다. 만든 상황이 물량 보존 계약을 지켜야 후속 판단을 신뢰할 수 있기 때문이다.',
        '- 그 뒤 YR-331-a: 독립 자료에서 안전 수용 상한을 확인한다. 이번 정규화 분모를 최적 혼잡도나 처리 능력으로 오해하지 않기 위해서다.',
        '- 결과를 보고 즉석에서 비중이나 반복 수를 바꾸지 않았다. 보상 구현의 완료와 성능 채택 판정은 구별한다.', ''])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    claims={k:result[k] for k in ('trained_models','update_attempts','updates_applied','frozen_evaluations')}
    claims.update({f'reduction_vs_{k}':round(v,2) for k,v in reduction.items()})
    write_json(root/'reported-values.json',claims)


def audit(root):
    ref = reference_config()
    assert sha(REFERENCE) == sha(root/'reference/operational_reference.json')
    assert ref['scales'] == read(root/'reference/scales.json')['scales']
    assert read(root/'reference/report.json')['optimizer_updates'] == 0
    campaign = read(root/'campaign/campaign.json')
    assert campaign['all_completed'] and len(campaign['completed']) == 15
    reports, manifests, traces = {}, {}, {}
    for run in campaign['completed']:
        assert run['exit_code'] == 0
        name = run['name']; folder = root/'campaign'/name
        report, manifest = read(folder/'report.json'), read(folder/'manifest.json')
        reports[name], manifests[name] = report, manifest
        assert report['protocol_complete'] and report['status'] == 'complete'
        assert report['invariants'] and report['checkpoint_reload_exact'] and report['cf_calls'] == 0
        assert manifest['code']['git_dirty'] is False
        assert manifest['prereg_sha256'] == sha(root/'prereg-executed.md')
        assert read(folder/'container_contract.json')['passed']
        admission = report['admission']
        assert admission['skipped'] == admission['vessel_failed'] == 0
        bundle_name = f"{report['mode']}-{report['seed']}-cost"
        assert manifest['input_sha256'] == sha(Path('outputs/reports/yr331_training')/bundle_name/'seed-data.json.gz')
        train = report['mode'] == 'train'
        assert len(report['updates']) == (24 if train else 0)
        assert report['learning_intervals'] == (1440 if train else 0)
        assert (report['parameter_l2_change'] > 0) if train else (report['parameter_l2_change'] == 0)
        assert report['shaping_reward'] == 0
        assert [x['train'] for x in read(folder/'days.json')] == ([False,True,False] if train else [False]*3)
        saved=[torch.load(folder/name,map_location='cpu',weights_only=True)['policy']
               for name in ('initial.pt','day_01.pt','day_02.pt','day_03.pt','policy-final.pt')]
        equal=lambda a,b: all(torch.equal(a[k],b[k]) for k in a)
        assert equal(saved[0],saved[1]) if train else all(equal(saved[0],s) for s in saved[1:])
        assert equal(saved[2],saved[3]) and equal(saved[3],saved[4])
        for row in report['updates']:
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
        rows = np.asarray(read(folder/'physical-trace.json')['rows'])
        traces[name] = rows
        assert np.isfinite(rows).all() and np.all(np.diff(rows[:,0]) > 0)
        assert np.min(np.diff(rows[:,1:],axis=0)) >= -1e-6
        measured = rows[rows[:,0]==172800][0,1:] - rows[rows[:,0]==86400][0,1:]
        np.testing.assert_allclose(measured,[report['measured_physical'][k] for k in KEYS],rtol=1e-12)
        assert math.isclose(normalized_loss(dict(zip(KEYS,measured)),ref), report['measured_objective'],rel_tol=1e-12)
        if report['reward_mode'] == 'operational':
            assert report['reward_normalization'] == ref and report['config']['reward_scale_krw'] is None
            assert math.isclose(-report['team_reward'],report['objective']-report['initial_objective'],rel_tol=1e-10)
        else:
            assert report['reward_mode'] == 'legacy-krw'
            assert math.isclose(-report['team_reward']*report['config']['reward_scale_krw'],
                                report['cost_krw']-report['initial_cost_krw'],rel_tol=1e-10)
        cargo = report['evaluation_endpoint']['cargo']
        assert cargo['physical_inventory'] == cargo['expected_inventory']
    pairs=[]
    assert len({reports[f'train-{seed}-operational']['initial_parameter_sha256']
                for seed in (9931000,9941000,9951000)}) == 3
    for train_seed,eval_seed in zip((9931000,9941000,9951000),(9961000,9971000,9981000)):
        a,b = [reports[f'train-{train_seed}-{arm}'] for arm in ('operational','legacy')]
        assert a['initial_parameter_sha256'] == b['initial_parameter_sha256']
        rng=[torch.load(root/f'campaign/train-{train_seed}-{arm}/initial.pt',
                        map_location='cpu',weights_only=True)['action_rng'] for arm in ('operational','legacy')]
        assert torch.equal(*rng)
        warm = [traces[f'train-{train_seed}-{arm}'] for arm in ('operational','legacy')]
        np.testing.assert_array_equal(warm[0][warm[0][:,0]<=86400],warm[1][warm[1][:,0]<=86400])
        arms = {arm:reports[f'eval-{eval_seed}-{arm}'] for arm in ('operational','legacy','rule')}
        for arm in ('operational','legacy'):
            trained = reports[f'train-{train_seed}-{arm}']
            evaluated = arms[arm]
            assert trained['final_parameter_sha256'] == evaluated['initial_parameter_sha256'] == evaluated['final_parameter_sha256']
            assert manifests[f'eval-{eval_seed}-{arm}']['checkpoint_sha256'] == sha(root/f'campaign/train-{train_seed}-{arm}/policy-final.pt')
        row = dict(train_seed=train_seed,eval_seed=eval_seed,
            objectives={arm:r['measured_objective'] for arm,r in arms.items()},
            physical={arm:r['measured_physical'] for arm,r in arms.items()},
            guards={arm:guard_values(r) for arm,r in arms.items()}, comparisons={})
        for base in ('legacy','rule'):
            diffs={k:row['guards']['operational'][k]-row['guards'][base][k] for k in row['guards'][base]}
            row['comparisons'][base]=dict(
                reduction_percent=100*(1-row['objectives']['operational']/row['objectives'][base]),
                guard_deltas=diffs, guards_pass=all(v<=1e-6 for v in diffs.values()))
        pairs.append(row)
    means = {arm:statistics.mean(p['objectives'][arm] for p in pairs) for arm in ('operational','legacy','rule')}
    reduction={arm:100*(1-means['operational']/means[arm]) for arm in ('legacy','rule')}
    guards_pass=all(p['comparisons'][arm]['guards_pass'] for p in pairs for arm in ('legacy','rule'))
    numerical_improvement=all(v>0 for v in reduction.values())
    result=dict(task='YR-331-e', implementation_verified=True, registered_runs=15, completed_runs=15,
        trained_models=6, update_attempts=sum(len(r['updates']) for r in reports.values()),
        updates_applied=sum(r['updates_applied'] for r in reports.values()), frozen_evaluations=9,
        normalization=ref,pairs=pairs,mean_objectives=means,mean_reduction_percent=reduction,
        numerical_improvement=numerical_improvement,guards_pass=guards_pass,
        diagnostic_success=numerical_improvement and guards_pass,
        statistical_confirmation=False,real_terminal_claim=False,
        execution_commit=reports['train-9931000-operational']['source']['git_head'])
    assert read(root/'package-check.json')['passed']
    write_json(root/'result.json',result)
    render_report(root,result)
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifacts.json','gates.json','.gitattributes')]
    write_json(root/'artifacts.json',dict(files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]))
    return result


def gates(root,board_commit,remote_ref):
    result=read(root/'result.json')
    hashes={str(root/r['path']):r['sha256'] for r in read(root/'artifacts.json')['files']}
    outcomes=[]
    for folder in [root/'reference',*sorted((root/'campaign').iterdir())]:
        if not folder.is_dir(): continue
        manifest=read(folder/'manifest.json')
        stamp=dict(code=manifest['code'],seeds={'run':[manifest['seed']]},
                   prereg=str(root/'prereg-executed.md'),params=manifest)
        outcomes.append(judge_runtime_evidence(stamp,artifact_hashes=hashes))
    assert all(r.status==GateStatus.PASS for r in outcomes)
    dashboard=audit_dashboard(Path('.'),task_id='YR-331-e',expected_state='done',
        spec_path=Path('.claude/docs/dashboard-task-specs/YR-331-e-operational-normalization.md'),
        evidence_paths=(root/'result.json',root/'report.md'),evidence_commits=(board_commit,),
        remote_ref=remote_ref,pin_commit=board_commit)
    raw={k:result[k] for k in ('trained_models','update_attempts','updates_applied','frozen_evaluations')}
    raw.update({f'reduction_vs_{k}':v for k,v in result['mean_reduction_percent'].items()})
    alignment=judge_claim_alignment(read(root/'reported-values.json'),raw,absolute_tolerance=.005)
    reliability=combine_reliability(outcomes[-1],dashboard,alignment)
    status=GateStatus.INCONCLUSIVE if result['guards_pass'] else GateStatus.FAIL
    performance=GateOutcome('performance',status,
        '원화 없는 운영 목적의 사전등록 세 쌍 진단이며 통계적 확증은 아님',
        (() if result['guards_pass'] else ('원요청 대기 또는 잔여 작업의 비악화 조건 미충족',)),
        dict(diagnostic_success=result['diagnostic_success'],mean_reduction_percent=result['mean_reduction_percent'],
             objective='frozen normalized physical metrics; not economic cost'))
    scenario=judge_scenario_validity(internal_checks={'physical_constraints':True,'ledger_conservation':True},
        flow_checks={'fixed_measurement_window':True},anchors={},continuous_operation=True,
        request_real_terminal_claim=False)
    return attach_common_gates(dict(task='YR-331-e',scope='currency-free normalization diagnostic',
        result_sha256=sha(root/'result.json'),per_run_source_audits=[r.as_dict() for r in outcomes],
        unresolved_existing_input_supply_issue='YR-331-c'),ResearchGateReport(performance,reliability,scenario))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('outputs/reports/yr331_operational'))
    p.add_argument('--board-commit'); p.add_argument('--remote-ref')
    args=p.parse_args()
    if args.board_commit:
        result=gates(args.root,args.board_commit,args.remote_ref)
        write_json(args.root/'gates.json',result)
        print({k:v['status'] for k,v in result['common_gates'].items() if isinstance(v,dict) and 'status' in v})
    else:
        result=audit(args.root)
        print({k:result[k] for k in ('completed_runs','updates_applied','mean_reduction_percent','diagnostic_success')})
