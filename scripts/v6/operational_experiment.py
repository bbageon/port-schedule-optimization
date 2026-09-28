"""Pinned three-pair physical-normalization pilot; retain all failed runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from yard_rl.v6.ppo.checkpoint import load_policy, save_checkpoint
from yard_rl.v6.ppo.journal import RunJournal, write_json
from yard_rl.v6.ppo.model import BlockPolicy
from yard_rl.v6.ppo.provenance import code_stamp, file_sha256
from yard_rl.v6.ppo.runtime import DebugStop, PPOConfig, PPORuntime
from yard_rl.v6.ppo.workload_experiment import ReferenceRuntime
from yard_rl.v6.reward.counterfactual import rollout_calls
from yard_rl.v6.reward.operational import KEYS, normalized_loss, reference_config, runtime_totals
from yard_rl.v6.stage.month import plan_days
from yard_rl.v6.stage.month_run import run_month
from yard_rl.v6.stage.seed_bundle import load_seed_bundle

PREREG = Path('.claude/docs/dashboard-task-specs/YR-331-e-operational-normalization.md')


def parameter_hash(policy):
    h = hashlib.sha256()
    for name, value in policy.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def original_request_totals(rt, document, t):
    total, unfinished, requested = 0., 0, 0
    for order in document['orders']:
        request = order['in_out_reserve_s']
        if request > t: continue
        rec = rt.bridge.records[order['doc_key']]
        end = min(t,rec.gate_out_s) if rec.gate_out_s is not None else t
        total += max(0., end-request)
        requested += 1
        unfinished += int(rec.gate_out_s is None or rec.gate_out_s > t)
    return dict(original_request_s=total, requested=requested, unfinished=unfinished)


def run_one(args):
    torch.set_num_threads(1)
    stamp = code_stamp()
    document, audit = load_seed_bundle(args.seed_bundle, expected_sha256=file_sha256(args.seed_bundle))
    days = plan_days(args.seed, loads=[row['load'] for row in document['days']])
    config = PPOConfig(reward_mode='legacy-krw' if args.arm == 'legacy' else 'operational')
    reference = reference_config(gamma=config.gamma,time_unit_s=config.time_unit_s)
    torch.manual_seed(args.init_seed)
    policy = load_policy(args.checkpoint) if args.checkpoint else BlockPolicy()
    before = {k:v.clone() for k,v in policy.state_dict().items()}
    smoke = args.mode == 'smoke'
    training = args.mode in ('train','smoke')
    window = (0.,7200.) if smoke else (86400.,172800.)
    stop = 7200. if smoke else 259200. if training else None
    endpoint = 7200. if smoke else 259200.
    expected_updates = 2 if smoke else 24 if training else 0
    out = Path(args.output)
    manifest = dict(code=stamp, seed=args.seed, init_seed=args.init_seed, ppo=asdict(config),
        arguments=vars(args), prereg_sha256=file_sha256(PREREG),
        input_sha256=file_sha256(args.seed_bundle), input_audit=audit,
        learning_window_s=list(window), learning_days=(window[1]-window[0])/86400 if training else 0,
        stop_s=stop,
        action_mode='sample-all-days' if args.arm != 'rule' else 'KEEP+SF_SPT',
        normalization=reference, initial_parameter_sha256=parameter_hash(policy),
        checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None)
    journal = RunJournal(out,days,manifest)
    snapshots, trace = {}, []
    def boundary(rt):
        raw = rt.physical if rt.physical is not None else runtime_totals(rt,rt.time_s)
        if not trace or trace[-1][0] != rt.time_s:
            trace.append([rt.time_s]+[raw[k] for k in KEYS])
        if rt.time_s in (0.,*window,endpoint):
            snapshots[str(int(rt.time_s))] = dict(physical=raw, objective=normalized_loss(raw,reference),
                request=original_request_totals(rt,document,rt.time_s),
                cargo=rt.mbt.cargo_report(), updates=len(rt.updates))
            write_json(out/'snapshots.json',snapshots)
        journal.boundary(rt)
    cls = ReferenceRuntime if args.arm == 'rule' else PPORuntime
    rt = cls(policy,config=config,seed=args.seed,training=training,sample_actions=True,
        stop_s=stop,learning_window_s=window,
        on_boundary=boundary,on_update=journal.update)
    save_checkpoint(out/'initial.pt',rt)
    started, calls = time.perf_counter(), rollout_calls()
    try:
        try:
            result = run_month(seed=args.seed,days=days,ppo=rt,seed_data=document,
                on_admission=journal.admission,on_day=journal.day_report,
                on_container_contract=journal.container_contract)
            write_json(out/'month-result.json',asdict(result))
        except DebugStop:
            if not training or rt.time_s != stop: raise
        for sim in rt.mbt.blocks.values(): sim.check_invariants()
        assert rollout_calls() == calls
        assert len(rt.updates) == expected_updates
        change = float(sum((value-before[k]).square().sum() for k,value in policy.state_dict().items()).sqrt())
        if training: assert change > 0 and any(row['minibatches'] > 0 for row in rt.updates)
        else: assert change == 0
        assert all(np.isfinite(v) for row in rt.updates for v in row.values() if isinstance(v,float))
        journal.save_admissions()
        assert journal.admissions['skipped'] == journal.admissions['vessel_failed'] == 0
        # The existing announcer admits notified future work before its service day.
        expected_admitted = (sum((max(0.,e['arrival_s']-e['lead_s'])//60)*60 <= stop
                                 for e in document['schedule']) if training else len(document['schedule']))
        assert journal.admissions['admitted'] == expected_admitted
        if args.arm == 'rule': assert rt.rule_exceptions['n'] == 0
        save_checkpoint(out/'policy-final.pt',rt)
        restored = load_policy(out/'policy-final.pt')
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items())
        PPORuntime(restored,config=config,training=False)
        write_json(out/'physical-trace.json',dict(keys=KEYS,rows=trace))
        start, end = [snapshots[str(int(t))] for t in window]
        report = rt.report() | dict(status='complete', protocol_complete=True,
            whole_input_completed=not training, measured_physical={k:end['physical'][k]-start['physical'][k] for k in KEYS},
            measured_objective=end['objective']-start['objective'],
            measured_original_request_s=end['request']['original_request_s']-start['request']['original_request_s'],
            evaluation_endpoint=snapshots[str(int(endpoint))], updates_applied=sum(x['minibatches']>0 for x in rt.updates),
            mode=args.mode,arm=args.arm,seed=args.seed,source=stamp,initial_parameter_sha256=manifest['initial_parameter_sha256'],
            final_parameter_sha256=parameter_hash(policy),parameter_l2_change=change,
            admission=journal.admissions,invariants=True,checkpoint_reload_exact=True,cf_calls=0,
            elapsed_s=time.perf_counter()-started,performance_claim=False)
        write_json(out/'report.json',report)
        write_json(out/'status.json',dict(status='complete',time_s=rt.time_s,updates=len(rt.updates)))
        print(json.dumps(dict(status='complete',arm=args.arm,mode=args.mode,updates=len(rt.updates),measured_objective=report['measured_objective'])),flush=True)
    except BaseException as exc:
        journal.fail(exc,rt)
        write_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


def campaign(output):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    jobs, completed = [], []
    base=Path('outputs/reports/yr331_training').resolve()
    def invoke(job):
        name, args=job
        command=[sys.executable,__file__,*args,'--output',str(root/name)]
        with (root/f'{name}.log').open('w',encoding='utf-8') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        return dict(name=name,exit_code=rc,command=command)
    for i,seed in enumerate((9931000,9941000,9951000)):
        for arm in ('operational','legacy'):
            name=f'train-{seed}-{arm}'
            jobs.append((name,['--mode','train','--arm',arm,'--seed',str(seed),'--init-seed',str(202609281+i),
                              '--seed-bundle',str(base/f'train-{seed}-cost/seed-data.json.gz')]))
    def execute(batch):
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(invoke,job) for job in batch]):
                row=future.result(); completed.append(row)
                write_json(root/'progress.json',dict(completed=completed,registered_runs=15))
                print(json.dumps(row),flush=True)
    execute(jobs)
    evals=[]
    for i,seed in enumerate((9961000,9971000,9981000)):
        for arm in ('operational','legacy','rule'):
            command=['--mode','eval','--arm',arm,'--seed',str(seed),'--init-seed',str(202609281+i),
                     '--seed-bundle',str(base/f'eval-{seed}-cost/seed-data.json.gz')]
            if arm!='rule':
                checkpoint=root/f'train-{9931000+i*10000}-{arm}/policy-final.pt'
                if not checkpoint.exists(): continue  # Missing comparison is reported, never substituted.
                command.extend(['--checkpoint',str(checkpoint)])
            evals.append((f'eval-{seed}-{arm}',command))
    execute(evals)
    write_json(root/'campaign.json',dict(completed=completed,registered_runs=15,
        all_completed=len(completed)==15 and all(row['exit_code']==0 for row in completed)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['train','eval','smoke'])
    p.add_argument('--arm',choices=['operational','legacy','rule'])
    p.add_argument('--seed',type=int)
    p.add_argument('--init-seed',type=int)
    p.add_argument('--seed-bundle')
    p.add_argument('--checkpoint')
    a=p.parse_args()
    campaign(a.output) if a.campaign else run_one(a)
