"""Exact same-input training replay followed by a read-only residual snapshot."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import os
import subprocess

from capture_residual_state import capture, json_default


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def comparable_report(report):
    return {k:v for k,v in report.items() if k != 'code'}


def main():
    from yard_rl.v5.ppo.continuous import run_continuous
    from yard_rl.v5.ppo.provenance import code_stamp
    from yard_rl.v5.ppo.runtime import PPOConfig
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original-run',type=Path,required=True)
    ap.add_argument('--seed-bundle',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--debug-load',type=int)
    args = ap.parse_args()
    manifest,expected = read(args.original_run/'manifest.json'),read(args.original_run/'report.json')
    require(expected['state']=='completed','Only replay a completed reference')
    require(not args.output.exists(),'Choose a new output directory')
    stamp = code_stamp()
    require(stamp['source_sha256']==manifest['code']['source_sha256'],'Policy/environment source changed')
    require(stamp['torch']==manifest['code']['torch'] and stamp['numpy']==manifest['code']['numpy'],
            'Numerical environment changed')
    require(len(os.sched_getaffinity(0))==1,'Replay must be restricted to one CPU')
    require(sha(args.seed_bundle)==manifest['fixed_seed']['sha256'],'Input bytes differ')
    original_hashes = {p.name:sha(p) for p in args.original_run.iterdir() if p.is_file()}
    runtime,report = run_continuous(output=args.output,seed=manifest['seed'],n_days=len(manifest['days']),
        load=args.debug_load,config=PPOConfig(**manifest['ppo']),seed_bundle=args.seed_bundle,
        seed_sha256=manifest['fixed_seed']['sha256'])
    new_manifest = read(args.output/'manifest.json')
    setting_keys = ('seed','days','ppo','learning_window_s','initial_occupancy','action_mode','fixed_seed')
    settings_match = all(new_manifest[k]==manifest[k] for k in setting_keys)
    report_match = comparable_report(report)==comparable_report(expected)
    days_match = read(args.output/'days.json')==read(args.original_run/'days.json')
    checkpoints = {p.name:sha(p)==sha(args.output/p.name) for p in args.original_run.glob('*.pt')}
    # Preserve the observed state even if reference replay equivalence fails.
    before = runtime.mbt.cargo_report()
    rng_before = runtime.action_rng.get_state().clone()
    snapshot = capture(runtime)
    require(before==runtime.mbt.cargo_report(),'Observer changed cargo counters')
    require(bool((rng_before==runtime.action_rng.get_state()).all()),'Observer consumed action randomness')
    payload = (json.dumps(snapshot,ensure_ascii=False,sort_keys=True,default=json_default)+'\n').encode('utf-8')
    target = args.output/'residual-state.json.gz'
    with target.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as stream:
            stream.write(payload)
    original_preserved = all(sha(args.original_run/name)==value for name,value in original_hashes.items())
    matching = settings_match and report_match and days_match and all(checkpoints.values()) and original_preserved
    summary = dict(state='verified' if matching else 'mismatch',task='YR-306',
        captured_utc=datetime.now(timezone.utc).isoformat(),original_run=str(args.original_run),
        observer_code=stamp,original_code=manifest['code'],settings_match=settings_match,
        report_match=report_match,days_match=days_match,checkpoints_match=checkpoints,
        original_files_preserved=original_preserved,original_artifacts_sha256=original_hashes,
        residual_snapshot_sha256=sha(target),residual_count=len(snapshot['jobs']),
        categories=snapshot['categories'],categories_by_flow=snapshot['categories_by_flow'],
        claim_scope='NO_PERFORMANCE_CLAIM',extension_or_policy_change=False)
    (args.output/'residual-diagnosis.json').write_bytes((json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    require(matching,'Replay differs from original; snapshot is NOT original-run evidence')


if __name__ == '__main__':
    main()
