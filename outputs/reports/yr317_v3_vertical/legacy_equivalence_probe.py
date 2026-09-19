"""Bounded legacy-default regression probe. Outputs only in this directory."""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
FROZEN = Path('/home/geonu/yr317-v3-block-recovery-de630ec')
SEED = 99_194_001


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_manifest(root):
    paths = sorted((root/'src/yard_rl/v3').rglob('*.py'))
    paths += [root/'configs/terminals/dgt_armg.yaml']
    result = {}
    for path in paths:
        raw = path.read_bytes()
        result[path.relative_to(root).as_posix()] = dict(
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            lf_sha256=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
    return result


def child(root, label):
    os.chdir(root)
    sys.path.insert(0, str(root/'src'))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from yard_rl.v3.stage.month import DayPlan
    from yard_rl.v3.stage.month_run import run_month
    from yard_rl.v3.reward import reset_rollout_calls, rollout_calls

    before = source_manifest(root)
    observations = []
    days = [DayPlan(index=0, load=21, label='legacy-equivalence',
                    seed=SEED+1000, t0=0.0, n_days=1)]
    start = time.monotonic()
    print(json.dumps(dict(event='started', label=label, seed=SEED)), flush=True)
    reset_rollout_calls()
    result = run_month(seed=SEED, arm='NO_REALLOC', days=days,
        admission_mode='PRESERVE', supply_mode='COUNT_BALANCED',
        capture_requests=True, diagnose_admissions=True, capture_daily=True,
        on_observation=observations.append)
    elapsed = time.monotonic()-start
    payload = dict(result=dataclasses.asdict(result), observations=observations,
                   rollout_calls=rollout_calls())
    dump(OUT/f'{label}-result.json', payload)
    loaded = []
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            try:
                rel = Path(name).resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if rel in before:
                loaded.append(rel)
    after = source_manifest(root)
    changed_loaded = sorted(p for p in set(loaded) if before[p] != after[p])
    meta = dict(root=str(root), seed=SEED, elapsed_s=elapsed, cpu_affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                result_sha256=digest(payload), source_before=before, source_after=after,
                loaded_sources_changed_during_run=changed_loaded,
                observations=len(observations), requested=result.request_summary['requested'],
                rollout_calls=rollout_calls())
    dump(OUT/f'{label}-meta.json', meta)
    print(json.dumps(dict(event='finished', label=label, elapsed_s=elapsed,
                         observations=len(observations), requested=result.request_summary['requested'])), flush=True)


def differences(a, b, path='$'):
    if type(a) is not type(b):
        return [path+': type mismatch']
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        return [item for key in keys for item in
                ([path+'.'+key+': missing'] if key not in a or key not in b
                 else differences(a[key], b[key], path+'.'+key))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path+': length mismatch']
        return [item for i, (x, y) in enumerate(zip(a, b))
                for item in differences(x, y, f'{path}[{i}]')]
    return [] if a == b else [path+': value mismatch']


def compress_artifacts(record):
    """Deterministic gzip files retain full, exactly comparable raw results."""
    artifacts = {}
    for name in ('frozen', 'workspace'):
        path = OUT/f'{name}-result.json'
        raw = path.read_bytes()
        buffer = io.BytesIO()
        with gzip.GzipFile(filename='', mode='wb', fileobj=buffer, mtime=0) as stream:
            stream.write(raw)
        packed = buffer.getvalue()
        assert gzip.decompress(packed) == raw
        compressed = path.with_suffix('.json.gz')
        compressed.write_bytes(packed)
        payload = json.loads(gzip.decompress(packed))
        if name == 'workspace':
            assert payload['result'].pop('environment_manifest') == {}
        normalized = digest(payload)
        assert normalized == record['common_result_sha256']
        meta = OUT/f'{name}-meta.json'
        artifacts[name] = dict(path=compressed.name, compressed_bytes=len(packed),
            compressed_sha256=hashlib.sha256(packed).hexdigest(),
            uncompressed_bytes=len(raw), uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
            normalized_semantic_sha256=normalized, source_manifest_path=meta.name,
            source_manifest_sha256=hashlib.sha256(meta.read_bytes()).hexdigest())
    record['artifacts'] = artifacts
    record['compression'] = 'gzip mtime=0, empty filename; decompressed full results preserved byte-for-byte'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=('frozen', 'workspace'))
    args = parser.parse_args()
    os.sched_setaffinity(0, {19})
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.child:
        child(FROZEN if args.child == 'frozen' else WORKSPACE, args.child)
        return 0

    initial = {name: source_manifest(root) for name, root in [('frozen', FROZEN), ('workspace', WORKSPACE)]}
    common = set(initial['frozen']) & set(initial['workspace'])
    changed = sorted(p for p in common if initial['frozen'][p]['lf_sha256'] != initial['workspace'][p]['lf_sha256'])
    added = sorted(set(initial['workspace'])-set(initial['frozen']))
    removed = sorted(set(initial['frozen'])-set(initial['workspace']))
    allowed_changes = {'src/yard_rl/v3/stage/month_run.py', 'src/yard_rl/v3/stage/month_engine.py'}
    unexpected = sorted(set(changed)-allowed_changes)
    record = dict(schema='yr317.v3.vertical.legacy-equivalence.v1',
        started_at=datetime.now(timezone.utc).isoformat(), seed=SEED, cpu=19, threads=1,
        case=dict(days=1, trucks=21, arm='NO_REALLOC', admission_mode='PRESERVE',
                  supply_mode='COUNT_BALANCED', capture_requests=True,
                  diagnose_admissions=True, capture_daily=True, environment_spec='omitted'),
        source_difference=dict(lf_changed=changed, added=added, removed=removed,
            unexpected_changed=unexpected, basis='Raw and LF-normalized hashes saved in per-run metadata; only intended opt-in stage edits differ.'),
        compared='Complete dataclass MonthResult, every daily observation, and rollout count; only empty workspace result.environment_manifest removed.',
        new_training_runs=0, claim_eligible=False,
        limits='One small default-path regression, not H/V performance or complete equivalence for all policies/inputs.')
    dump(OUT/'legacy-equivalence.json', record | {'state': 'running', 'passed': False})
    if unexpected or removed or any(not p.startswith('src/yard_rl/v3/layouts/') for p in added):
        raise RuntimeError('Unexpected source differences; inspect before claiming equivalence')
    for name in ('frozen', 'workspace'):
        with (OUT/f'legacy-{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--child', name],
                           env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        print(json.dumps(dict(event='probe_done', label=name)), flush=True)
    a, b = [json.loads((OUT/f'{name}-result.json').read_text()) for name in ('frozen', 'workspace')]
    empty_manifest = b['result'].get('environment_manifest') == {}
    if empty_manifest:
        del b['result']['environment_manifest']
    diffs = differences(a, b)
    metas = {name: json.loads((OUT/f'{name}-meta.json').read_text()) for name in ('frozen', 'workspace')}
    checks = dict(empty_new_environment_manifest=empty_manifest, exact_serialized_results=digest(a)==digest(b),
                  no_differences=not diffs, expected_request_count=all(m['requested']==21 for m in metas.values()),
                  source_stable=all(not m['loaded_sources_changed_during_run'] for m in metas.values()),
                  no_rollouts=all(m['rollout_calls']==0 for m in metas.values()),
                  cpu_one_thread=all(m['cpu_affinity']==[19] and m['torch_threads']==m['interop_threads']==1 for m in metas.values()))
    record.update(state='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                  passed=all(checks.values()), checks=checks, differences=diffs[:30],
                  common_result_sha256=digest(a), workspace_normalized_result_sha256=digest(b),
                  runs={name: {k:v for k,v in m.items() if k not in ('source_before','source_after')} for name,m in metas.items()})
    if record['passed']:
        compress_artifacts(record)
    dump(OUT/'legacy-equivalence.json', record)
    print(json.dumps(dict(event='equivalence', passed=record['passed'], differences=diffs[:5])), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
