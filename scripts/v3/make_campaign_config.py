"""Assemble the confirmatory campaign config from its frozen parts (YR-318).

Everything this writes is mechanical: seeds come from the bank, hashes are read
off disk, and the design (columns, comparisons, primary) is transcribed from the
preregistration by the flags below. Writing it by hand invites a mistyped hash
in a run that costs a thousand CPU-hours, so it is generated and re-checked.

The config is refused unless every declared file exists and every training run
supplies a checkpoint, a starting checkpoint and a manifest, so an unfinished
training run cannot silently become an evaluated arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def training_run(folder):
    """One finished training run: weights, starting weights and manifest."""
    folder = Path(folder)
    weights = folder / 'ckpt_029.pt'
    for required in (weights, folder / 'ckpt_init.pt', folder / 'training_manifest.json'):
        if not required.exists():
            raise SystemExit(f'Training run is incomplete: {required} is missing')
    manifest = json.loads((folder / 'training_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('admission_mode') != 'PRESERVE' or manifest.get('supply_mode') != 'COUNT_BALANCED':
        raise SystemExit(f'{folder}: trained under a contract the campaign does not evaluate')
    if manifest.get('candidate_pruning') != 'feasible_first':
        raise SystemExit(f'{folder}: trained on the unrepaired engine')
    return dict(folder=relative(folder), checkpoint=relative(weights),
                checkpoint_sha256=sha(weights),
                init_checkpoint=relative(folder / 'ckpt_init.pt'),
                init_checkpoint_sha256=sha(folder / 'ckpt_init.pt'),
                manifest=relative(folder / 'training_manifest.json'),
                manifest_sha256=sha(folder / 'training_manifest.json'),
                environment_seed=manifest.get('seed'), init_seed=manifest.get('init_seed'))


def build(args):
    bank = Path(args.bank)
    summary = json.loads((bank / 'summary.json').read_text(encoding='utf-8'))
    if not summary.get('passed'):
        raise SystemExit('Seed bank did not pass its own verification')
    seeds = [row['seed'] for row in summary['rows']]
    runs = [training_run(folder) for folder in args.training]
    if len(runs) < 2:
        raise SystemExit('Training-run stability needs at least two runs')

    labels = [f'RL_TIME_n{i + 1}' for i in range(len(runs))]
    specs = [dict(label='NO_REALLOC', arm='NO_REALLOC'),
             dict(label='SLOT_LL', arm='SLOT_LL'),
             dict(label='SPACE_TIME_LL', arm='SPACE_TIME_LL')]
    for label, run in zip(labels, runs):
        specs.append(dict(label=label, arm='RL_TIME', checkpoint=run['checkpoint'],
                          checkpoint_sha256=run['checkpoint_sha256']))
    specs.append(dict(label='RL_n1', arm='RL', checkpoint=runs[0]['checkpoint'],
                      checkpoint_sha256=runs[0]['checkpoint_sha256']))
    specs.append(dict(label='RL_NOVETO_n1', arm='RL_NOVETO', checkpoint=runs[0]['checkpoint'],
                      checkpoint_sha256=runs[0]['checkpoint_sha256']))

    mean = 'RL_TIME_mean'
    comparisons = [['SLOT_LL', mean]] + [['SLOT_LL', label] for label in labels] + [
        ['NO_REALLOC', mean], ['NO_REALLOC', 'SLOT_LL'], ['SLOT_LL', 'SPACE_TIME_LL'],
        ['SPACE_TIME_LL', 'RL_n1'], [mean, 'RL_n1'], ['RL_n1', 'RL_NOVETO_n1']]

    files = {relative(args.prereg): sha(args.prereg),
             relative(bank / 'summary.json'): sha(bank / 'summary.json')}
    for run in runs:
        files[run['checkpoint']] = run['checkpoint_sha256']
        files[run['manifest']] = run['manifest_sha256']
    for seed in seeds:
        item = Path(args.supply_inputs) / f'seed-{seed}.json'
        if not item.exists():
            raise SystemExit(f'Supply-corrected input missing: {item}')
        files[relative(item)] = sha(item)

    cfg = dict(schema='yr318.confirmatory-campaign.v1', seeds=seeds, arm_specs=specs,
               checkpoint=runs[0]['checkpoint'], checkpoint_sha256=runs[0]['checkpoint_sha256'],
               candidate_pruning='feasible_first', measure_latency=True,
               prereg=relative(args.prereg), bank=relative(bank),
               supply_inputs=relative(args.supply_inputs),
               comparisons=comparisons, derived_columns={mean: labels},
               primary_comparisons=[f'28d:SLOT_LL-{mean}'], bootstrap_seed=args.bootstrap_seed,
               training_runs=runs, workers_max=args.workers_max, worker_budget_gib=3,
               cpu_limit=args.cpu_limit,
               solver_sha256=sha(ROOT / 'src/yard_rl/v3/stage/supply_plan.py'), files=files)

    from independent_eval_checks import campaign_specs
    resolved_seeds, resolved_specs = campaign_specs(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + '\n',
                              encoding='utf-8')
    print(json.dumps(dict(config=str(args.out), months=len(resolved_seeds),
        columns=[item['label'] for item in resolved_specs],
        runs=len(resolved_seeds) * len(resolved_specs),
        primary=cfg['primary_comparisons'], config_sha256=sha(args.out)), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', required=True, type=Path, help='new-band seed bank run directory')
    p.add_argument('--supply-inputs', required=True, type=Path)
    p.add_argument('--training', required=True, nargs='+', type=Path,
                   help='finished training run directories, in declared order')
    p.add_argument('--prereg', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--bootstrap-seed', type=int, default=9_900_727)
    p.add_argument('--workers-max', type=int, default=16)
    p.add_argument('--cpu-limit', type=int, default=20)
    build(p.parse_args())


if __name__ == '__main__':
    main()
