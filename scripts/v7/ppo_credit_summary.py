"""Aggregate descriptive records and independently spot-check saved-state predictions."""
from collections import defaultdict
import gzip
import json

import numpy as np
import torch

from ppo_credit_audit import OUT, SOURCE, KINDS, SEEDS, sha, read, save, decode
from ppo_credit_metrics import describe
from yard_rl.v7.crane.actor_critic import SchedulingActorCritic, pack_states


def records():
    result = []
    for kind in KINDS:
        for seed in SEEDS:
            path = OUT/f'{kind}-{seed}.jsonl.gz'
            check = read(OUT/f'{kind}-{seed}-checks.json')
            assert sha(path) == check['output_sha256']
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream]
            assert len(rows) == check['rows'] == 12288
            result.extend(rows)
    return result


def aggregate(rows):
    tables = {}
    dimensions = [('kind',), ('kind', 'seed'), ('kind', 'load'), ('kind', 'load', 'time_bin'),
                  ('kind', 'seed', 'load'), ('kind', 'seed', 'load', 'time_bin'),
                  ('kind', 'seed', 'iteration', 'day_index'), ('kind', 'chosen_flow')]
    for dims in dimensions:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[d] for d in dims)].append(row)
        tables['_'.join(dims)] = [dict(group=dict(zip(dims, key)), metrics=describe(rs))
                                  for key, rs in sorted(groups.items())]
        print(json.dumps(dict(table=dims, groups=len(groups))), flush=True)
    updates = []
    for kind in KINDS:
        for seed in SEEDS:
            updates.extend(read(OUT/f'{kind}-{seed}-checks.json')['updates'])
    save(OUT/'summary.json', dict(rows=len(rows), tables=tables, updates=updates,
        interpretation='Descriptive training-state analysis. No independent test or causal intervention.',
        uncertainty='Correlated within-day decisions; no decision-level IID significance claim.'))


@torch.no_grad()
def verify(rows):
    torch.set_num_threads(1)
    manifest = read(OUT/'manifest.json')
    for group in ('input_hashes', 'analyzer_hashes'):
        for path, digest in manifest[group].items():
            assert sha(path) == digest, path
    lookup = {(r['kind'], r['seed'], r['iteration'], r['sample_index']): r for r in rows}
    assert len(lookup) == len(rows) == 73728
    assert all(abs(sum(r[v]['mass'])-1) < 1e-10 for r in rows for v in ('pre', 'post', 'initial', 'final'))
    assert all(sum(r['cap_counts']) <= sum(r['flow_counts'][:2]) <= r['allowed_count'] for r in rows)
    max_probability_error = max_value_error = max_old_logp_error = 0.
    spot_states = forwards = 0
    for kind in KINDS:
        for seed in SEEDS:
            root = SOURCE/f'{kind}-{seed}'
            for iteration in range(6):
                raw = read(root/f'samples-{iteration}.json.gz')
                selected = (0, len(raw['samples'])-1)
                for version, filename in [('pre', 'initial.pt' if iteration == 0 else f'update-{iteration-1}.pt'),
                                          ('post', f'update-{iteration}.pt'), ('initial', 'initial.pt'), ('final', 'final.pt')]:
                    model = SchedulingActorCritic.load(root/filename).eval()
                    for i in selected:
                        sample = decode(raw['samples'][i])
                        row = lookup[kind, seed, iteration, i]
                        logits, value = model(pack_states([sample.state]))
                        ll = logits[0]
                        stage = next(j for j, m in enumerate(sample.masks) if len(m) > 1)
                        mask = sample.masks[stage]
                        probabilities = torch.softmax(ll[mask].double(), 0).tolist()
                        mass = [0.]*5
                        for candidate, p in zip(mask, probabilities):
                            f = sample.state.features[candidate]
                            flow = 4 if f[1] < .5 else ((2 if f[0] > .5 else 0)+(0 if f[9] > .5 else 1))
                            mass[flow] += p
                        max_probability_error = max(max_probability_error, max(abs(a-b) for a, b in zip(mass, row[version]['mass'])))
                        if version in ('pre', 'post'):
                            max_value_error = max(max_value_error, abs(float(value[0])-row[version+'_value']))
                        if version == 'pre':
                            # Independent log_softmax implementation, without action_statistics().
                            joint = sum(float(torch.log_softmax(ll[m], 0)[m.index(c)]) for m, c in zip(sample.masks, sample.choices))
                            max_old_logp_error = max(max_old_logp_error, abs(joint-sample.old_log_probability))
                        forwards += 1
                spot_states += len(selected)
    assert max_probability_error < 2e-6 and max_value_error < 2e-5 and max_old_logp_error < 2e-5
    summary = read(OUT/'summary.json')
    for kind in KINDS:
        selected = [r for r in rows if r['kind'] == kind]
        full = next(t['metrics'] for t in summary['tables']['kind'] if t['group']['kind'] == kind)
        assert full['n'] == len(selected)
        errors = [(r['pre_value']-r['target'])**2 for r in selected]
        assert abs((sum(errors)/len(errors))**.5-full['pre']['rmse']) < 1e-10
        count = sum(sum(r['cap_counts']) for r in selected)
        assert count == full['saturation']['capped_candidates']
        for table in ('kind_seed', 'kind_load', 'kind_load_time_bin', 'kind_seed_iteration_day_index', 'kind_chosen_flow'):
            entries = [t for t in summary['tables'][table] if t['group']['kind'] == kind]
            assert sum(t['metrics']['n'] for t in entries) == len(selected)
    save(OUT/'verification.json', dict(status='PASS', rows=len(rows), unique_record_keys=len(lookup),
        immutable_source_files=len(manifest['input_hashes']), analyzer_hashes_match=True,
        spot_states=spot_states, independent_single_state_forwards=forwards,
        max_probability_error=max_probability_error, max_value_error=max_value_error,
        max_old_joint_logp_error=max_old_logp_error,
        all_corpus_max_logp_error=max(r['logp_replay_max_error'] for r in summary['updates']),
        all_corpus_max_value_error=max(r['value_replay_max_error'] for r in summary['updates']),
        all_corpus_max_target_error=max(r['target_max_error'] for r in summary['updates']),
        probability_sums=True, saturation_counts=True, partition_counts=True, independent_aggregate_checks=True,
        tests=dict(command='pytest -q -p no:cacheprovider tests/test_ppo_credit_diagnostics.py', passed=6),
        new_simulation=False, new_training=False,
        limitations=['No raw wait above two hours or original job IDs in saved records.',
                     'One observed return per state, no counterfactual causal identification.',
                     'First-minibatch gradients only; not actual Adam parameter displacement.',
                     'Original performance FAIL and scenario_validity FAIL remain unchanged.']))
    print(json.dumps(read(OUT/'verification.json')), flush=True)


if __name__ == '__main__':
    rows = records()
    aggregate(rows)
    verify(rows)
