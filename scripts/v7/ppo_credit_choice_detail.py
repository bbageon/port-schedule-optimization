"""Supplementary same-state choice audit after primary aggregates (still no fitting).

Distinguish one-crane choices from joint-construction ordering and soft probabilities
from deterministic argmax. This extension is descriptive, not a new causal test.
"""
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch

from ppo_credit_audit import OUT, SOURCE, KINDS, SEEDS, read, save, sha, decode, infer
from ppo_credit_metrics import first_choice, flow_id
from yard_rl.v7.crane.actor_critic import SchedulingActorCritic


def main():
    torch.set_num_threads(1)
    aggregates = defaultdict(list)
    examples = {}
    early_high = defaultdict(lambda: [0, 0])
    for kind in KINDS:
        for seed in SEEDS:
            root = SOURCE/f'{kind}-{seed}'
            history = read(root/'training.json')['history']
            models = [SchedulingActorCritic.load(root/f'{s}.pt').eval() for s in ('initial', 'final')]
            for iteration in range(6):
                source = root/f'samples-{iteration}.json.gz'
                corpus = read(source)
                samples = [decode(x) for x in corpus['samples']]
                prediction = [infer(m, samples) for m in models]
                boundary = history[iteration]['days'][0]['accounting']['samples']
                for i, sample in enumerate(samples):
                    day = history[iteration]['days'][int(corpus['indices'][i] >= boundary)]
                    stage, mask, chosen = first_choice(sample)
                    cranes = {int(sample.state.cranes[c]) for c in mask}
                    groups = np.array([flow_id(sample.state.features[c]) for c in mask])
                    pair = []
                    detail = dict(kind=kind, seed=seed, load=day['case']['load'],
                                  single_crane=len(cranes) == 1, mixed_flow=0 in groups and 1 in groups)
                    for name, result in zip(('initial', 'final'), prediction):
                        logits = result[0][i][mask].double()
                        probs = torch.softmax(logits, 0).numpy()
                        top = int(logits.argmax())
                        ordered = logits.sort(descending=True).values
                        detail[name] = dict(normalized_entropy=float(-(probs*np.log(probs)).sum()/math.log(len(mask))),
                            top_probability=float(probs[top]), top_vs_uniform=float(probs[top]*len(mask)),
                            top_logit_gap=float(ordered[0]-ordered[1]), top_flow=int(groups[top]),
                            inbound_conditional=(float(probs[groups == 0].sum()/probs[groups < 2].sum())
                                                 if detail['mixed_flow'] else None))
                        pair.append(probs)
                    aggregates[kind, seed, 0].append(detail)
                    aggregates[kind, seed, detail['load']].append(detail)
                    if detail['single_crane'] and detail['mixed_flow'] and detail['initial']['top_flow'] == 1 and detail['final']['top_flow'] == 0:
                        key = kind, seed, detail['load']
                        if key not in examples:
                            examples[key] = dict(source=source.as_posix(), source_sha256=sha(source), sample_index=i,
                                stage=stage, mask=mask, initial_probability=pair[0].tolist(), final_probability=pair[1].tolist(),
                                features=sample.state.features[mask].tolist(),
                                operations=sample.state.operations[mask].tolist(), **detail)
                print(f'{kind} {seed} update {iteration}: choice detail', flush=True)
    tables = []
    for key, rows in sorted(aggregates.items()):
        singles = [r for r in rows if r['single_crane']]
        mixed = [r for r in singles if r['mixed_flow']]
        stats = dict(kind=key[0], seed=key[1], load=key[2], n=len(rows), single_crane_n=len(singles), mixed_single_crane_n=len(mixed))
        for version in ('initial', 'final'):
            stats[version] = dict(mean_normalized_entropy=float(np.mean([r[version]['normalized_entropy'] for r in rows])),
                median_top_vs_uniform=float(np.median([r[version]['top_vs_uniform'] for r in rows])),
                median_top_logit_gap=float(np.median([r[version]['top_logit_gap'] for r in rows])),
                mixed_single_crane_inbound_probability=float(np.mean([r[version]['inbound_conditional'] for r in mixed])),
                mixed_single_crane_top_inbound_fraction=float(np.mean([r[version]['top_flow'] == 0 for r in mixed])))
        tables.append(stats)
    save(OUT/'choice-detail.json', dict(scope='Post-primary descriptive refinement of same-state rankings/probabilities',
        motivation='Separate free scheduling choices from multi-crane construction order; quantify soft/argmax difference.',
        input_manifest_sha256=sha(OUT/'manifest.json'), script_sha256=sha(Path(__file__)),
        rows=tables, examples=list(examples.values()), example_rule='first source-ordered single-crane outbound-to-inbound flip per seed/load',
        examples_are_illustrative=True, new_fitting=False))


if __name__ == '__main__':
    main()
