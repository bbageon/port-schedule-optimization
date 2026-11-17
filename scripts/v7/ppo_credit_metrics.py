"""Pure, descriptive measurements for the saved-experience YR-332-ax audit."""
from collections import defaultdict
import math

import numpy as np
import torch

FLOWS = ('inbound', 'outbound', 'discharge', 'loading', 'other')
VERSIONS = ('pre', 'post', 'initial', 'final')


def flow_id(features):
    if features[1] < .5:
        return 4
    return (2 if features[0] > .5 else 0) + (0 if features[9] > .5 else 1)


def first_choice(sample):
    for stage, (mask, choice) in enumerate(zip(sample.masks, sample.choices)):
        if len(mask) > 1:
            return stage, mask, choice
    raise ValueError('Stored sample has no nontrivial choice')


def normalized_advantages(samples, seed):
    """Match the original update's float32 reduction order, then undo its permutation."""
    rng = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(samples), generator=rng)
    target = torch.tensor([samples[i].target for i in permutation])
    value = torch.tensor([samples[i].old_value for i in permutation])
    raw = target - value
    z = (raw-raw.mean()) / raw.std(unbiased=False).clamp_min(1e-8)
    restored = torch.empty_like(z)
    restored[permutation] = z
    return restored.tolist(), permutation, rng


def choice_metrics(logits, mask, chosen, features):
    """All metrics condition on exactly the same legal first nontrivial mask."""
    ll = logits[mask].double()
    probability = torch.softmax(ll, 0).numpy()
    fs = features[mask].numpy()
    groups = np.array([flow_id(f) for f in fs])
    truck = groups < 2
    capped = truck & (fs[:, 2] >= 2.-1e-6)
    position = mask.index(chosen)
    scores = ll.numpy()
    tied = np.isclose(scores, scores[position], rtol=0., atol=1e-7)
    rank = 1 + int((scores > scores[position]+1e-7).sum()) + (int(tied.sum())-1)/2
    top = int(np.argmax(scores))  # same first-index tie convention as deterministic argmax
    return dict(mass=[float(probability[groups == f].sum()) for f in range(5)],
                capped_mass=float(probability[capped].sum()),
                chosen_probability=float(probability[position]), chosen_rank=rank,
                top_flow=int(groups[top]), top_capped=bool(capped[top]),
                top_candidate=int(mask[top]),
                top_ties=int(np.isclose(scores, scores[top], rtol=0., atol=1e-7).sum())), probability


def mask_description(sample, mask, chosen):
    fs = sample.state.features[mask].numpy()
    groups = np.array([flow_id(f) for f in fs])
    capped = (groups < 2) & (fs[:, 2] >= 2.-1e-6)
    # Equal features do not imply equal Transformer scores: operation/crane graph context can differ.
    by_crane = defaultdict(list)
    for j, candidate in enumerate(mask):
        if capped[j]:
            by_crane[int(sample.state.cranes[candidate])].append(tuple(fs[j].tolist()))
    duplicates = sum(len(values)-len(set(values)) for values in by_crane.values())
    return dict(candidate_count=len(sample.state.features), allowed_count=len(mask),
                flow_counts=[int((groups == f).sum()) for f in range(5)],
                cap_counts=[int((capped & (groups == f)).sum()) for f in range(2)],
                chosen_flow=int(groups[mask.index(chosen)]), chosen_capped=bool(capped[mask.index(chosen)]),
                identical_capped_features_same_crane=duplicates)


def fit_metrics(rows, value_key):
    y = np.array([r['target'] for r in rows], dtype=float)
    v = np.array([r[value_key] for r in rows], dtype=float)
    error = v-y
    variance = float(np.var(y))
    return dict(n=len(rows), target_mean=float(y.mean()), prediction_mean=float(v.mean()),
                bias=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                explained_variance=(1.-float(np.var(error))/variance if variance > 1e-12 else None))


def between_group_share(rows, group='day_index', strata=None):
    """Within-stratum SS fraction attributable to observed group means, not causal variance."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[strata] if strata else 0].append(r)
    between = total = 0.
    for rs in buckets.values():
        values = np.array([r['raw_advantage'] for r in rs])
        mean = float(values.mean())
        total += float(np.sum((values-mean)**2))
        groups = defaultdict(list)
        for r in rs:
            groups[r[group]].append(r['raw_advantage'])
        between += sum(len(v)*(float(np.mean(v))-mean)**2 for v in groups.values())
    return dict(between_ss=between, total_ss=total, share=between/total if total > 1e-12 else None)


def describe(rows):
    assert rows
    advantage = np.array([r['advantage'] for r in rows])
    result = dict(n=len(rows), pre=fit_metrics(rows, 'pre_value'), post=fit_metrics(rows, 'post_value'),
                  positive_advantage_fraction=float(np.mean(advantage > 0)),
                  mean_advantage=float(advantage.mean()))
    fc = np.array([r['flow_counts'] for r in rows]).sum(0)
    cc = np.array([r['cap_counts'] for r in rows]).sum(0)
    truck_candidates = int(fc[:2].sum())
    cap_candidates = int(cc.sum())
    result['saturation'] = dict(truck_candidates=truck_candidates, capped_candidates=cap_candidates,
        candidate_fraction=cap_candidates/truck_candidates if truck_candidates else None,
        flow_counts=fc.tolist(), cap_counts=cc.tolist(),
        states_with_any=sum(sum(r['cap_counts']) > 0 for r in rows),
        states_with_multiple=sum(sum(r['cap_counts']) >= 2 for r in rows),
        states_with_identical_features=sum(r['identical_capped_features_same_crane'] > 0 for r in rows))
    mixed_flow = [r for r in rows if r['flow_counts'][0] and r['flow_counts'][1]]
    mixed_cap = [r for r in rows if 0 < sum(r['cap_counts']) < sum(r['flow_counts'][:2])]
    result['mixed_flow_n'], result['mixed_cap_n'] = len(mixed_flow), len(mixed_cap)
    for name in VERSIONS:
        result[name+'_choice'] = dict(mean_flow_mass=np.mean([r[name]['mass'] for r in rows], 0).tolist(),
            top_flow_counts=[sum(r[name]['top_flow'] == f for r in rows) for f in range(5)],
            mean_chosen_probability=float(np.mean([r[name]['chosen_probability'] for r in rows])),
            mean_chosen_rank=float(np.mean([r[name]['chosen_rank'] for r in rows])),
            mixed_inbound_probability=(float(np.mean([r[name]['mass'][0]/sum(r[name]['mass'][:2])
                                                      for r in mixed_flow])) if mixed_flow else None),
            mixed_capped_probability=(float(np.mean([r[name]['capped_mass']/sum(r[name]['mass'][:2])
                                                     for r in mixed_cap])) if mixed_cap else None),
            mixed_capped_top_fraction=(float(np.mean([r[name]['top_capped'] for r in mixed_cap]))
                                       if mixed_cap else None))
    for before, after in (('pre', 'post'), ('initial', 'final')):
        label = before+'_'+after
        result[label] = dict(top_candidate_changed_fraction=float(np.mean([
            r[before]['top_candidate'] != r[after]['top_candidate'] for r in rows])),
            mean_total_variation=float(np.mean([r[label+'_tv'] for r in rows])),
            top_flow_transitions=[[sum(r[before]['top_flow'] == a and r[after]['top_flow'] == b for r in rows)
                                   for b in range(5)] for a in range(5)])
    result['advantage_sign_agrees_recorded_logp_change_fraction'] = float(np.mean([
        (r['post_logp']-r['pre_logp'])*r['advantage'] > 0 for r in rows]))
    return result
