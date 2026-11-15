"""Recompute the saved-training facts used by the read-only architecture review."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    out = Path('outputs/reports/yr332_network_review')
    out.mkdir(parents=True, exist_ok=True)
    paths = list(sorted(Path('outputs/reports/yr332_linked_retraining').glob('model-*/training.json')))
    models = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = data['rows']
        models.append(dict(model=path.parent.name, config=data['model_config'],
            collected_pairs=sum(r['stats']['labeled'] for r in rows),
            train_pairs_total=sum(r['fit']['n'] // 2 for r in rows),
            validation_pairs_total=sum(r['fit']['n_val'] // 2 for r in rows),
            train_pairs_per_update=[r['fit']['n'] // 2 for r in rows],
            last_update_train_pairs=rows[-1]['fit']['n'] // 2,
            updates=len(rows), steps_each=[r['fit']['steps'] for r in rows],
            max_recorded_teacher_candidate_count=max(c['candidate_count'] for r in rows for c in r['stats']['witness_search'])))
    paths += [Path(p) for p in ('src/yard_rl/v7/crane/policy.py', 'src/yard_rl/v7/crane/fit.py',
        'scripts/v7/retrain_linked_environment.py', 'outputs/reports/yr332_linked_retraining/campaign.json',
        'docs/research/v7-discharge/network-learning-review.md')]
    value = dict(scope='read-only review; no model/optimizer/checkpoint changes',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), models=models,
        sources={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        classification='supervised normalized counterfactual cost-difference regression; not PPO/DQN updates',
        behavior='frozen initial model collects labels; each student update fits only its current corpus',
        architecture='shared 10-64-64-1 candidate scorer; variable candidate count; batched inference',
        papers=[dict(url='https://www.sciencedirect.com/science/article/pii/S0278612525003140', finding='April 2026; review of 143 studies'),
            dict(url='https://www.sciencedirect.com/science/article/pii/S136655452500657X', finding='CNN-MLP and shared-policy PPO; crane allocation among blocks'),
            dict(url='https://link.springer.com/article/10.1007/s40747-025-02174-3', finding='multi-head self-attention PPO; online 2025-12-29, 2026 volume'),
            dict(url='https://www.sciencedirect.com/science/article/pii/S0952197625029434', finding='weighted multi-graph and relational graph convolution; AGV/YC loading')])
    (out / 'review.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(models, indent=2))


if __name__ == '__main__':
    main()
