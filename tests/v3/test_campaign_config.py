"""The campaign config generator (YR-318).

A mistyped hash or a half-finished training run reaching this config costs a
thousand CPU-hours, so the generator refuses both rather than warning.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3'))

import make_campaign_config as maker                      # noqa: E402
from independent_eval_checks import campaign_specs        # noqa: E402

BAND = [22_000_000 + 100_000 * i for i in range(3)]


def make_bank(root, seeds=BAND, passed=True):
    bank = root / 'bank'
    bank.mkdir(parents=True)
    (bank / 'summary.json').write_text(json.dumps(
        dict(passed=passed, rows=[dict(seed=s) for s in seeds])), encoding='utf-8')
    return bank


def make_supply(root, seeds=BAND):
    folder = root / 'supply'
    folder.mkdir(parents=True)
    for seed in seeds:
        (folder / f'seed-{seed}.json').write_text(json.dumps(dict(seed=seed)), encoding='utf-8')
    return folder


def make_training(root, name, *, pruning='feasible_first', admission='PRESERVE', complete=True):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / 'ckpt_029.pt').write_bytes(name.encode())
    if complete:
        (folder / 'ckpt_init.pt').write_bytes(b'init-' + name.encode())
    (folder / 'training_manifest.json').write_text(json.dumps(
        dict(seed=9_900_701, init_seed=9_900_701, admission_mode=admission,
             supply_mode='COUNT_BALANCED', candidate_pruning=pruning)), encoding='utf-8')
    return folder


def args_for(root, training, prereg, out):
    from types import SimpleNamespace
    return SimpleNamespace(bank=make_bank(root), supply_inputs=make_supply(root),
        training=training, prereg=prereg, out=out, bootstrap_seed=9_900_727,
        workers_max=16, cpu_limit=20)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """The generator writes repo-relative paths, so it needs a repo-shaped root."""
    (tmp_path / 'src/yard_rl/v3/stage').mkdir(parents=True)
    (tmp_path / 'src/yard_rl/v3/stage/supply_plan.py').write_text('solver', encoding='utf-8')
    monkeypatch.setattr(maker, 'ROOT', tmp_path)
    return tmp_path


def test_it_writes_a_config_the_runner_accepts(workspace, capsys):
    prereg = workspace / 'prereg.md'
    prereg.write_text('declared design', encoding='utf-8')
    training = [make_training(workspace, f'month-n{i}') for i in (1, 2, 3)]
    out = workspace / 'config.json'
    maker.build(args_for(workspace, training, prereg, out))

    cfg = json.loads(out.read_text(encoding='utf-8'))
    seeds, specs = campaign_specs(cfg)
    assert list(seeds) == BAND
    assert [item['label'] for item in specs] == ['NO_REALLOC', 'SLOT_LL', 'SPACE_TIME_LL',
        'RL_TIME_n1', 'RL_TIME_n2', 'RL_TIME_n3', 'RL_n1', 'RL_NOVETO_n1']
    # Each training run supplies its own weights; the rule arms fall back to the first.
    weights = {item['label']: item['checkpoint_sha256'] for item in specs}
    assert len({weights[f'RL_TIME_n{i}'] for i in (1, 2, 3)}) == 3
    assert weights['RL_n1'] == weights['RL_TIME_n1'] == weights['NO_REALLOC']
    # One primary comparison, against the rule baseline, on the averaged column.
    assert cfg['primary_comparisons'] == ['28d:SLOT_LL-RL_TIME_mean']
    assert cfg['derived_columns'] == {'RL_TIME_mean': ['RL_TIME_n1', 'RL_TIME_n2', 'RL_TIME_n3']}
    assert cfg['candidate_pruning'] == 'feasible_first' and cfg['measure_latency'] is True
    # Every declared file is hashed from disk.
    for name, digest in cfg['files'].items():
        assert maker.sha(workspace / name) == digest
    assert json.loads(capsys.readouterr().out)['runs'] == len(BAND) * len(specs)


def test_an_unfinished_training_run_cannot_become_an_arm(workspace):
    prereg = workspace / 'prereg.md'
    prereg.write_text('x', encoding='utf-8')
    training = [make_training(workspace, 'month-n1'),
                make_training(workspace, 'month-n2', complete=False)]
    with pytest.raises(SystemExit, match='incomplete'):
        maker.build(args_for(workspace, training, prereg, workspace / 'config.json'))


@pytest.mark.parametrize('bad, message', [
    (dict(pruning='legacy'), 'unrepaired engine'),
    (dict(admission='LEGACY'), 'contract the campaign does not evaluate'),
])
def test_a_run_trained_off_contract_is_refused(workspace, bad, message):
    prereg = workspace / 'prereg.md'
    prereg.write_text('x', encoding='utf-8')
    training = [make_training(workspace, 'month-n1'),
                make_training(workspace, 'month-n2', **bad)]
    with pytest.raises(SystemExit, match=message):
        maker.build(args_for(workspace, training, prereg, workspace / 'config.json'))


def test_a_single_training_run_cannot_claim_stability(workspace):
    prereg = workspace / 'prereg.md'
    prereg.write_text('x', encoding='utf-8')
    with pytest.raises(SystemExit, match='at least two runs'):
        maker.build(args_for(workspace, [make_training(workspace, 'month-n1')], prereg,
                             workspace / 'config.json'))


def test_a_missing_supply_input_stops_the_config(workspace):
    prereg = workspace / 'prereg.md'
    prereg.write_text('x', encoding='utf-8')
    training = [make_training(workspace, f'month-n{i}') for i in (1, 2)]
    args = args_for(workspace, training, prereg, workspace / 'config.json')
    (args.supply_inputs / f'seed-{BAND[-1]}.json').unlink()
    with pytest.raises(SystemExit, match='Supply-corrected input missing'):
        maker.build(args)
