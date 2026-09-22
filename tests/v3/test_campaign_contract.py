"""The confirmatory-campaign plumbing (YR-318).

These guard a 779 CPU-hour run: the frozen 20x3 contract must stay unbreakable,
a new campaign must declare its own design, and the paired statistics must
refuse a partial or unpaired result set rather than report one.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3'))

from campaign_checks import campaign_summary, write_campaign_report   # noqa: E402
from independent_eval_checks import (ARMS, FROZEN_SCHEMA, SEEDS,     # noqa: E402
                                     campaign_contract, campaign_specs)

BAND = [22_000_000 + 100_000 * i for i in range(4)]
CAMPAIGN_ARMS = ['NO_REALLOC', 'SLOT_LL', 'RL_TIME']


WEIGHTS = dict(checkpoint='outputs/v3/month-n1/ckpt_029.pt', checkpoint_sha256='a' * 64)


def campaign_cfg(**over):
    cfg = dict(schema='yr318.confirmatory-campaign.v1', seeds=list(BAND),
               arms=list(CAMPAIGN_ARMS), **WEIGHTS)
    cfg.update(over)
    return cfg


def month(seed, arm, cost):
    return dict(seed=seed, arm=arm, cost_28d_krw=cost, cost_30d_krw=cost * 1.07,
                requested=1000, completed=1000, vessel_all_completed=True,
                unfinished_vessel_jobs=0, requested_identity_sha256=f'input-{seed}')


def rows_for(arms=CAMPAIGN_ARMS, seeds=BAND):
    base = {'NO_REALLOC': 100.0, 'SLOT_LL': 96.0, 'RL_TIME': 90.0}
    return [month(seed, arm, base[arm] + i) for i, seed in enumerate(seeds) for arm in arms]


def test_the_frozen_design_cannot_be_edited():
    frozen = dict(schema=FROZEN_SCHEMA, seeds=list(SEEDS), arms=list(ARMS), **WEIGHTS)
    assert campaign_contract(frozen) == (SEEDS, ARMS)
    with pytest.raises(ValueError, match='20 x 3'):
        campaign_contract({**frozen, 'arms': ['NO_REALLOC', 'RL']})
    with pytest.raises(ValueError, match='20 x 3'):
        campaign_contract({**frozen, 'seeds': list(SEEDS)[:19]})


def test_a_new_campaign_declares_its_own_design():
    assert campaign_contract(campaign_cfg()) == (tuple(BAND), tuple(CAMPAIGN_ARMS))


@pytest.mark.parametrize('bad, message', [
    (dict(seeds=[20_000_000, 22_000_000]), 'already used for judgement'),
    (dict(seeds=[9_900_700, 9_900_800]), 'Diagnostic band'),
    (dict(arms=['NO_REALLOC', 'NO_REALLOC']), 'distinct'),
    (dict(arms=['NO_REALLOC', 'RL_IMAGINARY']), 'Unknown arms'),
    (dict(seeds=[]), 'non-empty'),
])
def test_a_campaign_cannot_borrow_used_seeds_or_invent_arms(bad, message):
    with pytest.raises(ValueError, match=message):
        campaign_contract(campaign_cfg(**bad))


def test_summary_reports_the_declared_primary_comparison(tmp_path):
    summary = campaign_summary(rows_for(), seeds=BAND, arms=CAMPAIGN_ARMS,
        comparisons=[('NO_REALLOC', 'RL_TIME'), ('SLOT_LL', 'RL_TIME')],
        primary=['28d:SLOT_LL-RL_TIME'], bootstrap_seed=9_900_725, bootstrap_samples=500)
    rule = summary['comparisons']['28d:SLOT_LL-RL_TIME']
    assert rule['primary'] and rule['interval_level'] == 0.95      # one primary: no split
    assert rule['wins'] == len(BAND) and rule['mean_saving_krw'] == pytest.approx(6.0)
    # Against the weaker baseline the saving is larger, and it stays descriptive.
    loose = summary['comparisons']['28d:NO_REALLOC-RL_TIME']
    assert not loose['primary'] and loose['interval_level'] == 0.95
    assert loose['mean_saving_krw'] > rule['mean_saving_krw']
    assert summary['policy_runs'] == len(BAND) * len(CAMPAIGN_ARMS)


def test_three_training_runs_average_into_one_primary_column():
    """The convention fixes one primary comparison. Three training runs are one
    procedure, so they average into a single column rather than three tests."""
    arms = ['SLOT_LL', 'RL_TIME:n1', 'RL_TIME:n2', 'RL_TIME:n3']
    costs = {'SLOT_LL': 100.0, 'RL_TIME:n1': 88.0, 'RL_TIME:n2': 94.0, 'RL_TIME:n3': 91.0}
    rows = [month(seed, arm, costs[arm] + i) for i, seed in enumerate(BAND) for arm in arms]
    summary = campaign_summary(rows, seeds=BAND, arms=arms,
        derived={'RL_TIME:mean': ['RL_TIME:n1', 'RL_TIME:n2', 'RL_TIME:n3']},
        comparisons=[('SLOT_LL', 'RL_TIME:mean'), ('SLOT_LL', 'RL_TIME:n2')],
        primary=['28d:SLOT_LL-RL_TIME:mean'], bootstrap_seed=9_900_727,
        bootstrap_samples=500)
    averaged = summary['comparisons']['28d:SLOT_LL-RL_TIME:mean']
    assert averaged['primary'] and averaged['interval_level'] == 0.95
    assert averaged['mean_saving_krw'] == pytest.approx(100 - (88 + 94 + 91) / 3)
    # The weakest single run stays visible and stays descriptive.
    weakest = summary['comparisons']['28d:SLOT_LL-RL_TIME:n2']
    assert not weakest['primary']
    assert weakest['mean_saving_krw'] < averaged['mean_saving_krw']
    assert summary['derived_columns'] == {
        'RL_TIME:mean': ['RL_TIME:n1', 'RL_TIME:n2', 'RL_TIME:n3']}


def test_a_derived_column_cannot_shadow_a_measured_one():
    with pytest.raises(ValueError, match='collides'):
        campaign_summary(rows_for(), seeds=BAND, arms=CAMPAIGN_ARMS,
            derived={'RL_TIME': ['RL_TIME']},
            comparisons=[('SLOT_LL', 'RL_TIME')], primary=['28d:SLOT_LL-RL_TIME'],
            bootstrap_seed=9_900_727, bootstrap_samples=100)


def test_the_confirmatory_family_splits_one_error_budget(tmp_path):
    """Three training runs mean three confirmatory tests of one claim, so each
    interval must be wider than a lone 95% interval, not the same width."""
    family = ['28d:SLOT_LL-RL_TIME', '30d:SLOT_LL-RL_TIME', '28d:NO_REALLOC-RL_TIME']
    summary = campaign_summary(rows_for(), seeds=BAND, arms=CAMPAIGN_ARMS,
        comparisons=[('NO_REALLOC', 'RL_TIME'), ('SLOT_LL', 'RL_TIME')],
        primary=family, bootstrap_seed=9_900_725, bootstrap_samples=500)
    assert summary['primary_alpha'] == pytest.approx(0.05 / 3)
    confirmed = summary['comparisons']['28d:SLOT_LL-RL_TIME']
    descriptive = summary['comparisons']['30d:NO_REALLOC-RL_TIME']
    assert confirmed['interval_level'] == pytest.approx(1 - 0.05 / 3)
    assert not descriptive['primary'] and descriptive['interval_level'] == 0.95
    report = write_campaign_report(tmp_path / 'results.md', rows_for(), summary)
    assert '98.3333% interval' in report.read_text(encoding='utf-8')

    report = write_campaign_report(tmp_path / 'results.md', rows_for(), summary)
    text = report.read_text(encoding='utf-8')
    assert '**(primary)**' in text and 'SLOT_LL' in text


def test_summary_refuses_a_partial_or_unpaired_result_set():
    complete = rows_for()
    with pytest.raises(ValueError, match='no partial inference'):
        campaign_summary(complete[:-1], seeds=BAND, arms=CAMPAIGN_ARMS,
            comparisons=[('SLOT_LL', 'RL_TIME')], primary=['28d:SLOT_LL-RL_TIME'],
            bootstrap_seed=9_900_725, bootstrap_samples=100)
    unpaired = [dict(row) for row in complete]
    unpaired[0]['requested_identity_sha256'] = 'different-input'
    with pytest.raises(ValueError, match='Unpaired'):
        campaign_summary(unpaired, seeds=BAND, arms=CAMPAIGN_ARMS,
            comparisons=[('SLOT_LL', 'RL_TIME')], primary=['28d:SLOT_LL-RL_TIME'],
            bootstrap_seed=9_900_725, bootstrap_samples=100)


def test_a_campaign_without_a_primary_comparison_is_refused():
    with pytest.raises(ValueError, match='at least one primary'):
        campaign_summary(rows_for(), seeds=BAND, arms=CAMPAIGN_ARMS,
            comparisons=[('SLOT_LL', 'RL_TIME')], primary=[],
            bootstrap_seed=9_900_725, bootstrap_samples=100)


def test_one_arm_can_appear_once_per_training_run():
    """Training-seed stability is measured inside one campaign: the same engine
    arm appears as several columns, each with its own weights."""
    specs = [dict(label='SLOT_LL', arm='SLOT_LL'),
             dict(label='RL_TIME_n1', arm='RL_TIME', checkpoint='m1.pt',
                  checkpoint_sha256='1' * 64),
             dict(label='RL_TIME_n2', arm='RL_TIME', checkpoint='m2.pt',
                  checkpoint_sha256='2' * 64)]
    seeds, resolved = campaign_specs(campaign_cfg(arm_specs=specs))
    assert seeds == tuple(BAND)
    assert [item['label'] for item in resolved] == ['SLOT_LL', 'RL_TIME_n1', 'RL_TIME_n2']
    # An arm that names no weights of its own falls back to the campaign default.
    assert resolved[0]['checkpoint'] == WEIGHTS['checkpoint']
    assert [item['checkpoint'] for item in resolved[1:]] == ['m1.pt', 'm2.pt']
    assert campaign_contract(campaign_cfg(arm_specs=specs))[1] == (
        'SLOT_LL', 'RL_TIME_n1', 'RL_TIME_n2')


def test_a_label_that_cannot_be_a_directory_name_is_refused():
    """Each label becomes a folder under outputs/ on a Windows filesystem, so a
    colon or a slash would fail hours into the run instead of now."""
    for bad in ('RL_TIME:n1', 'RL/TIME', 'RL TIME', ''):
        specs = [dict(label=bad, arm='RL_TIME')]
        with pytest.raises(ValueError, match='filename-safe|distinct'):
            campaign_specs(campaign_cfg(arm_specs=specs))


def test_an_arm_without_weights_anywhere_is_refused():
    cfg = dict(schema='yr318.confirmatory-campaign.v1', seeds=list(BAND),
               arm_specs=[dict(label='RL_TIME', arm='RL_TIME')])
    with pytest.raises(ValueError, match='declare no weights'):
        campaign_specs(cfg)


def test_two_columns_cannot_share_a_label():
    specs = [dict(label='RL_TIME', arm='RL_TIME'), dict(label='RL_TIME', arm='RL')]
    with pytest.raises(ValueError, match='distinct'):
        campaign_specs(campaign_cfg(arm_specs=specs))
