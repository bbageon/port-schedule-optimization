"""Protect source order, axis isolation and hash-pinned input integration."""
import copy
import hashlib
import json

import pytest

from confirmed_demand_support import ordered_templates,fit_days
from yard_rl.v7.day_shape import build_bank,basis_hour_masses
from yard_rl.v7.input_bundle import InputBundle,SCHEMA


def test_numeric_hours_do_not_depend_on_json_key_order():
    original=dict(dates=['day1'],minute_counts={str(h):[[h]*60] for h in sorted(range(24),key=str)})
    saved=copy.deepcopy(original)
    result=ordered_templates(original)
    assert list(result['minute_counts']) == [str(h) for h in range(24)]
    assert result['minute_counts'] == saved['minute_counts'] and original == saved


@pytest.mark.parametrize('bad', ['missing_hour','negative','date_count'])
def test_invalid_template_cannot_enter_calibration(bad):
    d=dict(dates=['day1'],minute_counts={str(h):[[1]*60] for h in range(24)})
    if bad=='missing_hour':del d['minute_counts']['2']
    elif bad=='negative':d['minute_counts']['0'][0][0]=-1
    else:d['dates'].append('day2')
    with pytest.raises(ValueError):ordered_templates(d)


def test_curve_fit_requires_conserved_daily_counts():
    with pytest.raises(ValueError,match='conservation'):
        fit_days([dict(date='d1',orders=25,hour_counts=[1]*24)])


def bundle(tmp_path,confirmed=True):
    info=dict(schema='empirical-reservation-notice-pairs-v1',all_orders_reserved=True,
        source_event_contract=dict(log_clock_substitution=not confirmed),
        flows={f:dict(n=1,eta_offset_s=0,samples=[[0,30,1]]) for f in ('GATE_IN','GATE_OUT')})
    bank=build_bank([dict(date='d1',weights=[.4,.2,.2,.2]),dict(date='d2',weights=[.2,.3,.3,.2])],
        seed=9,sizes=dict(train=2,selection=1,holdout=1),basis=basis_hour_masses())
    bank['initial_stock_mode']='preserve'
    for rows in bank['splits'].values():
        for row in rows:row['gate_out_share']=.5
    demand=dict(schema='observed-truck-demand-marginals-v1',hour_counts=[1]*24,gate_out_share=.5,
                within_hour=dict(method='poisson'))
    refs={}
    for name,value in dict(information=info,bank=bank,demand=demand).items():
        path=tmp_path/f'{name}.json';path.write_text(json.dumps(value),encoding='utf-8')
        refs[name]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path=tmp_path/'bundle.json';path.write_text(json.dumps(dict(schema=SCHEMA,assets=refs)),encoding='utf-8')
    return path


def test_asset_tampering_is_rejected_before_generation(tmp_path):
    path=bundle(tmp_path)
    (tmp_path/'demand.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):InputBundle(path)


def test_unconfirmed_timestamps_cannot_be_silently_selected(tmp_path):
    with pytest.raises(ValueError,match='source-confirmed'):InputBundle(bundle(tmp_path,confirmed=False))


def test_case_mutation_does_not_contaminate_later_cases(tmp_path):
    inputs=InputBundle(bundle(tmp_path)); first=inputs.case('train',0,3500)
    expected=copy.deepcopy(first)
    first['demand_profile']['hour_counts'][0]=999
    first['information_profile']['flows']['GATE_IN']['samples'][0][0]=999
    assert inputs.case('train',0,3500)==expected
    assert inputs.case('selection',0,3500)['seed'] != expected['seed']


@pytest.mark.parametrize('args',[('train',-1,3500),('unknown',0,3500),('train',0,0)])
def test_invalid_case_is_rejected(tmp_path,args):
    with pytest.raises(ValueError):InputBundle(bundle(tmp_path)).case(*args)
