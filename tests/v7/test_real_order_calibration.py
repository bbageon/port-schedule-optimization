"""Source parsing and as-of joins must not learn from post-entry revisions."""
import csv
import json
import pytest

import calibrate_order_information as calibration


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fixture(root, duplicate=False):
    template=dict(source='VBS',status='complete',in_out='IN',copino_first_time='2026-11-04 08:00:00',
        in_out_reserve_time='2026-11-04 13:00:00',gate_in_time='2026-11-04 10:00:00',
        enter_block_time='2026-11-04 10:05:00',job_done_time='2026-11-04 10:20:00',
        gate_out_time='2026-11-04 10:25:00',visit_id='v1',doc_keys='d1',truck_hash='t1',con_hash='c1')
    rows=[template,dict(template,visit_id='v2',doc_keys='d2',truck_hash='t2',con_hash='c2',in_out='OUT'),
          dict(template,visit_id='v3',doc_keys='d3',truck_hash='t3',con_hash='c3',
               gate_in_time='2026-11-07 10:00:00',copino_first_time='2026-11-07 08:00:00')]
    if duplicate:rows.append(dict(template,visit_id='v4'))
    write(root/'기사/방문/visits.csv',rows)
    base=dict(log_time='2026-11-04 08:00:00',doc_key='d1',doc_status='9',
              copino_err_type='NM',in_out_reserve_time='2026-11-04 09:30:00',in_out='IN')
    orders=[base,dict(base,log_time='2026-11-04 11:00:00',in_out_reserve_time='2026-11-04 13:00:00'),
            dict(base,doc_key='d2',in_out='OUT',doc_status='1',in_out_reserve_time='2026-11-04 11:00:00'),
            dict(base,doc_key='d3',log_time='2026-11-07 08:00:00',in_out_reserve_time='2026-11-07 05:00:00')]
    write(root/'기사/오더/copino.csv',orders)


def test_first_pre_entry_booking_and_holdout_split(tmp_path,monkeypatch):
    root=tmp_path/'source';fixture(root);out=tmp_path/'result'
    monkeypatch.setattr(calibration,'OUT',out);calibration.audit(root)
    p=json.loads((out/'profile.json').read_text(encoding='utf-8'))
    assert p['flows']['GATE_IN']['samples']==[[1800,7200,1]]
    assert p['flows']['GATE_OUT']['samples']==[[-3600,7200,1]]
    assert p['flows']['GATE_IN']['eta_offset_s']==1800
    a=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    assert a['cohorts']['holdout_IN']['error_s']['n']==1


def test_duplicate_physical_visit_is_rejected_even_with_different_row_id(tmp_path,monkeypatch):
    fixture(tmp_path,duplicate=True);monkeypatch.setattr(calibration,'OUT',tmp_path/'result')
    with pytest.raises(AssertionError,match='duplicate physical'):
        calibration.audit(tmp_path)


@pytest.mark.parametrize('value,expected',[
    ('202610299:00','2026-10-29T09:00:00'),
    ('2026-11-04 23:59:00','2026-11-04T23:59:00'),
    ('2026110499:00',None),('',None),('unavailable',None)])
def test_timestamp_variants_and_invalid_values(value,expected):
    result=calibration.stamp(value)
    assert (result.isoformat() if result else None)==expected
