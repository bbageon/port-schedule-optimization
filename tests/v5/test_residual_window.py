"""Offline horizon bounds must not be mistaken for actual work feasibility."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location('residual_window',
    Path(__file__).resolve().parents[2]/'scripts/v5/analyze_residual_window.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
window_bounds = _module.window_bounds


def example(release=90, source=90):
    return dict(sources=[dict(container='BOX',source_kind='VESSEL_DISCHARGE',planned_source_s=source)],
                schedule=[dict(job_id='OUT',flow='GATE_OUT',day=0,arrival_s=release,travel_s=0,target='BOX')],
                vessels_by_day={})


@pytest.mark.parametrize('release,source,expected', [(90,90,0),(100,90,1),(90,100,1),(110,120,1)])
def test_release_or_source_at_cutoff_is_counted_once(release, source, expected):
    value = window_bounds(example(release, source), 100)
    assert value['lower_bound_unfinished'].get('GATE_OUT',0) == expected
    assert len(value['jobs']) == expected


def test_vessel_load_waits_for_source_and_discharge_has_independent_release():
    doc = example(source=120)
    doc['schedule'] = []
    doc['vessels_by_day'] = {'0': [
        dict(block='Y01',key='LOAD',work='LOAD',start_s=0,cadence_s=10,moves=1,targets=['BOX']),
        dict(block='Y01',key='DIS',work='DISCHARGE',start_s=90,cadence_s=10,moves=2)]}
    value = window_bounds(doc,100)
    assert value['lower_bound_unfinished'] == {'VESSEL_LOAD':1,'VESSEL_DISCHARGE':1}
    assert value['total_jobs'] == 3


def test_duplicate_job_rejected():
    doc = example()
    doc['schedule'] *= 2
    with pytest.raises(ValueError,match='Duplicate job'):
        window_bounds(doc,100)


def test_missing_source_rejected():
    doc = example()
    doc['sources'] = []
    with pytest.raises(KeyError):
        window_bounds(doc,100)
