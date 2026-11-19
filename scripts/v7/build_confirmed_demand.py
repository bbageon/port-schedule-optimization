"""Recalibrate the existing synthetic family from source-confirmed gate entries."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import time

from audit_visit_units import load_rows, gate_records, physical_visits, templates, MERGE_S
from build_gate_time_calibration import aggregate_orders, visit_patterns
from confirmed_demand_support import (BASE,OUT,BA,PREREG,OLD_TEMPLATES,OLD_FIT,OLD_MIX,OLD_MINUTES,
    OLD_DEMAND,OLD_BANK,read,save,sha,ordered_templates,fit_days,bind)
from gate_time_provenance import CONFIRMED,load_evidence
from yard_rl.integrated.repro import repro_stamp
from yard_rl.v7.day_mix import add_daily_mix
from yard_rl.v7.day_shape import build_bank
from yard_rl.v7.input_bundle import SCHEMA,InputBundle


def calibrate_minutes(template_path, folder):
    import calibrate_correlated_minutes as ar
    import calibrate_multiscale_minutes as multiscale
    import validate_correlated_implementation as runtime
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'multiscale-minute-calibration.json').exists():
        raise FileExistsError('Calibration output already exists')
    # Run the exact old formulas and seed loops; only source/output locations vary.
    for module in (ar,multiscale):
        with bind(module,OUT=folder,TEMPLATES=template_path,write_result=save), redirect_stdout(io.StringIO()):
            module.main()
    result = read(folder/'multiscale-minute-calibration.json')
    if not result['passed']:
        raise ValueError('Fixed minute model criteria failed; no threshold or seed adjustment')
    with bind(runtime,OUT=folder,TEMPLATES=template_path,write_result=save), redirect_stdout(io.StringIO()):
        runtime.main()


def make_bank(fitted, daily, previous, *, new_seeds):
    bank = add_daily_mix(build_bank(fitted['fits'],seed=previous['seed'],
        sizes={k:len(v) for k,v in previous['splits'].items()},basis=previous['basis_hour_masses'],family=previous['family']),daily)
    bank['initial_stock_mode'] = 'preserve'
    for k,split in enumerate(('train','selection','holdout')):
        for i,row in enumerate(bank['splits'][split]):
            old = previous['splits'][split][i]
            assert row['parents'] == old['parents'] and row['fraction'] == old['fraction']
            row['input_seed'] = 9976100+k*100+i if new_seeds else old['input_seed']
    return bank


def build(out):
    out = out.resolve()
    if (out/'build-record.json').exists() or (out/'bundle.json').exists():
        raise FileExistsError('Use a new output directory; prior results are immutable')
    started = time.perf_counter()
    manifest = read(BA/'source-manifest.json')
    root = BASE/'오더데이터'
    assert len(manifest) == 600 and all(sha(root/r['path']) == r['sha256'] for r in manifest)
    assert read(BA/'verification.json')['status'] == 'PASS'
    visits, visit_manifest = load_rows()
    proof = load_evidence((root/'기사/_raw').glob('*.gz'))
    entered = [r for r in visits if r['gate_in_time']]
    confirmed = [r for r in entered if proof.classify(r) in CONFIRMED]
    expected = read(BA/'arrival-impacts.json')
    cohorts = dict(legacy=entered,confirmed=confirmed)
    aggregates = {k:aggregate_orders(rows) for k,rows in cohorts.items()}
    assert aggregates['legacy'] == expected['order_records']['before']
    assert aggregates['confirmed'] == expected['order_records']['confirmed']
    allowed = read(OLD_TEMPLATES)['dates']
    source_summary = dict(schema='confirmed-demand-source-v1',source_manifest_sha256=sha(BA/'source-manifest.json'),
        gate_contract=read(BA/'profile-confirmed.json')['source_event_contract'],dates=allowed,
        legacy_orders=len(entered),confirmed_orders=len(confirmed),excluded_orders=len(entered)-len(confirmed),
        raw_source_contract='all 13 previously viewed development dates; incomplete whole-terminal coverage',cohorts={})
    save(out/'source-manifest.json',manifest)
    fitted = {}
    for arm,rows in cohorts.items():
        folder = out/arm
        physical = physical_visits(gate_records(rows),MERGE_S)
        templ = ordered_templates(templates(physical,visit_manifest))
        pats = visit_patterns(physical)
        agg = aggregates[arm]
        assert [r['date'] for r in agg['per_date']] == allowed
        stored_template = read(OLD_TEMPLATES if arm == 'legacy' else BA/'visit-arrival-templates-confirmed.json')
        assert templ['minute_counts'] == stored_template['minute_counts']
        assert pats == expected['patterns']['before' if arm == 'legacy' else 'confirmed']
        templ['source_event_contract'] = ('legacy-unfiltered' if arm == 'legacy' else source_summary['gate_contract'])
        save(folder/'minute-templates.json',templ)
        save(folder/'daily.json',dict(per_date=agg['per_date'],hour_counts=agg['hour_counts'],
            gate_out_share=agg['gate_out_share'],orders=agg['n'],visits=len(physical)))
        link = copy.deepcopy(read(OLD_DEMAND)['profiles']['observed_visit_links']['visit_link'])
        link['patterns'] = pats['patterns']
        link['source'] = dict(task='YR-332-bb',arm=arm,visits=len(physical),eligible=pats['known'],
            original_manifest_sha256=source_summary['source_manifest_sha256'])
        save(folder/'visit-link.json',link)
        fitted[arm] = fit_days(agg['per_date'])
        save(folder/'curve-fit.json',fitted[arm])
        if not fitted[arm]['input_family_pass']:
            raise ValueError('Existing curve family failed fixed per-date criterion')
        if arm == 'legacy':
            assert fitted[arm]['fits'] == read(OLD_FIT)['fits'],'legacy curve fit did not reproduce'
        calibrate_minutes(folder/'minute-templates.json',folder)
        if arm == 'legacy':
            for name in ('correlated-minute-calibration.json','multiscale-minute-calibration.json','minute-implementation-check.json'):
                actual,old = read(folder/name),read(OLD_MINUTES/name)
                for key in ('parameters','observed','mean','errors','passed','repeats'):
                    if key in old:assert actual[key] == old[key],(name,key,'legacy minute reproduction')
        source_summary['cohorts'][arm] = dict(orders=agg['n'],visits=len(physical),eligible_patterns=pats['known'],
            gate_out_share=agg['gate_out_share'],daily_orders=[r['orders'] for r in agg['per_date']],
            hour_counts=agg['hour_counts'],curve_passed_dates=fitted[arm]['passed_dates'])
        print(f'{arm}: source and all three minute stages verified',flush=True)
    old_bank = read(OLD_BANK)
    assert make_bank(fitted['legacy'],aggregates['legacy']['per_date'],old_bank,new_seeds=False) == old_bank
    new_bank = make_bank(fitted['confirmed'],aggregates['confirmed']['per_date'],old_bank,new_seeds=True)
    save(out/'bank.json',new_bank)
    save(out/'source-summary.json',source_summary)
    demand = copy.deepcopy(read(OLD_DEMAND)['profiles']['observed_visit_links'])
    demand.update(hour_counts=aggregates['confirmed']['hour_counts'],gate_out_share=aggregates['confirmed']['gate_out_share'],
        within_hour=dict(method='correlated_minutes',parameters=read(out/'confirmed/multiscale-minute-calibration.json')['parameters'],
            source=dict(task='YR-332-bb',calibration_sha256=sha(out/'confirmed/multiscale-minute-calibration.json'))),
        visit_link=read(out/'confirmed/visit-link.json'),
        source=dict(task='YR-332-bb',source_summary_sha256=sha(out/'source-summary.json'),
            source_time_contract='explicit-gate-request-time-v1',unit='entered container orders; arrivals grouped into truck visits',
            dates=allowed,orders=len(confirmed),scope='calibrated synthetic input; not source event replay'))
    save(out/'demand-base.json',demand)
    def ref(path):
        import os
        return dict(path=Path(os.path.relpath(path,out)).as_posix(),sha256=sha(path))
    bundle = dict(schema=SCHEMA,task='YR-332-bb',assets=dict(information=ref(BA/'profile-confirmed.json'),
        bank=ref(out/'bank.json'),demand=ref(out/'demand-base.json')),
        scope='Same synthetic family recalibrated from explicit gate times; no raw arrival replay or performance claim')
    save(out/'bundle.json',bundle)
    loaded = InputBundle(out/'bundle.json')
    assert loaded.case('train',0,3500)['seed'] == 9976100
    paths = [Path(__file__),BASE/'scripts/v7/confirmed_demand_support.py',BASE/PREREG,
        BASE/'scripts/v7/audit_day_shapes.py',BASE/'scripts/v7/audit_visit_units.py',
        BASE/'scripts/v7/calibrate_correlated_minutes.py',BASE/'scripts/v7/calibrate_multiscale_minutes.py',
        BASE/'scripts/v7/validate_correlated_implementation.py',BASE/'scripts/v7/synthetic_inputs.py',
        OLD_TEMPLATES,OLD_FIT,OLD_MIX,OLD_DEMAND,OLD_BANK,BA/'profile-confirmed.json',
        *sorted((BASE/'src/yard_rl/v7').rglob('*.py'))]
    cfg = dict(task='YR-332-bb',dates=allowed,source_files=600,excluded_orders=601,
        loads=[3500,7500,15000],splits={k:len(v) for k,v in new_bank['splits'].items()},
        curve_seed=9964000,input_cases=102,repeats=6,minute_fit_repeats=100,policy_execution=False)
    seeds = dict(input=[r['input_seed'] for rows in new_bank['splits'].values() for r in rows],
        curve_fit=list(range(9934000,9934013)),minute_ar=list(range(9936300,9936400)),
        minute_coarse=list(range(9936500,9936600)),minute_runtime=list(range(9936700,9936800)))
    save(out/'build-record.json',dict(configuration=cfg,source_sha256={p.relative_to(BASE).as_posix():sha(p) for p in paths},
        repro=repro_stamp(experiment='YR-332-bb',seeds=seeds,params=cfg,prereg=PREREG),
        bundle_sha256=sha(out/'bundle.json'),seconds=time.perf_counter()-started))
    print(json.dumps(dict(status='BUILT',excluded=len(entered)-len(confirmed),curves=sum(cfg['splits'].values())),ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    build(p.parse_args().out)
