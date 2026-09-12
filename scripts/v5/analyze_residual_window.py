"""Read-only lower-bound diagnosis of a saved fixed-cargo execution."""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def window_bounds(document, end_s):
    """A lower bound at/after the horizon cannot finish a positive-duration job.

    Earlier bounds do NOT establish feasibility: actual travel, queues, service,
    and source completions are deliberately not guessed from planned timings.
    """
    sources = {s['container']: s for s in document['sources']}
    if len(sources) != len(document['sources']):
        raise ValueError('Duplicate source identity')
    jobs, seen = [], set()
    gate_arrivals = {e['job_id']:float(e['arrival_s']) for e in document['schedule']}

    def add(jid, kind, day, release, source_id=None):
        if jid in seen:
            raise ValueError('Duplicate job identity: ' + jid)
        seen.add(jid)
        source = sources[source_id] if source_id is not None else None
        supply = 0.0 if source is None else float(source['planned_source_s'])
        if source is not None and source['source_kind'] == 'GATE_IN':
            # Reassignment can shorten the original route. Gate time alone is a
            # conservative bound; the original block travel time is NOT one.
            supply = gate_arrivals[source['source_job']]
        earliest = max(float(release), supply)
        if earliest >= end_s:
            jobs.append(dict(job_id=jid, flow=kind, day=day, release_lower_bound_s=release,
                             source=source_id, source_kind=None if source is None else source['source_kind'],
                             source_lower_bound_s=supply, earliest_lower_bound_s=earliest,
                             release_outside=release >= end_s, source_outside=supply >= end_s))

    for e in document['schedule']:
        add(e['job_id'], e['flow'], e['day'], float(e['arrival_s']),
            e['target'] if e['flow'] == 'GATE_OUT' else None)
    for day, rows in document['vessels_by_day'].items():
        for row in rows:
            for m in range(row['moves']):
                jid = f"{row['block']}:J-{row['key']}-{m:04d}"
                at = float(row['start_s']) + m * float(row['cadence_s'])
                add(jid, 'VESSEL_' + row['work'], int(day), at,
                    row['targets'][m] if row['work'] == 'LOAD' else None)
    counts = Counter(j['flow'] for j in jobs)
    return dict(end_s=end_s, total_jobs=len(seen), lower_bound_unfinished=dict(counts),
                by_day={str(d): dict(Counter(j['flow'] for j in jobs if j['day'] == d))
                        for d in sorted({j['day'] for j in jobs})},
                jobs=jobs)


def analyze(document, run):
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    cohorts = json.loads((run/'cohort_reports.json').read_text(encoding='utf-8'))
    result = window_bounds(document, float(report['time_s']))
    actual = report['cargo']['active_unfinished_jobs']
    result['actual_unfinished'] = actual
    result['not_explained_by_this_lower_bound'] = {
        k: actual.get(k, 0) - result['lower_bound_unfinished'].get(k, 0)
        for k in sorted(set(actual) | set(result['lower_bound_unfinished']))}
    if any(v < 0 for v in result['not_explained_by_this_lower_bound'].values()):
        raise ValueError('Actual results violate an input lower bound; investigate before reporting')
    result['unfinished_gate_out_records_by_arrival_day'] = [
        dict(day=r['index'] + 1, count=r['n_censored']) for r in cohorts['final'] if r['n_censored']]
    result['caveat'] = ('Aggregate job counts and cohort gate-out censoring do not identify individual '
                        'unfinished jobs. Bounds before cutoff do not establish physical feasibility.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed-bundle', type=Path, required=True)
    ap.add_argument('--seed-sha256', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if hashlib.sha256(args.seed_bundle.read_bytes()).hexdigest() != args.seed_sha256:
        raise ValueError('Input bundle checksum mismatch')
    with gzip.open(args.seed_bundle, 'rt', encoding='utf-8') as f:
        document = json.load(f)
    result = analyze(document, args.run)
    result.update(seed=document['seed'], seed_sha256=args.seed_sha256,
                  report_sha256=hashlib.sha256((args.run/'report.json').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in result.items() if k != 'jobs'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
