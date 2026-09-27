"""Copy completed immutable run evidence; compress large logs without dropping rows."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plain = ('manifest.json', 'report.json', 'status.json', 'days.json', 'day-snapshots.json',
        'input-audit.json', 'admissions.json', 'container_contract.json', 'month-result.json',
        'seed-data.json.gz', 'execution-records.json.gz', 'policy-final.pt', 'calibration.json',
        'day_01.pt', 'day_02.pt', 'day_03.pt', 'day_04.pt', 'day_05.pt')
    compressed = ('completed-jobs.json', 'workload-samples.json', 'events.jsonl', 'trade-ledger.json')
    copies = []
    for run in sorted(source.iterdir()):
        if not run.is_dir() or not (run.name.startswith(('train-', 'eval-')) or run.name == 'calibration-v2'):
            continue
        if not (run/'report.json').exists():
            continue
        if json.loads((run/'report.json').read_text(encoding='utf-8'))['status'] != 'complete':
            continue
        target = destination/run.name
        target.mkdir(exist_ok=True)
        for name in plain:
            src, dst = run/name, target/name
            if not src.exists():
                continue
            shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
            copies.append(dict(path=dst.relative_to(destination).as_posix(), sha256=sha(dst), bytes=dst.stat().st_size))
        for name in compressed:
            src, dst = run/name, target/(name+'.gz')
            if not src.exists():
                continue
            with dst.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed_file:
                    compressed_file.write(src.read_bytes())
            copies.append(dict(path=dst.relative_to(destination).as_posix(), sha256=sha(dst),
                               bytes=dst.stat().st_size, original_sha256=sha(src)))
    for name in ('prereg-executed.json', 'paired-controls.json', 'checkpoint-audit.json',
                 'event-audit.json', 'calibration-audit.json', 'input-preflight.json',
                 'pytest.txt', 'result.json', 'reported-values.json', 'report.md'):
        path = destination/name
        if path.exists():
            copies.append(dict(path=name, sha256=sha(path), bytes=path.stat().st_size))
    for path in sorted((destination/'validation').glob('*.json')):
        copies.append(dict(path=path.relative_to(destination).as_posix(), sha256=sha(path), bytes=path.stat().st_size))
    index = dict(schema='yr331.evidence.v1', source_directory=str(source), files=copies,
                 total_bytes=sum(x['bytes'] for x in copies), note='Incomplete runs are retained in source, never marked complete here.')
    (destination/'artifacts.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(files=len(copies), bytes=index['total_bytes'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, default=Path('outputs/reports/yr331_training'))
    a = p.parse_args()
    archive(a.source, a.destination)
