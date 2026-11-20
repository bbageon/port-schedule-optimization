"""Read-only source helpers for the inbound disparity audit; no identifiers exported."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics as st

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / 'outputs/reports/yr332_global_policy'
OUT = BASE / 'outputs/reports/yr332_inbound_realism'
RAW = BASE / '오더데이터'


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                                      allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def stats(values):
    x = sorted(values)
    if not x:
        return dict(n=0)
    return dict(n=len(x), mean=st.mean(x), median=st.median(x), min=x[0], max=x[-1],
                p90=x[round((len(x)-1)*.9)], negative=sum(v < 0 for v in x))


def cases():
    return read(SOURCE / 'campaign.json')['configuration']['evaluation_cases']


def label(case):
    if case['stage'] == 'initial':
        return 'initial'
    return case['arm']
