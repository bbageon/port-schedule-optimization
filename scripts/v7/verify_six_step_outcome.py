"""Verify retained experiment inputs, models, common trajectories and test evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from benchmark_default_eval import save


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder):
    root = Path(folder)
    learning = root / 'learning'
    checks = []

    def require(name, condition, **evidence):
        checks.append(dict(check=name, passed=bool(condition), **evidence))
        assert condition, checks[-1]

    tests = read(root / 'prepared-verification.json')
    require('recorded implementation checks', tests['all_pass'] and tests['unique_checks'] == 160)
    for path, digest in tests['evidence'].items():
        require('test evidence unchanged', sha(root / path) == digest, path=path)
    for path, digest in tests['source_sha256'].items():
        require('tested source unchanged', sha(path) == digest, path=path)

    markers = [read(learning / f'campaign-stage-{stage}.json') for stage in (4, 5, 6)]
    for marker in markers:
        require('campaign completed', len(marker['completed']) == {4: 36, 5: 12, 6: 12}[marker['stage']],
                stage=marker['stage'], jobs=len(marker['completed']), commit=marker['commit'])
        require('learning runtime identical', marker['sources'] == markers[0]['sources'], stage=marker['stage'])
        require('same runtime options', marker['workers'] == 4 and marker['existence_scan']
                and not marker['shared_slots'], stage=marker['stage'])
    for path, digest in markers[0]['sources'].items():
        require('campaign source unchanged', sha(path) == digest, path=path)

    evaluation_inputs = {}
    for path in sorted((learning / 'evaluation').glob('*.json.gz')):
        data = read(path)
        digest = data['probe']['metadata']['input_sha256']
        expected = evaluation_inputs.setdefault(data['load'], digest)
        require('same evaluation input for all arms', digest == expected, path=path.as_posix())
        if data['checkpoint']:
            require('evaluated checkpoint unchanged', sha(data['checkpoint']) == data['checkpoint_sha256'],
                    path=path.as_posix())
    require('all prescribed evaluations retained', len(list((learning / 'evaluation').glob('*.json.gz'))) == 51)

    for base in sorted(learning.glob('model-*')):
        for it in range(3):
            old = read(base / f'labels-4-{it}.json.gz')
            new = read(base / f'labels-5-{it}.json.gz')
            digest = sha(base / f'behavior-{it}.pt')
            require('identical collection model', old['behavior_sha256'] == new['behavior_sha256'] == digest,
                    model=base.name, iteration=it)
            require('identical collection input', (old['seed'], old['load']) == (new['seed'], new['load']),
                    model=base.name, iteration=it)
            a, b = old['result'], new['result']
            require('identical actual collection choices',
                    a['crane']['main_choices_sha256'] == b['crane']['main_choices_sha256'],
                    model=base.name, iteration=it)
            for key in ('phi_krw', 'admitted', 'c_wait', 'c_vessel', 'c_move', 'c_rehandle', 'n_censored'):
                require('identical actual collection outcome', a[key] == b[key],
                        model=base.name, iteration=it, metric=key)
            times = lambda x: [(r['key'][0], r['key'][1]) for r in x['crane']['comparisons']]
            require('identical comparison times and blocks', times(a) == times(b), model=base.name, iteration=it)
            for stage, data in ((4, old), (5, new)):
                cs = data['result']['crane']
                require('no reported branch first-action errors',
                        cs['factual_mismatch'] == cs['force_failed'] == cs['hook_errors'] == 0,
                        model=base.name, iteration=it, stage=stage)
        flow = read(base / 'stage-6.json')
        for row in flow['rows']:
            path = base / f"labels-5-{row['it']}.json.gz"
            require('flow input uses unchanged stage5 corpus', row['corpus_sha256'] == sha(path)
                    and row['collection_seconds'] == 0, model=base.name, iteration=row['it'])

    for stage in (4, 5, 6):
        summary = read(learning / f'stage{stage}-summary.json')
        for path, digest in summary['artifact_sha256'].items():
            require('stage snapshot artifact unchanged', sha(path) == digest, stage=stage, path=path)
    replay = read(learning / 'fit-replay-verification.json')
    require('all retained fits reproduced', replay['all_exact'] and len(replay['checks']) == 36)
    for item in replay['checks']:
        require('replayed checkpoint unchanged', sha(item['checkpoint']) == item['checkpoint_sha256'],
                path=item['checkpoint'])
    for path in root.rglob('*.md'):
        require('report within line limit', len(path.read_text(encoding='utf-8').splitlines()) <= 200,
                path=path.as_posix())
    result = dict(all_pass=True, checks=checks, n_checks=len(checks),
                  scope='artifact and controlled-comparison integrity, not statistical or port-realism validation',
                  limitation='branch runtime checks validate the first action, not every subsequent event')
    save(root / 'verification-final.json', result)
    print(f'{len(checks)} artifact and controlled-comparison checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    verify(parser.parse_args().folder)
