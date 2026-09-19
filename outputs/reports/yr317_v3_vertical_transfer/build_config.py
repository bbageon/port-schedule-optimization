"""Pin the current frozen checkout and existing evaluation inputs for the pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    source, workspace = args.source.resolve(), args.workspace.resolve()
    os.chdir(source)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('The source checkout must be clean')
    sys.path.insert(0, str(source/'src'))
    from yard_rl.v3.layouts import synthetic_vertical_spec
    from yard_rl.v3.world.integrated.profiles import build_h21_profile
    base = Path('outputs/reports/yr317_v3_independent_eval/config.json')
    prereg = Path('outputs/reports/yr317_v3_vertical_transfer/prereg.md')
    out = workspace/prereg.parent/'config.json'
    payload = dict(schema='yr317.vertical-comparison.v1', seed=20_000_000,
        arms=['NO_REALLOC', 'RL_TIME'], base_config=base.as_posix(),
        base_config_sha256=sha(workspace/base), source_commit=head,
        prereg=prereg.as_posix(), prereg_sha256=sha(workspace/prereg),
        environment_spec=synthetic_vertical_spec(build_h21_profile()))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(source_commit=head, config=str(out), config_sha256=sha(out))))


if __name__ == '__main__':
    main()
