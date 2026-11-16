"""Check that staged experiment metadata preserves the bytes its hashes describe."""
import hashlib
from pathlib import Path
import subprocess

from audit_day_shapes import read
from context_learning import OUT
from yard_rl.integrated.repro import write_result

ROOTS = [Path('outputs/reports') / p for p in (
    'yr332_synthetic_inputs', 'yr332_parametric_arrivals',
    'yr323_learning_diagnosis', 'yr332_context_learning')]


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', *map(str, ROOTS)])
    paths = {Path(p.decode('utf-8')) for p in tracked.split(b'\0') if p}
    paths = {p for p in paths if p.suffix in ('.json', '.sha256', '.md')}
    frozen = read(OUT / 'campaign.json')['sources']
    for name in frozen:
        paths.add(Path(name).resolve().relative_to(Path.cwd()))
    ordered = sorted(paths)
    names = [p.as_posix() for p in ordered]
    response = subprocess.check_output(['git', 'cat-file', '--batch'],
        input=''.join(':'+n+'\n' for n in names).encode('utf-8'))
    cursor = 0
    blobs = {}
    for name in names:
        end = response.index(b'\n', cursor)
        header = response[cursor:end].split()
        assert len(header) == 3 and header[1] == b'blob', (name, header)
        size = int(header[2]); start = end + 1
        blob = response[start:start + size]
        assert blob == Path(name).read_bytes(), ('Git changes artifact bytes', name)
        blobs[name] = blob
        cursor = start + size + 1
    assert cursor == len(response)
    for name, expected in frozen.items():
        rel = Path(name).resolve().relative_to(Path.cwd()).as_posix()
        assert hashlib.sha256(blobs[rel]).hexdigest() == expected, rel
    sidecar_count = 0
    for name, blob in blobs.items():
        if name.endswith('.sha256'):
            source = name[:-7]
            if source in blobs:
                assert hashlib.sha256(blobs[source]).hexdigest() == blob.decode('utf-8').strip(), source
                sidecar_count += 1
    result = dict(staged_metadata_equal_worktree=len(blobs), frozen_sources_equal_git=len(frozen),
                  sidecar_pairs_valid_in_git=sidecar_count,
                  roots=[p.as_posix() for p in ROOTS],
                  correction='Scope Git -text attributes to these four artifact trees; preserve original bytes and original hash values.',
                  scope='Packaging correction only. No simulation, model, label, or metric values changed. Earlier commits may normalize CRLF; this publication fixes the stored bytes.')
    write_result(OUT / 'publication-byte-audit.json', result)
    print(result)


if __name__ == '__main__':
    main()
