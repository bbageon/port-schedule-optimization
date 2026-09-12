"""Deterministic, checksummed JSON seed bundles (not executable pickle files)."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .container_contract import ContainerContractError
from .fixed_seed import canonical_bytes, fixed_seed_audit


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_seed_bundle(document, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve existing seed data: {path}')
    audit = fixed_seed_audit(document)
    if not audit['passed']:
        raise ContainerContractError('Refusing invalid seed data', report=audit)
    payload = canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as stream:
            stream.write(payload)
    return dict(path=path.name, sha256=file_sha256(path), bytes=path.stat().st_size,
                content_sha256=hashlib.sha256(payload).hexdigest(), uncompressed_bytes=len(payload))


def load_seed_bundle(path, *, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise ContainerContractError('Seed file checksum mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        document = json.load(source)
    audit = fixed_seed_audit(document)
    if not audit['passed']:
        raise ContainerContractError('Loaded seed data violates the input contract', report=audit)
    return document, audit
