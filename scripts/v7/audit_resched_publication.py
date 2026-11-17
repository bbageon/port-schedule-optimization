"""Verify that the published Git objects retain every referenced evidence byte."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from resched_ppo import OUT, save
from six_step_learning import sha


def main(ref):
    campaign=json.loads((OUT/'campaign.json').read_text('utf-8'))
    summary=json.loads((OUT/'summary.json').read_text('utf-8'))
    expected={**campaign['sources'],**summary['artifact_hashes']}
    reference=json.loads((OUT/'reference/sources.json').read_text('utf-8'))
    for path,obj in reference['files'].items():
        expected[(OUT/'reference'/path).as_posix()]=obj['sha256']
    for name in ('README.md','summary.json','campaign.json','schedule-window.json','schedule-window.png',
                 'schedule-window.svg','inference-benchmark.json','gradient-diagnostic.json','verification.json'):
        p=OUT/name;expected[p.as_posix()]=sha(p)
    prefix=':' if ref=='index' else ref+':'
    raw=subprocess.check_output(['git','cat-file','--batch'],
        input=''.join(prefix+p+'\n' for p in expected).encode('utf-8'))
    at=0;mismatch=[]
    for path,h in expected.items():
        end=raw.index(b'\n',at);head=raw[at:end].split()
        if len(head)!=3:raise ValueError(('Missing evidence from Git',path,head))
        size=int(head[2]);body=raw[end+1:end+1+size];at=end+size+2
        if hashlib.sha256(body).hexdigest()!=h:mismatch.append(path)
    result=dict(passed=not mismatch,ref=ref,files=len(expected),mismatches=mismatch)
    save(OUT/'publication-byte-audit.json',result)
    assert not mismatch,mismatch
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',default='index');main(p.parse_args().ref)
