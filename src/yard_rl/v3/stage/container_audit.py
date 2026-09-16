"""Observe physical container production/removal before daily job pruning.

No actions, inventory, requests or policy features are changed. Container IDs are
scoped to their physical block because initial synthetic IDs repeat across blocks.
"""
from collections import Counter
from ..world.domain.enums import JobStatus, ServiceMode


class ContainerFlowAudit:
    def __init__(self, mbt):
        self.sources = {(b, cid): {'source_job': None, 'source_flow': 'INITIAL', 'available_s': 0.0}
                        for b, sim in mbt.blocks.items() for cid in sim.stacks.containers}
        self.initial_count = len(self.sources)
        self.observed_jobs, self.consumed = set(), {}
        self.links, self.issues = [], []
        self.snapshots = 0

    def observe(self, mbt):
        completed = [(b, j) for b, sim in mbt.blocks.items() for j in sim.jobs.values()
                     if j.status == JobStatus.DONE and (b, j.job_id) not in self.observed_jobs]
        # A producer and its consumer may both finish within the same day.
        for b, j in completed:
            if j.service_mode != ServiceMode.STORE:
                continue
            key = (b, f'IN_{j.job_id}')
            if key in self.sources:
                self.issues.append(f'duplicate container production: {key}')
            self.sources[key] = {'source_job': j.job_id, 'source_flow': j.flow.value,
                                 'available_s': j.service_end}
        for b, j in completed:
            self.observed_jobs.add((b, j.job_id))
            if j.service_mode != ServiceMode.RETRIEVE:
                continue
            key = (b, j.target_container)
            source = self.sources.get(key)
            if source is None:
                self.issues.append(f'removal without source: {key} by {j.job_id}')
            elif (source['available_s'] is None or j.service_start is None
                  or source['available_s'] > j.service_start + 1e-6):
                self.issues.append(f'removal before physical supply: {j.job_id}')
            if key in self.consumed:
                self.issues.append(f'container removed twice: {key}')
            self.consumed[key] = j.job_id
            self.links.append({'job_id': j.job_id, 'flow': j.flow.value, 'block': b,
                               'target': j.target_container, 'service_start_s': j.service_start,
                               'service_end_s': j.service_end, 'source': source})
        expected = set(self.sources) - set(self.consumed)
        actual = {(b, cid) for b, sim in mbt.blocks.items() for cid in sim.stacks.containers}
        if expected != actual:
            self.issues.append(f'physical stock mismatch at snapshot {self.snapshots}: '
                               f'missing={len(expected-actual)}, unexpected={len(actual-expected)}')
        self.snapshots += 1

    def finish(self, mbt):
        self.observe(mbt)
        return {'passed': not self.issues, 'issues': self.issues, 'snapshots': self.snapshots,
                'initial_containers': self.initial_count,
                'produced_containers': len(self.sources) - self.initial_count,
                'completed_removals': len(self.links),
                'remaining_containers': len(set(self.sources) - set(self.consumed)),
                'source_flows': dict(Counter(r['source']['source_flow'] for r in self.links if r['source'])),
                'scope': 'physical quantity and identity chain, not real cargo-manifest or customer identity validation'}, self.links
