"""Comparison coverage at the same sampled times and the same rollout budget."""
from collections import Counter
import hashlib


class CoverageSampler:
    def __init__(self, seed):
        self.seed = int(seed)
        self.cranes = Counter()
        self.flows = Counter()
        self.n = 0

    def tie(self, *parts):
        return hashlib.sha256(repr((self.seed, self.n, parts)).encode()).digest()

    @staticmethod
    def flow(sim, gc):
        job = sim.jobs.get(gc.job_ref.job_id)
        return '' if job is None else job.flow.value

    def select(self, sim, block, choices, rank):
        crane, picked, alternatives = min(choices, key=lambda item:
            (self.cranes[(block, item[0])], self.tie(block, item[0])))
        picked_flow = self.flow(sim, picked)
        if self.n % 2 == 0:
            alt = min(alternatives, key=lambda g: (rank(sim, crane, g), g.job_ref.job_id))
        else:
            alt = min(alternatives, key=lambda g:
                (self.flows[(picked_flow, self.flow(sim,g))], self.tie(block, crane, g.job_ref.job_id)))
        self.cranes[(block, crane)] += 1
        self.flows[(picked_flow, self.flow(sim, alt))] += 1
        self.n += 1
        return crane, picked, alt
