"""Event-maintained job status membership; no extra policy information."""
from __future__ import annotations

import weakref

_MISSING = object()


class IndexedJobs(dict):
    """Dict-compatible ownership table, including direct Job.status assignments.

    A job has one owner. Removed jobs detach before cross-block insertion.
    Serialization reconstructs membership, never copies a callback to another world.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._by_status = {}
        self._ordered = {}
        self.update(*args, **kwargs)

    def _changed(self, job, old, new):
        if self.get(job.job_id) is not job or old == new:
            return
        self._by_status.setdefault(old, set()).discard(job.job_id)
        self._by_status.setdefault(new, set()).add(job.job_id)
        self._ordered.clear()

    def __setitem__(self, key, job):
        if key != job.job_id:
            raise ValueError('job key must equal job_id')
        owner_ref = getattr(job, '_job_index_ref', None)
        owner = owner_ref() if owner_ref else None
        if owner is not None and owner is not self:
            raise ValueError('remove job from its owner before moving it')
        if key in self:
            self.__delitem__(key)
        super().__setitem__(key, job)
        job._job_index_ref = weakref.ref(self)
        self._by_status.setdefault(job.status, set()).add(key)
        self._ordered.clear()

    def __delitem__(self, key):
        job = self[key]
        self._by_status[job.status].discard(key)
        job.__dict__.pop('_job_index_ref', None)
        super().__delitem__(key)
        self._ordered.clear()

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, default=_MISSING):
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self[key]
        del self[key]
        return value

    def popitem(self):
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def clear(self):
        for job in self.values():
            job.__dict__.pop('_job_index_ref', None)
        super().clear()
        self._by_status.clear()
        self._ordered.clear()

    def __ior__(self, other):
        self.update(other)
        return self

    def ids(self, *statuses):
        key = tuple(statuses)
        if key not in self._ordered:
            members = set().union(*(self._by_status.get(s, ()) for s in statuses))
            self._ordered[key] = tuple(sorted(members))
        return self._ordered[key]

    def count(self, status):
        return len(self._by_status.get(status, ()))

    def __reduce_ex__(self, protocol):
        return type(self), (), None, None, iter(self.items())


def job_ids(sim, *statuses):
    """Fast prefilter; callers retain the original status predicates.

    Plain dictionaries deliberately use the original full scan as an oracle.
    """
    if isinstance(sim.jobs, IndexedJobs):
        return sim.jobs.ids(*statuses)
    return tuple(sorted(sim.jobs))
