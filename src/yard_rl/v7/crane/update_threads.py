"""CPU threads for the policy update step only (YR-332-bu).

Month rollouts run side by side on one thread each. The update step runs alone, so it may use more of the CPU.
A multi-thread update can differ from a one-thread update in the last digits, because sums are split differently.
"""
from contextlib import contextmanager

import torch


UPDATE_THREADS = 16


@contextmanager
def update_threads(n=UPDATE_THREADS):
    """Use `n` intra-op threads inside the block and restore the previous count afterwards, also on error."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("Thread count must be a positive integer")
    before = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield n
    finally:
        torch.set_num_threads(before)
