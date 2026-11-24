"""YR-332-bo, added after the registered result: how much does the C1 error cut move with the fit's random start?

Same rows, split, features and fitting settings as vessel_progress_check; only the value-head seed changes.
This describes the margin of the registered verdict and does not replace it.
"""
import math

import numpy as np
import torch

from local_return_scope import contexts, residual
from vessel_progress_check import FIT, HELD_OUT, MONTHS, SEED, manifest, month
from vessel_progress_record import OUT, save


SEEDS = [SEED + k for k in range(10)]   # the first one is the registered fit


def main():
    order, blocks, block_of = manifest()
    rows = [r for k in range(MONTHS) for r in month(k, order, blocks, block_of)]
    target = np.array([r["target"] for r in rows])
    month_of = np.array([r["month"] for r in rows])
    fit = np.isin(month_of, FIT) & (target > 0)
    test = np.isin(month_of, HELD_OUT) & (target > 0)
    context = contexts(rows)
    number = torch.tensor([[math.log1p(r["number"])] for r in rows], dtype=context.dtype)
    widened = torch.cat([context, number], 1)
    runs = []
    for seed in SEEDS:
        without = float(np.sqrt(np.mean(residual(context, target, fit, test, seed) ** 2)))
        with_number = float(np.sqrt(np.mean(residual(widened, target, fit, test, seed) ** 2)))
        runs.append(dict(seed=seed, rmse_without=without, rmse_with=with_number, error_cut=1 - with_number / without))
        print(seed, round(without, 2), round(with_number, 2), round(runs[-1]["error_cut"], 4), flush=True)
    cuts = [r["error_cut"] for r in runs]
    save(OUT / "seed-sensitivity.json", dict(
        purpose="Added after the registered result; describes the margin of C1 and does not change the verdict",
        runs=runs, error_cut_min=min(cuts), error_cut_median=float(np.median(cuts)), error_cut_max=max(cuts),
        at_or_above_threshold=sum(c >= .30 for c in cuts), seeds=len(cuts)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
