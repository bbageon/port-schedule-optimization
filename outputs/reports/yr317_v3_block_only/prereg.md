# YR-317-d addendum: twenty fixed Block-only evaluations

2026-09-17: the user requested RL_SPACE in addition to the running three-policy batch.
This addendum is registered before any new independent RL_SPACE result is observed.
The original 20 x 3 preregistration/configuration and ongoing runs remain unchanged.
At addition, one baseline month was complete and other policies had partial daily results.
This is an explicit later addition, not a claim that four policies were registered initially.

## Fixed design

- Add RL_SPACE on exactly the same twenty seeds: 20,000,000 through 21,900,000, step 100,000.
- Use the same ckpt_029.pt checkpoint (SHA-256 50f2da2e8b3070dbba48273f0de5debdf5514fe96f0fa1f5a8f25b8a830cba99).
- Keep the model fixed; no training, exploration, replacement seeds or outcome-based exclusions.
- RL_SPACE is the existing RL candidate restriction: spatial actions permitted, temporal actions disabled.
- Retain the acceptance network and the same crane dispatch policy. This is not separately trained Block-only.
- Preserve the exact request curves, daily totals, original inputs, COUNT_BALANCED vessel plan and PRESERVE admissions.
- Fresh engine/process per seed, continuous thirty days plus the existing two-hour closing interval.
- Same 28 measurement days and supplementary thirty-day totals. One month, not a day, is an independent unit.
- Save the same daily observations, costs, requests, container links, unfinished work and input/model hashes.
- Do not add appointment quotas, rescheduling fees, peak-width changes or simulation-source changes.

## Checks and failure handling

- Use seed 9,900,722 with two days x sixty trucks for a non-performance diagnostic.
- Require the diagnostic to exercise spatial changes and record zero temporal changes.
- Every RL_SPACE month must have zero temporal changes, unchanged networks and matched frozen input hashes.
- Require existing recording, queue, physical container linkage and request conservation checks.
- Unfinished work and policy losses remain results; they do not justify deleting or replacing samples.
- A failed worker stops new add-on launches; other active add-on workers finish and save their evidence.
- Never terminate or mutate the primary supervisor or its children. Preserve original failure/partial artifacts.

## Resources

- All new simulation workers use cores 0 through 19 only; the existing unrelated core-23 run is untouched.
- Reserve the primary supervisor's entire CPU capacity until it finishes, including gaps between workers.
- Count the external simulation against a total sixteen-simulation limit; reserve three GiB per worker.
- Also reserve two GiB for supervisors and twenty percent of total physical memory.
- Limit new launches by remaining memory headroom. Initially at most two additional workers fit the total budget; available memory may reduce this further.
- After the primary batch finishes, its released cores and memory can serve remaining RL_SPACE months.

## Analysis fixed for the addition

- Merge only after all twenty seeds have all four verified policy results: eighty runs in total.
- Check request identities, daily plans, checkpoint, network identities and simulation/runtime contracts.
- Preserve the original three-policy analysis unchanged, under original_analysis.
- Additional comparisons: NO_REALLOC minus RL_SPACE; RL_SPACE minus RL. Positive means the latter policy saves cost.
- Bootstrap twenty paired months together, 20,000 resamples, RNG seed 9,900,723.
- For these two additional 28-day comparisons use 97.5 percent intervals; thirty-day comparisons use descriptive 95 percent intervals.
- The original comparison family and added comparison family are reported separately, with the addition date.
- Report per-seed costs, ratios, wins/losses, completion and residual work. Never promote lower cost alone to operational success.
- Group 28 measurement days by the existing planned daily load only for descriptive mechanism analysis.
- Low planned load does not itself prove low realized congestion; use saved backlog and queue observations to interpret it.
- Do not treat these groups as independent daily samples, or claim independently optimized spatial-policy superiority.
- Full study status remains in progress; acceptance ablation and other review tasks are not completed by this addition.
