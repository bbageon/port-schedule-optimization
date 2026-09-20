# Frozen-model independent evaluation

20 independent continuous months; three policies per month. No new training.

| Policy | Mean 28-day cost (KRW) | Mean 30-day cost (KRW) | Unfinished trucks (sum) | Unfinished vessel jobs (sum) |
|---|---:|---:|---:|---:|
| NO_REALLOC | 21,773,023,232 | 22,549,993,002 | 9,572 | 24,683 |
| RL | 19,378,728,527 | 20,152,497,448 | 12,688 | 22,336 |
| RL_TIME | 34,398,779,263 | 35,096,599,372 | 27,298 | 35,801 |

Lower recorded cost alone does not establish improvement when work is unfinished.

| Comparison (positive = saving) | Mean saving (KRW) | Interval | Ratio-of-sums saving | Wins / 20 |
|---|---:|---|---:|---:|
| 28d:NO_REALLOC-RL | 2,394,294,704 | 97.5% [-11,727,356,716, 26,385,129,467] | 11.00% | 4 |
| 28d:RL_TIME-RL | 15,020,050,736 | 97.5% [-4,795,002,032, 41,907,763,522] | 43.66% | 7 |
| 28d:NO_REALLOC-RL_TIME | -12,625,756,032 | 95% [-42,372,594,603, 16,947,321,062] | -57.99% | 16 |
| 30d:NO_REALLOC-RL | 2,397,495,554 | 95% [-10,586,459,490, 21,876,064,574] | 10.63% | 4 |
| 30d:RL_TIME-RL | 14,944,101,924 | 95% [-4,012,250,840, 37,976,889,768] | 42.58% | 7 |
| 30d:NO_REALLOC-RL_TIME | -12,546,606,370 | 95% [-42,380,253,588, 17,107,158,131] | -55.64% | 16 |

Intervals resample whole months (20,000 draws), not days. Two primary 28-day comparisons use 97.5% intervals.
Full vs time-only measures the contribution of permitting spatial actions with the same fixed networks.
The synthetic environment has no appointment quota or new external waiting/rescheduling charge.
Per-day observations, all seeds, unsuccessful outcomes, and exact input/model hashes are retained.
Final operational-validity and paper-claim judgment remains separate from this computed table.
