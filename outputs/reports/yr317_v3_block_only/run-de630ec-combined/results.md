# Four-policy fixed-model comparison

20 independent continuous months, four policies each, no additional training.

| Policy | Mean 28-day cost (KRW) | Mean 30-day cost (KRW) | Unfinished trucks | Unfinished vessel jobs |
|---|---:|---:|---:|---:|
| NO_REALLOC | 21,773,023,232 | 22,549,993,002 | 9,572 | 24,683 |
| RL | 19,378,728,527 | 20,152,497,448 | 12,688 | 22,336 |
| RL_TIME | 34,398,779,263 | 35,096,599,372 | 27,298 | 35,801 |
| RL_SPACE | 33,837,153,972 | 34,858,691,572 | 27,235 | 29,419 |

Lower recorded cost alone does not establish improvement when work is unfinished.

| Added comparison (positive = saving) | Mean saving (KRW) | Interval | Saving ratio | Wins / losses / ties |
|---|---:|---|---:|---|
| 28d:NO_REALLOC-RL_SPACE | -12,064,130,740 | 97.5% [-29,280,439,736, 14,039,065,064] | -55.41% | 3 / 17 / 0 |
| 28d:RL_SPACE-RL | 14,458,425,444 | 97.5% [7,805,246,931, 22,579,089,824] | 42.73% | 19 / 1 / 0 |
| 30d:NO_REALLOC-RL_SPACE | -12,308,698,569 | 95% [-27,886,910,301, 9,805,656,011] | -54.58% | 2 / 18 / 0 |
| 30d:RL_SPACE-RL | 14,706,194,123 | 95% [8,651,983,196, 21,737,580,581] | 42.19% | 19 / 1 / 0 |

Positive baseline-minus-block-only savings show the effect of allowing spatial actions alone.
Positive block-only-minus-full savings show the further effect of allowing temporal actions.
The original three-policy analysis is preserved unchanged in summary.json: original_analysis.
All intervals resample whole months, using 20,000 draws. The two added 28-day comparisons use
97.5% intervals as a separate comparison family; 30-day intervals are descriptive.

| Scheduled trucks/day | Baseline mean day cost | Full mean day cost | Time-only mean day cost | Block-only mean day cost |
|---:|---:|---:|---:|---:|
| 3500 | 359,702,808 | 278,328,431 | 528,940,472 | 443,935,439 |
| 5000 | 301,313,799 | 312,061,701 | 637,130,983 | 679,513,397 |
| 7500 | 419,123,434 | 502,856,622 | 1,103,628,124 | 1,124,649,781 |
| 12500 | 1,815,758,045 | 1,980,316,318 | 2,790,499,209 | 3,036,478,955 |
| 15000 | 4,106,842,460 | 2,552,626,838 | 4,485,185,964 | 3,861,110,246 |

Load groups describe dependent days 1..28, with carryover state; no day-level significance test is made.
The full per-month load-group costs are saved in summary.json. Lower volume does not guarantee low backlog.
All losses and unfinished work remain in the tables. Claim eligibility remains false pending scientific review.
This synthetic experiment adds neither appointment quotas nor external waiting/rescheduling charges.
