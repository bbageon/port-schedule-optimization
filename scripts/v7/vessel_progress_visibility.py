"""Can the value model see thirty minutes of quay progress? Read-only check on the stored YR-332-bl decisions.

The only vessel-progress input is the candidate's slack to its planned finish, clipped at +-2 hours.
Thirty minutes of progress changes that slack only when a vessel candidate is present and unclipped.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from local_return_scope import manifest, month, save, sha

OUT = Path("outputs/reports/yr332_predicted_tail_return")
SLACK, IS_VESSEL, CLIP = 3, 0, 2.0      # feature columns of yard_rl.v7.crane.features.crane_features


def main():
    blocks, streams = manifest()
    totals = dict(decisions=0, controlling=0, controlling_with_vessel_candidate=0,
                  controlling_with_unclipped_slack=0, vessel_rows=0, at_upper=0, at_lower=0, exactly_zero=0, inside=0)
    for update in range(4):
        for index in range(4):
            rows, _ = month(update, index, blocks, streams)
            for r in rows:
                x = r["state"].features
                vessel = x[x[:, IS_VESSEL] == 1, SLACK]
                totals["decisions"] += 1
                totals["vessel_rows"] += len(vessel)
                totals["at_upper"] += int((vessel >= CLIP - 1e-6).sum())
                totals["at_lower"] += int((vessel <= -CLIP + 1e-6).sum())
                totals["exactly_zero"] += int((vessel == 0).sum())
                totals["inside"] += int(((vessel.abs() < CLIP - 1e-6) & (vessel != 0)).sum())
                if r["ships"]:      # this block's part decides when at least one started ship finishes
                    totals["controlling"] += 1
                    totals["controlling_with_vessel_candidate"] += int(len(vessel) > 0)
                    totals["controlling_with_unclipped_slack"] += int(bool(((vessel.abs() < CLIP - 1e-6)
                                                                            & (vessel != 0)).any()))
            print("month", update, index, totals["decisions"], flush=True)
    c, v = totals["controlling"], totals["vessel_rows"]
    summary = dict(
        question="Is thirty minutes of quay progress visible in the state the value model reads?",
        counts=totals,
        share_controlling=c / totals["decisions"],
        controlling_with_vessel_candidate=totals["controlling_with_vessel_candidate"] / c,
        controlling_with_unclipped_slack=totals["controlling_with_unclipped_slack"] / c,
        vessel_candidates=dict(at_plus_2h=totals["at_upper"] / v, at_minus_2h=totals["at_lower"] / v,
                               exactly_zero=totals["exactly_zero"] / v, inside=totals["inside"] / v),
        inputs_not_seen=["remaining moves of this block's part", "progress of the same call at other blocks",
                         "slack beyond +-2 hours", "anything about a vessel when no vessel candidate is offered"],
        provenance=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        source=sha("scripts/v7/vessel_progress_visibility.py"), torch=torch.__version__))
    save(OUT / "vessel-visibility.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
