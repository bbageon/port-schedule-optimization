"""Policy-independent stock balances for the fixed turnaround cohort."""
from collections import Counter

from global_policy_inputs import bundle
from vessel_turnaround_experiment import DRAIN, LOADS, OUT, save
from yard_rl.v7.stage.orders import build_stage
from yard_rl.v7.world.integrated.profiles import build_h21_profile


def main():
    profile = build_h21_profile()
    geometry = profile.block
    capacity = geometry.bay_count * geometry.row_count * geometry.tier_max
    runs = [("train", i, LOADS[i % 3]) for i in range(12)]
    runs += [("holdout", i, load) for i in range(3) for load in LOADS]
    manifests, balances = [], []
    for split, index, load in runs + [("train",12,3500)]:
        built = build_stage(**bundle().case(split,index,load), profile=profile, vessel_drain_s=DRAIN)
        trucks = Counter((r["block"],r["flow"]) for r in built["schedule"])
        streams, clipped = [], []
        for r in built["vessel_schedule"]:
            block = r["block"]
            scenario = built["scenarios"][block]
            actual = scenario.vessels[0].plan.total_moves
            if r["moves"] != actual:
                clipped.append([block,r["moves"],actual])
            initial = len(scenario.containers)
            adds, removals = trucks[block,"GATE_IN"], trucks[block,"GATE_OUT"]
            discharge = r["work"] == "DISCHARGE"
            moves = r["moves"]
            final = initial + adds - removals + (moves if discharge else -moves)
            # Give the ship every outbound slot and every pipeline slot before
            # admitting even one inbound truck. A positive deficit is unavoidable.
            impossible = max(0,initial-removals+moves-capacity-profile.transfer.n_units) if discharge else 0
            streams.append(dict(block=block,vessel=r["vessel_id"],work=r["work"],moves=moves,
                                init=initial,gate_in=adds,gate_out=removals,cap=capacity,final=final,
                                unserviceable_even_without_gate_in=impossible))
        case = dict(split=split,index=index,load=load)
        balances.append(dict(**case,streams=streams))
        manifests.append(dict(**case,vessels=len(built["fleet"]),moves=sum(r["moves"] for r in streams),
                              input_moves=sum(s.moves for s in built["fleet"]),clipped=clipped))
    assert all(not r["clipped"] and r["moves"] == r["input_moves"] for r in manifests)
    save(OUT/"manifest-check.json",manifests)
    save(OUT/"capacity-check.json",sorted(balances,key=lambda r: r["index"] != 12))


if __name__ == "__main__":
    main()
