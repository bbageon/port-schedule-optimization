"""[[YR-242]] 실측 — job_id 로 갈린 33.6% 는 무엇인가 — 본선끼리인가?"""
import collections
from yard_rl.v3.world.integrated import resolver as R

WHO = collections.Counter()      # job_id 로 갈린 결정의 1·2등 정체
DUR = collections.Counter()      # 소요시간 값의 다양성
_res = R.CentralResolver.resolve


def resolve(self, sim, decision, gen_by_crane):
    for c in decision.crane_ids:
        serves = [g for g in gen_by_crane[c].items
                  if g.feasible and g.kind.name == "SERVE"]
        for g in serves:
            if g.plan is not None:
                DUR[round(g.plan.duration_s, 3)] += 1
        if len(serves) < 2:
            continue
        keyed = sorted((tuple(self.preference.rank(sim, c, g)), g) for g in serves)
        (ka, ga), (kb, gb) = keyed[0], keyed[1]
        # 앞 4자리가 모두 같으면 job_id 가 승부를 가른 것
        if ka[:4] == kb[:4]:
            va = bool(ga.job_ref.is_vessel) if ga.job_ref else None
            vb = bool(gb.job_ref.is_vessel) if gb.job_ref else None
            nm = {(True, True): "본선 ↔ 본선", (False, False): "트럭 ↔ 트럭"}
            WHO[nm.get((va, vb), "본선 ↔ 트럭")] += 1
    return _res(self, sim, decision, gen_by_crane)


R.CentralResolver.resolve = resolve

from yard_rl.v3.stage.episode import run_episode

run_episode(load=5_000, arm="NO_REALLOC", seed=9_900_777)
tot = sum(WHO.values())
print(f"\n== job_id 가 승부를 가른 {tot:,}건의 정체 ==")
for k, v in WHO.most_common():
    print(f"  {k:<14} {v:>7,}건  {v / tot * 100:5.1f}%")
n = sum(DUR.values())
print(f"\n== 소요시간 값 == 후보 {n:,}건이 서로 다른 값 {len(DUR):,}개를 나눠 씀")
print("  가장 흔한 값 8개 (이 값에 몰릴수록 동점이 잦다):")
for v, c in DUR.most_common(8):
    print(f"    {v:>9,.1f}초 : {c:>6,}건 ({c / n * 100:4.1f}%)")
