"""[[YR-242]] 실측 — SF-SPT 정렬키 (실작업?, 소요, 본선?, -누적대기, job_id) 에서
   실제로 1등을 정하는 자리가 어디인가 — 실측."""
import collections
from yard_rl.v3.world.integrated import resolver as R

POS = ["0 실작업인가", "1 소요시간", "2 본선인가", "3 누적대기", "4 job_id"]
DEPTH = collections.Counter()
NSERVE = collections.Counter()
VESSEL_PRESENT = collections.Counter()

_res = R.CentralResolver.resolve


def resolve(self, sim, decision, gen_by_crane):
    for c in decision.crane_ids:
        serves = [g for g in gen_by_crane[c].items
                  if g.feasible and g.kind.name == "SERVE"]
        NSERVE[min(len(serves), 9)] += 1
        if len(serves) < 2:
            continue
        # 본선·트럭이 함께 걸린 결정인가 (본선 우선이 '작동할 기회'가 있었나)
        kinds = {bool(g.job_ref.is_vessel) for g in serves if g.job_ref}
        mixed = len(kinds) > 1
        VESSEL_PRESENT["기회있음" if mixed else "동종뿐"] += 1
        keys = sorted(tuple(self.preference.rank(sim, c, g)) for g in serves)
        a, b = keys[0], keys[1]
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                DEPTH[i] += 1
                if mixed:
                    DEPTH[("mixed", i)] += 1
                break
        else:
            DEPTH[99] += 1
    return _res(self, sim, decision, gen_by_crane)


R.CentralResolver.resolve = resolve

from yard_rl.v3.stage.episode import run_episode

print("== 부하 5,000 · 하루 · SF-SPT · 재배치 없음 ==", flush=True)
run_episode(load=5_000, arm="NO_REALLOC", seed=9_900_777)
tot = sum(DEPTH[i] for i in range(5)) + DEPTH[99]
print(f"\nSERVE 후보가 2개 이상이던 크레인-결정: {tot:,}건\n")
print("── 1등과 2등이 **어느 자리**에서 갈렸나 ──")
for i, nm in enumerate(POS):
    n = DEPTH[i]
    if tot:
        print(f"  {nm:<12} {n:>8,}건  {n/tot*100:5.1f}%")
print(f"  {'완전동점':<12} {DEPTH[99]:>8,}건")
mt = sum(v for k, v in DEPTH.items() if isinstance(k, tuple))
print(f"\n── 본선·트럭이 **같이 걸린** 결정 {VESSEL_PRESENT['기회있음']:,}건 중 ──")
for i, nm in enumerate(POS):
    n = DEPTH[("mixed", i)]
    if mt:
        print(f"  {nm:<12} {n:>8,}건  {n/mt*100:5.1f}%")
print("\n── 한 크레인이 고를 수 있던 SERVE 후보 수 ──")
for k in sorted(NSERVE):
    print(f"  {k}{'개 이상' if k == 9 else '개'}: {NSERVE[k]:,}")
print(f"\nΦ = {r.phi:,.0f}원")
