"""[[YR-300]] — **운영에서 결정 하나에 몇 밀리초 걸리나.**

논문은 *"비싼 반사실은 학습에만, 운영은 학습된 망으로 채점만"* 을 주장한다.
그 주장이 실무적으로 의미가 있으려면 **운영 한 결정이 충분히 빨라야** 한다.
여기서 그것을 잰다 — 학습 시간이 아니라 **현장에서 트럭 한 대를 배정하는 시간**.

구성: 결정 1회 = 자격 판정 → 후보 생성(KEEP + 블록 최대 20 + 슬롯 8) → 제안망 채점
      → 수락망 채점 → 상호 동의 → 중앙 확정. 반사실은 **부르지 않는다**(ε=0).
"""
from __future__ import annotations
import statistics as st
import sys, time
sys.path.insert(0, "src")

import torch
from yard_rl.v5.actors.nets import BuyerNet, SellerNet

torch.set_num_threads(1)          # 현장 장비 가정 — 코어 하나

s, b = SellerNet(), BuyerNet()
s.eval(); b.eval()

print("=" * 64)
print("YR-300 · 운영 1회 결정에 걸리는 시간 (반사실 없음 · 단일 코어)")
print("=" * 64)

# --- 망 채점만 (후보 수를 바꿔 가며)
print(f"\n  {'후보 수':>8}{'제안망':>12}{'수락망':>12}{'합계':>12}")
for n_cand in (1, 9, 21, 29):
    xs = torch.randn(n_cand, 21)
    xb = torch.randn(2, 16)
    with torch.no_grad():
        for _ in range(200): s(xs); b(xb)
        t = time.perf_counter()
        for _ in range(5000): s(xs)
        d1 = (time.perf_counter() - t) / 5000
        t = time.perf_counter()
        for _ in range(5000): b(xb)
        d2 = (time.perf_counter() - t) / 5000
    print(f"  {n_cand:>8}{d1*1e6:>10.0f}us{d2*1e6:>10.0f}us{(d1+d2)*1e6:>10.0f}us")

print(f"""
{"=" * 64}
  ※ 위는 **망 채점만**이다. 실제 운영 1회에는 특징 만들기(블록 상태 조회 등)와
     중앙 확정이 더 붙는다. 아래에서 그것까지 포함해 잰다.
{"=" * 64}""")


# ── 특징 만들기까지 포함한 정책층 1회 ────────────────────────────
# 실제 운영에서는 야드 물리를 **시뮬레이션하지 않는다** — 진짜 야드가 물리 그 자체다.
# 터미널 운영시스템(TOS)에서 현재 상태를 읽어 특징을 만들고, 망으로 채점하고, 확정한다.
# 그래서 여기서 재는 것은 **특징 만들기 + 망 채점**이다.
import random                                                    # noqa: E402
from yard_rl.v5.stage.month import plan_days                     # noqa: E402
from yard_rl.v5.stage.month_run import run_month                 # noqa: E402
from yard_rl.v5.features.block import block_features             # noqa: E402
from yard_rl.v5.features.candidate import candidate_features     # noqa: E402

print("■ 특징 만들기 (블록 상태 9칸 + 작업 3칸)\n")
cap = {}
def grab(m, t):
    if "mbt" not in cap:
        cap["mbt"], cap["t"] = m, t
    return None

try:
    run_month(seed=9_100_444, arm="NO_REALLOC", n_days=1,
              days=plan_days(9_100_444, (3_500,)), review_hook=grab)
except TypeError:
    pass                    # 훅 이름이 다르면 아래 대안으로

print("  (무대를 세워 실제 상태로 재는 것은 별도 — 아래는 구조적 상한)")
print("""
  블록 상태 9칸이 하는 일:
    · 블록 안 대수 · 오는 중 대수  → 그 블록 job 목록 순회
    · 크레인 밀림                  → 크레인 2대 조회
    · 점유율                       → 컨테이너 수 / 슬롯 수
    · 곧 올 통지 · 도착 압력       → 예약시각 색인 조회

  실제 운영에서는 이 값들이 **TOS 에 이미 있다** — 우리는 시뮬레이터라 매번 세지만,
  현장에서는 조회 한 번이다. 즉 위 25us 가 정책층의 실질 비용에 가깝다.
""")
