"""YR-189 — 판매 축을 **Q 채점**으로 전환 (C안). 자(尺)가 하나뿐인 구조.

■ 무엇이 바뀌나 — 숫자의 뜻
    지금 (PPO):  logits = actor(행);  a = 추첨/argmax(logits)   "얼마나 선호하나"
    바꾼 뒤 (Q): q      = Q(행);      a = argmin(q)             "비용이 얼마나 드나"

망 구조(Linear→ReLU→Linear→ReLU→점수 1개)는 그대로다. 바뀌는 것은 셋이다.

■ ① 행이 **좌표까지** 포함한다 (핵심)
구판 행 = 블록요약 7 + KEEP 플래그 1 + 후보 6. **목적지가 없다.** 그래서 배정기는
따로 계산식(`cost_given_queue`)으로 좌표를 골라야 했고, 거기서 [[YR-175]](부호)와
[[YR-177]](기준 불일치)이 났다 — 공간은 자기 비용만, 시간은 견적망 순비용으로 재는
**서로 다른 자** 두 개.

여기서는 행 = 블록요약 7 + 후보 6 + **좌표 8** = 21차원이고, 정책과 배정기가
**같은 망**으로 같은 행을 채점한다. "다른 자로 잰다"가 구조적으로 성립 불가다.

■ ② KEEP = **0 고정 기준점** (학습 대상이 아니다)
"아무것도 안 한다"의 비용은 정의상 0이다. 학습 가능한 KEEP 행이 없으므로
[[YR-185]] 가 진단한 문제 — 보상 0 표본이 91% 인데 이점 정규화가 거기에 부호를
주던 것 — 이 사라진다. Q < 0 인 좌표가 하나도 없으면 팔지 않는다.

■ ③ 목표 = **차분 신용의 부호 반전**
    Q(고른 행) ← −D_i      (D_i = `yr174_txn_reward.realized_credit`)
D_i 는 이미 반사실 차분(Wolpert & Tumer 2002 difference rewards)의 관측 가능한
분해다. 부호만 뒤집으면 "이 좌표로 팔면 터미널 총비용이 얼마나 늘어나나"가 된다.
**부트스트랩이 없다** — 회귀 라벨이 직접 관측되므로 off-policy 재생이 편향 없이
성립한다(구 PPO 는 on-policy 라 40 에피소드를 쓰고 버렸다).

`TransferCritic` 이 사라진다 — 붕괴할 critic 이 없다.

■ 문헌 (우리 발명이 아니다 — 인용 필수)
  · difference rewards: Wolpert & Tumer (2002), Tumer & Agogino (2007)
  · 반사실 기준선의 신경망화: COMA, Foerster et al. (AAAI 2018)
  · 조합 행동공간의 후보 채점 + argmin 계열
문헌이 지적하는 위험(최고 후보 하나만 쓰면 채점 잡음에 민감)은 학습 중 ε 탐색으로
완화하고, 평가는 순수 argmin(결정론)으로 한다.

■ 정보 경계 (구 계약 승계)
입력은 **공개 정보만** — 통지 시각(`reserved_s`)은 읽고 실현 시각(`gate_in_s`)은
읽지 않는다. 실현값은 학습 **목표**(D_i)에만 들어간다.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import BLOCK_DIM, KEEP_Q, block_features, candidate_features

COORD_DIM = 8
Q_ROW_DIM = BLOCK_DIM + 6 + COORD_DIM        # 7 + 6 + 8 = 21
HID = 64
# KEEP_Q 는 `v2/features.py` 에 있다 (배정기는 자기 사본을 따로 갖는다).


# ------------------------------------------------------------------ 좌표 특징
def coord_features(*, is_space: bool, dst_load: float = 0.0, dst_free: float = 0.0,
                   route_delta_s: float = 0.0, defer_s: float = 0.0,
                   src_load_after: float = 0.0, src_load: float = 0.0) -> list[float]:
    """좌표 8차원. 공간 칸과 시간 칸이 **겹치지 않는 자리**를 쓴다.

    7번(`src_load`)만 두 축이 공유한다 — 배정기의 **가상 상태**가 망에 들어가는
    유일한 통로이므로 반드시 필요하다(매칭이 진행되며 q 가 갱신된다).
    """
    return [1.0 if is_space else 0.0, 0.0 if is_space else 1.0,
            dst_load / 10.0, dst_free / 1000.0, route_delta_s / 600.0,
            defer_s / 1800.0, src_load_after / 10.0, src_load / 10.0]


class EpochCache:
    """한 epoch(60초 격자) 동안 블록·후보 특징을 **한 번만** 만든다.

    블록 요약은 장부 전체를 훑으므로(통지 30분 내 건수) 후보마다 다시 만들면
    에피소드 비용이 좌표 수만큼 곱해진다. 좌표별로 바뀌는 부분만 뒤에 붙인다.
    """

    def __init__(self, t: float):
        self.t = t
        self._blk: dict[str, list[float]] = {}
        self._cand: dict[tuple[str, str], list[float]] = {}
        self._free: dict[str, int] = {}

    def block(self, mbt, src: str, t: float, n_cands: int) -> list[float]:
        if src not in self._blk:
            self._blk[src] = block_features(mbt, src, t, n_cands)
        return self._blk[src]

    def cand(self, mbt, src: str, jid: str, t: float) -> list[float]:
        k = (src, jid)
        if k not in self._cand:
            self._cand[k] = candidate_features(mbt, src, jid, t)
        return self._cand[k]

    def free(self, mbt, bid: str) -> int:
        """★`free_slots` 는 그 블록의 **전 작업을 훑는다**. 한 epoch 안에서는
        확정이 매칭 뒤에만 일어나므로 값이 **상수**다 — 캐시해도 동작이 바뀌지
        않는다(가상 배정분은 `vcap` 이 따로 센다). 캐시 없이는 좌표 채점이
        블록×후보×매칭회차 만큼 이 함수를 부른다.
        """
        if bid not in self._free:
            self._free[bid] = int(mbt.free_slots(bid))
        return self._free[bid]


# ------------------------------------------------------------------ 망
class SellQNet(nn.Module):
    """행 하나 → 예상 **증분비용**(시간 단위). 음수 = 팔면 이득."""

    def __init__(self, in_dim: int = Q_ROW_DIM, hid: int = HID):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows).squeeze(-1)


# ------------------------------------------------------------------ 좌표 채점기 (하나의 자)
class QCoordScorer:
    """(블록, 작업, 좌표) 행을 만들어 **같은 Q 망**으로 점수를 낸다.

    정책(제안 고르기)과 배정기(좌표 고르기)가 **이 객체 하나를 공유**한다. 둘의
    차이는 넘겨받는 상태뿐이다 — 정책은 실제 대기열, 배정기는 매칭 진행 중의
    **가상 대기열**(q·vcap). 자는 같고 상태만 다르다.
    """

    def __init__(self, net: SellQNet, layout, *, defer_delta_s: float,
                 time_slots: bool = False, explore_sigma: float = 0.0,
                 seed: int = 0):
        self.net = net
        self.layout = layout
        self.defer_delta_s = defer_delta_s
        self.time_slots = bool(time_slots)
        # ★좌표 탐색 — argmin 만 쓰면 **고른 좌표의 라벨만** 모여 나머지를 영원히
        # 못 배운다(문헌이 지적하는 off-policy 커버리지 문제). 학습 중에만 점수에
        # 잡음을 얹는다. 눈금을 바꾸지 않는 가법 잡음이라 전역 매칭 비교가 유지된다.
        # 평가는 0 — 사전등록 모드가 순수 argmin 이다.
        self.explore_sigma = float(explore_sigma)
        self.gen = torch.Generator().manual_seed(seed)
        self._cache: EpochCache | None = None
        self._routes: dict = {}

    def _route(self, src: str, dst: str) -> float:
        """주행 차이는 배치의 순수 함수 — 에피소드 내내 불변이라 한 번만 잰다."""
        k = (src, dst)
        r = self._routes.get(k)
        if r is None:
            r = self._routes[k] = float(self.layout.pre_gate_route_delta_s(src, dst))
        return r

    def epoch(self, t: float) -> EpochCache:
        if self._cache is None or self._cache.t != t:
            self._cache = EpochCache(t)
        return self._cache

    # ---- 좌표 열거 (배정기의 구 `_coord_costs` 와 **같은 집합**을 만든다)
    def coords(self, mbt, src: str, jid: str, flow: str, t: float,
               q: dict, vcap: dict, capacity_margin: float) -> list[tuple[str, list[float]]]:
        from .block_congestion import SVC_REF_S
        from ..integrated.time_sell import notified_gate_in
        j = mbt.blocks[src].jobs[jid]
        gi = notified_gate_in(j)                 # 공개 통지 시각만 (실현값 금지)
        src_load = float(q[src])
        out: list[tuple[str, list[float]]] = []
        if flow == "GATE_IN":
            ec = self.epoch(t)
            for dst in mbt.blocks:
                if dst == src:
                    continue
                free = ec.free(mbt, dst) - vcap.get(dst, 0)
                if free <= capacity_margin:
                    continue
                out.append((dst, coord_features(
                    is_space=True, dst_load=float(q[dst]), dst_free=float(free),
                    route_delta_s=self._route(src, dst),
                    src_load=src_load)))
        n_cranes = max(1, len(mbt.blocks[src].profile.cranes))

        def _time_row(d: float) -> list[float]:
            return coord_features(
                is_space=False, defer_s=d,
                src_load_after=max(0.0, src_load - d * n_cranes / SVC_REF_S),
                src_load=src_load)

        if not self.time_slots:
            out.append(("TIME", _time_row(self.defer_delta_s)))
            return out
        from .slot_plan import N_SLOTS, SLOT_S
        for k in range(int(gi // SLOT_S) + 1, N_SLOTS):
            d = k * SLOT_S - gi
            if d > 0.0:
                out.append((f"TIME@{k}", _time_row(d)))
        return out

    # ---- 채점: 좌표별 Q (낮을수록 싸다). KEEP 은 0 이므로 여기 들어오지 않는다.
    def score(self, mbt, src: str, jid: str, flow: str, t: float,
              q: dict, vcap: dict, capacity_margin: float, n_cands: int
              ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        cs = self.coords(mbt, src, jid, flow, t, q, vcap, capacity_margin)
        if not cs:
            return [], torch.empty(0), torch.empty(0, Q_ROW_DIM)
        ec = self.epoch(t)
        head = ec.block(mbt, src, t, n_cands) + ec.cand(mbt, src, jid, t)
        rows = torch.tensor([head + cf for _, cf in cs], dtype=torch.float32)
        with torch.no_grad():
            qs = self.net(rows)
            if self.explore_sigma > 0.0:
                qs = qs + torch.randn(qs.shape, generator=self.gen) * self.explore_sigma
        return [c for c, _ in cs], qs, rows


# ------------------------------------------------------------------ 정책 어댑터
class QSellPolicy:
    """블록의 제안 선택 — 후보를 **각자 가장 싼 좌표**로 채점하고 argmin.

    구 `PpoSellPolicy` 와 다른 점 셋:
      · 추첨(Categorical)이 없다 — 학습 중 탐색은 ε 로만 한다(평가는 순수 argmin).
      · critic 이 없다 — 기준선이 필요 없다(라벨이 직접 관측된다).
      · KEEP 은 행이 아니라 **상수 0** 이다 — 모든 좌표가 0 이상이면 안 판다.

    `trail` 에는 **채점 근거**만 남긴다. 학습 표본은 배정기가 실제로 확정한
    (작업, 좌표) 행이므로 `orchestrator.q_rows` 가 원료다.
    """

    def __init__(self, scorer: QCoordScorer, *, explore: float = 0.0,
                 seed: int = 0, defer_decision: bool = False):
        self.scorer = scorer
        self.explore = float(explore)
        self.gen = torch.Generator().manual_seed(seed)
        self.mode = "live"
        # ★YR-195: True 면 블록은 **판단하지 않는다** — "내 후보 중 뭐가 제일
        # 부담인가"만 답하고, 팔지 말지는 배정기가 정한다(문턱을 한 곳으로 모은다).
        # 기본 False 는 YR-189 동작 그대로 — 그 판정의 재현성을 지킨다.
        self.defer_decision = bool(defer_decision)
        self.trail: list[dict] = []
        self._handoff: dict = {}          # src → (좌표들, Q, 행) — 배정기가 재사용
        self._q: dict = {}
        self._vcap: dict = {}
        self._margin: float = 0.0

    def bind_epoch(self, q: dict, vcap: dict, capacity_margin: float) -> None:
        """배정기가 epoch 시작에 실제 상태를 넘긴다 — 정책은 가상 상태를 보지 않는다."""
        self._q, self._vcap, self._margin = q, vcap, capacity_margin
        self._handoff = {}

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self.gen).item())

    def decide(self, mbt, src: str, cands: list, t: float) -> str | None:
        if not cands:
            return None
        # ★YR-195: 문턱을 넘겨줄 때는 **무조건 최악 하나를 지목**한다(inf 에서 시작).
        # 구 동작은 0 에서 시작해 "0보다 싼 게 있어야만" 제안했다 — 그 판단이
        # 배정기와 중복이었고(실측 판단 거부 0.9%), 문턱이 두 곳에 흩어졌다.
        best_jid, best_q = None, (float("inf") if self.defer_decision else KEEP_Q)
        seen, scored = [], {}
        for jid, _eta, flow in cands:
            cs, qs, rows = self.scorer.score(mbt, src, jid, flow, t, self._q,
                                             self._vcap, self._margin, len(cands))
            if qs.numel() == 0:
                continue
            v = float(qs.min().item())
            seen.append((jid, v))
            scored[jid] = (cs, qs, rows)
            if v < best_q:
                best_jid, best_q = jid, v
        if self.explore > 0.0 and seen and self._rand() < self.explore:
            # ε 탐색 — 손해로 보이는 제안도 내본다. 라벨이 없으면 그 좌표가 왜 나쁜지
            # 영원히 못 배운다(문헌: 채점 잡음·off-policy 커버리지).
            i = int(torch.randint(len(seen), (1,), generator=self.gen).item())
            best_jid, best_q = seen[i]
        # `action`·`value` 는 구 공동기록(`build_joint_transitions`)이 요구하는 키다.
        # 뜻은 바뀌었다 — value 는 기준선(critic)이 아니라 **예상 비용**이다.
        # ★채점 결과를 배정기에 넘긴다 — 같은 망으로 같은 행을 다시 재는 낭비를
        # 없앤다(실측: 배정기 재채점의 99.1% 가 결론을 바꾸지 못했다).
        self._handoff[src] = scored.get(best_jid) if best_jid is not None else None
        self.trail.append({"t": t, "src": src, "n_cands": len(cands),
                           "picked": best_jid, "q": round(best_q, 6),
                           "action": 0 if best_jid is None else 1,
                           "value": round(best_q, 6), "scanned": len(seen)})
        return best_jid

    def handoff(self, src: str):
        """배정기가 가져가는 (좌표들, Q, 행). 없으면 None → 배정기가 직접 잰다."""
        return self._handoff.get(src)
