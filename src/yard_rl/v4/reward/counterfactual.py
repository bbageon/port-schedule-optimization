"""반사실 교사 — 행위자가 **자기 행동을 바꾼** 세계를 H=3시간 굴린다.

설계 정본: `.claude/docs/architecture/04b-학습-잣대.md` §3

■ 행위자마다 자기 이진 반사실을 갖는다
      Seller :  Φ_H(KEEP)   vs  Φ_H(SELL…)
      Buyer  :  Φ_H(REJECT) vs  Φ_H(BUY)
  같은 snapshot·같은 난수에서 갈라 굴린다. 선택지가 이산이라 **회귀만으로** 끝난다
  — PPO 도 critic 도 안 들인다(v1 이 무너진 자리).

■ ★두 라벨을 더하면 `D_i` 가 되지 않는다. 그게 정상이다
  각자 "내가 행동을 바꿨다면" 을 재므로 **겹치는 몫이 양쪽에 다 들어간다.**
  실제 Φ 에는 **일어난 일(factual)만 한 번** 기록하고 actor 비용은 합산하지 않는다.

■ 독립인데 실제 세계와 어긋나지 않으려면 (04b §3)
  ① **상대는 가정하지 않고 정책을 그대로 굴린다** — Seller 의 SELL 가지에서
     "Buyer 가 받아준다" 고 가정하면 실제로 거절당한 경우 그 세계는 도달 불가능하다.
     offer 를 내고 **Buyer 정책이 실제로 응답하게** 굴린다.
  ② **factual 을 공유해 세계가 넷이 아니라 셋**이다.
  ③ 상대가 학습하면 내 라벨이 낡는다 → **회차마다 다시 만든다**(유효기간).

■ 기계 검사 — 동일성 불변식
  **반사실 rollout 의 factual 가지 == 실제 궤적** (비트 단위). 같은 시드·같은
  분기점이면 정확히 같아야 한다. 통계가 아니라 동일성 검사라 한 번만 어긋나도 잡힌다.

■ 배포에서 부르면 안 된다
  1건 결정에 3시간 시뮬이다. 판정 실행의 `rollout_calls_during_eval` 은 0 이어야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import CF_HORIZON_S

#: 이 모듈이 굴린 rollout 수 — 판정 하드가드가 읽는다(교사 누출 금지).
_ROLLOUT_CALLS = 0


def rollout_calls() -> int:
    """지금까지 굴린 반사실 rollout 수. 판정 실행에서는 **0 이어야** 한다."""
    return _ROLLOUT_CALLS


def reset_rollout_calls() -> None:
    _ROLLOUT_CALLS_reset()


def _ROLLOUT_CALLS_reset() -> None:
    global _ROLLOUT_CALLS
    _ROLLOUT_CALLS = 0


def _count_rollout() -> None:
    global _ROLLOUT_CALLS
    _ROLLOUT_CALLS += 1


def add_rollout_calls(n: int) -> None:
    """★다른 프로세스에서 굴린 몫을 여기 더한다 ([[YR-219]]).

    반사실을 프로세스 풀로 나누면 작업자의 계수기는 그 프로세스 안에만 남는다.
    판정 하드가드(`rollout_calls_during_eval == 0`)가 부모에서 검사되므로,
    작업자가 굴린 세계 수를 **부모가 되받아 더해야** 가드가 살아 있다.
    """
    global _ROLLOUT_CALLS
    _ROLLOUT_CALLS += int(n)


@dataclass
class ActorLabel:
    """행위자 하나의 이진 반사실 라벨.

    `advantage = phi_alt − phi_factual` — **양수면 실제 선택이 옳았다**(대안이 더 비쌌다).
    학습 목표는 각 세계의 Φ 자체이고, 선택은 argmin 이다.
    """

    doc_key: str
    actor: str                  # "SELLER" | "BUYER"
    factual_action: str
    alt_action: str
    phi_factual: float
    phi_alt: float
    horizon_s: float = CF_HORIZON_S

    @property
    def advantage(self) -> float:
        return self.phi_alt - self.phi_factual


@dataclass
class TeacherResult:
    """한 거래(또는 KEEP) 결정에서 나온 라벨들과 검사 정보."""

    labels: list[ActorLabel] = field(default_factory=list)
    worlds: int = 0                       # 굴린 세계 수 (factual 포함)
    factual_matches: bool = True          # 동일성 불변식 통과 여부


class CounterfactualTeacher:
    """교사 — 망이 아니라 **시뮬레이터를 굴리는 절차**다. 학습할 파라미터가 없다.

    `rollout_fn(snapshot, overrides, horizon_s) -> float`
        주어진 분기 시점에서 `overrides` 를 적용해 `horizon_s` 만큼 굴리고
        그 구간의 Φ(원화)를 돌려주는 함수. 무대가 주입한다.
    `phi_of_factual(snapshot, horizon_s) -> float`
        아무것도 안 바꾸고 굴린 값 — 세 세계가 **공유**한다.
    """

    def __init__(self, rollout_fn, *, horizon_s: float = CF_HORIZON_S,
                 strict_identity: bool = True):
        self.rollout_fn = rollout_fn
        self.horizon_s = float(horizon_s)
        self.strict_identity = strict_identity

    # ------------------------------------------------------------------ 내부
    def _roll(self, snapshot, overrides) -> float:
        _count_rollout()
        return float(self.rollout_fn(snapshot, overrides, self.horizon_s))

    # ------------------------------------------------------------------ 공개
    def label_decision(self, snapshot, *, doc_key: str,
                       seller_action: str, buyer_action: str | None,
                       factual_phi: float | None = None) -> TeacherResult:
        """한 결정에 대해 행위자별 라벨을 만든다.

        | 실제로 일어난 일 | Seller 라벨 | Buyer 라벨 |
        |---|---|---|
        | `KEEP`           | factual vs `SELL`→Buyer 응답 | **없음**(물어보지도 않았다) |
        | `SELL ∧ BUY`     | factual vs `KEEP`            | factual vs `REJECT` |
        | `SELL ∧ REJECT`  | factual vs `KEEP` → **≈0**   | factual vs `BUY` |

        `SELL ∧ REJECT` 에서 Seller 라벨이 0 에 가까운 것은 **맞다** — 제안했다
        거절당하면 세상이 안 바뀐다. 그것도 배울 정보다.
        """
        out = TeacherResult()

        # ① factual — 셋이 공유한다
        phi_f = self._roll(snapshot, overrides={})
        out.worlds += 1
        if factual_phi is not None and self.strict_identity:
            out.factual_matches = abs(phi_f - factual_phi) <= 1e-6
            if not out.factual_matches:
                raise AssertionError(
                    f"{doc_key}: 반사실 factual 가지가 실제 궤적과 다르다 "
                    f"({phi_f!r} vs {factual_phi!r}) — rollout 구현이 틀렸다")

        # ② Seller 의 대안 — 상대는 **정책을 그대로 굴린다**(가정하지 않는다)
        seller_alt = "KEEP" if seller_action != "KEEP" else "SELL"
        phi_s = self._roll(snapshot, overrides={
            "actor": "SELLER", "doc_key": doc_key, "action": seller_alt,
            "buyer_policy": "AS_IS",          # ← Buyer 가 실제로 응답하게 굴린다
        })
        out.worlds += 1
        out.labels.append(ActorLabel(
            doc_key=doc_key, actor="SELLER",
            factual_action=seller_action, alt_action=seller_alt,
            phi_factual=phi_f, phi_alt=phi_s, horizon_s=self.horizon_s))

        # ③ Buyer 의 대안 — SELL 이 실제로 일어난 경우에만 정의된다
        if buyer_action is not None:
            buyer_alt = "REJECT" if buyer_action == "BUY" else "BUY"
            phi_b = self._roll(snapshot, overrides={
                "actor": "BUYER", "doc_key": doc_key, "action": buyer_alt,
            })
            out.worlds += 1
            out.labels.append(ActorLabel(
                doc_key=doc_key, actor="BUYER",
                factual_action=buyer_action, alt_action=buyer_alt,
                phi_factual=phi_f, phi_alt=phi_b, horizon_s=self.horizon_s))

        return out
