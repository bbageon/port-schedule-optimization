"""자격시험 A~F — **기여도를 믿어도 되는가** ([[YR-223]] 3단계 · 사전등록).

■ 왜 여섯 개나 되나
  모델은 언제나 뭔가를 학습한다. 문제는 그게 **우리가 재려던 그것이냐**다.

      A 예측     맞히기는 하는가            (못 맞히면 나머지가 무의미)
      B 행동제거 **행동**을 보고 맞히는가    ← 시각대만 외웠는지 잡는다
      C 반복성   초기값을 바꿔도 같은 답인가
      D 순서섞기 **시간 구조**를 쓰는가
      E 보존     Σ기여도 = Y 인가
      ★F 개입    실제로 개입하면 그 부호가 맞는가   ← **가장 강한 검사**

  A~E 는 "모델이 뭔가를 학습했다" 를 본다. F 만이 **"그게 인과냐"** 를 본다.

■ 필수와 선택
      필수  A · B · D · F        하나라도 미달이면 **사용 불가**
      선택  C · E                미달이면 "순위 용도만"(표본 추출기)로 제한

■ 사전등록 문턱 (착수 시 동결 — 결과를 보고 고치지 않는다)
      A  검증 상관 ≥ 0.5
      B  행동 제거 시 검증 손실이 **10% 이상 나빠진다**
      C  초기값 3개의 상위 20 epoch 순위 상관 ≥ 0.5
      D  순서 섞을 때 검증 손실이 **10% 이상 나빠진다**
      E  |Σ기여도 − Y| / |Y| ≤ 0.10
      F  상위 20 epoch 개입의 **부호 일치 ≥ 70%**
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: 사전등록 문턱. ⚠️ 결과를 본 뒤 바꾸면 사전등록이 아니다.
TH_A_CORR = 0.5
TH_B_WORSE = 0.10
TH_C_RANK = 0.5
TH_D_WORSE = 0.10
TH_E_REL = 0.10
TH_F_SIGN = 0.70
REQUIRED = ("A", "B", "D", "F")


@dataclass
class Gate:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""

    def as_dict(self) -> dict:
        return {"gate": self.name, "pass": self.passed, "value": self.value,
                "threshold": self.threshold, "note": self.note}


@dataclass
class GateReport:
    gates: list = field(default_factory=list)

    def add(self, g: Gate) -> None:
        self.gates.append(g)

    @property
    def required_ok(self) -> bool:
        by = {g.name: g for g in self.gates}
        return all(by.get(n) is not None and by[n].passed for n in REQUIRED)

    @property
    def verdict(self) -> str:
        """세 갈래 — 쓸 수 있다 / 순위만 / 못 쓴다."""
        if not self.required_ok:
            return "사용 불가"
        by = {g.name: g for g in self.gates}
        soft = [n for n in ("C", "E") if n in by and not by[n].passed]
        return "순위 용도만 (표본 추출기)" if soft else "사용 가능"

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "required_ok": self.required_ok,
                "gates": [g.as_dict() for g in self.gates]}


# ------------------------------------------------------------------ A 예측
def gate_a(rep) -> Gate:
    return Gate("A", rep.val_corr >= TH_A_CORR, rep.val_corr, TH_A_CORR,
                f"검증 창 {rep.n_val}개 · 평균오차 {rep.val_mae_krw:,.0f}원")


# ------------------------------------------------------------------ B/D 훼손
def _worse(base_loss: float, hurt_loss: float) -> float:
    """훼손했을 때 손실이 **몇 배 나빠졌나** − 1. 양수면 나빠진 것."""
    return (hurt_loss - base_loss) / max(1e-9, abs(base_loss))


def gate_b(base_rep, ablated_rep) -> Gate:
    v = _worse(base_rep.val_loss, ablated_rep.val_loss)
    return Gate("B", v >= TH_B_WORSE, v, TH_B_WORSE,
                f"검증손실 {base_rep.val_loss:.4f} → {ablated_rep.val_loss:.4f} "
                f"(행동 칸 제거)")


def gate_d(base_rep, shuffled_rep) -> Gate:
    v = _worse(base_rep.val_loss, shuffled_rep.val_loss)
    return Gate("D", v >= TH_D_WORSE, v, TH_D_WORSE,
                f"검증손실 {base_rep.val_loss:.4f} → {shuffled_rep.val_loss:.4f} "
                f"(순서 섞음)")


# ------------------------------------------------------------------ C 반복성
def gate_c(rank_corrs) -> Gate:
    vals = [float(v) for v in rank_corrs]
    v = sum(vals) / max(1, len(vals))
    return Gate("C", v >= TH_C_RANK, v, TH_C_RANK,
                f"초기값 짝 {len(vals)}쌍의 순위상관 평균")


# ------------------------------------------------------------------ E 보존
#: Y 가 이보다 작은 창은 보존 검사에서 뺀다 — 분모가 잡음이면 상대오차가 무의미하다.
E_MIN_Y_KRW = 1_000.0


def gate_e(cons_rows) -> Gate:
    """★|Y| 가 너무 작은 창은 **뺀다**. 0 원짜리 창의 상대오차는 잡음÷잡음이다."""
    vals = sorted(float(r["rel_err"]) for r in cons_rows
                  if abs(float(r["y_krw"])) >= E_MIN_Y_KRW)
    if not vals:
        return Gate("E", False, 1.0, TH_E_REL,
                    f"Y 가 {E_MIN_Y_KRW:,.0f}원 넘는 창이 없다 — 잴 것이 없다")
    v = vals[len(vals) // 2]
    return Gate("E", v <= TH_E_REL, v, TH_E_REL,
                f"창 {len(vals)}개(|Y|≥{E_MIN_Y_KRW:,.0f}원)의 상대오차 중앙값")


# ------------------------------------------------------------------ ★F 개입
def gate_f(pairs) -> Gate:
    """`pairs` = [(기여도_원화, 실제개입효과_원화), ...]

    부호가 같은 비율을 본다. 0 에 가까운 값은 부호가 의미 없으므로 **뺀다** —
    안 빼면 잡음이 반반씩 맞아 50% 근처로 수렴해 검사가 무뎌진다.
    """
    use = [(c, d) for c, d in pairs if abs(c) > 1.0 and abs(d) > 1.0]
    if not use:
        return Gate("F", False, 0.0, TH_F_SIGN, "부호를 볼 표본이 없다")
    ok = sum(1 for c, d in use if (c > 0) == (d > 0))
    v = ok / len(use)
    from .train import pearson
    r = pearson([c for c, _ in use], [d for _, d in use])
    return Gate("F", v >= TH_F_SIGN, v, TH_F_SIGN,
                f"표본 {len(use)}건 · 크기상관 {r:+.3f}")
