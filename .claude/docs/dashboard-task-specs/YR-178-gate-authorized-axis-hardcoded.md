# YR-178 — 게이트의 "지금 허가된 축" 이 하드코딩이라 틀린 축을 가리킨다

- **Epic**: Infra / **Priority**: 🟡 / **등록일**: 2026-08-17 / **상태**: backlog
- **3대 게이트 보정 대상**: 없음 — 게이트 하네스 자체의 결함
- **1줄**: 세 축을 다 계산해놓고, **무엇을 해도 되는지 알려주는 칸만** 두 갈래
  고정 문자열이라 현재 상태와 어긋난다.

## 실측 (2026-08-17 재발행 `805f9f8`)

| 축 | 판정 |
|---|---|
| reliability | PASS |
| scenario_validity | **PASS** |
| performance | INCONCLUSIVE ← 유일한 미해소 |

그런데 같은 JSON 의 `currently_authorized` 는 이렇게 나온다:

> `reliability PASS → YR-150: scenario_validity 단일축`

**scenario_validity 는 이미 PASS 인데 그걸 보정하라고 한다.** 맞는 값은
`performance 단일축` 이고, 실제로 바로 아래 `conditional_sequence` 가 그렇게
적고 있다 — 같은 파일 안에서 두 칸이 서로 다른 말을 한다.

## 원인 (`yr151_gate_report.py:122-127`)

```python
out["currently_authorized"] = (
    "reliability PASS → YR-150: scenario_validity 단일축"
    if reliability.status.value == "PASS" else "YR-151 0A: reliability 단일축(미해소)")
out["conditional_sequence"] = [
    "scenario_validity PASS 후 YR-151 0B: performance 단일축(...)"]
out["forbidden_next_scope"] = "위 미확정과 무관한 새 상태·보상·행동·가설 추가"
```

세 칸 모두 **하드코딩**이고, `currently_authorized` 는 `reliability` 하나만 보고
갈린다. `scenario_validity` 가 FAIL 이던 시절(2026-08-06~09)에 쓰인 문자열이
그대로 남아, 그 축이 PASS 로 바뀐 뒤에도 계속 같은 말을 한다.

## 왜 고쳐야 하나

이 칸은 **"다음에 무슨 작업을 해도 되는가"를 알려주는 자리**다. 사람이 여기를
읽고 다음 작업을 고르는데, 이미 닫힌 축을 가리키면 두 가지가 생긴다.

1. 이미 통과한 축을 다시 보정하러 간다 (헛일).
2. 반대로 "게이트 말이 안 맞네" 하고 이 칸 전체를 무시하게 된다 —
   게이트가 진짜로 막아야 할 때도 무시된다. **이쪽이 더 위험하다.**

지금까지 사고가 안 난 이유는 사람이 세 축의 원 판정을 직접 읽었기 때문이지
이 칸이 맞아서가 아니다.

## 교정

세 칸을 판정에서 **유도**한다 (문자열 상수 제거).

- 미해소 축이 있으면 → 그중 **의존 순서상 가장 앞선 축**이 허가 축.
  순서는 `reliability → scenario_validity → performance` (신뢰할 수 없는
  실행에서 현실성을 재고, 현실성 없는 무대에서 성능을 재는 것은 무의미).
- 전 축 PASS → 허가 = `확증·잠금평가` (AGENTS.md 3대 게이트 규칙과 일치).
- `conditional_sequence` 는 남은 축을 그 순서대로 나열.

## 검증

- 8가지 판정 조합(PASS/미PASS × 3축)에 대해 허가 축이 규칙과 맞는지 단위시험.
- 과거 게이트 JSON 재생성 시 **판정값 자체는 한 글자도 안 바뀔 것**
  (이 작업은 안내 문구만 고치는 것이지 판정 로직이 아니다).

## 하지 않는 것

- 판정 임계·상태값 계산에 손대는 것.
- 문자열만 오늘 상태에 맞게 갈아끼우는 것 (같은 결함이 다음 상태 변화에 재발).
