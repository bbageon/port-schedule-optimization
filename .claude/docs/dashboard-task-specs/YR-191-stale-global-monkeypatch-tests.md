# YR-191 — 낡은 전역 monkeypatch 시험 8건이 상시 빨간불 (거짓 실패)

- **Epic**: Infra / **Priority**: 🟡 / **등록일**: 2026-08-18 / **상태**: backlog
- **3대 게이트 보정 대상**: `dashboard-code-evidence`
- **1줄**: [[YR-160]] 이 전역을 없애고 config 주입으로 바꿨는데, 시험 8건이
  **없어진 전역을 그대로 흔들고 있어** 아무 효과 없이 실패한다.

## 실측 (2026-08-18 · YR-187 A단계 회귀 중 발견)

```
8 failed, 571 passed, 8 skipped   (WSL · tests/integrated/ 전량 · 349.98s)
```

**HEAD 에서도 동일한 8건이 실패**한다 — 그날 변경과 무관한 **기존 실패**임을
같은 8건 재현으로 확인했다(터미널 스키마 배선을 되돌린 트리에서 대조).

| 파일 | 실패 |
|---|---|
| `test_yr115_transfer_joint_confirm.py` | 3 |
| `test_yr141_bound_prepo.py` | 1 |
| `test_yr143_safety_only.py` | 2 |
| `test_yr147_defer.py` | 2 |

## 원인

```python
cand_mod.WAIT_MODE = "DEFER_ALL"     # ← 시험이 하는 일
assert g.defer_until is not None     # ← 실패. 아무 일도 안 일어난다
```

`candidates` 모듈의 전역(`WAIT_MODE`·`SAFETY_ONLY`·`BOUND_REPO`)은 **커밋
`8078ef4` (YR-160 "전역 완전 제거·config 명시 주입")에서 사라졌다.** 지금은
`ExecPolicyConfig` 를 인자로 넣어야 동작이 바뀐다. 시험은 옛 방식 그대로다.

## ★함께 발견 — 시험이 아닌 **본코드**도 같은 전역을 쓴다

```
src/yard_rl/experiments/yr143_no_repo.py:114     cand_mod.WAIT_MODE = "DEFER_ALL"
src/yard_rl/experiments/yr146_deploy_guard.py:189 cand_mod.WAIT_MODE = "DEFER_ALL"
```

이 둘은 **아무 효과 없는 설정을 하고 실험을 돌린다**. 과거 산출물이 의도한
조건에서 나온 것인지 재검증 대상이다. 시험 수리보다 이쪽이 중요하다.

## 할 일

1. 두 실험 모듈이 실제로 어떤 조건에서 돌았는지 판정(전역 제거 시점 전/후).
2. 시험 8건을 `ExecPolicyConfig` 주입 방식으로 옮긴다.
3. 게이트에 **"integrated 전량 녹색"** 을 조건으로 넣는다 — 지금은 8건이
   상시 빨간불이라 새 회귀가 묻힌다(이번에 실제로 그럴 뻔했다).

## 하지 않는 것

- assert 를 느슨하게 고쳐 초록으로 만드는 것.
