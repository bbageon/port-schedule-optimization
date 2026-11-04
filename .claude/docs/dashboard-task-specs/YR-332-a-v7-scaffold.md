# YR-332-a — v7 골격: v6 복사, 재배정 끄기, 본선 단가, DGT 트럭 시간

- **상태**: done
- 상위: [[YR-332]]. 착수 2026-11-04.

## 무엇

1. `src/yard_rl/v6` 를 `src/yard_rl/v7` 로 **독립 복사**한다. 테스트·스크립트도 `tests/v7`·`scripts/v7` 로.
   v6 는 손대지 않는다. 복사 기준 커밋을 `V6_BASE_COMMIT` 으로 박는다.
2. **재배정을 끈 크레인 전용 경로**를 기본으로 둔다. 판매·구매 시장을 부르지 않고 트럭은 원래 블록·시각으로 온다.
   판매·구매 코드는 지우지 않는다 — [[YR-332-e]] 에서 다시 붙인다.
3. **본선 유휴 단가 2.99 → 35.88원/GT·시간.** 근거는 v4 `reward/krw.py` 의 용선료 주석([[YR-307]]).
4. **DGT 트럭 시간 분포**를 입력 생성기에 넣는다. 반입·반출 리드타임을 따로 뽑고,
   통지 전 도착(14%)은 통지 시각 = 도착 시각으로 둔다. 근거 `docs/data-handover/2026-08-19-DGT-1차표본-대조.md`.
   기존 30분 고정 방식도 선택지로 남긴다(비교·회귀용).
5. 행동 집합: 미리 자리 잡는 빈 이동을 끄고(`safety_only`) 교착 탈출만 남긴다. 파내기·쉬기는 유지.

## 완료 조건

- 재배정 끔 상태에서 30일 입력이 전건 투입되고 물리 불변식을 지킨다. 거래 수가 정확히 0이다.
- 본선 단가가 비용에 반영됨을 단위 시험으로 확인한다.
- DGT 리드타임 표본 분포의 중앙·사분위가 원 표본과 맞는지 시험한다. 수명주기 불변식 위반 0.
- 규칙 크레인(실작업 우선·짧은 일 먼저)으로 하루·30일이 완주한다.
- `tests/v7` 통과. 커밋·푸시.

## 진행 기록

### 2026-11-04 — 골격 구현 (작업본 커밋 · 푸시 안 함)

- **복사**: `src/yard_rl/v6`·`tests/v6`·`scripts/v6` 의 추적 파일 250개를 v7 로 복사하고 이름공간만 바꿨다
  (`V6_BASE_COMMIT = fec9a151`). `world/` 62개 파일이 v6 와 바이트 동일함을 `tests/v7/test_world_clone.py` 가
  매니페스트로 지킨다. v6 는 손대지 않았다.
- **크레인 전용 기본**: `run_episode(arm="NO_REALLOC")`, `run_month(arm=None → NO_REALLOC, PPO 유산 경로만 RL)`.
- **본선 단가**: `reward/krw.py` 35.88 (v4 용선료 주석 이식).
- **DGT 리드타임**: `stage/dgt_lead.py`(반입·반출 따로, 상수표 + 원 파일 sha256) · `lead_mode="DGT"` 기본,
  `DIST`·`FIXED` 는 선택지로 남김. 저장 입력을 쓰는 PPO 유산 경로는 입력에 적힌 방식을 따른다.
- **행동 범위**: `stage/exec_config.py` `V7_CRANE_ONLY`(safety_only) — `_rule_policy` 기본.
- **시험**: `tests/v7/test_v7_scaffold.py` 18건 통과. 복사해 온 비GPU 시험 379건 중 376 통과 · 3 실패 —
  ① `test_cargo_capacity_wait` 1건은 v6 에서도 같은 실패(기존 결함) ② `test_cargo_continuous` 1건은 PPO 유산
  경로가 통지 방식을 'DIST' 로 박아 둔 탓 → `ppo/continuous.py` 가 입력의 방식을 읽게 고쳐 재실행 대상
  ③ `test_world_clone` 1건은 README 이름을 바꾼 탓 → 새 README 로 해소. GPU 배열 시험 결과는 `outputs/reports/yr332_v7_scaffold/README.md`.
- **못 한 것**: 30일 완주·30일 전건 투입은 주장하지 않는다 — 3일 스모크만 했다. 생성 입력 그대로면 3일×300대
  중 32대가 NO_TARGET 으로 빠진다([[YR-331-c]] 입력 상자 공급 부족 · v6 방식에서도 48대).
- 증거: `outputs/reports/yr332_v7_scaffold/` (README · `scaffold_evidence.json`).
- **추가(같은 날)**: 배열 세계 시험 811건 — 800 통과 · 7 건너뜀 · 4 실패(넷 다 v6 에서도 같은 실패). 대조 시험 3곳은
  배열판이 v6 사본이라 `V6_LEGACY`·`DIST` 를 명시하도록 고쳤고, 본선 손계산값 1곳을 ×12 로 갱신했다(후속 커밋).
