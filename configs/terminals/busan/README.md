# 부산항 터미널 환경 레지스트리 (YR-082-A) — index

> 계층형 환경 모듈화. **① nameplate 층**([manifest.yaml](manifest.yaml)) = 필드별 증거등급,
> **② archetype 층**(4개 구조군) = 선택 시 "현 엔진에서 실행 가능한가"를 게이트로 강제.
> 선택기: [`src/yard_rl/integrated/terminal_registry.py`](../../../src/yard_rl/integrated/terminal_registry.py).
> 원본(single source, 사용자 소유 로컬): `부산항_터미널별_강화학습_환경정보_상세조사.md` (938줄).
> spec: [YR-082](../../../.claude/docs/dashboard-task-specs/YR-082-busan-terminal-poc-profile.md).

## 주장 게이트 (먼저 읽기)

전 터미널 **Level 0~1**. 여기서 나오는 프로파일은 전부 **Level 1 stress** — 문헌 보정 assumed
physics 위에 공개 확인값을 극소 오버레이할 뿐이다. **터미널별 실운영 성능 주장 금지.**
가능한 주장 = "문헌·공개자료 보정 시뮬레이션"의 방법론 결과. Level 2(구조 재현)는 블록좌표·
Bay·레인그래프·크레인 통과규칙 확인 후, Level 3(성능)은 TOS/VBS/RTLS 실측 후, Level 4(현장)는
별도 승인 후에만 연다.

## 4개 구조군 (archetype 층) — 10개 전부 선택·실행 가능

| 구조군 | 터미널 | 충실도 | 근거 |
|---|---|---|---|
| `PROVISIONAL_RMG_ATC_YT` | PNIT·PNC·HJNC·HPNT | ✅ **faithful** | RMG/ATC·YT 잠정군. **HJNC만 수평 확인**, 나머지 배열방향 unresolved |
| `VERTICAL_ARMG_AGV` | DGT | ⚠️ nameplate | 블록당 2기 육/해측 역할분리·AGV — 미모형, 공용 substrate 근사 |
| `VERTICAL_TC_SC` | BNCT·BCT | ⚠️ nameplate | S/C 이동·인계가 YT/AGV 와 달라 미모형 |
| `CONVENTIONAL_MIXED_PROVISIONAL` | BPT 신선대·감만·HKT | ⚠️ nameplate | 북항 혼합 — 장비형식조차 unresolved |

**10개 전부 선택·실행 가능**하다. 단 **충실도 게이트**로 정직성을 지킨다: 수평형은 `faithful=True`
(엔진이 SHARED 2크레인·YT 구조를 충실히 실행), 수직·혼합형은 `faithful=False` — 역할분리·S/C·AGV
역학이 현 엔진에 없어 **공용 assumed substrate 위의 "이름표 stress"** 로 실행된다(`warnings` 동반).
이름표 stress 실행을 **"해당 터미널 성능"으로 주장하는 것은 금지**(claim gate) — 충실 실행은 YR-083
(도로·인계점·크레인 역할 런타임화) 이후에 열린다. claim-bearing 실험 코드는 `require_faithful=True`
로 수직·혼합형을 차단(`StructureBlockedError`)해 사고를 막는다.

## 선택기 사용법

```python
from yard_rl.integrated.terminal_registry import (
    build_stress_profile, list_terminals, faithful_terminals, StructureBlockedError)

list_terminals()          # 10개 (id·구조군·Level·selectable·faithful) 선택 표면 — 전부 selectable
faithful_terminals()      # ['PNIT', 'PNC', 'HJNC', 'HPNT'] — 구조 충실 실행 집합

env = build_stress_profile("DGT")    # 10개 전부 성공 (막지 않음)
env.profile               # build_calibrated_profile() 자리에 그대로 꽂는 IntegratedProfile
env.faithful              # False (DGT=nameplate stress)
env.data_grade            # "Level1-NAMEPLATE(구조미충실)"
env.warnings              # 미충실 사유·주장 금지 경고
env.physics_overlays      # 실제 물리에 반영한 확인/유도 오버레이 (예: PNIT 열폭 2.84)

build_stress_profile("DGT", require_faithful=True)   # 실험 보호: StructureBlockedError
```

## "터미널 선택"이 실제로 바꾸는 것 (정직성)

공개자료로 블록 물리에 닿는 확인값이 거의 없어, **같은 구조군 안에서는 시뮬레이션이 사실상
동일**하다. 검증 실측:

- `HJNC-STRESS` 물리 = 현 `SNP-ARMG-STD` **완전 동일** (이름표만 다름).
- 4개 수평형 중 **PNIT만 열폭 2.84m**(레일간격 28.4m÷10열, 유도값) — 유일한 실물리 미세차.
  PNC·HPNT 는 레일간격 미공개라 base(3.1m) 그대로.
- 나머지(크레인 속도·bay 수·lock 시간·도착과정·레인그래프·블록당 대수)는 **전 터미널 동일
  assumed** — RL 거동을 지배하는 값 전부가 미공개라 구조군 physics_base 를 공유한다.

즉 지금 "환경 선택"은 대부분 **증거·라벨·게이트**를 바꾸지, 동역학을 바꾸지 않는다. 동역학이
실제로 갈리려면 수직형 역할분리를 엔진이 소비해야 하고(YR-083), 그 전까진 정직하게 stress 로만
쓴다.

## 자료충족도 · YR-042 자격 판정

| 항목 | 판정 |
|---|---|
| Level 2 적격 터미널 | **0개** (HJNC·DGT 가 구조 확인 최다이나 블록좌표·Bay·레인그래프·통과규칙 부재) |
| YR-042 cross-terminal structure arm | **DATA-BLOCKED — NO ELIGIBLE PROFILE** (같은 구조군 Level2 2개 미만) |
| 허용되는 시험 | 수평형 assumed profile **stress arm** 만 (성능주장 금지, profile-shift stress 표기) |

## 현장 자료요청 체크리스트 (Level 2 승격 선결)

터미널별로 아래가 확인돼야 Level 2(YR-042 대상) 자격이 열린다 — 전부 현재 unresolved:

- **블록**: ID·좌표·Bay 수·진입/출구·작업면, (수직형) 육/해측 작업면 경계
- **크레인**: 블록당 실제 배치·담당 Bay·통과 여부·선후관계·최소 안전거리
- **레인/인계점**: 방향·중심선 길이·용량·허용차종, (DGT) AGV 레인 네트워크·게이트 레인 수
- **속도/시간**(Level 3): TOS/VBS/RTLS 실측 gantry/trolley/hoist·작업시간 분포·큐 등록규칙

## 다음

- **YR-083** — Level 2 구조계약 런타임화(도로·인계점·크레인 역할을 엔진이 실제 소비). 이게 되면
  수직형(DGT 등) 구조군이 `runnable_in_current_engine` 로 승격되고 선택이 동역학을 바꾼다.
- **YR-042** — 같은 구조군 Level 2 프로파일 2개 확보 시 무재학습 게이트 개방.
