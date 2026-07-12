# YR-015 — Phase 6: 검증 UI MVP (읽기 전용 replay)

- **Epic**: UI / **Priority**: 🟡 / **등록일**: 2026-07-12
- **배경**: [04](../../../docs/구현계획/04_시각화_검증_UI.md) 전체 — 목적은 발표 애니메이션이 아니라 **시뮬레이터·정책 검증** (작업 타당성, 준비행동 완료 여부, trade-off, 다중 YC 안전, Baseline/RL 차이 발생 지점). 기술: Streamlit + Plotly + Parquet replay, 시뮬레이터와 분리된 읽기 전용 구조. Exp-1·2 화면에 사전정보가 노출되지 않도록 InformationFilter 결과를 저장.
- **목표(수용 기준)**: 04 §9 MVP 완료조건 5항 — ① 운영일 단일 정책 전구간 재생 ② 임의 의사결정의 입력·후보·mask·선택 확인 ③ 동일 시나리오 Baseline/RL 동기 비교(manifest hash 검증) ④ UI 가 결과를 변경하지 않음이 regression 으로 보장 ⑤ 물리·안전 불변조건 위반 0. 성능: 1일 replay 3초 내 로딩, step 300ms 내 (04 §8.4).
- **범위 밖(non-goal)**: 실시간 시뮬레이션 결합, 수동 작업 재배정, 3D, FastAPI/React 스택(한계 확인 시에만 검토).
- **계획**: UI-1 recorder(데이터 계약) → UI-2 단일 replay → UI-3 정책 설명 패널 → UI-4 동기 비교 → UI-5 다중 YC 표시 (04 §9).
- **산출물**: `experiments/recorder.py`, `src/yard_rl/ui/`, UI 검증 테스트(04 §8).
- **의존**: recorder 스키마는 YR-006·008 산출물과 정합 필요. UI-5 는 YR-013 과 동반.
