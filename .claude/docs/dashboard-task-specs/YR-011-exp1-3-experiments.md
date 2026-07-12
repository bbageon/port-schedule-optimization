# YR-011 — Phase 5: Exp-1~3 정보시점·제어범위 요인실험

- **Epic**: Exp / **Priority**: 🟡 / **등록일**: 2026-07-12
- **배경**: [02 §3](../../../docs/구현계획/02_시뮬레이터_RL환경.md) 정보 공개시점(블록/게이트/사전정보) × 제어범위(sequence_only/plus_positioning/plus_pre_rehandle) 분리. [03 §3.1](../../../docs/구현계획/03_정책_실험_평가.md) — Exp-2 와 Exp-3A 는 행동공간을 같게 유지해 정보시점 효과만 식별.
- **목표(수용 기준)**: [05 §4 Phase 5](../../../docs/구현계획/05_테스트_로드맵_산출물.md) — 제공 ETA 는 Exp-3 외생입력으로만 사용되고 **ETA 정확도 실험이 없음**. sequence-only 순수 비교 + 포지셔닝·선재조작 단계별 ablation + 운영부하별 paired evaluation 완료. 미래정보 누출 자동검사 통과.
- **범위 밖(non-goal)**: ETA 오차·no-show 민감도 (연구 이후 별도 항목), 다중 YC(→ YR-013).
- **계획**: InformationFilter → control_scope 옵션 → 운영부하 시나리오(03 §3.2) → paired 실행·분석.
- **산출물**: 정보필터·실험 설정, Exp-1~3 결과 리포트 (H1~H3 판정 근거).
- **의존**: YR-009 게이트 통과, YR-010.
