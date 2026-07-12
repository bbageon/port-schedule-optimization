# YR-005 — Phase 1 후반: 데이터 전처리 파이프라인

- **Epic**: Data / **Priority**: 🟡 / **등록일**: 2026-07-12
- **배경**: [01 §6](../../../docs/구현계획/01_범위_아키텍처_데이터.md) 최소 데이터셋(job_events·eta_history·crane_tasks·container_snapshot·rehandle_log·equipment_status) + 표준화 규칙 (Asia/Seoul→UTC epoch, 비가역 hash 익명화, 날짜 단위 split).
- **목표(수용 기준)**: [05 §4 Phase 1](../../../docs/구현계획/05_테스트_로드맵_산출물.md) — 샘플 운영일이 오류·제외 보고서와 함께 로딩됨. 품질오류 자동삭제 금지 — 플래그·제외비율 보고.
- **범위 밖(non-goal)**: ETA 생성·보정.
- **계획**: loaders → anonymize → preprocess(정렬·표준화·품질 플래그) → train/val/test 분할 manifest → 소형 테스트 데이터 생성.
- **산출물**: `src/yard_rl/io/`, `data/schemas/`, 품질 리포트, 분할 manifest.
- **의존**: 실자료는 YR-002 이후. 스키마·가정 데이터 기반 구현은 선행 가능.
