# YR-332-be — v7 물리 코드 변경 기록 대조

- **상태**: backlog
- 상위: [[YR-332]]. 파생: [[YR-332-i]], 2026-11-21.
- 보정 대상: reliability — 코드·등록 변경·검사 근거의 일치.

## 무엇과 왜

물리 코드 4파일의 현재 내용이 복제 기준 기록과 다르다.
적하 하한을 고치기 전 `06d56e71`에서도 같은 실패가 재현돼 이번 산술 수정과 분리한다.
과거 변경이 승인된 것인지, 기록만 빠졌는지 확인해야 전체 검사 결과를 믿을 수 있다.

## 범위와 완료 기준

1. `engine.py`, `multiblock.py`, `order_schema.py`, `time_contract.py`의 변경 커밋을 추적한다.
2. 변경마다 Dashboard 명세·기존 실행 증거와 대응시킨다. 근거 없는 변경은 별도로 표시한다.
3. 확인된 변경만 `tests/v7/v6-copy-manifest.json`과 범위 검사에 등록한다. 기대 해시 일괄 덮어쓰기 금지.
4. 확인 과정에서 실제 결함이 발견되면 단일축 보정 범위를 먼저 등록한다.
5. 복제 검사·영향받은 동작 검사와 작업 목록 정합을 확인한다. 정책·물리 설정·보상 변경은 범위 밖이다.

현재는 원인 대조 전이며 코드가 잘못됐다는 판정이 아니다.
착수 판정은 `outputs/reports/yr332_load_bound/final-verdict.json`의 신뢰성 미통과를 근거로 한다.
월 재고 감소는 [[YR-332-bf]], 운반 외부 자료는 [[YR-332-i]]에서 다룬다.

[발견 보고서](../../../outputs/reports/yr332_load_bound/README.md) ·
[수정 전 재현](../../../outputs/reports/yr332_load_bound/baseline-check.json).
