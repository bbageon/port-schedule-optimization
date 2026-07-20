# YR-083 — Level 2 터미널 구조계약 런타임화

- **Epic**: Sim / **Priority**: 🟠 / **등록일**: 2026-07-20
- **배경**: YR-082가 Level 2 구조자료를 확보해도 현재 `IntegratedProfile`은 단일
  `block`, 역할 없는 동일 `CraneSpec`, 길이·방향·용량·허용차종이 없는 단순
  `LaneGraph`, 고정 평균 이송시간만 소비한다. HJNC·DGT도 가상 L1/L2 레인과 같은
  크레인 2대로 조립되므로 현재 코드는 실제 구조 차이를 실행하지 못한다.
- **목표**: YR-042가 평가할 블록의 물리구조를 시뮬레이터·후보·resolver가 실제로
  소비하게 만들고, 기존 체크포인트의 무재학습 호환 여부를 명시적으로 판정한다.

## 최소 계약

- `RoadSegment`: 방향, 중심선 길이, 속도·용량, 허용차종, 교차·합류 충돌그룹
- `TransferPoint`: 블록·작업면·정차위치, 허용차종, 동시 처리용량, 제어절차
- `CraneRole`: 소속 블록, 육측·해측·공유 역할, 허용작업·Bay·인계점
- `CraneInteraction`: 통과 가능 여부, 비통과 선후관계, 최소 안전거리
- `TransferFleet`: YT·AGV·S/C의 허용경로·배차·버퍼와 이동시간 파라미터 provenance.
  실제 시간분포 보정은 Level 3이며 Level 2에서는 `assumed`를 허용한다.

## 구현 순서

1. 현 프로파일을 새 계약으로 무손실 변환하는 호환 compiler를 만든다.
2. 후보 mask와 resolver가 역할·차종·인계점·레인·비통과 제약을 강제하게 한다.
3. 구조값을 바꿨는데 엔진 궤적이 바뀌지 않는 dead field가 없도록 변이검사를 둔다.
4. HJNC형 공통 후보풀과 DGT 육측·해측 분리 fixture를 별도 테스트한다.
5. 기존 상태 텐서 의미·차원이 유지되는지 검사한다. 정책 입력이 바뀌면 기존
   체크포인트의 zero-shot으로 부르지 않고 재증류·재학습 경로로 분리한다.

## 수용 기준

- 기존 문헌 보정 프로파일을 새 계약으로 변환했을 때 golden 궤적이 의도대로 보존된다.
- 방향·용량·역할·통과·인계점 제약 각각이 실행 결과를 바꾸는 양성/음성 테스트가 있다.
- 금지 차종·금지 작업면·중복작업·안전거리 위반이 후보 또는 resolver에서 0건이다.
- YR-082 Level 2 manifest를 실행 프로파일로 compile하고 미지원 필드는 조용히 버리지 않는다.
- `ZERO-SHOT COMPATIBLE / SCHEMA ADAPTATION REQUIRED / STRUCTURE UNSUPPORTED`를 판정한다.

- **산출물**: 확장 profile schema·loader, 구조 compiler, mask/resolver 연계, 회귀·변이 테스트,
  체크포인트 호환성 보고서
- **의존**: YR-082 증거 계약·Level 2 후보, YR-009 기준환경 manifest
- **후속**: 호환 프로파일 2개 이상이면 YR-042 실제 구조 arm, 아니면 stress arm만 실행
- **범위 밖**: 다중 블록 상위관제·가변 크레인 수(YR-081), 실제 운영로그 보정(Level 3)
