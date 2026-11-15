# 학습용 합성 항만 데이터의 선행연구와 적용 범위

검토일: 2026-11-15. 연결 작업: YR-332-ao(날짜별 도착 모양·작업 비율 변동).
연구 질문은 한 가지 하루에 맞춘 작업 순서가 다른 하루에도 통하는가이다.
합성 시나리오 생성의 근거와 학습·검증 방법의 근거를 구분한다.

## 1. Hartmann 2004 — 항만 물류 시나리오 생성

**서지**: Sönke Hartmann, *Generating Scenarios for Simulation and Optimization of Container Terminal Logistics*,
OR Spectrum 26, 171–192. DOI: 10.1007/s00291-003-0150-6.
[저자 공개 원문](https://www.hsba.de/fileadmin/user_upload/bereiche/_dokumente/6-forschung/profs-publikationen/Hartmann_2004_Generating_Scenarios_for_Simulation_and.pdf).

- §2.2·3·4: 운송수단별 물량, 요일·시간대 도착 비중, 컨테이너 속성·장치 기간을 매개변수로 지정한다.
  반입 운송수단과 반출 운송수단을 연결해 입력 분포들을 함께 만족시키는 시나리오를 만든다.
- 우리 적용: 오더 네 종류를 독립적으로 뿌리기보다 반입→적하, 양하→반출의 공급 관계와 재고를 보존한다.
  생성 조건과 생성 결과의 통계를 함께 저장한다.
- 한계: 이 논문의 시간대 내 모양은 날짜마다 같다는 가정이고 트럭은 한 상자 가정이다.
  날짜별 모양 변화와 여러 상자 방문을 이 논문이 이미 검증했다고 쓰지 않는다. 함부르크 수치를 부산에 복사하지 않는다.

## 2. ConFlowGen — 공개 항만 컨테이너 흐름 생성기

**서지**: Kastner, Grasse, Jahn (2022), *Container Flow Generation for Maritime Container Terminals*,
Dynamics in Logistics, 133–143, DOI: 10.1007/978-3-031-05359-7_11.
이번에 직접 확인한 방법 설명은 저자들의 [공식 배경 문서](https://conflowgen.readthedocs.io/latest/background.html)와
[버전 3.0.0 입력 분포 문서](https://conflowgen.readthedocs.io/v3.0.0/notebooks/input_distributions.html)다.
2022 장 원문은 이번 도구에서 열리지 않았으므로 세부 구현의 근거를 공식 문서로 표시한다.

- 선박 등 계획 운송수단의 일정과 트럭의 도착 분포, 컨테이너 장치 기간을 결합한다.
  시나리오 생성기를 운영 시뮬레이션과 분리해 같은 입력을 재사용한다.
- 트럭 분포는 일주일의 시간대별 확률이다. 예약 시스템 유무에 따른 예시가 있으며,
  운영자 인터뷰를 참고하고 비밀 유지를 위해 변형했다고 출처를 명시한다.
- 우리 적용: 생성식·설정·난수 시드로 입력을 재생성하고 모든 정책에 같은 입력을 공급한다.
  합성이라는 이유로 집계 통계의 활용 사실을 생략하지 않는다.
- 한계: 공개 기본 분포는 부산의 관측 분포가 아니다. 변형했다는 사실이 익명성 보장을 뜻하지도 않는다.
  이번에는 생성기 전체를 교체하지 않고 설계 원칙을 참고한다.

## 3. Zhang 등 2020 — 합성 작업장 문제로 작업 순서 학습

**서지**: Cong Zhang et al., *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning*,
NeurIPS 2020. [학회 원문](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf), §5.

- Taillard 방식으로 여러 규모의 작업장 문제를 생성했다. 각 규모에서 10,000번 갱신하며
  회당 독립 문제 4개를 사용했고, 학습 중 확인할 검증 문제 100개는 고정했다.
  생성 문제뿐 아니라 공개 문제와 학습보다 큰 문제에서도 평가했다.
- 우리 적용: 새로운 하루를 생성하는 학습과 고정된 선택용·최종 평가 입력을 분리한다.
  같은 곡선의 새 시드, 다른 곡선 조합, 학습 범위 밖 조건을 서로 다른 일반화 시험으로 보고한다.
- 한계: 정적 작업장 일정 문제이며 동적 항만 도착의 수치 근거가 아니다.
  위 학습 횟수를 우리에게 필요한 횟수나 효과 보장으로 가져오지 않는다.

## 4. Muratore 등 2018 — 여러 시뮬레이션 조건에서 정책 학습

**서지**: Fabio Muratore et al., *Domain Randomization for Simulation-Based Policy Optimization with Transferability Assessment*,
CoRL / PMLR 87, 700–713. [학회 원문 및 자료](https://proceedings.mlr.press/v87/muratore18a.html).

- 시뮬레이터의 매개변수를 바꾼 여러 모형에서 학습하고, 특정 모형에만 맞추는 편향을 평가했다.
- 우리 적용: 하루 시작에 도착 곡선의 조건을 뽑고 그 하루 동안 유지한다.
  조건이 달라져도 관측 가능한 상태로 작업 순서를 고르게 한다.
- 한계: 로봇 실험이다. 항만 피크에 ±10%를 주라는 근거나 항만 현장 이전의 보장은 아니다.
  변화 범위·함께 움직이는 변수의 관계는 별도로 정해야 한다.

## 5. Sargent 2013 — 용도에 따른 시뮬레이션 검증

**서지**: R. G. Sargent, *Verification and validation of simulation models*, Journal of Simulation 7, 12–24,
DOI: 10.1057/jos.2012.20. [원문](https://ics-websites.science.uu.nl/docs/vakken/msosi/VerificationValidationSargent.pdf), §1·4·8.

- 모형의 타당성은 목적과 적용 조건에 따라 판단한다. 평균뿐 아니라 분산 등 필요한 출력 특성을 보고,
  허용 정확도를 개발 초기에 정하도록 설명한다.
- 우리 적용: 평균 시간대 곡선, 날짜 차이, 분 단위 몰림, 재고·사건 순서, 결과 지표를 따로 검사한다.
  정책 성능을 보고 잘 맞는 입력만 골라 타당성 기준을 정하지 않는다.
- 한계: 이 논문은 우리의 오차 허용치나 현실성 통과 여부를 정해 주지 않는다.

## 6. Agarwal 등 2021 — 초기값과 평가 표본의 불확실성

**서지**: Rishabh Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*,
NeurIPS 2021. [학회 원문](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).

- 소수 학습 실행의 평균만으로 순위를 매기면 불확실성을 놓칠 수 있음을 보여 주고 구간 추정을 제시한다.
- 우리 적용: 입력 시드와 신경망 초기값을 분리하고, 같은 입력의 정책 간 차이와 초기값별 결과를 모두 보고한다.
  같은 하루의 오더 수천 건을 수천 개의 독립 실험으로 세지 않는다.
- 한계: 특정 모델 수가 항상 충분하다는 근거나 현재 항만 정책의 개선 증거는 아니다.

## 채택한 방향

항만 흐름 보존 + 조건별 합성 문제 생성 + 고정 검증 입력 + 조건·초기값별 불확실성 보고를 결합한다.
이는 선행연구를 바탕으로 한 **우리의 설계 선택**이며, 특정 논문의 알고리즘 전체를 재현한 것은 아니다.
곡선의 숫자 범위와 학습 예산은 아래 설계에 따른 입력·학습 진단 후 동결한다.
[구체 설계](synthetic-demand-design.md).
