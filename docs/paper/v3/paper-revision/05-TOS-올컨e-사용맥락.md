# 부산항·TOS·올컨e에 연결하는 사용자 맥락

대상: 리뷰 3. **관련 문헌과 함께 누가 제안을 보고 어떤 결정을 하는지 보여 주는 것이 최소 대응이다.**
실제 TOS(터미널 운영시스템) 연동이나 사용자 평가를 구현·수행한 것으로 서술하지 않는다.

## 1. 인용할 근거와 역할

| 출처 | 확인한 내용 | 원고에서 쓰는 역할 |
|---|---|---|
| 김영일·신재영·박형준, 2022 | 부산 신항 반출입 자료를 이용한 대기시간 예측; 예약·혼잡 정보와 실제 대기 사이 차이 문제 | 부산항에서 운전자·배차 담당자가 이용하는 정보와 의사결정의 필요성 |
| 원승환·조성우·이언경, 2025 | 국내 TOS의 기능·지원 수준과 기술·산업 활성화를 전문가 면담 등으로 검토 | 정책 제안이 전달될 터미널 업무 시스템의 배경 |
| 최용석, 2022 | 항만 운영시스템에서 인공지능 적용 우선순위를 분석하고 야드 계획 등을 다룸 | 제안 알고리즘을 운영시스템의 계획 지원 구성요소로 위치시킴 |
| 부산항만공사 올컨e 소개, 2022 | 모바일 차량반출입예약·환적 운송·정보 서비스 | 터미널 밖의 운송사·운전자와 연결되는 사용 접점의 실제 사례 |

출처: [부산 신항 대기시간 연구](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002873252), [TOS 연구](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003186700), [인공지능 적용 연구](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002828888), [올컨e 공식 소개](https://www.busanpa.com/board/view.do?boardId=BBS_0000031&dataSid=27966&menuCd=DOM_000000105002001000&paging=ok&startPage=154).
TOS 연구 전체가 부산항 현장 실험인 것은 아니다. 올컨e 소개는 학술논문이 아니라 운영 주체의 공식 자료로 구분해 인용한다.

## 2. 대상 사용자와 권한 — 제안 시나리오

| 사용자 | 현재 업무와 필요 | 제안 시스템에서의 역할 |
|---|---|---|
| 터미널 야드 계획·운영 담당자 | 블록 여유와 작업 흐름을 보며 배정을 관리 | 블록·시간 변경 권고의 사유와 영향 확인, 수정 또는 거절, 운영 배정 확정 |
| 운송사 배차 담당자 | 기사·차량·후속 운송 일정에 맞춰 방문 계획 관리 | 제시된 시간 변경 가능 여부 확인, 허용 범위 제시 또는 거절 |
| 트럭 운전자 | 확정된 방문 시각·목적지에 맞춰 이동 | 확정 안내와 변경 여부 확인, 수행 곤란 상황 전달 |

권한 표는 앞으로 적용할 때의 설계 제안이다. 실제 TOS·올컨e의 승인 권한이나 화면을 조사해 확정한 기능 명세가 아니다.
정책의 제안망·수락망은 소프트웨어 내부의 비용 판단이다. 제안망이 운영자, 수락망이 사람 운송사라는 일대일 대응은 하지 않는다.

## 3. 논문에 넣을 하나의 사용 사례

1. TOS의 기존 배정과 공개된 도착 계획이 입력된다. 예를 들어 한 반입 요청의 기존 블록과 예약 시각이 정해져 있다.
2. 내부 정책은 가능한 후보를 비교해 다른 블록 또는 나중 도착 시각을 권고한다. 두 값을 항상 동시에 바꾸는 것으로 묘사하지 않는다.
3. 운영 담당자는 기존/변경 배정, 결정 직전 혼잡 정보, 주행 차이와 변경 이유를 본다.
4. 시간 변경이면 운송사 배차 담당자가 실제 이동·후속 일정과의 양립 가능성을 확인한다. 이 협의는 현재 실험의 내부 수락망과 별개다.
5. 관련 주체가 실행 가능한 변경으로 확인한 뒤 확정 정보를 TOS와 운전자 접점으로 전달하는 흐름을 제안한다.
6. 거절·응답 지연·상황 변경이면 기존 배정 유지 또는 재검토 여부를 정한 운영 규칙을 따른다. 이 규칙의 구현·효과는 후속 과제다.

흐름 요약: **기존 배정 → 정책 권고 → 운영자 검토 → 필요한 운송사 협의 → 확정 안내**.
TOS·올컨e는 이 흐름이 놓일 수 있는 접점의 사례다. 공개 API나 실제 데이터 연동 가능성이 확인됐다고 주장하지 않는다.

## 4. 사용자가 알아야 할 정보

- 무엇이 바뀌는가: 기존 블록/시간과 권고 블록/시간.
- 왜 제안했는가: 현재 혼잡, 블록 간 처리 여력, 주행 차이처럼 실제 관측 가능한 이유.
- 언제까지 판단해야 하는가: 운영 계약에서 정한 응답·확정 시각. 현재 모델의 검토 주기와 사람의 응답 시간은 다름.
- 어떤 선택이 가능한가: 승인·가능한 시간 제시·거절과 그 결과.
- 무엇이 아직 불확실한가: 예약 정원·운송사 허용 구간·정보 갱신 시점 등 미확인 조건.

쌍 중심 비용 점수를 정확한 원화 절감액이나 신뢰 확률처럼 표시하지 않는다. 그런 표시에는 별도 보정과 검증이 필요하다.
사용자 화면에는 학습망 내부 번호나 반사실 분기 구조를 보여 주기보다 실제 결정을 돕는 정보를 제시한다.

## 5. 원고에 넣을 문단 초안

> The intended users are terminal yard planners and operations staff, with dispatchers and truck drivers involved when an arrival-time change is proposed. The architecture is intended as a recommendation component alongside a terminal operating system. Operators would review the current and proposed assignments and their operational context; time changes would additionally require coordination with the carrier. Busan-port studies on truck waiting information and the AllCONe service provide relevant usage contexts, rather than evidence of an implemented integration.

> The proposal and acceptance networks represent internal computational decisions. They should not be interpreted as human approval. The present study evaluates simulated operational costs; interaction design, user acceptance, and field integration remain to be assessed.

의미: 누구를 위한 지원 도구인지, 사람이 개입할 결정은 무엇인지, 현재 검증하지 않은 것은 무엇인지 한 문맥 안에서 밝힌다.
위 첫 문단의 관련 주장 옆에 부산항 대기시간·TOS·올컨e 출처를 붙인다. 일반적인 HCI 연결 문장만 따로 추가하지 않는다.

## 6. 답변서 전략과 한계

리뷰어가 직접 요구한 것은 대상 사용자, 의사결정에서의 역할, 사용 맥락의 명확화다. 이 세 가지를 역할 표·사용 사례·문헌으로 보강한다.
사용자 실험을 자동으로 필수 추가하지 않는다. 다만 사용성 향상·업무 부담 감소·사용자 신뢰 향상을 주장하려면 해당 사용자 평가가 필요하다.
인용을 몇 개 추가한 것만으로 HCI 기여가 충분해졌다고 단정할 수는 없다. 논문의 기여를 운영 의사결정 지원 알고리즘과 그 적용 맥락의 수준으로 제한한다.

현재 완료는 문헌 확인과 시나리오 초안 작성이다. 원고 반영·사용자 접촉·현장 연동·사용자 평가는 수행하지 않았다.
담당: [YR-317-f 사용자 맥락](../../../../.claude/docs/dashboard-task-specs/YR-317-f-v3-review-hci-context.md).
