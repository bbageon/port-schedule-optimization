# `.claude/Dashboard/` — 작업 board (index)

> Jira/Plane 스타일 작업 board. **상태별로 파일 분리** (각 state = 한 파일). 본 README 가 **index**
> (운영 규약 + state 링크 + epic + 현황 overview). 상세 evidence 는 설계문서/산출물/git commit 이
> single source — board 는 그 위의 index 일 뿐 중복 서술하지 않는다.
> board 규약 원본: [dashboard-board.md](../../dashboard-board.md)

## 상태별 파일

| State | 파일 | 현황 |
|---|---|---|
| 🗂️ Backlog | [backlog.md](backlog.md) | 미착수·미래 |
| 📋 Ready | [ready.md](ready.md) | 착수 준비 |
| 🟢 In Progress | [in-progress.md](in-progress.md) | 진행 중 (한 번에 1개) |
| ✅ Done | [done.md](done.md) | 완료 (evidence 박제) |
| 🚫 Cancelled | [cancelled.md](cancelled.md) | 폐기 (사유 박제) |

흐름: `Backlog → Ready → In Progress → Done / Cancelled`.

## 사용 규약 (요지)

- **Issue ID**: `YR-NNN` (yard_rl 패키지명 기반). 닫힌 ID 재사용 금지. **Priority**: 🔴/🟠/🟡/⚪. **Epic**: 아래 표.
- 상태 이동(pull/done/폐기)·evidence 박제·grooming **절차 상세**: [dashboard-ops skill](../skills/dashboard-ops.md) (index: [skills.md](../skills.md)).
- **순서 제시 원칙·1줄 index 원칙** 등 전역 규칙: [AGENTS.md](../../AGENTS.md). row 상세 명세는 [task-specs](../docs/dashboard-task-specs/) 의 `<ID>-<slug>.md` (spec 이 원본 문서 § 를 링크).

## 🧭 Epics

| Epic | 의미 | 상태 |
|---|---|---|
| Infra | 하네스·환경·도구·프로젝트 스캐폴드 | 상시 |
| Data | TOS·VBS·ETA·본선·장비·레인 통합 schema와 실자료 매핑 | active |
| Sim | 통합 이벤트 시뮬레이터·Hard Constraint·실측 validation | active |
| Baseline | 강한 동정보 휴리스틱·중앙 matching·paired runner | active |
| RL | 동적 후보 Double DQN·Total Cost·QMIX 협조학습 | active |
| Exp | 동일 통합정책의 locked 평가·ablation·민감도 | active |
| UI | 정책 replay·설명·운영자 승인/반려 피드백 | active |

## 📌 현재 상태 overview (한눈에)

- **최종 목표 전환 (2026-07-15, 사용자 결정)**: 별도 Exp 정책이 아니라 차량·본선·이송장비·레인·다중 YC를 처음부터 같은 State·Action·Total Cost 계약으로 다루는 단일 통합정책. [최종전략](../docs/부산항_레인_다중야드크레인_협조최적화_강화학습_최종전략.md)(설계 원본) · **[2026-07-16 실행 전략 정정판](../docs/strategy-history/2026-07-16-구현-RL전략-정정판.md)** (정정 트랙 완료 시점 기준 현행) · [결정 이력](../docs/strategy-history/2026-07-15-YR-034-final-integrated-strategy-pivot.md).
- **정책 구조**: 가변 후보 `Q_cost`를 Candidate Double DQN으로 평가하고 중앙 resolver가 공동제약을 보장한 뒤 QMIX 추가효과를 검증한다. 안전·물리·마감 위반은 보상이 아니라 mask다.
- **단일 야드 트랙 종료 (2026-07-15, 사용자 결정)**: "현 환경에서 greedy(SPT)는 near-optimal" 로 결론. 근거 사슬 — 격차 +1.195→+0.083(해상도)→+0.035(feature, 220k) 축소했으나 YR-033 이 정정: 동일 checkpoint 가 fresh 240k 에선 +0.111·최적선택 하한도 +0.111·winner's curse 기각(Spearman 0.96). 커버리지·초기화·구조(H-B)·해상도·학습기법·선택 전 축 소진 → robust 격차 ~+0.1 는 정책이 아니라 문제 성질. [종료 결론서](../docs/strategy-history/2026-07-15-single-yard-track-closure.md). RL 잔존가치(tail·oracle 상금 +0.182)는 통합전략(YR-034)에서 재탐색.
- **통합 파이프라인 토대 완성 (2026-07-15)**: YR-035 계약(itc-v1)·YR-036 이벤트 시뮬레이터(다중 YC·본선·이송·레인)·YR-037 동적 후보+중앙 joint resolver·YR-038 정규화 비용/보상 **모두 done** — 계약→환경→공동배정→비용 토대 완성. 각 태스크 설계→구현→적대리뷰. sim/(단일 YC) 전 구간 동결·golden 불변. 매핑: [YR-035](../docs/YR-035-통합계약-매핑.md)·[YR-036](../docs/YR-036-통합시뮬레이터-매핑.md)·[YR-037](../docs/YR-037-후보-resolver-매핑.md)·[YR-038](../docs/YR-038-비용-reward-매핑.md).
- **YR-039 승리 판정 무효 (2026-07-15 정정)**: "첫 형식 승리·−84%" 는 잘못 구현된 imbalance 항(누적 완료건수 pstdev·scale=1 placeholder)이 총비용의 97.6~99.9% 를 지배한 결과 + SPT baseline 은 REPOSITION 54%(val 5-seed)~81%(test seed) 퇴화 정책 — 어느 기준으로도 승리 주장 불가. 파이프라인 코드는 유효 잔존(로컬 utility 중간 단계). 외부 진단을 독립 재현(수치 일치)·2-agent 감사(7 주장 CONFIRMED)로 확정. [무효 판정 기록](../docs/strategy-history/2026-07-15-YR-039-무효판정-imbalance-지배.md) · [매핑](../docs/strategy-history/YR-039-학습기-매핑.md)
- **정정 트랙 YR-043·044 done (2026-07-15~16)**: YR-043 이 무효 사유 1(목적함수) — imbalance 를 작업부하 `(max−min)/ΣLoad` 로 재정의(episode_raw 14779.4→0.028)·**지배도 guard**(단일 항 >70% → run 실패)·WAIT 실행동 복구·물리 mask·λ 중립 — 을, YR-044 가 무효 사유 2(baseline 퇴화) — 고정 시간창 rollout baseline(ETA 전 역사값 93.4 < base 96.6)·**행동분포 건전성 계약**(`serve_when_available` <0.25 → 실패) — 를 코드로 제거. YR-044 는 **명세가 지정한 "즉시비용 argmin" 자체가 퇴화**함을 실측으로 발견(완료율 41%·대기 119.8분)해 재정의. 매핑: [YR-043](../docs/YR-043-목적함수-정정-매핑.md)·[YR-044](../docs/YR-044-baseline-매핑.md).
- **YR-048 ETA 경로 복구 (2026-07-16)**: 통합 시나리오 외부트럭에 ETA ±300초를 전용 난수 흐름으로 주입해 PRE_REHANDLE 후보를 0%→14~15%로 활성화. 동시에 ETA가 REPOSITION 후보도 3.3배 늘린다는 두 번째 경로를 확인해 YR-045를 `NO_ETA/ETA_NO_PRE/FULL` 3-arm으로 사전등록했다.
- **YR-050 ETA 결정 시점·연착 신호 (2026-07-17)**: SERVE 없는 한산기에도 ETA wake(`provided_eta−지평`, 도착 진실 미열람)가 선제 재조작 결정을 연다 — wake 1회당 크레인별 1회 질문(armed). 무제한 재질문 1차 설계는 결정 40→464건·REPO 88% 퇴화 실측으로 폐기. 연착 음수 gap 은 부호 보존(계약 itc-v1→**itc-v2**, golden 재생성·낮은 정보수준 거동 불변). [매핑](../docs/YR-050-결정시점-매핑.md).
- **torch 실행환경 복구 (2026-07-16, YR-046)**: Windows 11 스마트 앱 컨트롤이 서명 없는 DLL 을 차단해 torch 의존 8파일이 몇 달간 미실행이던 문제 해소 — Windows(python.org 3.12·순수파이썬/UI) + WSL(torch·학습) 2원 구성. 스마트 앱 컨트롤은 단방향이라 끄지 않음. **미실행이 감추고 있던 실제 실패 1건 발견**(YR-043 이 뒤집은 WAIT 배제 계약을 옛 가드가 붙들고 있었음) — 미실행 테스트는 통과가 아니라 미지. WSL 315 / Windows 274 passed. [환경 문서](../docs/개발환경-windows-wsl.md).
- **YR-045 locked 재실험 완료 (2026-07-18)**: **RL 3종 전부 6중 게이트 통과 0 — YR-039 무효 확증** (FULL 총비용 dueling 90.95 vs JointRollout 70.78, +28%). ETA 가치는 실재하되 **위치선점 지배(−29.1)·선제 재조작 유의 소폭(−1.6~−4.0)**. 게이트 전부 통과는 SF-SPT 4·FIFO 1 조건 — 총비용 최적과 다중 운영지표 우월이 분리됨. 전략적 WAIT 은 전 정책 손해(RL 미완주 6건 전부 허용 조건 — YR-052 판정 데이터). 실행 중 잠복 결함 2건 정정(generator 미전달·clean-tree 오탐)·BEAM==JR 완전동일 발견(YR-055). [사전등록+결과](../docs/strategy-history/2026-07-16-YR-045-corrected-locked-rerun-prereg.md) · [report](../../outputs/reports/yr045_locked_rerun/yr045_report.md).
- **YR-054 재감사·YR-055 BEAM 결함 완료 (2026-07-18)**: RL 전패의 원인은 누락이 아니라 **협조 실패 — interference(경합 양보)가 격차의 85%** ([재감사](../docs/YR-054-RL-재감사.md)). BEAM==JR 동일성은 tail 조기반환 **결함으로 확정·수정** — 수정 후에도 일관 우위는 없음 ([분석](../docs/YR-055-BEAM-tail-결함.md)).
- **YR-052·YR-056 완료 (2026-07-18)**: 전략적 WAIT 는 RL 행동공간에서 **기본 제외** (사용자 승인, `fcb377c`). 경량 협조 feature(itc-v3 COORD 5채널)는 **무효 판정(negative)** — 관측 추가로는 interference 격차가 줄지 않음 (COORD Δtotal +3.70 [+1.52,+5.93], JR 대비 격차 양 arm 잔존). 채널은 QMIX 입력으로 유지. [YR-056 §6](../docs/YR-056-협조feature-매핑.md).
- **YR-061 단일 DQN 퇴화 원인 특정 (2026-07-18)**: 11조건 진단(전 knob 무효·SERVE 붕괴) 후 3-phase 사전등록 사슬 — 미완료 페널티 기각(무발동: 완료 구조 강제, 퇴화 = 지연·공회전)·할인 정합 γ→1 기각(유의 악화) → **모방 이분 검정으로 병목 = TD 신용 희석 확정** (동일 망·인코딩의 BC 가 swa 0.491·SF_SPT +3.13 근접, 종전 RL 대비 −20%). 성능 경로 = YR-062(BC+미세조정)·YR-063(신용 개별화). [사전등록+3결과](../docs/strategy-history/2026-07-18-YR-061-미완료페널티-prereg.md). **YR-062 완료 (2026-07-18, negative)**: 전 lr 에서 TD 미세조정이 BC 를 15ep 내 파괴 — 신용 희석 2차 확증. **YR-063 완료 (2026-07-18, 부분 지지)**: 차분(counterfactual WAIT) 귀속이 scratch 학습 최초의 건강 정책(swa 0.322·완료율 1.0 — 인과 3차 확증), 단 성능은 전 비교군 열세(85.58, "일은 하나 순서를 모름"). **YR-064·065 완료 (2026-07-18)**: BC+차분 미세조정 기각(신호별 자기 고정점 — 600s 차분 천장 ≈ 85)·창 40분은 차분 계열 최초 유의 개선(78.91, swa 0.400)이나 비단조·천장 BC 미만. **단일 DQN 잠정 결론: 학습 채택 후보 = BC 56.25, 차분 개량(YR-066 🟡)은 수확 체감 — 더 큰 지렛대는 QMIX 트랙(병행 세션)·실측 게이트(YR-002)**. **(2026-07-19 사용자 판단 정정: BC 는 선생(SF-SPT)을 구조적으로 넘을 수 없어 연구 목표(휴리스틱 초과) 경로가 아님 — 진단 도구·안전 차선책 기록으로만 유지, BC×정규화 결합은 추진하지 않음. 최적화 주축 = 차분 신용 축 + 정규화 + 차분 표적 QMIX 결합.)**
- **YR-059 상태 정규화 완료 (2026-07-19, 갈린 판정)**: scale-only 정규화(itc-v4·P90 동결)가 **INDEP 를 전 tier 유의 개선**(80.51 — TD-RL 최고, JR 격차 +17.3→+12.2, P1 확증) / **QMIX 는 발산 서명만 소멸하고 성능은 +21~+29 악화** → "QMIX 발산 = 입력 표현" 가설 기각, 신용 축이 1차 용의자로 승격. 이후 RL 실험 state_norm 적용 권고 (YR-061~065 는 미적용이었음 — 결합 여지). [매핑 §4](../docs/YR-059-상태정규화-매핑.md).
- **YR-067 완료 (2026-07-19, 갈린 판정 — 주축 ① 집행)**: 정규화는 **증폭기** — scratch TD 엔 유의 악화(+14.48, 신용 희석 못 고침·YR-059 이득은 규모 의존), 차분 2400s 엔 유의 개선(**75.38·swa 0.475·대기 0.64분 — 주축 첫 합산 성공, 비-BC 학습 신기록**, SF_SPT 53.12 미달). [사전등록+결과](../docs/strategy-history/2026-07-19-YR-067-정규화결합-prereg.md).
- **YR-013c 차분 표적 QMIX 완료 (2026-07-19, G1 통과 — QMIX 계열 첫 유의 기여)**: 명시 신용 앵커(D_i) + mixer 팀 창비용 보정(λ=1) 결합이 DIFF2400_NORM 대비 **Δ −9.41 [−12.81, −5.86]** — 계열 궤적 1차 +24.8 악화 → 2차 +21~29 악화 → **3차 −9.4 개선**. 비-BC 학습 신기록 65.97·FIFO 동급, 규칙(SF-SPT 53.12)은 +12.85 미달(G2). 교훈: mixer 는 명시 신용이 놓친 교차항의 **보정 도구**. [prereg+결과](../docs/strategy-history/2026-07-19-YR-013c-차분표적QMIX-prereg.md)·[매핑 §7](../docs/YR-013-QMIX-매핑.md).
- **YR-002 재기준화 완료 — D5·D1 사용자 확정 (2026-07-19)**: 운영사 협약 트랙 **폐기** (개인 연구자 실행 불가 — 사용자 확인) → **문헌 보정 프로파일 v2** ("신항 표준 ARMG" `build_calibrated_profile` — HJNC·DGT 공개 스펙 종합, YR-022~042 yaml 재사용) + 부하 현실화(`calibrated_load_params` 40/56/80대·피크, 기본 골든 바이트 동일 계약) + **공개 turn-time 1차 대조 부분 정합**(반입 ✅ 9.9~12.1분 / 반출 하회 12.7~18.6 vs 19~33 — 보정 후보: 게이트 처리·장치율). 연구 정체성 = 문헌 보정 시뮬레이션 방법론 기여. [결정 기록](../docs/strategy-history/2026-07-19-YR-002-D1-D5-사용자결정.md).
- **YR-072 천장 재진단 (2026-07-19) — "한산 탓" 가설 기각, 헤드룸 부재는 문제 성질 (2차 확증)**: v2×문헌 부하에서 완벽정보 이득 **정확히 0 재현**·탐색 상금 mid −1.4%/high 소멸·**JR 고부하 붕괴 +15%** (YR-068 차분 급감과 같은 "고정 시간창의 규모 의존 약화" 서사). v2 지형에선 규칙(SF-SPT)이 비용·건강도(swa 0.70~0.84) 모두 우위. [진단](../docs/strategy-history/2026-07-19-YR-072-천장재진단-v2.md).
- **개정 전략 사다리 (2026-07-19, 사용자 피드백 반영 — 최종 승인 대기)**: 병렬 세션 탐색이 **OLD 목적의 2항(간섭+차선혼잡) 93% 지배**와 **NEW 목적(트럭대기 1차) JR 재실행 시 대기 최대 6배 감소**를 발견 — "헤드룸 부재 = 문제 성질"(YR-070/072)은 **OLD 목적 기준의 판정**으로 정정. 새 순서: **YR-071**(NEW 동결·v2×현실부하 G0 확증·완벽정보 NEW 재측정, ready) → **YR-073**(중앙 공동가치망 = JR_NEW 증류, QMIX 신용 분할 구조 제거) → YR-074(조건부 미세조정) → YR-075(결정권 확장 2단계). YR-066 재정의·YR-069 조건 변경. 공통 후속 후보 = 장치율 현실화 → YR-042 재개 → YR-014. **열린 문제·대응 종합**: [열린문제 전략](../docs/strategy-history/2026-07-16-열린문제-대응전략.md).
- **주장 게이트 (2026-07-19 재정의)**: "실운영 대비 개선율"·CURRENT_RULE·운영사 실측 validation 은 **영구 불가** (D5 — 협약 폐기). 가능한 주장 = "문헌 보정 시뮬레이션 조건"의 알고리즘 가설(H1~H5) 검증·방법론 기여. 외부 정합성은 YR-009(공개 실측 대조 — PNIT·HPNT turn-time)가 담당한다. 모든 결과 서술에 "문헌 보정 조건" 한정을 명시한다.
