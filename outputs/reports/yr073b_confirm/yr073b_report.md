# YR-073-b — 순위 증류 확증 재판정 (사전 지정 R1 × 신규 seed)

> 모델 outputs/reports/yr073_distill/student_v1.pt (commit 2c7410f 박제본, 재학습·재선택 없음) · 결정론 OK · **문헌 보정 시뮬레이션 조건**

## mid

- 학생 평균대기 **2.0193분** vs SF 3.7107 — Δ **-1.6914 [-2.3379, -1.1244]** → G1′ ✅ · ΔP95 -9.7996 [-12.859, -6.9701]
- REPO 0.364 (SF 0.326) · 집계 swa 0.318 · 퇴화 에피소드 3/20 (보고) · wall 0.89s/ep · guards {'completion_all1': True, 'backlog_all0': True, 'p95_ok': True, 'policy_mix_healthy': True, 'repo_relative_ok': True}

## high

- 학생 평균대기 **4.1277분** vs SF 6.74 — Δ **-2.6123 [-3.5075, -1.6811]** → G1′ ✅ · ΔP95 -9.5972 [-13.9238, -5.0952]
- REPO 0.302 (SF 0.259) · 집계 swa 0.42 · 퇴화 에피소드 2/20 (보고) · wall 1.08s/ep · guards {'completion_all1': True, 'backlog_all0': True, 'p95_ok': True, 'policy_mix_healthy': True, 'repo_relative_ok': True}

**판정: G1′ 통과** · 원자료 rows.jsonl
