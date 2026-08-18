# `.claude/skills/` index

> 구체적 기술·방법론·절차(skill)는 `skills/` 폴더에 파일로 두고, 본 파일이 그 목록을 관리한다.
> 새 skill 추가 시 반드시 여기에 1줄 등록한다.

| Skill | 파일 | 용도 (트리거) |
|---|---|---|
| RL 학습 전용 세션 | [skills/rl-training-session.md](skills/rl-training-session.md) | RL 재학습을 별도 세션에서 돌릴 때·두 세션 동시 작업 시 레인 분리·충돌 방지·WSL 실행·실행 전 필수 4항 |
| Dashboard 운영 절차 | [skills/dashboard-ops.md](skills/dashboard-ops.md) | 세션 시작·작업 착수/종료(commit·push)·후속 등록·폐기 시 board 갱신 절차 |
| 연구 진행 3대 게이트 | [skills/research-gates.md](skills/research-gates.md) | 실험 종료·다음 작업 착수 전 성능·증거 정합·시나리오 현실성 판정과 새 가설 허용 범위 검사 |
