# yard-rl — 부산항 야드크레인 강화학습 PoC

이벤트 기반 야드 시뮬레이터 위에서 priority-rule 선택형 강화학습(Tabular Q-learning, SMDP)으로
야드크레인 작업순서를 최적화하는 PoC. 설계 원본은 루트의
[실험설계안](부산항_야드크레인_강화학습_실험설계안_업데이트.md)과
[구현계획서](부산항_야드크레인_강화학습_구현계획서.md)(→ `docs/구현계획/01~05`) 참조.

## 설치·실행

```bash
pip install -e .[dev]
pytest                                  # 단위·불변조건·회귀 테스트
python -m yard_rl.cli run-exp1          # Exp-1 예비 PoC (합성 시나리오)
```

## 현재 상태

- **가정 프로파일 기반 예비 PoC** — `configs/terminals/poc_single_crane.yaml` 의 모든 수치는
  `assumed: true` (실측 아님). 실자료 확보(YR-002) 후 보정·validation(YR-009) 예정.
- 작업 현황은 `.claude/Dashboard/` board 참조.
