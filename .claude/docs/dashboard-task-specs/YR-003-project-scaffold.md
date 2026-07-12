# YR-003 — 프로젝트 스캐폴드 + git init

- **Epic**: Infra / **Priority**: 🟠 / **등록일**: 2026-07-12
- **배경**: [01 §4](../../../docs/구현계획/01_범위_아키텍처_데이터.md) 프로젝트 구조. board 의 done evidence 가 commit 기반이므로 git 저장소가 선행되어야 함.
- **목표(수용 기준)**: ① `pip install -e .` 성공 ② pytest 스위트(빈 테스트 포함) 통과 ③ 01 §4 디렉토리 골격 존재 ④ git init + 첫 commit (data/raw 등 형상관리 제외 설정 포함).
- **범위 밖(non-goal)**: 실제 도메인 로직·시뮬레이터 코드.
- **계획**: pyproject.toml → src/yard_rl 패키지 골격(domain·io·sim·envs·policies·experiments·ui) → tests/(unit·integration·invariants·regression) → configs/·data/·outputs/ → .gitignore → git init·commit.
- **산출물**: 저장소 골격 일체, 첫 commit 해시 (board evidence 체계 가동 시작).
