# LaTeX 원고 (영어판) — 빌드와 보는 방법

> 한국어판은 [../latex/](../latex/README-latex.md) 에 있다. **두 판은 내용이
> 같아야 한다** — 수치가 바뀌면 양쪽을 함께 고친다.

Springer LNCS 양식(`llncs.cls`)으로 조판한 최종 원고다. 템플릿 폴더는 그대로 두고
필요한 두 파일(`llncs.cls`·`splncs04.bst`)만 여기로 복사해 왔다.

```
v3/
  figures/                  ← 국문·영문이 함께 쓰는 그림 PDF/SVG/PNG
  latex-en/
    main.tex                ← 표제·초록·서론·관련연구·참고문헌
    sections/*.tex          ← 방법·환경·결과·재현방법·결론
    llncs.cls  splncs04.bst ← LNCS 템플릿 파일
```

## 1. 영어판은 pdfLaTeX 로도 된다

영어 본문이라 `pdflatex` 로 충분하다. 참고문헌에 한글 저자가 없도록 `(in Korean)` 표기로 옮겨 적었다.

```bash
pdflatex main.tex
pdflatex main.tex     # 상호참조·그림번호 확정을 위해 두 번
```

## 2. 보는 방법 — 셋 중 하나

이 저장소에서는 MiKTeX의 `pdflatex`로 빌드를 검증했다.

### ① Overleaf — 설치 없이 가장 빠름 (권장)

1. <https://overleaf.com> 에서 **New Project → Upload Project**
2. `v3/figures/`와 `v3/latex-en/`을 같은 구조로 묶어 올린다
3. Main document를 `latex-en/main.tex`으로 정한다
4. Recompile 하면 오른쪽에 PDF 가 나온다

기본 컴파일러(pdfLaTeX)로 그대로 된다.

### ② 윈도우에 MiKTeX 설치

```powershell
winget install MiKTeX.MiKTeX
# 설치 뒤 새 터미널에서
cd docs\paper\v3\latex-en
pdflatex main.tex
pdflatex main.tex
```

처음 빌드할 때 필요한 패키지를 자동으로 내려받겠느냐고 묻는다. **Always install** 을
고르면 편하다. 결과는 `main.pdf` 다.

### ③ WSL 에 TeX Live 설치

```bash
sudo apt update && sudo apt install -y texlive-latex-recommended texlive-latex-extra
cd /mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매/docs/paper/v3/latex-en
pdflatex main.tex && pdflatex main.tex
```

용량이 크다(약 2~3 GB). 자주 쓸 게 아니면 ①이 낫다.

## 3. 그림

그림은 `docs/paper/v3/figures/` **한 곳**에 있고 두 논문이 `\graphicspath{{../figures/}}`
로 같이 본다. 모두 matplotlib 으로 그린 벡터 PDF 이므로 변환이 필요 없다.

| 파일 | 내용 | 생성기 |
|---|---|---|
| `fig-arch.pdf` | 운영 추론과 오프라인 학습의 분리 | `scripts/v3/fig_arch.py` |
| `fig-demand.pdf` | 일일 수요 분포와 시간대별 도착 과정 | `scripts/v3/fig_demand.py` |
| `fig-mlp.pdf` | 후보별 비용 신경망의 구조 | `scripts/v3/fig_mlp.py` |

다시 만들려면 저장소 뿌리에서 돌린다. 수치를 손으로 적어 넣은 곳은 없다 —
`fig-demand` 는 `LOAD_WEIGHTS`·`DIURNAL_PEAKS` 를 구현에서 직접 읽는다.

```powershell
$env:PYTHONPATH = "src"
python scripts/v3/fig_arch.py
python scripts/v3/fig_demand.py
python scripts/v3/fig_mlp.py
```

**그림에는 제목을 넣지 않는다.** 설명은 LaTeX `\caption` 이 그림 아래에 붙인다.
그림은 LNCS 본문 폭(4.80 in)으로 그리므로 `width=\textwidth` 로 넣으면 축소되지
않는다 — 넓게 그려서 줄이면 글씨가 같이 줄어 읽히지 않는다.

`_provided/` 에는 쓰지 않는 원본 이미지가 이유와 함께 남아 있다.

## 4. 원고 본문을 고칠 때

LaTeX 는 [md 원고](../README.md)를 옮긴 것이다. **수치가 바뀌면 md 를 먼저 고치고**
여기에 반영한다. 표 번호와 그림 번호는 LaTeX 가 자동으로 매기므로 본문에서는
`\ref{tab:paired}` 처럼 이름으로 부른다.

| md 문서 | LaTeX 위치 |
|---|---|
| [README](../README.md) 초록 | `main.tex` `\begin{abstract}` |
| [1 서론](../1-서론.md) | `main.tex` `\section{서론}` |
| [2 관련연구](../2-관련연구.md) | `main.tex` `\section{관련 연구}` |
| [3 실험환경](../3-실험환경.md) | `sections/03-environment.tex` |
| [4 정책과 학습](../4-정책과-학습.md) | `sections/04-policy.tex` |
| [5 실험결과](../5-실험결과.md) | `sections/05-results.tex` |
| [6 재현방법](../6-재현방법.md)·[7 결론](../7-결론.md) | `sections/06-conclusion.tex` |
| [8 참고문헌](../8-참고문헌.md) | `main.tex` `thebibliography` |
