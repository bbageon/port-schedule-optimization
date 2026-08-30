# LaTeX 원고 (영어판) — 빌드와 보는 방법

> 한국어판은 [../latex/](../latex/README-latex.md) 에 있다. **두 판은 내용이
> 같아야 한다** — 수치가 바뀌면 양쪽을 함께 고친다.

Springer LNCS 양식(`llncs.cls`)으로 조판한 최종 원고다. 템플릿 폴더는 그대로 두고
필요한 두 파일(`llncs.cls`·`splncs04.bst`)만 여기로 복사해 왔다.

```
latex/
  main.tex                  ← 표제·초록·서론·관련연구·참고문헌
  sections/03-environment.tex
  sections/04-policy.tex
  sections/05-results.tex
  sections/06-conclusion.tex   ← 재현방법 + 결론
  figures/*.svg             ← 그림 7장 (아래 §3 참조)
  llncs.cls  splncs04.bst   ← 템플릿에서 복사
```

## 1. 영어판은 pdfLaTeX 로도 된다

영어 본문이라 `pdflatex` 로 충분하다. 참고문헌에 한글 저자가 없도록 `(in Korean)` 표기로 옮겨 적었다.

```bash
pdflatex main.tex
pdflatex main.tex     # 상호참조·그림번호 확정을 위해 두 번
```

## 2. 보는 방법 — 셋 중 하나

이 컴퓨터에는 지금 TeX 이 깔려 있지 않다(`xelatex`·`pdflatex` 둘 다 없음).

### ① Overleaf — 설치 없이 가장 빠름 (권장)

1. <https://overleaf.com> 에서 **New Project → Upload Project**
2. `latex/` 폴더를 통째로 zip 으로 묶어 올린다
4. Recompile 하면 오른쪽에 PDF 가 나온다

기본 컴파일러(pdfLaTeX)로 그대로 된다.

### ② 윈도우에 MiKTeX 설치

```powershell
winget install MiKTeX.MiKTeX
# 설치 뒤 새 터미널에서
cd docs\paper\v3\latex
pdflatex main.tex
pdflatex main.tex
```

처음 빌드할 때 필요한 패키지를 자동으로 내려받겠느냐고 묻는다. **Always install** 을
고르면 편하다. 결과는 `main.pdf` 다.

### ③ WSL 에 TeX Live 설치

```bash
sudo apt update && sudo apt install -y texlive-latex-recommended texlive-latex-extra
cd /mnt/c/Users/geonu/orca/workspaces/port_reinforcement/강화학습-판매/docs/paper/v3/latex
pdflatex main.tex && pdflatex main.tex
```

용량이 크다(약 2~3 GB). 자주 쓸 게 아니면 ①이 낫다.

## 3. 그림 — SVG 를 PDF 로 바꿔야 한다

`figures/` 에 있는 것은 **SVG** 인데 LaTeX 는 PDF·EPS·PNG 만 직접 넣을 수 있다.
`main.tex` 는 `fig1-architecture.pdf` 처럼 **PDF 이름으로** 부르고 있으므로 한 번
변환해야 한다.

### Inkscape 로 (품질이 가장 좋다 · 벡터 유지)

```powershell
winget install Inkscape.Inkscape
cd docs\paper\v3\latex\figures
Get-ChildItem *.svg | ForEach-Object {
  inkscape $_.FullName --export-type=pdf --export-filename="$($_.BaseName).pdf"
}
```

### Overleaf 에서 바로 (변환 없이)

`main.tex` 머리말에 아래를 넣고 `\includegraphics` 를 `\includesvg` 로 바꾸면
Overleaf 가 알아서 변환한다.

```latex
\usepackage{svg}
\svgpath{{figures/}}
% \includegraphics[width=...]{fig4-policies.pdf}
% → \includesvg[width=...]{fig4-policies}
```

### 브라우저에서 그림만 보고 싶으면

SVG 는 그냥 브라우저로 열면 된다.

```powershell
start docs\paper\v3\latex\figures\fig4-policies.svg
```

## 4. 그림을 다시 만들려면

수치가 바뀌면 원자료에서 다시 그린다. 하드코딩된 값은 없다.

```powershell
$env:PYTHONPATH = "src"
python scripts/v3/figures.py && python scripts/v3/figures_en.py
Copy-Item outputs/v3/figures/en/*.svg docs/paper/v3/latex-en/figures/
```

| 그림 | 내용 | 원자료 |
|---|---|---|
| fig1 | 재배치 결정 흐름과 학습·운영 분리 | (도식) |
| fig2 | 시간대별 도착밀도 | 식 (1)(2) |
| fig3 | 회차별 학습·검증 손실과 탐색 확률 | `outputs/v3/month-02/history.json` |
| fig4 | 정책별 28일 총비용 감소율 | `outputs/v3/judge-30d/arms/` |
| fig5 | 날짜별 비용 차이 (짝비교 분포) | 〃 |
| fig6 | 수요수준별·행동유형별 비용 감소 | 〃 |
| fig7 | 크레인 작업순서 규칙의 수요 민감도 | `outputs/v3/base-matrix/rows.json` |

## 5. 원고 본문을 고칠 때

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
