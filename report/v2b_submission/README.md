# V2b report draft

Team Lebmar (87) — Chengxuan YUAN.

This directory contains the current 7-page A4, 11pt English report draft
(`main.pdf`) and its LaTeX source (`main.tex`). It is a draft for participant
review, not an already submitted CMT file.

## Compile

```sh
cd report/v2b_submission
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The two passes resolve the cross-references. The figures are generated from the
frozen final arrays by:

```sh
python scripts/make_v2b_report_figures.py
```

The one-page onboarding flow is a standalone LaTeX/TikZ page in
`figures/q1_onboarding_flow.tex`; the report includes its cropped PDF.

## Scope

The report uses the frozen V2b final results and the development 1–120 Q1 product
screen. It does not fit, select or change a model. The final 121–180 sample was
previously exposed and is described as nonblind. CMT upload and final page/font
checks remain with the team leader/participant.
