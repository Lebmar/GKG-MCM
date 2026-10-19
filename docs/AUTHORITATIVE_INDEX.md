# Authoritative document index

**Read this first when preparing the report or checking a number.** The project
contains multiple generations of evidence. Historical files remain valid at their
own timestamps; they are not all current. This index prevents using a superseded
V1/V2/earlier-final number in the final submission.

Last updated: 2026-10-20 (Q1 guard generalization addendum). Current branch:
`codex/risk-budget-development`.

## Current authoritative chain

| Purpose | Authoritative file |
| --- | --- |
| Q1 guard generalization to B/E (development-only addendum, 2026-10-20) | [q1_guard_generalization_protocol.json](q1_guard_generalization_protocol.json), [judgment](q1_guard_generalization_judgment.json), [results](../results/q1_guard_generalization_v1/), [report note](../submission/Q1_GUARD_GENERALIZATION.md) |
| Locked V2b model, Q1 products/assignments, Q2 rule, final scope | [mitigation_main_model_registry.json](mitigation_main_model_registry.json) |
| Technical judgment, final rows and parameter meanings | [mitigation_technical_completion_judgment.json](mitigation_technical_completion_judgment.json) |
| Final 121–180 evaluation (14 objects, 60 paths) | [results/mitigation_locked_final_evaluation_v1/](../results/mitigation_locked_final_evaluation_v1/) |
| Independent final audit | [independent_audit.json](../results/mitigation_locked_final_evaluation_v1/independent_audit.json) |
| Isolated final replay | [results/mitigation_locked_final_reproduction_v1/](../results/mitigation_locked_final_reproduction_v1/) |
| Requirements/evidence closure | [technical_evidence_closure_2026_10_19.json](technical_evidence_closure_2026_10_19.json) |
| Last-mile diagnostics and verification | [technical_last_mile_completion_2026_10_19.json](technical_last_mile_completion_2026_10_19.json) |
| Task-facing answer sheet | [TASK_ANSWER_SHEET.md](../submission/TASK_ANSWER_SHEET.md) |
| Q1 5×3 product screen | [Q1_PRODUCT_COMPARISON.md](../submission/Q1_PRODUCT_COMPARISON.md) and [results/q1_product_comparison_v1/](../results/q1_product_comparison_v1/) |
| Q1 one-page onboarding flow | [Q1_ONBOARDING_FLOW.md](../submission/Q1_ONBOARDING_FLOW.md) |
| V2b report draft | [main.pdf](../report/v2b_submission/main.pdf) and [main.tex](../report/v2b_submission/main.tex) |
| Q3 sensitivity note | [Q3_SENSITIVITY.md](../submission/Q3_SENSITIVITY.md) |
| Compact final reproduction guide | [FINAL_REPRODUCTION.md](../submission/FINAL_REPRODUCTION.md) |
| Closure verification v2 (Q1 table, report, v4 package) | [results/submission_closure_v2/verification.json](../results/submission_closure_v2/verification.json) |
| Data provenance/licence | [DATA_NOTICE.md](DATA_NOTICE.md) |
| AI assistance record | [AI_USE.md](../submission/AI_USE.md) |

The final submission ZIP is [Team87_Lebmar_Technical_Source_v6.zip](../dist/Team87_Lebmar_Technical_Source_v6.zip)
(superseding v4/v5). It keeps the V2 core unchanged and adds the Q1 5x3 screen,
the one-page onboarding flow, the answer sheet, the Q3 note, the final report
source/PDF, the B/E guard-generalization study, the last-mile diagnostics chain
(producers, audits, protocols, freezes, results and notebook), this index and a
package README. Large upstream arrays (e.g. `results/state_budget_mitigation_v1`)
remain regenerable from the bundled data via the frozen core scripts; the full
historical record lives in the git history.

## Superseded or historical material — do not use for final claims

| Item | Status / reason |
| --- | --- |
| [current_model_contract_2026_10_19.json](current_model_contract_2026_10_19.json) | **Pre-final document.** Its Q1 targets/assignments are consistent, but its `current_C_service` says execution Review and its `freeze_and_delivery_boundary` says final 121–180 not read. Those fields predate the V2b final and are superseded by the registry/final evaluation above. |
| [main_model_registry.json](main_model_registry.json) | Older main-model registry; historical evidence only. |
| [RESULTS_REVIEW.md](RESULTS_REVIEW.md) and [VERIFICATION.md](VERIFICATION.md) | Archived V1 S2/R review and cached-result verification. Their old portfolios (e.g. Conservative 15/5/80) and E→Conservative assignment are **not** the current V2b products/assignments. |
| [results/final_locked_evaluation_v1/](../results/final_locked_evaluation_v1/) and [results/final_locked_analysis_v1/](../results/final_locked_analysis_v1/) | Earlier-model final archive; retained, not the V2b final. |
| [results/final_locked_checkpoint_v1/](../results/final_locked_checkpoint_v1/) | Earlier checkpoint; historical. |
| [report/](../report/) and [output/pdf/](../output/pdf/) | Archived V1/S2+R prototype; not regenerated for V2b. Do not quote its 3.66%/43.44% numbers as the current result. |
| 44-word explanation in `current_model_contract` | Older general wording; the final Q2 customer text is the 41-word text in [mitigation_customer_explanation_contract.json](mitigation_customer_explanation_contract.json). |
| `main` branch before the closure update | It was 80 commits behind the current branch. The closure pushes the current tree to `main` so the GitHub homepage matches this index. |

## Number conventions

- Growth: net real annualised log growth; CAGR = exp(G) − 1.
- Large loss: maximum real drawdown from the complete original peak, annual
  endpoints; larger is worse. It does not observe intrayear highs/lows.
- P10/P90: empirical inverse CDF; median: ordinary median.
- Fees: nominal units per initial 100, not an annual percentage.
- Trade fraction: periods with actual destination moved >1e-12 divided by the
  original horizon; the current final V2b value is 382/1,800 = 21.22%, including
  the mandatory initial allocation. Excluding the initial allocation:
  322/1,740 = 18.51%. Ordinary autonomous service is 350/1,800 = 19.44% and is a
  different metric.
- Q1 static B/C/E references are annual product stresses, not the Q2 guarded
  service and not autonomous approvals. Their adverse tails (B 11/60, C 8/60,
  E 5/60 with P10 −1.58%) must remain visible.

## Frozen-evidence boundary

The historical producers, audits, checkpoints and result archives are **not to be
edited**. This closure adds new documents and a new package; it does not rewrite a
frozen manifest. The 2026-10-20 Q1 guard generalization addendum follows the same
rule: it is purely additive (new protocol, freeze, producer, audit and results),
touches no locked artifact, and uses development scenarios 1–120 only. If a
historical checkpoint hashes an older README or file, that checkpoint remains a
true statement about its own timestamp, not about the current tree. The closure verification in
[results/submission_closure_v2/verification.json](../results/submission_closure_v2/verification.json)
records the Q1 table, report draft and v4 package checks. The earlier
[results/submission_closure_v1/verification.json](../results/submission_closure_v1/verification.json)
remains the historical v3-package record.
