# Q1 guard generalization — does the Q2 safeguard extend to B and E?

Team **Lebmar (87)** — Chengxuan YUAN. Companion note to
[TASK_ANSWER_SHEET.md](TASK_ANSWER_SHEET.md). Development scenarios 1–120 only
(resubstitution diagnostic); no final data parsed; no locked artifact changed.
Protocol: [q1_guard_generalization_protocol.json](../docs/q1_guard_generalization_protocol.json);
audit: [independent_audit.json](../results/q1_guard_generalization_v1/independent_audit.json)
(10,993 checks passed, independently recomputed from saved arrays).

## Question

Q1 assigns B → Conservative and E → Growth as **static product references**,
while the Q2 protective guard was only evaluated for C. Does the same
unchanged guard extend to B and E?

## Answer: yes — as a stricter funding gate, not as an execution policy

Applying the locked V2b rule (`release_0.5_uniform`, calendar support, 0.25%
fee) unchanged to each profile's own target, horizon and loss limit:

| Object | Target | H | L | One-year support floor | Guard initial decision |
| --- | --- | ---: | ---: | ---: | --- |
| B guard | Conservative 25/30/45 | 15 | 25% | 87.5 | **Pre-investment Review** (no feasible supported allocation) |
| E guard | Growth 65/5/30 | 6 | 35% | 82.5 | **Pre-investment Review** (no feasible supported allocation) |

For contrast, C's Balanced 45/5/50 faces floor 80.0 and passes the identical
screen (funded 60/60 in the locked final evaluation). The floor is the
declared normative convention (1 − L/2) × current real wealth; the selectivity
comes from the interaction of the L-derived floor and target riskiness, never
from a per-profile adjustment.

## Static references on the same 120 development paths (cross-check)

These reproduce the previously audited development 5×3 screen exactly:

| Profile product | Breaches/120 | Median CAGR | P10 CAGR | Median MDD | P90 MDD |
| --- | ---: | ---: | ---: | ---: | ---: |
| B Conservative annual | 9 | 3.04% | 0.10% | 13.25% | 24.22% |
| B Conservative pending-hold | 9 | 3.04% | 0.10% | 13.25% | 24.22% |
| B exact-thirds annual | 20 | 3.61% | 0.05% | 15.19% | 27.63% |
| E Growth annual | 6 | 5.00% | −3.69% | 13.89% | 31.18% |
| E Growth pending-hold | 6 | 5.00% | −3.79% | 13.89% | 31.18% |
| E exact-thirds annual | 0 | 3.52% | −2.32% | 8.90% | 21.93% |

## How to use this in the report

- The safeguard is **one coherent family**: a suitability screen at onboarding
  (A/D Review), then the same unchanged execution gate deciding whether an
  assigned product may be auto-funded. Only C passes both gates; B and E would
  be routed to human review **before** investment under the guarded service.
- This makes the "static B/C references exceed the 10% service convention"
  disclosure precise: Q1 assignments are suitability outcomes, never automatic
  execution approvals — and the guard confirms it by refusing unattended
  funding for B/E.
- **Do not** claim the guard reduces B/E breach rates in execution: no funded
  guard path exists for B/E. The finding concerns the gate's conservatism.
- E's equal-thirds reference shows 0/120 breaches but lower growth (median
  3.52% vs 5.00%) with negative P10 — a hypothetical comparator, not the
  assigned product.

## Limitations (must stay attached)

1. Resubstitution on development data; the guard was selected for C on the
   same 120 paths, so the gate's behaviour here is mechanically informed by
   development evidence.
2. The one-year support floor is a declared normative convention, not a
   calibrated loss probability.
3. A stricter funding gate increases human-review workload; no adviser
   capacity or response model is estimated.
4. No final-sample B/E guard counterpart exists or may be created without
   breaking the disclosed nonblind protocol.
