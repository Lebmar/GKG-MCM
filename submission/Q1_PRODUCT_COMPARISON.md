# Q1 — 5×3 profile × product comparison

Static annual target-restoration products with exact 0.25% fees and each
profile's own horizon and loss tolerance. **Development 1–120** is the
selection screen; **final 121–180** is a post-freeze descriptive check. The
frozen assignments were not changed by this table.

## Development 1–120 (selection screen)

| Profile | Product | Median CAGR | P10 CAGR | Median MDD | P90 MDD | Breaches | Frozen assignment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A (H=5, L=10%, high) | Conservative | 3.17% | -1.59% | 5.56% | 14.90% | 34/120 | Review |
| A (H=5, L=10%, high) | Balanced | 4.36% | -1.85% | 8.59% | 20.53% | 49/120 | Review |
| A (H=5, L=10%, high) | Growth | 5.10% | -2.73% | 11.94% | 27.46% | 66/120 | Review |
| B (H=15, L=25%, medium) | Conservative **selected** | 3.04% | 0.10% | 13.25% | 24.22% | 9/120 | Conservative |
| B (H=15, L=25%, medium) | Balanced | 4.08% | 0.10% | 17.35% | 32.65% | 27/120 | Conservative |
| B (H=15, L=25%, medium) | Growth | 5.26% | 0.02% | 23.18% | 42.07% | 49/120 | Conservative |
| C (H=30, L=40%, low) | Conservative | 2.97% | 1.02% | 17.37% | 31.74% | 1/120 | Balanced |
| C (H=30, L=40%, low) | Balanced **selected** | 4.03% | 1.44% | 22.60% | 38.04% | 9/120 | Balanced |
| C (H=30, L=40%, low) | Growth | 5.12% | 1.48% | 32.26% | 48.84% | 35/120 | Balanced |
| D (H=25, L=15%, medium) | Conservative | 3.19% | 0.90% | 16.71% | 29.37% | 67/120 | Review |
| D (H=25, L=15%, medium) | Balanced | 4.31% | 1.34% | 20.90% | 35.91% | 92/120 | Review |
| D (H=25, L=15%, medium) | Growth | 5.33% | 1.41% | 29.18% | 47.90% | 104/120 | Review |
| E (H=6, L=35%, high) | Conservative | 2.99% | -1.74% | 8.08% | 18.35% | 0/120 | Growth |
| E (H=6, L=35%, high) | Balanced | 3.75% | -2.33% | 10.29% | 23.11% | 2/120 | Growth |
| E (H=6, L=35%, high) | Growth **selected** | 5.00% | -3.69% | 13.89% | 31.18% | 6/120 | Growth |

All selected B/C/E products pass the declared empirical 10% loss-event screen
on development data: B 9/120, C 9/120, E 6/120. A and D have no
risk-feasible product (34–104/120 and 67–104/120 breaches respectively), so
they remain Review. The final selected-product table is the evaluation table
in [TASK_ANSWER_SHEET.md](TASK_ANSWER_SHEET.md); it retains the adverse final
drift (B 11/60, C 8/60, E 5/60, E P10 −1.58%).

## Final 121–180 (descriptive, not used to assign)

| Profile | Product | Median CAGR | P10 CAGR | Median MDD | P90 MDD | Breaches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A (H=5, L=10%, high) | Conservative | 2.61% | -1.57% | 5.26% | 14.85% | 16/60 |
| A (H=5, L=10%, high) | Balanced | 3.73% | -2.36% | 6.11% | 17.18% | 25/60 |
| A (H=5, L=10%, high) | Growth | 4.70% | -3.40% | 8.74% | 23.08% | 28/60 |
| B (H=15, L=25%, medium) | Conservative **selected** | 3.21% | 0.35% | 14.09% | 29.53% | 11/60 |
| B (H=15, L=25%, medium) | Balanced | 4.07% | 0.44% | 16.78% | 37.01% | 19/60 |
| B (H=15, L=25%, medium) | Growth | 5.27% | 0.21% | 22.06% | 47.67% | 27/60 |
| C (H=30, L=40%, low) | Conservative | 2.91% | 1.24% | 19.61% | 33.46% | 4/60 |
| C (H=30, L=40%, low) | Balanced **selected** | 3.56% | 1.48% | 27.49% | 43.54% | 8/60 |
| C (H=30, L=40%, low) | Growth | 4.45% | 1.68% | 36.20% | 54.64% | 18/60 |
| D (H=25, L=15%, medium) | Conservative | 3.24% | 0.74% | 17.83% | 31.95% | 38/60 |
| D (H=25, L=15%, medium) | Balanced | 4.07% | 1.39% | 22.75% | 40.00% | 48/60 |
| D (H=25, L=15%, medium) | Growth | 5.23% | 1.65% | 31.66% | 50.85% | 52/60 |
| E (H=6, L=35%, high) | Conservative | 2.54% | -0.38% | 7.12% | 19.20% | 0/60 |
| E (H=6, L=35%, high) | Balanced | 3.50% | -0.20% | 10.32% | 21.95% | 0/60 |
| E (H=6, L=35%, high) | Growth **selected** | 4.51% | -1.58% | 14.17% | 31.34% | 5/60 |

The final 5×3 table is a descriptive check only. In particular, B/C static
final frequencies exceed the 10% training reference and E Growth has negative
P10 growth; these are disclosed, not repaired by changing assignments.

## Validation

The selected B/C/E rows reproduce the published development and final Q1 tables; 30 checks, maximum absolute error 1.67e-16.

Source: [development CSV](../results/q1_product_comparison_v1/development_5x3.csv), [final CSV](../results/q1_product_comparison_v1/final_5x3.csv), [manifest](../results/q1_product_comparison_v1/manifest.json).
