# Fixed-state update-dependence analysis

The fixed search screened 137,430 candidate states and directly examined 512 disagreements.
It found **223 refinement-stable qualifying witnesses** under the prespecified .001 action-margin rule.

These are numerical action diagnostics. The intervention freezes marginal model weights while retaining conditional regime updates; it is not a coherent deletion of only model information.

| Candidate source | Screened | BA/WQ disagreements | Distinct disagreements | Direct checks |
|---|---:|---:|---:|---:|
| uniform_exploration_reached | 60000 | 3172 | 3172 | 256 |
| arbitrary_numerical_probe | 77430 | 3393 | 3376 | 256 |

Direct checks examine the fixed top 256 disagreements per source after table screening; null does not exclude other states.

Cross-family stability: checked_on_supplied_families. No reported action margin is an interval-certified true Bellman gap.

## Retained case 48408

Status: **refinement_stable_witness**. Recorded public prefix under uniform safe exploration; no scored-policy prevalence claim.

State: n=12, q=2, x=1, joint p=[0.025081359505147685, 0.4617276104866007, 0.3156190607925384, 0.19757196921571324].

| Quantity | BA choice | Weighted-Q choice |
|---|---:|---:|
| Action ID | 2 | 0 |
| Immediate expected reward | 0.04608809104 | 0.052 |
| One-period score including liquidation | 0.0004587800481 | -0.002 |
| Model information (nats) | 0.01153301425 | 0 |
| Conditional regime information (nats) | 0.02143066178 | 0 |

The same-BA-continuation frozen-weight action is 0. Complete legal-action scores and every failed qualification are retained in direct_candidates.jsonl and direct_scores.csv.

## Numerical scope

For the selected family, the broad theorem allowance at T=30 is 9.46506 objective units on the one-sided BA comparison, 15.1157 on grid nodes, and 15.5038 for a final off-grid value query.
These are ordinary floating evaluations of conservative mathematical formulas, with no solver-rounding or interval-arithmetic certificate. They do not establish an economic optimality gap of .002. Observed cross-cache score changes are a separate empirical diagnostic.

Cold/equal-conditional-belief and terminal suppression identities: passed.

All reached states come from separate namespace-410 public exploration, with no final-evaluation PnL used in selection. Parent campaign results are needed to assess the population economic contrast.
