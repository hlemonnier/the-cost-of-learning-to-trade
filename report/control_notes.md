# Numerical control: specification and verification protocol

This protocol was written before `src/trade_learning/control.py`. Values are in the synthetic price units of the task. It covers the mandatory reduced reference, not a certification of the continuous-belief optimum.

## Failure catalogue recorded before implementation

1. Confusing pre-decision and post-execution inventory in the risk charge.
2. Forgetting that executed units bear the current return; multiplying expected fills by expected returns would lose selection costs.
3. Charging terminal liquidation twice, dropping it, or allowing terminal liquidation to obey the discretionary one-unit limit.
4. Allowing an individually unsafe submitted side because the opposite side might fill.
5. Using posterior belief before the regime transition, or transitioning the regime before its current execution.
6. Multiplying regime-averaged side likelihoods instead of averaging the products conditional on a shared regime.
7. Treating missing quotes as non-fills or letting unselected depths inform beliefs.
8. Assuming unconditional independent bid/ask fills. Their covariance comes from a shared return innovation.
9. Dropping non-fill outcomes, return evidence, or current state-dependent fill probabilities.
10. Integrating both immediate reward and continuation with inadequate quadrature, hiding avoidable reward error.
11. Discretising beliefs with nearest-neighbour rather than declared linear interpolation, or extrapolating outside [0,1].
12. Failing to preserve kernel mass, finite values, exact terminal values, or invalid-action negative infinities.
13. Incorrect action indexing, especially abstention, market sides, and side exchange under sign symmetry.
14. Letting the full-information benchmark know future regimes or innovations, or using its values as a deployable competitor.
15. Giving the no-information-value ablation a different current reward, estimator, action set, horizon, inventory bound, or runtime belief update.
16. Confusing the no-learning planner's stored value with the realised value of its runtime policy, which does update its beliefs.
17. Claiming exact optimality from empirical convergence, or mixing integration and belief-grid changes in one refinement.
18. Counting tiny action ties as economically meaningful learning, or comparing different evidence states.
19. Returning float32 tables, silently changing the requested horizon, or loading stale cached parameters.
20. Tuning a controller or numerical tolerance after seeing its final evaluation advantage.

## Original stopping rule fixed before numerical results

The primary reference is theta=0.35, kappa=0.02, horizon=30, qmax=2. Belief refinement uses 41, 81, 161, and 321 uniformly spaced points with 81 Gauss--Hermite nodes held fixed. Integration refinement uses 21, 41, 81, and 161 nodes at 321 belief points. All planned levels are reported even when an earlier pair passes. If a axis fails, an additional doubled-minus-one level may be added on that axis; failure is reported explicitly until the condition is met.

For each pair of consecutive levels, compare the horizon-30 value and action scores on the coarser belief grid. The empirical stopping rule requires all of: initial-state value change at (q=0,x=0,b=0.5) <= 0.001; maximum absolute value change over this common state grid <= 0.005; and disagreement in selected actions <= 0.5% among states where the finer-grid best-versus-runner-up action gap exceeds 0.001. Raw disagreements and score gaps are also reported. Integration and grid conditions must pass separately. These tolerances are numerical diagnostics, not confidence intervals or rigorous error bounds. No Monte Carlo is used in this reference calculation.

## Recursion and conventions

After subtracting current marked wealth C+qP, state is (periods remaining,q,x,b). Cash and the absolute midpoint disappear because all transaction prices are midpoint-relative, return and execution laws are translation invariant, and terminal liquidation is linear in the midpoint. The terminal value is -|q|(h+cT).

For passive fills f, let dq=f_bid-f_ask and G=sum_s (h+k_s*DEPTH-cP)f_s. A market action has dq=s and G=-(h+cT). The reward is G+(q+dq)R-lambda*q^2. Its expectation is integrated analytically: each submitted passive side contributes p_s(d_s-cP+s*MU*x)-SIGMA*(2b-1)*theta*phi(Phi^-1(p_s)), in addition to q*MU*x-lambda*q^2. A market contribution is -h-cT+s*MU*x.

Active continuation integrates all submitted fill subsets and the realised return, applies the exact regime-mixture likelihood, predicts the next regime, linearly interpolates the value at the resulting belief, and averages the independent next signal. Gauss--Hermite quadrature is used only for continuation. The noinfo planner replaces the feedback posterior with the current prior before predicting the next regime, retaining identical rewards, fill probabilities, and inventory dynamics. Runtime policies using those tables still filter every realised observation.

Under sign reversal H -> -H, substitute z -> -z in the unconditional fill-vector law: it is unchanged. Thus fills alone (including resulting inventory) reveal no H, and the noinfo planner can be interpreted as planning without return--fill association. For a convex continuation in belief, feedback has nonnegative one-step value conditional on every fill subset. The runtime noinfo policy nevertheless differs from a permanently non-learning policy because its next action uses updated evidence.

The full-information benchmark observes the current regime only, has two exact regime states, and averages the next regime using kappa. The independent controller sets theta=0 in its planning model. The taker controller restricts the action set to abstention and market buys/sells. All modes keep the same liquidation, risk charge, and inventory safety rules. Ties are broken by the lowest action ID.

## Evidence to be produced

`verification/verify_control.py` was authored before the solver and independently checks one-step accounting using scalar adaptive quadrature, theta=0 equality, symmetry, invalid actions, kernel mass, convexity, full-information dominance, refinements, serialization, and a same-state feedback-suppression comparison using one fixed active continuation function. It saves its checks, settings, runtimes and refinement records in `outputs/reference/verification.json`. Large practical tables require separately disclosed refinement evidence; reference convergence must not be silently transferred to them.

## Practical-horizon refinement protocol (fixed before those calculations)

After the reduced-reference check passed, the primary agent requested independent approximation evidence at T=300 and qmax=5 for every one of the nine public parameter pairs, before final policy evaluation. Both active and noinfo planners are assessed. The declared grid sequence is 81, 161, 321 belief points at 81 quadrature nodes, followed by 81 versus 161 quadrature nodes at the final belief resolution. An axis failing the rule receives one further level: 641 beliefs or 321 quadrature nodes. Every calculated comparison is retained, including failures. No final-environment sample enters this assessment.

The practical tolerances, selected before running these calculations, are initial-state value change <=0.01, common-state maximum value change <=0.03, and <=0.5% action disagreement among states whose finer action gap exceeds 0.005. This comparison concerns the full set of inventories, signals and shared grid beliefs at 300 periods remaining. Both axes must pass separately. These are empirical approximation diagnostics, not certified error bounds. The practical active and noinfo controllers will use the same final resolution, selected at least as fine as the maximum level needed across the declared family. A failed condition at the maximum examined level must be reported, not suppressed. Reproduce with `PYTHONPATH=src .venv/bin/python verification/verify_control.py --practical --output outputs/reference/practical_refinement.json`.

## Recorded numerical findings

The reduced reference passes every prewritten verification check. At 321 beliefs and 161 quadrature nodes, the initial active value is 0.9895587848, the no-learning planner value is 0.7300294617, and the full-information value is 1.4719921721. The no-learning planner value must **not** be presented as the expected realised value of the filtered noinfo runtime policy.

| Reduced-reference comparison | Initial value change | Maximum common-state change | Robust action disagreements |
|---|---:|---:|---:|
| Beliefs 81 to 161; quadrature 81 | 0.000320543 | 0.000691179 | 0 |
| Beliefs 161 to 321; quadrature 81 | 0.000096013 | 0.000206334 | 0 |
| Quadrature 41 to 81; beliefs 321 | 0.000053842 | 0.000087437 | 0 |
| Quadrature 81 to 161; beliefs 321 | 0.000005123 | 0.000053614 | 0 |

Here robust means finer action gap above 0.001. Raw action disagreements are retained in the JSON; most lie on exact symmetry ties or very small margins. Independent one-period cash-flow integration differs from the solver by at most 8.7e-17. Sign symmetry, theta-zero equality, belief convexity, admissibility, terminal liquidation, serialization, taker restrictions, full-information dominance and kernel mass checks all pass.

There are states with an explicit current economic sacrifice for information. At 30 periods remaining, q=0, x=0, b=0.75, the active action submits both deep quotes, with expected immediate reward -0.00829323. Holding the very same active continuation fixed but suppressing this step's feedback selects abstention, with immediate reward zero. Active action value exceeds abstention by 0.00349678. This is a worked example selected after solving, not a separate statistical test. Across the horizon-30 reference grid, 778/4,815 choices change when suppressing current feedback, with 594 changes costing more than 0.001 in active Q; the largest cost is 0.0158799.

**Historical original replay (before the numerical correction).** The primary agent evaluated the reduced policies on 20,000 paired environment episodes (seed 260924301). Active mean objective 0.98772665 (MC SE 0.01600858) is 0.114 standard errors from its planning value. Full-information mean 1.45178509 (SE 0.01560408) is 1.295 standard errors from its planning value. Filtered runtime noinfo achieves 0.95649339, illustrating why its planning value cannot be used as its measured performance. Active minus runtime noinfo is 0.03123326 with paired 95% interval [0.01781759,0.04464893]; active minus myopic is 0.15467122. See the [original reduced record](https://github.com/hlemonnier/the-cost-of-learning-to-trade/blob/audit/original-c9d1d4c/outputs/reference/rollout/verification.json) and paired CSV at that tag; the current path contains corrected results. These are reduced known-parameter findings, separate from the finite-pilot practical deployment comparison.

The first practical-horizon sweep took 163.74 seconds and preserved all 18 active/noinfo model-family results. Seventeen families pass both predeclared final-axis checks. Active theta=0.35,kappa=0.002 needs 641 beliefs: its 321-to-641 initial change is 0.00517215. Active theta=0.65,kappa=0.002 narrowly misses the initial-value condition at that maximum examined level: 321-to-641 change 0.01073930 exceeds 0.01, while maximum state change 0.01531266 passes and there are zero robust action disagreements. Its maximum finer-Q loss for a changed coarse action is only 0.00001389. This residual value approximation limit is recorded explicitly in `outputs/reference/practical_refinement.json`; the experiment exits nonzero until its declared criteria are met or a further documented refinement resolves them. Across all active families, the largest final quadrature initial-value change is 0.00026360 and largest state change 0.00033843. Noinfo quadrature differences are below 9e-13. The maximum resolution examined in that sweep is 641 beliefs and 161 nodes.

## Additional refinement declared after the first sweep and before new results

The primary agent requested a further level rather than accepting the remaining failed value condition. For the public support point theta=0.65,kappa=0.002, compare 641 versus 1,281 beliefs holding 81 nodes fixed, then 81 versus 161 and 161 versus 321 nodes holding 1,281 beliefs fixed. Preserve the original tolerance and every previous failure. The selected practical resolution is intended to be 641 beliefs/161 nodes throughout the public family, with a 1,281-belief override for this support point in **both** active and noinfo modes if the additional checks pass. This override is attached to a candidate model supplied to every learner, never selected from a hidden test-environment label. Also check the noinfo override with the same belief refinement and 81-versus-161 integration refinement. Reproduce the extension, after the first sweep, with `PYTHONPATH=src .venv/bin/python verification/verify_control.py --practical-extra --output outputs/reference/practical_refinement.json`.

The extension passed every unchanged criterion in 95.48 seconds. Active 641-to-1,281 belief refinement changes initial value by 0.00142454 and common-state maximum by 0.00193946, with zero robust action disagreements. At 1,281 beliefs, 81-to-161 quadrature changes initial value by 0.00012763; 161-to-321 changes it by 0.00002505 and the state maximum by 0.00014527, again with zero robust action disagreements. The matching noinfo refinement passes. Thus the final practical settings are 641 beliefs/161 nodes, with 1,281 beliefs/161 nodes for the public candidate (0.65,0.002) in both modes. This support-point override supersedes the earlier intention to use one global belief-grid size, while preserving identical numerical settings between each matched active/noinfo pair. All 18 families now meet their final-axis criteria; original failures remain visible in the evidence. The criterion is an empirical refinement rule, and this result still does not certify exact optimality or a worst-case error bound.

The stress reference also needs precise wording: when nominal full-information tables are used in a misspecified stress environment, they define a privileged current-regime policy with nominal planning dynamics, **not** an oracle optimum for that alternative process. Fixed spells have mean length 50, compared with \(1/\kappa\) nominally (500 at \(\kappa=.002\)); only \(\kappa=.02\) matches that mean. Duration shape and trading opportunities also differ. Neither the nominal-to-stress objective change nor the gap to this frozen privileged policy isolates the cost of duration shape or avoidable filtering inefficiency.

## Corrective all-horizon verification

The historical numerical outcomes above describe the original study and remain preserved at `audit/original-c9d1d4c`. The post-review solver applies the absolute 1e-12 lowest-admissible-ID convention in both Bellman selection and runtime execution. Its refinement evidence checks **every remaining horizon**, not just the initial T-step state: initial-state drift, common-state drift and robust disagreement must meet the same thresholds at each horizon. Numerical ties and other near ties are counted separately.

The repeated 18-family sweep and prescribed extension pass. The same public (.65,.002) case requires 1281 beliefs; all other deployed matched tables retain 641, and all use 161 integration nodes. Full per-horizon records and earlier failed steps are in `outputs/reference/practical_refinement.json`. The compact CSV index points to those records. The corrected [reduced cold replay](../outputs/reference/rollout/verification.json) gives active mean 0.9866010082, runtime noinfo 0.9564933923 and myopic 0.8363515144. The paired active-minus-noinfo estimate is 0.0301076160 with 95% interval [0.0167131963, 0.0435020356]. These use the original episode seeds and are not additional independent evidence. The [secondary numerical-stack comparison](../outputs/verification/audit_corrections/cross_stack_comparison.json) records exact equality of all 20,000 episode objectives for each of the four policies on the two tested local stacks, without a universal portability claim.
