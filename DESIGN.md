# Research design and integration contract

Recorded before implementation and final evaluation, 2026-09-24. The selected scope is the full mandatory single-asset synthetic study; optional extensions are not completion requirements. All report deliverables will be in English.

## Shared implementation interface

Python package `src/trade_learning`; scripts run with `PYTHONPATH=src`. Float64 computation. Actions 0..8 enumerate `(bid_depth,ask_depth)` from `(-1,0,1)^2` in lexicographic order, where -1 means absent. Action 9 buys at market; action 10 sells at market. Action 0 is abstention. Signal indices 0,1,2 correspond to x=-1,0,1. Hidden-regime index 0 means -1, index 1 means +1. Observed fill arrays contain -1 for missing, 0 for an observed non-fill, 1 for a fill.

`model.py` exposes `K`, `ACTIONS` (11x2, market rows sentinel -2), `H=.025`, `DEPTH=.025`, `CP=.001`, `CT=.002`, `LAMBDA=.002`, `MU=.03`, `SIGMA=.30`, `admissible(q,qmax)` returning (...,11) bool, and `fill_probability(x,side,depth)` with numpy broadcasting.

Control interface: `solve_control(theta,kappa,horizon,qmax,belief_points,quadrature_points,mode)` returns an object exposing `q_values` with shape `(horizon+1,2*qmax+1,3,belief_points,11)` indexed by periods remaining (row 0 may be unused), `values` with shape `(horizon+1,2*qmax+1,3,belief_points)`, and `beliefs`. Modes `active`, `noinfo`, `independent`, `taker`, `full`. In full mode belief grid may be the two exact states [0,1]; document this. Invalid action Q values are -inf. Tie breaking is deterministic lowest action ID. Pre-decision inventory penalty and terminal liquidation exactly follow the statement. Save/load helpers and compact metadata are welcome; no dependency on evaluator metadata.

Environment/inference interface is to be documented by its owner before parent integration; use batched arrays and allow pre-generated evaluator-only exogenous streams. Policies must receive only copies of public observations, never simulator objects, seeds, parameter labels, latent regimes or counterfactual fills. Evaluator labels remain available only to diagnostic/oracle code.

## Statistical design fixed in advance

Nine nominal parameter pairs, 10 independent pilots per pair, 100 episodes of 300 periods per pilot. Final evaluation uses 100 fresh episodes per pilot and policy, separately in nominal and each prescribed stress (1,000 per pair/policy/variant). Pilots are shared across data-dependent policies within each replicate. Exogenous draws are paired across policies; public policy randomisation is independent. No fitting/tuning on final streams and no cumulative cross-test-episode updates.

Before any full evaluation, source review fixed distinct roots: smoke/debug seed 260924136, final seed 260924138, and restricted-reference rollout seed 260924301. An earlier smoke run had consumed root 260924137, so that root is retired from final evaluation. Changing both the debug profile and the final root prevents any reuse of previously inspected random prefixes.

Candidate model support is the public 3x3 parameter grid with a uniform prior, declared to every learner. Fit the joint grid likelihood using all 100 pilot episodes; no data-driven hyperparameter search is planned. Practical control may combine per-model regime-control Q tables under the parameter posterior; this is an approximation to joint Bayes-adaptive control, not an optimum. Its matched no-information-value ablation uses the same posterior, filter, action set, horizon and inventory penalty, changing only future-feedback valuation. Parameter weights and regime beliefs can update within each evaluation episode, starting afresh from the pilot posterior.

Numerical settings frozen after deterministic refinement and before final sampling: common 641 belief nodes and 161 Gaussian-Hermite nodes; the public support point (theta=.65,kappa=.002) uses 1281 belief nodes for both active and noinfo tables. The override applies identically regardless of the unknown environment. Reduced reference uses 321 belief/161 integration nodes. The full-horizon refinement criteria (initial value drift <=.01, common-state maximum drift <=.03, robust action disagreements <=.5% at gap>.005) pass; earlier failed coarser checks are retained in the artifact. These remain empirical approximation checks, not error certificates.

Primary deployable comparison is active versus matched noinfo, selected before evaluation. Deployment population is the uniform mixture of the nine nominal pairs with a fresh independent 100-episode pilot per environment. Acceptance requires a one-sided 95% lower confidence bound above 0.05 objective units/episode for the primary difference, and positive net PnL. Report cluster-based empirical uncertainty using independently generated pilots, with within-pilot episode variation separately. Per-environment or multi-policy claims use multiplicity-adjusted intervals. This is an empirical confidence assessment, not a distribution-free or regime-change guarantee. Stress evidence is diagnostic, never used for nominal policy selection.

## Failure modes to verify before relying on results

Incorrect cash-flow signs; double liquidation/fees; post- rather than pre-decision penalty; accepting unsafe simultaneous-fill actions; treating missing quotes as failed quotes; exposing current/future innovations or regime/parameter labels; independent rather than shared depth noise; unconditional side-independence in likelihood; dropping non-fill evidence; failing to transition the posterior; underflow; using final data to tune; hidden training through cross-episode updates; unmatched estimator/constraints in the ablation; mismatched references; premature quadrature/grid convergence; counting ticks as independent; failure to separate pilot variability from Monte Carlo error; changed marginals in stresses; inconsistent report and saved data.

## Acceptance evidence

End-to-end verification must save a machine-readable artifact: trajectory/accounting and information-contract checks; conditional marginal and depth tests; controlled filter diagnostics; independently derived wealth reconciliation; lower-bound derivation and numerical illustration; reference refinement and action stability; matched policy/ablation results; complete nominal/stress episode counts and hierarchical uncertainty; a clean-process small reproduction. Every numerical claim in the report must point to saved results. A requirements-to-evidence matrix is an audit record, not a local task backlog.

## Post-review correction contract (2026-09-24)

The original design and results remain at tag `audit/original-c9d1d4c`. The correction retains scientific support, budgets, baseline, economic threshold and all random roots. It replays already observed tapes; it is not an independent confirmation or an enlargement of the inferential sample. The fixed absolute tie tolerance is 1e-12 and the lowest admissible ID wins; Bellman V stores the chosen Q. Cache arrays are authenticated against source/build/settings/content/structure. Statistical and configuration helpers reject unsupported or incomplete evidence. Detailed failure cases were recorded before the corrective code in `verification/numerical_failure_modes.md`, `stats_config_failure_modes.md` and `campaign_failure_modes.md`.

Practical convergence now checks every positive remaining horizon, requiring the original tolerances at each horizon and reporting near ties separately. The repeated first sweep retains the original hard support point; its prescribed extension passes without changing the deployed 641/161 resolution or public 1281-belief override. The corrected production source is frozen at `0620e68`; source and configuration are rechecked at run completion.
