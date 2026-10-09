# Prespecified Bayes-adaptive extension

Author: Hugo Lemonnier, with disclosed Codex/Astra/Sol assistance. Protocol prepared
25 September 2026 (Europe/Paris), before extension economic evaluation.

## Scope

This separate synthetic study preserves the core market and accounting with its
own streams, outputs and model-uncertainty question. Its observations are not
pooled with the original finite-pilot experiment.

## Question and primary hypothesis

How costly is acting on posterior-weighted known-model optimal Q values when the
model will not actually be revealed? Does joint model-and-regime planning improve
economic decisions when persistence is genuinely uncertain?

The primary contrast is **Bayes-adaptive approximation minus posterior-weighted
known-model Q, with zero pilot episodes**, on the equal mixture of the two
prespecified models below. The economic threshold is **0.002 objective units per
30-period episode**, the fee on one market execution, excluding the spread.
Meaningful improvement requires the primary paired empirical 95% interval's lower
endpoint to exceed 0.002, together with resolved numerical checks. A null, adverse
or numerically unresolved finding is retained and reported, never tuned away.
The question is not conditional on the new policy winning.

## Fixed market, information and policies

Use the unchanged model/accounting: theta=0.35, kappa in {0.002,0.10}, T=30,
inventory cap 2, all original fees, signal transitions, returns and admissibility.
The equal initial model prior is known to every feasible policy. Hidden regimes
start independently at probability one half in each pilot/evaluation episode.
The true model is fixed within a pilot/evaluation replicate; labels, hidden paths
and unsubmitted fills stay evaluator-only.

All feasible policies maintain the exact public-data joint posterior over four
states (M1,-1),(M1,+1),(M2,-1),(M2,+1). Policy differences concern planning, not
the observations or the runtime posterior update:

1. `bayes`: full reduced-horizon numerical joint-belief Bellman recursion.
2. `weighted_q`: posterior average of separately solved known-model hidden-regime
   Q values, with the same stable lowest-ID absolute-1e-12 tie rule.
3. `no_feedback`: same state/action/reward model, but future planning uses only
   regime prediction and frozen model weights; runtime still filters every public
   observation. This removes future feedback valuation, not current inference.
4. `frozen_model`: future planning updates both conditional regime beliefs while
   holding model weights fixed; runtime uses the correct full posterior. This is
   a planning intervention, not a coherent Bayesian posterior or a new bound.
5. `known_parameter`: evaluator-only reference given the true model label, with
   hidden regime still filtered from selected public observations. It never sees
   current/future hidden regimes or future shocks.

When conditional regime beliefs are identical, one observation has equal model
likelihood under both kappas. In particular cold-start first-step model information
is exactly zero. The experiment must verify this temporal-identification feature.
Freezing model weights also affects the marginal regime prediction once M and H
are correlated; a matched ablation alone cannot establish a pure causal separation
between parameter and regime information. Entropy reduction alone is not benefit.

## Model-revelation analysis

Prove exact continuous-model statements, separately from numerical evidence:
known-model information at entry gives U^0; revealing M after d decisions gives
U^d; U^0 >= U^1 >= U^2 >= ... >= U^T = V_BA >= V_pi. U^1 equals the maximum
posterior-weighted exact known-model Q. Revealing M does not reveal H.
Compute numerical U^0,U^1,U^2,U^3 and full-horizon BA estimates at matched states.
The relaxed objective is not the realized value of its greedy policy. State
numerical-error assumptions and any proved broad error envelopes explicitly;
refinement or pointwise quadrature agreement alone is not a certified bound.

## Pilots, pairing and fixed evaluation

Use 30 independent pilot/evaluation replicates per true model, with 1,000 fresh
evaluation episodes each: 60,000 paired market trajectories. Generate five
independent 30-step uniform-admissible-action pilot episodes per replicate.
The predeclared ladder uses zero, the first one, and all five pilot episodes.
The one/five budgets are nested; all five policies receive the same public pilot
at each replicate/budget. Reset H to its initial half prior at each evaluation
episode, retaining only the pilot model posterior. No evaluation outcome fits a
shared pilot or another episode's policy state.

Reuse evaluation tapes across policies and pilot budgets. The result is 900,000
policy-episode records, not 900,000 independent markets. Compare policies within
the same (true_model,replicate,episode,budget) key. Pilot ladder comparisons are
paired sensitivity analyses, not independent data additions.

Root seed 260925508 is new and separate from all original roots. Derive each
stream from SeedSequence([root,namespace,model_index,replicate]); namespaces are
110=pilot market, 111=pilot action, 210=evaluation market, 310=numerical probes,
410=explanatory probes, 510=cold repeat smoke. Never replace a seed after seeing
an economic result. The full protocol JSON records every count and identifier.

## Inference and required outputs

For each contrast/metric, average paired episode differences within each
independent replicate, then take an equal-weight mean of the two true-model
stratum means. Estimate variance as one quarter of the sum of stratum
replicate-mean sample variances divided by 30, with Welch degrees of freedom.
Use empirical Student/Welch intervals; no distribution-free or future-market
guarantee is claimed. Reject incomplete budgets, missing policies/strata,
duplicate/broken pairs, nonfinite values and insufficient replicates.

The single primary objective comparison has an ordinary two-sided 95% interval.
The supplementary family comprises four comparisons (bayes-weighted_q,
bayes-no_feedback, bayes-frozen_model, known_parameter-bayes), three pilot budgets,
and objective/net-PnL metrics: **24 Bonferroni-adjusted two-sided intervals**.
Also show stratum results and raw policy means, with their scope labelled.

Retain complete episode objectives/PnL/ledger decomposition/inventory/action
counts, all actions, public pilots, reproducible market tapes or their exact
generation metadata, hashes, selected full posterior/action traces, posterior
entropy/true-model probability (evaluator diagnostics only), planning/runtime
cost and memory observations. Plot economic performance versus pilot budget,
feasible policy performance separately from numerical relaxed objectives.

## Numerical validation and acceptance

Write failure modes before implementing each isolated component. Start with
terminal/one-step and short-horizon direct integration cases, exact posterior
enumeration, impossible/missing feedback, symmetry, degenerate model support,
known-parameter reduction and same-model equivalence. Check transition mass,
reward/ledger chronology, action admissibility and stable ties.

Use a predeclared product-belief refinement ladder and independently refined
quadrature, with coordinates w,b1,b2. Planned joint grid levels are (9,17,17),
(17,33,33), (25,49,49), (33,65,65). Quadrature candidates are 15,25,41,61,121,241;
known-model scalar references use at least 321 beliefs and 161 nodes, compared
with independently refined settings. Solver feasibility measurements may change
implementation/storage without changing this statistical design.

Before final evaluation select the first level satisfying both independently
varied refinement axes on fixed predeclared probes, all horizons 1..30: max
initial-state value change <=0.001, max probe value/finite-Q change <=0.005,
and disagreement fraction <=0.005 where the finer best-to-second gap exceeds
0.001. Initial probes use q=x=0,b1=b2=.5,w in {0,.25,.5,.75,1}; remaining probes
include boundaries, symmetry and 2,048 fixed random joint states (namespace310).
Evaluate scores at identical off-grid states, not mismatched cell indices.
These are numerical resolution criteria, not uniform mathematical error bounds.

If the declared finite ladder cannot resolve the full numerical reference,
retain every failed comparison, label the full reference unresolved, and report
proved analytical bounds plus checked limited-lookahead results as the specified
fallback. Do not loosen tolerances based on economic outcomes. Do not call a
controller exact/Bayes-optimal solely because it was evaluated. Any change to
the protocol must be recorded before accessing new final outcomes.

## Controlled decision explanation and final gate

Search fixed grid/probe states for different BA/weighted-Q actions, then compare
same-state scores with identical BA continuation while suppressing only the
model-weight update. Keep rewards, outcome law, inventory, conditional-regime
updates and continuation fixed. Record both chosen actions and the complete Q
vectors, robust gaps, initial and posterior weights, and refinement stability.
Show a compact economic explanation figure if such a robust witness exists;
otherwise report the negative search and strongest checked case without claiming
successful identification. Do not select a favorable realized PnL trajectory.

Independent checks must replay the ledger from saved tapes/actions, enumerate
the joint filter, verify pilot identities and recalculate all contrasts without
production statistical imports. Re-extract the final review ZIP, verify every
hash, run fresh cold checks and verify the recorded baseline data.
The primary agent reviews each delegated deliverable and every new PDF page.
Keep the original eight-page main report intact, add a clearly labelled extension
appendix, source, results, limitations and reproduction instructions. No email or
external result is established by these checks.
