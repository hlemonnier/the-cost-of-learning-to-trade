# Joint-belief solver: pre-implementation design and failure catalogue

Prepared before `solver.py` implementation. This document covers numerical
control only; the shared protocol owns evaluation seeds, statistical decisions,
and the final refinement acceptance criteria. No economic outcomes were used to
choose this design.

## State and recursion

The state is inventory, current public signal, remaining horizon, and the four
joint masses in order `(M1,-1), (M1,+1), (M2,-1), (M2,+1)`. Internally a regular
product grid uses `w=P(M1)`, `b1=P(H=+1|M1)`, and `b2=P(H=+1|M2)`. This is a
coordinate system for a joint belief, not separate optimization by model.
At zero model weight its conditional regime probability is immaterial.

Both models share theta. A selected fill/non-fill likelihood is computed
conditional on the observed Gaussian return innovation and current hidden sign.
The joint posterior first conditions on that likelihood, then each model's
own transition predicts the next hidden sign. The public signal transitions
independently according to the existing matrix. Immediate expected reward is
analytic, with the pre-action inventory penalty and exact terminal liquidation.

The Bayes-adaptive recursion maximizes one common action over the expectation of
the preceding horizon's joint-belief value. Revelation at delay zero uses the
posterior-weighted known-model hidden-regime values. Delays one, two and three
apply the common-action joint Bellman recursion before that revelation, with
one less delay in the continuation. The one-step revelation Q is the weighted
known-model Q, not the achieved value of the greedy weighted-Q policy.

All policy execution uses the same exact public-data posterior maintained by
the evaluation harness. `no_feedback` ignores future conditioning in planning,
while retaining stochastic fills, inventory changes and common reward.
`frozen_model` conditions each within-model regime belief but freezes the
marginal model weights inside planning. The latter is an artificial
information-state intervention: it is not a coherent Bayesian posterior and
does not isolate model-learning economics without qualifications. A separate
one-step intervention can score the unchanged Bayes-adaptive continuation under
either suppressed update, avoiding continuation-policy confounding.

## Predeclared numerical architecture proposal

- Float64 computations and trilinear interpolation on `[0,1]^3`; include all
  boundaries and the exactly symmetric cold-start point.
- Gauss-Hermite normal integration for continuation; analytic reward avoids
  integrating its smooth linear component repeatedly.
- Reuse outcome kernels across inventory states, and integrate the next public
  signal before each belief-kernel application.
- Store full-horizon values, with optional action-value retention for efficient
  batched evaluation; source/core remain untouched.
- Candidate refinement ladder, subject to the shared pre-evaluation freeze:
  `(w,b,GH)=(9,17,15),(17,33,25),(25,49,41),(33,65,61)`.
- Numerical tables approximate continuous-belief Bellman quantities. An
  analytical information inequality does not make these numerical values
  certified upper or lower bounds. Observed refinement is empirical evidence,
  not a global interpolation or quadrature error certificate.

## Failure catalogue, declared before coding

1. Updating each hidden belief while failing to update the model weights loses
   the joint posterior and implements a different controller.
2. Multiplying marginal side probabilities after mixing hidden states gives the
   wrong two-sided likelihood; sides mix jointly conditional on the return.
3. Treating an unsubmitted side as a non-fill invents evidence. No-quote and
   market actions must carry no fill likelihood or model evidence.
4. Predicting the regime before conditioning on current fills swaps timing;
   using one kappa for both models erases the intended uncertainty.
5. Revealing the hidden regime together with the model creates a different
   comparator. Known-model controls still integrate hidden regimes.
6. Averaging model-specific maxima in a feasible action recursion grants
   unearned information. Only the delay-zero revelation continuation may do so.
7. Off-by-one revelation timing can make delay one equal to immediate revelation
   or make revelation after termination change terminal liquidation.
8. Delayed recursions may accidentally continue with their own delay rather than
   delay minus one; short horizons must collapse to the BA recursion.
9. Incorrect common-action maximization or baseline interpolation can invalidate
   weighted-known-Q agreement and the numerical information hierarchy.
10. Suppression ablations may change likelihood mass, reward, inventory,
    observation access at execution, or continuation in addition to the stated
    intervention. Their exact semantics must be visible in metadata.
11. Frozen model weights are not a Bayes posterior; calling this a rigorous
    information-removal bound or pure causal model-information value overclaims.
12. Zero model weights, impossible hidden signs, near-degenerate likelihoods,
    underflow, or a zero outcome mass may yield NaN, divide by zero or revive an
    excluded model. These cases need defined harmless conditional coordinates.
13. Uniform-grid interpolation may leave the cube, use negative weights,
    transpose axes, omit boundary endpoints, or mix independent beliefs in the
    wrong flattening order.
14. Interpolation at weight zero/one must ignore the unused model's hidden
    probability; duplicate conditional coordinates may create numerical ghosts.
15. A kernel may lose/duplicate mass, assign an unsafe inventory transition, or
    omit a visible fill subset. Outcome transition masses must sum to one for
    every admissible state/action.
16. Return quadrature may be underresolved, especially near posterior decision
    boundaries. More nodes may change actions without large value changes.
17. A belief mesh may look converged at the cold start while disagreeing at
    reached states, boundaries, or short horizons. Refinement must examine all
    horizons and independent states, with action regrets as well as values.
18. Signal-transition orientation, market-order mark-to-market reward, passive
    execution covariance, pre-action risk timing, and terminal liquidation are
    easy to misalign. Reuse verified model constants and audit identities.
19. Inadmissible actions or unused horizon-zero Q entries must not become
    finite through interpolation, overflow, or a masked dot product.
20. Ties must use the verified absolute tolerance and lowest action ID; ordinary
    `argmax` or relative closeness changes established controller semantics.
21. A cache can be stale, partial, corrupted, generated with a different source
    or specification, or loaded with swapped model/grid axes. Authenticate its
    source, configuration, arrays, shapes and numerical stack before reuse.
22. A temporary cache or new extension output must not overwrite canonical
    artifacts. No modification to existing `src/trade_learning` is authorized here.
23. Float32 storage, nondeterministic parallel reductions, or insufficient memory
    can silently change policies. Record dtype/build/timing and fail explicitly
    rather than degrading precision without a frozen amendment.
24. Reporting a table maximum as an executable policy's achieved expectation,
    or calling approximate inequalities certified bounds, confuses distinct
    objects and must be prevented in documentation and output names.
25. Numerical level choice based on favorable policy outcomes would contaminate
    evaluation. Resource pilots and validity checks precede final evaluation;
    choose refinement only under the frozen protocol.

The E2E harness is maintained separately. No post-implementation unit test suite
is to be added; it should produce repeatable numerical and policy artifacts from
the declared failure modes.

## Implemented architecture and numerical acceptance clarification

The frozen shared `config.json` is authoritative. Known-model tables use 321
belief points and 161 quadrature nodes, independently compared with 641/161 and
321/321. Joint numerical refinement retains full float64 Q histories for
`bayes`, `no_feedback`, and `frozen_model`; the three delayed revelation values
are retained as full-horizon V histories. Every array is a disk-backed NPY file.
Outcome kernels are shared across inventory states and are discarded after a
mode's completed recursion. Each complete family authenticates its source,
numerical stack, specification, array shapes and byte hashes.

The frozen finite-Q/value criterion is evaluated on all six joint V histories
and all three feasible joint Q histories. Weighted known-model Q is checked
separately with the scalar reference. Delay-one nodal V interpolation is not the
same off-grid quantity as the maximum of directly evaluated weighted known-model
Q; both are exposed separately. Delay zero weights interpolated known-model V,
not the model-revealed current-regime value. The API's `revelation_value` makes
these distinctions explicit.

Refinement starts with the second declared grid and second quadrature level.
At each candidate it compares the preceding grid at fixed quadrature, and the
preceding quadrature at fixed grid. Only a failing axis advances. A selected
setting must pass both independent comparisons; exhaustion of a failing axis is
recorded as unresolved. No criterion depends on economic outcomes. The maximum
robust-action disagreement fraction is taken separately for every policy and
remaining horizon, then maximized, a conservative interpretation of the shared
fraction rule.

The diagnostic states comprise the prescribed 2,048 draws from a uniform joint
simplex law, generated from the frozen numerical namespace, plus 405 boundary
states crossing all inventory/signal values and coordinates in `{0,.5,1}`, and
128 reflected inventory/signal counterparts. Initial-state probes are exactly
the five weights specified in the shared protocol. Saved probe bytes identify
the identical states used across all numerical comparisons.

Before economic evaluation, resource-only measurements on this host found:

| Joint grid and quadrature | Full-update kernel build | One Q backup | CSR bytes |
|---|---:|---:|---:|
| 9 x 17 x 17, GH 15 | 0.85 s | 0.042 s | 103,792,608 |
| 17 x 33 x 33, GH 25 | 10.20 s | 0.484 s | 1,288,333,020 |

Both operator mass errors were at most 4.45e-16. These are isolated resource
measurements, not complete campaign timings or policy results. Full-family
metadata records actual timings and final array sizes.

## Implementation revision v2, before economic evaluation

The original implementation is preserved byte-for-byte in
`revisions/solver_v1.py`. Its completed artifacts and failed refinement
comparisons remain identifiable under the original cache/output directories.
The 33 x 65 x 65, GH 25 run was stopped with SIGINT after completing three
horizons: a 11.85 GiB operator required repeated disk reads, and its second
three-continuation step took 96.68 seconds. Its incomplete directory is marked
`ABANDONED.json`, not authenticated as a complete control family.

Revision v2 batches the already available BA, delay-one and delay-two
continuations at remaining horizon `n-1` in a single sparse operation to obtain
BA, delay-two and delay-three Q values at `n`. Their equations, state ordering,
kernel, summation order over outcomes, precision and action selection are
unchanged. Future Q-file slices are assigned completely when calculated instead
of first being filled with unused negative infinities. Horizon zero remains
negative infinity for all action tables.

All v2 results use a fresh cache prefix and source fingerprint. Complete small
families are compared against the preserved v1 implementation across every
stored V and Q element before the full numerical refinement is restarted.
Existing v1 caches are never relabelled as v2 artifacts. The anticipated v2
temporary action-score array at a conditional 49 x 97 x 97 grid is about
1.83 GB; large kernels continue to be streamed from disk.
