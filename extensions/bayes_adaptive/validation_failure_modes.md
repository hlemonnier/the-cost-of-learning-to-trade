# Independent validation failure catalogue

Frozen before implementing `verify_extension.py` on 2026-09-25. This is an
end-to-end acceptance plan for the separate Bayes-adaptive extension. It does
not certify the original synthetic experiment or numerical global optimality.

## Model, information and numerical recursion

1. **Wrong hidden-state indexing.** The four states are `(model 0,H=-1)`,
   `(model 0,H=+1)`, `(model 1,H=-1)`, `(model 1,H=+1)`. A model label is fixed
   within an episode; only the sign changes with that model's `kappa`.
2. **Wrong public observation likelihood.** The Gaussian return has the same
   law under all four states. Its observed innovation must condition each
   selected-side fill likelihood. A submitted non-fill is evidence; an
   unsubmitted side and any counterfactual depth provide none. Bid and ask
   outcomes are conditionally independent given the return and state, but not
   after mixing hidden states. Market orders provide no fill evidence.
3. **Timing/leakage error.** The action uses the pre-decision joint belief.
   Selected fills and the return update the current-state posterior, then the
   regime transition predicts the next belief. The next signal is observed
   only at the next decision. Feasible policies cannot receive the true model,
   hidden sign, future innovation, counterfactual fill or evaluator seed.
4. **Invalid posterior.** Probabilities must be finite, nonnegative and sum to
   one. Zero-likelihood cases must be handled without NaNs or silently
   reviving excluded models. Pilot episodes reset the hidden regime to one
   half while sharing their fixed model; online evaluation starts from the
   declared pilot posterior and a fresh regime prior.
5. **False Bayes-adaptive recursion.** Every continuation node must optimize
   on one joint posterior, not use separately model-optimized continuation
   values. Immediate reward, posterior transition, next-signal transition,
   terminal liquidation and tie rule must agree with the stated model.
6. **Invalid numerical claim.** Independently enumerate likelihoods and
   one/two-step Bellman values with direct Gaussian quadrature at diagnostic
   beliefs/actions. Check the singleton-model reduction against the old
   hidden-regime problem. Compare resolution refinements on values, legal
   actions and relevant decisions; treat convergence as numerical evidence,
   never a certified bound. Reject a result with missing/nonfinite refinement
   entries, mass drift or unexplained action instability.
7. **Bound confusion.** Model revelation after an action can relax the
   information constraint, so an exact revelation value upper-bounds the
   exact Bayes value. Its maximizing executable first action does not itself
   achieve that upper bound. Delayed-revelation bounds must be monotone in
   revelation delay. Approximate table values are not certified upper bounds
   without rigorous integration/discretization error control.

## Economic replay and experimental design

8. **Illegal decisions or fill encoding.** Action IDs must be in `0..10`;
   boundary-safe legality must hold for every possible fill subset. The quote
   side/depth code, filled/unfilled/missing mask and taker exclusivity must be
   exact. Action and feedback sequence lengths must equal the declared
   horizon, with no dropped episode or duplicate key.
9. **Wrong exogenous data.** Every policy in a paired cell uses the identical
   signal, hidden sign and Gaussian innovation tape, with different realized
   fills only because of its actions. Recompute selected fills from that tape
   and the threshold. Check signal transitions, hidden-sign transitions,
   return identity `R=.03X+.30Z`, tape finiteness and declared seed/variant.
   Cross-build primitive bit hashes are compared only where the same build is
   promised; economic replay must still reconcile.
10. **Wrong ledger or terminal accounting.** Replay cash, inventory, midpoint,
    fees, spread, marked wealth, pre-decision `Q^2` penalty, terminal
    best-quote liquidation and net PnL from actions and exogenous tapes.
    Inventory penalty is separate from transaction costs. Verify reported
    objective and PnL for every episode, not only summary means.
11. **Pilot pairing or identity fraud.** Pilot IDs must cover the declared
    budget; pilot arrays/hashes must be distinct across independent
    replicates and shared exactly across policies in each replicate. Evaluation
    tapes must use fresh seeds and be paired exactly by pilot and episode.
    A cold-start condition must not import a fitted pilot posterior. A pilot
    ladder must use its declared counts, not silently reuse additional data.
12. **Policy identity drift.** Compare frozen policy/configuration/source
    identity with the manifest. Verify the expected policy set, model family,
    numerical setting, seed partition, pilot count, condition labels and
    predeclared primary contrast. A renamed/missing policy, copied rows or
    policy receiving privileged model/sign data fails closed.
13. **Wrong inference.** Recompute paired differences from raw episode
    outcomes. Aggregate within independent pilot replicate before a
    pilot-aware interval; do not count episode decisions as independent
    parameter-learning replicates. Declare the primary population, contrast,
    alpha, interval type and any multiplicity adjustment. Reject incomplete,
    duplicate, nonfinite, unbalanced or unpaired raw data even if a summary
    still parses.
14. **Unsupported mechanism story.** A changed action, lower entropy or
    posterior concentration alone does not establish economic model-learning
    value. The feedback-value ablation must share state, immediate rewards,
    action set, evaluation tape and continuation specification, changing only
    the declared update. Any selected example must be reproducible from a
    recorded state and independently evaluated action scores.

## Fail-closed artifact checks

15. **Missing/stale evidence.** Manifest must authenticate input source,
    frozen protocol, raw tapes, pilot evidence, action ledger, row-level
    outcomes, numerical checks and summary. Check every expected file and
    shape, content hashes, complete key Cartesian product and raw-to-summary
    reconciliation. Never accept a producer `passed` flag by itself.
16. **Deliberate corruption probes.** An E2E validation run must demonstrate
    rejection of at least: missing file/row, duplicate or unpaired key,
    nonfinite outcome, altered legal action, altered selected fill, changed
    tape identity, altered terminal accounting, pilot reuse, and stale manifest
    hash. Keep a machine-readable report of each injected fault and rejection.
17. **Reproducibility scope.** Retain a command, environment fingerprint and
    result hashes so a reviewer can repeat validation. Where practical,
    compare two isolated cold executions of the extension. Distinguish
    same-build byte equality from cross-build numerical agreement, and record
    any failure rather than silently repairing evidence.

Acceptance requires the independent verifier to pass the actual final export,
all prescribed corruption probes to reject, and the primary agent to inspect
both the methods and their evidence. A partial pilot or smoke run must be
labelled as such.

## Separate explanatory-candidate prefix replay

This check was added to the validation scope before writing its replay code.
It consumes only the separately frozen explanatory public artifact, not the
final evaluation outcomes or hidden tape.

18. **Candidate transport drift.** A missing or altered public/candidate NPZ,
    changed design/source manifest, malformed shape, duplicate candidate ID,
    extra file or nonfinite joint belief must fail before interpretation.
19. **False reachability.** Every reached candidate must map exactly to its
    model/episode/time public prefix, with correct remaining horizon,
    signal, inventory and pre-decision four-state posterior. Arbitrary probes
    cannot be mislabelled reached.
20. **Impossible public histories.** Every sampled action must be legal at the
    saved inventory; unsubmitted sides must have missing-fill markers,
    submitted sides binary selected outcomes. Inventory must follow only those
    outcomes, cold episodes must reset at q=x=0 and joint masses .25, and the
    independent selected likelihood must recreate every next prior without a
    true-model label or hidden tape.

## Additional whole-solver structural reductions

Declared before their fresh small-family E2E run; these are new settings with
unchanged algorithm, not extra statistical outcome trials.

21. **Duplicate model identities.** With both kappas equal, the model label is
    observationally and economically redundant. Bayes control must approach
    the known-kappa score for the marginal hidden-regime belief as the mesh
    refines. At a state with equal conditional regime beliefs across labels,
    direct weighted-known-model Q must equal that same known-model Q, including
    legal-action masks. A discrepancy would reveal erroneous model-dependent
    dynamics, likelihoods or interpolation.
22. **No observation information.** At theta zero, selected fills carry no
    hidden-state or model information. Bayes, prediction-only no-feedback,
    and frozen-model recursions must produce the same full value and Q tables
    for every state/horizon, not merely the same initial action. A difference
    would reveal incorrect update timing or intervention semantics.

## Cold same-array reconstruction identity

This contract was approved before changing the archived V1 verifier. Normal
verification continues to require the exact evaluated artifact hash. The
explicit `--allow-array-equivalent-build` mode is solely for an independently
rebuilt family whose timing-bearing metadata changes while every numerical
array stays byte-identical on the same source/build/specification.

23. **Wrong table format.** The V2 solver writes exact format version 2. A
    validator expecting version 1 would reject genuine results; a validator
    accepting both indiscriminately could admit stale structure. Require 2.
24. **False array equivalence.** A copied or rebuilt `complete.json` signature,
    static source/build/precision/retained-Q/specification field, listed array
    record, actual byte SHA, shape, dtype, legal-action mask or finite cell may
    differ. Reject each even if another field or a prose statement says the
    build is equivalent. Never modify/restamp the study's original manifest.
25. **Receipt substitution.** The original full numerical acceptance must
    continue binding the originally evaluated artifact and study hash, not the
    new timing-bearing artifact hash. The verifier receipt must identify both
    original and supplied artifacts and label whether identity was exact or
    exact-array cold-equivalent. A flag without a supplied table must reject.
26. **Cross-build relaxation.** Array equivalence is allowed only if source,
    numerical contract and full build fingerprint remain equal to the original
    study. Different binaries, environment or arrays require an independent
    run and cannot be waved through by this flag.
