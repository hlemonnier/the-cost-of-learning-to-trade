# Explanatory analysis: design and failure catalogue

Frozen before implementing `explain.py`, generating explanatory tapes, or reading
new explanatory action comparisons. The protocol/configuration at `8db6208` owns
the market, economic estimand, numerical acceptance rules, and final evaluation.
This separate analysis does not select policies from realized final PnL.

Design record: version 1, frozen at 2026-09-24 22:53 UTC (25 September 00:53
Europe/Paris). The root agent explicitly accepted this design before generation.
Read `AGENTS.md`, the main `WORKLOG.md`, and the extension `WORKLOG.md` before
using or resuming it. Protocol commit:
`8db62080ef4e5170413d01b7f5fdac33bfd2b1e5`; configuration SHA-256:
`8681e5a8c48f25967f95a4b7d42b76531cbc4e8e00d013114c3e6efe93aee462`.
Every generated manifest records this design file's actual hash and its code
source identity. A correction requires a new output directory with the same
declared seeds, retaining the earlier evidence.

## Fixed design

- Market: theta .35, kappas .002/.10, horizon 30, inventory cap 2; start every
  episode with all four joint probabilities equal to .25.
- Generate 1,000 independent episodes per true model under uniform random
  selection among all currently admissible actions. Market seed entropy is
  `[260925508,410,0,model_index]`; independent action entropy is
  `[260925508,410,1,model_index]`. Neither uses final-evaluation namespace 210.
  Save all public observations and the full pre-decision joint posterior. The
  evaluator model index identifies provenance but is never supplied to a score
  function or posterior update.
- The candidate set contains all 60,000 pre-decision states from those episodes,
  plus the already prespecified numerical-probe states at every remaining
  horizon 1 through 30. The latter are arbitrary admissible joint priors, whose
  reachability under the initial prior is not asserted. Reached states are
  reached under uniform exploration, not necessarily under any scored policy.
- First score every candidate using the selected family's interpolated BA and
  weighted-known-model-Q tables. Retain every state, chosen action, best-to-second
  gap, and traversal identity. For each source group (reached/arbitrary), retain
  BA/WQ action disagreements, deduplicate exact `(n,q,x,p)` state bytes, and sort
  by descending minimum of the two best-to-second gaps, then candidate ID.
  Directly examine at most the first 256 states in each source group. The cap is
  a disclosed resource limit; a negative result is not an exhaustive theorem.
- Direct scores use `ControlFamily.bellman_q('bayes',...)` with the same stored
  BA continuation and with update modes `bayes`, `frozen_model`, and
  `no_feedback`. The weighted-Q comparator uses the family's known-model Q
  tables. Retain all eleven action entries, including explicit nulls for illegal
  actions, and independently calculate analytic expected reward and model /
  conditional-regime information gain using 241-node normal quadrature.
- A selected-resolution candidate qualifies only when direct BA and deployed
  table BA choose the same action, BA differs from WQ, the single-backup frozen
  model intervention chooses WQ's action, all three direct BA/FM/WQ best-to-second
  gaps are at least .001, and model information for the BA action exceeds 1e-10.
  Qualification is numerical and descriptive, not a certified action theorem.
- Check each directly examined state's scores on every comparison family
  provided by the primary agent. A refinement-stable witness requires the same
  three chosen actions and the same .001 margin/information criterion at every
  supplied family. With no comparison family, stability is unresolved.
- Prefer refinement-stable witnesses reached under the saved uniform policy;
  within that class maximize the minimum BA/FM/WQ direct margin, with candidate
  ID breaking ties. Only if none exists choose an arbitrary-prior witness with
  its reachability limitation. If no stable witness qualifies, preserve the
  negative result and the strongest checked candidate, using the same
  reached-first / minimum-margin / ID rule. Never alter thresholds, seeds,
  action semantics, candidate counts, or the shortlist after seeing results.
- Also check fixed cold/equal-conditional-belief, singleton-model, boundary and
  terminal-horizon states. Equal conditional beliefs imply zero model MI and
  zero same-continuation frozen-weight difference; every update suppression
  has the same score at horizon one. Retain diagnostic violations explicitly.
- Error envelopes use THEORY.md equations (27)-(32), exact mathematical stage
  envelope .231 and liquidation span .054. Compute one-dimensional Gaussian
  transport distance to each positive normalized quadrature rule. Mark all
  floating evaluations `certified_bound=false`; preserve one-sided, nodal
  two-sided and off-grid allowances as separate fields. Observed cross-cache
  score changes are not substituted for a global error allowance.

## Output contract

`prepare` writes `exploration_public.npz`, `candidate_states.npz` and a manifest
containing configuration/design/source hashes, seeds, counts, shapes and file
hashes. The full public history suffices to replay each reached belief.

`analyze` writes `traversal.csv`, `direct_candidates.jsonl`, `direct_scores.csv`,
`explanation.json`, `error_envelopes.json`, `EXPLANATION.md` and a manifest.
Each traversal record contains its candidate/source IDs, `(n,q,x,p)`, selected
BA/WQ actions and gaps, and whether it was directly examined. Each JSONL entry
contains the complete selected and comparison score vectors, reward and
continuation decomposition, information quantities, flags, provenance, and
selection margins. The CSV score table has one row per candidate/family/legal
action and explicit legality rows for all eleven actions. Invalid scores are
null, never fabricated zero scores. Cache arrays are referenced by authenticated
manifest identity, not copied into the result.

`envelopes` and `diagnostics` are lightweight standalone paths. The latter
retains an E2E numerical-check artifact for information decompositions, normal
transport integration, analytic reward consistency, and rejection probes. It
does not construct a control table or run a new solver.

## Failure catalogue declared before code

1. Reusing namespace 210, adapting seeds, importing final economic outcomes, or
   changing the candidate distribution after a preferred policy loses leaks
   evaluation into the explanation.
2. Supplying true model labels, hidden signs, unselected fills or future returns
   to posterior/scoring code creates an inadmissible controller. Model labels
   appear only in provenance; the public reconstruction must suffice.
3. Uniform action draws may include actions unsafe for one possible fill subset,
   or assume opposite-side fills protect a boundary quote. Use the observed
   admissibility mask and preserve the chosen action and selected outcomes.
4. Carrying a previous episode's regime or model posterior into a cold episode
   changes the prior. Each independent explanatory episode resets all masses
   to .25 and begins at q=x=0.
5. Predicting before observing, multiplying mixed side likelihoods, treating a
   missing side as an observed non-fill, or silently dropping selected non-fills
   breaks the four-state filter and both information measures.
6. MI of the future predicted regime confounds observed information with new
   switching uncertainty. Compute current-regime posterior information before
   prediction and report conditional-regime and marginal-regime MI separately.
7. Entropy or model truth probability used as a score bonus would change the
   policy. Information quantities are diagnostics only; true-model information
   is never part of action selection.
8. Comparing recursively frozen-model tables with a BA table and calling it a
   same-continuation intervention changes two ingredients. Direct scores must
   call the BA continuation for every update mode and keep the outcome law,
   immediate reward and stochastic inventory identical.
9. Interpolated BA and direct Bellman BA may choose different actions off-grid.
   Record both and disqualify a deployed-policy witness when they disagree.
10. Ties, invalid actions, NaNs or infinity subtraction can fabricate a gap.
    Preserve the established absolute-1e-12 stable-lowest-ID tie rule and fail
    on nonfinite legal scores. Illegal scores use a separate legality mask.
11. The table screen may miss direct-score disagreements or omit lower-ranked
    witnesses. Report screened count, disagreements, deduplication and cap; a
    null is a result for this fixed search only.
12. A witness chosen solely from an arbitrary belief cube may be unreachable.
    Keep the provenance classification and all public prefixes for reached
    states; never imply on-policy prevalence from an explanatory example.
13. Full Q vectors or negative cases may be omitted while reporting only chosen
    actions. Retain every traversal, every directly scored candidate, all
    actions and comparison families, including disqualification reasons.
14. Cross-cache changes are empirical resolution evidence, not certified error.
    No comparison family means unresolved stability; failed refinements remain
    adverse evidence even if one attractive example is stable.
15. Freezing marginal model weights is not a coherent removal of only model
    information, and alters the marginal regime forecast. Preserve the
    qualifications proved in THEORY.md; do not claim pure causal separation.
16. Gaussian rewards are unbounded. The .231 envelope bounds conditional
    expected rewards and policy alpha values, not realized PnL support.
17. Computing transport distance with unsorted/negative/unnormalized weights,
    wrong tail endpoints, a variance-two Hermite rule or invalid CDF cuts gives
    a false envelope. Check mass/moments and independent integration at a
    simple rule; explicitly retain any roundoff treatment.
18. Omitting the final off-grid interpolation term, using n instead of n-1 in
    a continuation envelope, or reusing unbudgeted known-model errors invalidates
    the bound. Distinguish BA allowances from delayed-revelation estimates.
19. Evaluating (29)-(32) in ordinary floating point does not certify their
    numerical values. Mark missing interval arithmetic, rounding, and solver
    arithmetic allowances; no .002 optimality certificate is inferred.
20. Cache identity can drift while another agent edits/solves. Load only complete
    authenticated families with the same model/horizon/known-table contract;
    never trigger a solve from explanation code or copy giant caches to output.
21. Corrupt/missing candidate arrays, duplicate IDs, nonfinite beliefs, bad
    dimensions or shifted n/t must fail closed rather than produce a partial
    apparently successful result. Verify manifests before analysis.
22. Overwriting original scientific data, baseline archive, prior evidence, or another
    agent's source would violate the freeze. New outputs use fresh directories;
    this component owns only its own source/design and explanatory artifacts.

The command artifact is the verification mechanism. No unit test file is to be
written after the implementation, and no original source or scientific output
is changed by this analysis.

## Provenance correction declared before implementation

On 25 September 2026, after the authorized resume and before final candidate
scoring, the primary review identified that the optional numerical receipt was
only logged. A receipt saying `passes_predeclared_rule=true` could therefore
refer to another family. This correction changes no candidate, seed, score,
ranking, margin or economic hypothesis. Preserve `revisions/explain_v2.py` and
the complete V2 evidence; regenerate the same candidates once under V3 source
identity and require both NPZ files to remain byte-identical.

Failure cases to reject before creating analysis outputs or screening candidates:
an accepted receipt with a missing or different selected artifact, specification,
source identity, or build identity. A malformed acceptance flag must also reject.
A false acceptance flag is retained as unresolved evidence, never upgraded to a
pass. The bound receipt hash must remain unchanged throughout the analysis.
The command-level verification uses one authenticated existing family, a matched
synthetic receipt explicitly marked as diagnostic, and one altered field at a
time. It checks rejection before any candidate/output work. Synthetic acceptance
fixtures are not real numerical acceptance or new explanatory observations.
