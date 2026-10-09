# Mechanism-study failure catalogue — written before implementation

1. The JSON advertises settings, policies, seeds or contrasts different from execution.
2. A root seed reuses canonical/final/pilot/reduced streams or episodes fail to reset.
3. A policy sees H, future shocks, unsubmitted outcomes or evaluator metadata.
4. Forecast deletion also removes X from fills, changes K, selection correlation,
   simulated returns or the filter's true .03X centering.
5. Forecast deletion removes q*mu*x but misses predictable returns on new fills,
   reverses ask signs, or affects invalid-action sentinels.
6. The blind comparator assumes theta=0 or uses a different Q table/horizon.
7. Blind beliefs quietly incorporate feedback, or filtered comparators do not.
8. An inventory ablation changes actual inventory, immediate risk, safety masks,
   transaction costs or the common continuation function as well as its q argument.
9. Inventory suppression erases final liquidation, applies liquidation early,
   interpolates the wrong horizon or gets the pre/post-decision q sign wrong.
10. The inventory deletion fails direct fill enumeration, or is nonzero when
    continuation is independent of inventory. At the final step it must be zero.
11. Same-continuation inventory/feedback examples use different states or select
    different continuation values, or an action-invariant offset changes tie handling.
12. Reward and continuation quadrature errors are mixed, or refinements omit
    shorter remaining horizons, near ties, failed steps or the no-forecast planner.
13. Float32 arrays, nonfinite legal scores, finite invalid scores, stale caches,
    malformed belief grids or source/build/content identities are accepted.
14. Treatment effects are inferred from planning scores instead of actual rollout
    outcomes under the unchanged true simulator.
15. Ledger reconciliation omits/doubles liquidation, uses post-decision risk,
    or treats the risk penalty as a cash fee.
16. Policy comparisons do not share the exact exogenous tapes/episode IDs, or
    duplicates, missing policies/episodes and nonfinite outcomes are accepted.
17. Statistical SE uses periods or policy rows as independent observations,
    rather than paired independent episodes; family size or tail quantile is wrong.
18. An unadjusted significant result is presented as simultaneous evidence;
    negative findings are omitted, sample sizes increased, or policies retuned.
19. Corrupt action arrays, outcome bytes, trace files, manifest hashes, protocol,
    build provenance or source changes are silently accepted by verification.
20. Reusing a warm cache is described as a cold repeat; failures or timing-only
    changes are hidden; replay changes the episode batching and hence RNG stream.
21. Same-policy fresh runs differ, or exact repeat is claimed across all platforms
    based only on the two tested local processes.
22. Interacting intervention effects are summed as an additive PnL decomposition;
    inventory-mediated signal opportunities are described as pure risk management.
23. The full-information comparator is passed future H/shocks; its planning-value
    calibration is incorrectly applied to filtering noinfo or ablated scores.
24. The supplementary study changes canonical production source/data/primary,
    loses earlier adverse evidence, or is represented as external scientific sign-off.

Acceptance uses the complete command-line study twice from cold tables, an
independent raw-data/statistical and scalar-ledger verifier, analytical operator
checks specified in the protocol, and saved current-byte evidence. Fault probes
must reject wrong protocol, missing/duplicate/unpaired IDs, nonfinite outcomes,
wrong scalar ledger, corrupted action data and stale file/source hashes. No unit
test is added after implementing the behavior it tests.
