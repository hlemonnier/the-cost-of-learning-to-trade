# Numerical contract correction: failure inventory and acceptance protocol

Written before implementing AUD-01/AUD-02. Original snapshot/results are retained; all corrective artifacts use `outputs/verification/audit_corrections/`. No full-horizon campaign is launched by this harness.

## Prespecified decision rule

Use float64 scores and an **absolute tolerance of 1e-12 objective units**, relative tolerance zero. Among admissible finite actions whose score is within that tolerance of the maximum, choose the lowest action ID. This tolerance is fixed from roundoff scale and the audit's mathematical tie, not from PnL. It is nine orders below the reduced-reference 0.001 meaningful-action threshold and more than nine below the practical 0.005 threshold. It also lies far below the smallest execution fee (0.001). For a fixed numerical transition model, choosing such actions at each of T steps changes the Bellman supremum value by at most T times the tolerance; this is not a continuous-state error certificate.

The same rule governs Bellman policy/value selection, practical policies, reference rollouts, and refinement action comparisons. Bellman V stores the Q of the selected action. A terminal row has no legal decision. A legal NaN/infinite score, or a row with no admissible action, must fail rather than select an arbitrary index.

## Failure modes covered before implementation

1. Exact mathematical ties resolve through incidental floating-point rounding.
2. The myopic q=-1,x=0,adverse-regime abstain/market-buy tie changes under sub-tolerance perturbations.
3. Symmetric passive actions use different tie rules; a lower-ID illegal action is selected.
4. Runtime, oracle, reduced replay, Bellman value, and convergence comparisons use inconsistent rules.
5. A genuinely better action outside the fixed tolerance is discarded, or tolerance grows with score magnitude.
6. Terminal/invalid remaining horizons silently choose abstention; all-invalid rows, NaNs or positive infinities pass.
7. Finite cache corruption survives because only completion-marker existence is checked.
8. Wrong dtype/shape, malformed/nonuniform beliefs, illegal finite Q values, missing legal Q values, corrupt terminal sentinels, or inconsistent stored V survives loading.
9. Missing arrays, malformed/incomplete metadata, partial writes, or a stale format is silently rebuilt/reused.
10. Wrong settings, source hashes, numerical contract, Python/NumPy/SciPy/build fingerprint are accepted.
11. Metadata/content digests do not bind all three arrays and the declared admissibility mask.
12. A cache changes after initial loading but the campaign still claims unchanged evaluated artifacts.
13. Cache identity/fingerprints depend on nondeterministic solve duration, or metadata returned to consumers can rewrite the recorded fingerprint.
14. Existing original artifacts are overwritten by a correction verification command.
15. Refinement examines only the initial horizon, hides near-tie action changes inside robust changes, or averages away a bad intermediate horizon.

## Acceptance experiments

The new `verify_numerical_contract.py` executes tiny real solves, runtime policy selection, save/load, fresh-bank cache loads, deliberately corrupted scratch caches, and post-load mutation detection. It persists JSON evidence, including hashes and rejection reasons. The legal-action and myopic fixtures are specified before the implementation. Structural corruptions are also tested after refreshing their declared byte hashes, to distinguish semantic validation from mere file-hash checks. Those refreshed manifests are deliberately invalid fixtures, not genuine research artifacts.

Cache content is SHA-256 bound to settings, source files, numerical contract, numerical runtime/build and all array bytes. Timing is provenance but excluded from deterministic identity. Read-only memory maps are used during evaluation. Existing invalid or incomplete logical cache entries raise an explicit error; an operator may deliberately remove/rebuild a named entry or use a new cold-cache directory. No silent recovery occurs. These hashes detect changes relative to a trusted recorded manifest; they are not a signature against a filesystem owner rewriting both evidence and data.

Refinement now checks every remaining horizon n=1..T. Preserve the original tolerances, report the T-step initial value drift and maximum initial-state drift over horizons, the maximum common-state drift, per-horizon and aggregate robust disagreements, and numerical/near ties separately. Passing requires the initial-state tolerance at every horizon, the common-state tolerance over every horizon, and the robust-disagreement tolerance at every horizon. This expansion may fail formerly passing settings; preserve failures and choose any extra numerical resolution only from those checks.

Bounded commands after implementation: the numerical-contract harness; corrected reduced control checks; corrected reduced paired rollout using a new explicit output/cache path. Separate fresh-process cold-cache smoke and full corrected campaign are owned by the primary agent. Cross-stack exact equality is not asserted from a single supported-stack run.

## Declared secondary-stack check

Before examining its results, rerun the same fixed 20,000 reduced episodes and
seed 260924301 using the already installed Python 3.14.6 / NumPy 2.5.0 /
SciPy 1.18.0 / pandas 2.3.3 environment, with a separate cold cache. The primary
stack remains Python 3.12.14 / NumPy 2.5.3 / SciPy 1.18.1 / pandas 3.0.6.
Report per-episode and mean differences, whether zero or nonzero. Do not change
the tie rule, any scientific setting or the final campaign based on this check.
This tests two local macOS stacks, not Linux or every possible numerical build.
