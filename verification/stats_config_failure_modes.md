# Statistics and configuration acceptance contract (AUD-03 / AUD-04)

Recorded before the corrective implementation. This document enumerates failure probes; it is not a task backlog. The executable acceptance experiment is `verify_stats_config.py`. Original data/results are read-only inputs, and correction artifacts are written separately under `outputs/verification/audit_corrections/`.

## Statistical failures to reject

1. A single fitted pilot, including the audit's nonconstant `[0,10,20]` example, cannot identify independent-pilot uncertainty.
2. Empty data, missing identifiers/outcomes, NaN/positive or negative infinity, nonnumeric or complex outcomes, and invalid confidence levels cannot produce inference.
3. Duplicate `(pilot,episode)` observations, nonintegral/negative/out-of-budget IDs, mixed policies/variants/environments passed to a single-stratum helper, unbalanced samples, and fewer than two episodes per pilot cannot silently pass.
4. A declared pilot or episode budget must match every stratum; relabelled IDs cannot disguise a missing pilot or episode.
5. Stratified inference must require the explicitly declared protocol/population. Dropping, adding, or relabelling an environment must not silently change its estimand.
6. A campaign must contain exactly the declared environments, variants, policies, independent pilots and episode IDs; every paired key must be unique and every comparator present.
7. A missing row, entire comparator, environment, variant or pilot, or a duplicate inserted to restore total row count must be rejected before deployment assessment.
8. Environment names must agree with numeric parameter metadata. Public candidate support and actual policy implementation must match the validated manifest specification.
9. Objective, PnL and financial/accounting outcomes must be finite. Intentionally unavailable privileged/public monitoring diagnostics may remain NaN and must not be mistaken for broken economic outcomes.
10. Recorded pilot identities must be present, equal across paired policies/variants, and distinct across independently claimed pilot datasets. Learning policies must share the same starting-posterior fingerprint per pilot. These structural checks do not prove statistical independence; it still rests on the separately audited streams.
11. Arithmetic overflow after otherwise finite input must fail closed. Zero observed between-pilot variance with at least two valid independent pilots is not itself a failure; preserve the canonical estimator rather than imposing an unrequested variance floor.
12. No truncated campaign, or smoke-only integration population, may emit an affirmative deployment decision for the nine-environment population.
13. On the original full balanced dataset, every existing numeric summary field, including all primary/secondary means, standard errors and intervals, must agree within absolute `1e-12`. Validation must not change the estimator.

## Configuration failures to reject

14. Unsupported changes to theta/kappa support, order, duplicates, policy set/order, variants, primary policy/baseline, threshold, population label, prior, online-update rule or selection rule must fail before table construction or simulation.
15. Unknown/misspelled fields, missing required fields, JSON duplicate keys/nonfinite literals, unsupported profile/version, booleans masquerading as integers and nonpositive or nonintegral dimensions/resolutions must fail closed.
16. The fixed full benchmark retains `T=300`, inventory bound 5, ten pilots, 100 training and 100 final episodes; unsupported budget edits fail, including raw-file edits that smoke overrides would otherwise conceal.
17. The explicitly supported smoke profile retains `T=12`, bound 2, two pilots, 12 training and 16 final episodes, one evaluation pair `(0.35,0.02)`, all seven policies/three variants and the full nine-model public prior. Its population label and no-deployment status must match those facts.
18. Valid explicit numerical-resolution overrides are honored in the resolved protocol and the actual TableBank arguments; invalid overrides fail. The default full support-point refinement remains explicit and common to active/noinfo.
19. `execute` must revalidate direct caller protocols rather than trusting only CLI parsing. Invalid execution must fail before creating an output directory, constructing a table bank or drawing a simulator stream.
20. The final run manifest must include the resolved protocol fingerprint and authenticated control source/build/content fingerprints, verified again after evaluation. Configuration metadata cannot advertise a different experiment from the row groups and inference support.

## Evidence commands

`PYTHONPATH=tmp/audited-snapshot/src .venv/bin/python verification/verify_stats_config.py --original tmp/audited-snapshot/outputs/full --capture-before` preserves the original confirmed defects and canonical summary before corrections. The normal command, with corrected `PYTHONPATH=src` and explicit `--original tmp/audited-snapshot/outputs/full`, runs malformed-data/configuration probes and reconciles the original raw campaign against its saved canonical summaries. `--smoke-run` additionally executes one small cold-cache campaign and checks its resolved manifest and control fingerprints; no full-horizon table or full evaluation is built by this harness.
