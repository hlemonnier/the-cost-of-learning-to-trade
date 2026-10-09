# Experiment and packaging failure modes, before implementation

- Wrong true-model prior/label reaches a feasible controller; evaluator and public
  records must remain separate. The privileged reference receives M only, never H.
- Updating weights using unsubmitted fills, or treating missing as nonfill; use
  selected feedback and independently reconstruct it from action/tape.
- Resetting model weights each trading step, carrying posterior between evaluation
  episodes, or carrying final pilot H into a new episode biases learning.
- Pilot budgets use different/non-nested pilots across policies, overlap evaluation
  streams, or share action randomness with market innovations.
- Count 900,000 policy records as independent trajectories, or treat 0/1/5 budgets
  and same-stream policies as independent samples. Cluster on independent pilots
  and preserve balanced model strata and paired episode keys.
- Missing model/replicate/budget/policy/episode, duplicated keys, nonfinite metrics,
  wrong number of pilots, incomplete terminal ledger or altered action arrays must
  fail closed before reporting uncertainty. One pilot is insufficient evidence.
- Average conditional-model Q maxima instead of maximizing the weighted actions;
  confuse its relaxed planning score with achieved policy performance.
- Call approximate revelation values certified upper bounds, conceal a hierarchy
  violation, or choose numerical settings after seeing evaluation PnL.
- Freeze a model update but change reward, observation law, inventory, conditional
  regime update or continuation; a different action alone does not identify why.
- First-step model weights change under uniform conditional H despite identical
  observation laws across kappas; reject this incorrect chronology.
- Choose favorable pilots/seeds/action witnesses; keep prespecified streams,
  complete data and deterministic state-search provenance, including negatives.
- A full-run receipt references stale source/config or trusts warm damaged tables;
  bind source/build/input/content hashes and rerun cold checks independently.
- Large artifacts/caches leak into Git or the review ZIP, original science is
  overwritten, ZIP extraction inherits a parent .git, or source-export commands
  need private paths; use explicit fresh destinations and verify final extraction.
- A PDF removes scientific caveats, overstates a gain/near-optimality certificate,
  exceeds the original main-report limit or has unreadable/clipped figures; keep
  core PDF unchanged and inspect every new rendered appendix page.
