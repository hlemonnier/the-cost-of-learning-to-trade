# Independent final explanation verification design

Recorded 27 September 2026 before implementing `verify_explanation.py`. The
existing independent candidate reader checks the fixed namespace-410 public
prefixes and state array. The final analysis still needs an independent reader
for the producer's direct scores, reward and information diagnostics. This is a
read-only evidence check; it cannot turn unresolved numerical refinement into
acceptance or certify a Bellman optimality bound.

## Failure modes to reject

1. A source, candidate, analysis, or control-family manifest can be stale,
   incomplete, re-signed around changed files, or from the wrong build. Bind
   exact input bytes, manifest digests, all eleven table arrays and explicit V3
   physical axes before reading analytical claims.
2. A traversal may omit candidates, reorder IDs, change public state/provenance,
   conceal a BA/weighted-Q disagreement, or choose a shortlist other than the
   fixed top 256 distinct disagreements per source. Recompute the screen from
   raw Q tables and reconcile every state, ordering and aggregate count.
3. A directly examined state may use a different action, belief, horizon or
   inventory from the authenticated candidate population. Rejoin every JSONL
   and CSV row by candidate ID and family; require all eleven action IDs with
   explicit nulls for illegal scores and no duplicated or missing case.
4. Direct Bellman backups may accidentally use the frozen/no-feedback policy's
   own continuation, advance the regime before observing, use unselected fills,
   lose the common return node, change the public-signal transition or reward,
   or import model truth. Recompute the three update modes with one authenticated
   BA continuation table, independent selected-outcome likelihood and physical
   interpolation, over every legal action and supplied family.
5. The immediate reward, one-period liquidation score or information measures
   can be fabricated or numerically inconsistent. Recompute rewards analytically
   and integrate model, conditional-regime and regime information with the
   declared positive 241-node normal quadrature; check probability mass and
   the common-emission chain relation.
6. A close action tie, illegal-action infinity, false robust gap or changed
   threshold can create a qualifying witness. Apply the established absolute
   1e-12 stable-lowest-ID action rule and recompute every margin, qualification,
   cross-family stability flag, maximum score change and winner from raw data.
7. A partial analysis may pass by checking only a chosen witness or one family.
   Require every directly examined case and every selected/comparison family;
   report counts and a hash-bound receipt. A smaller bounded development check
   must be labelled as such and never substitute for the final full audit.
8. A family may have the same grid counts but a different geometry, or a
   receipt may claim accepted numerics from another selected artifact. Check
   physical axis records and the separate numerical acceptance binding; retain
   nonpassing or absent acceptance honestly.
9. Memory-mapping several full families and replaying millions of random
   table reads can exhaust RAM or filesystem cache. Stream candidates and
   horizon blocks, keep read-only maps, and record the exact scope and runtime.
10. An explanatory allowance can silently use the sine parameter width rather
    than the largest physical belief interval, an average cell width, a wrong
    quadrature transport distance, or a missing terminal contribution. Rebuild
    each reported physical axis, integrate Gaussian transport independently and
    check all horizon rows; keep `certified_bound=false` explicit.

The final validation command will produce a JSON receipt naming all input
digests, replay counts, maximum numerical differences and any rejected fault
probes. No unit-test file will be created.
