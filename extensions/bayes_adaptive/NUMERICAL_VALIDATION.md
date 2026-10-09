# Independent raw numerical-check audit

This audit is read-only. It reconciles saved joint-control refinement rows
against authenticated value and retained-Q arrays at the frozen probes. It does
not solve a new control problem, certify a uniform error bound, or turn an
unresolved receipt into numerical acceptance. `verify_numerical_raw.py` does
not import the production solver's coordinate, interpolation, comparison, or
selection functions.

## Failure modes recorded before implementation

1. A receipt, statistical/numerical configuration, probe file, solver source,
   core source, or amendment identity may be absent, changed, or mismatched.
   A partially written receipt may lack final amendment fields; its rows can
   still be audited but it cannot be declared accepted.
2. A cached family may be incomplete, from another source or numerical build,
   or have changed array bytes, shape, dtype, action order, or specification.
   The complete manifest and all arrays used by the audit must authenticate.
3. A joint probability may be invalid or a zero-weight model may produce a
   spurious division. The inactive conditional belief must use the documented
   canonical one-half coordinate.
4. Grid axes or flattened indices may be transposed, endpoint interpolation
   may go out of bounds, or weights may become negative or fail to sum to one.
   Coarse and fine values must be queried at the same physical probe states.
5. Invalid actions may be treated as finite, or valid actions may contain
   NaN/infinity. Interpolation of `-inf` can create NaNs unless the independent
   admissibility mask is applied before accumulation.
6. A close numerical tie may use a different action rule. The lowest legal
   action within absolute `1e-12` of the maximum must be selected. A robust
   action disagreement uses the *finer* best-to-second gap above `0.001` and
   the denominator for that policy and horizon, not all probes or an average
   over horizons.
7. Rows may omit a horizon or policy, permute them, report a false metric or
   threshold result, or skip an axis/ladder level. Maxima and pass flags must
   be recomputed from actual arrays for every audited comparison.
8. A selected subset can reconcile while another comparison remains wrong.
   The CLI must label partial coverage. A reconciled failed gate remains an
   unresolved numerical result; receipt claims are reported separately from
   raw-array reconciliation.
9. Hashing or loading large arrays can exhaust available disk bandwidth or
   memory during a solve. Open arrays as read-only memory maps, process one
   family/horizon at a time, and allow a single axis/pair to be selected.

The bounded first check is the existing first belief-axis comparison at
`9x17x17 -> 17x33x33`, GH25. It exercises all 30 horizons, six V modes,
three retained-Q modes, and all frozen probes. That historical uniform
comparison **reconciled and failed** the unchanged convergence thresholds.

## Endpoint-geometry amendment: failure modes fixed before verifier changes

Recorded 27 September 2026 before changing the independent verifiers. The new
`endpoint_sine` branch is a numerical geometry change only. Historical uniform
receipts and failures remain evidence; a new source/format identity must prevent
them from being presented as current accepted results.

1. A receipt may claim endpoint geometry while the selected family or any
   comparison family used uniform axes, or vice versa. Require one geometry in
   every specification and independently authenticate the actual three axes in
   every family manifest, including generation rule, order, endpoints and hash.
2. An axis may have the right length and hash but the wrong physical points,
   ordering, symmetry, midpoint, or flattened-coordinate convention. Reconstruct
   both conditional axes from the declared sine-squared rule and the weight axis
   from a uniform linspace. Reject altered, missing, nonmonotone or swapped axes.
3. Zero-weight models can give an undefined conditional probability. Preserve the
   canonical one-half coordinate and locate all probes by their physical values;
   search each authenticated axis for its enclosing interval rather than using
   a uniform-grid scaling formula. Check exact endpoints and convex corner weights.
4. A format, specification, source, build, amendment document, wrapper, or array
   record may be stale or forged. Authenticate these identities independently,
   reject mixed V2/V3 caches, and preserve every original array hash and family
   identity. A cold rebuild must match all eleven actual arrays on the same build.
5. A change to geometry may accidentally change the frozen experiment: thresholds,
   2,581 probes, 30 horizons, six value modes, three retained-Q modes, known-model
   reference, evaluation seeds, or statistical design. Recreate these from the
   frozen configuration and reject a shortened or selectively passing receipt.
6. A changed receipt might report a passing status while raw interpolation
   disagrees, a failed earlier setting is hidden, or only one refinement axis
   passes. Recompute all selected rows and maxima from authenticated physical
   arrays; retain partial-coverage labels and never promote a failed raw audit.
7. A cold reproduction may reuse an old uniform reference smoke, compare only
   metadata, or accept its own newly produced outputs as the reference. Bind the
   reference source/geometry/array records first, build into a fresh cache, and
   compare all eleven array records and every nontiming smoke file byte for byte.
8. The portable review ZIP nests the original core under
   `Trade Learning_Hugo_Lemonnier/`, while keeping the extension beside it. A verifier
   that assumes `outer/src/` may reject valid extracted evidence or silently
   use a local checkout. Resolve an explicit core root (or the documented
   fallback) and hash its actual source bytes before accepting the receipt.
