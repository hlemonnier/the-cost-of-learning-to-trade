# Endpoint-resolved numerical geometry amendment

Recorded 27 September 2026 (Europe/Paris), before implementation of solver v3
and before generating any final namespace-210 economic outcomes. Hugo explicitly
resumed the full extension and confirmed that Pro's reported implementation and
artifacts are unavailable. Its reported Linux result is unverified research
input; this local reconstruction must establish its own evidence.

## Scope and fixed acceptance rule

The original statistical protocol and `config.json` remain unchanged. The
earlier uniform meshes, failures and interrupted families remain identifiable.
The models, posterior law, policies, rewards, quadrature candidates, independent
known-model reference, seeds, numerical probe bytes and every acceptance
threshold remain fixed. Both independent refinement axes still cover all 30
horizons, six value modes and all three retained Q modes, including the recursive
`frozen_model` policy. This is an explicit numerical amendment, not part of the
original freeze and not a relaxation of an acceptance criterion.

The previous numerical-only amendment's optional 49 x 97 x 97 level is retained.
Use the same candidate grid counts and independent-axis selection rule, starting
at the second grid and quadrature levels. Each comparison uses the same geometry
on both coarse and fine grids. No cross-geometry comparison substitutes for
refinement. Stop at the first setting satisfying all existing numerical gates;
retain unresolved status if the amended finite ladder is exhausted.

## Geometry and rationale

Use a uniform model-weight axis and conditional-regime axes

    w_i = i/(W-1),       b_i = sin(pi*i/(2*(B-1)))**2.

The lower half of each conditional axis is evaluated in float64, the upper half
is its reflected complement, and the endpoints and odd-count midpoint are set
exactly to 0, 1 and 0.5. This clusters points where conditional regime beliefs
approach certainty. At the persistent model's kappa=.002, regime prediction
allows beliefs near .002 and .998; a uniform 49-point conditional mesh has
spacing about .0208 there. The amendment resolves these regions without changing
the state, observation or control model. It is motivated by the numerical
failure of uniform meshes, not by any final policy-return comparison.

Interpolation remains trilinear in the **physical probabilities** w,b0,b1.
Interval search identifies adjacent physical nodes and uses the ordinary linear
barycentric fraction. Interpolation in the sine parameter would change the
approximation and is not implemented. The uniform option preserves the v2
arithmetic exactly, so the revision can be checked against a preserved v2 source.

## Storage and authenticity

Format v3 uses a fresh cache namespace. `SolverSpec.belief_geometry` is either
`uniform` or `endpoint_sine`. Every complete manifest embeds the physical axes,
their canonical hash, generation recipe and interpolation recipe under
`grid_geometry`; the loader reconstructs and checks the entire geometry record.
Source, build, specification, all eleven numerical array records and the usual
completion digest remain authenticated. V2 caches are never relabelled.

The numerical arrays retain their existing shape, float64 dtype and summation
order. Each completed contiguous Q horizon is flushed before optional advisory
release of its mapped pages. This changes residency, not any Bellman operation.
All required arrays must be complete before a completion manifest is written.

## Failure modes declared before implementation

1. Using uniform cell indices on nonuniform nodes, or interpolating the sine
   parameter rather than physical probabilities, changes the Bellman operator.
2. Reversing an axis, transposing flattening order, duplicating nodes, losing an
   endpoint or rounding away the symmetric midpoint creates invalid weights.
3. Boundary roundoff can produce negative weights, extrapolation, or dependence
   on the conditional probability of an excluded model.
4. Nonuniform interpolation can lose mass or fail to reproduce a multilinear
   physical-coordinate function; transition masses and analytic one-step
   liquidation scores must still reconcile.
5. Changing shared-theta likelihoods, regime-update timing, revelation timing,
   reward chronology, action admissibility or stable tie semantics would change
   the scientific model. Uniform full-array equivalence must detect regressions.
6. A geometry field without authenticated physical nodes can let uniform and
   endpoint tables share a cache identity. Changed geometry names, axes or
   interpolation recipes must reject even after a manifest is re-signed.
7. Premature flushing, advisory eviction or mapping closure can leave a partial
   table, invalidate a live continuation or change its bytes. Cold rebuilding
   must reproduce every array; partial directories must not load as complete.
8. A local short-horizon pass cannot establish 30-horizon convergence or certify
   continuous-belief bounds. The final full all-mode refinement gate remains
   mandatory before economics.
9. Choosing the geometry, thresholds or probes after seeing final economic
   returns contaminates the experiment. Freeze this amendment first and bind
   its hash in the numerical wrapper's receipt.

## Bounded predeclared E2E evidence

Before editing the solver, `geometry_end_to_end.py` is written to execute fresh
small complete families. It compares all eleven uniform arrays against the
unchanged v2 source at two short-horizon configurations, rebuilds endpoint tables
in two separate cold processes, checks physical-coordinate interpolation and
analytic one-step liquidation, and rejects altered authenticated geometry.
It saves manifests and a hash-bound verification receipt. This evidence is a
production implementation check; the separate independent verifier and full
frozen refinement still own numerical acceptance.
