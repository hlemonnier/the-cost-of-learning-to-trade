# Resource-envelope measurement protocol (PER-503)

Written before `verification/profile_resources.py`. This is an isolated resource measurement; it does not change the solver or its research protocol.

## Failure modes to rule out

1. A loaded control table or reused directory can make a nominal run warm. Require a fresh child process and an output path that does not exist before solving. Record that the child calls `solve_control` directly and `save_control` only after the solve.
2. Profiling a smaller grid, shorter horizon, wrong parameter pair, or no-information mode understates the canonical active table. Pin theta=.35, kappa=.02, T=300, Qmax=5, beliefs=641, Gauss-Hermite nodes=161, mode=active; check returned metadata and saved artifact specification.
3. Python object sizes and serialized array bytes omit sparse operator, temporary arrays, allocator overhead and imported libraries. Report OS process high-water resident memory from `wait4`/`rusage`, with platform units, and separately report sampled RSS as a lower bound. Do not call either total system RAM or a proven minimum.
4. Sampling can miss a short peak. Use the kernel high-water mark as the peak; if unavailable, mark the result incomplete rather than promoting the sampled maximum.
5. A subprocess killed by memory pressure, signal, nonzero exit, serialization failure, or missing `metadata.json` completion record is a failed measurement. Record exit/signal and refuse a success result. The serializer atomically publishes the directory with `metadata.json`; it does not create `complete.json`.
6. File-system block allocation, apparent file size, directory metadata and cache-directory naming differ. Record both logical file bytes and allocated blocks for every persistent output, and distinguish them from the transient peak disk requirement.
7. Process monitoring itself consumes resources and concurrent workloads can affect wall time or OOM. Sample only the isolated solver child; record host, Python/numerical versions, load context and elapsed time. Avoid simultaneous heavy local processes where possible.
8. A cold single-table resource profile does not prove a full cold campaign fits the same envelope. Describe the measured scope and a conservative suggested RAM capacity; do not label an untested capacity the minimum.
9. A prior refinement pass flag could be stale or partial. A refreshed practical record must contain 18 families, both modes, the prescribed 3x3 public parameter grid, comparisons across every remaining horizon 1..300, successful final-axis rules, and the prescribed extension. Preserve the old output; write refreshed evidence to a new path.

## Acceptance artifact

`outputs/verification/reaudit/resource_envelope.json` records the exact command, build, child status, kernel high-water RSS, sampled RSS, elapsed time, output sizes, and table metadata. The persisted cold control artifact remains in an ignored cache directory; its specification and hashes support a repeatable check. `report/resource_envelope.md` translates these measurements into a practical reproducibility envelope, explicitly separating tested observations from suggestions and from a full cold campaign.
