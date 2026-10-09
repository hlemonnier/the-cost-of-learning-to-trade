# Numerical resource diagnosis

Observed 2026-09-27T19:53:28.333232+00:00 while worker 22987 continued the frozen
33 x 65 x 65 / GH25 family. This is a read-only performance diagnosis, not
numerical acceptance. Source and geometry-amendment hashes were unchanged.

- The current Bayes operator contains 69 CSR kernels / 207 NPY files:
  13,347,074,220 payload bytes (12.430 GiB), compared with
  4.886 GiB at 25 x 49 x 49 / GH25.
- Successive Bayes horizon median: 33.647 s
  across 13 observed intervals, versus
  3.765 s across 29 intervals at 25 x 49 x 49.
- The recorded disk0 samples are 424.26, 449.75, 280.98 MB/s,
  **host-wide**, not worker-exclusive. A single full operator sweep at that
  displayed throughput would take about
  31.5 s.
  This is an explanatory estimate, not a measured per-worker I/O time.
- `vmmap` excerpts, before/after CPU/RSS and host VM counters, exact file sizes,
  and timestamps are retained in `resource_diagnosis.json`. RSS includes resident
  file mappings; large virtual mapped files do not imply equally large RAM use.

The code traverses every CSR once per horizon. The operator's size is consistent
with exceeding available file-cache capacity on this shared 16 GiB machine. The
observed disk traffic and memory compression support this explanation for the
sharp slowdown. The active array shapes imply a 526.5 MiB three-score result,
175.5 MiB rewards, and two roughly 47.9 MiB continuation/expectation arrays,
plus selection temporaries; those estimates exclude resident file-cache pages.
Q histories are flushed and advised discard after each completed slice. Full V
histories and all current-mode kernels remain mapped. The observed mapping size
is consistent with the current family. **This does not prove the absence of a
memory leak.**

Keep this frozen run unchanged and avoid concurrent memory-intensive jobs. Do
not purge its useful cache, delete mapped files, or change source mid-run. The
actual 16 GiB host supports execution but is I/O-limited at this mesh. For faster
reproduction, recommend **at least 32 GiB RAM** to give the operator and active
arrays room in cache; this is a resource recommendation, not a measured speedup.
The cold rebuild must record its own runtime, memory, disk and exact array checks.

A later source-only optimization could benchmark bounded score buffers and
retirement of completed V pages. Merely opening and closing each CSR cannot
eliminate rereading the complete operator. Preserve summation/tie ordering and
all eleven arrays, and require new source/cache identities and full equivalence
checks before using any revised implementation. No such edit was made here.
