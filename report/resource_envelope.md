# Practical control resource envelope (PER-503, 24 September 2026)

The canonical active control table at the public benchmark's central model was **built in a fresh Python process, without a cache read**, on a 16 GiB Apple Silicon Mac. The measured peak process resident set size was **3,628,892,160 bytes (3.380 GiB)**. Solving took 17.99 s; the child completed solve plus atomic serialization in 19.09 s (19.50 s including process startup and monitoring). The saved authenticated table occupies **611,250,986 logical bytes (582.93 MiB)**, or 611,262,464 allocated bytes on this filesystem. The solver reported 46,148,150 sparse transition nonzeros and 554,708,536 bytes of transition storage; its `values`, `q_values`, and beliefs arrays total 611,242,216 bytes in memory.

The exact run was `theta=.35`, `kappa=.02`, `T=300`, `Qmax=5`, 641 belief points, 161 Gauss-Hermite nodes, and `mode=active`. The build was macOS arm64, CPython 3.12.14, NumPy 2.5.3, SciPy 1.18.1. The control-source fingerprint was `c90b00d6369c3da0626bd1378062a94badbb9217e97d2f306f31be5ec849d2af`; the saved artifact fingerprint was `25a4a541b674c0c458643055b04dd26d40f2aef68a891d33e1af13cfd5281861`. Full source-file and numerical-build fingerprints, child status, file sizes and command are in [`resource_envelope.json`](../outputs/verification/reaudit/resource_envelope.json). The profiler then loaded the saved table read-only and validated its completion record and array hashes.

The peak is the kernel's `wait4`/`ru_maxrss` high-water **resident process memory** on Darwin, measured for the child that both solved and serialized this single table. A separate `ps` sample approximately every 50 ms observed 3,628,564,480 bytes; sampling can miss a brief peak, so the kernel value is the headline. The number is neither total system use nor a tested minimum capacity. It may change with numerical libraries, source, operating system and concurrent jobs. The independent reviewer's 4 GiB container OOM is therefore consistent with this close-to-limit process footprint, but the environments are different and this run does not establish a universal 4 GiB threshold.

For reproducibility, **use a dedicated machine with 16 GiB RAM and 30 GiB free disk as a conservative working recommendation**, serializing heavy control jobs. This Mac had 16 GiB physical RAM and ample free disk for the measured table. The recommendation is not a measured minimum and has not been tested at lower capacities. The already completed corrected full campaign's cache directory occupies approximately 12.54 GiB allocated (`outputs/control_cache/corrective_full_0620e68`, 29 artifacts), and its `outputs/full` directory approximately 126 MiB. Atomic cache publication also needs room for a new table's temporary files. The cold single-table profile does not measure the whole campaign's peak RAM, temporary disk high-water or wall time; the corrected campaign's own manifest remains the source for its recorded full-run runtime. These regenerable cache directories are excluded from the review package.

Repeat the cold table profile from the repository root with a **new, nonexistent** artifact and output path:

```sh
.venv/bin/python verification/profile_resources.py \
  --artifact outputs/control_cache/reaudit_canonical_cold_NEW \
  --output outputs/verification/reaudit/resource_envelope_NEW.json
```

The profiler launches a new child and calls `solve_control` directly, then `save_control`. A reused artifact path is rejected; no warm-cache result can masquerade as this cold build. The failure protocol was recorded before the harness in [`resource_failure_modes.md`](../verification/resource_failure_modes.md).

The **refreshed practical numerical validation** ran the existing `verification/verify_control.py` on the same local numerical stack, without reading control caches:

```sh
PYTHONPATH=src .venv/bin/python verification/verify_control.py --practical \
  --output outputs/verification/reaudit/practical_refinement.json
PYTHONPATH=src .venv/bin/python verification/verify_control.py --practical-extra \
  --output outputs/verification/reaudit/practical_refinement.json
```

The initial 18-family sweep took 218.23 s and, as expected from the retained failure history, left `theta=.65, kappa=.002, active` short of its stopping rule. The declared extension took 112.62 s. Its final record passes every final-axis rule for all **18 model/mode families** and contains comparisons at every remaining horizon **1–300** in each of 61 comparison records. It recommends common settings 641 beliefs/161 nodes and the public-model override of 1281 beliefs/161 nodes for `(.65,.002)` in active and noinfo modes. The full refreshed record is [`practical_refinement.json`](../outputs/verification/reaudit/practical_refinement.json), SHA-256 `da4c464e2bf82d3fa130397b366d24811d934f675f7cc616d9d182bab70f0e9e` (13,394,054 bytes). An independent structural check confirmed the exact 3×3×2 family set, all 300 horizon entries per comparison, and final passes; the 18 final settings and pass statuses agree with the earlier published refinement. This is a recalculation of numerical convergence, **not** a full cold replay of the 189,000-record evaluation campaign and not an independent new economic sample.
