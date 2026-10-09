# Reproduce the study

All commands run from the repository root. Synthetic experiments require
no API key, external data or GPU after dependency installation.

## Install

Python 3.12 is the recorded interpreter. Each run records its numerical build.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

For a smaller smoke installation, use `python -m pip install -e .` alone.

## End-to-end reproduction

```bash
python verification/verify_pipeline.py --out outputs/reproduction/pipeline
```

Two fresh subprocesses run separate cold caches. Acceptance checks exact equality
of 672 episode records, 21 tables per run, accounting, admissibility, liquidation,
policy/stress coverage and pilot/posterior resets. Evidence is saved as `first/`,
`repeat/` and `verification.json`. Choose a new destination for each run.

GitHub Actions runs the same experiment and exports its evidence artifact.
A single smoke run is also available:

```bash
python -m trade_learning.run --profile smoke --out outputs/reproduction/smoke
```

## Reconcile the recorded campaign

```bash
python verification/verify_campaign.py --root . --out outputs/reproduction/campaign-audit.json
```

The independent reader validates 189,000 policy records, pilot identities,
paired keys, economic decomposition, uncertainty estimates and report exports.
The recorded campaign predates the namespace change. Its source identity is
checked against the explicit migration map in `docs/publication-manifest.json`.
Recorded data reconciliation is distinct from a new simulation.

## Full simulation

```bash
python -m trade_learning.run --profile full \
  --out outputs/reproduction/full \
  --cache outputs/control_cache/reproduction_full
```

Use unused output/cache paths. The full design includes nine parameter pairs,
ten pilots per environment, seven policies and three variants. Fresh manifests
bind current source, benchmark, seeds, numerical build and control arrays.

One recorded canonical solve used 3.380 GiB peak process RSS and produced a
582.93 MiB table. That measures one table. A dedicated 16 GiB RAM host with
30 GiB free disk is a conservative working budget; solve heavy tables sequentially.

## Bayes-adaptive extension

Read the [extension guide](extensions/bayes_adaptive/REPRODUCING.md).
Restore its large gzip from four authenticated chunks:

```bash
python extensions/bayes_adaptive/data_transport.py reconstruct
```

Every chunk and the complete gzip SHA-256 are checked. The extension has a separate
horizon, support, seed namespaces, numerical protocol and economic criterion.
Its observations must not be pooled with the core study.

## Build the PDFs

Sources are `report/Market_Making_Study.tex`, `report/Market_Making_Appendix.tex`,
`extensions/bayes_adaptive/report/Bayes_Adaptive_Extension.tex` and
`extensions/bayes_adaptive/report/Mathematical_Appendix.tex`.

From `report/`, with Tectonic installed:

```bash
tectonic --untrusted --outdir ../output/pdf Market_Making_Study.tex
tectonic --untrusted --outdir ../output/pdf Market_Making_Appendix.tex
```

Compile the extension sources from `extensions/bayes_adaptive/report/`.
Generated numerical narratives and scientific figures are shipped with them.

## Evidence scope

A smoke pass establishes installation and integration, not full numerical precision.
Saved-data reconciliation establishes accounting and statistical consistency.
Exact theoretical claims, numerical error and sampling uncertainty remain
separate. See [EVIDENCE.md](EVIDENCE.md) for the acceptance record.
