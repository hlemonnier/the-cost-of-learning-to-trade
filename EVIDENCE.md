# Evidence and provenance

The core and extension are distinct synthetic studies. Model assumptions,
sampling uncertainty and numerical error remain separate.

| Study | Recorded data | Primary result |
|---|---|---|
| Core hidden-regime experiment | 189,000 policy records; 27,000 paired episodes; 90 pilots | Active − matched noinfo: +1.39256132, 95% interval [1.29533830, 1.48978434] |
| Matched mechanism experiment | 140,000 policy records on 20,000 paired episodes | Four interventions separate forecasting, inference, inventory continuation and future feedback |
| Joint-belief extension | 900,000 records on 60,000 paired trajectories | Bayes − weighted Q: +0.00449837, 95% interval [-0.00106562, 0.01006236]; the 0.002 criterion is not met |

Core statistics are in `report/generated/report_claims.json` and
`outputs/full/summary.json`. Extension statistics are in
`extensions/bayes_adaptive/outputs/full/summary.json`.

## Source and data integrity

Raw ledgers, pilots, tapes, actions, statistical tables and scientific plots are
retained byte-for-byte. Four authenticated chunks transport the extension gzip.

Publication changes names, document navigation, report headings and presentation.
`docs/publication-manifest.json` records original/published source hashes,
the original aggregate core identity and retained numeric asset hashes.
The independent campaign reader verifies published bytes against this explicit
migration map before using an earlier source fingerprint.

Fresh runs generate their own manifests. Current source/build checks reject
stale caches; recorded identities are not relabelled as fresh execution evidence.

## Publication acceptance

The contract was written before changes in
`verification/publication_failure_modes.md`. Evidence under `outputs/publication/`:

- `pipeline/verification.json`: installed-package cold repeat and exact episode equality.
- `campaign-audit.json`: independent full core ledger and statistical reconciliation.
- `extension-math.json`: independent joint-belief mathematical checks.
- `acceptance.json`: content, links, PDFs, source migration and data preservation.

GitHub Actions repeats the E2E experiment and exports its artifacts. Final
publication acceptance also uses a fresh GitHub clone.

The checks do not rerun the expensive full campaigns. The extension's primary
meaningful-improvement criterion remains unmet; no real-market profitability
or small global optimality-gap certificate is established.
