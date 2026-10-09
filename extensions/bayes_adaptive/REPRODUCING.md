# Reproduce the joint-belief study

Install the package from the repository root using [the main guide](../../REPRODUCING.md).

## Restore the recorded ledger

From the repository root:

```bash
python extensions/bayes_adaptive/data_transport.py reconstruct
```

Four chunks and the complete gzip are authenticated against
`data_transport/manifest.json`. This restores the recorded 900,000-policy-record
campaign; it does not run a new simulation.

## Independent mathematical checks

```bash
python extensions/bayes_adaptive/verify_extension.py --math-only \
  --out outputs/reproduction/extension-math.json
```

The verifier implements its observation, filtering and mathematical oracle
independently of the production market, ledger and statistics modules.

## Fresh control and evaluation

```bash
python extensions/bayes_adaptive/solver.py --help
python extensions/bayes_adaptive/run.py --help
```

Generate control tables in a new cache and run smoke evaluation in an unused
destination. Full evaluation requires numerical acceptance for those exact tables;
a successful coarse solve is insufficient.

The selected recorded specification is 33 × 65 × 65 / GH25 with endpoint-sine
belief geometry. Large control arrays are regenerable and excluded from Git.
Recorded statistical outcomes are in `outputs/full/summary.json`.

## Recorded versus fresh identities

[The evidence guide](../../EVIDENCE.md) and
[publication manifest](../../docs/publication-manifest.json) document source
migration. Recorded evidence belongs to its original numerical build. A new
run binds its own source, arrays and outputs.
