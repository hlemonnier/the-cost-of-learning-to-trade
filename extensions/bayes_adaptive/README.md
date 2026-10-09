# The price of assuming you will know

A separate synthetic study of **joint model-and-regime planning**.

When a model remains uncertain, are posterior-weighted known-model Q values a
good decision rule? The solver maintains a joint belief over two persistence
models and two hidden regimes, and compares that recursion with matched
planning interventions.

Read the [paper](report/Bayes_Adaptive_Extension.pdf), the
[mathematical appendix](report/Mathematical_Appendix.pdf), the
[frozen protocol](PROTOCOL.md) and the [reproduction guide](REPRODUCING.md).

## Primary finding

With zero pilot episodes, joint-belief planning minus weighted Q has mean
**+0.00449837 objective units per 30-period episode**, with paired 95% interval
**[-0.00106562, +0.01006236]**. The prespecified requirement that the lower
endpoint exceed **0.002** is not met.

The campaign retains 900,000 policy records on 60,000 paired trajectories,
including pilots, tapes, every action and selected posterior traces.
Sampling intervals and numerical approximation error are reported separately.

## Model and controls

The model fixes theta at 0.35 and uses kappa in {0.002, 0.10}, a 30-period
horizon and inventory cap 2. Feasible policies observe the same selected feedback:

- `bayes`: joint-belief Bellman planning.
- `weighted_q`: posterior average of known-model Q values.
- `no_feedback`: removes future feedback valuation, with runtime inference retained.
- `frozen_model`: fixes model weights during planning, with runtime inference retained.
- `known_parameter`: evaluator reference with the true model and a hidden regime.

The selected resolution is 33 × 65 × 65 with GH25 integration and endpoint-sine
belief geometry. Numerical relaxation values are approximations; no small
global optimality-gap certificate is claimed.

## Files

| Path | Role |
|---|---|
| [THEORY.md](THEORY.md) | Observation law, joint posterior and revelation inequalities |
| [SOLVER_DESIGN.md](SOLVER_DESIGN.md) | Numerical representation and failure modes |
| [solver.py](solver.py) | Joint-belief control solver |
| [run.py](run.py) | Paired pilot/evaluation experiment |
| [verify_extension.py](verify_extension.py) | Independent mathematical and raw-data reader |
| [outputs/full/](outputs/full/) | Complete outcomes, tapes, pilots, actions and summaries |
| [data_transport/](data_transport/) | Exact-byte chunks for the compressed ledger |

The study has its own seed namespaces and economic criterion. Its observations
do not enlarge the core experiment's sampling population.
