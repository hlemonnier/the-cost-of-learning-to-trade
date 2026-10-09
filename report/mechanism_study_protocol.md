# Matched mechanism study — frozen before implementation and evaluation

This supplementary study closes PDF page 1 and §5.3, tracked by PER-506. It uses
the prescribed **known-parameter reduced model**: theta .35, kappa .02, T=30,
inventory bound 2, initial q=x=0 and belief .5. The canonical 189,000-record
finite-pilot campaign, its primary endpoint and its 162-contrast family are
unchanged. This is new synthetic evaluation, not new training or a parameter
learning claim. Negative or zero treatment effects are acceptable findings.

## Decision interventions

Write r for the exact immediate marked-wealth reward less the pre-decision risk
charge, G(q)=-.027|q| for terminal liquidation, psi(b)=.02+.96b, and V^N for the
existing noinfo planner. All policies retain the same actual simulator, public
signal and selected observations, admissible actions, numerical tie rule and
terminal cash accounting. No policy receives latent regimes or future shocks;
the separately labelled full-information reference alone observes current H.

1. **Forecast use:** compare filtered noinfo with `no_forecast`. Solve the same
   noinfo recursion with only the planning reward changed to
   r^0=r-.03x E[q'|x,a] at every remaining horizon. Keep the true drift .03 in
   simulation and in return centering for the filter; keep signal-dependent fill
   probabilities, the signal transition K, regime dependence, constraints, risk
   and terminal G. This measures use of the specified predictive drift, not the
   value of hiding the whole public signal or re-estimating a forecasting model.
2. **Current regime inference use:** compare filtered noinfo with `blind_regime`.
   Use the identical V^N/Q^N table, but pass unconditional belief .5 to the latter
   at every decision. Because the symmetric regime starts at stationarity,
   psi(.5)=.5. The blind rule discards its own return/fill association when
   deciding; it still observes q and x and assumes the true nonzero theta.
   Neither planner values future feedback. The treatment includes the consequent
   change in action-selected evidence, rather than a hypothetical free signal.
3. **Action-induced inventory continuation:** compare filtered noinfo with
   `inventory_off`. At m>=2 use
   Q^I_m(q,x,b,a)=r(q,x,b,a)+sum_{x'}K_xx' V^N_{m-1}(q,x',psi(b)),
   replacing only the post-execution q' argument by current q in the **same**
   continuation function. At m=1 use exactly Q^N_1, including E[G(q')].
   The real inventory still changes and is charged/liquidated normally. This
   is a deliberately ablated decision score, not an optimal alternative-world
   Bellman solution. It ignores the action's inventory consequence in future
   planning before the real final step; it does not move liquidation earlier.
   The exact score deletion is
   D_m=E[V^N_{m-1}(q',X',psi(b))-V^N_{m-1}(q,X',psi(b))].
   Since X' and psi(b) do not depend on the action, this is the entire
   action-dependent continuation under noinfo. Future signal opportunities
   mediated by inventory are part of this treatment; it is not a pure risk-cost
   effect or a signal process change. The existing myopic rule, which includes
   E[G(q')] at every decision, is retained as a separate descriptive comparator.
4. **Future information acquisition:** compare active with filtered noinfo.
   Only feedback conditioning in the planning recursion changes; current filter,
   drift use, inventory dynamics and economics are matched. Retain the existing
   same-active-continuation negative-immediate-reward witness. The fresh study
   provides an independent reduced-model evaluation of this already fixed pair.

These are four one-at-a-time policy interventions around filtered noinfo. They
are not four noninteracting components of total PnL. No additive attribution,
universal inference premium, exact optimality or commercial profitability is
claimed. The noinfo planning value is not the value of its filtering runtime
policy. No-forecast and inventory-off Q scores are not the true-policy values.

## Frozen budgets, streams and inference

Evaluate seven policies (`active`, `noinfo`, `no_forecast`, `blind_regime`,
`inventory_off`, `myopic`, `full_information`) on **20,000 new paired independent
episodes**, exogenous root seed **260924506**. The generator spawns independent
signal, regime, return and side-noise streams as in the original simulator.
This root is separate from canonical seed 260924138 and reduced seed 260924301.
There are no pilots, fitting or tuning. All episodes reset each policy's belief
and ledger. Record all seven policies, all 140,000 outcome rows, complete action
arrays, tape/array/source/build hashes, action frequencies and accounting checks.
Retain small public decision traces for independent filter/score inspection.

The four fixed contrasts, in order, are noinfo minus no_forecast (forecast),
noinfo minus blind_regime (current inference), noinfo minus inventory_off
(inventory continuation), and active minus noinfo (future information).
The endpoint is expected penalised objective per 30-decision episode. Report
the paired mean, sample SE across 20,000 independent **episode differences**,
df 19,999, ordinary two-sided Student 95% interval and a simultaneous-family
interval using t_(1-.05/(2*4),19999). These are approximate Monte Carlo intervals,
not rigorous concentration guarantees. Policy net PnL, inventory exposure and
myopic/full-reference outcomes are descriptive. Do not enlarge the statistical
family with post hoc significance claims or change budgets in response to sign.

## Numerical and end-to-end acceptance fixed before code

Cold-build at 321 beliefs and 161 Gauss-Hermite nodes. For active, noinfo and
no-forecast recursions, and the inventory-off decision scores, check separately
161->321 beliefs at 161 nodes and 81->161 nodes at 321 beliefs. Check every
remaining horizon 1..30 on shared grids: initial score/value change <=.001,
maximum common-state score/value change <=.005, and <=.5% action disagreement
among fine-score gaps >.001. Raw disagreements and maximum action loss must be
retained. If a criterion fails, preserve the failure and double-minus-one that
axis until the same rule passes; all policies use a common resulting resolution.
This is an empirical stability rule, not a certified approximation error bound.

Before accepting rollout numbers, verify the protocol and all array structures;
validate reward-drift deletion independently; verify inventory D with direct
fill enumeration and a nonconstant synthetic continuation; require exact
last-step agreement and zero D for inventory-independent continuation; verify
runtime belief routing, selected observations and true return centering. Retain
same-state action changes and score decompositions as illustrations, not tests
chosen for significance. The pre-implementation failure catalogue is
`verification/mechanism_failure_modes.md`.

Run the entire study in two separate processes with no shared control cache.
Outcomes and action/content hashes must match on this stack (runtime timings
may differ). Independently reconstruct all four contrasts and accounting from
raw outcomes, reject deliberately incomplete/corrupt inputs, and independently
replay the stored actions through a scalar cash ledger. Save repeatable evidence.
The active and full-information means must agree with their correctly matched
planning values within four Monte Carlo SE; the blind-regime mean should likewise
match the no-learning planner's .5 initial value within four SE. These checks
are model/implementation checks and are not gates requiring active to win.

## Reproduction

From the project root, after installing the pinned dependencies:

```sh
PYTHONPATH=src .venv/bin/python verification/mechanism_study.py --out tmp/mechanism-fresh
PYTHONPATH=src .venv/bin/python verification/verify_mechanism_study.py --study tmp/mechanism-fresh --out tmp/mechanism-independent.json
```

The versioned accepted output resides in
`outputs/verification/reaudit/mechanism_study/`. Regenerable Q arrays and temporary
repeats stay outside version control. The result manifest binds this protocol
and the authoritative JSON settings to their exact bytes.
