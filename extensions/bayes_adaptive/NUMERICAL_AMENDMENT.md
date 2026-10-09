# Numerical-only amendment before final evaluation

Recorded 25 September 2026 (Europe/Paris), after the 17x33x33 and 25x49x49
refinement comparisons, before any final namespace210 economic outcomes.
The original statistical protocol is frozen at commit 8db6208; config.json remains
byte-identical, including its model, hypotheses, priors, pilot/evaluation budgets,
comparators, thresholds, streams and action-witness design.

The frozen-model planning ablation is the limiting numerical component. At GH25,
17x33x33 to 25x49x49 changes its maximum probed Q by 0.0187719673 and initial
value by 0.0026526305. These fail the original 0.005 and 0.001 tolerances.
The BA recursion is closer to resolution, and independent quadrature passes.
The 33x65x65 level is being computed. No threshold is relaxed.

If the original ladder remains unresolved, extend only the joint-belief ladder
by one level: **49x97x97**. Retain every original comparison and use the same
independent-axis selection rule, all 30 horizons, fixed probes, six value modes,
three action-score modes, robust-action rule and quadrature ladder. Stop at the
first jointly passing level. If this additional level fails, explicitly retain
the unresolved status and use the proved/checked fallback described in PROTOCOL.

This amendment pursues more numerical accuracy after diagnosing a failure; it
does not select a method from evaluation PnL. The existing 720-row development
smoke was used only to verify the pipeline and is not evaluation evidence.
Its outcomes did not select the resolution or change the registered hypothesis.
The explanatory states and their ordering were already fixed independently.

`numerical_config.json` is an exact JSON-value copy of config.json except for
that appended grid. `refine_numerical.py` validates this restriction and calls
the unchanged solver. Its receipt binds both actual numerical configuration and
original statistical configuration, the amendment, wrapper, solver, build and
table content. The evaluated source still uses the original statistical config.
This record is a transparent protocol amendment, not part of the original freeze.

Feasibility estimate before execution: 461,041 belief states; 66.888 GB retained
arrays per quadrature setting, plus roughly 40 GB temporary full-update kernels.
Arrays/kernels are memory mapped. The two main quadrature settings may take 1–3 h
on this 16 GiB machine due to disk traffic. These are resource estimates, not
claimed timings. Preserve enough disk/time for independent and cold validation.
