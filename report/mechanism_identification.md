# Mechanism identification: matched policy interventions

The original PDF asks to separate forecasting, inventory planning, current-regime
inference and deliberate acquisition of future information. The initial report's
noinfo–myopic gap and local arguments did not identify all four policy effects.
This supplementary study implements the missing comparisons in the prescribed
reduced model: \(T=30, |q|\le2, \theta=.35, \kappa=.02\).

The [protocol](mechanism_study_protocol.md) and
[failure catalogue](../verification/mechanism_failure_modes.md) were committed at
ac7b70b before implementation or results. The separate seed 260924506 supplies
20,000 independent episodes shared by seven fixed policies: 140,000 policy
records, with no fitting, tuning or pilot uncertainty. This new experiment does
not replace or pool with the original 189,000-record finite-pilot campaign.

## Four matched comparisons and results

All rules act in the **same true simulator**, with the same cash accounting,
inventory constraints, fees, horizon, terminal liquidation and observable public
signal. The four effects below are paired differences in penalised objective per
episode. Brackets give approximate **simultaneous 95% Student/Bonferroni intervals
for the four prespecified contrasts**, with 19,999 degrees of freedom.

| Channel | Changed decision input or operation | Effect and simultaneous interval |
|---|---|---:|
| Predictive drift use | Filtered noinfo minus no-forecast; remove only \(.03xE[q']\) from every planning reward, keeping signal-dependent fills, transitions and correct filtering. | +0.190707 [0.156094, 0.225320] |
| Current regime inference use | Filtered noinfo minus blind-regime; identical noinfo tables, but the blind rule routes belief .5 into every decision. It keeps nonzero true dependence and does not use its shadow filter. | +0.241601 [0.214178, 0.269023] |
| Action-induced inventory continuation | Filtered noinfo minus inventory-off; replace only \(q'\) by current \(q\) in the same continuation before the actual last step. Keep the last-step scores exactly equal. | +0.085447 [0.056881, 0.114013] |
| Future information valuation | Active minus filtered noinfo; only feedback conditioning in future planning changes. Both update current beliefs from their own selected observations. | +0.026197 [0.009232, 0.043161] |

Each interval is above zero on this fixed synthetic configuration. This supplies
four distinct operational policy effects. **They are not additive components of
PnL**, and they do not estimate a universal premium for each mechanism. Policy
actions subsequently change which evidence and inventories are encountered.
The first three comparisons are anchored at the same filtered noinfo policy;
only the fourth changes future-feedback valuation.

## Exact changes to the decision rules

Let \(r\) be the expected current marked-wealth increment less the pre-decision
risk charge, \(G(q)=-(h+c_T)|q|\), and
\(\psi(b)=\kappa+(1-2\kappa)b\). The noinfo planning recursion is

\[
Q_m^N(q,x,b,a)=r(q,x,b,a)+
E_b[V_{m-1}^N(q',X',\psi(b))],\qquad V_0^N=G.
\]

The runtime noinfo policy still filters its realised return/fill feedback. Its
stored no-learning planning value therefore differs from its realised objective.

For **forecast use**, solve this same recursion with
\(r^{F0}=r-\mu xE[q'\mid q,x,a]\). A passive action has
\(E[q']=q+\sum_{s\in S(a)}s p_{s,k_s}(x)\); a market order has \(E[q']=q+s\).
The actual simulator and filter retain \(\mu=.03\); in particular,
\(z=(R-.03x)/.30\) remains the observed innovation. The policy still uses \(X\)
for fill probabilities and future public transitions. This comparison measures
the use of the specified predictive **return drift**, not the value of hiding
the entire signal or fitting an unknown forecasting model.

For **current inference**, the enabled rule selects using \(Q_m^N(q,x,b_t,a)\)
and the blind rule uses \(Q_m^N(q,x,.5,a)\). The symmetric regime begins at its
stationary prior, so \(\psi(.5)=.5\). The blind rule's shadow posterior is saved
for verification but never routed into decisions. This comparison includes the
economic consequences of using evidence the policy has actually acquired.
It does not assume information was free, or force two diverging policies to have
identical histories. It does not change the execution model to \(\theta=0\), as
the older independence baseline does.

For **inventory continuation**, use exactly the same \(V^N\) and define

\[
Q_m^{I0}(q,x,b,a)=
\begin{cases}
Q_1^N(q,x,b,a),&m=1,\\
r(q,x,b,a)+E[V_{m-1}^N(q,X',\psi(b))],&m\ge2.
\end{cases}
\]

Before the last step, this deletes precisely

\[
D_m(q,x,b,a)=E[V_{m-1}^N(q',X',\psi(b))-
V_{m-1}^N(q,X',\psi(b))].
\]

Neither \(K\) nor \(\psi(b)\) depends on the action. Consequently, all
action-dependent future continuation in noinfo is mediated by the change from
\(q\) to \(q'\). The marginal selected-fill vector is also sign-invariant:
substituting \(z\mapsto-z\) exchanges the two regime likelihoods under the
symmetric normal density. This includes joint two-sided fills.

The ablated score intentionally omits the effect of today's action on inventory
supplied to continuation. It is **not** the optimum of an alternative physical
market. The simulator still changes inventory, enforces the same safe actions
and charges the same current risk; the rule observes the real inventory again
next period. At the last decision it retains \(E[G(q')]\) exactly, with one real
terminal liquidation. Before then its action-independent continuation means it
maximises immediate reward. This differs from the existing myopic rule, which
includes a hypothetical immediate liquidation at every decision.

Future signal opportunities that depend on inherited inventory remain part of
\(D_m\). The measured effect is **multi-period use of action-induced inventory
consequences**, including interactions with forecasts, constraints and future
executions. It is not a pure contribution of \(\lambda q^2\), nor a change in
the public-signal process. This exact common-continuation construction removes
the earlier ambiguity of a generic noinfo–myopic comparison.

For **future information**, active replaces \(\psi(b)\) in the planning
continuation by \(\psi(B)\), where \(B\) is the feedback posterior. The original
controlled witness remains: at \((m,q,x,b)=(30,0,0,.75)\), both deep quotes have
immediate expected reward \(-.00829323\); suppressing current feedback valuation
with the very same active continuation instead selects abstention, with an
active-Q gap .00349678. This is a state illustration selected after solving,
not an extra statistical test.

## Policy outcomes and validation

| Policy | Mean objective | Monte Carlo SE | Mean net PnL |
|---|---:|---:|---:|
| Active | 0.990439 | 0.016146 | 1.071387 |
| Filtered noinfo | 0.964242 | 0.016480 | 1.052758 |
| No forecast | 0.773536 | 0.013283 | 0.813878 |
| Blind regime | 0.722642 | 0.016060 | 0.788921 |
| Inventory off | 0.878795 | 0.018261 | 0.993802 |
| Myopic with immediate liquidation | 0.839049 | 0.013645 | 0.886818 |
| Current-regime full-information reference | 1.459878 | 0.015778 | 1.544889 |

The active, blind-regime and full-information means agree with their correctly
matched planning values within 0.77 Monte Carlo SE. No such calibration claim is
made for filtering noinfo, no-forecast or inventory-off planning scores.
The [raw summary](../outputs/verification/reaudit/mechanism_study/summary.json)
also reports ordinary 95% intervals, fees, liquidation, inventory exposure and
all eleven action frequencies. Raw episode rows and complete selected-action
arrays allow every comparison and ledger to be checked.

Cold numerical builds use 321 beliefs and 161 integration nodes. Separate
161-to-321 belief and 81-to-161 quadrature comparisons pass the unchanged reduced
thresholds at every horizon for active, noinfo, no-forecast and inventory-off
scores. These are empirical stability checks, not certified bounds. Independent
fill enumeration checks 1,845 inventory-deletion cases with maximum error
\(8.2\times10^{-16}\); the drift identity error is below
\(2.0\times10^{-17}\). Exact last-step equality, a nonconstant synthetic
continuation and zero effect for inventory-independent continuation also pass.
The comparisons cover all 144,450 grid states/horizons; aggregate disagreement
counts, maximum score losses and selected illustrations are retained.

Two fresh processes sharing no control cache reproduce all saved episode,
action, trace, summary and numerical-check bytes and all table fingerprints on
the recorded stack. The standalone independent verifier imports no production
simulator, filter or statistical code: it regenerates paired tapes, replays all
4.2 million selected decisions from the cash-flow equations, reconstructs the
2,520 saved public decision traces and filters, and recalculates all four
intervals. Its corruption probes reject incomplete, stale or altered inputs.
These are implementation-team checks, not a new external reviewer sign-off.

## Earlier evidence retained separately

The canonical nine-environment active-minus-noinfo primary remains +1.3925613223,
pilot-aware approximate 95% interval [1.2953383014, 1.4897843432]. Its corrected
replay uses the original streams and adds no independent sample. The new reduced
study uses a separate root seed and known parameters; it cannot expand the
canonical deployment population or establish persistent parameter learning.

The earlier nominal noinfo-minus-myopic contrast remains **post hoc and
descriptive**: +2.4774821599 [2.2825822518, 2.6723820680] across 90 fitted pilots.
It changes continuation and internal liquidation timing together and is not the
inventory treatment defined above. Its specification, results and independent
arithmetic remain in Git history and the mechanism_contrasts.json artifact
alongside the new study; it is outside both prespecified contrast families.
Local drift sensitivities and the hypothetical free-signal Jensen argument remain
valid analytical context, not replacements for the implemented interventions.
