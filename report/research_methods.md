# Supplementary implementation and statistical methods

## Finite-pilot estimator and practical policy

The public parameter support is the Cartesian product of the three prescribed dependence magnitudes and three switching probabilities, with equal prior masses. The learner does not receive the actual environment index. Each of 100 pilot episodes starts at hidden prior one half. For candidate \(j\), its likelihood is the product of successive predictive selected-fill likelihoods conditional on the genuinely observed returns. Summing log likelihoods over 100 independent episodes and normalizing yields the pilot posterior \(w_j^{\rm pilot}\). There is no hyperparameter search or separate hidden training simulator. All 100 episodes fit this prespecified estimator; none is silently added as validation data.

At each evaluation episode, initialize \(w_j=w_j^{\rm pilot}\) and \(b_j=1/2\). The joint update in the mathematical appendix is exact on this finite support. Decisions use

\[
\widehat Q_m(q,x,(w,b),a)=\sum_j w_j Q_{j,m}(q,x,b_j,a).
\]

For the active policy, \(Q_{j,m}\) plans future regime observations under known candidate \(j\). For the matched noinfo policy, those future regime observations are suppressed inside planning only. Both policies filter and update parameters after each real observation. The numerical resolution, prior support, estimator, available pilot, horizon, action constraints and risk treatment match. Each candidate's continuation optimizes future actions separately for that model, implicitly allowing future model-specific knowledge. This optimistic relaxation resembles QMDP on the static parameter; hidden regimes remain filtered. It does **not** optimize the future evolution of parameter weights or correctly value parameter identification. It is not full Bayes-adaptive control and no posterior-averaged-Q optimality theorem is claimed.

The original nominal campaign nearly identified the true support point in every pilot, as recorded in the retained pilot diagnostics. Its practical comparison therefore primarily tests learning the changing hidden regime after parameter identification. It gives little evidence about persistent parameter uncertainty, off-grid models or the value of deliberately acquiring parameter information. Each campaign's concentration diagnostics delimit its own claim; numerical reproducibility corrections alone do not establish performance under persistent parameter uncertainty.

The myopic policy uses the same posterior/filter but each model contributes its one-remaining-step Q table: immediate marked wealth, the current pre-decision risk penalty and expected terminal liquidation of post-action inventory. This is an explicitly stated liquidation convention, not a silently changed cash objective. The independence baseline plans with theta zero, so passive sides and returns are conditionally independent given the public signal. The taker baseline solves the full-horizon inventory problem restricted to market buys, market sells and abstention. Both use the same public return model and fees. Abstention starts and remains at zero inventory and earns exactly zero.

The privileged reference alone receives the nominal true parameters and current hidden sign. It never receives future shocks or regimes. Under stresses its nominal tables remain frozen; it is a privileged misspecified policy, not the optimum in the modified environment.

## Public observation boundary and randomness

Policies receive detached public records containing time, horizon, inventory bound, midpoint, signal, cash, inventory and the admissible-action mask. Post-action records contain the chosen action, current signal, observed return, submission mask, selected fills, execution prices/quantities/fees and the next public observation. Unsubmitted fills are -1; submitted non-fills are zero. Primitive current-period noise, latent regimes, unselected outcomes, environment labels and seeds never appear in those records. Read-only defensive copies prevent accidental shared-buffer access; this is a data contract, not a security sandbox against malicious Python code.

Private exogenous tapes allow common random numbers across policies without exposing counterfactual feedback. Within a tape, independent child streams generate public signals, hidden signs/switches, return innovations, each side's noise and fixed-duration phases. Initial hidden signs are equiprobable and every episode resets public state. The two potential depths share one side-specific noise. Pilot behaviour randomization uses a separate namespace and samples a rank among valid actions, logging the reciprocal valid-action count.

The final root is **260924138**. Root 260924137 was used by an early smoke run and retired before any final sampling. Current smoke uses 260924136; reduced-reference rollout uses 260924301. For final fitting/evaluation, SeedSequence receives `(namespace, environment_index, pilot_index, root)` and emits a private tape seed. Namespaces 11 and 12 cover pilot exogenous data and behaviour; 21, 22 and 23 cover nominal, signal-dependent and fixed-duration final evaluation. All 450 final IDs are distinct and disjoint from consumed smoke IDs. Component spawning and namespaces follow NumPy's documented design [parallel RNG documentation](https://numpy.org/doc/stable/reference/random/parallel.html).

The original policy structure and numerical settings were frozen before final sampling. The post-review corrective replay keeps those scientific settings and the original seeds; its fixed numerical tie convention is absolute 1e-12 with the lowest admissible action ID. It is a paired correction of the existing experiment, not newly independent evidence. Both Bellman values and executed controls use the same convention, which was chosen for numerical stability rather than observed PnL. The code hash covers all production Python modules; the run rejects an unnoticed source change during evaluation. The corrected manifest additionally binds the configuration and resolved protocol, plus each evaluated control table's source, numerical build, content hashes, dtype, shape and action-mask structure. Cached bytes and semantic structure are checked before use and rechecked before completion. Two fresh processes with separate cold caches test reproducibility. Unsupported benchmark edits and incomplete strata, pairing, budgets or nonfinite economic outcomes fail closed; at least two independent pilots are required for pilot uncertainty. The fixed full campaign still uses ten. Evaluator manifests containing true parameters are never learner inputs. Public pilot files and posterior fingerprints permit auditing equality of training data across competitors and fresh state across test episodes.

## Paired uncertainty and deployment assessment

For environment \(e\), independent pilot \(j=1,\ldots,M\) and final episode \(k=1,\ldots,n\), let \(D_{ejk}\) be the objective difference between two policies evaluated on the same exogenous tape. Here \(M=10,n=100\). Define

\[
\overline D_{ej}=n^{-1}\sum_kD_{ejk},\qquad
\widehat\Delta_e=M^{-1}\sum_j\overline D_{ej},\qquad
\widehat{\mathrm{SE}}_e^2=S_e^2/M,
\]

where \(S_e^2\) is the sample variance of the ten pilot means. A descriptive two-sided interval is \(\widehat\Delta_e\pm t_{M-1,1-\alpha/2}\widehat{\mathrm{SE}}_e\). This treats pilot datasets as independently sampled units and retains the variability of fitting. It does not pretend that 300 sequential decisions are independent. If the pilot means are non-Gaussian, the Student interval is an empirical approximation, particularly with ten pilots.

To expose the two noise sources, let \(s_{ej}^2\) be the within-pilot sample variance of paired episode differences. The estimated contribution of Monte Carlo noise to a pilot mean is \(W_e=M^{-1}\sum_j s_{ej}^2/n\). Report the observed pilot-mean variance \(S_e^2\), \(W_e\), and the method-of-moments excess \(\max(S_e^2-W_e,0)\). Truncating that component at zero does **not** replace \(S_e^2\) in the reported standard error.

For the fixed uniform mixture of nine environments,

\[
\widehat\Delta=\tfrac19\sum_e\widehat\Delta_e,\quad
v_e=\widehat{\mathrm{SE}}_e^2/81,\quad
\widehat{\mathrm{SE}}^2=\sum_ev_e,\quad
\nu=\frac{(\sum_ev_e)^2}{\sum_ev_e^2/(M-1)}.
\]

The aggregate uses Welch--Satterthwaite Student quantiles with \(\nu\) degrees of freedom. The environment grid is a fixed population with uniform weights, not nine random draws from all markets. Independently generated pilot/test streams across environments make the stratified variance calculation appropriate to this declared population. Policy differences are paired at episode level within each pilot; isolated marginal policy intervals cannot substitute for their paired difference.

The primary comparison was fixed as active versus matched noinfo on the nominal mixture. The acceptance threshold is 0.05 penalised-objective units per episode. Its one-sided 95% lower bound must exceed that threshold, and active net PnL must have a positive one-sided lower bound. This intersection does not select among many winners; it assesses one preselected candidate. Supplemental active-versus-six-comparator contrasts across nine environments and three variants form a family of 162. Their saved tables provide both descriptive 95% intervals and Bonferroni-adjusted intervals using alpha/162. Adjustment addresses multiplicity under the interval approximation; it does not turn that approximation into an exact finite-sample guarantee.

Downside summaries are the empirical fifth PnL percentile, mean of observations at or below it, and loss frequency. They are descriptive tail estimates, not rare-event certificates. Gaussian innovations yield unbounded returns. No bounded-reward concentration result is used. Stress evidence is not used to choose a nominal policy, and nominal confidence says nothing distribution-free about arbitrary new regimes.

## Accounting attribution and monitoring

### Four mechanisms: matched supplementary policy interventions

The original four-part explanatory objective is now tested using a separately
frozen known-parameter reduced study (T=30, qmax=2, theta=.35, kappa=.02),
20,000 fresh paired episodes and seven fixed policies. The
[protocol](mechanism_study_protocol.md) was committed before code/results; the
[identification note](mechanism_identification.md) gives the exact equations,
outcomes, all held-fixed quantities, independent verification and scope limits.

| Mechanism | Enabled versus disabled rule | Paired objective effect; simultaneous approximate 95% interval |
|---|---|---|
| Predictive drift use | Filtered noinfo versus no-forecast: remove only \(\mu xE[q']\) in planning at every horizon, keeping signal-dependent execution, K and the true drift in simulation/filtering. | +0.190707 [0.156094, 0.225320] |
| Current regime inference use | Identical noinfo tables with selected-feedback belief versus unconditional .5; nonzero execution dependence and future-feedback suppression are unchanged. | +0.241601 [0.214178, 0.269023] |
| Action-induced inventory continuation | Same noinfo continuation evaluated at post-execution q' versus current q for m>=2; exact m=1 scores, actual inventory dynamics and terminal liquidation are preserved. | +0.085447 [0.056881, 0.114013] |
| Future information valuation | Active versus filtered noinfo: change only feedback conditioning in planning, with identical current filtering and economics. | +0.026197 [0.009232, 0.043161] |

These are four one-at-a-time intervention effects around noinfo, with a separate
four-comparison Student/Bonferroni family. Their Monte Carlo SEs use independent
paired episodes, without pilot clusters because no parameter fitting occurs in
this reduced study. They do not add to PnL. Forecast use is not the value of hiding
all X; inventory continuation includes interactions with future signal and
execution opportunities, not just the risk penalty. The inventory-off rule is a
deliberately ablated decision score, not a Bellman optimum for another physical
market. Once actions diverge, each policy encounters its own selected feedback.

For passive actions, the exact expected immediate reward is
\[
r(q,x,b,a)=q\mu x-\lambda q^2+
\sum_{s\in S(a)}\{p_s(d_{k_s}-c_P+s\mu x)
-\sigma\theta(2b-1)\phi(\Phi^{-1}(p_s))\}.
\]
The drift deletion is exactly \(\mu xE[q']\). Under noinfo, the next public-signal
law and predicted belief are action independent, so all action-dependent
continuation passes through q'. The inventory comparison removes that argument's
effect using the same continuation function, and keeps actual final-step
liquidation. It is distinct from the older myopic rule's repeated hypothetical
one-period liquidation.

The original future-information witness and full nominal primary remain unchanged:
active/noinfo +1.3925613223 [1.2953383014, 1.4897843432], with pilot-aware uncertainty
for the declared uniform nine-environment population. The corrected original
reduced rollout remains +0.0301076160 [0.0167131963, 0.0435020356] on its original
seed. Neither same-stream correction is a fresh sample; the new seed is used only
by the supplementary study.

The earlier noinfo–myopic continuation comparison remains post hoc and descriptive:
+2.4774821599 [2.2825822518, 2.6723820680]. It includes changed internal liquidation
timing, and is not relabelled as the new inventory treatment. Its independently
checked artifact, mechanism_contrasts.json, is retained. The canonical primary,
162-contrast family, production source and full raw dataset are unchanged.

All per-episode results preserve cash PnL separately from the penalised objective. The exact split is passive spread capture minus passive fees, discretionary market spread/fees and terminal liquidation, plus:

* directional exposure: \(\sum_t[Q_tR_t+\Delta Q_t\mu X_t]\);
* contemporaneous passive selection: \(\sum_t\Delta Q_t^P(R_t-\mu X_t)\);
* market-order innovation: \(\sum_t\Delta Q_t^M(R_t-\mu X_t)\).

This separates known drift exposure from the contemporaneous return/fill relation without claiming every realised component has predictable profit. The independent wealth sum is reconciled against actual cash and marked inventory. Turnover includes forced liquidation; passive/market fills, side counts, action counts, terminal fees, inventory magnitude/squares and maximum exposure are stored separately.

The learner's public residual is

\[
e_t=\sum_{s\in S_t} sF_{t,s}Z_t+
\left[\sum_j w_{j,t}\theta_j(2b_{j,t}-1)\right]
\sum_{s\in S_t}\phi(a_{s,k_s}(X_t)).
\]

The first term uses only the return observed after acting and submitted fills. Under a correct predictive model the full residual has conditional expectation zero. Residual sums and submitted-side counts are stored by signal; pooled ratios are descriptive monitoring summaries. No alarm threshold, false-alarm rate or detection power has been calibrated. Empty quotation periods yield no execution evidence. The privileged reference's public-prediction residual is deliberately unavailable rather than incorrectly centred on rho=0.

The conditional fill log score excludes the common Gaussian return density and includes non-fill evidence. Its level depends on action mix and conditional entropy, so comparing raw scores across policies is not an isolated calibration test. Hidden-label Brier scores use pre-decision predicted sign probabilities. Erroneous confidence counts wrong probabilities above .9 or below .1. Switch-relative error examines the first ten periods after an actual hidden switch, excluding initial unknown phase; its numerator and denominator are saved for weighted aggregation. These label-based measures are retrospective evaluator diagnostics, not deployable monitors.

Both stress variants preserve the stated return model and marginal fill functions, which are checked empirically by signal, side and depth. The state-dependent copula violates the learner's constant-magnitude assumption. Fixed durations violate memoryless switching and also replace geometric mean complete-spell length \(1/\kappa\) by 50 periods. For nominal \(\kappa=.002,.02,.10\), those means are 500, 50 and 10 periods respectively; the hidden uniform phase also determines the initial residual spell. Nominal kappa remains only a training label.

For the fixed-duration stress, a nominal-to-stress PnL or objective difference combines a changed opportunity process, duration-shape mismatch and the controller's response. Except at \(\kappa=.02\), it also changes the mean switching rate. It is descriptive evidence of robustness limits, not an isolated estimate of avoidable filtering loss or the effect of non-geometric shape. Hidden-label errors can establish miscalibration without identifying how much of the economic change it causes. The frozen privileged policy is not a stress-optimal counterfactual. No matched-mean, stress-aware attribution experiment is claimed; any such follow-up would need a separate frozen question, budget, comparison and fresh streams.

## Reproduction scope and disclosure

The repository separates a tiny clean-process reproduction, reduced-reference rollouts and the full independent campaign. All source and result assertions are traceable through `EVIDENCE.md`. Public-data pilot NPZs, per-episode CSVs, fit summaries and manifests are saved. Large deterministic Q-table caches are regenerable computation, not target-environment data, and need not be shipped in the review archive. The runtime reports cold table computation separately from total campaign wall time.

The original email fixed the design-and-solve process and deadline. The user-provided local statement refines implementation details, including the explicit ten-pilot minimum and central reduced-reference parameters. The stricter local requirements are met; no assignment wording or objective has been silently rewritten. Optional extensions have not displaced the required investigation.

Codex and delegated AI agents assisted with mathematics, code, verification and prose. External methodological sources are cited in the report and mathematical appendix. Synthetic constants define the benchmark and are not estimated from a real exchange. Author judgement remains separate from recorded computational checks.
