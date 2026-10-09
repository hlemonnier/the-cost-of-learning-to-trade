# Synthetic market-making benchmark

The market model, information boundary, accounting and frozen evaluation design.
All prices and fees use arbitrary synthetic units.

# 1. Market model and event chronology

### 1.1 State, horizon and units

An episode contains periods $t=0,\ldots,T-1$, with $T=300$. Initially, $P_0=1{,}000$, $C_0=0$, $Q_0=0$, and $X_0=0$. Here $P_t$ is the midpoint, $C_t$ is cash, and $Q_t$ is pre-decision inventory. Prices and fees use consistent arbitrary price units; one order is one unit of the asset. A period has no asserted real-world duration.

Inventory is restricted to $Q_t\in\{-5,-4,\ldots,5\}$. No inventory or cash carries between episodes. The trader has no market impact.

### 1.2 Public signal and midpoint

The public signal $X_t\in\{-1,0,1\}$ has transition matrix

$$
K=\begin{pmatrix}
0.75&0.20&0.05\\
0.10&0.80&0.10\\
0.05&0.20&0.75
\end{pmatrix},
\qquad K_{ij}=\Pr(X_{t+1}=j\mid X_t=i).
$$

Rows and columns are ordered as $(-1,0,1)$. The return and next midpoint are

$$
R_t=0.03X_t+0.30Z_t,\qquad Z_t\sim\mathcal N(0,1),\qquad P_{t+1}=P_t+R_t.
$$

### 1.3 Hidden execution regime

The regime $H_t\in\{-1,+1\}$ starts with equal probabilities. In nominal environments,

$$
\Pr(H_{t+1}=-H_t\mid H_t)=\kappa,\qquad
\Pr(H_{t+1}=H_t\mid H_t)=1-\kappa,\qquad \rho_t=H_t\theta.
$$

The signal and regime transitions are independent of one another and of the Gaussian innovations. Innovations are independent across periods. The role of $\rho_t$ is defined by the execution mechanism in Section 2; it is not an extra drift term in the public return model.

### 1.4 Order of events

At the start of period $t$, expose $P_t$, $X_t$, $C_t$, $Q_t$, and the trader's prior observations. The trader selects one admissible action before any current-period innovation or fill is revealed. A market order executes immediately; submitted passive quotes are fixed for that period.

Generate $R_t$ and potential passive fills jointly using the current $X_t$ and $H_t$. Book only actual executions at their specified prices. At period end, reveal $R_t$ and submitted-quote outcomes, expire all passive quotes, and update the cash and inventory ledger. Then transition to $X_{t+1}$ and $H_{t+1}$; only the new public signal is disclosed. After the last period, liquidate at $P_T$.

This is a reduced-form joint model, not a simulated intra-period price path. It does not specify queue position, routing latency, partial fills, market impact, or intra-period cancellation. Future returns may influence the simulated joint outcome but are never available to the policy before acting.

---

# 2. Actions and passive execution

### 2.1 Admissible actions

Let $s=+1$ mean a purchase by the market maker and $s=-1$ a sale. Each passive side offers no quote, a near quote, or a deeper quote. Encode that choice as $k_s\in\{\varnothing,0,1\}$. Alternatively, submit one market buy or one market sell, denoted $M_b$ and $M_a$. The action sets are

$$
\mathcal A_P=\{\varnothing,0,1\}^2,\qquad \mathcal A=\mathcal A_P\cup\{M_b,M_a\}.
$$

There are nine passive action pairs before inventory restrictions, including abstention, plus two market actions. A passive pair and a market order cannot be combined. An action is admissible only if every possible submitted-fill subset respects the inventory bound; do not rely on an opposite-side fill to make an otherwise inadmissible quote safe. Both sides may fill in the same period.

Set

$$
h=0.025,\qquad \Delta=0.025,\qquad c_P=0.001,\qquad c_T=0.002.
$$

A passive quote at depth $k\in\{0,1\}$ executes at $P_t-s(h+k\Delta)$. A market order executes at $P_t+sh$. These prices remain fixed for the action; a passive quote does not follow the midpoint during the period.

### 2.2 Marginal fills and dependence

Write $g(u)=(1+e^{-u})^{-1}$ and let $\Phi$ be the standard normal cumulative distribution function. The specified marginal probability is

$$
p_{s,k}(x)=g(-0.3-0.7k-0.2sx).
$$

For each side, draw $U_{t,s}\sim\mathcal N(0,1)$, independent of the other side and of $Z_t$. Define

$$
V_{t,s}=\rho_t sZ_t+\sqrt{1-\rho_t^2}\,U_{t,s},
\qquad F_{t,s,k}=\mathbf 1\{V_{t,s}\leq\Phi^{-1}(p_{s,k}(X_t))\}.
$$

Use the same $U_{t,s}$ for the two potential depths on that side. The trader can submit only one of those depths. Potential outcomes may exist internally for simulation and paired evaluation, but only selected outcomes may enter the learner's observations.

An unfilled submitted order is an observed zero. An unsubmitted order has no observed outcome and must be stored as missing, not as a zero. For submitted orders, executions are either zero or one unit. Passive fees are charged on each fill; an unfilled quote has no execution fee.

Although the two side-specific $U$ innovations are independent, both sides share the same return innovation. Derive the observation likelihood under the actual conditioning information rather than assuming that all execution outcomes are independent.

### 2.3 Model validation requirement

Verify the conditional return distribution and marginal fill functions empirically, including both depths and both sides. Check the shared-innovation construction, depth consistency, action masking, and simultaneous fills. These checks validate the specified synthetic model, not its realism as an exchange model.

---

# 3. Information, accounting and objective

### 3.1 What the trader observes

Let $O_t$ contain $R_t$ and the selected passive outcomes after period $t$. For a market action or abstention, it contains no passive-fill observations. Let $\mathcal I_t$ denote the information available immediately before action $A_t$:

$$
\mathcal I_t=\sigma(P_0,X_0,C_0,Q_0,\{A_u,O_u,X_{u+1},C_{u+1},Q_{u+1}\}_{u<t}).
$$

The policy may use independent private randomisation. It must not receive $H_t$, future innovations, unselected fills, a true-parameter label, or a simulator seed. Observed returns can be transformed using the known return model; prohibiting hidden state does not prohibit transformations of genuinely observed data. Simulator information needed by diagnostic or oracle code must be kept separate.

### 3.2 Cash and inventory ledger

Let $d_k=h+k\Delta$, and let $f_{t,s}$ be the realised fill of the depth submitted on side $s$. For a passive action, sum only over submitted sides:

$$
Q_{t+1}=Q_t+\sum_s sf_{t,s},\qquad
C_{t+1}=C_t+\sum_s f_{t,s}(-sP_t+d_{k_s}-c_P).
$$

For a market action in direction $s$,

$$
Q_{t+1}=Q_t+s,\qquad C_{t+1}=C_t-sP_t-h-c_T.
$$

For abstention, $C_{t+1}=C_t$ and $Q_{t+1}=Q_t$. Abstaining does not liquidate existing inventory. All next-period inventory is marked at $P_{t+1}$, not at $P_t$.

### 3.3 Terminal liquidation and objective

$C_T$ and $Q_T$ denote balances immediately before terminal liquidation. The terminal cash PnL and inventory-penalised objective are

$$
\Pi_T=C_T+Q_TP_T-|Q_T|(h+c_T),
$$

$$
J(\pi)=\mathbb E_\pi\!\left[\Pi_T-\lambda\sum_{t=0}^{T-1}Q_t^2\right],
\qquad \lambda=0.002.
$$

Terminal liquidation is mandatory and can liquidate multiple units at the stated constant per-unit spread and fee. It is outside the one-unit discretionary action constraint. Do not apply the liquidation cost twice.

The penalty uses **pre-decision** $Q_t$, exactly as written. It is a utility penalty, not a cash payment. Post-execution exposure may be reported as an additional risk diagnostic, but must not silently replace the specified objective. Report actual net PnL and the penalised objective separately.

Construct unit tests for each signed cash flow, both-side fills, existing inventory under abstention, boundary actions, and terminal liquidation. Derive an independent wealth-reconciliation identity as a check on the implementation.

---

# 4. Information and the economic cost of learning

### 4.1 Identify the available evidence

Characterise the information about the hidden regime in public returns, the signal, single-sided execution feedback, two-sided feedback, and periods with no submitted quotes. Distinguish observing a fill indicator on its own from observing it jointly with the subsequent return.

With known $\theta$ and $\kappa$, derive a correct online filter. The pre-decision regime belief may be denoted

$$
b_t=\Pr(H_t=+1\mid\mathcal I_t).
$$

Derive the action-dependent observation likelihood, the posterior update, and the transition to the next decision-time belief. Account for filled and unfilled submitted quotes, both submitted sides, and periods with no passive feedback. Specify numerically stable treatment of small likelihoods.

Validate inference in configurations whose truth is available to the evaluator. State which diagnostics require privileged regime labels and which can be computed by a deployed trader. Do not expose evaluator labels to fitting, tuning, or decisions.

### 4.2 Quantitative learning limitation

Construct a stationary, repeated one-step reduction with two fully specified environments, $\mathcal E_0$ and $\mathcal E_1$, that imply different optimal trading decisions. State what is fixed, reset, hidden, or removed relative to the sequential model. In particular, state the treatment of public context, inventory, liquidation, and regime switching.

Let $n$ be the number of opportunities and $\widehat e_n$ an environment-identification decision. Choose and state an error level $\delta\in(0,1/2)$. Establish a non-asymptotic lower bound on the evidence needed to achieve

$$
\max_{e\in\{0,1\}}\Pr_{\mathcal E_e}(\widehat e_n\ne e)\leq\delta.
$$

The bound must account for adaptive action selection and the feedback actually observed. Define whether the evidence count is elapsed periods, submitted sides, or informative observations; these are not interchangeable. A quoted theorem must be connected explicitly to the observation laws in the reduction.

### 4.3 Translate evidence into an economic trade-off

Define an environment-specific, attainable comparator and a regret or opportunity-cost quantity. Derive a bound connecting the requirement to learn with an economic cost in at least one of the environments. State the assumptions, constants or parameter dependence, and the scope of the result.

Do not claim that every informative action loses money, that a bound on observations automatically implies the same bound on regret, or that the stationary result applies unchanged to the switching model. A precise, useful result for the reduction is preferable to an unsupported general theorem.

**Required evidence:** a proof or complete derivation; a worked numerical illustration using the reduction; and a short explanation of what the result predicts, and does not predict, for the full trading problem.

---

# 5. Known-parameter control and reference solution

### 5.1 Formulate the decision problem

Initially assume $\theta$ and $\kappa$ are known but $H_t$ is not. Derive a finite-horizon control formulation using a sufficient decision state. Justify any removal of the absolute midpoint or cash from the optimisation state rather than assuming it.

Account for the current inventory, remaining horizon, uncertainty about the regime, action-dependent feedback, trading cash flows, and terminal liquidation. Derive the boundary condition and the transition used by the control recursion. Resolve the timing of rewards, observations, and regime transitions consistently with Sections 1–3.

### 5.2 Numerically validated restricted problem

Implement a numerical reference at

$$
T_{\mathrm{ref}}=30,\qquad |Q_t|\leq2,
\qquad(\theta,\kappa)=(0.35,0.02).
$$

All other market and accounting parameters are unchanged. The central parameter pair fixes the primary reference calculation; additional reduced configurations may be used to investigate the economic result.

Numerical integration and belief discretisation are permitted. Provide refinements in integration accuracy and state resolution, and quantify changes in values and action choices. State the stopping criterion before examining the preferred policy's advantage. Separate Monte Carlo error from numerical approximation error.

A numerically stable result is not automatically an exact solution or certified bound. Use terminology consistent with the evidence, and do not call a heuristic full-information calculation an oracle optimum.

### 5.3 Controls needed to interpret the result

Compute a full-information reference that observes the current $H_t$ and knows parameters, but not future regimes or innovations. It must have the same horizon, accounting, inventory constraints, and actions. It is a privileged-information benchmark, not an implementable competitor.

Compare the partially observed control with a myopic policy and a no-information-value control. The latter should update its beliefs from realised feedback but omit the benefit of feedback-conditioned future decisions when evaluating present actions. Specify the approximation precisely. A plug-in regime estimate may be an additional certainty-equivalent baseline, but must not substitute for a matched information-value ablation.

Use these comparisons to separate inventory planning, current regime inference, and deliberate information acquisition. Demonstrate whether any states produce different choices specifically because of future information, using a controlled comparison at the same state and with the same current evidence.

A richer controller need not win materially. Report action ties, economically negligible differences, and regions where the simpler policy is effectively equivalent. Do not manufacture an information-value finding through an artificially weak comparator.

---

# 6. Learning unknown parameters from a finite pilot

### 6.1 Historical data budget

In the practical problem, the nominal model family is known but the true $\theta$ and $\kappa$ are not supplied to the learner. For each environment and each pilot replicate, generate 100 independent episodes of length 300 under a behaviour policy that is uniform over currently admissible actions:

$$
\beta(a\mid\mathcal I_t)=\frac{1}{|\mathcal A(Q_t)|},\qquad a\in\mathcal A(Q_t).
$$

At an inventory boundary, renormalise over admissible actions rather than sampling an illegal action and silently replacing it. Log the selected action probability. Every pilot episode resets $P_0,C_0,Q_0,X_0$ and independently draws $H_0$ with equal probabilities.

The learner receives public observations, actions, action probabilities, its own fill outcomes, and the cash and inventory ledger. It receives neither hidden-state sequences nor counterfactual outcomes. The 100 episodes are the total target-environment data budget, including any fitting, validation, or tuning split.

### 6.2 Policy design and parameter uncertainty

Develop a practical policy for $T=300$ and $\left\vert Q_t\right\vert\leq5$. It may estimate parameters, maintain uncertainty, and update using newly observed feedback during an episode. Explain how parameter uncertainty interacts with regime inference and action selection.

The evaluation grid is public benchmark information, but the true parameter pair and environment identifier must not be passed to the learner. Any grid-based prior or restricted parameter support must be declared. Do not infer labels from filenames, configuration objects, seeds, or metadata.

The method is unrestricted. Model-based optimisation, controlled approximations, or learned policies are acceptable. Explain why the chosen complexity is justified by the observed sample size. Distinguish active information acquisition from learning passively from whichever observations a profit-seeking controller generates.

### 6.3 Separate evidence from computation

Model-based integration or rollouts from a learner-fitted model are allowed and must be logged as computation. They are not additional observations from the true environment. Extra interactive training on the actual hidden-parameter environment is outside the pilot budget.

Known-parameter reference calculations may use model parameters only in clearly separated reference code. Do not transfer true-parameter-specific policies, labels, or privileged state histories into the practical learner unless they are explicitly available under a declared public prior and the same rule is used across environments.

Freeze algorithm structure, tuning rules, allowed online updates, and exploration settings before final evaluation. Each independent evaluation episode starts from a fresh copy of the same pilot-fitted state; online updates may operate within that episode but do not accumulate across evaluation episodes. This distinguishes within-episode adaptation from obtaining an ever-growing evaluation training set.

**Required ablation:** keep the estimator, action set, risk treatment, and available pilot data matched, while removing deliberate information acquisition. Explain what this comparison does and does not isolate.

---

# 7. Evaluation, attribution and uncertainty

### 7.1 Nominal environments and repetitions

Evaluate every combination in

$$
\theta\in\{0.15,0.35,0.65\},\qquad
\kappa\in\{0.002,0.02,0.10\}.
$$

Use independent random streams for pilot generation, any permitted tuning, and final evaluation. Obtain at least 1,000 final episodes per parameter pair and policy, distributed across at least ten independently generated 100-episode pilot datasets. A minimum balanced design is ten pilots followed by 100 fresh test episodes per pilot.

All data-dependent competitors should receive the same pilot data in a replicate. Paired exogenous random streams may be used to reduce comparison variance, provided counterfactual observations remain inaccessible to the policy. Document the pairing and preserve the marginal law of each strategy's experiment.

### 7.2 Required comparison set

**Abstention.** Never initiate a trade; with zero initial inventory, its net PnL is zero.

**Taker-only control.** Use public information and inventory planning with only market orders and abstention; account for every spread and fee.

**Conditional-independence control.** Use the supplied return and marginal-fill models but assume conditional independence between execution and returns. Define its execution model explicitly.

**Myopic control.** Optimise a declared one-period criterion without multi-period planning; state the continuation and liquidation convention.

**No-information-value / certainty-equivalent control.** Use current execution evidence but omit the value of feedback-conditioned future choices. Include the matched ablation from Sections 5–6.

**Proposed learning-and-control policy.** Use only permitted pilot and online feedback.

**Known-parameter references.** Compare against the restricted partially observed numerical reference where available and the full-information control under the appropriate configuration. Label privileged information and numerical error.

### 7.3 Economic and statistical reporting

Report expected net terminal PnL, the inventory-penalised objective, terminal liquidation costs, passive and market fees, turnover, action frequencies, fill counts, and inventory exposure. Include a downside-risk summary and the performance gap to the relevant reference.

Provide uncertainty for paired policy differences, not just isolated point estimates. Distinguish within-pilot evaluation noise from between-pilot fitting variation. Treat an episode, rather than every sequential event, as the basic independent evaluation unit conditional on the pilot. Do not pool adaptation histories as independent observations.

Separate spread capture, directional exposure, execution selection, and inventory costs through ledger-based attribution and controlled ablations. Explain which findings are stable across parameter pairs and which depend on a particular environment. Do not select the winning configuration after seeing the results and present it as representative.

---

# 8. Misspecification and the deployment decision

### 8.1 Protocol common to both stress tests

For each nominally trained policy, freeze its structure, tuning choices, and pilot-fitted starting state. Preserve only the online updates declared before evaluation. Run each stress separately using fresh episodes and the nominal comparison design. Stress outcomes may not be used to select or retrain the reported nominal policy.

Retain the public signal model, return model, quoted prices, fees, and marginal-fill functions. Verify those retained components empirically rather than assuming a simulator modification preserved them.

### 8.2 State-dependent execution dependence

Replace the nominal dependence parameter with

$$
\rho_t=H_t\theta(1+0.25X_t).
$$

Retain the nominal regime-switching probability. For the prescribed grid the dependence parameter remains in the valid interval. The learner is not told that the dependence strength now varies with the signal.

### 8.3 Non-geometric regime durations

In a separate experiment, retain $\rho_t=H_t\theta$ but replace nominal switching with deterministic 50-period regime durations. Draw an unobserved initial phase $D$ uniformly from $\{0,\ldots,49\}$, independently of $H_0$, and set

$$
H_t=H_0(-1)^{\lfloor(t+D)/50\rfloor}.
$$

Here $H_0$ remains equally likely to be $-1$ or $+1$. The nominal training value of $\kappa$ no longer controls test-time switching; retain it in reporting as a label of the training environment, not as the realised stress process.

### 8.4 Detection and failure analysis

Determine which observable diagnostics respond to each misspecification and which may appear satisfactory while economic performance deteriorates. Separate monitoring that can run online from retrospective evaluation using hidden labels. Examine adaptation speed, erroneous confidence, and policies that stop collecting useful feedback.

Document at least one material failure or limitation. If the prescribed tests do not reveal a substantial failure, conduct a further exploratory stress test, clearly distinguish it from preregistered evaluation, and do not retrain on it. An empirical stress grid is not a guarantee against all alternative environments.

### 8.5 Deployment decision

Specify a baseline, a policy-selection rule, a stationary deployment population, and an economically meaningful acceptance threshold. Explain which independent evidence supports deployment, which supports retaining the baseline, and which leaves the decision unresolved.

Account for selecting among policies and repeated comparisons. State whether the evidence is an empirical confidence assessment or a formal guarantee, and justify any claimed guarantee under explicit assumptions. Gaussian innovations make rewards unbounded; do not invoke a bounded-reward concentration result without satisfying its assumptions. Explain why nominal evidence does not automatically protect against either stress process or arbitrary future regime change.

---

# 9. Completion criteria and extensions

### 9.2 Core completion gate

The core is complete only when accounting and observation-boundary tests pass; the lower-bound argument is documented; the restricted control calculation has refinement evidence; all required competitors and the matched learning ablation are evaluated; both stress tests are reported; uncertainty and data budgets are explicit; and the report and code agree.

An unsuccessful conjecture or weak learned policy is not a reason to change the objective or conceal a result. Explain the finding and retain the original comparison. Mathematical correctness, reliable accounting, and a defensible conclusion take precedence over model complexity and positive PnL.

### 9.3 Optional extensions — outside the mandatory scope

Only after freezing and reproducing the core investigation, select an extension that answers a specific unresolved question. Each extension must retain a core baseline, declare additional assumptions and data, and report its incremental benefit and cost. No extension is a prerequisite for completing this task.

**Multi-asset trading.** Add a small correlated asset system to test whether shared execution information or inventory hedging changes the value of learning. Define covariance, joint regimes, costs, and portfolio constraints; do not compare against an unmatched single-asset risk budget.

**Hawkes or other event-driven order flow.** Replace the reduced execution mechanism with an explicit clustered-event model to examine whether action-dependent information findings survive a richer observation process. Define event types, intensities, stability conditions, and a valid fill rule before experimentation.

**Deep reinforcement learning.** Approximate a documented control problem and compare against the checked reference and model-based learner under disclosed information and training budgets. Additional simulator interaction is a different data regime and must be reported as such.

**Exchange connectivity.** A read-only public-data or sandbox adapter may demonstrate observation handling and shadow decisions. It does not validate live fills or profitability. Live orders, funded execution, and production credentials are outside scope.

Choose depth over implementing all four. Optional work must appear separately and must not retroactively change the core task, metric, or evaluation protocol.

---

# Appendix A. Configuration and research context

### A.1 Fixed configuration at a glance

| Item | Specification |
|:--|:--|
| Episode / initial state | $T=300$; $P_0=1{,}000$; $C_0=Q_0=X_0=0$ |
| Signal | States $(-1,0,1)$; transition matrix in Section 1 |
| Midpoint change | $R_t=0.03X_t+0.30Z_t$ |
| Hidden regime | $H_t\in\{-1,+1\}$; equal initial probabilities |
| Nominal dependence | $\rho_t=H_t\theta$ |
| Parameter grid | $\theta\in\{0.15,0.35,0.65\}$; $\kappa\in\{0.002,0.02,0.10\}$ |
| Inventory / order size | $\left\vert Q_t\right\vert\leq5$; one unit per order |
| Half-spread / depth step | $h=0.025$; $\Delta=0.025$ |
| Passive / market fees | $c_P=0.001$; $c_T=0.002$ per executed unit |
| Inventory penalty | $\lambda=0.002$; charged on pre-decision $Q_t^2$ in the objective only |
| Fill probabilities | $p_{s,k}(x)=g(-0.3-0.7k-0.2sx)$ |
| Pilot budget | 100 independent episodes per environment and pilot replicate |
| Minimum final evaluation | 10 independent pilots × 100 fresh episodes per policy and parameter pair |
| Reduced reference | $T=30$; $\left\vert Q_t\right\vert\leq2$; $(\theta,\kappa)=(0.35,0.02)$ |
| Stress 1 | $\rho_t=H_t\theta(1+0.25X_t)$ |
| Stress 2 | Fixed 50-period regime durations; hidden uniform initial phase |

### A.2 Research context and attribution

The task draws on established work in partially observed market making, action-dependent feedback, and learning costs. These references provide context; their models and assumptions differ from this benchmark, so their results must not be imported without checking applicability.

[1] Diego Zabaljauregui and Luciano Campi. *Optimal market making under partial information with general intensities*. arXiv:1902.01157, version 3, 2020. Relevant to filtering and control with hidden market factors, and to separating partial- and full-information comparisons.

[2] Tor Lattimore and Csaba Szepesvari. *An Information-Theoretic Approach to Minimax Regret in Partial Monitoring*. arXiv:1902.00470, version 2, 2019. Relevant to information-theoretic analysis of sequential decisions with restricted feedback.

[3] Nicolò Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Luigi Foscari, and Vinayak Pathak. *Market Making without Regret*. arXiv:2411.13993, version 2, 2025. Relevant to sequential market-making feedback and the economic cost of exploration.

The numerical constants, action set, data budget, and evaluation protocol in this document define this benchmark. They are synthetic design choices, not empirical estimates from a market .

---
