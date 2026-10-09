# Environment and inference implementation record

## Failure analysis recorded before implementation

The executable end-to-end verification is authored before production implementation. It exercises complete public-observation trajectories and independently reconstructs cash, inventory, wealth, and inference. Its final JSON evidence is reproducible from the command recorded below.

Possible failures are: wrong signed trade cash flows; fee charged on a non-fill; forgetting the spread on a market action; double-charging terminal liquidation; silently liquidating on abstention; unsafe paired quotes at inventory boundaries; risk penalty using post-decision inventory or debiting cash; missing feedback encoded as an observed zero; a passive/market action combination; actions sampled uniformly before rather than after masking; mutable observations sharing simulator buffers; hidden labels, seeds, innovations, future signals, or unselected fills passed to a policy; wrong event ordering; correlated random components or shared episode starts; independent depth noises; altered fill marginals in either stress; a phase off-by-one in deterministic durations; omission of non-fill information; unconditional side independence in a mixture likelihood; missing posterior regime transition; failure to reset regime priors between pilot episodes; density underflow; mixing linear and log parameter weights; fitting with evaluator labels; and reporting conditional fill likelihood as full return-and-fill likelihood.

## Public interface fixed before implementation

- `model.py`: constants and action mapping specified in `DESIGN.md`.
- `generate_exogenous(episodes, horizon, theta, kappa, seed, variant="nominal")`: evaluator-only random tapes, with signal shape `(episodes,horizon+1)` and latent/innovation arrays `(episodes,horizon)`. Component RNG streams are spawned separately.
- `BatchedEnvironment(tapes, qmax=5)`: `.observe()` returns a frozen record of copied public pre-decision arrays; `.step(actions)` returns public feedback only; `.finalize()` liquidates exactly once and returns evaluator metrics. Policies must be called with public observations and feedback, never this evaluator object.
- `generate_uniform_pilot(episodes=100, horizon=300, theta=..., kappa=..., seed=..., policy_seed=..., qmax=5)`: returns only `PilotData`. Public state arrays include both endpoints and have shape `(episodes,horizon+1)`; action and return arrays have shape `(episodes,horizon)`; fills/submission/price/quantity arrays end in side dimension two. `.as_dict()` is suitable for `numpy.savez_compressed` and never includes random seeds or true parameters.
- `observation_log_likelihood(signal, return_, actions, fills, candidate_theta)` returns `(episodes,models,2)` in hidden-sign order `(-1,+1)`. It is **conditional fill log likelihood given the observed return**, excluding the common Gaussian return density.
- `filter_step(beliefs, log_weights, signal, return_, actions, fills, candidate_theta, candidate_kappa)` returns predicted next-period regime probabilities, normalized updated model log weights, and the model-averaged conditional fill log score for the current feedback.
- `fit_grid_posterior(pilot, candidate_theta, candidate_kappa)` returns `GridFit` with `weights`, `log_weights`, `log_likelihood`, `episode_log_likelihood`, and the declared candidate vectors. It resets hidden-state belief to one half for every pilot episode, sums episode likelihoods, and uses a uniform model prior by default.

## Accounting attribution

For old inventory `q`, return `r`, and signed newly executed quantity `dq`, the marked-wealth increment is passive quoted-spread capture minus passive fees minus discretionary market spread and fees, plus `q*r + dq*r`. Liquidation then subtracts `abs(q_T)*(h+c_T)`. The independent sum of these terms must match the explicit cash ledger. Inventory penalty is separately `lambda*sum(q_t**2)` using pre-decision inventory. Turnover includes terminal liquidation; discretionary passive and market fills are also stored separately.

The reporting decomposition separates `directional_exposure = sum(q_old*r + dq*mu*x)`, `execution_selection = sum(passive_dq*(r-mu*x))`, and `market_innovation = sum(market_dq*(r-mu*x))`. This prevents unpredictable taker noise or the public signal's predictable fill exposure being labelled execution selection. The acceptance harness independently reconstructs all three terms before these additional attribution fields are implemented.

## Observation likelihood and online update

At period end the public return reveals `z=(r-mu*x)/sigma`. For a submitted side `s` at depth `k`, let `a=Phi^{-1}(p_{s,k}(x))`. Conditional on the current regime sign `h`, its fill probability is

\[
 \ell_{h,s}(z)=\Phi\left(\frac{a-h\theta s z}{\sqrt{1-\theta^2}}\right).
\]

Conditional on **both** this observed return and the regime, the side-specific noises are independent. Thus `L_h` is the product over submitted sides of `ell**fill * (1-ell)**(1-fill)`. A non-fill contributes its survival probability. An unsubmitted side contributes one. The likelihood must mix these **joint** side likelihoods over the regime; mixing each side separately would wrongly assume unconditional independence. All computations use `log_ndtr`, including the survival probability as `log_ndtr(-argument)`.

For pre-decision `b=P(H_t=+1 | history)`, the posterior on the current regime is

\[
 \widehat b=\frac{bL_{+}}{(1-b)L_{-}+bL_{+}},\qquad
 b_{t+1}=\kappa+(1-2\kappa)\widehat b.
\]

The public return density, signal transition, and already selected action probabilities are common to the candidate models conditional on the public history; they do not change posterior odds. Cash and inventory are deterministic consequences of recorded observations and supply no additional likelihood factor. When no passive quote is submitted, `L_-=L_+=1`: there is no current regime evidence, and the posterior only undergoes the Markov prediction.

With a finite parameter grid, maintain one regime belief per parameter pair and a joint posterior model weight. Each model's predictive fill probability multiplies its weight before normalization; each model also updates its own regime belief. This is an exact filter for the declared finite-grid family, although the controller that consumes it can be approximate. A new independent episode resets every regime belief to one half and starts from the pilot parameter posterior. Evaluation feedback is never carried to another episode. The fitting routine resets hidden beliefs separately for each of the pilot episodes and adds their marginal log likelihoods.

The conditional fill log score excludes the common Gaussian return density. It is useful for model assessment on its own selected observations, but its raw level also changes with quote mix and conditional entropy; differences across policies alone do not establish superior calibration.

## Verification command and evidence

Run `PYTHONPATH=src .venv/bin/python verification/verify_environment.py`. It writes `outputs/environment/verification.json` only after all required assertions pass. Conditional distribution tolerances are declared in the harness before simulation; they are simultaneous conservative Monte Carlo checks, not fitted tolerances. Calibration diagnostics use true regimes only inside this evaluator verification.

The recorded run passed 209 action/inventory legality combinations, all eleven forced action trajectories, a 30,000-row uniform pilot with independent cash reconstruction, exact short-sequence hidden-path enumeration, and 3.24 million distribution-check periods over all nine pairs and all three variants. Maximum absolute cash/wealth reconciliation error was `8.77e-12`; the maximum standardized fill-marginal discrepancy was `3.21`, below the predeclared eight-standard-error threshold. The short-sequence marginal log likelihood matched independent hidden-path enumeration to the recorded floating-point precision. A controlled known-parameter filter at `(theta,kappa)=(.65,.02)` achieved predictive Brier score `0.095815` versus `0.25` for the uninformative predictor. The calibration-bin frequencies in the JSON are descriptive, with serial dependence within each of the 100 independent episodes; they are not binomial confidence guarantees.
