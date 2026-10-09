"""Episode pairing and independent-pilot uncertainty; no tick-level inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .protocol import validate_protocol, environment_names


def _columns(frame, names):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("statistical evidence must be a nonempty DataFrame")
    missing = set(names)-set(frame.columns)
    if missing:
        raise ValueError(f"missing statistical evidence columns: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("duplicate evidence column names are not supported")


def _finite(frame, name):
    column = frame[name]
    if (not pd.api.types.is_numeric_dtype(column.dtype)
            or pd.api.types.is_bool_dtype(column.dtype)
            or pd.api.types.is_complex_dtype(column.dtype)):
        raise ValueError(f"{name} must contain real numeric outcomes")
    values = column.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or nonfinite outcomes")
    return values


def _ids(frame, name, count=None):
    values = _finite(frame, name)
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError(f"{name} IDs must be nonnegative integers")
    if count is not None and np.any(values >= count):
        raise ValueError(f"{name} IDs exceed the declared budget")


def _alpha(alpha):
    if isinstance(alpha, (bool, str)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite real probability strictly between zero and one")


def _finite_result(result):
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite statistical result: arithmetic overflow or invalid evidence")
    return result


def _validate_cluster(frame, value, expected_pilots=None, expected_episodes=None):
    _columns(frame, ["pilot", "episode", value])
    _finite(frame, value)
    _ids(frame, "pilot", expected_pilots)
    _ids(frame, "episode", expected_episodes)
    for name in ("environment", "variant", "policy"):
        if name in frame and (frame[name].isna().any() or frame[name].nunique() != 1):
            raise ValueError(f"a clustered estimate requires exactly one {name} stratum")
    if frame.duplicated(["pilot", "episode"]).any():
        raise ValueError("duplicate paired pilot/episode keys")
    n = frame.pilot.nunique()
    if n < 2:
        raise ValueError("at least two independent pilots are required to estimate pilot uncertainty")
    if expected_pilots is not None and n != expected_pilots:
        raise ValueError("incomplete declared pilot budget")
    if set(frame.pilot.unique()) != set(range(n)):
        raise ValueError("pilot IDs must cover every declared contiguous pilot")
    sizes = frame.groupby("pilot", observed=True).size()
    if sizes.nunique() != 1 or sizes.iloc[0] < 2:
        raise ValueError("balanced evidence with at least two episodes per pilot is required")
    count = int(sizes.iloc[0])
    if expected_episodes is not None and count != expected_episodes:
        raise ValueError("incomplete declared per-pilot episode budget")
    # With unique nonnegative integer IDs, count observations below count covers
    # exactly 0..count-1; this rejects shifted IDs even in standalone calls.
    _ids(frame, "episode", count)


def validate_campaign(data, protocol):
    """Reject incomplete, mislabeled or unpaired evidence before any summary.

    The distinct pilot fingerprints check prevents accidental relabelling of
    one dataset as independent replicates; independence itself is established
    by the separately recorded simulator streams, not inferred from outcomes.
    """
    protocol = validate_protocol(protocol)
    keys = ["environment", "variant", "policy", "pilot", "episode"]
    # These ledger and inventory outcomes exist for every policy and episode.
    # Missing values here are corruption, unlike unavailable monitoring ratios.
    financial = ["objective", "pnl", "inventory_penalty", "risk_penalty",
                 "liquidation_cost", "liquidation_fees", "liquidation_spread",
                 "passive_fees", "market_fees", "market_spread_cost", "turnover",
                 "spread_capture", "directional_pnl", "execution_exposure_pnl",
                 "directional_exposure", "execution_selection", "market_innovation",
                 "reconciled_pnl", "reconciliation_error", "inventory_sq_sum",
                 "inventory_post_sq_sum", "inventory_abs_sum", "max_abs_inventory",
                 "mean_abs_inventory", "terminal_inventory", "terminal_units",
                 "terminal_inventory_after_liquidation", "passive_fills", "market_fills",
                 "bid_fill_count", "ask_fill_count"]
    execution = ["max_quote_gap", "submitted_periods", "submitted_sides",
                 "submitted_sides_x-1", "submitted_sides_x0", "submitted_sides_x1"]
    actions = [f"action_{a}" for a in range(11)]
    required = keys + financial + execution + actions + ["theta", "kappa", "pilot_data_hash", "starting_posterior_hash",
                                                        "score_sum_x-1", "score_sum_x0", "score_sum_x1"]
    _columns(data, required)
    for name in financial+execution+actions+["theta", "kappa"]:
        _finite(data, name)
    expected_sets = {"environment": environment_names(protocol), "variant": protocol["variants"], "policy": protocol["policies"]}
    for name, expected in expected_sets.items():
        if data[name].isna().any() or set(data[name].unique()) != set(expected):
            raise ValueError(f"campaign {name} strata differ from the declared protocol")
    _ids(data, "pilot", protocol["pilots"])
    _ids(data, "episode", protocol["test_episodes"])
    if data.duplicated(keys).any():
        raise ValueError("duplicate campaign paired episode key")
    expected_index = pd.MultiIndex.from_product(
        [expected_sets["environment"], protocol["variants"], protocol["policies"], range(protocol["pilots"])],
        names=keys[:-1])
    counts = data.groupby(keys[:-1], observed=True).size().reindex(expected_index)
    if counts.isna().any() or not (counts == protocol["test_episodes"]).all():
        raise ValueError("incomplete declared campaign pilot/episode budgets")
    if len(data) != len(expected_index)*protocol["test_episodes"]:
        raise ValueError("campaign total differs from the declared complete design")
    for name, (theta, kappa) in zip(expected_sets["environment"], protocol["environments"]):
        group = data[data.environment == name]
        if not ((group.theta == theta) & (group.kappa == kappa)).all():
            raise ValueError("environment label disagrees with numeric parameter metadata")
    action_values = data[actions].to_numpy()
    if (np.any(action_values < 0) or np.any(action_values != np.floor(action_values))
            or not np.all(action_values.sum(axis=1) == protocol["horizon"])):
        raise ValueError("action counts do not match complete episode horizons")
    if not data.pilot_data_hash.astype("string").str.fullmatch(r"[0-9a-f]{64}", na=False).all():
        raise ValueError("missing or invalid pilot dataset fingerprint")
    groups = data.groupby(["environment", "pilot"], observed=True)
    if not (groups.pilot_data_hash.nunique() == 1).all():
        raise ValueError("paired policies or variants received different pilot datasets")
    identities = groups.pilot_data_hash.first()
    if identities.duplicated().any():
        raise ValueError("one pilot dataset was reused as independent pilot replicates")
    if data.starting_posterior_hash.isna().any():
        raise ValueError("missing starting policy-state fingerprint")
    learning = data[data.policy.isin(["active", "noinfo", "myopic"])]
    if not learning.starting_posterior_hash.astype("string").str.fullmatch(r"[0-9a-f]{64}", na=False).all():
        raise ValueError("invalid initial learning-state fingerprint")
    if not (learning.groupby(["environment", "pilot"], observed=True).starting_posterior_hash.nunique() == 1).all():
        raise ValueError("learning policies did not reset to the same pilot-fitted state")
    return protocol


def clustered_estimate(frame, value, alpha=.05, *, expected_pilots=None, expected_episodes=None):
    _alpha(alpha)
    _validate_cluster(frame, value, expected_pilots, expected_episodes)
    groups = frame.groupby('pilot')[value]
    means = groups.mean().to_numpy()
    n = len(means)
    between_observed = float(np.var(means, ddof=1))
    within = float((groups.var(ddof=1).fillna(0.)/groups.size()).mean())
    se = np.sqrt(between_observed/n)
    critical = float(student_t.ppf(1-alpha/2, n-1))
    mean = float(np.mean(means))
    return _finite_result(dict(mean=mean, se=float(se), low=mean-critical*se,
                high=mean+critical*se, pilots=n, episodes=len(frame),
                df=n-1,
                observed_pilot_mean_variance=between_observed,
                within_pilot_mean_variance=within,
                estimated_between_pilot_variance=max(between_observed-within, 0.)))


def stratified_estimate(frame, value, alpha=.05, *, protocol):
    """Uniform fixed environment mixture; randomness is pilot and test streams."""
    protocol = validate_protocol(protocol)
    _alpha(alpha)
    _columns(frame, ["environment", "pilot", "episode", value])
    if frame.environment.isna().any() or set(frame.environment.unique()) != set(environment_names(protocol)):
        raise ValueError("stratified evidence differs from the explicitly declared population")
    estimates = [clustered_estimate(g, value, alpha, expected_pilots=protocol['pilots'],
                                   expected_episodes=protocol['test_episodes'])
                 for _, g in frame.groupby('environment')]
    count = len(estimates)
    mean = float(np.mean([r['mean'] for r in estimates]))
    contributions = np.array([r['se']**2/count**2 for r in estimates])
    variance = float(contributions.sum())
    denominator = float(sum(v*v/r['df'] for v, r in zip(contributions, estimates)))
    df = variance**2/denominator if denominator > 0 else 1.
    se = np.sqrt(variance)
    return _finite_result(dict(mean=mean, se=float(se), df=float(df),
                low=float(mean-student_t.ppf(1-alpha/2, df)*se),
                high=float(mean+student_t.ppf(1-alpha/2, df)*se),
                one_sided_lower=float(mean-student_t.ppf(1-alpha, df)*se),
                environments=count, pilots=sum(r['pilots'] for r in estimates),
                episodes=len(frame)))


def summarize(data, protocol):
    protocol = validate_campaign(data, protocol)
    budget = dict(expected_pilots=protocol['pilots'], expected_episodes=protocol['test_episodes'])
    per_environment = []
    additional = ['pnl', 'inventory_penalty', 'liquidation_cost', 'passive_fees',
                  'market_fees', 'turnover', 'spread_capture', 'directional_exposure',
                  'execution_selection', 'mean_abs_inventory', 'max_quote_gap',
                  'regime_brier', 'wrong_confident_fraction', 'post_switch_error_10',
                  'conditional_logscore', 'score_residual_per_side']
    for (variant, environment, policy), group in data.groupby(
            ['variant', 'environment', 'policy']):
        record = dict(variant=variant, environment=environment, policy=policy,
                      objective=clustered_estimate(group, 'objective', **budget),
                      pnl=clustered_estimate(group, 'pnl', **budget))
        record['metrics'] = {c: float(group[c].mean()) for c in additional if c in group}
        if 'post_switch_window_periods' in group and group.post_switch_window_periods.sum() > 0:
            record['metrics']['post_switch_error_10'] = float(
                group.post_switch_error_count.sum()/group.post_switch_window_periods.sum())
        if policy in {'active', 'noinfo', 'myopic'} and group.submitted_sides.sum() > 0:
            record['metrics']['score_residual_per_side'] = float(
                group[[f'score_sum_x{x}' for x in [-1, 0, 1]]].sum().sum()/group.submitted_sides.sum())
        quantile = float(group.pnl.quantile(.05))
        record['downside'] = {'pnl_5pct_quantile': quantile,
                              'pnl_lower_5pct_mean': float(group.loc[group.pnl <= quantile, 'pnl'].mean()),
                              'loss_probability': float((group.pnl < 0).mean())}
        record['action_frequencies'] = [float(group[f'action_{a}'].mean()/protocol['horizon'])
                                        for a in range(11)]
        per_environment.append(record)
    keys = ['variant', 'environment', 'pilot', 'episode']
    pivot = data.pivot(index=keys, columns='policy', values='objective')
    differences = []
    primary_policy, baseline_policy = protocol['primary_policy'], protocol['primary_baseline']
    comparators = [policy for policy in protocol['policies'] if policy != primary_policy]
    for baseline in comparators:
        delta = (pivot[primary_policy]-pivot[baseline]).rename('difference').reset_index()
        for (variant, env), group in delta.groupby(['variant', 'environment']):
            # These intervals jointly cover all six contrasts, environments and variants
            # by a Bonferroni adjustment to the empirical Student approximation.
            family = len(comparators)*len(protocol['environments'])*len(protocol['variants'])
            differences.append(dict(variant=variant, environment=env,
                                    baseline=baseline,
                                    paired=clustered_estimate(group, 'difference', **budget),
                                    family_adjusted=clustered_estimate(group, 'difference', .05/family, **budget),
                                    family_size=family))
    delta = (pivot[primary_policy]-pivot[baseline_policy]).rename('difference').reset_index()
    nominal = delta[delta.variant == 'nominal']
    primary = stratified_estimate(nominal, 'difference', protocol=protocol)
    primary['policy'] = primary_policy
    primary['baseline'] = baseline_policy
    primary['threshold'] = protocol['deployment_threshold']
    primary['population'] = protocol['deployment_population']
    active_nominal = data[(data.policy == primary_policy) & (data.variant == 'nominal')]
    pnl = stratified_estimate(active_nominal, 'pnl', protocol=protocol)
    accepted = (primary['one_sided_lower'] > protocol['deployment_threshold']
                and pnl['one_sided_lower'] > 0)
    primary['active_net_pnl'] = pnl
    primary['decision'] = ('prefer_active_in_prespecified_synthetic_nominal_population'
                           if accepted else 'retain_matched_noinfo_baseline_pending_stronger_evidence')
    primary['guarantee'] = False
    if protocol['profile'] != 'full':
        primary['decision'] = 'smoke_only_no_deployment_decision'
    overall = []
    for (variant, policy), group in data.groupby(['variant', 'policy']):
        overall.append(dict(variant=variant, policy=policy,
                            objective=stratified_estimate(group, 'objective', protocol=protocol),
                            pnl=stratified_estimate(group, 'pnl', protocol=protocol)))
    return dict(primary_comparison=primary, overall=overall,
                per_environment=per_environment, paired_differences=differences,
                uncertainty_method=('Paired episode differences aggregated within independent pilots; '
                                    'Student intervals over pilot means and stratified Welch-Satterthwaite '
                                    'approximation for the uniform environment mixture. Within-pilot Monte '
                                    'Carlo variance and excess between-pilot variance reported separately. '
                                    'One prespecified primary comparison; other contrasts also provide '
                                    'Bonferroni family-adjusted intervals. Empirical approximations, not '
                                    'finite-sample distribution-free guarantees.'),
                reference_note=('Full-information policy knows nominal parameters and current H, not future '
                                'innovations. Under stresses it remains a frozen nominal-model privileged '
                                'reference, not the optimum for the altered environment.'))
