"""Fail-closed configuration for two explicitly supported benchmark profiles.

This is a fixed research benchmark, not a configurable model-family framework.
Unsupported scientific edits are rejected before any experiment is started.
Numerical resolution overrides are explicit and are honored in the manifest.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json


CONFIGURATION_CONTRACT = "fixed-trade-learning-benchmark-v2"
SMOKE_POPULATION = "smoke integration only: theta=0.35,kappa=0.02, fresh 12-episode pilot; no deployment inference"
_FIXED_FULL = {
    "version": 1, "horizon": 300, "qmax": 5,
    "theta_grid": [.15, .35, .65], "kappa_grid": [.002, .02, .10],
    "pilots": 10, "pilot_episodes": 100, "test_episodes": 100,
    "variants": ["nominal", "state_dependence", "fixed_duration"],
    "policies": ["abstain", "taker", "independent", "myopic", "noinfo", "active", "full_information"],
    "master_seed": 260924138, "primary_policy": "active", "primary_baseline": "noinfo",
    "deployment_threshold": .05,
    "deployment_population": "uniform mixture of nine nominal environments, fresh 100-episode pilot",
    "parameter_prior": "uniform over declared public 3x3 grid",
    "online_updates": "joint parameter/regime posterior within each episode only",
    "algorithm_selection": "fixed in advance; no final-data tuning",
}
_NUMERICAL_FIELDS = {"belief_points", "quadrature_points", "belief_overrides"}
_INTEGER_FIELDS = {"version", "horizon", "qmax", "pilots", "pilot_episodes", "test_episodes", "master_seed"}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python/JSON integer >= {minimum}, not a boolean")


def _numerical(protocol):
    _integer(protocol["belief_points"], "belief_points", 2)
    _integer(protocol["quadrature_points"], "quadrature_points")
    overrides = protocol["belief_overrides"]
    if not isinstance(overrides, dict) or set(overrides) - {"0.65,0.002"}:
        raise ValueError("only the declared public (0.65,0.002) belief refinement is supported")
    for key, value in overrides.items():
        _integer(value, f"belief_overrides[{key}]", 2)


def _same(actual, expected, name):
    # bool == 1 is not a valid scientific configuration match.
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"unsupported {name}: this fixed benchmark requires {expected!r}, got {actual!r}")


def validate_configuration(raw):
    """Validate the raw full-profile JSON before smoke can override any field."""
    expected_keys = set(_FIXED_FULL) | _NUMERICAL_FIELDS
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        supplied = set(raw) if isinstance(raw, dict) else set()
        raise ValueError(f"configuration fields differ: missing={sorted(expected_keys-supplied)}, unknown={sorted(supplied-expected_keys)}")
    for name in _INTEGER_FIELDS:
        _integer(raw[name], name, 0 if name == "master_seed" else 1)
    for name, expected in _FIXED_FULL.items():
        _same(raw[name], expected, name)
    _numerical(raw)
    return deepcopy(raw)


def load_configuration(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON configuration key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON configuration literal: {value}")

    return validate_configuration(json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=nonfinite))


def resolve_protocol(raw, profile, belief_points=None, quadrature_points=None):
    result = validate_configuration(raw)
    if profile not in ("full", "smoke"):
        raise ValueError("profile must be 'full' or 'smoke'")
    result["profile"] = profile
    result["environments"] = [list(pair) for pair in itertools.product(result["theta_grid"], result["kappa_grid"])]
    if profile == "smoke":
        result.update(horizon=12, qmax=2, pilots=2, pilot_episodes=12, test_episodes=16,
                      environments=[[.35, .02]], master_seed=260924136,
                      belief_points=21, quadrature_points=15, belief_overrides={},
                      deployment_population=SMOKE_POPULATION)
    if belief_points is not None:
        result["belief_points"] = belief_points
        result["belief_overrides"] = {}
    if quadrature_points is not None:
        result["quadrature_points"] = quadrature_points
    return validate_protocol(result)


def validate_protocol(protocol):
    """Validate an already resolved protocol, including direct execute callers.

    Original full-run manifests have the same fields and remain analyzable.
    Legacy smoke manifests that falsely name the full deployment population do
    not pass: explicitly resolve the supported smoke profile instead.
    """
    expected_keys = set(_FIXED_FULL) | _NUMERICAL_FIELDS | {"profile", "environments"}
    if not isinstance(protocol, dict) or set(protocol) != expected_keys:
        supplied = set(protocol) if isinstance(protocol, dict) else set()
        raise ValueError(f"resolved protocol fields differ: missing={sorted(expected_keys-supplied)}, unknown={sorted(supplied-expected_keys)}")
    profile = protocol["profile"]
    if profile not in ("full", "smoke"):
        raise ValueError("resolved profile must be 'full' or 'smoke'")
    expected = deepcopy(_FIXED_FULL)
    environments = [list(pair) for pair in itertools.product(expected["theta_grid"], expected["kappa_grid"])]
    if profile == "smoke":
        expected.update(horizon=12, qmax=2, pilots=2, pilot_episodes=12, test_episodes=16,
                        master_seed=260924136, deployment_population=SMOKE_POPULATION)
        environments = [[.35, .02]]
        if protocol["belief_overrides"] != {}:
            raise ValueError("the smoke profile does not use a public-model belief override")
    for name in _INTEGER_FIELDS:
        _integer(protocol[name], name, 0 if name == "master_seed" else 1)
    for name, value in expected.items():
        _same(protocol[name], value, name)
    _same(protocol["environments"], environments, "environments")
    _numerical(protocol)
    return deepcopy(protocol)


def validate_inference_support(protocol, candidate_theta, candidate_kappa, policies):
    protocol = validate_protocol(protocol)
    expected = list(itertools.product(protocol["theta_grid"], protocol["kappa_grid"]))
    if len(candidate_theta) != len(expected) or len(candidate_kappa) != len(expected):
        raise ValueError("actual inference support differs from the resolved public grid")
    if list(zip(candidate_theta, candidate_kappa)) != expected:
        raise ValueError("actual inference support differs from the resolved public grid")
    if list(policies) != protocol["policies"]:
        raise ValueError("actual policy implementation differs from the resolved policy set")


def environment_names(protocol):
    return [f"theta{theta:g}_kappa{kappa:g}" for theta, kappa in protocol["environments"]]


def protocol_fingerprint(protocol):
    resolved = validate_protocol(protocol)
    return hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
