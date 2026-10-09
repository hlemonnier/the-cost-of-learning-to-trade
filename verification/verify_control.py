"""Pre-implementation control verification, with reproducible JSON evidence.

Run: PYTHONPATH=src .venv/bin/python verification/verify_control.py [--quick]
The expected one-step calculation deliberately does not use solver internals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from trade_learning.control import (
    solve_control, save_control, load_control, feedback_suppressed_q,
)
from trade_learning.numerics import stable_argmax, numerical_contract, TIE_ABSOLUTE_TOLERANCE


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = [(bid, ask) for bid in (-1, 0, 1) for ask in (-1, 0, 1)]
MIRROR = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8, 10, 9])
INITIAL_TOL = 0.001
SUP_TOL = 0.005
ROBUST_GAP = 0.001
ROBUST_DISAGREEMENT = 0.005


def normal_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def one_step_independent(theta, kappa, qmax, beliefs):
    """Scalar integration of actual marked cash flows plus liquidation."""
    out = np.full((2 * qmax + 1, 3, len(beliefs), 11), -np.inf)
    for qi, q in enumerate(range(-qmax, qmax + 1)):
        for xi, x in enumerate((-1, 0, 1)):
            for bi, b in enumerate(beliefs):
                for a, (kb, ka) in enumerate(ACTIONS):
                    if (kb >= 0 and q == qmax) or (ka >= 0 and q == -qmax):
                        continue
                    sides = [(1, kb), (-1, ka)]
                    sides = [(s, k) for s, k in sides if k >= 0]

                    def integrand(z):
                        total = 0.0
                        for h, hprob in ((-1, 1-b), (1, b)):
                            probs = [ndtr((ndtri(1/(1+math.exp(.3+.7*k+.2*s*x)))
                                           - h*theta*s*z)/math.sqrt(1-theta*theta))
                                     for s, k in sides]
                            for bits in range(1 << len(sides)):
                                dq = 0
                                cash_relative = 0.0
                                prob = hprob
                                for j, (s, k) in enumerate(sides):
                                    f = (bits >> j) & 1
                                    prob *= probs[j] if f else 1-probs[j]
                                    dq += s*f
                                    cash_relative += f*(.025+.025*k-.001)
                                total += prob*(cash_relative+(q+dq)*(.03*x+.30*z)
                                               -.002*q*q-.027*abs(q+dq))
                        return normal_pdf(z)*total

                    out[qi, xi, bi, a] = quad(integrand, -11, 11, epsabs=2e-11,
                                               epsrel=2e-11, limit=200)[0]
                for a, s in ((9, 1), (10, -1)):
                    if abs(q+s) <= qmax:
                        out[qi, xi, bi, a] = -.027+(q+s)*.03*x-.002*q*q-.027*abs(q+s)
    return out


def interpolate_last_axis(arr, source, target):
    flat = arr.reshape(-1, arr.shape[-1])
    return np.stack([np.interp(target, source, row) for row in flat]).reshape(
        *arr.shape[:-1], len(target))


def compare_solutions(coarse, fine, initial_tol=INITIAL_TOL, sup_tol=SUP_TOL,
                      robust_gap=ROBUST_GAP, robust_disagreement=ROBUST_DISAGREEMENT):
    if (coarse.horizon, coarse.qmax, coarse.theta, coarse.kappa, coarse.mode) != (
            fine.horizon, fine.qmax, fine.theta, fine.kappa, fine.mode) or coarse.horizon < 1:
        raise ValueError('Refinement must compare the same positive-horizon control problem')
    horizons = []
    for n in range(1, coarse.horizon+1):
        vc = coarse.values[n]
        vf = interpolate_last_axis(fine.values[n], fine.beliefs, coarse.beliefs)
        qf = np.moveaxis(fine.q_values[n], -1, -2)
        qf = interpolate_last_axis(qf, fine.beliefs, coarse.beliefs)
        qf = np.moveaxis(qf, -2, -1)
        qc = coarse.q_values[n]
        ac, af = stable_argmax(qc), stable_argmax(qf)
        ordered = np.sort(qf, axis=-1)
        gap = ordered[..., -1]-ordered[..., -2]
        robust = gap > robust_gap
        numerical_tie = gap <= TIE_ABSOLUTE_TOLERANCE
        near_tie = ~robust
        disagreement = ac != af
        sup = float(np.max(np.abs(vc-vf)))
        initial = abs(float(np.interp(.5, coarse.beliefs, vc[coarse.qmax, 1])
                            - np.interp(.5, coarse.beliefs, vf[coarse.qmax, 1])))
        robust_fraction = float(np.mean(disagreement[robust])) if robust.any() else 0.0
        regret = np.max(qf, axis=-1)-np.take_along_axis(qf, ac[..., None], -1)[..., 0]
        horizons.append({
            'remaining': n, 'states': int(disagreement.size),
            'initial_value_change': initial, 'maximum_common_state_value_change': sup,
            'action_disagreements': int(disagreement.sum()),
            'robust_states': int(robust.sum()),
            'robust_action_disagreements': int(np.sum(disagreement & robust)),
            'robust_action_disagreement_fraction': robust_fraction,
            'numerical_tie_states': int(numerical_tie.sum()),
            'numerical_tie_action_disagreements': int(np.sum(disagreement & numerical_tie)),
            'near_tie_states': int(near_tie.sum()),
            'near_tie_action_disagreements': int(np.sum(disagreement & near_tie)),
            'maximum_action_loss_under_fine_scores': float(np.max(regret)),
            'passes_predeclared_rule': initial <= initial_tol and sup <= sup_tol
                                      and robust_fraction <= robust_disagreement,
        })
    totals = {key: sum(row[key] for row in horizons) for key in
              ('states', 'action_disagreements', 'robust_states', 'robust_action_disagreements',
               'numerical_tie_states', 'numerical_tie_action_disagreements',
               'near_tie_states', 'near_tie_action_disagreements')}
    return {
        "coarse": [len(coarse.beliefs), coarse.quadrature_points],
        "fine": [len(fine.beliefs), fine.quadrature_points],
        'remaining_horizons_checked': list(range(1, coarse.horizon+1)),
        'horizon_coverage': 'every positive remaining horizon; terminal row excluded',
        'numerical_contract': numerical_contract(),
        'initial_value_change': horizons[-1]['initial_value_change'],
        'maximum_initial_state_value_change_over_horizons': max(row['initial_value_change'] for row in horizons),
        'maximum_common_state_value_change': max(row['maximum_common_state_value_change'] for row in horizons),
        'raw_action_disagreement_fraction': totals['action_disagreements']/totals['states'],
        'robust_action_disagreement_fraction': totals['robust_action_disagreements']/max(1, totals['robust_states']),
        'maximum_per_horizon_robust_action_disagreement_fraction': max(row['robust_action_disagreement_fraction'] for row in horizons),
        'maximum_action_loss_under_fine_scores': max(row['maximum_action_loss_under_fine_scores'] for row in horizons),
        'near_tie_definition': 'fine top-two gap <= robust_gap; includes numerical ties <= 1e-12',
        **totals, 'per_horizon': horizons,
        'passes_predeclared_rule': all(row['passes_predeclared_rule'] for row in horizons),
    }


def feedback_check(solution):
    n = solution.horizon
    observed = solution.q_values[n]
    suppressed = feedback_suppressed_q(solution, n)
    active_choice = stable_argmax(observed)
    blind_choice = stable_argmax(suppressed)
    finite = np.isfinite(observed)
    bonus = np.zeros_like(observed)
    bonus[finite] = observed[finite]-suppressed[finite]
    mask = active_choice != blind_choice
    loss = np.max(observed, -1)-np.take_along_axis(observed, blind_choice[..., None], -1)[..., 0]
    examples = []
    indices = np.argwhere(mask)
    order = sorted(indices, key=lambda loc: loss[tuple(loc)], reverse=True)[:12]
    for qi, xi, bi in order:
        idx = (int(qi), int(xi), int(bi))
        a, a0 = int(active_choice[idx]), int(blind_choice[idx])
        examples.append({"remaining": n, "q": int(qi)-solution.qmax,
                         "x": int(xi)-1, "belief": float(solution.beliefs[bi]),
                         "active_action": a, "suppressed_action": a0,
                         "advantage_in_active_Q": float(loss[idx]),
                         "active_immediate_feedback_bonus": float(bonus[idx+(a,)]),
                         "suppressed_immediate_feedback_bonus": float(bonus[idx+(a0,)]),
                         "active_Q": float(observed[idx+(a,)]),
                         "suppressed_action_active_Q": float(observed[idx+(a0,)])})
    # A report illustration selected after solving, not a statistical comparison.
    bi = int(np.argmin(np.abs(solution.beliefs-.75)))
    b = float(solution.beliefs[bi])
    idx = (solution.qmax, 1, bi)
    a, a0 = int(active_choice[idx]), int(blind_choice[idx])

    def zero_state_stage_reward(action):
        if action >= 9:
            return -.027
        total = 0.
        for side, depth in zip((1, -1), ACTIONS[action]):
            if depth >= 0:
                p = 1/(1+math.exp(.3+.7*depth))
                total += p*(.025+.025*depth-.001)
                total -= .30*(2*b-1)*solution.theta*normal_pdf(float(ndtri(p)))
        return total

    illustration = {"selection": "worked state illustration after solving; not a statistical test",
                    "remaining": n, "q": 0, "x": 0, "belief": b,
                    "active_action": a, "suppressed_action": a0,
                    "active_expected_stage_reward": zero_state_stage_reward(a),
                    "suppressed_expected_stage_reward": zero_state_stage_reward(a0),
                    "active_Q_advantage": float(loss[idx]),
                    "active_feedback_bonus": float(bonus[idx+(a,)])}
    return {"same_continuation": "active V[remaining-1] used for both choices",
            "action_differences": int(mask.sum()), "states": int(mask.size),
            "meaningful_differences_above_0.001": int(np.sum(loss > ROBUST_GAP)),
            "maximum_active_score_loss_from_suppressing_feedback": float(np.max(loss)),
            "minimum_feedback_bonus": float(np.min(bonus[finite])),
            "economic_illustration": illustration,
            "examples": examples}


def practical_refinement(output):
    """A predeclared numerical experiment at the actual deployment horizon."""
    started = time.perf_counter()
    result = {"protocol": "report/control_notes.md: practical-horizon protocol",
              "horizon": 300, "qmax": 5,
              "tolerances": {"initial": .01, "sup": .03,
                             "robust_gap": .005, "robust_disagreement": .005},
              "families": []}

    def comparison(coarse, fine):
        return compare_solutions(coarse, fine, .01, .03, .005, .005)

    def run(theta, kappa, mode, b, g, family):
        solution = solve_control(theta, kappa, 300, 5, b, g, mode)
        family["runs"].append({"belief_points": b, "quadrature_points": g,
                               "initial_value": float(solution.values[-1, 5, 1, b//2]),
                               "metadata": solution.metadata})
        return solution

    for theta in (.15, .35, .65):
        for kappa in (.002, .02, .10):
            for mode in ("active", "noinfo"):
                family = {"theta": theta, "kappa": kappa, "mode": mode,
                          "runs": [], "belief_grid": [], "quadrature": []}
                coarse = run(theta, kappa, mode, 81, 81, family)
                for b in (161, 321):
                    fine = run(theta, kappa, mode, b, 81, family)
                    family["belief_grid"].append(comparison(coarse, fine))
                    coarse = fine
                if not family["belief_grid"][-1]["passes_predeclared_rule"]:
                    fine = run(theta, kappa, mode, 641, 81, family)
                    family["belief_grid"].append(comparison(coarse, fine))
                    coarse = fine
                b = len(coarse.beliefs)
                fine = run(theta, kappa, mode, b, 161, family)
                family["quadrature"].append(comparison(coarse, fine))
                coarse = fine
                if not family["quadrature"][-1]["passes_predeclared_rule"]:
                    fine = run(theta, kappa, mode, b, 321, family)
                    family["quadrature"].append(comparison(coarse, fine))
                    coarse = fine
                family["final_settings"] = [b, coarse.quadrature_points]
                family["all_final_axis_checks_passed"] = (
                    family["belief_grid"][-1]["passes_predeclared_rule"]
                    and family["quadrature"][-1]["passes_predeclared_rule"])
                result["families"].append(family)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, indent=2)+"\n")
                print(json.dumps({"theta":theta,"kappa":kappa,"mode":mode,
                                  "settings":family["final_settings"],
                                  "passed":family["all_final_axis_checks_passed"]}), flush=True)
                del coarse, fine
    result["all_final_axis_checks_passed"] = all(
        f["all_final_axis_checks_passed"] for f in result["families"])
    result["recommended_common_settings"] = [
        max(f["final_settings"][0] for f in result["families"]),
        max(f["final_settings"][1] for f in result["families"])]
    result["elapsed_seconds"] = time.perf_counter()-started
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output":str(output),
                      "all_final_axis_checks_passed":result["all_final_axis_checks_passed"],
                      "recommended_common_settings":result["recommended_common_settings"],
                      "elapsed_seconds":result["elapsed_seconds"]}), flush=True)
    if not result["all_final_axis_checks_passed"]:
        raise SystemExit(1)


def practical_extra_refinement(output):
    """Declared additional level for the first sweep's single failing family."""
    started = time.perf_counter()
    result = json.loads(output.read_text())
    result.setdefault("first_sweep_all_final_axis_checks_passed", result["all_final_axis_checks_passed"])
    result["extension_protocol"] = "report/control_notes.md: additional refinement declared before new results"
    for mode in ("active", "noinfo"):
        family = next(f for f in result["families"]
                      if f["theta"] == .65 and f["kappa"] == .002 and f["mode"] == mode)

        def run(b, g):
            s = solve_control(.65, .002, 300, 5, b, g, mode)
            family["runs"].append({"belief_points":b,"quadrature_points":g,
                                   "initial_value":float(s.values[-1,5,1,b//2]),
                                   "metadata":s.metadata,"phase":"additional_refinement"})
            print(json.dumps({"phase":"additional_refinement","mode":mode,
                              "belief_points":b,"quadrature_points":g,
                              "seconds":s.metadata["solve_seconds"]}),flush=True)
            return s

        coarse = run(641, 81)
        fine = run(1281, 81)
        family["belief_grid"].append(compare_solutions(coarse,fine,.01,.03,.005,.005))
        coarse = fine
        for g in ((161,321) if mode == "active" else (161,)):
            fine = run(1281,g)
            family["quadrature"].append(compare_solutions(coarse,fine,.01,.03,.005,.005))
            coarse = fine
        family["largest_checked_settings"] = [1281,coarse.quadrature_points]
        family["final_settings"] = [1281,161]
        family["all_final_axis_checks_passed"] = (
            family["belief_grid"][-1]["passes_predeclared_rule"]
            and all(c["passes_predeclared_rule"] for c in family["quadrature"][-2:]))
        output.write_text(json.dumps(result,indent=2)+"\n")
        del coarse,fine
    result["all_final_axis_checks_passed"] = all(f["all_final_axis_checks_passed"]
                                                  for f in result["families"])
    result["recommended_common_settings"] = [641,161]
    result["recommended_public_model_overrides"] = [
        {"theta":.65,"kappa":.002,"belief_points":1281,"quadrature_points":161,
         "modes":["active","noinfo"]}]
    result["extension_elapsed_seconds"] = time.perf_counter()-started
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(output),
                      "all_final_axis_checks_passed":result["all_final_axis_checks_passed"],
                      "extension_elapsed_seconds":result["extension_elapsed_seconds"]}),flush=True)
    if not result["all_final_axis_checks_passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--practical", action="store_true")
    parser.add_argument("--practical-extra", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tables-out", type=Path,
                        help="Optionally persist reduced tables into a new directory; existing artifacts are refused")
    args = parser.parse_args()
    if args.practical_extra:
        return practical_extra_refinement(args.output or ROOT/"outputs/verification/audit_corrections/practical_refinement.json")
    if args.practical:
        return practical_refinement(args.output or ROOT/"outputs/verification/audit_corrections/practical_refinement.json")
    args.output = args.output or ROOT/"outputs/verification/audit_corrections/reduced_control.json"
    started = time.perf_counter()
    evidence = {"protocol": "report/control_notes.md; verification/numerical_failure_modes.md",
                'numerical_contract': numerical_contract(),
                "quick": args.quick, "checks": {}, "runs": [], "refinements": {},
                "tolerances": {"initial": INITIAL_TOL, "sup": SUP_TOL,
                               "robust_gap": ROBUST_GAP,
                               "robust_disagreement": ROBUST_DISAGREEMENT}}

    def solve(theta=.35, kappa=.02, horizon=30, qmax=2, b=81, g=81, mode="active"):
        s = solve_control(theta, kappa, horizon, qmax, b, g, mode)
        evidence["runs"].append({"theta":theta,"kappa":kappa,"horizon":horizon,
                                  "qmax":qmax,"belief_points":len(s.beliefs),
                                  "quadrature_points":g,"mode":mode,
                                  "initial_value":float(np.interp(.5,s.beliefs,s.values[-1,qmax,1])),
                                  "metadata":s.metadata})
        return s

    def check(name, condition, **details):
        evidence["checks"][name] = {"passed": bool(condition), **details}

    s1 = solve(horizon=1,b=5,g=81)
    expected = one_step_independent(.35,.02,2,s1.beliefs)
    finite = np.isfinite(expected)
    error = float(np.max(np.abs(expected[finite]-s1.q_values[1][finite])))
    check("one_step_independent_accounting", error < 2e-8, maximum_error=error)
    check("one_step_invalid_actions", np.array_equal(np.isfinite(expected),np.isfinite(s1.q_values[1])))
    terminal = -.027*np.abs(np.arange(-2,3))[:,None,None]
    check("terminal_liquidation",np.allclose(s1.values[0],terminal,atol=1e-14))

    hz = 8 if args.quick else 30
    zero = [solve(theta=0,horizon=hz,b=41,g=21,mode=m)
            for m in ("active","noinfo","independent","full")]
    zeroerr = max(float(np.max(np.abs(z.values-zero[0].values))) for z in zero[1:3])
    fullzeroerr = float(np.max(np.abs(zero[-1].values[...,0]-zero[0].values[...,0])))
    check("zero_dependence_mode_equality",max(zeroerr,fullzeroerr)<1e-10,
          maximum_error=max(zeroerr,fullzeroerr))

    if args.quick:
        reference = solve(horizon=hz,b=41,g=41)
    else:
        grid = [solve(b=b,g=81) for b in (41,81,161,321)]
        evidence["refinements"]["belief_grid"] = [compare_solutions(a,b) for a,b in zip(grid,grid[1:])]
        integration = [solve(b=321,g=g) if g !=81 else grid[-1] for g in (21,41,81,161)]
        evidence["refinements"]["quadrature"] = [compare_solutions(a,b) for a,b in zip(integration,integration[1:])]
        reference = integration[-1]
        check("belief_grid_stopping_rule",evidence["refinements"]["belief_grid"][-1]["passes_predeclared_rule"])
        check("quadrature_stopping_rule",evidence["refinements"]["quadrature"][-1]["passes_predeclared_rule"])

    noinfo = solve(horizon=hz,b=len(reference.beliefs),g=reference.quadrature_points,mode="noinfo")
    full = solve(horizon=hz,b=2,g=reference.quadrature_points,mode="full")
    b = reference.beliefs
    fullmix = full.values[...,0,None]*(1-b)+full.values[...,1,None]*b
    full_gap = float(np.min(fullmix-reference.values))
    info_gap = float(np.min(reference.values-noinfo.values))
    check("full_information_dominance",full_gap>=-1e-8,minimum_gap=full_gap)
    check("active_planning_dominates_no_learning_planning",info_gap>=-1e-8,minimum_gap=info_gap,
          note="Stored noinfo V is its planning value, not its runtime filtered policy value.")

    mirrored = reference.q_values[:,::-1,::-1,:,MIRROR]
    finite = np.isfinite(reference.q_values)&np.isfinite(mirrored)
    symerr=float(np.max(np.abs(reference.q_values[finite]-mirrored[finite])))
    check("sign_symmetry",symerr<2e-10,maximum_error=symerr)
    convex = float(np.min(np.diff(reference.values,n=2,axis=-1)))
    check("belief_convexity",convex>=-1e-8,minimum_second_difference=convex)
    check("transition_mass",reference.metadata["max_transition_mass_error"]<1e-12,
          maximum_error=reference.metadata["max_transition_mass_error"])
    check("float64_tables",reference.values.dtype==np.float64 and reference.q_values.dtype==np.float64)

    taker=solve(horizon=hz,b=11,g=21,mode="taker")
    check("taker_action_restriction",np.all(np.isneginf(taker.q_values[1:,...,1:9])))
    with tempfile.TemporaryDirectory() as tmp:
        save_control(Path(tmp)/"table",reference)
        restored=load_control(Path(tmp)/"table",mmap_mode="r")
        check("serialization",np.array_equal(restored.q_values,reference.q_values)
              and np.array_equal(restored.values,reference.values)
              and np.array_equal(restored.beliefs,reference.beliefs)
              and restored.theta==reference.theta and restored.kappa==reference.kappa)

    evidence["same_state_feedback_contrast"]=feedback_check(reference)
    evidence["planner_comparison"]={
        "active_initial":float(np.interp(.5,reference.beliefs,reference.values[-1,2,1])),
        "noinfo_planning_initial":float(np.interp(.5,noinfo.beliefs,noinfo.values[-1,2,1])),
        "full_initial":float(np.mean(full.values[-1,2,1])),
        "taker_initial":float(np.interp(.5,taker.beliefs,taker.values[-1,2,1])),
        "active_noinfo_action_differences":int(np.sum(stable_argmax(reference.q_values[-1])
                                                        !=stable_argmax(noinfo.q_values[-1])))}
    evidence["elapsed_seconds"]=time.perf_counter()-started
    evidence["all_checks_passed"]=all(x["passed"] for x in evidence["checks"].values())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,indent=2)+"\n")
    if not args.quick and args.tables_out:
        save_control(args.tables_out/"reduced_active",reference)
        save_control(args.tables_out/"reduced_full",full)
        save_control(args.tables_out/"reduced_noinfo",noinfo)
    print(json.dumps({"output":str(args.output),"all_checks_passed":evidence["all_checks_passed"],
                      "elapsed_seconds":evidence["elapsed_seconds"],"checks":evidence["checks"]},indent=2))
    if not evidence["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
