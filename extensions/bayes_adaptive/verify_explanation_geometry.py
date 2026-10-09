"""Prewritten bounded E2E for physical-grid conservative error allowances.

Run --prepare before the code correction, then --verify against its saved
uniform evidence. No Bellman solve, candidate search or economics is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri, roots_hermitenorm

import explain


HERE = Path(__file__).resolve().parent
LEVELS = [(9, 17), (17, 33), (25, 49), (33, 65), (49, 97)]


def require(value, message):
    if not value:
        raise AssertionError(message)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def specifications(geometry):
    return [{"weight_points": w, "belief_points": b, "quadrature_points": gh,
             "belief_geometry": geometry} for w, b in LEVELS for gh in (1, 25)]


def emit(mode, output):
    config = explain._config(HERE/"config.json")
    if mode == "uniform":
        result = {"implicit": explain.error_envelopes(config),
                  "explicit": explain.error_envelopes(config, specifications("uniform"))}
    else:
        result = explain.error_envelopes(config, specifications(mode))
    write(output, result)


def transport_reference(count):
    nodes, weights = roots_hermitenorm(count)
    weights = weights/math.sqrt(2*math.pi)
    weights /= weights.sum()
    cuts = ndtri(np.clip(np.r_[0., np.cumsum(weights)[:-1], 1.], 0., 1.))
    total = 0.
    for low, high, node in zip(cuts[:-1], cuts[1:], nodes):
        if low == high:
            continue
        middle = min(max(node, low), high)
        for left, right in ((low, middle), (middle, high)):
            if left != right:
                total += quad(lambda z: abs(z-node)*math.exp(-z*z/2)/math.sqrt(2*math.pi),
                              left, right, epsabs=2e-13, epsrel=2e-13)[0]
    return float(total)


def physical_axes(weights, beliefs):
    w = [i/(weights-1) for i in range(weights)]
    b = [math.sin(math.pi*i/(2*(beliefs-1)))**2 for i in range(beliefs)]
    lower = list(b[:(beliefs+1)//2])
    b[beliefs//2:] = [1-value for value in reversed(lower)]
    b[0], b[-1] = 0., 1.
    if beliefs % 2:
        b[beliefs//2] = .5
    return dict(weight0=w, conditional_plus0=b, conditional_plus1=b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--emit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.emit:
        emit(args.emit, out)
        return
    if args.prepare:
        out.mkdir(parents=True, exist_ok=False)
        emit("uniform", out/"uniform-before.json")
        write(out/"preparation.json", {"explain_sha256": explain.file_sha256(HERE/"explain.py"),
              "solver_sha256": explain.file_sha256(HERE/"solver.py"),
              "geometry_amendment_sha256": explain.file_sha256(HERE/"GEOMETRY_AMENDMENT.md"),
              "config_sha256": explain.file_sha256(HERE/"config.json"),
              "uniform_before_sha256": explain.file_sha256(out/"uniform-before.json"),
              "failure_note_sha256": explain.file_sha256(HERE/"EXPLANATION_GEOMETRY_CHECK.md"),
              "harness_sha256": explain.file_sha256(Path(__file__))})
        print(json.dumps({"prepared": True, "output": str(out)}))
        return
    require(not (out/"verification.json").exists(), "Verification receipt already exists")
    preparation = json.loads((out/"preparation.json").read_text())
    require(explain.file_sha256(out/"uniform-before.json") == preparation["uniform_before_sha256"],
            "Uniform reference has changed")
    for filename, key in (("solver.py", "solver_sha256"), ("GEOMETRY_AMENDMENT.md", "geometry_amendment_sha256"),
                          ("config.json", "config_sha256")):
        require(explain.file_sha256(HERE/filename) == preparation[key], "Frozen numerical input changed")
    record = {"passed": False, "scope": "Geometry-aware error-envelope E2E, no solve or economic evaluation",
              "explain_sha256": explain.file_sha256(HERE/"explain.py"),
              "preparation_sha256": explain.file_sha256(out/"preparation.json"), "commands": []}

    def execute(name, expected_success=True):
        command = [sys.executable, str(Path(__file__).resolve()), "--emit", name, "--out", str(out/(name+".json"))]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out/(name+".txt")).write_text(completed.stdout)
        record["commands"].append({"command": command, "returncode": completed.returncode,
                                   "log_sha256": explain.file_sha256(out/(name+".txt"))})
        require((completed.returncode == 0) == expected_success, f"Unexpected CLI result: {name}")
        return completed

    try:
        execute("uniform")
        require((out/"uniform-before.json").read_bytes() == (out/"uniform.json").read_bytes(),
                "Original uniform records changed")
        execute("endpoint_sine")
        endpoint = json.loads((out/"endpoint_sine.json").read_text())
        require(endpoint["certified_bound"] is False and endpoint["solver_rounding_allowance_included"] is False,
                "Envelope became a numerical certificate")
        distances = {gh: transport_reference(gh) for gh in (1, 25)}
        errors = {"axis": 0., "width": 0., "transport": 0., "envelope": 0.}
        summaries = []
        for level in endpoint["levels"]:
            wp, bp, gh = (level[key] for key in ("weight_points", "belief_points", "quadrature_points"))
            axes = physical_axes(wp, bp)
            widths = {name: max(right-left for left, right in zip(axis[:-1], axis[1:])) for name, axis in axes.items()}
            geometry = level["grid_geometry"]
            require(geometry["name"] == "endpoint_sine" and
                    geometry["axes_sha256"] == explain.canonical_hash(geometry["axes"]), "Invalid geometry identity")
            for name in axes:
                errors["axis"] = max(errors["axis"], float(np.max(np.abs(np.array(geometry["axes"][name])-axes[name]))))
                errors["width"] = max(errors["width"], abs(widths[name]-level["max_cell_widths"][name]))
            delta = widths["weight0"]+max(widths["conditional_plus0"], widths["conditional_plus1"])
            require(delta > 1/(wp-1)+1/(bp-1), "Endpoint allowance incorrectly used uniform widths")
            errors["width"] = max(errors["width"], abs(delta-level["delta_grid"]))
            errors["transport"] = max(errors["transport"], abs(distances[gh]-level["normal_transport_distance"]))
            require(level["certified_bound"] is False and len(level["by_horizon"]) == 31, "Incomplete or certified envelope")
            for row in level["by_horizon"]:
                n = row["remaining"]
                cumulative = sum(.231*k+.027 for k in range(n))
                one_sided = 4*(.35/math.sqrt(1-.35**2)/math.sqrt(2*math.pi))*distances[gh]*cumulative
                nodal = cumulative*(delta+4*(.35/math.sqrt(1-.35**2)/math.sqrt(2*math.pi))*distances[gh])
                final = (.231*n+.027)*delta if n else 0.
                expected = {"sum_D_continuations": cumulative, "one_sided_allowance": one_sided,
                            "two_sided_nodal_allowance": nodal, "additional_final_off_grid_interpolation": final,
                            "two_sided_off_grid_allowance": nodal+final}
                for key, value in expected.items():
                    require(math.isfinite(row[key]) and row[key] >= 0, "Nonfinite or negative envelope")
                    errors["envelope"] = max(errors["envelope"], abs(value-row[key]))
                    if n == 0:
                        require(row[key] == 0., "Nonzero terminal allowance")
            summaries.append({"weight_points": wp, "belief_points": bp, "quadrature_points": gh,
                              "independent_max_cell_widths": widths, "independent_delta_grid": delta,
                              "final": level["by_horizon"][-1]})
        rejected = execute("misspelled_endpoint", expected_success=False)
        require("belief_geometry" in rejected.stdout and not (out/"misspelled_endpoint.json").exists(),
                "Unknown geometry was not rejected explicitly")
        require(errors["axis"] < 3e-15 and errors["width"] < 3e-15 and
                errors["transport"] < 3e-12 and errors["envelope"] < 5e-10,
                "Independent physical-width or conservative-envelope reconciliation failed")
        require(endpoint["excludes"] == json.loads((out/"uniform.json").read_text())["implicit"]["excludes"],
                "Floating-error exclusions changed")
        record.update(passed=True, original_uniform_records_identical=True, independently_recomputed_levels=len(summaries),
                      independently_recomputed_horizon_rows=sum(len(level["by_horizon"]) for level in endpoint["levels"]),
                      maximum_absolute_errors=errors, independent_normal_transport=distances,
                      unknown_geometry_rejected=True, numerical_certification_claimed=False, levels=summaries,
                      output_sha256={name: explain.file_sha256(out/name) for name in ("uniform.json", "endpoint_sine.json")})
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write(out/"verification.json", record)
    print(json.dumps({"passed": record["passed"], "horizon_rows": record["independently_recomputed_horizon_rows"],
                      "maximum_absolute_errors": record["maximum_absolute_errors"]}))


if __name__ == "__main__":
    main()
