"""Prewritten bounded end-to-end check for the v3 numerical geometry amendment.

This executes real complete short-horizon control families in cold processes.
It is implementation evidence, never the final 30-horizon numerical gate.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from trade_learning.numerics import canonical_hash, file_sha256


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def worker(args):
    if args.revision == "v2":
        spec = importlib.util.spec_from_file_location("preserved_solver_v2", HERE/"revisions/solver_v2.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        import solver as module
    options = dict(horizon=args.horizon, weight_points=args.weight_points,
                   belief_points=args.belief_points, quadrature_points=args.quadrature_points)
    if args.revision != "v2":
        options["belief_geometry"] = args.belief_geometry
    family = module.solve_family(module.SolverSpec(**options), directory=args.cache)
    print(json.dumps({"artifact_sha256": family.metadata["artifact_sha256"]}), flush=True)


def physical_function(c):
    w, b0, b1 = np.moveaxis(c, -1, 0)
    return 1+2*w-3*b0+.5*b1+1.25*w*b0-.3*w*b1+.7*b0*b1+.2*w*b0*b1


def verify(args):
    import solver
    out, cache = args.out.resolve(), args.cache.resolve()
    require(not out.exists() and not cache.exists(), "E2E output and cache must be fresh")
    out.mkdir(parents=True)
    cache.mkdir(parents=True)
    result = {"passed": False, "scope": "Bounded production E2E; no full-horizon acceptance or economic evaluation",
              "solver_sha256": file_sha256(HERE/"solver.py"),
              "preserved_v2_sha256": file_sha256(HERE/"revisions/solver_v2.py"),
              "harness_sha256": file_sha256(Path(__file__)),
              "amendment_sha256": file_sha256(HERE/"GEOMETRY_AMENDMENT.md"),
              "uniform_equivalence": [], "commands": [], "rejections": []}

    def build(label, revision, geometry, dimensions):
        horizon, weights, beliefs, quadrature = dimensions
        destination = cache/label
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--revision", revision,
                   "--belief-geometry", geometry, "--horizon", str(horizon), "--weight-points", str(weights),
                   "--belief-points", str(beliefs), "--quadrature-points", str(quadrature), "--cache", str(destination)]
        with (out/f"{label}.txt").open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        result["commands"].append({"label": label, "command": command, "returncode": completed.returncode,
                                   "log_sha256": file_sha256(out/f"{label}.txt")})
        require(completed.returncode == 0, f"{label} failed; see preserved log")
        manifest = json.loads((destination/"complete.json").read_text())
        for item in manifest["arrays"]:
            require(file_sha256(destination/item["file"]) == item["sha256"], "Array content differs from manifest")
        write(out/f"{label}-complete.json", manifest)
        return destination, manifest

    try:
        for index, dimensions in enumerate(((4, 5, 7, 15), (7, 5, 9, 25))):
            _, old = build(f"uniform-v2-{index}", "v2", "uniform", dimensions)
            _, new = build(f"uniform-v3-{index}", "v3", "uniform", dimensions)
            require(old["arrays"] == new["arrays"], "V3 storage/geometry revision changed uniform numerical arrays")
            require(len(new["arrays"]) == 11, "Expected all eleven full-horizon numerical arrays")
            result["uniform_equivalence"].append({"dimensions": dimensions, "array_count": len(new["arrays"]),
                "arrays_identical": True, "v2_artifact": old["artifact_sha256"], "v3_artifact": new["artifact_sha256"]})

        path, first = build("endpoint-first", "v3", "endpoint_sine", (7, 5, 9, 25))
        _, cold = build("endpoint-cold", "v3", "endpoint_sine", (7, 5, 9, 25))
        require(first["arrays"] == cold["arrays"], "Cold endpoint family is not byte-identical")
        family = solver.load_family(path)
        rng = np.random.default_rng(260927530)
        coordinates = np.concatenate((rng.uniform(size=(257, 3)),
            np.stack(np.meshgrid([0., .5, 1.], [0., .5, 1.], [0., .5, 1.], indexing="ij"), axis=-1).reshape(-1, 3)))
        nodal = physical_function(solver.grid_coordinates(family.spec))
        interpolated, mass = np.zeros(len(coordinates)), np.zeros(len(coordinates))
        for indices, coefficients in solver._corners(coordinates, family.spec):
            require(np.min(coefficients) >= -1e-15, "Negative physical interpolation coefficient")
            interpolated += coefficients*nodal[indices]
            mass += coefficients
        interpolation_error = float(np.max(np.abs(interpolated-physical_function(coordinates))))
        require(interpolation_error < 4e-15 and np.max(np.abs(mass-1)) < 2e-15,
                "Physical multilinear reconstruction or interpolation mass failed")
        q = rng.integers(-2, 3, size=len(coordinates))
        x = rng.integers(-1, 2, size=len(coordinates))
        joint = solver.coordinates_to_joint(coordinates)
        reference = family.q_values("weighted_q", 1, q, x, joint)
        one_step = {}
        for kind in solver.DEFAULT_RETAIN_Q:
            actual = family.q_values(kind, 1, q, x, joint)
            require(np.array_equal(np.isfinite(actual), np.isfinite(reference)), "One-step admissibility changed")
            mask = np.isfinite(actual)
            error = float(np.max(np.abs(actual[mask]-reference[mask])))
            require(error < 2e-10, "One-step liquidation differs from independent high-resolution known-model control")
            one_step[kind] = error
        # Mutate only metadata in the bounded fixture and restore it after every
        # attempt. Re-signing must not override geometry/schema validation.
        manifest_path = path/"complete.json"
        original_text = manifest_path.read_text()
        mutations = {
            "v2_format": lambda m: m.update(format_version=2),
            "geometry_name": lambda m: m["specification"].update(belief_geometry="uniform"),
            "axis_values": lambda m: m["grid_geometry"]["axes"]["conditional_plus0"].__setitem__(1, .123),
            "interpolation_recipe": lambda m: m["grid_geometry"].update(interpolation="sine-parameter-interpolation"),
            "axis_digest": lambda m: m["grid_geometry"].update(axes_sha256="0"*64),
        }
        for name, mutation in mutations.items():
            altered = copy.deepcopy(first)
            mutation(altered)
            altered["artifact_sha256"] = canonical_hash({k: v for k, v in altered.items() if k != "artifact_sha256"})
            write(manifest_path, altered)
            try:
                solver.load_family(path)
            except (ValueError, KeyError) as error:
                result["rejections"].append({"case": name, "rejected": True, "message": str(error)})
            else:
                raise AssertionError(f"Changed geometry loaded: {name}")
            finally:
                manifest_path.write_text(original_text)
        result.update(passed=True, cold_arrays_identical=True, cold_array_count=len(cold["arrays"]),
                      physical_interpolation_max_error=interpolation_error,
                      one_step_max_errors=one_step, endpoint_artifact=first["artifact_sha256"],
                      endpoint_geometry=first["grid_geometry"],
                      maximum_transition_mass_error=max(op["max_transition_mass_error"] for op in first["metadata"]["operators"]))
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write(out/"verification.json", result)
    print(json.dumps({"passed": result["passed"], "uniform_complete_families": len(result["uniform_equivalence"]),
                      "cold_array_count": result["cold_array_count"], "rejections": len(result["rejections"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--revision", choices=("v2", "v3"), default="v3")
    parser.add_argument("--belief-geometry", choices=("uniform", "endpoint_sine"), default="endpoint_sine")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--weight-points", type=int, default=5)
    parser.add_argument("--belief-points", type=int, default=9)
    parser.add_argument("--quadrature-points", type=int, default=25)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)


if __name__ == "__main__":
    main()
