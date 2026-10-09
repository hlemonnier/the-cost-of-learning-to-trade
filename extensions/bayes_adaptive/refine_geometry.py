"""Run the explicit endpoint-belief mesh amendment without changing the experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from refine_numerical import amended_config, amendment_identity
from run import HERE, file_sha256, require, write_json
from solver import run_numerical_refinement


def geometry_identity():
    return {**amendment_identity(),
            "belief_geometry": "endpoint_sine",
            "geometry_amendment_sha256": file_sha256(HERE / "GEOMETRY_AMENDMENT.md"),
            "geometry_wrapper_sha256": file_sha256(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    identity = geometry_identity()
    result = run_numerical_refinement(amended_config(), args.output, args.cache,
                                     progress=lambda row: print(json.dumps(row), flush=True),
                                     belief_geometry="endpoint_sine")
    require(identity == geometry_identity(), "Geometry amendment inputs changed during refinement")
    result.update(identity)
    result["amendment_scope"] = ("Endpoint-resolved conditional-belief axes and bounded storage; "
                                 "registered statistical design and numerical acceptance criteria unchanged")
    write_json(args.output / "numerical_checks.json", result)
    print(json.dumps({"status": result["status"],
                      "passes_predeclared_rule": result["passes_predeclared_rule"],
                      "selected_resolution": result["selected_resolution"]}), flush=True)


if __name__ == "__main__":
    main()
