"""Execute the explicitly amended numerical ladder, preserving the statistical freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run import HERE, CONFIG_SHA256, file_sha256, read_protocol, require, write_json
from solver import run_numerical_refinement


def amended_config():
    original = read_protocol()
    path = HERE / "numerical_config.json"
    numeric = json.loads(path.read_text())
    expected = {**original, "joint_grid_levels": original["joint_grid_levels"] + [[49, 97, 97]]}
    require(numeric == expected, "Numerical amendment may only append the approved extra grid")
    return path


def amendment_identity():
    return {"statistical_protocol_sha256": CONFIG_SHA256,
            "numerical_amendment_sha256": file_sha256(HERE / "NUMERICAL_AMENDMENT.md"),
            "numerical_wrapper_sha256": file_sha256(Path(__file__)),
            "numerical_config_file": "numerical_config.json",
            "numerical_config_sha256": file_sha256(amended_config())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    identity = amendment_identity()
    result = run_numerical_refinement(amended_config(), args.output, args.cache,
                                     progress=lambda row: print(json.dumps(row), flush=True))
    require(identity == amendment_identity(), "Numerical amendment inputs changed during execution")
    result.update(identity)
    result["amendment_scope"] = "One extra joint mesh; original statistical design and all tolerances unchanged"
    write_json(args.output / "numerical_checks.json", result)
    print(json.dumps({"status": result["status"],
                      "passes_predeclared_rule": result["passes_predeclared_rule"],
                      "selected_resolution": result["selected_resolution"]}), flush=True)


if __name__ == "__main__":
    main()
