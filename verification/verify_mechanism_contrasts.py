"""Post hoc paired noinfo-minus-myopic horizon-planning contrast.

The frozen question, failure catalogue and interpretation limits are in
report/mechanism_identification.md. This script imports no production code.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


ENVIRONMENTS = [f"theta{theta:g}_kappa{kappa:g}"
                for theta in (.15, .35, .65) for kappa in (.002, .02, .1)]
POLICIES = ("noinfo", "myopic")
PILOTS = 10
EPISODES = 100


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(paired: dict[tuple[str, int, int], float], environment: str) -> dict:
    pilot_samples = [np.array([paired[(environment, pilot, episode)]
                               for episode in range(EPISODES)], dtype=np.float64)
                     for pilot in range(PILOTS)]
    means = np.array([float(values.mean()) for values in pilot_samples])
    mean = float(means.mean())
    observed = float(means.var(ddof=1))
    within = float(np.mean([values.var(ddof=1) / EPISODES for values in pilot_samples]))
    se = float(np.sqrt(observed / PILOTS))
    critical = float(t.ppf(.975, PILOTS - 1))
    return {
        "mean": mean, "se": se, "low": mean - critical * se,
        "high": mean + critical * se, "df": PILOTS - 1,
        "pilots": PILOTS, "paired_episodes": PILOTS * EPISODES,
        "observed_pilot_mean_variance": observed,
        "within_pilot_mean_variance": within,
        "estimated_between_pilot_variance": max(observed - within, 0.0),
    }


def compute(root: Path) -> dict:
    raw = root / "outputs/full/episodes.csv.gz"
    manifest_path = root / "outputs/full/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    require(manifest["status"] == "complete", "Campaign is incomplete")
    require(manifest["rows"] == 189000, "Full row budget differs")
    digest = sha256(raw)
    require(digest == manifest["csv_sha256"], "CSV hash differs from manifest")
    require(protocol["pilots"] == PILOTS and protocol["test_episodes"] == EPISODES,
            "Pilot or episode budget differs")
    require(protocol["variants"] == ["nominal", "state_dependence", "fixed_duration"],
            "Campaign variant design differs")
    require(protocol["environments"] == [[theta, kappa]
                                         for theta in (.15, .35, .65)
                                         for kappa in (.002, .02, .1)],
            "Environment grid differs")
    require(set(POLICIES).issubset(protocol["policies"]), "Required policies absent")

    records = {}
    pilot_hashes = defaultdict(set)
    posterior_hashes = defaultdict(set)
    full_rows = 0
    with gzip.open(raw, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"variant", "environment", "pilot", "episode", "policy",
                    "objective", "pilot_data_hash", "starting_posterior_hash"}
        require(required.issubset(reader.fieldnames or []), "Missing raw columns")
        for row in reader:
            full_rows += 1
            if row["variant"] != "nominal" or row["policy"] not in POLICIES:
                continue
            environment = row["environment"]
            require(environment in ENVIRONMENTS, "Unexpected nominal environment")
            try:
                pilot, episode = int(row["pilot"]), int(row["episode"])
                value = float(row["objective"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid nominal paired record") from exc
            require(0 <= pilot < PILOTS and 0 <= episode < EPISODES,
                    "Nominal pilot or episode ID exceeds budget")
            require(np.isfinite(value), "Nonfinite nominal objective")
            key = (environment, pilot, episode, row["policy"])
            require(key not in records, "Duplicate nominal paired ID")
            records[key] = value
            pilot_hashes[(environment, pilot)].add(row["pilot_data_hash"])
            posterior_hashes[(environment, pilot)].add(row["starting_posterior_hash"])
    require(full_rows == manifest["rows"], "CSV row count differs from manifest")
    expected = {(environment, pilot, episode, policy)
                for environment in ENVIRONMENTS for pilot in range(PILOTS)
                for episode in range(EPISODES) for policy in POLICIES}
    require(set(records) == expected, "Missing or extra paired nominal records")
    require(len(pilot_hashes) == 90 and all(len(values) == 1 for values in pilot_hashes.values()),
            "Pilot dataset differs within matched pair")
    require(len({next(iter(values)) for values in pilot_hashes.values()}) == 90,
            "Pilot dataset reused across nominal replicates")
    require(all(len(values) == 1 for values in posterior_hashes.values()),
            "Starting posterior differs within matched pair")
    require(all(len(next(iter(values))) == 64 for values in posterior_hashes.values()),
            "Invalid starting posterior fingerprint")
    paired = {(environment, pilot, episode):
              records[(environment, pilot, episode, "noinfo")]
              - records[(environment, pilot, episode, "myopic")]
              for environment in ENVIRONMENTS for pilot in range(PILOTS)
              for episode in range(EPISODES)}

    export_path = root / "outputs/tables/pilot_paired_moments.csv"
    exported = {}
    with export_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "nominal" or row["baseline"] not in POLICIES:
                continue
            key = (row["environment"], int(row["pilot"]), row["baseline"])
            require(key not in exported, "Duplicate pilot-moment export")
            exported[key] = float(row["mean"])
            require(np.isfinite(exported[key]), "Nonfinite pilot-moment export")
    expected_export = {(environment, pilot, policy)
                       for environment in ENVIRONMENTS for pilot in range(PILOTS)
                       for policy in POLICIES}
    require(set(exported) == expected_export, "Incomplete pilot-moment export")
    max_export_error = 0.0
    for environment in ENVIRONMENTS:
        for pilot in range(PILOTS):
            raw_mean = float(np.mean([paired[(environment, pilot, episode)]
                                      for episode in range(EPISODES)]))
            exported_mean = (exported[(environment, pilot, "myopic")]
                             - exported[(environment, pilot, "noinfo")])
            max_export_error = max(max_export_error, abs(raw_mean - exported_mean))
    require(max_export_error <= 1e-10, "Raw contrast differs from pilot-moment export")

    per_environment = {environment: summarize(paired, environment)
                       for environment in ENVIRONMENTS}
    components = np.array([per_environment[environment]["se"] ** 2 / 81.0
                           for environment in ENVIRONMENTS])
    mixture_mean = float(np.mean([per_environment[environment]["mean"]
                                  for environment in ENVIRONMENTS]))
    mixture_variance = float(components.sum())
    mixture_se = float(np.sqrt(mixture_variance))
    denominator = float(np.sum(components ** 2 / (PILOTS - 1)))
    mixture_df = float(mixture_variance ** 2 / denominator) if denominator else float("inf")
    mixture_critical = float(t.ppf(.975, mixture_df))
    mixture = {"mean": mixture_mean, "se": mixture_se, "df": mixture_df,
               "low": mixture_mean - mixture_critical * mixture_se,
               "high": mixture_mean + mixture_critical * mixture_se,
               "environments": 9, "pilots_per_environment": PILOTS,
               "paired_episodes_per_environment": PILOTS * EPISODES}
    return {
        "status": "verified_secondary_descriptive_contrast",
        "contrast": "noinfo_minus_myopic_objective",
        "variant": "nominal",
        "interpretation": "remaining-horizon continuation package, including inventory carry and internal terminal timing; not pure inventory value",
        "primary_unchanged": "active_minus_noinfo",
        "original_family_size_unchanged": 162,
        "confidence_method": "paired episode differences; ten pilot means per environment; Student t9 and fixed-uniform-grid Welch-Satterthwaite approximate two-sided 95% intervals",
        "source": {"raw_csv": str(raw.relative_to(root)), "raw_csv_sha256": digest,
                   "manifest": str(manifest_path.relative_to(root)),
                   "manifest_sha256": sha256(manifest_path),
                   "pilot_moments": str(export_path.relative_to(root)),
                   "pilot_moments_sha256": sha256(export_path),
                   "script_sha256": sha256(Path(__file__).resolve())},
        "validated": {"full_rows": full_rows, "nominal_paired_records": len(records),
                      "paired_episode_differences": len(paired), "distinct_pilots": len(pilot_hashes),
                      "pilot_export_comparisons": PILOTS * len(ENVIRONMENTS),
                      "max_pilot_export_difference": max_export_error},
        "per_environment": per_environment,
        "uniform_mixture": mixture,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/verification/reaudit/mechanism_contrasts.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = compute(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"{output}: {result['status']}; mixture={result['uniform_mixture']['mean']:.10f}")


if __name__ == "__main__":
    main()
