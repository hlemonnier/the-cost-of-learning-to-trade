"""Generate extension report figures/tables from the independently checked full run.

This does not alter the original report or any scientific outcome file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAMES = {"bayes": "Joint-belief", "weighted_q": "Weighted-Q", "no_feedback": "No future feedback",
         "frozen_model": "Frozen model weights", "known_parameter": "Known model; hidden regime"}
COLORS = {"bayes": "#076b87", "weighted_q": "#ba6633", "no_feedback": "#888888",
          "frozen_model": "#835d9d", "known_parameter": "#3c8055"}
ACTION_NAMES = ["Abstain", "Ask touch", "Ask deep", "Bid touch", "Both touch",
                "Bid touch / ask deep", "Bid deep", "Bid deep / ask touch", "Both deep",
                "Market buy", "Market sell"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_canonical(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def interval(row, simultaneous=False):
    prefix = "simultaneous_" if simultaneous else ""
    return f"{row['mean']:+.5f} [{row[prefix+'low']:+.5f}, {row[prefix+'high']:+.5f}]"


def savefig(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def checked_numerical_raw(path, numerical_path, numerical):
    raw = read_json(path)
    comparisons = numerical.get("comparisons")
    require(isinstance(comparisons, list) and len(comparisons) >= 2 and
            len(comparisons) % 2 == 0 and numerical.get("passes_predeclared_rule") is True and
            numerical.get("selected_resolution") is not None,
            "A complete accepted two-axis numerical ladder is required")
    require(raw.get("audit_status") == "raw_rows_reconciled" and
            raw.get("all_saved_joint_comparisons_audited") is True and
            raw.get("last_pair_joint_gate_passed_if_audited") is True and
            raw.get("receipt_claims_numerical_acceptance") is True and
            raw.get("raw_reconciliation_is_numerical_acceptance") is False and
            raw.get("numerical_receipt_sha256") == sha(numerical_path) and
            raw.get("verifier_sha256") == sha(HERE / "verify_numerical_raw.py") and
            raw.get("belief_geometry") == numerical.get("belief_geometry") == "endpoint_sine" and
            raw.get("receipt_selected_resolution") == numerical["selected_resolution"] and
            raw.get("receipt_last_attempted_resolution") == numerical["selected_resolution"] and
            raw.get("probe_sha256") == numerical.get("probe_sha256") and
            raw.get("probe_count") == 2581 and
            raw.get("current_source_identity") == numerical.get("source") and
            raw.get("receipt_build_sha256") == numerical.get("build", {}).get("sha256"),
            "Independent numerical raw replay is absent, partial, stale or from another geometry")
    rows = raw.get("audited_comparisons")
    require(isinstance(rows, list) and len(rows) == len(comparisons),
            "Independent numerical replay omits a saved refinement comparison")
    for index, (row, comparison) in enumerate(zip(rows, comparisons)):
        require(row.get("pair_index") == index // 2 and
                row.get("axis") == comparison.get("axis") == ("belief" if index % 2 == 0 else "quadrature") and
                row.get("coarse_artifact_sha256") == comparison.get("coarse_artifact_sha256") and
                row.get("fine_artifact_sha256") == comparison.get("fine_artifact_sha256") and
                row.get("horizon_policy_rows_reconciled") == 180 and
                row.get("passes_unchanged_joint_gate") is comparison.get("passes_predeclared_rule"),
                f"Independent numerical comparison {index} differs from saved raw evidence")
    return raw


def checked_appendix():
    receipt_path = HERE / "report/Mathematical_Appendix_verification.json"
    receipt = read_json(receipt_path)
    source, pdf = HERE / "report/Mathematical_Appendix.tex", HERE / "report/Mathematical_Appendix.pdf"
    expected_checks = {"ten_pdf_pages", "ten_sections", "equations_1_to_33",
                       "ten_rendered_pages", "nonempty_page_text", "no_unconverted_markdown",
                       "conservative_allowance", "geometry_agnostic_cell_width",
                       "references_visible", "pilots_visible", "revelation_proof_visible",
                       "three_primary_links"}
    require(receipt.get("source", {}).get("sha256") == sha(source) and
            receipt.get("source", {}).get("bytes") == source.stat().st_size and
            receipt.get("pdf", {}).get("sha256") == sha(pdf) and
            receipt.get("pdf", {}).get("bytes") == pdf.stat().st_size and
            receipt.get("pdf", {}).get("pages") == 10 and
            receipt.get("theory_source", {}).get("sha256") == sha(HERE / "THEORY.md") and
            expected_checks <= receipt.get("content_checks", {}).keys() and
            all(receipt["content_checks"][key] is True for key in expected_checks) and
            receipt.get("visual_check", {}).get("all_ten_pages_reviewed") is True and
            receipt.get("primary_agent_review", {}).get("all_ten_pages_visually_inspected") is True and
            receipt.get("primary_agent_review", {}).get("pdf_sha256") == sha(pdf),
            "Standalone mathematical appendix or primary-agent review is missing or stale")
    return receipt


def checked_independent_explanation(path, directory, candidates_path, numerical_path,
                                    analysis_manifest, candidate_manifest, explanation):
    receipt = read_json(path)
    numerical = read_json(numerical_path)
    analysis = receipt.get("analysis", {})
    candidate = receipt.get("candidate", {})
    source = receipt.get("source", {})
    direct = analysis.get("direct", {})
    fixed = analysis.get("fixed_state", {})
    envelopes = analysis.get("error_envelopes", {})
    binding = receipt.get("numerical_acceptance_binding", {})
    require(receipt.get("schema_version") == 1 and
            receipt.get("validation_completion") is True and
            receipt.get("scope") == "full_endpoint_explanation_independent_raw_recomputation" and
            receipt.get("certified_bound") is False and
            source.get("verifier_sha256") == sha(HERE / "verify_explanation.py") and
            source.get("candidate_reader_sha256") == sha(HERE / "verify_extension.py") and
            source.get("physical_reader_sha256") == sha(HERE / "verify_numerical_raw.py") and
            source.get("solver_source") == numerical.get("source") and
            candidate.get("manifest_sha256") == sha(candidates_path / "manifest.json") and
            candidate.get("artifact_sha256") == candidate_manifest["artifact_sha256"] and
            candidate.get("states") == 137430 and
            candidate.get("public_decisions_replayed") == 60000 and
            candidate.get("frozen_numerical_probe_sha256") == numerical.get("probe_sha256") and
            candidate.get("frozen_numerical_probe_count") == 2581,
            "Independent explanation candidate/source replay is missing, partial or stale")
    require(analysis.get("manifest_sha256") == sha(directory / "manifest.json") and
            analysis.get("artifact_sha256") == analysis_manifest.get("artifact_sha256") and
            analysis.get("file_records") == analysis_manifest.get("files") and
            analysis.get("traversal_rows_recomputed") == 137430 and
            direct.get("direct_cases") == explanation.get("direct_candidates") and
            direct.get("direct_family_cases") == 3 * explanation.get("direct_candidates", -1) and
            direct.get("csv_action_rows_checked") == 33 * explanation.get("direct_candidates", -1) and
            ((direct.get("direct_cases") == 0 and
              direct.get("direct_legal_scores_recomputed") == 0) or
             (direct.get("direct_cases", 0) > 0 and
              direct.get("direct_legal_scores_recomputed", 0) > 0)) and
            direct.get("selected_qualifiers") == explanation.get("selected_resolution_qualifiers") and
            direct.get("refinement_stable_witnesses") == explanation.get("refinement_stable_witnesses") and
            direct.get("winner_candidate_id") ==
            (None if explanation.get("winner") is None else explanation["winner"].get("candidate_id")) and
            fixed.get("states_per_family") == 151 and
            fixed.get("family_state_checks") == 453 and fixed.get("passed") is True and
            envelopes.get("families") == 3 and
            envelopes.get("horizon_rows_recomputed") == 93 and
            isinstance(envelopes.get("quadrature_rules_integrated"), int) and
            envelopes["quadrature_rules_integrated"] >= 2 and
            envelopes.get("passed") is True,
            "Independent explanation direct/fixed/envelope replay is incomplete")
    if explanation.get("winner") is not None:
        require(analysis.get("illustrative_posteriors", {}).get("branches_checked", 0) > 0,
                "Independent explanation did not replay illustrative posterior branches")
    family_specs = analysis_manifest.get("family_specifications")
    families = receipt.get("families", {})
    require(isinstance(family_specs, list) and len(family_specs) == 3 and
            set(families) == {"selected", "comparison_1", "comparison_2"},
            "Independent explanation omits a refinement family")
    for row in family_specs:
        label = row["label"]
        family = families[label]
        specification = {key: value for key, value in row.items()
                         if key not in ("label", "artifact_sha256")}
        require(family.get("artifact_sha256") == row["artifact_sha256"] and
                family.get("specification") == specification and
                len(family.get("array_records", [])) == 11 and
                isinstance(family.get("manifest_sha256"), str) and
                len(family["manifest_sha256"]) == 64 and
                isinstance(family.get("grid_axes_sha256"), str) and
                len(family["grid_axes_sha256"]) == 64,
                f"Independent explanation family identity is incomplete: {label}")
    comparisons = numerical.get("comparisons", [])
    require(len(comparisons) >= 2 and
            [item.get("axis") for item in comparisons[-2:]] == ["belief", "quadrature"],
            "Explanatory comparison axes are absent from the accepted numerical pair")
    for label, comparison in zip(("comparison_1", "comparison_2"), comparisons[-2:]):
        require(families[label]["artifact_sha256"] ==
                comparison["coarse_artifact_sha256"] and
                families[label]["specification"] == comparison["coarse_specification"],
                f"Independent explanation used a wrong-axis comparison: {label}")
    require(binding.get("receipt_sha256") == sha(numerical_path) and
            binding.get("selected_artifact_sha256") == numerical.get("selected_artifact_sha256") and
            binding.get("passes_predeclared_rule") is True and
            binding.get("binding_status") == "accepted_family_authenticated" and
            families["selected"]["artifact_sha256"] == numerical["selected_artifact_sha256"],
            "Independent explanation does not bind accepted numerical family")
    return receipt


def checked_explanation(directory, numerical_path, validation, candidates_path):
    manifest = read_json(directory / "manifest.json")
    require(manifest["kind"] == "explanatory_analysis" and
            manifest["fixed_state_checks_passed"] is True and
            manifest.get("artifact_sha256") == sha_canonical({
                key: value for key, value in manifest.items() if key != "artifact_sha256"}),
            "Complete explanatory analysis is required, including negative outcomes")
    record = read_json(directory / "explanation.json")
    for item in manifest["files"]:
        path = directory / item["file"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                f"Explanatory input changed: {item['file']}")
    for name in ("explain.py", "explain_failure_modes.md", "THEORY.md", "solver.py"):
        require(manifest["source"]["files"][name] == sha(HERE / name), "Stale explanatory source")
    numerical = read_json(numerical_path)
    binding = record["numerical_selection_evidence"]
    require(binding["acceptance_bound_to_selected_family"] is True and
            all(binding["binding_checks"].values()) and binding["sha256"] == sha(numerical_path) and
            binding["selected_family_artifact_sha256"] == numerical["selected_artifact_sha256"],
            "Explanation must use the same accepted family as the economic study")
    candidate_manifest = read_json(candidates_path / "manifest.json")
    candidate_check = validation.get("explanatory_candidates", {})
    require(candidate_manifest.get("kind") == "explanatory_candidates" and
            candidate_manifest.get("config_sha256") == sha(HERE / "config.json") and
            candidate_manifest.get("artifact_sha256") == sha_canonical({
                key: value for key, value in candidate_manifest.items()
                if key != "artifact_sha256"}) and
            candidate_manifest.get("reached_states") == 60000 and
            candidate_manifest.get("total_states") == 137430 and
            candidate_manifest.get("source") == manifest.get("source") and
            candidate_manifest.get("build") == manifest.get("build") and
            candidate_check.get("checked") is True and
            candidate_check.get("public_decisions_replayed") == 60000 and
            candidate_check.get("candidate_states_checked") == 137430 and
            candidate_check.get("artifact_sha256") == candidate_manifest.get("artifact_sha256") ==
            record["candidate_manifest_sha256"] == manifest["candidate_artifact_sha256"],
            "Current endpoint candidate provenance lacks complete independent replay")
    for item in candidate_manifest["files"]:
        path = candidates_path / item["file"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha(path) == item["sha256"], f"Candidate input changed: {item['file']}")
    require(record["comparison_families_supplied"] == 2, "Both independent refinement axes must check the explanation")
    return record, read_json(directory / "error_envelopes.json"), candidate_manifest


def explanation_page(record, out):
    total = sum(row["screened"] for row in record["search"])
    stable = record["refinement_stable_witnesses"]
    lines = [f"The fixed search screened {total:,} states and directly checked "
             f"{record['direct_candidates']} distinct disagreements. It found {stable} cases "
             "meeting the complete action-margin and cross-refinement witness criterion."]
    case = record["winner"]
    if case is None:
        lines.append("No screened disagreement was available for direct analysis. This is a negative result "
                     "for this fixed candidate set; it does not rule out differences elsewhere.")
    else:
        selected = case["families"]["selected"]
        ba, wq = selected["actions"]["direct_bayes"], selected["actions"]["weighted_q"]
        frozen = selected["actions"]["same_continuation_frozen_model"]
        coordinates = ", ".join(f"{p:.6f}" for p in case["joint"])
        reached = case["source"] == "uniform_exploration_reached"
        status = ("A refinement-stable witness" if case["refinement_stable_witness"] else
                  "The strongest retained nonqualifying case")
        lines.append(f"{status} is candidate {case['candidate_id']}, with "
                     f"$n={case['remaining']}$, $q={case['inventory']}$, $x={case['signal']}$ and "
                     f"$p=({coordinates})$. It is " +
                     ("reached by the separate public exploration process, not selected from policy evaluation."
                      if reached else "an arbitrary-prior probe, not a claimed reachable trading history."))
        lines.append(f"The direct joint-belief backup selects {ACTION_NAMES[ba].lower()} (ID {ba}); "
                     f"weighted-Q selects {ACTION_NAMES[wq].lower()} (ID {wq}); keeping the same "
                     f"joint-belief continuation but freezing model weights selects {ACTION_NAMES[frozen].lower()} "
                     f"(ID {frozen}).")
        if not case["refinement_stable_witness"]:
            lines.append("This case fails at least one declared requirement. It must not be treated as a "
                         "demonstration of a robust model-update-sensitive decision. All failed checks are retained.")
        legal = np.flatnonzero(np.asarray(case["legal_actions"], dtype=bool))
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), constrained_layout=True)
        score_specs = [("direct_bayes", "Joint backup", "#076b87"),
                       ("same_continuation_frozen_model", "Same continuation; frozen weights", "#835d9d"),
                       ("weighted_q", "Weighted-Q", "#ba6633")]
        index = np.arange(len(legal))
        for offset, (name, label, color) in enumerate(score_specs):
            scores = np.asarray(selected["scores"][name], dtype=float)
            axes[0].bar(index+(offset-1)*.24, scores[legal]-scores[wq], width=.23, label=label, color=color)
        axes[0].axhline(0, color="#333333", linewidth=.5)
        axes[0].set(xticks=index, xticklabels=legal, xlabel="Safe action ID", ylabel="Score minus that method's\nscore of weighted-Q's chosen action")
        axes[0].legend(fontsize=5.8)
        mi = np.asarray(case["information"]["model_information"], dtype=float)
        axes[1].bar(index, mi[legal], color=["#076b87" if a == ba else "#ba6633" if a == wq else "#cccccc" for a in legal])
        axes[1].set(xticks=index, xticklabels=legal, xlabel="Safe action ID", ylabel="One-observation model information\n(nats; conditional on this state)")
        savefig(fig, out, "action_explanation")
        lines += [r"\begin{center}\includegraphics[width=\linewidth]{generated/action_explanation.pdf}\end{center}",
                  r"{\small\textbf{Decision diagnostic.} Each score is centered on that same method's score of the weighted-Q action; absolute relaxed and feasible values are not subtracted. Positive information does not itself establish economic benefit.}",
                  r"\begin{center}\small\begin{tabular}{lrr}\toprule",
                  r"Quantity & Joint-belief choice & Weighted-Q choice\\\midrule"]
        for label, values in [("Immediate expected reward", case["immediate_expected_reward"]),
                              ("Model information (nats)", case["information"]["model_information"]),
                              ("Regime information conditional on model", case["information"]["conditional_regime_information"])]:
            lines.append(f"{label} & {values[ba]:+.7f} & {values[wq]:+.7f}"+r"\\")
        lines.append(r"\bottomrule\end{tabular}\end{center}")
        rows = []
        for family, entry in case["families"].items():
            for action in legal:
                rows.append({"family": family, "action": int(action), "action_name": ACTION_NAMES[action],
                             "immediate_expected_reward": case["immediate_expected_reward"][action],
                             "model_information": mi[action],
                             **{name: values[action] for name, values in entry["scores"].items()}})
        pd.DataFrame(rows).to_csv(out / "action_explanation.csv", index=False, float_format="%.17g")
    (out / "action_explanation.tex").write_text("\n".join(lines)+"\n")


def generate(study, numerical_path, numerical_raw_path, validation_path, explanation_path,
             explanation_verification_path, candidates_path, cold_path, out):
    summary, manifest = read_json(study / "summary.json"), read_json(study / "manifest.json")
    numerical, validation = read_json(numerical_path), read_json(validation_path)
    require(manifest["status"] == "complete" and manifest["profile"] == "full" and
            summary["rows"] == 900000 and summary["paired_market_trajectories"] == 60000,
            "Report requires the complete prespecified full campaign")
    require(validation.get("passed") is True and validation.get("manifest_sha256") == sha(study / "manifest.json"),
            "Independent verification does not authenticate this study")
    require(validation.get("profile") == "full" and validation.get("rows") == 900000 and
            validation.get("policy_decisions_recomputed") == 27000000 and
            validation.get("trace_rows") == 108000 and validation.get("pilot_records") == 60 and
            validation.get("verifier_sha256") == sha(HERE / "verify_extension.py") and
            validation.get("numerical_acceptance", {}).get("passed") is True,
            "Report requires current-source complete independent decision and numerical replay")
    require(manifest["numerical_acceptance_sha256"] == sha(numerical_path) and
            numerical["passes_predeclared_rule"] is True, "Numerical acceptance missing or stale")
    checked_numerical_raw(numerical_raw_path, numerical_path, numerical)
    checked_appendix()
    for name, digest in manifest["files"].items():
        require(sha(study / name) == digest, f"Scientific input changed: {name}")
    explanation, envelopes, candidate_manifest = checked_explanation(
        explanation_path, numerical_path, validation, candidates_path)
    checked_independent_explanation(
        explanation_verification_path, explanation_path, candidates_path, numerical_path,
        read_json(explanation_path / "manifest.json"), candidate_manifest, explanation)
    cold = read_json(cold_path)
    require(cold.get("passed") is True and cold.get("arrays_identical") is True and
            cold.get("smoke_economic_files_identical") is True and cold.get("array_count") == 11 and
            cold.get("reference_table_artifact_sha256") == manifest["table_artifact_sha256"] and
            cold.get("verifier_sha256") == sha(HERE / "verify_cold_reproduction.py") and
            cold.get("independent_verifier_sha256") == sha(HERE / "verify_extension.py"),
            "Report requires the selected resolution's current-source cold reproduction")
    require(not out.exists(), "Use a new report-output directory to preserve previous evidence")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "grid.linewidth": .5,
                         "pdf.fonttype": 42, "axes.axisbelow": True})
    means = pd.DataFrame(summary["policy_means"])
    contrasts = pd.DataFrame(summary["contrasts"])
    plans = pd.read_csv(study / "planning_values.csv", float_precision="round_trip")
    diagnostics = pd.read_csv(study / "diagnostics.csv", float_precision="round_trip")
    timings = pd.read_csv(study / "timings.csv", float_precision="round_trip")
    budgets = [0, 1, 5]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    for j, policy in enumerate(NAMES):
        d = means[(means.policy == policy) & (means.metric == "objective")].sort_values("budget")
        axes[0].errorbar(np.arange(3)+(j-2)*.055, d["mean"],
                        yerr=[d["mean"]-d.low, d.high-d["mean"]],
                        color=COLORS[policy], marker="o", markersize=3.5, capsize=2,
                        linestyle="--" if policy == "known_parameter" else "-", label=NAMES[policy])
    axes[0].set(xticks=np.arange(3), xticklabels=budgets, xlabel="Pilot episodes", ylabel="Mean net objective / episode")
    axes[0].legend(fontsize=6.3, loc="best")
    d = contrasts[(contrasts.policy == "bayes") & (contrasts.baseline == "weighted_q") &
                  (contrasts.metric == "objective")].sort_values("budget")
    axes[1].errorbar(np.arange(3), d["mean"], yerr=[d["mean"]-d.low, d.high-d["mean"]],
                    color=COLORS["bayes"], marker="o", capsize=4)
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].axhline(.002, color="#ba6633", linestyle=":", label="Prespecified economic threshold")
    axes[1].set(xticks=np.arange(3), xticklabels=budgets, xlabel="Pilot episodes",
                ylabel="Joint-belief minus weighted-Q\npaired objective difference")
    axes[1].legend(fontsize=6, loc="best")
    savefig(fig, out, "economic_performance")

    info = diagnostics.groupby(["budget", "policy"], sort=True).mean(numeric_only=True).reset_index()
    info.to_csv(out / "information.csv", index=False, float_format="%.17g")
    compute = timings.groupby(["budget", "policy"], sort=True)[["choice_seconds", "decisions"]].sum().reset_index()
    compute["microseconds_per_decision"] = compute.choice_seconds / compute.decisions * 1e6
    compute.to_csv(out / "computation.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    pilot_entropy = plans.groupby("budget").initial_model_entropy.mean().reindex(budgets)
    axes[0].plot(range(3), pilot_entropy, color="black", linestyle=":", label="After pilot; before trading")
    for policy in NAMES:
        d = info[info.policy == policy].sort_values("budget")
        if policy != "known_parameter":
            axes[0].plot(range(3), d.model_entropy_final, marker="o", markersize=3,
                         color=COLORS[policy], label=NAMES[policy])
        c = compute[compute.policy == policy].sort_values("budget")
        axes[1].plot(range(3), c.microseconds_per_decision, marker="o", markersize=3, color=COLORS[policy])
    axes[0].set(xticks=range(3), xticklabels=budgets, xlabel="Pilot episodes", ylabel="Model-posterior entropy (nats)")
    axes[0].legend(fontsize=6, loc="best")
    axes[1].set(xticks=range(3), xticklabels=budgets, xlabel="Pilot episodes", ylabel="Choice time\n(microseconds / decision)")
    savefig(fig, out, "information_and_compute")

    numerical_rows = []
    for comparison in numerical["comparisons"]:
        spec = comparison["fine_specification"]
        numerical_rows.append({"axis": comparison["axis"], "weight_points": spec["weight_points"],
                               "belief_points": spec["belief_points"], "quadrature_points": spec["quadrature_points"],
                               "passes": comparison["passes_predeclared_rule"], **comparison["summary"]})
    ref = pd.DataFrame(numerical_rows)
    ref.to_csv(out / "refinement.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    for ax, metric, limit, label in zip(axes, ["max_finite_q_change", "max_initial_value_change"],
                                      [.005, .001], ["Max finite-Q change", "Max initial-value change"]):
        for axis, color in [("belief", "#076b87"), ("quadrature", "#ba6633")]:
            r = ref[ref.axis == axis]
            ax.plot(r.weight_points, r[metric], "o-", color=color, label=axis)
        ax.axhline(limit, color="black", linestyle=":", label="Prespecified tolerance")
        ax.set(xlabel="Fine grid: model-weight points", ylabel=label, yscale="log")
        ax.legend(fontsize=7)
    savefig(fig, out, "numerical_refinement")

    plan_fields = ["revelation_0", "revelation_1_direct", "reveal2", "reveal3", "bayes"]
    revelation = plans.groupby("budget")[plan_fields].mean().reindex(budgets)
    revelation.to_csv(out / "revelation.csv", float_format="%.17g")
    contrasts.to_csv(out / "all_contrasts.csv", index=False, float_format="%.17g")
    means.to_csv(out / "all_policy_means.csv", index=False, float_format="%.17g")
    gap_rows = []
    for budget in budgets:
        for policy in ("weighted_q", "bayes"):
            row = means[(means.budget == budget) & (means.policy == policy) & (means.metric == "objective")].iloc[0]
            for field in plan_fields[:-1]:
                gap_rows.append({"budget": budget, "policy": policy, "planning_quantity": field,
                                 "approximate_planning_value": revelation.loc[budget, field],
                                 "actual_policy_mean": row["mean"], "policy_interval_low": row.low,
                                 "policy_interval_high": row.high,
                                 "uncertified_planning_minus_return": revelation.loc[budget, field]-row["mean"]})
    pd.DataFrame(gap_rows).to_csv(out / "uncertified_planning_gaps.csv", index=False, float_format="%.17g")

    primary = summary["primary"]
    conclusion = ("The prespecified economically meaningful improvement criterion is met."
                  if primary["low"] > .002 else
                  "The prespecified economically meaningful improvement criterion is not met.")
    macros = {"PrimaryMean": f"{primary['mean']:+.6f}", "PrimaryLow": f"{primary['low']:+.6f}",
              "PrimaryHigh": f"{primary['high']:+.6f}", "PrimaryConclusion": conclusion,
              "EvaluationMinutes": f"{manifest['seconds']/60:.1f}",
              "GridWeights": str(manifest["table_specification"]["weight_points"]),
              "GridBeliefs": str(manifest["table_specification"]["belief_points"]),
              "GHNodes": str(manifest["table_specification"]["quadrature_points"]),
              "MaxHierarchyViolation": f"{validation['secondary_tables']['max_numerical_hierarchy_violation']:.6g}"}
    (out / "numbers.tex").write_text("\n".join("\\newcommand{\\"+key+"}{"+value+"}" for key, value in macros.items())+"\n")
    table = [r"\begin{tabular}{lrrr}\toprule", r"Approximate planning quantity & Pilot 0 & Pilot 1 & Pilot 5\\\midrule"]
    labels = [r"$U^0$: model known at entry", r"$U^1$: revealed after 1 decision", r"$U^2$: after 2 decisions",
              r"$U^3$: after 3 decisions", r"$V^{\rm BA}$: no revelation"]
    for key, label in zip(plan_fields, labels):
        table.append(label+" & "+" & ".join(f"{revelation.loc[b,key]:.6f}" for b in budgets)+r"\\")
    table.append(r"\bottomrule\end{tabular}")
    (out / "revelation.tex").write_text("\n".join(table)+"\n")
    table = [r"\begin{tabular}{llrr}\toprule", r"Pilot & Comparison & Objective difference & Net-PnL difference\\\midrule"]
    for budget in budgets:
        for policy, baseline in manifest["protocol"]["contrasts"]:
            select = contrasts[(contrasts.budget == budget) & (contrasts.policy == policy) & (contrasts.baseline == baseline)]
            label = ("Known model -- joint" if policy == "known_parameter" else
                     {"weighted_q": "Joint -- weighted-Q", "no_feedback": "Joint -- no feedback", "frozen_model": "Joint -- frozen weights"}[baseline])
            values = [interval(select[select.metric == metric].iloc[0], True) for metric in ["objective", "pnl"]]
            table.append(f"{budget} & {label} & "+" & ".join(values)+r"\\")
    table.append(r"\bottomrule\end{tabular}")
    (out / "contrasts.tex").write_text("\n".join(table)+"\n")
    interpretation = ("The implemented joint-belief policy clears the registered empirical threshold in this population. "
                      "This supports its practical use here, without certifying the true optimal policy."
                      if primary["low"] > .002 else
                      "The implemented joint-belief policy underperforms weighted-Q in the primary condition. "
                      "The extension therefore does not support paying for its extra sophistication here."
                      if primary["high"] < 0 else
                      "The experiment does not establish the registered economically meaningful improvement. "
                      "The observed estimate and interval are retained even if other comparisons look more favorable.")
    (out / "result_interpretation.tex").write_text(interpretation+"\n")
    wq0 = means[(means.budget == 0) & (means.policy == "weighted_q") & (means.metric == "objective")].iloc[0]
    (out / "planning_interpretation.tex").write_text(
        f"At cold start, the numerical $U^1$ minus achieved weighted-Q mean is "
        f"{revelation.loc[0,'revelation_1_direct']-wq0['mean']:+.6f}; delaying revelation to $U^3$ "
        f"reduces the corresponding diagnostic to {revelation.loc[0,'reveal3']-wq0['mean']:+.6f}. "
        "These differences combine planning approximation and Monte Carlo error. They are not certified "
        "upper bounds on the heuristic's suboptimality. The complete budget-by-policy diagnostics "
        "are exported in the accompanying tables.\n")
    explanation_page(explanation, out)
    final = numerical["comparisons"][-2:]
    numerical_text = ["The final two independently varied axes both satisfy the frozen rule. "
                      "Earlier failed meshes and the resource-driven V1 interruption remain in the evidence."]
    for comparison in final:
        s = comparison["summary"]
        numerical_text.append(f"For the {comparison['axis']} comparison, maximum initial-value, probe-value and "
                              f"finite-Q changes are {s['max_initial_value_change']:.6g}, "
                              f"{s['max_probe_value_change']:.6g} and {s['max_finite_q_change']:.6g}; "
                              f"the robust-action disagreement fraction is {s['max_robust_disagreement_fraction']:.6g}.")
    allowance = envelopes["levels"][0]["by_horizon"][-1]
    numerical_text.append(f"The selected-family conservative one-sided allowance at $T=30$ is "
                          f"{allowance['one_sided_allowance']:.5f}, and the two-sided off-grid allowance is "
                          f"{allowance['two_sided_off_grid_allowance']:.5f} objective units. They are much "
                          "larger than .002; their floating evaluation is not an interval-arithmetic certificate.")
    (out / "numerical_validation.tex").write_text("\n\n".join(numerical_text)+"\n")
    (out / "final_interpretation.tex").write_text(
        interpretation+" The numerical relaxation ladder explains the shortcut's information optimism, "
        "but the present conservative error envelope does not prove a small global optimality gap. "
        f"The fixed action search supplies {explanation['refinement_stable_witnesses']} qualifying cases; "
        "their frequency in a designed search is not an estimate of their prevalence under either policy.\n")
    ledger_error = max(validation["maximum_ledger_errors"].values())
    (out / "final_validation.tex").write_text(
        "The independent verifier reconstructs all 900,000 outcome records, all 27 million policy decisions, "
        "108,000 posterior-trace rows and 60 pilot sets from the saved primitive tapes and permitted public feedback. "
        "It imports neither the producer's runner nor its statistical routines. All 24 supplementary contrasts, "
        "30 policy means and 180 initial planning-value rows reconcile. "
        f"The largest absolute ledger discrepancy is {ledger_error:.6g}; the largest public-trace discrepancy is "
        f"{validation['maximum_trace_error']:.6g}.\n\n"
        "A fresh process builds the selected family in a previously nonexistent cache. All eleven array-content "
        "records and actual file hashes match the reference build. A separate smoke process produces 720 policy "
        "records on 48 paired tapes; its pilots, tapes, actions, outcomes, traces, planning values and summaries "
        "match exactly, with all 21,600 decisions independently replayed. Measured timing metadata retains its "
        "new identity. These are repeated checks of the same seeds, not independent economic observations.\n\n"
        f"The final economic execution took {manifest['seconds']/60:.1f} minutes on the recorded stack; "
        f"the selected-resolution cold reproduction took {cold['seconds']/60:.1f} minutes. "
        "Both durations are workload observations. The actual review archive has its own packaging and "
        "extracted-execution receipt; those checks are separate from numerical and statistical validation.\n")
    payload = {"primary": primary, "conclusion": conclusion, "macro_values": macros,
               "study_manifest_sha256": sha(study / "manifest.json"),
               "numerical_sha256": sha(numerical_path),
               "independent_numerical_raw_sha256": sha(numerical_raw_path),
               "independent_verification_sha256": sha(validation_path),
               "explanation_manifest_sha256": sha(explanation_path / "manifest.json"),
               "explanation_sha256": sha(explanation_path / "explanation.json"),
               "independent_explanation_sha256": sha(explanation_verification_path),
               "candidate_manifest_sha256": sha(candidates_path / "manifest.json"),
               "candidate_artifact_sha256": candidate_manifest["artifact_sha256"],
               "cold_verification_sha256": sha(cold_path),
               "mathematical_appendix_verification_sha256": sha(
                   HERE / "report/Mathematical_Appendix_verification.json"),
               "builder_sha256": sha(Path(__file__)),
               "scope": "Tables generated from authenticated outputs; plotting is not independent scientific validation."}
    (out / "report_inputs.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--numerical-checks", type=Path, required=True)
    parser.add_argument("--numerical-raw-verification", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--explanation", type=Path, required=True)
    parser.add_argument("--explanation-verification", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cold-validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "report/generated")
    args = parser.parse_args()
    print(json.dumps(generate(args.study, args.numerical_checks, args.numerical_raw_verification,
                              args.validation, args.explanation, args.explanation_verification,
                              args.candidates,
                              args.cold_validation, args.out), indent=2))


if __name__ == "__main__":
    main()
