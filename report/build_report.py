"""Build report inputs from the frozen campaign and separate matched study.

Artifact acceptance follows DESIGN.md. This does not train/select a new policy.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypandoc
from delivery_receipts import invalidate, record_build

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trade_learning.run import source_hash, pilot_hash
from trade_learning.statistics import stratified_estimate, validate_campaign

GEN = ROOT / "report/generated"
FIG = ROOT / "outputs/figures"
TABLE = ROOT / "outputs/tables"
POLICIES = ["abstain", "taker", "independent", "myopic", "noinfo", "active", "full_information"]
LABEL = dict(abstain="Abstain", taker="Taker", independent="Independence",
             myopic="Myopic", noinfo="Matched noinfo", active="Active",
             full_information="Privileged reference", full="Privileged reference")
VARIANTS = ["nominal", "state_dependence", "fixed_duration"]
VLABEL = dict(nominal="Nominal", state_dependence="Signal dependence", fixed_duration="Fixed durations")


def read(relative):
    return json.loads((ROOT / relative).read_text())



def mechanism_study():
    base = ROOT / "outputs/verification/reaudit/mechanism_study"
    manifest = json.loads((base / "manifest.json").read_text())
    receipt = json.loads((base / "independent_verification.json").read_text())
    summary = json.loads((base / "summary.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest["status"] == "complete" and receipt["passed"]
    assert receipt["manifest_sha256"] == sha(base / "manifest.json")
    assert receipt["verifier_sha256"] == sha(ROOT / "verification/verify_mechanism_study.py")
    assert receipt["repeat"]["passed"] and receipt["repeat"]["exact_five_file_content"]
    assert receipt["fault_probes_rejected"] == 10
    for name, digest in manifest["files"].items():
        assert sha(base / name) == digest == receipt["file_sha256"][name]
    for name, digest in manifest["input_sha256"].items():
        assert sha(ROOT / name) == digest
    assert len(summary["contrasts"]) == 4 and summary["rows"] == 140000
    assert len(receipt["recomputed"]["contrasts"]) == 4
    for left, right in zip(summary["contrasts"], receipt["recomputed"]["contrasts"]):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], float):
                assert abs(left[key]-right[key]) < 5e-10
            else:
                assert left[key] == right[key]
    return summary, {"manifest_sha256": sha(base / "manifest.json"),
                     "independent_verification_sha256": sha(base / "independent_verification.json"),
                     "summary_sha256": sha(base / "summary.json")}


def put(name, content):
    (GEN / f"{name}.tex").write_text(content.strip()+"\n")


def interval(row, decimals=3):
    return f"[{row['low']:.{decimals}f}, {row['high']:.{decimals}f}]"


def artifact_audit(frame, summary, manifest):
    """Reconcile saved full-run artifacts against predeclared acceptance criteria."""
    validate_campaign(frame, manifest["protocol"])
    assert manifest["status"] == "complete"
    assert len(frame) == manifest["rows"] == 189000 and len(manifest["streams"]) == 90
    assert manifest["protocol"]["master_seed"] == 260924138
    assert manifest["code_hash"] == source_hash()
    assert hashlib.sha256((ROOT / "outputs/full/episodes.csv.gz").read_bytes()).hexdigest() == manifest["csv_sha256"]
    assert hashlib.sha256((ROOT / "BENCHMARK.md").read_bytes()).hexdigest() == manifest["specification_hash"]
    keys = ["environment", "pilot", "variant", "policy", "episode"]
    assert not frame.duplicated(keys).any()
    counts = frame.groupby(keys[:-1]).size()
    assert len(counts) == 1890 and (counts == 100).all() and frame.environment.nunique() == 9
    assert set(frame.policy) == set(POLICIES) and set(frame.variant) == set(VARIANTS)
    assert np.isfinite(frame[["pnl", "objective", "reconciliation_error"]]).all().all()
    assert (frame.max_abs_inventory <= 5).all()
    assert (frame.terminal_inventory_after_liquidation == 0).all()
    assert (frame[[f"action_{a}" for a in range(11)]].sum(axis=1) == 300).all()
    assert frame.reconciliation_error.abs().max() < 1e-8
    assert np.max(np.abs(frame.pnl-frame.inventory_penalty-frame.objective)) < 1e-10
    ledger = (frame.spread_capture-frame.passive_fees-frame.market_fees-frame.market_spread_cost
              -frame.liquidation_cost+frame.directional_exposure+frame.execution_selection+frame.market_innovation)
    assert np.max(np.abs(ledger-frame.pnl)) < 1e-8
    assert (frame.loc[frame.policy == "abstain", ["pnl", "objective"]] == 0).all().all()
    assert (frame.groupby(["environment", "pilot"]).pilot_data_hash.nunique() == 1).all()
    learning = frame[frame.policy.isin(["active", "noinfo", "myopic"])]
    assert (learning.groupby(["environment", "pilot"]).starting_posterior_hash.nunique() == 1).all()
    fits = read("outputs/full/pilot_fits.json")
    assert len(fits) == 90
    identities = []
    for index, stream in enumerate(manifest["streams"]):
        env_index, pilot_index = divmod(index, 10)
        with np.load(ROOT / f"outputs/full/pilots/{env_index:02d}_{pilot_index:02d}.npz") as archive:
            arrays = {name: archive[name] for name in archive.files}
            assert arrays["actions"].shape == (100, 300)
            assert pilot_hash(arrays) == stream["pilot_data_hash"]
            assert not any(k in arrays for k in ["h", "z", "theta", "kappa", "u_bid", "u_ask"])
        assert fits[index]["pilot_data_hash"] == stream["pilot_data_hash"]
        identities.extend([stream["pilot_seed"], stream["behavior_seed"], *stream["final"].values()])
    assert len(identities) == len(set(identities)) == 450
    for file, key in [("outputs/verification/seed_separation.json", "passed"),
                      ("outputs/reference/practical_refinement.json", "all_final_axis_checks_passed"),
                      ("outputs/reference/verification.json", "all_checks_passed"),
                      ("outputs/reference/rollout/verification.json", "passed"),
                      ("outputs/verification/pipeline/verification.json", "passed")]:
        assert read(file)[key], file
    assert read("outputs/environment/verification.json")["status"] == "PASS"
    assert read("outputs/theory/verification.json")["status"] == "passed"
    pivot = frame[frame.variant == "nominal"].pivot(
        index=["environment", "pilot", "episode"], columns="policy", values="objective")
    delta = (pivot.active-pivot.noinfo).rename("difference").reset_index()
    primary = stratified_estimate(delta, "difference", protocol=manifest["protocol"])
    for key in ["mean", "se", "low", "high", "one_sided_lower"]:
        assert abs(primary[key]-summary["primary_comparison"][key]) < 1e-10
    assert len(summary["per_environment"]) == 189 and len(summary["paired_differences"]) == 162
    record = dict(passed=True, episodes=len(frame), pilot_datasets=90, pilot_episodes=9000,
                  final_periods=len(frame)*300, total_stream_ids=450, final_stream_ids=270,
                  pilot_and_behavior_stream_ids=180, policy_variant_environment_cells=189,
                  max_ledger_error=float(frame.reconciliation_error.abs().max()),
                  max_attribution_error=float(np.max(np.abs(ledger-frame.pnl))),
                  production_code_hash=manifest["code_hash"], final_csv_sha256=manifest["csv_sha256"],
                  primary_comparison_recomputed=primary,
                  scope="Complete frozen campaign and existing verification reconciled; PDF visual review is separate.")
    (ROOT / "outputs/verification/final_artifact_audit.json").write_text(json.dumps(record, indent=2)+"\n")
    return record


def export_tables(frame, summary, fits, refinement):
    rows = []
    columns = [c for c in frame.select_dtypes("number").columns if c not in ["episode", "pilot", "theta", "kappa"]]
    for record in summary["per_environment"]:
        variant, env, policy = [record[k] for k in ["variant", "environment", "policy"]]
        group = frame[(frame.variant == variant) & (frame.environment == env) & (frame.policy == policy)]
        row = dict(variant=variant, environment=env, policy=policy)
        row.update({f"mean_{c}": float(group[c].mean()) for c in columns})
        for key in ["objective", "pnl"]:
            row.update({f"{key}_{k}": v for k, v in record[key].items()})
        row["pooled_post_switch_error_10"] = record["metrics"]["post_switch_error_10"]
        row["pooled_score_residual_per_side"] = record["metrics"]["score_residual_per_side"]
        for x in [-1, 0, 1]:
            denominator = group[f"submitted_sides_x{x}"].sum()
            row[f"pooled_public_residual_x{x}"] = (group[f"score_sum_x{x}"].sum(min_count=1)/denominator
                                                 if denominator > 0 else np.nan)
        for a in range(11):
            row[f"action_frequency_{a}"] = float(group[f"action_{a}"].mean()/300)
        row.update(record["downside"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLE / "all_economic_metrics.csv", index=False)
    pd.json_normalize(summary["paired_differences"], sep="_").to_csv(TABLE / "paired_differences.csv", index=False)
    pd.json_normalize(summary["overall"], sep="_").to_csv(TABLE / "mixture_performance.csv", index=False)
    # Small text exports let GitHub readers rederive every pilot-aware contrast
    # even when their connector cannot transfer private compressed binaries.
    moments = frame.groupby(["variant","environment","pilot","policy"],sort=True).agg(
        episodes=("objective","size"), objective_mean=("objective","mean"),
        objective_variance=("objective","var"), pnl_mean=("pnl","mean"), pnl_variance=("pnl","var"))
    moments.reset_index().to_csv(TABLE/"pilot_objective_moments.csv",index=False,float_format="%.17g")
    pivot=frame.pivot(index=["variant","environment","pilot","episode"],columns="policy",values="objective")
    paired_moments=[]
    for baseline in [p for p in POLICIES if p != "active"]:
        difference=(pivot.active-pivot[baseline]).rename("difference").reset_index()
        aggregate=difference.groupby(["variant","environment","pilot"],sort=True).agg(
            episodes=("difference","size"),mean=("difference","mean"),variance=("difference","var")).reset_index()
        aggregate["baseline"]=baseline
        paired_moments.append(aggregate)
    pd.concat(paired_moments,ignore_index=True).to_csv(TABLE/"pilot_paired_moments.csv",index=False,float_format="%.17g")
    records = []
    for fit in fits:
        w = np.array(fit["weights"])
        th, ka = [float(s[5:]) for s in fit["environment"].split("_")]
        truth = (np.array(fit["candidate_theta"]) == th) & (np.array(fit["candidate_kappa"]) == ka)
        records.append(dict(environment=fit["environment"], pilot=fit["pilot"],
                            true_parameter_weight=float(w[truth].sum()),
                            entropy=float(-np.sum(w*np.log(np.maximum(w, 1e-300)))),
                            maximum_weight=float(w.max())))
    pd.DataFrame(records).to_csv(TABLE / "pilot_parameter_diagnostics.csv", index=False)
    refinements = []
    for family in refinement["families"]:
        for axis in ["belief_grid", "quadrature"]:
            for r in family[axis]:
                # Full per-horizon evidence lives in practical_refinement.json;
                # this CSV is its compact, one-row-per-refinement index.
                compact = {k:v for k,v in r.items() if k not in
                           {"per_horizon", "remaining_horizons_checked", "numerical_contract"}}
                refinements.append(dict(theta=family["theta"], kappa=family["kappa"], mode=family["mode"], axis=axis, **compact))
    pd.DataFrame(refinements).to_csv(TABLE / "all_refinement_steps.csv", index=False)
    return pd.DataFrame(records)


def make_figures(frame, overall, paired):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 240, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 2, figsize=(9.3, 2.7), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, (p, color) in enumerate(zip(POLICIES, ["#64748b"]*5+["#176b98", "#b66c1d"])):
        v = overall["nominal", p]["objective"]
        ax[0].errorbar(v["mean"], i, xerr=[[v["mean"]-v["low"]], [v["high"]-v["mean"]]],
                      fmt="o", color=color, capsize=2)
    ax[0].set_yticks(range(7), [LABEL[p] for p in POLICIES]); ax[0].invert_yaxis()
    ax[0].axvline(0, color="#cbd5e1", lw=.8); ax[0].set_xlabel("Objective per episode (J units)")
    ax[0].grid(axis="x", alpha=.18)
    grid = np.array([[paired["nominal", f"theta{th:g}_kappa{ka:g}", "noinfo"]["paired"]["mean"]
                      for ka in [.002, .02, .1]] for th in [.15, .35, .65]])
    limit = max(.05, np.max(np.abs(grid)))
    ax[1].imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax[1].text(j, i, f"{grid[i,j]:+.3f}", ha="center", va="center",
                       color="white" if abs(grid[i,j]) > .65*limit else "#182c3c")
    ax[1].set_xticks(range(3), [".002", ".02", ".10"]); ax[1].set_xlabel("Switch probability κ")
    ax[1].set_yticks(range(3), [".15", ".35", ".65"]); ax[1].set_ylabel("Dependence θ")
    ax[1].set_title("Active − matched noinfo (J units)", fontsize=10)
    fig.tight_layout(pad=.6); fig.savefig(FIG / "nominal.png"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9.3, 3.1))
    for offset, (p, color) in enumerate([("active", "#176b98"), ("noinfo", "#b66c1d"),
                                        ("myopic", "#64748b"), ("independent", "#936ca9")]):
        vals = [overall[v, p]["objective"] for v in VARIANTS]
        ax[0].errorbar(np.arange(3)+(offset-1.5)*.07, [v["mean"] for v in vals],
                      yerr=[[v["mean"]-v["low"] for v in vals], [v["high"]-v["mean"] for v in vals]],
                      marker="o", markersize=3, lw=1, capsize=2, color=color, label=LABEL[p])
    ax[0].set_xticks(range(3), ["Nominal", "Signal dep.", "Fixed duration"])
    ax[0].set_ylabel("Objective per episode"); ax[0].grid(axis="y", alpha=.2)
    ax[0].legend(fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.02))
    for variant, color in zip(VARIANTS, ["#176b98", "#b66c1d", "#64748b"]):
        g = frame[(frame.policy == "active") & (frame.variant == variant)]
        values = [g[f"score_sum_x{x}"].sum()/g[f"submitted_sides_x{x}"].sum() for x in [-1, 0, 1]]
        ax[1].plot([-1, 0, 1], values, marker="o", markersize=3, lw=1, color=color, label=VLABEL[variant])
    ax[1].axhline(0, color="#111827", lw=.6); ax[1].set_xticks([-1, 0, 1]); ax[1].set_xlabel("Public signal x")
    ax[1].set_ylabel("Public residual / submitted side"); ax[1].grid(axis="y", alpha=.2); ax[1].legend(fontsize=9)
    fig.tight_layout(pad=.6); fig.savefig(FIG / "robustness.png"); plt.close(fig)


def write_appendix(refinement):
    source = (ROOT / "report/theory_appendix.md").read_text()
    intro, body = source.split("## 1. Observation law and filtering", 1)
    text = intro.split("## Pre-implementation")[0]+"## 1. Observation law and filtering"+body
    methods = (ROOT / "report/research_methods.md").read_text()
    stats = methods.split("## Paired uncertainty and deployment assessment\n", 1)[1].split("## Accounting attribution", 1)[0]
    text += "\n\n## 8. Pilot-aware statistical estimators\n\n"+stats
    text += ("\n\n## 9. Practical-horizon numerical refinement\n\n"
             "Both matched modes were checked independently at every public parameter point. "
             "Every remaining horizon 1 through 300 was checked. The table gives the last accepted "
             "belief-axis and integration-axis initial-state changes, maximized over horizons, "
             "with maximum common-state drift and worst per-horizon robust action disagreement across these two steps. "
             "The deployed belief grid is at least as fine as the accepted belief setting. "
             "All campaign tables use 161 integration nodes. For active (0.65, 0.002), "
             "the last integration check is 161 to 321 nodes at 1,281 beliefs; "
             "321 nodes were a verification refinement and were not deployed. "
             "Other last integration checks compare 81 to 161 nodes. "
             "Earlier failures and all steps remain in outputs/tables/all_refinement_steps.csv. "
             "The limits, enforced at every horizon, were initial drift 0.01, common-state drift 0.03, "
             "and disagreement 0.005 among states with fine-grid action gap above 0.005. "
             "Numerical ties use the prespecified absolute 1e-12 rule; near-tie counts are separate. "
             "These checks are not certified error bounds.\n\n"
             "| $\\theta$ | $\\kappa$ | Mode | Belief change | Integration change | Max drift | Robust mismatch |\n"
             "|--:|--:|:--|--:|--:|--:|--:|\n")
    for f in refinement["families"]:
        b, q = f["belief_grid"][-1], f["quadrature"][-1]
        text += (f"| {f['theta']} | {f['kappa']} | {f['mode']} | "
                 f"{b['maximum_initial_state_value_change_over_horizons']:.6f} | {q['maximum_initial_state_value_change_over_horizons']:.6f} | "
                 f"{max(b['maximum_common_state_value_change'],q['maximum_common_state_value_change']):.6f} | "
                 f"{max(b['maximum_per_horizon_robust_action_disagreement_fraction'],q['maximum_per_horizon_robust_action_disagreement_fraction']):.4f} |\n")
    mechanism = (ROOT / "report/mechanism_identification.md").read_text()
    mechanism = mechanism.split("\n", 1)[1].replace("\n## ", "\n### ")
    text += "\n" + r"\newpage" + "\n\n## 10. Matched mechanism interventions\n" + mechanism
    md = GEN / "appendix.md"; md.write_text(text)
    header = GEN / "appendix_header.tex"
    header.write_text("\\usepackage{amsmath,amssymb}\n\\allowdisplaybreaks\n\\emergencystretch=2em\n")
    pypandoc.convert_file(str(md), to="latex", format="markdown+tex_math_single_backslash",
                         outputfile=str(ROOT / "report/Market_Making_Appendix.tex"),
                         extra_args=["--standalone", "--variable=papersize:a4", "--variable=geometry:margin=21mm", "--variable=fontsize:10pt",
                                     "--variable=colorlinks:true", "--include-in-header="+str(header)])


def main():
    invalidate(ROOT)
    for d in [GEN, FIG, TABLE]: d.mkdir(parents=True, exist_ok=True)
    manifest = read("outputs/full/manifest.json")
    if manifest["status"] != "complete": raise RuntimeError("Full campaign is not complete.")
    summary = read("outputs/full/summary.json")
    frame = pd.read_csv(ROOT / "outputs/full/episodes.csv.gz")
    fits, refinement = read("outputs/full/pilot_fits.json"), read("outputs/reference/practical_refinement.json")
    audit = artifact_audit(frame, summary, manifest)
    diagnostics = export_tables(frame, summary, fits, refinement)
    overall = {(r["variant"], r["policy"]): r for r in summary["overall"]}
    cells = {(r["variant"], r["environment"], r["policy"]): r for r in summary["per_environment"]}
    paired = {(r["variant"], r["environment"], r["baseline"]): r for r in summary["paired_differences"]}
    primary, active = summary["primary_comparison"], overall["nominal", "active"]
    put("key_numbers", "% Empirical values are generated from outputs/full.\n"
        +rf"\newcommand{{\PilotTruthMin}}{{{100*diagnostics.true_parameter_weight.min():.2f}\%}}")
    put("executive_results",
        rf"The frozen campaign comprises \textbf{{189,000 policy episodes}} on 27,000 paired exogenous trajectories, "
        rf"with 90 independent pilots, nine environments, seven controls and three variants. "
        rf"On the uniform nominal population, active objective is "
        rf"{active['objective']['mean']:.3f}; its paired advantage over the matched filtering baseline is "
        rf"{primary['mean']:+.3f} (95\% interval {interval(primary)}). Active net PnL is {active['pnl']['mean']:.3f}. "
        rf"The prespecified one-sided lower advantage bound is {primary['one_sided_lower']:.3f} against "
        rf"an acceptance threshold of .05 objective units. These pilot-aware intervals are empirical, not finite-sample guarantees.")
    ref = read("outputs/reference/rollout/verification.json")
    rows = [r"\begin{center}\begin{tabular}{lrr}\toprule Policy & Evaluated $J$ (MC SE) & Initial planner value\\\midrule"]
    for r in ref["policies"]:
        v = f"{r['planner_initial_value']:.5f}" if "planner_initial_value" in r else "--"
        rows.append(rf"{LABEL[r['policy']]} & {r['objective_mean']:.5f} ({r['objective_mc_se']:.5f}) & {v}\\")
    rows.append(r"\bottomrule\end{tabular}\end{center}"); put("reference_table", "\n".join(rows))
    deltas = [r["family_adjusted"] for r in summary["paired_differences"] if r["variant"] == "nominal" and r["baseline"] == "noinfo"]
    put("nominal_narrative",
        rf"The same policy is evaluated everywhere. Its uniform-mixture objective is {active['objective']['mean']:.3f}, "
        rf"versus {overall['nominal','noinfo']['objective']['mean']:.3f} for the matched baseline and "
        rf"{overall['nominal','full_information']['objective']['mean']:.3f} for the privileged reference. "
        rf"Of nine family-adjusted paired intervals, {sum(r['low']>0 for r in deltas)} are entirely positive "
        rf"and {sum(r['high']<0 for r in deltas)} entirely negative. This does not imply a uniform advantage.")
    rows = [r"\begin{center}\small\begin{tabular}{rrrrrrl}\toprule",
            r"$\theta$ & $\kappa$ & Active & Noinfo & Myopic & Privileged & $\Delta J$ [adjusted CI]\\\midrule"]
    for th, ka in manifest["protocol"]["environments"]:
        env = f"theta{th:g}_kappa{ka:g}"
        means = [cells["nominal", env, p]["objective"]["mean"] for p in ["active", "noinfo", "myopic", "full_information"]]
        d = paired["nominal", env, "noinfo"]["family_adjusted"]
        rows.append(f"{th:g} & {ka:g} & "+" & ".join(f"{m:.3f}" for m in means)+rf" & {d['mean']:+.3f} {interval(d)}\\")
    rows.append(r"\bottomrule\end{tabular}\end{center}"); put("nominal_table", "\n".join(rows))
    g = frame[(frame.variant == "nominal") & (frame.policy == "active")]
    m = g.select_dtypes("number").mean(); q05 = g.pnl.quantile(.05); tail = g.loc[g.pnl <= q05, "pnl"].mean()
    put("attribution_narrative",
        rf"Active nominal mean spread capture is {m.spread_capture:.3f}, passive selection {m.execution_selection:+.3f}, "
        rf"directional exposure {m.directional_exposure:+.3f}, passive fees {m.passive_fees:.3f}, liquidation cost "
        rf"{m.liquidation_cost:.3f}, and inventory penalty {m.inventory_penalty:.3f}. The empirical 5th PnL "
        rf"percentile is {q05:.3f}, with mean below it {tail:.3f}; tail estimates have limited precision. "
        rf"Complete fees, turnover, bid/ask fills, action frequencies and exposure are saved for all 189 cells.")
    parts = []
    for v in VARIANTS[1:]:
        vals = [overall[v, p]["objective"]["mean"] for p in ["active", "noinfo", "myopic"]]
        parts.append(f"{VLABEL[v]}: active {vals[0]:.3f}, noinfo {vals[1]:.3f}, myopic {vals[2]:.3f} objective units.")
    put("stress_narrative", " ".join(parts)+" These are separate frozen-policy experiments.")
    monitor = {}
    for variant in VARIANTS:
        g = frame[(frame.policy == "active") & (frame.variant == variant)]
        monitor[variant] = dict(
            switch_error=float(g.post_switch_error_count.sum()/g.post_switch_window_periods.sum()),
            gap=float(g.max_quote_gap.mean()),
            residuals=[float(g[f"score_sum_x{x}"].sum()/g[f"submitted_sides_x{x}"].sum()) for x in [-1, 0, 1]])
    residuals = monitor["state_dependence"]["residuals"]
    put("monitoring_narrative",
        rf"In nominal/signal-dependent/fixed-duration order, active classification error during the first "
        rf"ten periods after a switch is "
        +"/".join(f"{100*monitor[v]['switch_error']:.1f}" for v in VARIANTS)
        +rf"\%; its mean longest gap without quotes is "
        +"/".join(f"{monitor[v]['gap']:.1f}" for v in VARIANTS)
        +rf" periods. Signal-dependent public residuals at $x=-1,0,1$ are "
        +", ".join(f"{r:+.4f}" for r in residuals)
        +". Residuals use public feedback; switch errors require hidden labels. The plot supplies the nominal comparison.")
    noinfo = frame[(frame.policy == "noinfo") & (frame.variant == "nominal")]
    fixed_noinfo = frame[(frame.policy == "noinfo") & (frame.variant == "fixed_duration")]
    with (GEN / "monitoring_narrative.tex").open("a") as file:
        file.write(f"\nMatched noinfo's mean longest quote gap is {noinfo.max_quote_gap.mean():.1f} periods nominally "
                   f"and {fixed_noinfo.max_quote_gap.mean():.1f} under fixed durations: avoiding quotes slows access to evidence. "
                   "The fixed-duration public residual remains much closer to zero than the signal-dependent "
                   "residual, despite worse hidden-state errors; this pooled monitor alone misses much of that failure.\n")
    (TABLE / "monitoring_summary.json").write_text(json.dumps(monitor, indent=2)+"\n")
    drops = []
    for v in VARIANTS[1:]:
        for th, ka in manifest["protocol"]["environments"]:
            env = f"theta{th:g}_kappa{ka:g}"; s, n = cells[v, env, "active"], cells["nominal", env, "active"]
            drops.append((s["objective"]["mean"]-n["objective"]["mean"], v, env, th, ka, s, n))
    drop, v, env, th, ka, s, n = min(drops, key=lambda r:r[0])
    failure = dict(illustration_selection="Largest active objective deterioration across prescribed stresses, retrospective only",
                   variant=v, environment=env, objective_change=drop, nominal=n, stress=s, matched_contrast=paired[v,env,"noinfo"])
    (TABLE / "failure_illustration.json").write_text(json.dumps(failure, indent=2)+"\n")
    put("failure_narrative",
        rf"\textbf{{Failure illustration (retrospective).}} The largest observed active deterioration occurs under "
        rf"{VLABEL[v].lower()} after training at $(\theta,\kappa)=({th:g},{ka:g})$: $J$ changes from "
        rf"{n['objective']['mean']:.3f} to {s['objective']['mean']:.3f} ({drop:+.3f}). Hidden-label Brier error changes "
        rf"from {n['metrics']['regime_brier']:.3f} to {s['metrics']['regime_brier']:.3f}; wrong high-confidence predictions "
        rf"from {100*n['metrics']['wrong_confident_fraction']:.2f}\% to {100*s['metrics']['wrong_confident_fraction']:.2f}\%. "
        rf"This selected diagnostic illustrates vulnerability; it is not a prespecified hypothesis test or policy-selection rule.")
    accepted = primary["decision"].startswith("prefer_active")
    conclusion = "prefer active within the nominal synthetic population" if accepted else "retain the matched no-information-value baseline"
    put("deployment_narrative",
        rf"\textbf{{Decision: {conclusion}.}} Active-minus-noinfo mean is {primary['mean']:+.4f}, with paired 95\% "
        rf"interval {interval(primary,4)} and one-sided 95\% lower bound {primary['one_sided_lower']:.4f}. "
        rf"The latter {'exceeds' if primary['one_sided_lower']>.05 else 'does not exceed'} the fixed .05 threshold. "
        rf"Active net PnL has one-sided lower bound {primary['active_net_pnl']['one_sided_lower']:.3f}. "
        rf"This rule assesses a fresh nominally generated 100-episode pilot and the uniform nine-point stationary population. "
        rf"It gives no approval for either misspecified process or real-market trading.")
    if not accepted:
        distinction = ("The two-sided upper bound is also below .05: the estimated benefit is too small "
                       "for the stipulated threshold, rather than merely failing a significance check."
                       if primary["high"] <= .05 else
                       "The interval still includes economically acceptable benefits: the magnitude remains unresolved "
                       "at this evidence level, even though the operational rule retains the baseline.")
        with (GEN / "deployment_narrative.tex").open("a") as file:
            file.write("\n"+distinction+"\n")
    computation = manifest["control_computation"]
    seconds = sum(c.get("seconds", 0) for c in computation)
    put("runtime_narrative",
        rf"The corrective full campaign, starting with an empty cache, took {manifest['wall_seconds']/60:.1f} minutes on ARM macOS with "
        rf"{manifest['hardware']['cpu_count']} reported logical CPUs; measured table-build times sum to {seconds:.1f} seconds. "
        rf"Python {manifest['python']}, NumPy {manifest['software']['numpy']} and SciPy {manifest['software']['scipy']} "
        rf"accompany the frozen code hash. The maximum final cash/wealth discrepancy was {audit['max_ledger_error']:.2e}. "
        rf"All 90 pilot datasets and 450 stream identifiers were reconciled to the manifest.")
    resources = read("outputs/verification/reaudit/resource_envelope.json")
    assert resources["passed"] and resources["child_exit_code"] == 0
    with (GEN / "runtime_narrative.tex").open("a") as file:
        file.write("\n"+rf"A fresh central-model $T=300$, $Q_{{\max}}=5$, 641-belief/161-node active table "
                   rf"peaked at {resources['peak_process_rss_bytes']/2**30:.2f} GiB process RSS and wrote "
                   rf"{resources['artifact_bytes']['logical_bytes']/2**20:.0f} MiB. "
                   r"Use a dedicated 16 GiB host and 30 GiB free disk as a conservative working envelope, "
                   r"not a measured minimum; the full-run peak was not measured. "
                   r"The reproduction guide separates smoke verification from full cold computation."+"\n")
    study, study_identity = mechanism_study()
    pd.DataFrame(study["contrasts"]).to_csv(TABLE / "mechanism_contrasts.csv", index=False)
    pd.DataFrame([{key: row[key] for key in ("policy", "mean", "se", "pnl_mean",
                                          "inventory_sq_mean_per_period")}
                  for row in study["policies"]]).to_csv(TABLE / "mechanism_policies.csv", index=False)
    make_figures(frame, overall, paired); write_appendix(refinement)
    claims = dict(primary=primary, active_nominal=active, failure_illustration=failure,
                  pilot_true_weight_min=float(diagnostics.true_parameter_weight.min()),
                  pilot_true_weight_median=float(diagnostics.true_parameter_weight.median()), audit=audit,
                  mechanism_study={"identity": study_identity, "contrasts": study["contrasts"],
                                   "episodes": study["episodes"], "rows": study["rows"]})
    (GEN / "report_claims.json").write_text(json.dumps(claims, indent=2)+"\n")
    effects = {row["mechanism"]: row for row in study["contrasts"]}
    specifications = [
        ("forecast", "Forecast", r"Remove only $\mu x\E[q']$ from every noinfo planning reward; keep $X$ in fills and the true filter."),
        ("current_inference", "Current inference", r"Same noinfo table; route $.5$ instead of the selected-feedback posterior into each decision."),
        ("inventory_continuation", "Inventory", r"Same noinfo continuation; replace $q'$ by $q$ for $m\ge2$. Preserve exact last-step scores."),
        ("future_information", "Future feedback", r"Active versus noinfo; change only feedback conditioning in future planning. Both filter at runtime."),
    ]
    rows = []
    for key, label, description in specifications:
        result = effects[key]
        rows.append(label + " & " + description + " & $"
                    + f"{result['mean']:+.4f}" + r"$\newline"
                    + rf"$[{result['simultaneous_low']:.4f},{result['simultaneous_high']:.4f}]$\\")
    put("mechanism_table", r"""
\textbf{Matched mechanism study (reduced model).} A separately frozen study uses
20,000 fresh paired episodes, seven policies and four planned contrasts.
\begin{center}\small
\begin{tabular}{p{.14\linewidth}p{.54\linewidth}p{.24\linewidth}}\toprule
Channel & Decision intervention & $\Delta J$ [simultaneous 95\%]\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{center}
The first three rows compare filtered noinfo with the named ablation. All retain
the same true simulator, accounting, constraints and final liquidation.
Intervals use paired-episode Student inference adjusted for these four contrasts;
this separate family leaves the canonical primary and 162 contrasts unchanged.
Effects are not additive. Inventory continuation includes downstream forecast
opportunities; it is not a pure risk-cost effect. The appendix defines every
intervention, including the inventory-off rule's deliberately suppressed
continuation argument, and gives cold-repeat and independent ledger checks.
""")
    receipt = record_build(ROOT)
    print(json.dumps({key: receipt[key] for key in ("state", "generated", "episodes", "primary", "worst_stress_drop")}, indent=2))


if __name__ == "__main__":
    main()
