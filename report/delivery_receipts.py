"""Acyclic, exact-byte provenance for generated inputs and compiled documents.

See verification/delivery_failure_modes.md. These receipts certify local build
identity, not an external review or correctness of an economic conclusion.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

DOCUMENTS = {"main": "Market_Making_Study", "appendix": "Market_Making_Appendix"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def inputs(root):
    """All scientific inputs and report dependencies, excluding status/PDF outputs."""
    root = Path(root)
    paths = {root / p for p in (
        "configs/protocol.json", "configs/mechanism_study.json",
        "verification/mechanism_study.py", "verification/verify_mechanism_study.py",
        "verification/verify_control.py", "verification/mechanism_failure_modes.md",
        "BENCHMARK.md",
        "Trade Learning_Part_1_Task_Statement.pdf", "outputs/full/manifest.json",
        "outputs/full/episodes.csv.gz", "outputs/full/summary.json",
        "outputs/full/pilot_fits.json", "outputs/reference/practical_refinement.json",
        "outputs/verification/reaudit/mechanism_contrasts.json",
        "outputs/verification/reaudit/resource_envelope.json",
        "outputs/reference/rollout/verification.json",
        "outputs/reference/verification.json", "outputs/environment/verification.json",
        "outputs/theory/verification.json", "outputs/verification/seed_separation.json",
        "outputs/verification/pipeline/verification.json",
        "outputs/verification/final_artifact_audit.json")}
    paths.update((root / "outputs/full/pilots").glob("*.npz"))
    paths.update(p for p in (root / "outputs/verification/reaudit/mechanism_study").glob("*")
                 if p.is_file())
    for folder, suffixes in (("src/trade_learning", {".py"}),
                             ("report", {".py", ".tex", ".md", ".json"}),
                             ("outputs/tables", {".csv", ".json"}),
                             ("outputs/figures", {".png", ".pdf"})):
        paths.update(p for p in (root / folder).rglob("*")
                     if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def identity(root):
    root = Path(root)
    manifest = read(root / "outputs/full/manifest.json")
    code = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                                  for p in sorted((root / "src/trade_learning").glob("*.py")))).hexdigest()
    require(code == manifest["code_hash"], "Production source differs from campaign")
    raw = sha(root / "outputs/full/episodes.csv.gz")
    require(raw == manifest["csv_sha256"], "Raw ledger differs from campaign")
    claims = read(root / "report/generated/report_claims.json")
    summary = read(root / "outputs/full/summary.json")
    require(claims["primary"] == summary["primary_comparison"], "Claims and summary primary differ")
    require(claims["audit"]["production_code_hash"] == code, "Claim source identity differs")
    require(claims["audit"]["final_csv_sha256"] == raw, "Claim raw identity differs")
    return {"production_source_sha256": code, "raw_csv_sha256": raw,
            "manifest_sha256": sha(root / "outputs/full/manifest.json"),
            "claims_sha256": sha(root / "report/generated/report_claims.json")}


def invalidate_package(root, state="needs_rebuild"):
    write(Path(root) / "output/review_package_verification.json",
          {"passed": False, "state": state, "recorded_utc": timestamp(),
           "scope": "The existing ZIP, if present, is historical until the current delivery is packaged and verified."})


def invalidate(root, state="building"):
    invalidate_package(root)
    for name in ("report_build", "main_compile", "appendix_compile"):
        write(Path(root) / f"outputs/{name}.json",
              {"schema": 2, "state": state, "generated": False, "recorded_utc": timestamp(),
               "scope": "A current build is incomplete; previous success is not carried forward."})


def record_build(root, state="inputs_generated"):
    root = Path(root)
    claims = read(root / "report/generated/report_claims.json")
    result = {"schema": 2, "state": state, "generated": True, "recorded_utc": timestamp(),
              "scope": "Local generation identity; PDF visual review and external sign-off are separate.",
              "episodes": claims["audit"]["episodes"], "primary": claims["primary"],
              "worst_stress_drop": claims["failure_illustration"]["objective_change"],
              "identity": identity(root), "input_sha256": inputs(root)}
    if state == "complete":
        result["compile_receipt_sha256"] = {
            f"outputs/{name}_compile.json": sha(root / f"outputs/{name}_compile.json")
            for name in DOCUMENTS}
        result["pdf_sha256"] = {
            f"output/pdf/{stem}.pdf": sha(root / f"output/pdf/{stem}.pdf")
            for stem in DOCUMENTS.values()}
    write(root / "outputs/report_build.json", result)
    return result


def validate(root):
    from pypdf import PdfReader
    root = Path(root)
    build = read(root / "outputs/report_build.json")
    require(build.get("schema") == 2 and build.get("generated") is True
            and build.get("state") == "complete", "Current build receipt is incomplete or historical")
    current_identity, current_inputs = identity(root), inputs(root)
    require(build["identity"] == current_identity, "Build identity is stale")
    require(build["input_sha256"] == current_inputs, "Build inputs changed after generation")
    claims = read(root / "report/generated/report_claims.json")
    require(build["episodes"] == claims["audit"]["episodes"] == 189000, "Build budget differs")
    require(build["primary"] == claims["primary"], "Build receipt has stale primary results")
    require(build["worst_stress_drop"] == claims["failure_illustration"]["objective_change"],
            "Build receipt has stale stress results")
    pdfs = {}
    for name, stem in DOCUMENTS.items():
        path = f"outputs/{name}_compile.json"
        receipt = read(root / path)
        pdf = f"output/pdf/{stem}.pdf"
        require(build["compile_receipt_sha256"][path] == sha(root / path), "Compile receipt changed: " + name)
        require(receipt.get("state") == "complete" and receipt.get("exit_code") == 0,
                "Compilation was incomplete: " + name)
        require(receipt["identity"] == current_identity and receipt["input_sha256"] == current_inputs,
                "Compilation used different inputs: " + name)
        require(receipt["pdf"] == pdf and receipt["pdf_sha256"] == build["pdf_sha256"][pdf] == sha(root / pdf),
                "Compiled PDF changed: " + name)
        pages = len(PdfReader(root / pdf).pages)
        require(pages == receipt["pages"] and pages >= 1, "PDF page count differs: " + name)
        if name == "main":
            require(pages <= 8, "Main report exceeds eight pages")
        pdfs[pdf] = {"sha256": sha(root / pdf), "pages": pages}
    return {"passed": True, "identity": current_identity,
            "checked_input_files": len(current_inputs), "pdfs": pdfs,
            "build_receipt_sha256": sha(root / "outputs/report_build.json"),
            "scope": "Current source/data/claim/input/PDF receipts agree; no visual or external sign-off inferred."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = validate(args.root.resolve())
    if args.out:
        write(args.out, result)
    print(json.dumps(result, indent=2))
