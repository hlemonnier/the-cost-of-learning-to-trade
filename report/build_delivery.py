"""Generate, compile and bind both report PDFs to the current scientific inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from pypdf import PdfReader
from delivery_receipts import (DOCUMENTS, identity, inputs, invalidate, record_build,
                               sha, timestamp, validate, write)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tectonic", default="tectonic", help="Tectonic executable for a portable build")
    parser.add_argument("--compiler-helper", type=Path,
                        help="Optional Codex LaTeX skill compile_latex.py helper")
    args = parser.parse_args()
    invalidate(ROOT)
    try:
        subprocess.run([sys.executable, str(ROOT / "report/build_report.py")], check=True)
        subprocess.run([sys.executable, str(ROOT / "report/export_markdown.py")], check=True)
        before, provenance = inputs(ROOT), identity(ROOT)
        output = ROOT / "output/pdf"
        output.mkdir(parents=True, exist_ok=True)
        if not args.compiler_helper and not shutil.which(args.tectonic):
            raise FileNotFoundError("Tectonic is required; pass --tectonic /path/to/tectonic")
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="delivery-compile-", dir=ROOT / "tmp") as temp:
            staged = Path(temp)
            receipts = {}
            for name, stem in DOCUMENTS.items():
                if args.compiler_helper:
                    command = [sys.executable, str(args.compiler_helper.resolve()),
                               str(ROOT / f"report/{stem}.tex"), "--compiler", "tectonic",
                               "--output-directory", str(staged), "--json"]
                else:
                    command = [args.tectonic, "--untrusted", "--outdir", str(staged), f"{stem}.tex"]
                compiled = subprocess.run(command, cwd=ROOT / "report", text=True, capture_output=True)
                log = compiled.stdout + compiled.stderr
                (ROOT / f"tmp/{name}-delivery-compile.log").write_text(log)
                if compiled.returncode:
                    raise RuntimeError(f"{name} compilation failed: {log[-2000:]}")
                if args.compiler_helper:
                    result = json.loads(compiled.stdout)
                    if result.get("exitCode") != 0 or not result.get("pdfExists"):
                        raise RuntimeError(f"Compiler helper failed for {name}")
                produced = staged / f"{stem}.pdf"
                if not produced.is_file():
                    raise RuntimeError(f"Compiler did not produce a new {name} PDF")
                if inputs(ROOT) != before:
                    raise RuntimeError("Report inputs changed during compilation")
                pages = len(PdfReader(produced).pages)
                if pages < 1 or (name == "main" and pages > 8):
                    raise ValueError(f"Invalid {name} PDF page count: {pages}")
                receipts[name] = {
                    "schema": 2, "state": "complete", "recorded_utc": timestamp(), "exit_code": 0,
                    "command": command, "compiler": "tectonic", "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
                    "identity": provenance, "input_sha256": before, "pdf": f"output/pdf/{stem}.pdf",
                    "pdf_sha256": sha(produced), "pages": pages,
                    "scope": "New PDF from an initially empty compilation directory; visual inspection remains separate."}
            for name, stem in DOCUMENTS.items():
                (staged / f"{stem}.pdf").replace(output / f"{stem}.pdf")
                write(ROOT / f"outputs/{name}_compile.json", receipts[name])
        record_build(ROOT, "complete")
        print(json.dumps(validate(ROOT), indent=2))
    except BaseException:
        invalidate(ROOT, "failed")
        raise


if __name__ == "__main__":
    main()
