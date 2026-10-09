"""Repeatable acceptance for a public research checkout and its frozen evidence."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
from pypdf import PdfReader


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("outputs/reproduction/publication.json"))
    parser.add_argument("--forbid", action="append", default=[], help="Content regex to reject; repeat for multiple rules")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")[:-1]
    migration = json.loads((root / "docs/publication-manifest.json").read_text())
    issues = []
    for relative, expected in migration["preserved_numeric_assets"].items():
        if digest(root / relative) != expected:
            issues.append(f"Numeric asset differs: {relative}")
    for relative, record in migration["source_migration"].items():
        if digest(root / relative) != record["published_sha256"]:
            issues.append(f"Published source differs: {relative}")
    rules = [re.compile(rule, re.I) for rule in args.forbid]
    pdfs = {}
    for relative in paths:
        p = root / relative
        if not p.is_file():
            issues.append(f"Missing tracked file: {relative}")
            continue
        text = ""
        if p.suffix == ".pdf":
            reader = PdfReader(p)
            text = "\n".join(page.extract_text() or "" for page in reader.pages) + str(reader.metadata)
            pdfs[relative] = {"pages": len(reader.pages), "sha256": digest(p)}
        elif p.suffix == ".gz":
            with gzip.open(p, "rt", errors="replace") as stream:
                carry = ""
                for block in iter(lambda: stream.read(1024 * 1024), ""):
                    sample = carry + block
                    if any(rule.search(sample) for rule in rules):
                        issues.append(f"Content rule failed: {relative}")
                        break
                    carry = sample[-256:]
        elif p.suffix in {".md", ".py", ".json", ".toml", ".tex", ".txt", ".csv", ".tsv", ".yml", ".yaml", ".cff", ".svg"} or p.name in {"LICENSE", ".gitignore"}:
            text = p.read_text()
        if any(rule.search(relative) or rule.search(text) for rule in rules):
            issues.append(f"Content rule failed: {relative}")
        if p.suffix == ".md":
            plain = re.sub(r"```.*?```", "", text, flags=re.S)
            targets = re.findall(r"\[[^\]]*\]\(([^\n)]+)\)", plain)
            targets += re.findall(r'(?:href|src)="([^"]+)"', plain)
            for raw in targets:
                target = raw.strip().split(' "')[0].strip("<>")
                if target.startswith(("#", "http:", "https:", "mailto:", "codex:")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if target and not (p.parent / target).exists():
                    issues.append(f"Broken relative link: {relative} -> {target}")
    result = {
        "passed": not issues,
        "tracked_files_checked": len(paths),
        "content_rules_checked": len(rules),
        "preserved_numeric_assets": len(migration["preserved_numeric_assets"]),
        "source_migration_files": len(migration["source_migration"]),
        "relative_links_valid": not any("relative link" in issue for issue in issues),
        "pdfs": pdfs,
        "issues": issues,
        "scope": "Publication content, recorded data hashes and explicit source migration; numerical simulation is checked separately.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pdfs"}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

