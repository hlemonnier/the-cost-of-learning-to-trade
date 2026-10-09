"""Explicit publication migration checks for recorded scientific identities."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _manifest(root: Path) -> dict:
    value = json.loads((root / "docs/publication-manifest.json").read_text())
    if value.get("schema") != 1:
        raise ValueError("Unsupported publication migration schema")
    return value


def recorded_digest(root: Path, relative: str) -> str:
    record = _manifest(root)["source_migration"][relative]
    path = root / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["published_sha256"]:
        raise ValueError(f"Published source changed after its migration: {relative}")
    return record["recorded_sha256"]


def recorded_core_hash(root: Path) -> str:
    manifest = _manifest(root)
    expected = manifest["core_files"]
    actual = sorted(str(p.relative_to(root)) for p in (root / "src/trade_learning").glob("*.py"))
    if actual != expected:
        raise ValueError("Published core source inventory differs from migration")
    for relative in actual:
        recorded_digest(root, relative)
    return manifest["recorded_core_source_sha256"]

