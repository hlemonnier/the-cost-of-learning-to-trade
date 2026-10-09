"""Run the opaque-gzip transport CLI end to end on a small disposable fixture."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
TOOL = HERE / "data_transport.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_e2e() -> dict:
    with tempfile.TemporaryDirectory(prefix="trade-learning-data-transport-e2e-") as temporary:
        root = Path(temporary)
        source = root / "episodes.csv.gz"
        raw = b"".join(hashlib.sha256(index.to_bytes(4, "little")).digest() for index in range(1024))
        source.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        store = root / "data_transport"
        split = call("split", "--source", str(source), "--store", str(store), "--part-bytes", "4096")
        require(split.returncode == 0, f"Fixture split failed: {split.stderr}")
        split_result = json.loads(split.stdout)
        manifest = json.loads((store / "manifest.json").read_text())
        require(len(manifest["parts"]) >= 3 and max(item["bytes"] for item in manifest["parts"]) <= 4096,
                "Fixture did not exercise multiple bounded parts")
        destination = root / "fresh" / "episodes.csv.gz"
        restored = call("reconstruct", "--store", str(store), "--destination", str(destination))
        require(restored.returncode == 0 and json.loads(restored.stdout)["status"] == "reconstructed" and
                destination.read_bytes() == source.read_bytes(), "Fresh reconstruction is not byte-identical")
        repeated = call("reconstruct", "--store", str(store), "--destination", str(destination))
        require(repeated.returncode == 0 and json.loads(repeated.stdout)["status"] == "already_present" and
                destination.read_bytes() == source.read_bytes(), "Identical existing destination was not authenticated")

        faults = []

        def reject(label: str, mutation, expected: str, existing: bytes | None = None) -> None:
            folder = root / f"fault-{label}"
            copied = folder / "data_transport"
            shutil.copytree(store, copied)
            mutation(copied)
            target = folder / "result" / "episodes.csv.gz"
            if existing is not None:
                target.parent.mkdir(parents=True)
                target.write_bytes(existing)
            result = call("reconstruct", "--store", str(copied), "--destination", str(target))
            require(result.returncode != 0 and expected in result.stderr,
                    f"{label} did not reject for its intended reason: {result.stderr}")
            require((not target.exists() if existing is None else target.read_bytes() == existing),
                    f"{label} published or overwrote a destination")
            require(not list(target.parent.glob(".episodes.csv.gz.stage-*")),
                    f"{label} left a partial staging file")
            faults.append({"label": label, "returncode": result.returncode,
                           "expected_error": expected, "no_partial_output": True})

        def edit_manifest(copied: Path, edit) -> None:
            path = copied / "manifest.json"
            value = json.loads(path.read_text())
            edit(value)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

        second = manifest["parts"][1]["file"]
        reject("missing_part", lambda copied: (copied / second).unlink(), "Unexpected or missing transport file")

        def corrupt(copied: Path) -> None:
            path = copied / second
            value = bytearray(path.read_bytes())
            value[0] ^= 0x40
            path.write_bytes(value)

        reject("corrupt_part", corrupt, "Part digest mismatch")
        reject("reordered_parts", lambda copied: edit_manifest(
            copied, lambda value: value["parts"].reverse()), "Unexpected part index")
        reject("duplicate_part", lambda copied: edit_manifest(
            copied, lambda value: value["parts"].__setitem__(1, dict(value["parts"][0]))),
            "Unexpected part index")
        reject("parent_escape", lambda copied: edit_manifest(
            copied, lambda value: value["parts"][1].__setitem__("file", "../outside")),
            "Unexpected part filename")
        reject("absolute_escape", lambda copied: edit_manifest(
            copied, lambda value: value["parts"][1].__setitem__("file", "/tmp/outside")),
            "Unexpected part filename")
        reject("symlink_part", lambda copied: ((copied / second).unlink(),
               (copied / second).symlink_to(source)), "Missing or nonregular file")
        reject("different_existing_destination", lambda copied: None,
               "Destination differs from authenticated original", existing=b"different bytes")

        occupied_before = {path.name: sha(path) for path in store.iterdir()}
        occupied = call("split", "--source", str(source), "--store", str(store))
        require(occupied.returncode != 0 and "Transport store already exists" in occupied.stderr and
                occupied_before == {path.name: sha(path) for path in store.iterdir()},
                "Occupied store was changed by a second split")
        faults.append({"label": "occupied_store", "returncode": occupied.returncode,
                       "expected_error": "Transport store already exists", "no_partial_output": True})

        oversized = root / "oversized-store"
        large = call("split", "--source", str(source), "--store", str(oversized),
                     "--part-bytes", str(32 * 1024 * 1024 + 1))
        require(large.returncode != 0 and "Part size must be between 1 and 32 MiB" in large.stderr and
                not oversized.exists(), "Oversized part limit was not rejected")
        faults.append({"label": "oversized_part_limit", "returncode": large.returncode,
                       "expected_error": "Part size must be between 1 and 32 MiB", "no_partial_output": True})

        return {"kind": "opaque_gzip_data_transport_e2e", "passed": True,
                "scope": "Small disposable CLI fixture only; no scientific data read or modified",
                "replay_command": "python3 extensions/bayes_adaptive/data_transport_e2e.py",
                "tool_sha256": sha(TOOL), "e2e_script_sha256": sha(Path(__file__)),
                "fixture_bytes": source.stat().st_size, "fixture_sha256": sha(source),
                "manifest_sha256": sha(store / "manifest.json"),
                "part_count": len(manifest["parts"]), "part_bytes": manifest["part_bytes"],
                "split_status": split_result["status"],
                "fresh_status": json.loads(restored.stdout)["status"],
                "existing_status": json.loads(repeated.stdout)["status"],
                "reconstructed_sha256": sha(destination), "compressed_bytes_identical": True,
                "faults": faults}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path,
                        default=HERE / "outputs/data_transport_validation/receipt.json")
    args = parser.parse_args()
    args.receipt.unlink(missing_ok=True)
    result = run_e2e()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "receipt": str(args.receipt), "faults": len(result["faults"])}))


if __name__ == "__main__":
    main()
