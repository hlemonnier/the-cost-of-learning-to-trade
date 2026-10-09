"""Split and reconstruct the exact episode gzip bytes for Git publication.

The review ZIP carries the unsplit original. This tool never parses or
recompresses the gzip stream and never overwrites an existing destination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "outputs/full/episodes.csv.gz"
DEFAULT_STORE = HERE / "data_transport"
MAX_PART_BYTES = 32 * 1024 * 1024
PART_PREFIX = "episodes.csv.gz.part-"
MANIFEST_NAME = "manifest.json"
MANIFEST_KEYS = {"format_version", "target", "part_bytes", "source_bytes", "source_sha256", "parts"}
PART_KEYS = {"index", "file", "bytes", "sha256"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_int(value: object) -> bool:
    return type(value) is int


def is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def regular_file(path: Path) -> os.stat_result:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"Missing or nonregular file: {path}")
    return info


def digest_file(path: Path) -> tuple[int, str]:
    before = regular_file(path)
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before), f"File changed before reading: {path}")
        while block := stream.read(1024 * 1024):
            digest.update(block)
        require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before) and
                fingerprint(regular_file(path)) == fingerprint(before), f"File changed while reading: {path}")
    return before.st_size, digest.hexdigest()


def read_json_file(path: Path) -> object:
    before = regular_file(path)
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before), f"File changed before reading: {path}")
        raw = stream.read()
        require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before) and
                fingerprint(regular_file(path)) == fingerprint(before), f"File changed while reading: {path}")
    return json.loads(raw)


def part_name(index: int) -> str:
    return f"{PART_PREFIX}{index:06d}"


def split(source: Path, store: Path, part_bytes: int) -> dict:
    require(is_int(part_bytes) and 0 < part_bytes <= MAX_PART_BYTES, "Part size must be between 1 and 32 MiB")
    require(not store.exists() and not store.is_symlink(), "Transport store already exists")
    source_before = regular_file(source)
    require(source_before.st_size > 0, "Source is empty")
    store.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{store.name}.stage-", dir=store.parent))
    store_created = False
    try:
        full_hash = hashlib.sha256()
        parts = []
        with os.fdopen(os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
            require(fingerprint(os.fstat(stream.fileno())) == fingerprint(source_before), "Source changed before split")
            while block := stream.read(part_bytes):
                index = len(parts)
                name = part_name(index)
                with (staged / name).open("xb") as output:
                    output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                full_hash.update(block)
                parts.append({"index": index, "file": name, "bytes": len(block),
                              "sha256": hashlib.sha256(block).hexdigest()})
            require(fingerprint(os.fstat(stream.fileno())) == fingerprint(source_before) and
                    fingerprint(regular_file(source)) == fingerprint(source_before),
                    "Source changed during split")
        manifest = {"format_version": 1, "target": "outputs/full/episodes.csv.gz",
                    "part_bytes": part_bytes, "source_bytes": source_before.st_size,
                    "source_sha256": full_hash.hexdigest(), "parts": parts}
        require(sum(item["bytes"] for item in parts) == source_before.st_size,
                "Source byte count changed during split")
        with (staged / MANIFEST_NAME).open("x") as output:
            json.dump(manifest, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        store.mkdir()  # Exclusive claim; the manifest is installed last.
        store_created = True
        for item in parts:
            os.link(staged / item["file"], store / item["file"])
        os.link(staged / MANIFEST_NAME, store / MANIFEST_NAME)
        return {"status": "split", "store": str(store), "source_bytes": source_before.st_size,
                "source_sha256": full_hash.hexdigest(), "part_count": len(parts),
                "manifest_sha256": digest_file(store / MANIFEST_NAME)[1]}
    except Exception:
        if store_created:
            shutil.rmtree(store)
        raise
    finally:
        shutil.rmtree(staged)


def read_manifest(store: Path) -> dict:
    require(not store.is_symlink() and store.is_dir(), "Transport store is missing or a symlink")
    manifest_path = store / MANIFEST_NAME
    manifest = read_json_file(manifest_path)
    require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS, "Malformed transport manifest")
    require(is_int(manifest["format_version"]) and manifest["format_version"] == 1 and
            manifest["target"] == "outputs/full/episodes.csv.gz",
            "Unsupported transport manifest")
    size, limit, parts = manifest["source_bytes"], manifest["part_bytes"], manifest["parts"]
    require(is_int(size) and size > 0 and is_int(limit) and 0 < limit <= MAX_PART_BYTES and
            is_sha(manifest["source_sha256"]) and isinstance(parts, list) and parts,
            "Malformed transport sizes or digest")
    require(len(parts) == (size + limit - 1) // limit, "Wrong transport part count")
    for index, item in enumerate(parts):
        require(isinstance(item, dict) and set(item) == PART_KEYS and
                is_int(item["index"]) and item["index"] == index,
                "Unexpected part index, order, or duplicate")
        require(item["file"] == part_name(index), "Unexpected part filename or path escape")
        expected_bytes = min(limit, size - index * limit)
        require(is_int(item["bytes"]) and item["bytes"] == expected_bytes and
                is_sha(item["sha256"]), "Malformed part size or digest")
    expected = {MANIFEST_NAME, *(item["file"] for item in parts)}
    require({path.name for path in store.iterdir()} == expected, "Unexpected or missing transport file")
    for item in parts:
        part = store / item["file"]
        require(regular_file(part).st_size == item["bytes"], f"Missing or nonregular part: {part}")
    return manifest


def stream_parts(store: Path, manifest: dict, output) -> None:
    full_hash = hashlib.sha256()
    total = 0
    for item in manifest["parts"]:
        path = store / item["file"]
        before = regular_file(path)
        part_hash = hashlib.sha256()
        with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
            require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before), f"Part changed before reading: {path}")
            while block := stream.read(1024 * 1024):
                part_hash.update(block)
                full_hash.update(block)
                total += len(block)
                if output is not None:
                    output.write(block)
            require(fingerprint(os.fstat(stream.fileno())) == fingerprint(before) and
                    fingerprint(regular_file(path)) == fingerprint(before),
                    f"Part changed while reading: {path}")
        require(part_hash.hexdigest() == item["sha256"], f"Part digest mismatch: {path}")
    require(total == manifest["source_bytes"] and full_hash.hexdigest() == manifest["source_sha256"],
            "Reconstructed stream size or digest mismatch")


def reconstruct(store: Path, destination: Path) -> dict:
    manifest = read_manifest(store)
    if destination.exists() or destination.is_symlink():
        size, sha = digest_file(destination)
        require(size == manifest["source_bytes"] and sha == manifest["source_sha256"],
                "Destination differs from authenticated original; refusing overwrite")
        stream_parts(store, manifest, None)
        return {"status": "already_present", "destination": str(destination),
                "source_bytes": size, "source_sha256": sha, "part_count": len(manifest["parts"])}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.stage-",
                                         dir=destination.parent, delete=False) as output:
            staged = Path(output.name)
            stream_parts(store, manifest, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(staged, destination)  # Atomic and exclusive, unlike os.replace.
        except FileExistsError:
            size, sha = digest_file(destination)
            require(size == manifest["source_bytes"] and sha == manifest["source_sha256"],
                    "Destination appeared with different bytes; refusing overwrite")
            status = "already_present"
        else:
            status = "reconstructed"
        return {"status": status, "destination": str(destination),
                "source_bytes": manifest["source_bytes"], "source_sha256": manifest["source_sha256"],
                "part_count": len(manifest["parts"])}
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    splitter = commands.add_parser("split", help="Publish exact gzip bytes as <=32 MiB parts")
    splitter.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    splitter.add_argument("--store", type=Path, default=DEFAULT_STORE)
    splitter.add_argument("--part-bytes", type=int, default=MAX_PART_BYTES)
    restorer = commands.add_parser("reconstruct", help="Verify parts and restore the exact gzip bytes")
    restorer.add_argument("--store", type=Path, default=DEFAULT_STORE)
    restorer.add_argument("--destination", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    try:
        result = (split(args.source, args.store, args.part_bytes) if args.command == "split"
                  else reconstruct(args.store, args.destination))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"data_transport: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
