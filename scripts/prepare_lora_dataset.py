#!/usr/bin/env python3
"""Materialize verified character reference images for distributed CPU training (v2).

The coordinator supplies a committed training request. Each worker can use
the same request+source snapshot to materialize an identical source dataset.
Also supports legacy direct character-ID preparation for local use.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from library import ROOT, read_pack, pack_path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character-id")
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.request_file) == bool(args.character_id):
        parser.error("Provide exactly one of --character-id or --request-file")

    if args.request_file:
        data = json.loads(args.request_file.read_text(encoding="utf-8"))
        if data.get("contract_version") != "v1":
            parser.error("Unsupported distributed CPU training request")
        records = data["dataset"]["images"]
        character_id = data["dataset"]["character_id"]
        revision = data["dataset"]["character_revision"]
        dataset_sha = data["dataset_sha256"]
        expected_commit = data["source_commit"]
        # File paths originate in the canonical Phase 1 library.
        images = [
            (ROOT / r["repo_path"], r["sha256"], r["role"])
            for r in records
        ]
    else:
        character = read_pack("character", args.character_id)
        base = pack_path("character", character["id"]).resolve()
        images = [
            (base / r["path"], r["sha256"], r["role"])
            for r in character.get("references", [])
        ]
        character_id = character["id"]
        revision = character["revision"]
        dataset_sha = None
        expected_commit = None

    unique = []
    seen = set()
    package = pack_path("character", character_id).resolve()
    for file, expected_hash, role in images:
        path = file.resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError(f"Source reference missing or outside asset pack: {file}")
        payload = path.read_bytes()
        if sha(payload) != expected_hash:
            raise ValueError(f"Source image checksum mismatch: {file}")
        if expected_hash in seen:
            continue
        seen.add(expected_hash)
        unique.append((path, expected_hash, role))
    if len(unique) < 3:
        raise ValueError(
            f"At least three distinct verified reference images required; got {len(unique)}"
        )
    args.destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, (source, fingerprint, role) in enumerate(unique, 1):
        suffix = source.suffix.lower()
        if suffix not in (".png", ".jpg", ".gif", ".webp"):
            raise ValueError(f"Unsupported reference type: {source}")
        dest = args.destination / f"image-{i:03d}{suffix}"
        shutil.copyfile(source, dest)
        outputs.append({"file": dest.name, "sha256": fingerprint, "role": role})
    provenance = {
        "version": "v2", "character_id": character_id,
        "character_revision": revision, "image_count": len(outputs),
        "dataset_sha256": dataset_sha, "source_commit": expected_commit,
        "images": outputs,
    }
    manifest = args.destination.parent / "training-manifest-v2.json"
    manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(outputs)} SHA-verified character images for CPU trainer; "
          f"dataset_sha256={dataset_sha or 'direct-library'}")


if __name__ == "__main__":
    main()
