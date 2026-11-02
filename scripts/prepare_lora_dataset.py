#!/usr/bin/env python3
"""Prepare a verified character reference set for upstream SDXL LoRA training (v1)."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from library import ROOT, read_pack, pack_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    data = read_pack("character", args.character_id)
    base = pack_path("character", data["id"]).resolve()
    images = []
    seen = set()
    for item in data["references"]:
        path = (base / item["path"]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError("Unsafe or missing character reference")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != item["sha256"]:
            raise ValueError(f"Character reference changed: {item['path']}")
        if sha in seen:
            continue
        seen.add(sha)
        images.append((path, sha, item["role"]))
    if len(images) < 3:
        raise ValueError(
            f"LoRA training requires at least 3 distinct saved character images; "
            f"{args.character_id} has {len(images)}"
        )
    args.destination.mkdir(parents=True, exist_ok=True)
    items = []
    for n, (path, sha, role) in enumerate(images, 1):
        suffix = path.suffix.lower()
        if suffix not in (".png", ".jpg", ".gif", ".webp"):
            raise ValueError("Unsupported character reference format")
        dest = args.destination / f"image-{n:03d}{suffix}"
        shutil.copyfile(path, dest)
        items.append({"role": role, "sha256": sha, "file": dest.name})
    manifest = {
        "version": "v1",
        "character_id": data["id"],
        "character_revision": data["revision"],
        "image_count": len(items),
        "images": items,
    }
    (args.destination.parent / "training-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(items)} verified character references for GPU training")


if __name__ == "__main__":
    main()
