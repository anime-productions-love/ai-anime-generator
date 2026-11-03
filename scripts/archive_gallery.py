#!/usr/bin/env python3
"""Archive complete anime workflow outputs in PRIVATE gallery GitHub Releases (v3).

The archive is not an expiring Actions artifact. Each source workflow run gets
its own release tag and a ZIP containing every file in the designated outputs.
Nothing is deleted automatically. Verify remote release assets before allowing
the production chain to advance.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

GALLERY = "anime-productions-love/gallery"


def gh(args, capture=True):
    env = os.environ.copy()
    token = env.get("VIDEO_REPO_TOKEN") or env.get("GALLERY_REPO_TOKEN")
    if not token:
        raise RuntimeError(
            "Archive requires VIDEO_REPO_TOKEN (fine-grained PAT with "
            "Contents:write on private anime-productions-love/gallery) "
            "in the ai-anime-generator Actions secrets."
        )
    env["GH_TOKEN"] = token
    return subprocess.run(
        ["gh", *args], check=True, capture_output=capture,
        text=True, env=env,
    ).stdout.strip() if capture else subprocess.run(
        ["gh", *args], check=True, env=env,
    )


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kind", required=True, choices=(
        "wan", "movie", "enhanced", "comic", "storyboard", "story-director", "kaggle"
    ))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("work/gallery-archive"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{4,24}", args.run_id):
        parser.error("run-id must be numeric GitHub Actions run ID")
    base = args.source.resolve()
    if not base.is_dir():
        parser.error("Source output folder not found")
    paths = sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink())
    if not paths:
        parser.error("Nothing to archive")
    for file in paths:
        if not file.resolve().is_relative_to(base):
            parser.error("Unsafe source path")
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"anime-archive-{args.run_id}"
    bundle = args.out / f"{args.kind}-{args.run_id}-complete.zip"
    inventory = {
        "schema_version": "v1",
        "origin_repository": "anime-productions-love/ai-anime-generator",
        "origin_workflow_run_id": args.run_id,
        "kind": args.kind,
        "source_directory": str(args.source),
        "retention": "GitHub Release assets: no automatic Actions artifact expiration",
        "files": [
            {
                "path": p.relative_to(base).as_posix(),
                "size": p.stat().st_size,
                "sha256": checksum(p),
            } for p in paths
        ],
    }
    # Deflate efficiently compressible metadata, without modifying source files.
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for path in paths:
            z.write(path, path.relative_to(base).as_posix())
        z.writestr("gallery-archive-manifest-v1.json",
                   json.dumps(inventory, indent=2)+"\n")
    # Only create this release when it does not already exist. Retrying a
    # partial archival operation cannot overwrite a previously uploaded asset.
    try:
        gh(["release", "view", tag, "--repo", GALLERY, "--json", "tagName"])
    except subprocess.CalledProcessError:
        gh([
            "release", "create", tag, "--repo", GALLERY,
            "--title", f"Anime production run {args.run_id} ({args.kind})",
            "--notes", (
                "Permanent private gallery copy of anime generation outputs "
                f"from workflow run {args.run_id}; no automatic deletion."
            ),
            "--latest=false",
        ])
    release = json.loads(gh([
        "release", "view", tag, "--repo", GALLERY,
        "--json", "assets,url,tagName",
    ]))
    remote_names = {a["name"]: a for a in release.get("assets", [])}
    if bundle.name not in remote_names:
        gh(["release", "upload", tag, str(bundle), "--repo", GALLERY])
    release = json.loads(gh([
        "release", "view", tag, "--repo", GALLERY,
        "--json", "assets,url,tagName",
    ]))
    asset = next((a for a in release["assets"] if a["name"] == bundle.name), None)
    if not asset or asset.get("size") != bundle.stat().st_size:
        raise RuntimeError("Private gallery ZIP asset size was not verified")
    receipt = {
        **{k: inventory[k] for k in (
            "origin_repository", "origin_workflow_run_id", "kind"
        )},
        "schema_version": "v1",
        "gallery_repository": GALLERY,
        "release_tag": tag,
        "release_url": release["url"],
        "asset_name": bundle.name,
        "zip_sha256": checksum(bundle),
        "size_bytes": bundle.stat().st_size,
        "file_count": len(paths),
        "archive_verified": True,
    }
    (args.out / "archive-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified permanent private-gallery release: {release['url']}")
    print(f"{len(paths)} files archived; ZIP size {bundle.stat().st_size} bytes")


if __name__ == "__main__":
    main()
