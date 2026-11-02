#!/usr/bin/env python3
"""Export or dispatch one canonical storyboard image to the Wan video workflow (v4).

Dispatch is opt-in and uses the built-in GITHUB_TOKEN with Actions:write
on this same unified anime repository. Acceptance does NOT mean video generated.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from library import ROOT, validate_id
from story import sequence_dir

TARGET = "anime-productions-love/ai-anime-generator"
SOURCE = "anime-productions-love/ai-anime-generator"
ENGINES = {
    "wan": "generate-wan22-ti2v-turbo.yml",
}


def git_revision():
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Cannot determine immutable source Git commit")
    return revision


def canonical(story_id, sequence_id, shot_id):
    seqdir = sequence_dir(story_id, sequence_id)
    seqfile = seqdir / "manifest.json"
    if not seqfile.is_file():
        raise ValueError("Sequence does not exist")
    seq = json.loads(seqfile.read_text(encoding="utf-8"))
    if seq.get("schema_version") != "v1" or shot_id not in seq.get("shots", []):
        raise ValueError("Shot not present in published sequence")
    file = seqdir / "shots" / shot_id / "manifest.json"
    if not file.is_file():
        raise ValueError("Shot metadata missing")
    shot = json.loads(file.read_text(encoding="utf-8"))
    image_path = shot["selected_image"]
    if not isinstance(image_path, str) or not image_path.startswith("stories/"):
        raise ValueError("Unsupported image path")
    path = (ROOT / image_path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Canonical image missing or outside repository")
    if hashlib.sha256(path.read_bytes()).hexdigest() != shot["selected_sha256"]:
        raise ValueError("Selected image digest mismatch")
    if shot.get("shot_id") != shot_id:
        raise ValueError("Shot manifest identity mismatch")
    return shot, path


def dispatch(workflow, payload, token):
    if not token:
        raise ValueError(
            "Dispatch requires GITHUB_TOKEN with Actions:write on this repository"
        )
    api = f"https://api.github.com/repos/{TARGET}/actions/workflows/{workflow}/dispatches"
    request = urllib.request.Request(
        api,
        data=json.dumps({"ref": "main", "inputs": payload}).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "AnimeStoryboardBridge/1.0",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 204:
                raise RuntimeError(f"Unexpected GitHub workflow dispatch response HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        message = exc.read(1600).decode("utf-8", "replace")
        raise RuntimeError(
            f"GitHub video workflow dispatch rejected (HTTP {exc.code}): {message}"
        ) from exc
    print("GitHub accepted a video workflow_dispatch request. "
          "This does not confirm the target run started or generated a video.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--shot-id", required=True)
    parser.add_argument("--engine", choices=tuple(ENGINES), default="wan")
    parser.add_argument("--action", choices=("export", "dispatch"), default="export")
    parser.add_argument("--motion-prompt", default="")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--handoff-id", required=True)
    args = parser.parse_args()
    for value in (args.story_id, args.sequence_id):
        validate_id(value)
    if not re.fullmatch(r"[0-9]{4,}", args.shot_id):
        parser.error("--shot-id must be the four-digit published shot ID")
    if not args.handoff_id.isdigit():
        parser.error("--handoff-id must be numeric GitHub run ID")
    if not 1 <= args.fps <= 60 or not 5 <= args.frames <= 121:
        parser.error("fps must be 1-60, frames must be 5-121")
    if args.width % 16 or args.height % 16 or not 256 <= args.width <= 1536 or not 256 <= args.height <= 1536:
        parser.error("Video width and height must be multiples of 16 in 256-1536")
    shot, selected = canonical(args.story_id, args.sequence_id, args.shot_id)
    commit = git_revision()
    repo_path = selected.relative_to(ROOT).as_posix()
    image_url = (f"https://raw.githubusercontent.com/{SOURCE}/{commit}/"
                 f"{urllib.parse.quote(repo_path, safe='/')}")
    motion = args.motion_prompt.strip() or (
        "The pictured anime character(s) move naturally with subtle expressions, "
        "hair and clothing shift gently, smooth cinematic camera movement. "
        + shot.get("prompt", "")
    )
    if len(motion) > 5000:
        parser.error("motion prompt exceeds 5000 characters")
    inputs = {
        "source_mode": "image", "image": image_url,
        "prompt": motion, "negative_prompt": shot.get("composed_negative_prompt", "")[:3000],
        "fit_mode": "contain-blur", "frames": str(args.frames),
        "fps": str(args.fps), "width": str(args.width),
        "height": str(args.height), "seed": str(shot["seed"]),
    }
    receipt_path = (
        sequence_dir(args.story_id, args.sequence_id) / "video-handoffs"
        / ("handoff-" + args.handoff_id + ".json")
    )
    if receipt_path.exists():
        raise ValueError("Handoff already exists; refusing to overwrite")
    if args.action == "dispatch":
        dispatch(ENGINES[args.engine], inputs, os.environ.get("GITHUB_TOKEN", ""))
    receipt = {
        "schema_version": "v1",
        "source_repo": SOURCE,
        "source_commit": commit,
        "story_id": args.story_id,
        "sequence_id": args.sequence_id,
        "shot_id": args.shot_id,
        "selected_image": repo_path,
        "selected_sha256": shot["selected_sha256"],
        "direct_image_url": image_url,
        "target_repo": TARGET,
        "target_workflow": ENGINES[args.engine],
        "target_ref": "main",
        "workflow_inputs": inputs,
        "mode": args.action,
        "dispatch_accepted": args.action == "dispatch",
        "target_run_id": None,
        "video_created": False,
        "target_actions_url": f"https://github.com/{TARGET}/actions/workflows/{ENGINES[args.engine]}",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Created video handoff: {receipt_path.relative_to(ROOT)}")
    if args.action == "export":
        print("Export only. No video job dispatched.")
    else:
        print("Dispatch accepted. Check the target repository Actions page for actual run status.")


if __name__=="__main__":
    main()
