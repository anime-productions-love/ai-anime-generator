#!/usr/bin/env python3
"""Four-runner Anime Story Director: verified models and draft-only inference (v1).

No training, no production dispatch and no unapproved model-authored writes.
The workflow calls prepare/download/infer in order; model weights are sourced
from immutable Hugging Face revisions and checked against registered digests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "scripts/story-director/models-v1.json"
MAX_STORY = 16_000

CREATIVE = """You are a specialized anime screenwriter and creative director.
Faithfully adapt the supplied story; preserve character motivations, dialogue
intention, pacing, emotional beats, and continuity. Propose vivid but physically
achievable short continuous clips. Output ONE valid JSON object only with:
summary (string), character_bible (object), environment_bible (object),
continuity_notes (array of strings), clips (array of exactly N objects with
index (integer 1..N), prompt (string), narrative_goal (string)).
Each clip continues from the LAST FRAME of the preceding clip. Do not
invent new starting images, camera cuts, technical settings or characters that
contradict the story. All clips share identical technical generation settings.
You only propose a draft; a human must approve before rendering."""

PRODUCTION = """You are an anime production director. Optimize a narrative
for fixed-configuration Wan2.2 continuous image-to-video generation. Preserve
original story and character identity. Every successor begins from the
previous video final frame. No new image sources or changes in technical
settings. Output ONE JSON object with keys: summary (string),
character_bible (object), environment_bible (object),
continuity_notes (array of strings), clips (array of exactly N objects each
with index 1..N, prompt, narrative_goal). Fit motion to clip duration.
Do not claim you reviewed unseen video. Draft requires human approval."""

EFFICIENT = """You are a resource-conscious anime production planner.
Generate precise feasible motion/camera instructions for a fixed technical
Wan video configuration and preceding-frame-only continuation. Budget action
complexity to exact clip duration. Output ONE JSON object with keys:
summary, character_bible, environment_bible, continuity_notes, clips.
clips must contain exactly N objects each with index, prompt, narrative_goal.
This draft requires human approval; do not dispatch rendering."""


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_entry(model):
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if data.get("version") != "v1" or model not in data["models"]:
        raise ValueError("Unknown or unregistered Story Director model")
    info = data["models"][model]
    if not re.fullmatch(r"[0-9a-f]{64}", info["sha256"]):
        raise ValueError("Model digest not pinned")
    return info


def prepare(args):
    info = registry_entry(args.model)
    if args.story_file:
        given = (ROOT / args.story_file).resolve()
        if not given.is_relative_to(ROOT) or not given.is_file():
            raise ValueError("story_file must point to an existing in-repo file")
        if "stories" not in given.relative_to(ROOT).parts:
            raise ValueError("story_file must reside within stories/")
        story = given.read_text(encoding="utf-8")
    else:
        story = os.environ.get("DIRECTOR_STORY_TEXT", "")
    story = story.strip()
    if not 10 <= len(story) <= 16_000:
        raise ValueError("Story text must have 10–16000 characters")
    if not 2 <= args.clips <= 33:
        raise ValueError("Proposed scene must have 2–33 clips")
    if not 5 <= args.frames <= 121 or not 1 <= args.fps <= 60:
        raise ValueError("Invalid shared frames or FPS")
    plan = {
        "version": "v1",
        "status": "DRAFT_NOT_APPROVED",
        "model": args.model,
        "model_role": info["role"],
        "clips_requested": args.clips,
        "shared_frames": args.frames,
        "shared_fps": args.fps,
        "clip_seconds": round(args.frames / args.fps, 2),
        "story_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
        "story_text": story,
        "requested_style": args.style[:2000],
        "character_ids": args.characters[:2000],
    }
    write_json(args.output, plan)
    print(json.dumps({"status": plan["status"], "model": args.model,
                      "clips": args.clips, "story_sha256": plan["story_sha256"]}),
          flush=True)


def download(args):
    info = registry_entry(args.model)
    request = urllib.request.Request(
        "https://huggingface.co/api/models/" + info["repository"],
        headers={"User-Agent": "anime-story-director-v1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        metadata = json.load(response)
    commit = metadata.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hugging Face did not return an immutable commit SHA")
    filename = info["file"]
    matches = [x for x in metadata.get("siblings", [])
               if x.get("rfilename") == filename]
    if len(matches) != 1:
        raise ValueError("Pinned GGUF filename missing from upstream repository")
    dest = Path(args.destination)
    dest.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(dest)
    free = fs.f_frsize * fs.f_bavail
    required = int(info["approx_bytes"]) + 2_500_000_000
    if free < required:
        raise RuntimeError(
            f"Controller SSD preflight failed: {free/1e9:.2f} GB free, "
            f"needs at least {required/1e9:.2f} GB. Four-runner RPC "
            "shares RAM, not download storage; no silent low-quality fallback."
        )
    outfile = dest / filename
    url = f"https://huggingface.co/{info['repository']}/resolve/{commit}/{filename}"
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "5", "--retry-all-errors",
        "--retry-delay", "5", "--connect-timeout", "30",
        "--output", str(outfile), url,
    ], check=True)
    if outfile.stat().st_size < int(info["approx_bytes"]) * 0.9:
        raise RuntimeError("Incomplete model download")
    actual = digest(outfile)
    if actual != info["sha256"]:
        outfile.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model failed registered SHA-256 integrity check")
    receipt = {
        "version": "v1", "model": args.model,
        "repository": info["repository"], "revision": commit,
        "filename": filename, "file_sha256": actual,
        "file_bytes": outfile.stat().st_size,
        "source_url": info["source"],
        "vision_projector_included": False,
        "status": "MODEL_BYTES_VERIFIED_NOT_INFERRED",
    }
    write_json(dest / "model-provenance-v1.json", receipt)
    print(json.dumps({"model_file": str(outfile.resolve()),
                      "sha256": actual, "revision": commit}), flush=True)


def parse_object(content):
    raw = content.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + fence + r"(?:json)?\s*", "", raw, count=1)
        raw = re.sub(r"\s*" + fence + r"$", "", raw, count=1)
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("Model did not return a JSON object")
    return obj


def infer(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if args.model != plan["model"]:
        raise ValueError("Inferred model differs from source plan")
    info = registry_entry(plan["model"])
    system = {
        "creative-director": CREATIVE,
        "production-director": PRODUCTION,
        "efficient-production-director": EFFICIENT,
    }[info["role"]]
    prompt = (
        f"Create exactly N={plan['clips_requested']} sequential anime clips.\n"
        f"Each clip: {plan['shared_frames']} frames at {plan['shared_fps']} FPS, "
        f"approximately {plan['clip_seconds']} seconds.\n"
        f"Character IDs: {plan['character_ids']}\n"
        f"Style: {plan['requested_style']}\n"
        f"Source story:\n{plan['story_text']}"
    )
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.5 if info["role"] == "creative-director" else 0.25,
        "max_tokens": min(6500, max(1500, 250 * plan["clips_requested"])),
        "stream": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        response = json.load(resp)
    content = response["choices"][0]["message"]["content"]
    result = {
        "version": "v1", "status": "DRAFT_NEEDS_HUMAN_REVIEW",
        "model": args.model,
        "model_provenance": json.loads(Path(args.provenance).read_text()),
        "prompt_source_sha256": plan["story_sha256"],
        "source_clips_requested": plan["clips_requested"],
        "raw_model_response": content,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        parsed = parse_object(content)
        clips = parsed.get("clips")
        if (not isinstance(clips, list) or len(clips) != plan["clips_requested"]
            or any(not isinstance(clip, dict) or clip.get("index") != n
                   or not isinstance(clip.get("prompt"), str)
                   or not 1 <= len(clip["prompt"]) <= 5000
                   for n, clip in enumerate(clips, 1))):
            raise ValueError("Wrong clip count, numbering, or empty prompt")
        result["draft"] = parsed
        result["status"] = "DRAFT_STRUCTURALLY_COMPLETE_NOT_APPROVED"
    except (ValueError, TypeError) as exc:
        result["format_error"] = str(exc)
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "model": args.model,
        "clips_requested": plan["clips_requested"],
        "source_sha256": plan["story_sha256"],
    }), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--model", required=True)
    a.add_argument("--clips", type=int, required=True)
    a.add_argument("--frames", type=int, default=42)
    a.add_argument("--fps", type=int, default=4)
    a.add_argument("--story-file", default="")
    a.add_argument("--characters", default="")
    a.add_argument("--style", default="")
    a.add_argument("--output", required=True)
    a = sub.add_parser("download")
    a.add_argument("--model", required=True)
    a.add_argument("--destination", required=True)
    a = sub.add_parser("infer")
    a.add_argument("--model", required=True)
    a.add_argument("--plan", required=True)
    a.add_argument("--provenance", required=True)
    a.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        {"prepare": prepare, "download": download, "infer": infer}[args.mode](args)
    except Exception as exc:
        p.exit(1, f"Story Director {args.mode} stopped: {exc}\n")


if __name__ == "__main__":
    main()
