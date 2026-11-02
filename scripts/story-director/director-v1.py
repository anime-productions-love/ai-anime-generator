#!/usr/bin/env python3
"""Four-runner Anime Story Director: verified models and draft-only episode planning (v4).

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

CREATIVE = """You are a creative anime series director. Preserve the supplied story,
character arcs, emotional motivations, pacing, and artistic identity. Propose
a coherent episode, with as many scenes as the storytelling needs."""
PRODUCTION = """You are an anime episode production director. Build realistic,
connected scenes for continuous Wan image-to-video clips. A new scene may begin
with its own approved image and new framing, but successive clips WITHIN a
scene inherit the exact final frame of the preceding clip. Never conflate a
scene boundary with an internal clip boundary."""
EFFICIENT = """You are an anime episode planner who preserves narrative
quality while making shot duration and motion achievable on CPU-rendered Wan.
Estimate shot counts from actions and pacing, never from arbitrary quotas."""

OUTLINE_SYSTEM = """Return exactly ONE JSON OBJECT with:
title (string), logline (string), character_bible (object),
environment_bible (object), episode_continuity (array of strings),
scenes (array of next ordered scene objects), has_more (boolean).
Each scene has index (starting at the supplied next index), title,
narrative_goal, location, characters (array), starting_image_brief,
transition_from_previous, continuity_instructions, clip_count (positive
integer). Include at most BATCH_SIZE scenes in this response, continuing
the story naturally from the prior outline. NO hard upper bound on total
scenes or clips. Set has_more=true ONLY when there are actually more
story beats requiring additional scenes. Do not write every clip prompt yet.
The starting image for each scene must be selected/approved by a human.
Do not fabricate an existing image or send production instructions."""

SCENE_SYSTEM = """Return exactly ONE JSON OBJECT with:
scene_index (integer), clips (array of the next ordered clip objects).
Each clip includes index (starting at the given NEXT_CLIP), prompt (nonempty,
concrete character/action/camera instructions), narrative_goal (nonempty).
Produce exactly the requested BATCH_COUNT clips for THIS batch, preserving
the scene's fixed location, characters and established continuity. The first
clip of a scene uses its operator-approved initial image; every successor
inherits the preceding verified final frame, even across technical batches.
No new starting images or camera cuts within one continuous scene.
Never claim an image has already been approved. This is a draft only."""

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
    if not 10 <= len(story) <= MAX_STORY:
        raise ValueError("Story text must have 10–16000 characters")
    if not 5 <= args.frames <= 121 or not 1 <= args.fps <= 60:
        raise ValueError("Invalid shared Wan frames or FPS")
    if args.scenes < 0:
        raise ValueError("Optional preferred scene count cannot be negative")
    plan = {
        "version": "v2", "kind": "episode",
        "status": "DRAFT_NOT_APPROVED",
        "model": args.model,
        "model_role": info["role"],
        "preferred_scenes": args.scenes or None,
        "clip_count_policy": "story_director_decides_no_creative_limit",
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
                      "preferred_scenes": plan["preferred_scenes"],
                      "story_sha256": plan["story_sha256"]}), flush=True)

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
        # GGUF Q4 weights exceed the 14 GB advertised GitHub-runner SSD.
        # Use the pinned HTTPS-range FUSE reader rather than downgrade the
        # model to a substantially lower-quality quantization.
        write_json(dest / "remote-manifest-v1.json", {
            "version": "v1", "model": args.model,
            "repository": info["repository"], "revision": commit,
            "filename": filename, "expected_sha256": info["sha256"],
            "approx_bytes": info["approx_bytes"],
            "source_url": info["source"],
            "remote_url": f"https://huggingface.co/{info['repository']}/resolve/{commit}/{filename}",
        })
        print(json.dumps({
            "status": "REMOTE_HTTPS_RANGE_MOUNT_REQUIRED",
            "free_bytes": free, "required_bytes": required,
            "model": args.model, "revision": commit,
        }), flush=True)
        return
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


def director_request(model, system, prompt, temperature):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature, "max_tokens": 5500,
        "stream": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=2400) as resp:
        response = json.load(resp)
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model did not return an answer")
    return parse_object(content)


def positive_index(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_scene(scene, index):
    if not isinstance(scene, dict) or scene.get("index") != index:
        raise ValueError(f"Scene {index} is missing or out of order")
    for k in ("title", "narrative_goal", "location", "starting_image_brief",
              "transition_from_previous", "continuity_instructions"):
        if not isinstance(scene.get(k), str) or not scene[k].strip():
            raise ValueError(f"Scene {index} missing {k}")
    if not isinstance(scene.get("characters"), list):
        raise ValueError(f"Scene {index} lacks characters list")
    positive_index(scene.get("clip_count"), f"Scene {index} clip_count")


def validate_clip(clip, index):
    if not isinstance(clip, dict) or clip.get("index") != index:
        raise ValueError(f"Clip {index} is missing or out of order")
    for key in ("prompt", "narrative_goal"):
        if not isinstance(clip.get(key), str) or not clip[key].strip():
            raise ValueError(f"Clip {index} missing {key}")
        if len(clip[key]) > 5000:
            raise ValueError(f"Clip {index} {key} exceeds 5000 characters")


def infer(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if args.model != plan["model"] or plan.get("kind") != "episode":
        raise ValueError("Source episode/model identity mismatch")
    role = registry_entry(args.model)["role"]
    persona = {
        "creative-director": CREATIVE,
        "production-director": PRODUCTION,
        "efficient-production-director": EFFICIENT,
    }[role]
    stage = args.stage
    result = {
        "version": "v2", "kind": "episode-director-draft",
        "stage": stage, "status": "DRAFT_PARTIAL_CONTINUATION_AVAILABLE",
        "model": args.model,
        "model_provenance": json.loads(
            Path(args.provenance).read_text(encoding="utf-8")
        ),
        "prompt_source_sha256": plan["story_sha256"],
        "scene_index": args.scene_index if stage == "scene" else None,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.resume_draft:
        prior = json.loads(Path(args.resume_draft).read_text(encoding="utf-8"))
        if (prior.get("version") != "v2"
            or prior.get("stage") != stage
            or prior.get("model") != args.model
            or prior.get("prompt_source_sha256") != plan["story_sha256"]
            or prior.get("scene_index") != result["scene_index"]):
            raise ValueError("Resume draft does not match source/story/stage")
        result.update({
            key: prior[key] for key in ("episode", "scene", "last_error")
            if key in prior
        })

    deadline = time.monotonic() + 205 * 60  # Runner time budget, NOT story length.
    rounds = 0
    while True:
        if time.monotonic() >= deadline:
            result["status"] = "DRAFT_PARTIAL_CONTINUATION_AVAILABLE"
            break
        if stage == "episode":
            episode = result.get("episode") or {
                "title": "", "logline": "", "character_bible": {},
                "environment_bible": {}, "episode_continuity": [],
                "scenes": [], "has_more": True,
            }
            if not episode.get("has_more", True):
                result["status"] = "EPISODE_OUTLINE_COMPLETE_NOT_APPROVED"
                break
            n = len(episode["scenes"])
            preferred = plan.get("preferred_scenes")
            if preferred and n >= preferred:
                result["status"] = "EPISODE_OUTLINE_COMPLETE_NOT_APPROVED"
                break
            # Eight scenes is an inference request/output budget, NOT a
            # maximum number of scenes in the episode.
            batch = min(8, preferred - n) if preferred else 8
            past = [{"index": item["index"], "title": item["title"],
                     "narrative_goal": item["narrative_goal"]}
                    for item in episode["scenes"]]
            prompt = (
                f"Storyboard the episode in chronological order. Next scene "
                f"index: {n+1}; BATCH_SIZE={batch}. "
                f"Preferred total scene count (0 means author decides): "
                f"{preferred or 0}. "
                f"Story: {plan['story_text']}\n"
                f"Existing episode scenes: {json.dumps(past, ensure_ascii=False)}\n"
                f"Characters: {plan['character_ids']}; style: "
                f"{plan['requested_style']}; "
                f"clip duration {plan['clip_seconds']} seconds. "
                f"Do not create a false ending merely because a batch ends."
            )
            try:
                part = director_request(args.model, persona + OUTLINE_SYSTEM,
                                        prompt, 0.55)
                new = part.get("scenes")
                if not isinstance(new, list) or not 1 <= len(new) <= batch:
                    raise ValueError("Outline batch missing scene items")
                for j, item in enumerate(new, n + 1):
                    validate_scene(item, j)
                if type(part.get("has_more")) is not bool:
                    raise ValueError("Outline batch has_more must be boolean")
                for key in ("title", "logline"):
                    if n == 0 and not isinstance(part.get(key), str):
                        raise ValueError(f"Episode lacks {key}")
                for key in ("character_bible", "environment_bible"):
                    if n == 0 and not isinstance(part.get(key), dict):
                        raise ValueError(f"Episode lacks {key}")
                if n == 0:
                    episode.update({
                        key: part[key] for key in
                        ("title", "logline", "character_bible",
                         "environment_bible")
                    })
                    episode["episode_continuity"] = part.get(
                        "episode_continuity", []
                    )
                episode["scenes"].extend(new)
                episode["has_more"] = part["has_more"]
                result["episode"] = episode
                result["status"] = (
                    "DRAFT_PARTIAL_CONTINUATION_AVAILABLE"
                    if part["has_more"] else "EPISODE_OUTLINE_COMPLETE_NOT_APPROVED"
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result["last_error"] = str(exc)
                write_json(args.output, result)
                raise
        else:
            if not args.outline:
                raise ValueError("Scene detail requires a prior episode outline")
            source = json.loads(Path(args.outline).read_text(encoding="utf-8"))
            if (source.get("stage") != "episode"
                or source.get("prompt_source_sha256") != plan["story_sha256"]
                or source.get("model") != args.model
                or not source.get("episode")):
                raise ValueError("Scene detail uses incorrect episode outline")
            scenes = source["episode"].get("scenes", [])
            scene = next((item for item in scenes
                          if item.get("index") == args.scene_index), None)
            if scene is None:
                raise ValueError("Requested scene is not in episode outline")
            total = positive_index(scene.get("clip_count"), "scene clip count")
            output_scene = result.get("scene") or {
                "index": args.scene_index,
                "title": scene["title"], "clip_count": total, "clips": [],
                "starting_image_brief": scene["starting_image_brief"],
                "initial_image_status": "OPERATOR_SELECTION_REQUIRED",
            }
            if output_scene["clip_count"] != total:
                raise ValueError("Resumed scene changed clip count")
            n = len(output_scene["clips"])
            if n >= total:
                result["status"] = "SCENE_CLIPS_COMPLETE_NOT_APPROVED"
                break
            # Ten prompts is a per-call budget, not a clip limit.
            batch = min(10, total - n)
            prompt = (
                f"Create next BATCH_COUNT={batch} clips in scene "
                f"{args.scene_index}. NEXT_CLIP={n+1}. "
                f"Episode title: {source['episode']['title']}. "
                f"Scene: {json.dumps(scene, ensure_ascii=False)}\n"
                f"Episode continuity: {json.dumps(source['episode'].get('episode_continuity', []), ensure_ascii=False)}\n"
                f"Previous clips in this scene: "
                f"{json.dumps([{'index': x['index'], 'prompt': x['prompt']} for x in output_scene['clips'][-10:]], ensure_ascii=False)}\n"
                f"Frames={plan['shared_frames']}, FPS={plan['shared_fps']}, "
                f"duration={plan['clip_seconds']} seconds per clip. "
                f"No arbitrary scene or clip count ceiling."
            )
            try:
                part = director_request(args.model, persona + SCENE_SYSTEM,
                                        prompt, 0.4)
                new = part.get("clips")
                if part.get("scene_index") != args.scene_index:
                    raise ValueError("Scene index response mismatch")
                if not isinstance(new, list) or len(new) != batch:
                    raise ValueError("Scene batch has incorrect clip count")
                for j, item in enumerate(new, n + 1):
                    validate_clip(item, j)
                output_scene["clips"].extend(new)
                result["scene"] = output_scene
                result["status"] = (
                    "SCENE_CLIPS_COMPLETE_NOT_APPROVED"
                    if len(output_scene["clips"]) == total
                    else "DRAFT_PARTIAL_CONTINUATION_AVAILABLE"
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result["last_error"] = str(exc)
                write_json(args.output, result)
                raise
        rounds += 1
        result["batches_completed_this_run"] = rounds
        # Persist after every actual model batch; the next workflow run may
        # continue from the last accepted source-bound draft.
        write_json(args.output, result)
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "stage": stage,
        "model": args.model,
        "scenes_so_far": len(result.get("episode", {}).get("scenes", [])),
        "clips_so_far": len(result.get("scene", {}).get("clips", [])),
        "source_sha256": plan["story_sha256"],
        "batches": rounds,
    }), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--model", required=True)
    a.add_argument("--scenes", type=int, default=0,
                   help="Optional preferred scene count; 0 = AI decides")
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
    a.add_argument("--stage", choices=("episode", "scene"), default="episode")
    a.add_argument("--scene-index", type=int, default=1)
    a.add_argument("--outline", default="")
    a.add_argument("--resume-draft", default="")
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
