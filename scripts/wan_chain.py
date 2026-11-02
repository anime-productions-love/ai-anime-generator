#!/usr/bin/env python3
"""Chain multiple Wan video scenes across separate GitHub Actions runs (v5).

Each run is a single <=350-minute Wan job; ONLY the initial clip uses
an external image, and each later clip uses the prior clip's last_frame.png. A form-generated JSON plan supplies prompts and one shared configuration.
The built-in GITHUB_TOKEN dispatches the next workflow_dispatch on success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WAN_WORKFLOW = "generate-wan22-ti2v-turbo.yml"
MOVIE_WORKFLOW = "assemble-anime-movie.yml"
ARTIFACT_PATTERN = re.compile(r"wan22-turbo-v[0-9]+-([0-9]+)")
DEFAULT_NEGATIVE = "frozen frame, flicker, jitter, warped face, deformed anatomy, text, watermark"
DEFAULTS = {
    "negative_prompt": DEFAULT_NEGATIVE,
    "fit_mode": "contain-blur",
    "frames": 42, "fps": 6, "width": 480, "height": 832, "seed": 42,
}


def fail(message):
    raise ValueError(message)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, capture=False, stdin=None):
    process = subprocess.run(
        [str(x) for x in command],
        cwd=ROOT, input=stdin, text=True, capture_output=capture,
        check=True,
    )
    return process.stdout if capture else None


def gh_api(path):
    return json.loads(run(["gh", "api", path], capture=True))


def validate_plan_path(value):
    if not re.fullmatch(r"chains/[A-Za-z0-9_-]{1,80}\.json", value or ""):
        fail("chain_plan must be chains/<name>.json (no nested paths)")
    return ROOT / value


def positive_int(value, label, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        fail(f"{label} must be an integer between {lo} and {hi}")
    return value


def parse_config(path):
    """Validate a continuous scene: one image, one immutable settings set.

    No per-video source/size/FPS/seed/negative-prompt overrides. The only
    property that changes between segments is their written motion prompt.
    """
    data = json_read(path)
    if not isinstance(data, dict) or data.get("version") != "v2":
        fail("Sequence plan must be a JSON object with version='v2'")
    if set(data) - {"version", "title", "initial_image", "settings", "prompts", "assembly", "postprocess"}:
        fail("Unknown sequence plan property")
    image = data.get("initial_image")
    if not isinstance(image, str) or not image.startswith("https://"):
        fail("initial_image must be an HTTPS image/video URL (private gallery supported)")
    settings = data.get("settings")
    if not isinstance(settings, dict) or set(settings) != set(DEFAULTS):
        fail("settings must contain exactly negative_prompt, fit_mode, frames, fps, "
             "width, height and seed; no per-segment overrides")
    if not isinstance(settings["negative_prompt"], str) or len(settings["negative_prompt"]) > 3000:
        fail("Negative prompt must be text of <= 3000 characters")
    if settings["fit_mode"] not in ("contain", "contain-blur", "crop"):
        fail("fit_mode must be contain, contain-blur or crop")
    positive_int(settings["frames"], "frames", 5, 121)
    positive_int(settings["fps"], "fps", 1, 60)
    positive_int(settings["width"], "width", 256, 1536)
    positive_int(settings["height"], "height", 256, 1536)
    if settings["width"] % 16 or settings["height"] % 16:
        fail("width and height must be multiples of 16")
    positive_int(settings["seed"], "seed", 0, 2147483647)
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not 2 <= len(prompts) <= 33:
        fail("Continuous sequence requires 2–33 written video prompts")
    for n, prompt in enumerate(prompts, 1):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 5000:
            fail(f"Video prompt {n} must have 1–5000 characters")
    resolved = [{**settings, "prompt": prompt.strip()} for prompt in prompts]
    assembly = data.get("assembly")
    if not isinstance(assembly, dict):
        fail("assembly configuration must be an object")
    if set(assembly) - {"enabled", "width", "height", "fps", "transition",
                         "fade_seconds", "quality_crf"}:
        fail("Unsupported assembly option")
    assembly = {
        "enabled": True, "width": settings["width"],
        "height": settings["height"], "fps": settings["fps"],
        "transition": "cut", "fade_seconds": "0", "quality_crf": 20,
        **assembly,
    }
    if assembly["enabled"] is not True:
        fail("Continuous sequences always assemble the final movie")
    if (assembly["width"] != settings["width"]
            or assembly["height"] != settings["height"]
            or assembly["fps"] != settings["fps"]):
        fail("Movie assembly must preserve the same resolution and FPS as every clip")
    positive_int(assembly["quality_crf"], "assembly.quality_crf", 15, 35)
    if assembly["transition"] not in ("cut", "motion-seam"):
        fail("Continuous sequence assembly supports clean cut or motion-seam")
    if str(assembly["fade_seconds"]) not in ("0", "0.0"):
        fail("Continuous sequence cuts require fade_seconds=0")
    assembly["fade_seconds"] = "0"
    postprocess = data.get("postprocess", {
        "method": "motion", "target_fps": 24, "speed": 1.5,
    })
    if not isinstance(postprocess, dict) or set(postprocess) != {
        "method", "target_fps", "speed"
    }:
        fail("postprocess must set method, target_fps and speed")
    if postprocess["method"] not in ("motion", "rife"):
        fail("postprocess method must be motion or rife")
    positive_int(postprocess["target_fps"], "postprocess.target_fps", 8, 60)
    if postprocess["speed"] not in (0.75, 1, 1.0, 1.25, 1.5, 2, 2.0, 3, 3.0, 4, 4.0):
        fail("postprocess speed is unsupported")
    return data, resolved, assembly


def prepare(args):
    source = validate_plan_path(args.plan_path)
    if not source.is_file() or source.stat().st_size > 128 * 1024:
        fail("Chain plan does not exist or exceeds 128 KiB")
    content_sha = sha_file(source)
    data, clips, assembly = parse_config(source)
    index = positive_int(args.index, "chain_step", 1, len(clips))
    run_id = positive_int(args.run_id, "run ID", 1, 10**24)
    if args.previous_run:
        previous = positive_int(args.previous_run, "previous run", 1, 10**24)
    else:
        previous = None
    chain_id = args.chain_id or str(run_id)
    if not re.fullmatch(r"[0-9]{1,24}", chain_id):
        fail("chain ID must be the original numeric GitHub run ID")
    if index == 1 and (previous is not None or chain_id != str(run_id)):
        fail("First segment must be a fresh chain with no previous run or custom chain ID")
    if index > 1 and (previous is None or chain_id == str(run_id) or previous >= run_id):
        fail("Continuation needs the earlier successful run ID and original chain ID")
    spec = clips[index - 1]
    result = {
        "contract": "wan-chain-v1",
        "config_path": args.plan_path,
        "config_sha256": content_sha,
        "chain_id": chain_id,
        "step": index,
        "total": len(clips),
        "run_id": run_id,
        "previous_run_id": previous,
        "scene": spec,
        "initial_image": data["initial_image"],
        "assembly": assembly,
        "postprocess": data.get("postprocess", {
            "method": "motion", "target_fps": 24, "speed": 1.5,
        }),
    }
    write_json(args.output, result)
    print(f"Prepared chain {chain_id} scene {index}/{len(clips)} "
          f"({spec['frames']} frames @ {spec['fps']} fps, "
          f"{spec['width']}x{spec['height']})", flush=True)


def read_plan(path):
    data = json_read(path)
    if data.get("contract") != "wan-chain-v1":
        fail("Unknown chain plan format")
    config = validate_plan_path(data["config_path"])
    if sha_file(config) != data["config_sha256"]:
        fail("Chain config changed after start. Do not silently mix revisions; "
             "restore the original file or start a new chain.")
    return data


def previous_artifact(plan, destination):
    prior = plan.get("previous_run_id")
    if not prior:
        fail("Previous run is missing")
    repo = os.environ.get("GITHUB_REPOSITORY", "anime-productions-love/ai-anime-generator")
    if repo != "anime-productions-love/ai-anime-generator":
        fail("This chain belongs to the unified anime generator repository")
    matches = []
    # GitHub's artifact listing can lag briefly behind upload-artifact
    # completion. Check only the specified predecessor, never an unrelated
    # "latest successful" run. Give the freshly uploaded artifact time to
    # become visible before treating it as a real failure.
    for attempt in range(5):
        artifacts = gh_api(f"repos/{repo}/actions/runs/{prior}/artifacts?per_page=100")
        matches = [
            a for a in artifacts.get("artifacts", [])
            if not a.get("expired") and ARTIFACT_PATTERN.fullmatch(a.get("name", ""))
            and ARTIFACT_PATTERN.fullmatch(a["name"]).group(1) == str(prior)
        ]
        if matches or attempt == 4:
            break
        print(f"Predecessor artifact not indexed yet ({attempt+1}/5); retrying.", flush=True)
        time.sleep(10)
    if len(matches) != 1:
        fail(f"Run {prior} needs exactly one unexpired Wan artifact; got {len(matches)}")
    destination.mkdir(parents=True, exist_ok=True)
    run([
        "gh", "run", "download", str(prior), "--repo", repo,
        "--name", matches[0]["name"], "--dir", destination,
    ])
    state_files = list(destination.rglob("chain-state.json"))
    if len(state_files) != 1:
        fail("Previous Wan artifact lacks exactly one chain-state.json")
    state = json_read(state_files[0])
    required = {
        "contract": "wan-chain-v1",
        "chain_id": plan["chain_id"],
        "config_sha256": plan["config_sha256"],
        "step": plan["step"] - 1,
        "run_id": prior,
        "total": plan["total"],
    }
    for key, expected in required.items():
        if state.get(key) != expected:
            fail(f"Previous chain artifact has mismatched {key}")
    ids = state.get("completed_run_ids")
    if (not isinstance(ids, list) or len(ids) != plan["step"] - 1
            or ids[-1] != prior or len(ids) != len(set(ids))
            or any(not isinstance(x, int) for x in ids)):
        fail("Previous chain history is missing or inconsistent")
    frames = list(destination.rglob("last_frame.png"))
    if len(frames) != 1:
        fail("Previous Wan artifact must contain exactly one last_frame.png")
    if sha_file(frames[0]) != state.get("last_frame_sha256"):
        fail("Previous chain frame failed SHA-256 verification")
    videos = list(destination.rglob("video.mp4"))
    if len(videos) != 1 or sha_file(videos[0]) != state.get("video_sha256"):
        fail("Previous chain video failed SHA-256 verification")
    return state, frames[0]


def source(args):
    plan = read_plan(args.plan)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    prev_state = None
    prev_frame = None
    if plan["step"] > 1:
        # Even if this particular shot starts from a new image, the prior
        # completed chain history must be verified and preserved.
        prev_state, prev_frame = previous_artifact(plan, ROOT / "work" / "previous-wan")
    # Only the first scene may resolve an external image. Every subsequent
    # scene MUST use the exact immediately preceding run's verified frame.
    input_url = plan["initial_image"] if plan["step"] == 1 else None
    if input_url:
        run([
            sys.executable, "scripts/resolve_scene_input.py",
            "--source-mode", "auto", "--input", input_url,
            "--workflow-file", WAN_WORKFLOW,
            "--repository", os.environ.get("GITHUB_REPOSITORY", ""),
            "--current-run-id", plan["run_id"], "--output", target,
        ])
        print("Used the first and ONLY externally selected image URL", flush=True)
    elif prev_frame:
        shutil.copyfile(prev_frame, target)
        print(f"Continued from exact run {plan['previous_run_id']}'s final frame", flush=True)
    else:
        fail("First scene needs initial_image or an explicit source")
    if not target.is_file() or target.stat().st_size == 0:
        fail("No usable source image was resolved")
    write_json(ROOT / "work" / "chain-history.json", {
        "completed_run_ids": prev_state["completed_run_ids"] if prev_state else [],
    })


def generate(args):
    plan = read_plan(args.plan)
    image = ROOT / "inputs" / "downloaded_image"
    if not image.is_file() or image.stat().st_size == 0:
        fail("Source resolution did not create inputs/downloaded_image")
    clip = plan["scene"]
    run([
        sys.executable, "-u", "scripts/generate_wan22.py",
        "--image", image, "--prompt", clip["prompt"],
        "--negative-prompt", clip["negative_prompt"],
        "--scene-name", "scene-" + str(plan["run_id"]),
        "--fit-mode", clip["fit_mode"], "--frames", clip["frames"],
        "--fps", clip["fps"], "--width", clip["width"],
        "--height", clip["height"], "--seed", clip["seed"],
        "--steps", "4", "--output-dir", args.output_dir,
    ])


def record(args):
    plan = read_plan(args.plan)
    history = json_read(ROOT / "work" / "chain-history.json")
    ids = history["completed_run_ids"]
    if len(ids) != plan["step"] - 1:
        fail("Prior chain history is incomplete")
    scene = Path(args.output_dir) / ("scene-" + str(plan["run_id"]))
    video = scene / "video.mp4"
    last_frame = scene / "last_frame.png"
    if any(not x.is_file() or x.stat().st_size == 0 for x in (video, last_frame)):
        fail("Wan must produce complete video.mp4 and last_frame.png first")
    state = {
        "contract": "wan-chain-v1",
        "chain_id": plan["chain_id"], "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "step": plan["step"], "total": plan["total"],
        "run_id": plan["run_id"], "previous_run_id": plan["previous_run_id"],
        "completed_run_ids": ids + [plan["run_id"]],
        "video_sha256": sha_file(video),
        "last_frame_sha256": sha_file(last_frame),
        "fps": plan["scene"]["fps"], "frames": plan["scene"]["frames"],
    }
    write_json(Path(args.output_dir) / "chain-state.json", state)
    # Also include complete chain history in the artifact: it survives
    # ephemeral GitHub runners without adding a database or Cloudflare Worker.
    print("Recorded completed Wan scene "
          f"{plan['step']}/{plan['total']} with verified video and final frame",
          flush=True)


def dispatch(workflow, inputs, repo, ref):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        fail("Invalid GitHub repository")
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", ref) or ".." in ref:
        fail("Invalid Git ref")
    payload = json.dumps({"ref": ref, "inputs": inputs})
    run([
        "gh", "api", "--method", "POST",
        f"repos/{repo}/actions/workflows/{workflow}/dispatches",
        "--input", "-",
    ], stdin=payload)
    print(f"GitHub accepted {workflow} workflow_dispatch. "
          "This does not prove the new run completed.", flush=True)


def advance(args):
    plan = read_plan(args.plan)
    state = json_read(Path(args.output_dir) / "chain-state.json")
    if state.get("run_id") != plan["run_id"] or state.get("step") != plan["step"]:
        fail("Cannot dispatch from mismatched chain completion record")
    if state.get("completed_run_ids", [])[-1:] != [plan["run_id"]]:
        fail("Missing successful current run in chain history")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    ref = os.environ.get("GITHUB_REF_NAME", "main")
    if repo != "anime-productions-love/ai-anime-generator":
        fail("Refusing workflow dispatch outside the intended repository")
    if plan["step"] < plan["total"]:
        dispatch(WAN_WORKFLOW, {
            "chain_plan": plan["config_path"],
            "chain_step": str(plan["step"] + 1),
            "chain_previous_run": str(plan["run_id"]),
            "chain_id": plan["chain_id"],
            "source_mode": "auto",
        }, repo, ref)
        print(f"Next clip {plan['step']+1}/{plan['total']} dispatched automatically.", flush=True)
    elif plan["assembly"].get("enabled", True):
        assembly = plan["assembly"]
        dispatch(MOVIE_WORKFLOW, {
            "scene_runs": ",".join(map(str, state["completed_run_ids"])),
            "width": str(assembly["width"]), "height": str(assembly["height"]),
            "fps": str(assembly["fps"]), "transition": assembly["transition"],
            "fade_seconds": assembly["fade_seconds"],
            "quality_crf": str(assembly["quality_crf"]),
            "postprocess_method": plan["postprocess"]["method"],
            "postprocess_fps": str(plan["postprocess"]["target_fps"]),
            "postprocess_speed": str(plan["postprocess"]["speed"]),
        }, repo, ref)
        print(f"All {plan['total']} clips generated. Final movie assembly dispatched.", flush=True)
    else:
        print(f"Chain complete: {plan['total']} clips. "
              "Auto assembly disabled; run IDs: " +
              ",".join(map(str, state["completed_run_ids"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--plan-path", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--previous-run", type=int, default=0)
    p.add_argument("--chain-id", default="")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--output", type=Path, default=Path("work/wan-chain-plan.json"))
    p = sub.add_parser("source")
    p.add_argument("--plan", type=Path, default=Path("work/wan-chain-plan.json"))
    p.add_argument("--output", type=Path, default=Path("inputs/downloaded_image"))
    p = sub.add_parser("generate")
    p.add_argument("--plan", type=Path, default=Path("work/wan-chain-plan.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wan22"))
    for cmd in ("record", "advance"):
        p = sub.add_parser(cmd)
        p.add_argument("--plan", type=Path, default=Path("work/wan-chain-plan.json"))
        p.add_argument("--output-dir", type=Path, default=Path("outputs/wan22"))
    args = parser.parse_args()
    try:
        {
            "prepare": prepare, "source": source, "generate": generate,
            "record": record, "advance": advance,
        }[args.operation](args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError,
            subprocess.CalledProcessError) as error:
        parser.exit(1, f"Wan chain {args.operation} failed: {error}\n")


if __name__ == "__main__":
    main()
