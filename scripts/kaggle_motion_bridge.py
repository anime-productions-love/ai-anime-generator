#!/usr/bin/env python3
"""GitHub-to-Kaggle motion rendering bridge (v2).

Submit a PRIVATE single-render kernel via Kaggle's official CLI or collect
results from a precisely identified kernel. No implicit generation or tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

KERNEL_SLUG = "anime-motion"
OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,49}$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DATASET = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
KERNEL_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
SIMPLE_FILE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")

def cmd(*args):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if p.returncode:
        # Do not echo authentication environment variables.
        raise RuntimeError(
            f"Kaggle {' '.join(args[:2])} failed: "
            f"{(p.stderr or p.stdout).strip()[:900]}"
        )
    return (p.stdout + "\n" + p.stderr).strip()

def resolve_username(explicit):
    if explicit:
        if not OWNER.fullmatch(explicit):
            raise ValueError("Invalid Kaggle username")
        return explicit
    # New KAGGLE_API_TOKEN credentials are self-identifying to the CLI.
    info = cmd("config", "view")
    matched = re.search(r"(?:^|\n)\s*-?\s*username:\s*([A-Za-z0-9_-]+)", info, re.I)
    if matched and OWNER.fullmatch(matched.group(1)):
        return matched.group(1)
    raise RuntimeError("Kaggle username not available from authenticated CLI; supply the workflow kaggle_username input")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def prepare(args):
    username = resolve_username(args.kaggle_username)
    if not DATASET.fullmatch(args.dataset):
        raise ValueError("dataset_source must look like kaggle-user/private-asset-dataset")
    for value in (args.image, args.motion):
        if not SIMPLE_FILE.fullmatch(value) or value in (".", ".."):
            raise ValueError("Image/motion must be filenames at the dataset root; no URL or directory paths")
    if not re.fullmatch(r"[0-9]{4,24}", args.run_id):
        raise ValueError("Run identifier must be the originating GitHub run ID")
    if not args.prompt.strip() or len(args.prompt) > 2000:
        raise ValueError("A nonempty motion prompt up to 2000 characters is required")
    if args.frames < 5 or args.frames > 81 or (args.frames - 1) % 4:
        raise ValueError("frames must be 4n+1, between 5 and 81")
    if args.steps < 8 or args.steps > 50 or not 1 <= args.fps <= 30:
        raise ValueError("Steps must be 8–50, FPS 1–30")
    if args.width % 16 or args.height % 16 or not (256 <= args.width <= 832 and 256 <= args.height <= 832):
        raise ValueError("Width/height must be 256–832 multiples of 16")
    kernel = f"{username}/{KERNEL_SLUG}-{args.run_id}"
    folder = Path(args.path)
    folder.mkdir(parents=True, exist_ok=True)
    source_renderer = Path(__file__).resolve().parents[1] / "kaggle/renderer/render.py"
    write_json(folder / "kernel-metadata.json", {
        "id": kernel,
        "title": f"Anime Motion Render {args.run_id}",
        "code_file": "render.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [args.dataset],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    write_json(folder / "job-v1.json", {
        "version": "v1", "model": "vace-1.3b",
        "source_github_run_id": args.run_id,
        "dataset_source": args.dataset,
        "image_path": args.image, "motion_path": args.motion,
        "prompt": args.prompt, "negative_prompt": args.negative,
        "frames": args.frames, "fps": args.fps,
        "width": args.width, "height": args.height,
        "steps": args.steps, "seed": args.seed,
        "approval_status": "RENDER_REQUESTED_ONLY"
    })
    # Kaggle's push API treats code_file as its authoritative script;
    # embed the job in that script so data survives even if extra
    # sibling JSON files are not copied into the notebook version.
    script = source_renderer.read_text(encoding="utf-8")
    token = "EMBEDDED_JOB_JSON = None  # Filled by the GitHub submit bridge"
    if script.count(token) != 1:
        raise ValueError("Renderer job-injection marker missing or ambiguous")
    job_json = (folder / "job-v1.json").read_text(encoding="utf-8")
    script = script.replace(token, "EMBEDDED_JOB_JSON = " + repr(job_json))
    (folder / "render.py").write_text(script, encoding="utf-8")
    receipt = {
        "version": "v1", "status": "PREPARED_NOT_SUBMITTED",
        "origin_github_run_id": args.run_id, "kernel": kernel,
        "dataset_source": args.dataset, "gpu_requested": "NvidiaTeslaT4",
        "gpu_expected": "T4 x2",
        "generation_model": "Wan2.1 VACE 1.3B",
        "human_review_required": True,
    }
    write_json(args.receipt, receipt)
    print(f"Kaggle renderer source/metadata prepared: {kernel}")

def submit(args):
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if receipt.get("status") != "PREPARED_NOT_SUBMITTED":
        raise ValueError("Job receipt not in prepared state")
    cmd("kernels", "push", "-p", args.path, "--accelerator", "NvidiaTeslaT4")
    receipt["status"] = "SUBMITTED_TO_KAGGLE_NOT_YET_RENDERED"
    write_json(args.receipt, receipt)
    print(f"Kaggle accepted kernel submission: {receipt['kernel']}")
    print(f"View: https://www.kaggle.com/code/{receipt['kernel']}")

def select_kernel(args):
    if args.kernel:
        if not KERNEL_REF.fullmatch(args.kernel):
            raise ValueError("Expected owner/kernel-slug")
        return args.kernel
    if not re.fullmatch(r"[0-9]{4,24}", args.source_run_id):
        raise ValueError("Provide a source GitHub run ID for retrieval")
    return f"{resolve_username(args.kaggle_username)}/{KERNEL_SLUG}-{args.source_run_id}"

def status(args):
    kernel = select_kernel(args)
    print(cmd("kernels", "status", kernel))
    print("Kaggle kernel: " + kernel)

def collect(args):
    kernel = select_kernel(args)
    status_text = cmd("kernels", "status", kernel)
    print("Kaggle reported: " + status_text[:1200])
    if re.search(r"\b(error|failed|cancelled|canceled)\b", status_text, flags=re.I):
        raise RuntimeError("Kaggle render failed; inspect Kaggle notebook execution logs")
    if re.search(r"\b(pending|queued|running|in progress)\b", status_text, flags=re.I):
        raise RuntimeError("Kaggle render is still pending/running; re-run retrieve later (do not resubmit)")
    if not re.search(r"\b(complete|completed|success)\b", status_text, flags=re.I):
        raise RuntimeError("Unrecognized Kaggle status; refusing to download a possibly unfinished version")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cmd("kernels", "output", kernel, "-p", str(output), "-o")
    manifests = list(output.rglob("result-v1.json"))
    videos = list(output.rglob("video.mp4"))
    frames = list(output.rglob("last_frame.png"))
    if not(len(manifests) == len(videos) == len(frames) == 1):
        raise RuntimeError("Kaggle output missing one exact result, video or final-frame file")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("status") != "RENDERED_UNREVIEWED":
        raise RuntimeError("Kaggle did not produce a completed unreviewed render")
    if sha256(videos[0]) != manifest.get("video_sha256") or sha256(frames[0]) != manifest.get("last_frame_sha256"):
        raise RuntimeError("Output SHA-256 does not match Kaggle render manifest")
    if videos[0].stat().st_size < 1000 or frames[0].stat().st_size == 0:
        raise RuntimeError("Output media is empty")
    write_json(output / "bridge-result-v1.json", {
        "version": "v1", "status": "RETRIEVED_VERIFIED_NEEDS_HUMAN_REVIEW",
        "kernel": kernel, "model": manifest["model"],
        "video_sha256": manifest["video_sha256"],
        "last_frame_sha256": manifest["last_frame_sha256"],
        "approved_to_chain": False,
    })
    print("Verified Kaggle video and last-frame checksums. NOT approved to chain.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--kaggle-username", default="")
    a.add_argument("--run-id", required=True)
    a.add_argument("--dataset", required=True)
    a.add_argument("--image", required=True)
    a.add_argument("--motion", required=True)
    a.add_argument("--prompt", required=True)
    a.add_argument("--negative", default="")
    a.add_argument("--frames", type=int, default=17)
    a.add_argument("--fps", type=int, default=8)
    a.add_argument("--width", type=int, default=512)
    a.add_argument("--height", type=int, default=288)
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--path", default="work/kaggle-stage")
    a.add_argument("--receipt", default="work/kaggle-receipt-v1.json")
    a = sub.add_parser("submit")
    a.add_argument("--path", required=True)
    a.add_argument("--receipt", required=True)
    for act in ("status", "collect"):
        a = sub.add_parser(act)
        a.add_argument("--kaggle-username", default="")
        a.add_argument("--kernel", default="")
        a.add_argument("--source-run-id", default="")
        if act == "collect":
            a.add_argument("--output", required=True)
    args = p.parse_args()
    {"prepare": prepare, "submit": submit, "status": status,
     "collect": collect}[args.action](args)

if __name__ == "__main__":
    main()
