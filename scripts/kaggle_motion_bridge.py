#!/usr/bin/env python3
"""GitHub-to-Kaggle motion rendering bridge (v10).

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
import time
from urllib.parse import urlparse, parse_qs

KERNEL_SLUG = "anime-motion-render"
OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,49}$")
KERNEL_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")

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

def drive_file_id(share_url):
    """Accept only standard Google Drive file links; never embed signed URLs."""
    parsed = urlparse(share_url)
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com" or parsed.port is not None:
        raise ValueError("Use an https://drive.google.com/file/d/FILE_ID/view link")
    m = re.fullmatch(r"/file/d/([A-Za-z0-9_-]{15,128})(?:/(?:view|preview|edit))?/?", parsed.path)
    if m:
        return m.group(1)
    if parsed.path in ("/open", "/uc"):
        ids = parse_qs(parsed.query).get("id", [])
        if len(ids) == 1 and re.fullmatch(r"[A-Za-z0-9_-]{15,128}", ids[0]):
            return ids[0]
    raise ValueError("Unsupported Google Drive URL; use Share > Copy link for a file")


def prepare(args):
    username = resolve_username(args.kaggle_username)
    image_id = drive_file_id(args.image_drive)
    motion_id = drive_file_id(args.motion_drive)
    if image_id == motion_id:
        raise ValueError("Source image and motion reference must be different Drive files")
    if not re.fullmatch(r"[0-9]{4,24}", args.run_id):
        raise ValueError("Run identifier must be the originating GitHub run ID")
    if not args.prompt.strip() or len(args.prompt) > 2000:
        raise ValueError("A nonempty motion prompt up to 2000 characters is required")
    if args.engine not in ("vace-1.3b", "scail2-q6k"):
        raise ValueError("Unsupported Kaggle rendering engine")
    if not 5 <= args.frames <= 81:
        raise ValueError("Requested frames must be between 5 and 81")
    requested_frames = args.frames
    if args.engine == "vace-1.3b":
        # Existing VACE inference contract remains unchanged.
        args.frames = 5 + 4 * ((requested_frames - 5 + 2) // 4)
    if args.steps < (4 if args.engine == "scail2-q6k" else 8) or args.steps > 50 or not 1 <= args.fps <= 30:
        raise ValueError("Invalid engine-specific steps or FPS")
    if not 1 <= args.duration_seconds <= 5:
        raise ValueError("First SCAIL-2 clip must be 1–5 seconds")
    if args.width % 16 or args.height % 16 or not (256 <= args.width <= 832 and 256 <= args.height <= 832):
        raise ValueError("Width/height must be 256–832 multiples of 16")
    kernel = f"{username}/{KERNEL_SLUG}-{args.run_id}"
    folder = Path(args.path)
    folder.mkdir(parents=True, exist_ok=True)
    render_file = "scail2.py" if args.engine == "scail2-q6k" else "render.py"
    source_renderer = Path(__file__).resolve().parents[1] / "kaggle/renderer" / render_file
    # Optional privately uploaded model datasets, reusable across Kaggle runs.
    datasets = [part.strip() for part in args.kaggle_datasets.split(";") if part.strip()]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", item) for item in datasets):
        raise ValueError("Kaggle dataset sources must be semicolon-separated owner/slug identifiers")
    write_json(folder / "kernel-metadata.json", {
        "id": kernel,
        "title": f"Anime Motion Render {args.run_id}",
        "code_file": render_file, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": datasets,
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    write_json(folder / "job-v1.json", {
        "version": "v1", "model": args.engine,
        "source_github_run_id": args.run_id,
        "source_kind": "google_drive_public_link",
        "image_drive_file_id": image_id, "motion_drive_file_id": motion_id,
        "prompt": args.prompt, "negative_prompt": args.negative,
        "frames": args.frames, "requested_frames": requested_frames,
        "fps": args.fps,
        "width": args.width, "height": args.height,
        "steps": args.steps, "seed": args.seed,
        "duration_seconds": args.duration_seconds if args.engine == "scail2-q6k" else None,
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
    (folder / render_file).write_text(script, encoding="utf-8")
    receipt = {
        "version": "v1", "status": "PREPARED_NOT_SUBMITTED",
        "origin_github_run_id": args.run_id, "kernel": kernel,
        "source_kind": "google_drive_public_link",
        "image_drive_file_id": image_id, "motion_drive_file_id": motion_id,
        "requested_frames": requested_frames,
        "render_frames": args.frames,
        "gpu_requested": "NvidiaTeslaT4",
        "gpu_expected": "T4 x2",
        "generation_model": ("SCAIL-2 Q6_K" if args.engine == "scail2-q6k"
                             else "Wan2.1 VACE 1.3B"),
        "dataset_sources": datasets,
        "human_review_required": True,
    }
    write_json(args.receipt, receipt)
    print(f"Kaggle renderer source/metadata prepared: {kernel}")

def submit(args):
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if receipt.get("status") != "PREPARED_NOT_SUBMITTED":
        raise ValueError("Job receipt not in prepared state")
    print("Kaggle CLI push output: " + cmd("kernels", "push", "-p", args.path, "--accelerator", "NvidiaTeslaT4")[:800], flush=True)
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

def collect(args, allow_status_denied=False):
    kernel = select_kernel(args)
    try:
        status_text = cmd("kernels", "status", kernel)
    except RuntimeError as exc:
        if not allow_status_denied or "kernels.get" not in str(exc):
            raise
        # A private kernel can become temporarily unreadable via status API.
        # Never infer completion: require actual retrievable output + hashes.
        status_text = ""
        print("Kaggle status access denied; requiring actual complete render output", flush=True)
    if status_text:
        print("Kaggle reported: " + status_text[:1200])
    if re.search(r"\b(error|failed|cancelled|canceled)\b", status_text, flags=re.I):
        raise RuntimeError("Kaggle render failed; inspect Kaggle notebook execution logs")
    if re.search(r"\b(pending|queued|running|in progress)\b", status_text, flags=re.I):
        raise RuntimeError("Kaggle render is still pending/running; re-run retrieve later (do not resubmit)")
    if status_text and not re.search(r"\b(complete|completed|success)\b", status_text, flags=re.I):
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

def salvage(args):
    """Retrieve and account for ALL extant outputs even from a failed notebook.

    A salvaged result is never a successful render and never approved to chain.
    This is independent of Kaggle's private-kernel status endpoint.
    """
    kernel = select_kernel(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        cmd("kernels", "output", kernel, "-p", str(output), "-o")
    except RuntimeError as error:
        # The monitor may already have fetched some actual outputs.
        present = [p for p in output.rglob("*") if p.is_file()]
        if not present:
            raise RuntimeError("No Kaggle failed-job artifacts retrievable: " +
                               str(error)[:400]) from error
        print("Kaggle output API denied; archiving already retrieved files", flush=True)
    media = [
        p for p in output.rglob("*") if p.is_file() and not p.is_symlink()
        and p.name != "bridge-result-v1.json"
    ]
    if not media:
        raise RuntimeError("Kaggle returned no files to salvage")
    # Verify any checksums written by the remote renderer, but keep a
    # failure clearly distinct from a complete/visually approved render.
    for manifest in output.rglob("raw-output-v1.json"):
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        parent = manifest.parent
        for filename, key in (("video.mp4", "video_sha256"),
                              ("last_frame.png", "last_frame_sha256")):
            expected = raw.get(key)
            path = parent / filename
            if expected and (not path.is_file() or sha256(path) != expected):
                raise RuntimeError("Failed render artifact hash mismatch: " + filename)
    assets = [{
        "path": path.relative_to(output).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path)
    } for path in sorted(media)]
    report = {
        "version": "v1", "status": "FAILED_OR_UNVERIFIED_OUTPUTS_SALVAGED",
        "kernel": kernel, "files": assets,
        "video_available": any(p.name == "video.mp4" for p in media),
        "generated_frames_available": any(p.parent.name == "generated-frames" for p in media),
        "approved_to_chain": False, "human_review_required": True
    }
    write_json(output / "bridge-result-v1.json", report)
    print(f"Preserved {len(assets)} files from {kernel}; "
          f"video available: {report['video_available']}. NOT approved to chain.", flush=True)


def monitor(args):
    """Observe exact Kaggle kernel until verified success, real failure, or deadline."""
    kernel = select_kernel(args)
    deadline = time.monotonic() + args.timeout_seconds
    previous = None
    while True:
        try:
            status_text = cmd("kernels", "status", kernel)
        except RuntimeError as exc:
            if "kernels.get" not in str(exc):
                raise
            # Kaggle intermittently returns HTTP 403 even for the same private
            # kernel that was previously RUNNING and successfully authenticated.
            # Try verified output retrieval without resubmitting GPU work.
            status_text = "STATUS_ACCESS_DENIED"
            if status_text != previous:
                print("Kaggle status became inaccessible; retrying verified output retrieval", flush=True)
                previous = status_text
            try:
                collect(args, allow_status_denied=True)
                return
            except RuntimeError as output_exc:
                detail = str(output_exc)
                retryable = ("kernels output failed" in detail or
                             "Kaggle output missing one exact" in detail or
                             "Kaggle render is still pending" in detail)
                if not retryable:
                    raise
                print("Output not yet retrievable: " + detail[:500], flush=True)
            if time.monotonic() >= deadline:
                raise RuntimeError("Kaggle status denied and verified output unavailable until monitor deadline")
            time.sleep(min(60, max(1, deadline - time.monotonic())))
            continue
        if status_text != previous:
            print("Remote Kaggle status: " + status_text[:1000], flush=True)
            previous = status_text
        if re.search(r"\b(error|failed|cancelled|canceled)\b", status_text, flags=re.I):
            failure_dir = Path(args.output) / "remote-failure"
            failure_dir.mkdir(parents=True, exist_ok=True)
            try:
                cmd("kernels", "output", kernel, "-p", str(failure_dir), "-o")
                for failure in failure_dir.rglob("failure-v1.json"):
                    print("Remote render failure manifest: " + failure.read_text(encoding="utf-8")[:14000], flush=True)
            except Exception as exc:
                print("Could not download Kaggle failure output: " + str(exc)[:600], flush=True)
            raise RuntimeError("Kaggle execution failed: " + kernel)
        if re.search(r"\b(complete|completed|success)\b", status_text, flags=re.I):
            collect(args)
            return
        if not re.search(r"\b(pending|queued|running|in progress)\b", status_text, flags=re.I):
            raise RuntimeError("Unrecognized Kaggle kernel state: " + status_text[:500])
        if time.monotonic() >= deadline:
            raise RuntimeError("Kaggle still running when GitHub monitor deadline expired; do not resubmit without checking kernel")
        time.sleep(min(30, max(1, deadline - time.monotonic())))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--kaggle-username", default="")
    a.add_argument("--engine", choices=("vace-1.3b", "scail2-q6k"), default="vace-1.3b")
    a.add_argument("--kaggle-datasets", default="")
    a.add_argument("--duration-seconds", type=int, default=2)
    a.add_argument("--run-id", required=True)
    a.add_argument("--image-drive", required=True)
    a.add_argument("--motion-drive", required=True)
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
    for act in ("status", "collect", "monitor", "salvage"):
        a = sub.add_parser(act)
        a.add_argument("--kaggle-username", default="")
        a.add_argument("--kernel", default="")
        a.add_argument("--source-run-id", default="")
        if act in ("collect", "monitor", "salvage"):
            a.add_argument("--output", required=True)
        if act == "monitor":
            a.add_argument("--timeout-seconds", type=int, default=4200)
    args = p.parse_args()
    {"prepare": prepare, "submit": submit, "status": status,
     "collect": collect, "monitor": monitor, "salvage": salvage}[args.action](args)

if __name__ == "__main__":
    main()
