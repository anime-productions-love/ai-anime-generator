#!/usr/bin/env python3
"""Kaggle dual-T4 experimental motion-guided renderer (v6).

One PRIVATE Kaggle notebook run processes ONE source image + ONE driving clip.
Images and MP4/animated WebP/GIF references are downloaded from link-viewable Google Drive files.
No visual quality approval is inferred from a successfully rendered video.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

OUT = Path("/kaggle/working/anime-render")
MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
EMBEDDED_JOB_JSON = None  # Filled by the GitHub submit bridge
FAIL_NEGATIVE = (
    "body horror, gore, dismemberment, torn skin, separated limbs, fused bodies, "
    "extra arms, extra legs, deformed hands, uncanny faces, distorted anatomy, "
    "melted characters, uncanny facial expression, unwanted violence"
)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for data in iter(lambda: src.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()

def download_drive_source(file_id, target_name):
    """Download only a constrained Drive file ID, not an arbitrary remote URL."""
    if not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{15,128}", file_id):
        raise ValueError("Invalid Google Drive file identifier")
    if target_name not in ("source-image.bin", "motion-reference.mp4"):
        raise ValueError("Unsupported source destination")
    import gdown
    directory = Path("/kaggle/working/source-input")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / target_name
    try:
        downloaded = gdown.download(id=file_id, output=str(target), quiet=True)
    except Exception as exc:
        raise RuntimeError(
            "Google Drive download failed; check Anyone with the link -> Viewer, "
            "file availability, and Drive download quota"
        ) from exc
    if not downloaded or not target.is_file() or not (1 <= target.stat().st_size <= 1024 * 1024 * 1024):
        raise RuntimeError("Google Drive source is missing, empty, or too large")
    return target

def frames_from_reference(video, count, width, height):
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    def edge_image(frame, rgb=False):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 65, 150)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    # gdown stores the source under a fixed local name, so detect by file
    # signature rather than relying on its misleading .mp4 suffix.
    try:
        animated = Image.open(video)
    except (UnidentifiedImageError, OSError):
        animated = None
    if animated is not None:
        with animated:
            if animated.format not in ("WEBP", "GIF"):
                raise ValueError("Motion reference is a still image; provide animated WebP, GIF, or a video")
            total = int(getattr(animated, "n_frames", 1))
            if total < 2:
                raise ValueError("Animated motion reference must contain at least two frames")
            samples = []
            for n in range(count):
                idx = round(n * (total - 1) / max(1, count - 1))
                animated.seek(idx)
                samples.append(edge_image(np.asarray(animated.convert("RGB")), rgb=True))
            print(f"Reference motion decoded: {total} animated {animated.format} frames")
            return samples

    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            raise ValueError("Video reference has fewer than two decodable frames; check file type or codec")
        samples = []
        for n in range(count):
            idx = round(n * (total - 1) / max(1, count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not decode reference video frame {idx}")
            samples.append(edge_image(frame))
        print(f"Reference motion decoded: {total} video frames")
        return samples
    finally:
        cap.release()

def main():
    # Notebook code itself carries the approved JSON so Kaggle CLI upload
    # need not preserve arbitrary sidecar files when pushing a kernel.
    # Install documented Wan VACE dependencies before importing the pipeline.
    # Kaggle base images change, so do not trust preinstalled versions.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "-q", "diffusers==0.35.2", "transformers==4.57.1",
        "accelerate==1.11.0", "ftfy", "sentencepiece", "gdown>=5.2,<6"
    ], check=True)
    import torch
    import cv2
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    OUT.mkdir(parents=True, exist_ok=True)
    job = json.loads(EMBEDDED_JOB_JSON) if EMBEDDED_JOB_JSON is not None else json.loads(\
        Path(__file__).with_name("job-v1.json").read_text(encoding="utf-8"))
    if job.get("version") != "v1" or job.get("model") != "vace-1.3b":
        raise ValueError("Unsupported source-bound Kaggle job")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU allocated; refusing expensive CPU fallback")
    cards = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not cards or "T4" not in cards[0]:
        raise RuntimeError("Expected Kaggle T4 accelerator; actual " + repr(cards))
    # The pipeline uses ONE T4 with CPU offload. A second T4 is available
    # but is NOT automatically combined into a 32 GB memory pool.
    if job.get("source_kind") != "google_drive_public_link":
        raise ValueError("Unsupported input storage source")
    image_file = download_drive_source(job["image_drive_file_id"], "source-image.bin")
    motion_file = download_drive_source(job["motion_drive_file_id"], "motion-reference.mp4")
    width, height = int(job["width"]), int(job["height"])
    frames = int(job["frames"])
    steps = int(job["steps"])
    fps = int(job["fps"])
    if not (width % 16 == height % 16 == 0 and 256 <= width <= 832
            and 256 <= height <= 832 and 5 <= frames <= 81
            and (frames - 1) % 4 == 0 and 8 <= steps <= 50 and 1 <= fps <= 30):
        raise ValueError("Out-of-bounds Wan VACE inference settings")
    if len(job["prompt"]) > 2000 or not job["prompt"].strip():
        raise ValueError("Empty or overly long prompt")
    custom_negative = job.get("negative_prompt") or ""
    if not isinstance(custom_negative, str) or len(custom_negative) > 2000:
        raise ValueError("Invalid or overly long negative prompt")

    image = ImageOps.fit(Image.open(image_file).convert("RGB"), (width, height))
    control = frames_from_reference(motion_file, frames, width, height)
    control[0] = image  # exact first-frame guidance
    mask = [Image.new("L", (width, height), 0)]
    mask.extend(Image.new("L", (width, height), 255) for _ in range(frames - 1))
    control[0].save(OUT / "source-image.png")
    control[1].save(OUT / "control-edge-preview.png")
    pipe = None
    started = time.time()
    try:
        vae = AutoencoderKLWan.from_pretrained(
            MODEL, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanVACEPipeline.from_pretrained(
            MODEL, vae=vae, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        pipe.enable_model_cpu_offload()
        generated = pipe(
            video=control,
            mask=mask,
            reference_images=[image],
            prompt=job["prompt"],
            negative_prompt=FAIL_NEGATIVE + (", " + custom_negative.strip() if custom_negative.strip() else ""),
            height=height, width=width, num_frames=frames,
            num_inference_steps=steps, guidance_scale=5.0,
            generator=torch.Generator(device="cpu").manual_seed(int(job["seed"])),
        ).frames[0]
        target = OUT / "video.mp4"
        export_to_video(generated, str(target), fps=fps)
        if not target.is_file() or target.stat().st_size < 1000:
            raise RuntimeError("Model returned no usable video")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(target), "-vf", f"select=eq(n\\,{frames - 1})",
            "-vsync", "0", "-frames:v", "1", str(OUT / "last_frame.png")
        ], check=True)
        if not (OUT / "last_frame.png").is_file():
            raise RuntimeError("Missing last frame")
        result = {
            "version": "v1", "status": "RENDERED_UNREVIEWED",
            "model": MODEL, "motion_control": "experimental_edge_guidance",
            "gpus": cards, "source_kind": "google_drive_public_link",
            "image_drive_file_id": job["image_drive_file_id"],
            "motion_drive_file_id": job["motion_drive_file_id"],
            "image_sha256": sha256(image_file),
            "reference_sha256": sha256(motion_file),
            "video_sha256": sha256(target),
            "last_frame_sha256": sha256(OUT / "last_frame.png"),
            "frames": frames, "fps": fps, "steps": steps, "seed": int(job["seed"]),
            "duration_seconds": round(time.time() - started, 2),
            "requires_human_approval_before_chaining": True
        }
        (OUT / "result-v1.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("KAGGLE_RENDER_COMPLETED_UNREVIEWED")
    finally:
        del pipe
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "failure-v1.json").write_text(json.dumps({
            "version": "v1", "status": "RENDER_FAILED", "reason": str(exc)
        }, indent=2) + "\n", encoding="utf-8")
        raise
