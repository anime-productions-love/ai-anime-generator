#!/usr/bin/env python3
"""Kaggle dual-T4 experimental motion-guided renderer (v1).

One PRIVATE Kaggle notebook run processes ONE source image + ONE driving clip.
Input files live in a Kaggle dataset, not inside a public GitHub workflow input.
No visual quality approval is inferred from a successfully rendered video.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path("/kaggle/working/anime-render")
IN = Path("/kaggle/input")
MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
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

def required_source(dataset, name):
    if not dataset or not isinstance(name, str) or not name:
        raise ValueError("A Kaggle dataset and relative input filename are required")
    if ".." in Path(name).parts or Path(name).is_absolute():
        raise ValueError("Unsafe Kaggle dataset path")
    candidate = (IN / dataset.split("/", 1)[-1] / name).resolve()
    if not candidate.is_relative_to(IN.resolve()):
        raise ValueError("Source escaped the Kaggle input directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Input not found: {candidate}; inspect attached Kaggle dataset paths")
    return candidate

def frames_from_reference(video, count, width, height):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        raise ValueError("Driving video has fewer than two decodable frames")
    samples = []
    for n in range(count):
        idx = round(n * (total - 1) / max(1, count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Could not decode driving reference frame {idx}")
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        # Edge conditioning is a first-stage motion approximation, not
        # human pose/face retargeting. It removes most reference video color.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 65, 150)
        edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        samples.append(Image.fromarray(edge_rgb))
    cap.release()
    return samples

def main():
    import torch
    import cv2
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    OUT.mkdir(parents=True, exist_ok=True)
    job = json.loads(Path(__file__).with_name("job-v1.json").read_text(encoding="utf-8"))
    if job.get("version") != "v1" or job.get("model") != "vace-1.3b":
        raise ValueError("Unsupported source-bound Kaggle job")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU allocated; refusing expensive CPU fallback")
    cards = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not cards or "T4" not in cards[0]:
        raise RuntimeError("Expected Kaggle T4 accelerator; actual " + repr(cards))
    # The pipeline uses ONE T4 with CPU offload. A second T4 is available
    # but is NOT automatically combined into a 32 GB memory pool.
    dataset = job["dataset_source"]
    image_file = required_source(dataset, job["image_path"])
    motion_file = required_source(dataset, job["motion_path"])
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
            negative_prompt=job.get("negative_prompt") or FAIL_NEGATIVE,
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
            "-sseof", "-0.1", "-i", str(target), "-frames:v", "1",
            str(OUT / "last_frame.png")
        ], check=True)
        if not (OUT / "last_frame.png").is_file():
            raise RuntimeError("Missing last frame")
        result = {
            "version": "v1", "status": "RENDERED_UNREVIEWED",
            "model": MODEL, "motion_control": "experimental_edge_guidance",
            "gpus": cards, "dataset_source": dataset,
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
