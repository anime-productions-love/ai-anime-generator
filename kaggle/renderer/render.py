#!/usr/bin/env python3
"""Kaggle T4 motion-guided renderer (v11).

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
import threading
import time
import traceback

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
    import numpy as np
    from PIL import Image, ImageOps, ImageStat
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
    # CUDA placement is managed by Accelerate sequential offload; model
    # components are not split across GPUs because Wan VACE prepares VAE
    # inputs on its execution device and balanced placement caused a mismatch.
    print("CUDA GPUs available: " + repr(cards), flush=True)
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
    # VACE interprets BLACK mask regions as control/conditioning; white
    # regions request unconstrained generation. Previously every motion edge
    # frame had a white mask, so its guidance was thrown away. For C2V keep
    # the first-frame image and every following edge map as conditioning.
    mask = [Image.new("L", (width, height), 0) for _ in range(frames)]
    control[0].save(OUT / "source-image.png")
    control[1].save(OUT / "control-edge-preview.png")
    # Read actual per-device activity, not just torch.cuda.device_count().
    # Sampling is observational and never initiates GPU work of its own.
    gpu_samples = []
    gpu_stop = threading.Event()
    def observe_gpus():
        while not gpu_stop.is_set():
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, check=True, timeout=5)
                entries = []
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 3:
                        entries.append({"gpu": int(parts[0]), "utilization_percent": int(parts[1]),
                                        "memory_mib": int(parts[2])})
                if entries:
                    gpu_samples.append({"elapsed_seconds": round(time.time() - started, 1),
                                        "devices": entries})
            except Exception as exc:
                print("GPU telemetry unavailable: " + str(exc)[:200], flush=True)
                break
            if gpu_stop.wait(5):
                break

    pipe = None
    started = time.time()
    gpu_thread = threading.Thread(target=observe_gpus, name="gpu-activity", daemon=True)
    gpu_thread.start()
    try:
        # The pinned VACE pipeline runs VAE.encode() on tensors located on
        # its execution device. "balanced" placement with an external VAE
        # left the VAE weights on CPU while its inputs were on CUDA. Use
        # Accelerate's submodule-level sequential offload instead, so
        # weights move to CUDA as their forward hooks execute.
        vae = AutoencoderKLWan.from_pretrained(
            MODEL, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanVACEPipeline.from_pretrained(
            MODEL, vae=vae, torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        pipe.vae.enable_tiling(
            tile_sample_min_height=256, tile_sample_min_width=256
        )
        pipe.enable_sequential_cpu_offload(gpu_id=0)
        memory_strategy = "sequential_cpu_offload"
        print(
            "Memory strategy: sequential CPU offload on cuda:0; "
            "VAE dtype=" + str(pipe.vae.dtype) +
            "; execution device=" + str(pipe._execution_device),
            flush=True
        )
        torch.cuda.empty_cache()
        print("Starting Wan VACE inference (25+ frames may use substantial memory)", flush=True)

        def reject_invalid_latents(_pipeline, step, _timestep, callback_kwargs):
            # FP16 on T4 can silently overflow; run the full VACE model in
            # FP32 and still reject any non-finite denoising state.
            latents = callback_kwargs["latents"]
            if not bool(torch.isfinite(latents).all().item()):
                raise RuntimeError(f"Invalid non-finite denoising latents at step {step}")
            # Keep one overwritten, clearly non-rendered latent checkpoint.
            # A failure mid-inference must not erase all expensive GPU progress.
            # This is for inspection/recovery and is NOT a resumable video.
            if (step + 1) % 5 == 0:
                try:
                    tmp = OUT / "progress-latents.pt.tmp"
                    torch.save({"step": step + 1, "latents": latents.detach().to(
                        device="cpu", dtype=torch.float16)}, tmp)
                    os.replace(tmp, OUT / "progress-latents.pt")
                    (OUT / "progress-v1.json").write_text(json.dumps({
                        "version": "v1", "status": "PARTIAL_LATENTS_NOT_VIDEO",
                        "completed_denoising_steps": step + 1,
                        "total_steps": steps, "resumable": False
                    }, indent=2) + "\n", encoding="utf-8")
                    print(f"Preserved non-video denoising checkpoint at step {step + 1}/{steps}", flush=True)
                except Exception as checkpoint_exc:
                    print("Checkpoint preservation warning: " + str(checkpoint_exc)[:300], flush=True)
            return callback_kwargs

        generated = pipe(
            video=control,
            mask=mask,
            reference_images=[image],
            prompt=job["prompt"],
            negative_prompt=FAIL_NEGATIVE + (", " + custom_negative.strip() if custom_negative.strip() else ""),
            height=height, width=width, num_frames=frames,
            num_inference_steps=steps, guidance_scale=5.0,
            generator=torch.Generator(device="cpu").manual_seed(int(job["seed"])),
            callback_on_step_end=reject_invalid_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]
        # FIRST preserve the expensive result. Validation must never run
        # before raw frames are independently durable in /kaggle/working.
        # Diffusers returns NumPy RGB frames here, not PIL images.
        frame_folder = OUT / "generated-frames"
        frame_folder.mkdir(parents=True, exist_ok=True)
        saved_frames = []
        for index, frame in enumerate(generated):
            if isinstance(frame, Image.Image):
                rgb = frame.convert("RGB")
            else:
                array = np.asarray(frame)
                if array.ndim != 3 or array.shape[2] not in (3, 4):
                    raise RuntimeError(f"Unexpected decoded frame shape at index {index}: {array.shape}")
                if array.dtype != np.uint8:
                    if not bool(np.isfinite(array).all()):
                        raise RuntimeError(f"Non-finite decoded frame at index {index}")
                    if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1:
                        array = array * 255
                    array = np.clip(array, 0, 255).astype(np.uint8)
                rgb = Image.fromarray(array).convert("RGB")
            frame_path = frame_folder / f"frame-{index:04d}.png"
            rgb.save(frame_path)
            saved_frames.append(rgb)
        (OUT / "last_frame.png").write_bytes((frame_folder / f"frame-{len(saved_frames)-1:04d}.png").read_bytes())
        raw = {
            "version": "v1", "status": "GENERATED_FRAMES_SAVED_UNREVIEWED",
            "renderer_version": "v10", "frames": len(saved_frames), "fps": fps,
            "frame_directory": "generated-frames",
            "last_frame_sha256": sha256(OUT / "last_frame.png"),
            "human_review_required": True, "approved_to_chain": False,
        }
        (OUT / "raw-output-v1.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"Preserved {len(saved_frames)} raw generated PNG frames before checks", flush=True)

        target = OUT / "video.mp4"
        export_to_video(saved_frames, str(target), fps=fps)
        if not target.is_file() or target.stat().st_size < 1000:
            raise RuntimeError("Model returned no usable video; PNG frames have been preserved")
        raw.update(status="GENERATED_VIDEO_SAVED_UNREVIEWED", video_sha256=sha256(target))
        (OUT / "raw-output-v1.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print("Preserved raw video.mp4 before quality checks", flush=True)

        # Optional quality rejection cannot delete or obscure the video/PNGs.
        frame_luma = [round(ImageStat.Stat(f.convert("L").resize((64, 64))).mean[0], 2)
                      for f in saved_frames]
        input_luma = float(ImageStat.Stat(image.convert("L").resize((64, 64))).mean[0])
        cutoff = max(12.0, input_luma * 0.22)
        collapsed_count = sum(v < cutoff for v in frame_luma)
        print("Frame luminance: input=%.2f cutoff=%.2f frames=%s" %
              (input_luma, cutoff, frame_luma), flush=True)
        if input_luma >= 45 and collapsed_count >= max(3, len(frame_luma) // 3):
            raise RuntimeError("Rendered frame brightness collapsed compared to source image; "
                               f"{collapsed_count}/{len(frame_luma)} frames below {cutoff:.1f}; "
                               "raw video and frames were retained")
        result = {
            "version": "v1", "status": "RENDERED_UNREVIEWED",
            "model": MODEL, "motion_control": "experimental_edge_guidance",
            "gpus": cards, "memory_strategy": memory_strategy,
            "gpu_execution": "single_gpu_cuda0",
            "gpu_activity_report": "gpu-activity-v1.json",
            "source_kind": "google_drive_public_link",
            "image_drive_file_id": job["image_drive_file_id"],
            "motion_drive_file_id": job["motion_drive_file_id"],
            "image_sha256": sha256(image_file),
            "reference_sha256": sha256(motion_file),
            "video_sha256": sha256(target),
            "last_frame_sha256": sha256(OUT / "last_frame.png"),
            "frames": frames, "fps": fps, "steps": steps, "seed": int(job["seed"]),
            "renderer_version": "v10", "control_mask": "black_all_frames",
            "model_dtype": "float32", "input_luma": round(input_luma, 2),
            "frame_luma": frame_luma,
            "duration_seconds": round(time.time() - started, 2),
            "requires_human_approval_before_chaining": True
        }
        (OUT / "result-v1.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("KAGGLE_RENDER_COMPLETED_UNREVIEWED")
    finally:
        gpu_stop.set()
        gpu_thread.join(timeout=7)
        summaries = []
        for index, name in enumerate(cards):
            records = [entry for row in gpu_samples for entry in row["devices"]
                       if entry["gpu"] == index]
            summaries.append({
                "gpu": index, "name": name, "observations": len(records),
                "max_utilization_percent": max(
                    (r["utilization_percent"] for r in records), default=None),
                "peak_memory_mib": max(
                    (r["memory_mib"] for r in records), default=None),
                "active_samples_ge_10pct": sum(
                    r["utilization_percent"] >= 10 for r in records)
            })
        report = {
            "version": "v1", "renderer_version": "v11",
            "execution_strategy": "sequential_cpu_offload_cuda0_only",
            "both_gpus_assigned_to_inference": False,
            "samples": gpu_samples, "summary": summaries
        }
        (OUT / "gpu-activity-v1.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Actual GPU usage summary: " + json.dumps(summaries), flush=True)
        del pipe
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for item in sorted(OUT.rglob("*")):
            if item.is_file() and not item.is_symlink():
                artifacts.append({
                    "path": item.relative_to(OUT).as_posix(),
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256(item)
                })
        (OUT / "failure-v1.json").write_text(json.dumps({
            "version": "v1", "status": "RENDER_FAILED_OUTPUTS_PRESERVED",
            "reason": str(exc), "traceback": traceback.format_exc()[-12000:],
            "available_artifacts": artifacts,
            "approved_to_chain": False
        }, indent=2) + "\n", encoding="utf-8")
        raise
