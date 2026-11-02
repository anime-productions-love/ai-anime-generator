#!/usr/bin/env python3
"""Wan 2.2 TI2V-5B Turbo scene generation via stable-diffusion.cpp, v1."""
import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

MODEL_ID = "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF"
ENGINE_COMMIT = "290148d2e31ca6ef9757d567905776b5f3646db5"
DEFAULT_NEGATIVE = (
    "frozen frame, still picture, unnatural jerk, flicker, extra limbs, "
    "distorted eyes, deformed hands, text, watermark, low quality"
)


def parse_args():
    parser = argparse.ArgumentParser(description="CPU-only Wan2.2 Turbo image-to-video scene")
    parser.add_argument("--image", required=True, help="Already downloaded and validated local image")
    parser.add_argument("--prompt", required=True, help="Describe movement, character action, and camera")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--frames", type=int, default=17, help="Desired output frame count; internally rounded to 4n+1")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-name", default="scene-001")
    parser.add_argument("--fit-mode", choices=("contain-blur", "contain", "crop"), default="contain-blur")
    parser.add_argument("--model-dir", type=Path, default=Path("models/wan22"))
    parser.add_argument("--sd-cli", type=Path, default=Path("sd-cpp/build/bin/sd-cli"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wan22"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if min(args.width, args.height) < 128 or args.width % 16 or args.height % 16:
        parser.error("width and height must be multiples of 16 and at least 128")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.scene_name):
        parser.error("--scene-name accepts letters, numbers, hyphens and underscores only")
    return args


def frame_image(source, width, height, fit_mode):
    image = Image.open(source).convert("RGB")
    native_size = list(image.size)
    if fit_mode == "crop":
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS), native_size
    foreground = ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)
    if fit_mode == "contain-blur":
        backdrop = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        backdrop = backdrop.filter(ImageFilter.GaussianBlur(radius=max(10, width // 32)))
    else:
        backdrop = Image.new("RGB", (width, height), (16, 16, 16))
    backdrop.paste(foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2))
    return backdrop, native_size


def main():
    args = parse_args()
    scene = args.output_dir / args.scene_name
    scene.mkdir(parents=True, exist_ok=True)
    image_path = scene / "framed_input.png"
    raw_video = scene / "raw.avi"
    mp4 = scene / "video.mp4"
    last_frame = scene / "last_frame.png"

    image, source_dimensions = frame_image(args.image, args.width, args.height, args.fit_mode)
    image.save(image_path)
    (scene / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")
    (scene / "negative_prompt.txt").write_text(args.negative_prompt + "\n", encoding="utf-8")

    # Wan's temporal compression requires 4n+1 frames. Keep the user's
    # unrestricted numeric output request and trim the produced video.
    native_frames = max(5, 4 * math.ceil((args.frames - 1) / 4) + 1)
    print(f"Requested frames: {args.frames}; model frames: {native_frames}", flush=True)
    if args.steps != 4:
        print("CAUTION: Turbo is distilled for four steps; other counts are experimental.", flush=True)
    model_files = {
        "diffusion": args.model_dir / "Wan2_2-TI2V-5B-Turbo-Q4_K_M.gguf",
        "vae": args.model_dir / "Wan2.2_VAE.safetensors",
        "text": args.model_dir / "umt5-xxl-encoder-Q3_K_S.gguf",
    }
    for kind, file in model_files.items():
        if not file.is_file() or file.stat().st_size < 100_000_000:
            raise RuntimeError(f"Missing or incomplete {kind} model: {file}")

    cmd = [
        str(args.sd_cli.resolve()), "-M", "vid_gen",
        "--diffusion-model", str(model_files["diffusion"].resolve()),
        "--vae", str(model_files["vae"].resolve()),
        "--t5xxl", str(model_files["text"].resolve()),
        "-p", args.prompt, "-n", args.negative_prompt,
        "-i", str(image_path.resolve()),
        "--cfg-scale", "1.0", "--sampling-method", "euler",
        "--steps", str(args.steps), "--flow-shift", "3.0",
        "--video-frames", str(native_frames), "--fps", str(args.fps),
        "-W", str(args.width), "-H", str(args.height),
        "-s", str(args.seed),
        "--vae-tiling", "--temporal-tiling",
        "--vae-tile-size", "256x256",
        "--extra-tiling-args", "temporal_tile_frames=4,temporal_decode_frames=1",
        "-o", str(raw_video.resolve()),
    ]
    start = time.monotonic()
    print("Starting CPU-only Wan 2.2 TI2V Turbo inference.", flush=True)
    subprocess.run(cmd, check=True)
    if not raw_video.is_file() or raw_video.stat().st_size == 0:
        raise RuntimeError("Inference finished without an AVI file")

    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(raw_video), "-frames:v", str(args.frames),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(mp4)
    ], check=True)
    # The exact last delivered frame is needed for optional scene continuation.
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(mp4), "-vf", f"select=eq(n\\,{args.frames - 1})",
        "-vsync", "0", "-frames:v", "1", str(last_frame)
    ], check=True)
    if not mp4.is_file() or not last_frame.is_file():
        raise RuntimeError("Scene conversion did not produce all deliverables")

    metadata = {
        "version": "v1",
        "model": MODEL_ID,
        "engine_commit": ENGINE_COMMIT,
        "device": "cpu",
        "image_source": args.image,
        "original_image_size": source_dimensions,
        "fit_mode": args.fit_mode,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "scene_name": args.scene_name,
        "frames_requested": args.frames,
        "frames_generated": native_frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (scene / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Drop uncompressed intermediate so it cannot inflate the GitHub artifact.
    raw_video.unlink()
    print(f"Scene saved to {scene} (MP4, final frame, and metadata)", flush=True)


if __name__ == "__main__":
    main()
