#!/usr/bin/env python3
"""SVD-XT image-to-video inference on a CPU-only runner (v4)."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image, ImageFilter, ImageOps

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
WIDTH, HEIGHT = 1024, 576


def arguments():
    parser = argparse.ArgumentParser(description="Generate an SVD-XT clip with Diffusers on CPU")
    parser.add_argument("--image", required=True, help="Public HTTP(S) image URL or local path")
    parser.add_argument("--frames", type=int, default=25, help="Positive number of frames; >25 is experimental and may exhaust RAM")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion steps (1-100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=6, help="MP4 playback FPS")
    parser.add_argument("--motion-bucket-id", type=int, default=180)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--decode-chunk-size", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--fit-mode", choices=("contain-blur", "contain", "crop"), default="contain-blur")
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be a positive integer")
    if not 1 <= args.steps <= 100:
        parser.error("--steps must be between 1 and 100")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    if not 1 <= args.decode_chunk_size <= args.frames:
        parser.error("--decode-chunk-size must be between 1 and --frames")
    if not 1 <= args.motion_bucket_id <= 255:
        parser.error("--motion-bucket-id must be between 1 and 255")
    if not 0.0 <= args.noise_aug_strength <= 1.0:
        parser.error("--noise-aug-strength must be between 0 and 1")
    return args


def compose_image(image, mode):
    """Keep the entire source visible in SVD's mandatory 1024x576 frame."""
    if mode == "crop":
        return ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)

    foreground = ImageOps.contain(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
    if mode == "contain-blur":
        backdrop = ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
        backdrop = backdrop.filter(ImageFilter.GaussianBlur(radius=32))
    else:
        backdrop = Image.new("RGB", (WIDTH, HEIGHT), (16, 16, 16))
    offset = ((WIDTH - foreground.width) // 2, (HEIGHT - foreground.height) // 2)
    backdrop.paste(foreground, offset)
    return backdrop


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Free GitHub public runners provide four CPU cores. Avoid oversubscription.
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.set_num_interop_threads(1)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    source_image = load_image(args.image).convert("RGB")
    original_size = list(source_image.size)
    image = compose_image(source_image, args.fit_mode)
    image.save(args.output_dir / "framed_input.png")
    print(f"Input {original_size} -> {WIDTH}x{HEIGHT}, framing={args.fit_mode}", flush=True)

    print(f"Loading {MODEL_ID} on CPU (model weights downloaded as needed)", flush=True)
    print(f"Precision={args.precision}, frames={args.frames}, steps={args.steps}", flush=True)
    if args.frames > 25:
        print("WARNING: SVD-XT was trained on 25 frames. More frames are experimental, "
              "may reduce coherence, and can exceed runner RAM or time limits.", flush=True)
    started = time.monotonic()
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipeline.to("cpu")

    # Diffusers 0.35.2 preprocesses the VAE conditioning image as float32,
    # while _encode_vae_image only moves it to the device. The resulting
    # float32 image and bfloat16 VAE weights cause a convolution dtype error.
    # Match the VAE input to its weight dtype before noise augmentation.
    # Keeping the VAE itself in bf16 also preserves compatible decoding.
    if args.precision == "bf16":
        original_preprocess = pipeline.video_processor.preprocess

        def preprocess_for_vae(*preprocess_args, **preprocess_kwargs):
            conditioning_image = original_preprocess(*preprocess_args, **preprocess_kwargs)
            return conditioning_image.to(dtype=pipeline.vae.dtype)

        pipeline.video_processor.preprocess = preprocess_for_vae

    # Unlike the documentation's GPU example, CPU execution must not enable
    # GPU/CPU offload. Attention and feed-forward slicing lower peak memory.
    pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipeline.enable_attention_slicing(1)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    print("Beginning CPU denoising. This may take hours.", flush=True)
    with torch.inference_mode():
        generated = pipeline(
            image,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            fps=args.fps,
            decode_chunk_size=args.decode_chunk_size,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=args.noise_aug_strength,
            generator=generator,
        ).frames[0]

    video_path = args.output_dir / "video.mp4"
    last_frame_path = args.output_dir / "last_frame.png"
    export_to_video(generated, str(video_path), fps=args.fps)
    generated[-1].save(last_frame_path)

    metadata = {
        "version": "v4",
        "model": MODEL_ID,
        "device": "cpu",
        "precision": args.precision,
        "image": args.image,
        "original_size": original_size,
        "fit_mode": args.fit_mode,
        "width": WIDTH,
        "height": HEIGHT,
        "frames": len(generated),
        "steps": args.steps,
        "fps": args.fps,
        "seed": args.seed,
        "motion_bucket_id": args.motion_bucket_id,
        "noise_aug_strength": args.noise_aug_strength,
        "decode_chunk_size": args.decode_chunk_size,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {video_path} and {last_frame_path}", flush=True)


if __name__ == "__main__":
    main()
