#!/usr/bin/env python3
"""Generate anime storyboard stills with Animagine XL 3.1 on CPU (v2).

One-shot/sequence continuity is planned by story.py. This module performs
SDXL text-to-image or single-reference image-to-image inference only.
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

MODEL = "cagliostrolab/animagine-xl-3.1"
MODEL_REVISION = "483f0c322568ed13697ed01dd0be07204746d12b"


def fit_reference(image, width, height):
    source = Image.open(image).convert("RGB")
    # Preserve all artwork rather than center-cropping portrait or landscape.
    fitted = ImageOps.contain(
        source, (width, height), method=Image.Resampling.LANCZOS
    )
    base = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
    base = base.filter(ImageFilter.GaussianBlur(radius=max(12, width // 36)))
    base.paste(
        fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2)
    )
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "v1":
        parser.error("Unsupported shot plan")

    # Imports happen after the manifest and referenced assets have been
    # validated, so invalid shots fail before heavyweight model loading.
    import torch
    from diffusers import (
        StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.set_num_interop_threads(1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    dtype = torch.bfloat16
    print(
        f"Loading {MODEL} on CPU using {dtype}; model files downloaded "
        "from Hugging Face. The first run can require many GB of RAM/disk.",
        flush=True,
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL, revision=MODEL_REVISION,
        torch_dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True,
        add_watermarker=False,
    ).to("cpu")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True
    )
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    pipeline.enable_attention_slicing(1)

    root = Path(__file__).resolve().parents[1]
    conditioning = plan.get("conditioning_image")
    if conditioning:
        path = root / conditioning
        if not path.is_file():
            raise RuntimeError(f"Conditioning image is missing: {conditioning}")
        init_image = fit_reference(path, plan["width"], plan["height"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        init_image.save(args.output_dir / "conditioning_input.png")
        img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipeline)
        active = img2img
        print(
            f"Using single-reference img2img: {conditioning}; "
            f"strength={plan['strength']}. Additional references inform "
            "text prompts only until multi-adapter Phase 3.",
            flush=True,
        )
    else:
        active = pipeline
        init_image = None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print("No visual references available; using text-to-image generation.", flush=True)

    prompt = plan["composed_prompt"]
    negative_prompt = plan["composed_negative_prompt"]
    if len(prompt) > 10000 or len(negative_prompt) > 10000:
        raise ValueError("Composed prompt is too long")
    for number in range(1, plan["candidates"] + 1):
        seed = plan["seed"] + number - 1
        if seed >= 2**63:
            raise ValueError("Seed is too large")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        options = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=plan["steps"],
            guidance_scale=plan["cfg"],
            generator=generator,
            num_images_per_prompt=1,
            output_type="pil",
        )
        if init_image is not None:
            options.update(image=init_image, strength=plan["strength"])
        else:
            options.update(width=plan["width"], height=plan["height"])
        print(
            f"Generating candidate {number}/{plan['candidates']}, "
            f"seed {seed}, {plan['width']}x{plan['height']}",
            flush=True,
        )
        with torch.inference_mode():
            output = active(**options).images[0]
        filename = args.output_dir / f"candidate-{number:02d}.png"
        output.save(filename)
        print(f"Created {filename}", flush=True)


if __name__ == "__main__":
    main()
