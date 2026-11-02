#!/usr/bin/env python3
"""Animagine XL CPU story illustration inference with optional Phase 3 guides (v6).

Guidance is opt-in to preserve Phase 2's low-memory baseline:
  base         one img2img reference or text-to-image
  ip-adapter   multi-character/scene references, one SDXL IP-Adapter
  controlnet   canny structural guide from saved scene layout or prior shot
  both         both reference and structural conditioning (high RAM)
Registered character LoRAs are loaded from canonical HF Hub repositories.
"""
import argparse
import json
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

MODEL = "cagliostrolab/animagine-xl-3.1"
MODEL_REVISION = "483f0c322568ed13697ed01dd0be07204746d12b"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
CONTROL_MODEL = "diffusers/controlnet-canny-sdxl-1.0"


def fit_reference(image, width, height):
    source = Image.open(image).convert("RGB")
    fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    base = ImageOps.fit(source, (width, height), Image.Resampling.LANCZOS)
    base = base.filter(ImageFilter.GaussianBlur(radius=max(12, width // 36)))
    base.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return base


def verified_image(root, item):
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Reference is missing or unsafe: {item['path']}")
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"Reference has changed: {item['path']}")
    return path


def edge_guide(image):
    # Saved role canny is treated as a preprocessed edge guide. For a
    # layout/background illustration, compute actual Canny edges.
    import cv2
    import numpy as np
    arr = np.asarray(image.convert("RGB"))
    edges = cv2.Canny(arr, 100, 200)
    stacked = np.repeat(edges[:, :, None], 3, axis=2)
    return Image.fromarray(stacked, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "v1":
        parser.error("Unsupported shot plan")
    mode = plan.get("guidance_mode", "base")
    if mode not in ("base", "ip-adapter", "controlnet", "both"):
        parser.error("Unknown guidance mode")

    root = Path(__file__).resolve().parents[1]
    use_ip = mode in ("ip-adapter", "both")
    use_control = mode in ("controlnet", "both")
    ip_refs = plan.get("ip_adapter_references", [])
    if use_ip and not ip_refs:
        parser.error("IP-Adapter requested, but chosen characters/scene have no reference images.")
    # For ControlNet we can derive the edges from the prior selected shot,
    # or fall back to an explicitly stored layout reference.
    control_record = plan.get("control_reference")
    if use_control and not (control_record or plan.get("conditioning_image")):
        parser.error("ControlNet requested but no previous shot or scene layout reference exists.")

    import torch
    from diffusers import (
        ControlNetModel, StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        DPMSolverMultistepScheduler,
    )

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.set_num_interop_threads(1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    dtype = torch.bfloat16

    print(f"Loading {MODEL} (CPU, {dtype}), guidance={mode}", flush=True)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=dtype,
        use_safetensors=True, low_cpu_mem_usage=True, add_watermarker=False,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True
    )
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()
    pipeline.enable_attention_slicing(1)

    # Use only explicit, registered LoRA repos. Do not silently infer weights
    # from a free-form prompt or download a raw executable/pickle checkpoint.
    lora_names = []
    for number, item in enumerate(plan.get("character_loras", []), 1):
        repo_id = item["repository"]
        adapter_name = f"character_{number}"
        pipeline.load_lora_weights(
            repo_id, weight_name="pytorch_lora_weights.safetensors",
            adapter_name=adapter_name,
            revision=item.get("revision"),
            use_safetensors=True,
        )
        lora_names.append(adapter_name)
    if lora_names:
        pipeline.set_adapters(lora_names, adapter_weights=[0.75] * len(lora_names))
        print(f"Loaded {len(lora_names)} registered character LoRA(s)", flush=True)

    cached_ip_embeds = None
    if use_ip:
        # Reuse SHA-bound safe-tensor cache for one reference; multiple
        # references still use the canonical Diffusers image encoder.
        if len(ip_refs) == 1:
            from safetensors import safe_open
            from safetensors.torch import load_file
            ref = ip_refs[0]
            name = "characters" if ref["kind"] == "character" else "scenes"
            cached_path = (
                root / "assets" / name / ref["asset_id"] / "embeds" /
                f"{ref['sha256'][:20]}-ip-plus-v1.safetensors"
            )
            if cached_path.is_file():
                with safe_open(str(cached_path), framework="pt", device="cpu") as sf:
                    meta = sf.metadata() or {}
                if (meta.get("reference_sha256") == ref["sha256"] and
                    meta.get("ip_adapter_weight") == IP_WEIGHT and
                    meta.get("base_model") == MODEL and
                    meta.get("model_revision") == MODEL_REVISION and
                    meta.get("classifier_free_guidance") == "true"):
                    data = load_file(str(cached_path), device="cpu")
                    cached_ip_embeds = [data[k] for k in sorted(data) if k.startswith("embed_")]
                    if not cached_ip_embeds:
                        raise RuntimeError("Cached reference contains no image embeddings")
                    print(f"Reusing verified cache: {cached_path.relative_to(root)}", flush=True)
        if cached_ip_embeds is None:
            from transformers import CLIPVisionModelWithProjection
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                IP_ADAPTER_REPO, subfolder="models/image_encoder",
                torch_dtype=dtype,
            )
            pipeline.register_modules(image_encoder=encoder)
        pipeline.load_ip_adapter(
            IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=IP_WEIGHT,
            image_encoder_folder=None if cached_ip_embeds is not None else "models/image_encoder",
        )
        pipeline.set_ip_adapter_scale(0.55)
        print(f"IP-Adapter: {len(ip_refs)} saved character/scene references", flush=True)

    if use_control:
        controlnet = ControlNetModel.from_pretrained(
            CONTROL_MODEL, torch_dtype=dtype, use_safetensors=True
        )
        if plan.get("conditioning_image"):
            active = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
                pipeline, controlnet=controlnet
            )
        else:
            active = StableDiffusionXLControlNetPipeline.from_pipe(
                pipeline, controlnet=controlnet
            )
        print("Enabled SDXL Canny ControlNet", flush=True)
    elif plan.get("conditioning_image"):
        active = StableDiffusionXLImg2ImgPipeline.from_pipe(pipeline)
    else:
        active = pipeline
    # On a CPU-only runner do not use CUDA offload; memory use may exceed 16 GiB
    active.to("cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    init_image = None
    conditioning = plan.get("conditioning_image")
    if conditioning:
        source = (root / conditioning).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            raise RuntimeError(f"Invalid conditioning image: {conditioning}")
        init_image = fit_reference(source, plan["width"], plan["height"])
        init_image.save(args.output_dir / "conditioning_input.png")

    ip_images = []
    if use_ip and cached_ip_embeds is None:
        for ref in ip_refs:
            p = verified_image(root, ref)
            with Image.open(p) as source:
                ip_images.append(source.convert("RGB"))
        ip_images = [ip_images] if len(ip_images) > 1 else ip_images[0]

    canny = None
    if use_control:
        if control_record:
            path = verified_image(root, control_record)
            with Image.open(path) as source:
                control_source = source.convert("RGB")
            # Structural guides must not be blurred as if they were
            # conditioning artwork. Preserve the full map in black padding.
            control_fitted = ImageOps.contain(
                control_source, (plan["width"], plan["height"]),
                Image.Resampling.LANCZOS,
            )
            control_canvas = Image.new("RGB", (plan["width"], plan["height"]), (0, 0, 0))
            control_canvas.paste(
                control_fitted,
                ((plan["width"] - control_fitted.width) // 2,
                 (plan["height"] - control_fitted.height) // 2),
            )
            canny = control_canvas if control_record["role"] == "canny" else edge_guide(control_canvas)
        else:
            canny = edge_guide(init_image)
        canny.save(args.output_dir / "control_canny.png")

    for number in range(1, plan["candidates"] + 1):
        seed = plan["seed"] + number - 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        options = dict(
            prompt=plan["composed_prompt"],
            negative_prompt=plan["composed_negative_prompt"],
            num_inference_steps=plan["steps"],
            guidance_scale=plan["cfg"],
            generator=generator,
            num_images_per_prompt=1,
            output_type="pil",
        )
        if init_image is not None:
            options.update(image=init_image, strength=plan["strength"])
        elif not use_control:
            options.update(width=plan["width"], height=plan["height"])
        if canny is not None:
            if init_image is not None:
                options["control_image"] = canny
            else:
                options["image"] = canny
                options.update(width=plan["width"], height=plan["height"])
            options["controlnet_conditioning_scale"] = 0.65
        if use_ip:
            if cached_ip_embeds is not None:
                options["ip_adapter_image_embeds"] = cached_ip_embeds
            else:
                options["ip_adapter_image"] = ip_images
        print(f"Generating candidate {number}/{plan['candidates']} "
              f"with {mode}, seed={seed}", flush=True)
        with torch.inference_mode():
            image = active(**options).images[0]
        destination = args.output_dir / f"candidate-{number:02d}.png"
        image.save(destination)
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
