#!/usr/bin/env python3
"""Build reusable IP-Adapter Plus SDXL reference image embeddings (v1).

Writes safe tensor weights rather than pickle files. A cache is tied to the
exact image SHA-256, selected adapter checkpoint, and CFG-enabled batch shape.
"""
import argparse
import hashlib
import os
from pathlib import Path

from PIL import Image
from safetensors.torch import save_file
from library import ROOT, read_pack, pack_path
from generate_shot import MODEL, MODEL_REVISION, IP_ADAPTER_REPO, IP_WEIGHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("character", "scene"))
    parser.add_argument("--id", required=True)
    parser.add_argument("--role", default="main")
    args = parser.parse_args()

    record = read_pack(args.kind, args.id)
    refs = [r for r in record["references"] if r["role"] == args.role]
    if not refs:
        raise ValueError(f"No saved {args.role} reference for {args.kind}/{args.id}")
    # Only the first reference with this role is selected. Other roles
    # and images can be cached with their own invocation.
    ref = refs[0]
    base = pack_path(args.kind, args.id).resolve()
    path = (base / ref["path"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("Unsafe reference path")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != ref["sha256"]:
        raise ValueError("Reference image checksum mismatch")

    import torch
    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    dtype = torch.bfloat16
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO, subfolder="models/image_encoder", torch_dtype=dtype
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=dtype,
        use_safetensors=True, low_cpu_mem_usage=True, image_encoder=encoder,
    )
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=IP_WEIGHT
    )
    pipe.to("cpu")
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    with torch.inference_mode():
        embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=image, ip_adapter_image_embeds=None,
            device="cpu", num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
    payload = {
        f"embed_{i}": tensor.detach().to("cpu").contiguous()
        for i, tensor in enumerate(embeds)
    }
    destination = base / "embeds" / f"{sha[:20]}-ip-plus-v1.safetensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(destination), metadata={
        "version": "v1",
        "reference_sha256": sha,
        "ip_adapter_repo": IP_ADAPTER_REPO,
        "ip_adapter_weight": IP_WEIGHT,
        "base_model": MODEL,
        "model_revision": MODEL_REVISION,
        "classifier_free_guidance": "true",
    })
    print(f"Saved reusable IP-Adapter embedding cache: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
