#!/usr/bin/env python3
"""Reference-based image continuity evaluation for storyboard candidates (v1).

CLIP cosine similarity is a comparative visual signal only, not an
identity verification, story-consistency guarantee, or security gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    source = plan.get("ip_adapter_references") or []
    if not source and plan.get("conditioning_image"):
        from_path = plan["conditioning_image"]
        src = (root / from_path).resolve()
        if not src.is_relative_to(root.resolve()) or not src.is_file():
            raise ValueError("Missing previous-shot reference image")
        source = [{
            "path": from_path, "kind": "previous-shot",
            "asset_id": plan.get("continuity_from_shot"),
            "role": "canonical", "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        }]
    references = []
    for item in source:
        p = (root / item["path"]).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError(f"Reference missing or unsafe: {item['path']}")
        if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Reference hash mismatch: {item['path']}")
        references.append((item, p))

    files = [args.candidate_dir / f"candidate-{n:02d}.png"
             for n in range(1, plan["candidates"] + 1)]
    if any(not f.is_file() for f in files):
        raise ValueError("Cannot evaluate without every candidate")
    report = {
        "version": "v1",
        "method": "CLIP ViT-B/32 normalized image-embedding cosine similarity",
        "purpose": "Comparative visual similarity only; NOT facial identity or prompt accuracy",
        "reference_count": len(references),
        "candidates": [],
        "recommendation": None,
    }
    if references:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(min(4, os.cpu_count() or 4))
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", use_safetensors=True
        ).to("cpu").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def embedding(path):
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                vector = model.get_image_features(**inputs)
                # Some transformers versions return an object with .pooler_output.
                if hasattr(vector, "pooler_output"):
                    vector = vector.pooler_output
                if not isinstance(vector, torch.Tensor):
                    raise TypeError("Unsupported CLIP image features output")
                vector = vector.float()
                return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        reference_vectors = [(item, embedding(p)) for item, p in references]
        for n, path in enumerate(files, 1):
            vector = embedding(path)
            similarities = [{
                "reference_kind": item["kind"],
                "reference_asset_id": item["asset_id"],
                "reference_role": item["role"],
                "reference_path": item["path"],
                "cosine_similarity": round(float((vector * ref).sum().item()), 6),
            } for item, ref in reference_vectors]
            mean_score = sum(x["cosine_similarity"] for x in similarities) / len(similarities)
            report["candidates"].append({
                "number": n,
                "mean_cosine_similarity": round(mean_score, 6),
                "reference_scores": similarities,
            })
        # Ranking is advisory. The user still picks the canonical image;
        # never automatically replace a shot based on CLIP similarity alone.
        report["recommendation"] = max(
            report["candidates"], key=lambda x: x["mean_cosine_similarity"]
        )["number"]
    else:
        report["method"] = "not evaluated: no saved visual references"
        report["candidates"] = [
            {"number": number, "mean_cosine_similarity": None, "reference_scores": []}
            for number in range(1, len(files) + 1)
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Continuity evaluation: {len(references)} reference images, "
          f"recommended candidate {report['recommendation']}", flush=True)


if __name__ == "__main__":
    main()
