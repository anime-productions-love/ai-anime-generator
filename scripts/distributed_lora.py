#!/usr/bin/env python3
"""Prepare and verify external distributed CPU character-LoRA training (v1).

This is a handoff and result acceptance adapter, not a fake local trainer.
The separately managed multi-runner trainer consumes immutable Git sources,
performs CPU model-state sharding, and publishes weights + provenance to a
private Hugging Face repo. Only an actual verified result is registered.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from library import ROOT, pack_path, read_pack, save_manifest, stamp, validate_id

MODEL = "cagliostrolab/animagine-xl-3.1"
MODEL_REVISION = "483f0c322568ed13697ed01dd0be07204746d12b"
WEIGHTS_FILE = "pytorch_lora_weights.safetensors"
PROVENANCE_FILE = "training_provenance-v1.json"
REPO = "anime-productions-love/ai-anime-generator"
ID = re.compile(r"^lora-[0-9]{1,24}$")
HUB = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
HASH = re.compile(r"^[a-f0-9]{64}$")


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def request_path(request_id):
    if not ID.fullmatch(request_id or ""):
        raise ValueError("Request ID must look like lora-<GitHub-run-id>")
    return ROOT / "training" / "requests" / (request_id + ".json")


def read_request(request_id):
    path = request_path(request_id)
    if not path.is_file():
        raise ValueError(f"Training request not found: {path.relative_to(ROOT)}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("contract_version") != "v1" or data.get("request_id") != request_id:
        raise ValueError("Training request contract/version mismatch")
    return data, sha256(path.read_bytes())


def snapshot():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()


def references(character):
    base = pack_path("character", character["id"]).resolve()
    images = []
    seen = set()
    for record in character.get("references", []):
        path = (base / record["path"]).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"Reference missing or unsafe: {record['path']}")
        payload = path.read_bytes()
        digest = sha256(payload)
        if digest != record["sha256"]:
            raise ValueError(f"Character reference modified: {record['path']}")
        if digest in seen:
            continue
        seen.add(digest)
        images.append({
            "repo_path": path.relative_to(ROOT).as_posix(),
            "sha256": digest, "role": record["role"],
            "size_bytes": len(payload),
        })
    if len(images) < 3:
        raise ValueError("At least three distinct saved character reference images are required")
    return images


def prepare(args):
    validate_id(args.character_id)
    if not HUB.fullmatch(args.hf_repo or ""):
        raise ValueError("--hf-repo must be namespace/model-name")
    if not 100 <= args.training_steps <= 1200:
        raise ValueError("Training steps must be 100-1200")
    if args.resolution not in (256, 384, 512, 640, 768):
        raise ValueError("Use one of 256, 384, 512, 640 or 768 resolution")
    if not 4 <= args.lora_rank <= 32:
        raise ValueError("LoRA rank must be 4-32")
    if not 2 <= args.workers <= 20:
        raise ValueError("Distributed worker count must be 2-20")
    path = request_path(args.request_id)
    if path.exists():
        raise ValueError(f"Training request already exists: {args.request_id}; not overwriting")

    character = read_pack("character", args.character_id)
    records = references(character)
    source_commit = snapshot()
    dataset = {
        "source_repository": REPO,
        "source_commit": source_commit,
        "character_id": character["id"],
        "character_revision": character["revision"],
        "images": records,
    }
    dataset_sha = sha256(encoded(dataset))
    request = {
        "contract_version": "v1",
        "request_id": args.request_id,
        "status": "prepared",
        "created_at": stamp(),
        "source_repository": REPO,
        "source_commit": source_commit,
        "dataset": dataset,
        "dataset_sha256": dataset_sha,
        "base_model": MODEL,
        "base_model_commit": MODEL_REVISION,
        "output": {
            "hub_repository": args.hf_repo,
            "private": True,
            "weights_file": WEIGHTS_FILE,
            "provenance_file": PROVENANCE_FILE,
        },
        "training": {
            "device": "cpu",
            "distributed_backend": "gloo",
            "partitioning": "fsdp-full-shard-or-equivalent-true-model-state-sharding",
            "world_size": args.workers,
            "model_component": "unet-lora",
            "text_encoders": "frozen",
            "rank": args.lora_rank,
            "max_train_steps": args.training_steps,
            "resolution": args.resolution,
            "batch_size_per_rank": 1,
            "gradient_accumulation_steps": 4,
            "optimizer": "adamw",
            "precision": "fp32-or-supported-cpu-bf16",
            "checkpoint_interval_steps": 50,
            "resumable_across_github_job_timeouts": True,
        },
    }
    save(path, request)
    print(
        f"Prepared {path.relative_to(ROOT)} from {len(records)} verified images; "
        f"dataset_sha256={dataset_sha}, target {args.workers} CPU workers. "
        "Training NOT started: the external distributed network must consume "
        "this request and produce a verified result.",
        flush=True,
    )


def inspect_safetensors(path):
    """Validate file envelope/shape without importing PyTorch or executing pickle."""
    size = path.stat().st_size
    if size < 1_000_000 or size > 1024 * 1024 * 1024:
        raise ValueError("LoRA weights outside plausible 1 MiB–1 GiB bounds")
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated safetensors file")
        (header_bytes,) = struct.unpack("<Q", length_bytes)
        if not 64 <= header_bytes <= 16 * 1024 * 1024 or header_bytes + 8 >= size:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(header_bytes).decode("utf-8"))
    tensors = [(key, value) for key, value in header.items() if key != "__metadata__"]
    if not tensors or not any("lora" in key.lower() for key, _ in tensors):
        raise ValueError("Output contains no recognizable LoRA tensors")
    data_size = size - 8 - header_bytes
    for key, entry in tensors:
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Missing tensor offsets for {key}")
        start, end = offsets
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in offsets):
            raise ValueError(f"Invalid tensor offsets for {key}")
        if start < 0 or end <= start or end > data_size:
            raise ValueError(f"Out-of-bounds tensor {key}")
    return {"tensors": len(tensors), "sha256": sha256(path.read_bytes()), "size_bytes": size}


def finalize(args):
    request, request_hash = read_request(args.request_id)
    if args.character_id != request["dataset"]["character_id"]:
        raise ValueError("Requested character ID does not match immutable training request")
    if args.hf_repo != request["output"]["hub_repository"]:
        raise ValueError("HF model ID does not match immutable training request")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("The secret HF_TOKEN is required to verify private weights")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    info = api.model_info(args.hf_repo, files_metadata=False)
    if not info.private:
        raise ValueError("Cannot register a public model: LoRA destination must be private")
    commit = info.sha
    if not HASH.fullmatch(commit or ""):
        raise ValueError("Hugging Face did not return an immutable model commit")
    filenames = {item.rfilename for item in info.siblings}
    if not {WEIGHTS_FILE, PROVENANCE_FILE}.issubset(filenames):
        raise ValueError("Trainer did not publish both weights and provenance")
    provenance_path = hf_hub_download(
        args.hf_repo, filename=PROVENANCE_FILE, revision=commit, token=token
    )
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    expected = {
        "contract_version": "v1",
        "request_id": args.request_id,
        "request_sha256": request_hash,
        "dataset_sha256": request["dataset_sha256"],
        "source_commit": request["source_commit"],
        "base_model_commit": request["base_model_commit"],
        "character_id": args.character_id,
        "training_status": "completed",
        "weights_file": WEIGHTS_FILE,
        "model_component": "unet-lora",
    }
    mismatches = [key for key, value in expected.items()
                  if provenance.get(key) != value]
    if mismatches:
        raise ValueError("Training provenance does not match request: " + ", ".join(mismatches))
    if provenance.get("device") != "cpu":
        raise ValueError("Expected CPU training; no GPU-only training reports accepted")
    if provenance.get("distributed_world_size", 0) < request["training"]["world_size"]:
        raise ValueError("Trainer reports fewer CPU ranks than requested")
    if provenance.get("steps_completed", 0) < request["training"]["max_train_steps"]:
        raise ValueError("Trainer has not completed the requested steps")
    if provenance.get("distributed_backend") != "gloo":
        raise ValueError("Expected PyTorch Gloo backend")
    if provenance.get("partitioning") != request["training"]["partitioning"]:
        raise ValueError("Trainer did not report required model-state sharding")
    weight_path = hf_hub_download(
        args.hf_repo, filename=WEIGHTS_FILE, revision=commit, token=token
    )
    image = inspect_safetensors(Path(weight_path))

    # Only now update canonical character record, using an immutable model
    # commit rather than a mutable HF repo main pointer.
    character = read_pack("character", args.character_id)
    if character["revision"] < request["dataset"]["character_revision"]:
        raise ValueError("Character manifest older than training snapshot")
    target = f"hf://{args.hf_repo}@{commit}"
    if character["adapters"]["character_lora"] != target:
        character["adapters"]["character_lora"] = target
        character["revision"] += 1
        character["updated_at"] = stamp()
        save_manifest("character", args.character_id, character)
    receipt = {
        "contract_version": "v1",
        "request_id": args.request_id,
        "status": "verified",
        "verified_at": stamp(),
        "character_id": args.character_id,
        "hub_repository": args.hf_repo,
        "hub_commit": commit,
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": image["sha256"],
        "weights_size_bytes": image["size_bytes"],
        "lora_tensor_count": image["tensors"],
        "request_sha256": request_hash,
        "dataset_sha256": request["dataset_sha256"],
        "steps_completed": provenance["steps_completed"],
        "distributed_world_size": provenance["distributed_world_size"],
    }
    path = ROOT / "training" / "receipts" / f"{args.request_id}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("weights_sha256") != receipt["weights_sha256"]:
            raise ValueError("Already finalized request has different model bytes")
        print(f"Receipt already exists for {args.request_id} (verified)", flush=True)
        return
    save(path, receipt)
    print(
        f"Verified {args.request_id}: {image['tensors']} LoRA tensors from "
        f"{args.hf_repo}@{commit}. Character manifest updated.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("prepare", "finalize"))
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--training-steps", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "finalize": finalize}[args.operation](args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Distributed LoRA request failed: {exc}\n")


if __name__ == "__main__":
    main()
