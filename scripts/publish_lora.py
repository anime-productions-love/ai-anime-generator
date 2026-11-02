#!/usr/bin/env python3
"""Upload a finished SDXL LoRA to a private HF model repo and verify presence (v2)."""
import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    repo_id = args.hf_repo
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        parser.error("Hugging Face destination must be namespace/model-name")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", args.character_id):
        parser.error("Invalid character ID")
    if not args.weights.is_file() or args.weights.stat().st_size < 1_000_000:
        parser.error("LoRA weights missing or implausibly small; refusing to register")
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("GitHub secret HF_TOKEN required for model upload and later inference")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    # create_repo(exist_ok=True) does not make an existing public repo private.
    # Never accidentally publish a character's private LoRA weights.
    before = api.model_info(repo_id, files_metadata=False)
    if not before.private:
        parser.error(
            "Destination model repository is public. Choose a new private repo "
            "or manually make this one private before uploading."
        )
    api.upload_file(
        path_or_fileobj=str(args.weights), path_in_repo="pytorch_lora_weights.safetensors",
        repo_id=repo_id, repo_type="model", commit_message="Publish character LoRA weights v1",
    )
    provenance = args.weights.parent.parent / "training-manifest.json"
    if provenance.is_file():
        api.upload_file(
            path_or_fileobj=str(provenance), path_in_repo="training-manifest-v1.json",
            repo_id=repo_id, repo_type="model", commit_message="Add training provenance v1",
        )
    model_info = api.model_info(repo_id, files_metadata=False)
    siblings = {file.rfilename for file in model_info.siblings}
    if "pytorch_lora_weights.safetensors" not in siblings:
        raise RuntimeError("Hugging Face model upload was not verified; do not register")
    print(f"Verified LoRA weights in Hugging Face model {repo_id}; private storage")


if __name__ == "__main__":
    main()
