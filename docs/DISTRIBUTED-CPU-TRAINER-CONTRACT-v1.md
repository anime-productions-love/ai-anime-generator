# Distributed CPU LoRA Trainer Integration Contract — v1

## Purpose and ownership

The anime image generator no longer requires a self-hosted GPU. It **prepares** a cryptographically pinned character training request, waits for the separately built multi-GitHub-runner CPU training network, then **verifies and registers** the resulting LoRA.

**This repository does not implement the network coordinator, transport, scheduler, checkpoint store, or cross-runner training engine.** Those belong to the parallel aggregation project. This specification is the exact compatibility contract for that agent. Reuse its existing capabilities rather than create a second coordinator.

Four GitHub CPU runners are **four separate machines**. Their 64 GiB aggregate RAM does not become one address space. The network must implement genuine model/optimizer/gradient sharding (e.g. FSDP full sharding with Gloo where supported) or equivalent pipeline/tensor partitioning, with reliable rendezvous, communication, timeouts and recovery. Simply launching the upstream DreamBooth training script once per runner will not combine usable model memory and will not satisfy this contract.

The number of workers is a **requested distributed configuration**, not an assertion that the external network has that many concurrently available hosts or that this model will train. An end-to-end CPU-only SDXL LoRA run has **not been validated**. No GPU is required or assumed by the request interface.

## Source of truth

- GitHub repository: `anime-productions-love/ai-anime-generator`.
- Training request: `training/requests/lora-<prepare-workflow-run-id>.json`, committed by [Train Character LoRA (distributed CPU)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/train-character-lora.yml).
- Immutable reference snapshot: `source_repository` + `source_commit`, with each saved reference's `repo_path` and `sha256`.
- Dataset digest: `dataset_sha256` = SHA-256 of canonical JSON `request.dataset` using UTF-8 serialization with sorted keys, no spaces and a terminal newline (`scripts/distributed_lora.py:encoded`).
- Training destination: the request's private `output.hub_repository`.
- Training request and character manifest are Git-backed records; model weights are **not** committed to the public GitHub repo.

## Coordinator intake sequence (required)

1. Detect or retrieve a prepared `training/requests/lora-<id>.json`.
2. Require `contract_version: "v1"` and `status: "prepared"`. Capture the **SHA-256 of the entire JSON file**, including its final newline, as `request_sha256`.
3. Read the pinned `source_commit`, checkout that exact source tree for the training data, and verify every listed `repo_path` stays inside `assets/characters/<character_id>/` and its bytes match the recorded `sha256`.
4. Materialize the training dataset with `python3 scripts/prepare_lora_dataset.py --request-file <REQUEST_JSON> --destination <TRAINING_DIRECTORY>`, using the installed script from this repo. It copies exactly the reference images listed in the request and verifies their SHA-256.
5. Train an SDXL **UNet-only LoRA** against the model `cagliostrolab/animagine-xl-3.1` at the exact `base_model_commit`. Keep text encoders frozen. The request includes target rank, steps, resolution, batch size, accumulation and CPU-only `gloo` backend. Use genuine state sharding `fsdp-full-shard-or-equivalent-true-model-state-sharding`; data parallelism alone is not enough.
6. Persist rank-consistent checkpoints and optimizer state outside ephemeral GitHub runners, so jobs can resume after GitHub's per-job time limit; never claim completion on the basis of a saved checkpoint or a submitted job.
7. After training **actually completes**, export Diffusers-compatible LoRA tensors named **`pytorch_lora_weights.safetensors`**, with ordinary LoRA tensor keys (not a model-specific tensor format that Diffusers cannot load).
8. Upload the weights **and** the provenance JSON below to the *private* Hugging Face destination repository, using its own credential management. Refuse a public destination.
9. Only after upload and remote file verification may the coordinator report the training result as ready for this repo's `finalize` action.

The coordinator must not edit `assets/characters/<id>/manifest.json`, bypass the finalizer, or register an unverified model; those remain the anime generator's responsibility.

## Required remote provenance: `training_provenance-v1.json`

The coordinator must publish a UTF-8 JSON object alongside the model weights with **at least** these fields:

```json
{
  "contract_version": "v1",
  "request_id": "lora-EXAMPLE_RUN_ID",
  "request_sha256": "<sha256 of committed request JSON bytes>",
  "dataset_sha256": "<request.dataset_sha256>",
  "source_commit": "<request.source_commit>",
  "base_model_commit": "483f0c322568ed13697ed01dd0be07204746d12b",
  "character_id": "aria-001",
  "training_status": "completed",
  "weights_file": "pytorch_lora_weights.safetensors",
  "model_component": "unet-lora",
  "device": "cpu",
  "distributed_backend": "gloo",
  "distributed_world_size": 4,
  "partitioning": "fsdp-full-shard-or-equivalent-true-model-state-sharding",
  "steps_completed": 300
}
```

Substitute **real**, request-bound values. These are schema examples, not fabricated completed evidence. The value of `distributed_world_size` must be at least `request.training.world_size`, and `steps_completed` at least `request.training.max_train_steps`. A truthful trainer may write extra diagnostic fields like `engine_version`, `checkpoint_id`, `optimizer`, `training_duration_seconds`, and per-rank peak memory.

**Output requirements:** a private Hugging Face repository containing `pytorch_lora_weights.safetensors` and `training_provenance-v1.json`. The anime generator uses the model's **immutable Hugging Face commit SHA** for generation, not the mutable `main` pointer.

## Finalization

After the network has actually completed the request, run the **same** GitHub Actions workflow again with `operation=finalize`, the original `character_id`, `hf_model_repo`, and `request_id` from the preparation run. Set GitHub secret `HF_TOKEN` to permit reading the private model repository; preparation does not require this secret.

The installed `scripts/distributed_lora.py` finalizer:

- checks that the request file exists and that the character/HF destination matches it;
- authenticates to Hugging Face and confirms the destination model is private;
- pins the remote revision to its immutable model commit;
- downloads provenance and compares request SHA, dataset SHA, character, source commit, base-model revision, CPU backend, world size, partitioning and completed steps;
- downloads weights and validates the safetensors envelope, per-tensor bounds and presence of LoRA tensor names without loading untrusted pickle;
- only then sets the character's `adapters.character_lora` to `hf://namespace/model@<immutable-commit>`;
- creates `training/receipts/<request-id>.json` and commits the verified receipt and character manifest together.

It does **not** prove the trainer's claims cryptographically or evaluate whether the learned LoRA is high quality: a trusted coordinator or model uploader could falsify provenance. End-to-end training verification, runner usage, and inference quality must be established from actual network logs and generated images.

## Failure/continuation policy

- Missing network connectivity or rank failure: the aggregator owns recovery; don't declare training complete.
- Runner expiration: resume from verified shard+optimizer checkpoint; do not restart at step 0 unless justified.
- Partial output/upload: **do not finalize**.
- Output/model privacy mismatch: **refuse registration**.
- Any mismatched dataset/model/weights/provenance: **refuse registration**.
- Request remains prepared until explicitly finalized; no asynchronous training is launched by this repository's prepare workflow.
- Do not put tokens, IP addresses, auth headers, or signed checkpoint URLs in this public repo's request or provenance manifests.
- No tests or live campaigns should be created or run unless separately authorized; this document defines integration, not training execution.

## Exact installed interfaces

```bash
# Phase 1 saved characters and references must already exist.
python3 scripts/distributed_lora.py --operation prepare \
  --request-id lora-123456789 \
  --character-id aria-001 \
  --hf-repo owner/aria-animagine-lora \
  --workers 4 --training-steps 300 --resolution 512 --lora-rank 8

# Run only AFTER the separate trainer actually completes and uploads.
python3 scripts/distributed_lora.py --operation finalize \
  --request-id lora-123456789 \
  --character-id aria-001 \
  --hf-repo owner/aria-animagine-lora
```

The GitHub Actions wrapper generates the request ID automatically during `prepare` and commits updates using existing repository write permissions. `finalize` uses the explicit earlier request ID.

**External network integration point:** one committed JSON request per character training session; no new coordinator, broker, or cloud networking service is created by the anime repository.
