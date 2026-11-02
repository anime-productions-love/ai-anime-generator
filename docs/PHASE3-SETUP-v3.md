# Phase 3 — Anime Identity, Pose and Continuity System (v3)

## Implementation status

Phase 3 source and GitHub Actions workflows are installed in the repository. **No image-generation, distributed CPU training, embedding-building, or similarity-scoring run has been executed as part of this revision.** Source commits and manifests are not proof that the full pipeline fits inside the free GitHub runner.

This phase extends the existing Phase 1 character/scene library and Phase 2 story-shot generator **without replacing their canonical records**.

## New capabilities

1. **Character LoRAs:** character manifests can register a trained SDXL LoRA model from Hugging Face, and the image generator loads registered adapters when the character appears in a shot.
2. **Reference image guidance (IP-Adapter Plus SDXL):** can use up to four saved character/scene images as image-level conditioning, not merely their text descriptions.
3. **Structural control (Canny ControlNet SDXL):** uses a scene's saved `canny`, `layout`, or `edges` reference, or extracts Canny edges from the previous canonical shot, to constrain composition.
4. **Cached reference embeddings:** a dedicated workflow can save a single image's IP-Adapter embedding as safe-tensor data in its canonical character/scene asset pack, keyed by the original image SHA-256.
5. **Continuity evaluation:** each generation produces a per-candidate CLIP image-embedding cosine-similarity report against saved references. Rankings are advisory; the first candidate remains initially canonical until you explicitly change it using the existing selector.

## Generate or continue a shot with Phase 3 guidance

Open [Generate or continue anime story shot (v3)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml), then **Run workflow**.

| Input | Example | Use |
|---|---|---|
| `mode` | `start` / `continue` | New sequence or next storyboard illustration |
| `story_id` | `story-001` | Persistent storyline identifier |
| `sequence_id` | `seq-001` | Persistent sequence identifier |
| `character_ids` | `aria-001` | Comma-separated character IDs, blank to inherit on continue |
| `scene_id` | `moon-forest` | Existing setting, blank to inherit on continue |
| `shot_prompt` | `Aria turns toward the distant castle, determined expression` | Action/emotion/camera |
| `size` | `512x768` | Width and height combined to fit GitHub's ten-field workflow limit |
| `guidance_mode` | `base` | See options below |
| `candidates` | `1` | One to four separate images |
| `seed` | `42` | Reproducibility seed |

**Available guidance modes:**

| Mode | Runtime behavior | Practical caveat |
|---|---|---|
| `base` | Original Phase 2 image-to-image/text-to-image plus any registered LoRAs | Least additional RAM; best starting point |
| `ip-adapter` | Phase 2 conditioning + saved character/scene image references via one IP-Adapter Plus model | Requires at least one saved reference; extra image encoder can exceed 16 GB |
| `controlnet` | Phase 2 conditioning + SDXL Canny structural control | Requires previous selected shot or saved scene layout; downloads another large neural network |
| `both` | IP-Adapter + ControlNet together | Most resource-intensive and likely unsuitable for the free 16 GB CPU runner |

For character identities, store images with roles like `face`, `three_quarter`, `full_body`, `main`. For a scene, use `main`, `background` for appearance, and `layout`, `edges`, or `canny` for composition control. Phase 3 collects one preferred reference per character and one environment reference (up to four total). It is not a spatially masked multi-character identity guarantee.

**ControlNet note:** `canny` references are assumed to be **already-produced Canny edge maps**. `layout` or `edges` references are treated as normal illustrations and converted to Canny edges at generation. That edge processing is not a pose-estimation or skeleton detector; it constrains visible edges and composition, not anatomically rigged poses.

If there are no usable visual references, `ip-adapter` fails early instead of pretending that image conditioning occurred. The `base` option remains usable with no saved assets.

## Train and save a character LoRA — distributed CPU path

The former **self-hosted GPU-only trainer has been replaced** by the [Prepare or finalize distributed CPU character LoRA (v2)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/train-character-lora.yml) workflow.

This is a **request/finalization interface** for the multi-workflow CPU training network under construction by a separate agent. It does **not** itself train the LoRA, launch multi-runner jobs, or make RAM from different GitHub runners automatically shared. The external network must implement actual model-state sharding and checkpointing before training can begin.

### First operation — prepare

Run the training workflow with:
- `operation=prepare`
- an existing `character_id` with at least **three distinct saved reference images**
- `hf_model_repo=namespace/model-name` for a future **private** Hugging Face model repository
- `training_steps` (100–1200; default 300), `resolution` (256/384/512/640/768), `lora_rank` (4–32; default 8)
- `workers` = 4 by default, meaning **four independently coordinated CPU ranks** rather than pooled virtual RAM
- leave `request_id` blank

On success the workflow commits `training/requests/lora-<run-id>.json`. The request stores the canonical character identity, exact image source paths and SHA-256 hashes, base model revision, CPU/Gloo and true sharding requirements, checkpoint policy, private model destination and a repeatable dataset fingerprint. It explicitly reports **TRAINING NOT STARTED**.

The new training interface needs **no GPU hardware**. However, it cannot do actual distributed training until the other agent's cross-runner coordinator, transport, sharded execution and checkpoint storage are operational.

### External training

The separate network reads the committed request and obtains the exact dataset from Git. It can materialize the validated images using:

```bash
python3 scripts/prepare_lora_dataset.py --request-file <request.json> --destination work/character-training
```

The network must then run real CPU-distributed SDXL UNet LoRA optimization, checkpoint across runner timeouts, and publish weights and a request-bound provenance file to the specified **private** Hugging Face model repository.

Do not execute a GPU-only DreamBooth script or pretend multiple independently training workers are shared memory. The GPU option is not required. LoRA is retained as the **character personalization format**, but the GPU training implementation has been replaced.

### Final operation — verify and register

Once the network actually completes and uploads the model, run the **same** workflow using `operation=finalize`, with the original `request_id`, `character_id`, and `hf_model_repo`. Provide a GitHub `HF_TOKEN` secret that has read permission for the private model.

The finalizer verifies a completed `training_provenance-v1.json`, the pinned source and dataset hashes, reported CPU world size and steps, the safetensors weight format, and the private model commit. Only then does it register the model in `adapters.character_lora` using `hf://namespace/model@<immutable-commit>` and commit a receipt under `training/receipts/`.

**Read the complete [distributed CPU trainer integration contract v1](DISTRIBUTED-CPU-TRAINER-CONTRACT-v1.md)** for the exact JSON interface the other agent must implement.

## Build reusable reference embeddings

Open [Cache anime reference embedding (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/cache-reference-embedding.yml).

Specify:
- `kind`: `character` or `scene`
- `asset_id`: existing ID such as `aria-001`
- `reference_role`: e.g. `face` or `main`

The workflow reads the **first** saved image with the selected role, verifies its SHA-256 against the canonical asset, loads the matching IP-Adapter Plus image encoder, computes its SDXL guidance embedding, and commits a **safe-tensor** cache named:

```text
assets/characters/aria-001/embeds/<sha20>-ip-plus-v1.safetensors
```

The cache has its original image SHA-256, model/checkpoint identity and CFG-shape metadata. When a subsequent shot uses **exactly one matching IP-Adapter reference**, the image generator can reuse that cache instead of re-encoding the image; when multiple image references are used, it uses live reference-image encoding. The cache workflow runs on an ordinary GitHub CPU runner but can still exceed memory/time limits due to SDXL plus the additional image encoder.

Embeddings are saved using the **safetensors** format rather than pickle to avoid arbitrary Python object deserialization. The original character/scene manifest remains the canonical source for reference identity. Rebuilding a cache creates a Git commit only if its content changes.

## Continuity quality report

The image workflow now runs `scripts/evaluate_continuity.py` after the generator produces candidate PNGs. It loads reference images from the saved library and checks SHA-256 hashes, then compares each generated candidate's CLIP ViT-B/32 image features against each reference.

The saved `work/continuity.json` and shot `continuity_evaluation` report includes:
- method and model family
- per-candidate mean cosine similarity
- per-reference similarity details (source image role, asset ID, reference path)
- suggested candidate number based on mean similarity, or `null` if none

**These numerical similarities are not an identity verification, a face-matching guarantee, an objective art-quality score, or proof of story correctness.** They may reward copying the reference instead of telling the story. No candidate is auto-approved or rejected based solely on a similarity score. The existing canonical-image selector remains the authority for choosing which frame becomes the next shot's reference.

If a shot has no visual references, the report records `not evaluated` rather than inventing a score. The workflow requires the CLIP evaluation step to finish successfully before publishing the shot.

## Installed file map

- `scripts/story.py` — v3, includes immutable Hugging Face LoRA model revision loading.
- `scripts/generate_shot.py` — v6, selects SDXL base/LoRA/IP-Adapter/ControlNet, supports revision-pinned LoRAs.
- `scripts/library.py` — v3, supports model-commit-pinned `register-lora` without changing Phase 1 records.
- `scripts/evaluate_continuity.py` — v1.
- `scripts/build_ip_embeddings.py` — v1.
- `scripts/prepare_lora_dataset.py` — v2, imports pinned distributed CPU training data; `scripts/distributed_lora.py` — v1, request and verification adapter.
- `.github/workflows/generate-story-shot.yml` — v3.
- `.github/workflows/train-character-lora.yml` — v2 (CPU request/verified result only; no local training).
- `.github/workflows/cache-reference-embedding.yml` — v1 (CPU runner, potentially resource-intensive).
- `requirements-image-v2.txt` — v2, includes Phase 3 model conditioning dependencies.
- `schemas/shot-v1.schema.json` — compatible optional Phase 3 fields, with descriptive schema revision v2.

## Scope and limitations

Phase 3 is **not** a guarantee that character faces, costumes or locations remain perfectly consistent across generated frames. The IP-Adapter and ControlNet models introduce large RAM/CPU requirements; `base` is the operational fallback for ordinary free runners. Training requests are CPU-only, with no GPU prerequisite. Actual optimization remains blocked until the external model-sharding network is operational; finalizing a private Hugging Face model requires an HF token. Character visual conditioning from multiple references does not imply each character will remain spatially isolated without masks.

Training preparation only creates a committed request; it does not dispatch or run the independent multi-runner trainer. The cache workflow also does not run automatically. The generator does not perform recurring retraining or update models behind your back. Source changes were committed to GitHub without executing training, image generation, or tests. The first live run is needed to establish actual end-to-end compatibility and resource use.

## Upstream references

- [Diffusers IP-Adapter documentation](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
- [Diffusers ControlNet documentation](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
- [Official SDXL LoRA training script](https://github.com/huggingface/diffusers/blob/v0.35.2/examples/dreambooth/train_dreambooth_lora_sdxl.py)
- [Animagine XL 3.1 model](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
