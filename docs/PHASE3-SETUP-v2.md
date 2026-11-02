# Phase 3 — Anime Identity, Pose and Continuity System (v2)

## Implementation status

Phase 3 source and GitHub Actions workflows are installed in the repository. **No image-generation, GPU-training, embedding-building, or similarity-scoring run has been executed as part of this installation.** Source commits and manifests are not proof that the full pipeline fits inside the free GitHub runner.

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

## Train and save a character LoRA

Open [Train character LoRA (GPU runner) (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/train-character-lora.yml). This invokes the official Hugging Face Diffusers **SDXL DreamBooth LoRA** training script, pinned to the `v0.35.2` upstream release.

**Prerequisites (important):**

- Install/attach an **owner-managed Linux x64 GitHub Actions self-hosted GPU runner with the `gpu` label**, working NVIDIA CUDA drivers, and enough VRAM (typically a modern high-VRAM GPU). Standard GitHub-hosted CPU-only runners do not qualify.
- Add GitHub Actions secret `HF_TOKEN` authorized to create/upload to your Hugging Face namespace. No token is stored in source control.
- Create an existing character record with **at least three distinct saved reference images**. More varied, high-quality images generally yield better identity coverage. Training results are not guaranteed.
- Select a destination Hugging Face model ID `namespace/model-name` to store the adapter privately.

Workflow inputs: `character_id`, `hf_model_repo`, `training_steps` (default 300), `resolution` (default 512), `lora_rank` (default 8).

The publication step explicitly refuses existing public Hugging Face repositories, even if the API call to create a private repository succeeds with `exist_ok=True`.

The workflow follows this dependency sequence: validate reference SHA-256 checksums → prepare the training image directory → run the official upstream training script on the GPU → upload `pytorch_lora_weights.safetensors` to a **private** Hugging Face model repository → read back the model's file listing → register `hf://namespace/model-name` into `adapters.character_lora` in the existing character manifest → commit the manifest update. **Registration must never happen if training or upload fails.**

Trained weights **are not committed to the public Git repository**. A registered private LoRA needs `HF_TOKEN` in the image-generation workflow (supported by the v3 workflow). The current adapter loader expects the exact file name `pytorch_lora_weights.safetensors`. Training is planned using Animagine XL 3.1 as the base, to stay compatible with the shot generator.

This workflow is only runnable if a correctly labeled, functioning self-hosted GPU runner is present. Otherwise GitHub queues it without executing. No GPU resources are provisioned automatically.

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

- `scripts/story.py` — v2, includes reference/adapter guide resolution and publishing continuity reports.
- `scripts/generate_shot.py` — v5, selects SDXL base/LoRA/IP-Adapter/ControlNet, can reuse cached safe embeddings.
- `scripts/library.py` — v2, adds `register-lora` operation without replacing existing Phase 1 library commands.
- `scripts/evaluate_continuity.py` — v1.
- `scripts/build_ip_embeddings.py` — v1.
- `scripts/prepare_lora_dataset.py` — v1; `scripts/publish_lora.py` — v2 (refuses public model destinations).
- `.github/workflows/generate-story-shot.yml` — v3.
- `.github/workflows/train-character-lora.yml` — v1 (GPU runner only).
- `.github/workflows/cache-reference-embedding.yml` — v1 (CPU runner, potentially resource-intensive).
- `requirements-image-v2.txt` — v2, includes Phase 3 model conditioning dependencies.
- `schemas/shot-v1.schema.json` — compatible optional Phase 3 fields, with descriptive schema revision v2.

## Scope and limitations

Phase 3 is **not** a guarantee that character faces, costumes or locations remain perfectly consistent across generated frames. The IP-Adapter and ControlNet models introduce large RAM/CPU requirements; `base` is the operational fallback for ordinary free runners. LoRA training requires an owner-provisioned GPU and an HF token; it cannot silently borrow unavailable GPU hardware. Character visual conditioning from multiple references does not imply each character will remain spatially isolated without masks.

The training and cache workflows do not automatically run; you decide when to invoke them. The generator does not perform recurring retraining or update models behind your back. All files were written to GitHub without executing training, image generation, or tests. The first live run is needed to establish actual end-to-end compatibility and resource use.

## Upstream references

- [Diffusers IP-Adapter documentation](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
- [Diffusers ControlNet documentation](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
- [Official SDXL LoRA training script](https://github.com/huggingface/diffusers/blob/v0.35.2/examples/dreambooth/train_dreambooth_lora_sdxl.py)
- [Animagine XL 3.1 model](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
