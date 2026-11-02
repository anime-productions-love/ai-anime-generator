# AI Anime Story Generator — v4

A Git-backed production system for **anime characters, reusable environments, sequential illustrations, and continuity-controlled storyboards**. The repo contains three layers: Phase 1 persistent library, Phase 2 SDXL shot generation, and Phase 3 optional trained character LoRAs, reference-image conditioning, structural guidance, and continuity evaluation.

## Open a workflow

| Workflow | What it does |
|---|---|
| [Manage character and scene library (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/manage-asset-library.yml) | Create/update character and environment packs and attach licensed reference images |
| [Generate or continue anime story shot (v3)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml) | Generate numbered illustrations using saved characters/scenes, a last selected frame, and optional Phase 3 guidance |
| [Select canonical anime story image (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/select-story-image.yml) | Select one of the saved candidates without calling the image generator |
| [Cache anime reference embedding (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/cache-reference-embedding.yml) | Save safe, SHA-bound IP-Adapter Plus embeddings for one character/scene reference |
| [Train character LoRA (GPU runner) (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/train-character-lora.yml) | Train a reusable character LoRA using an **owner-provisioned self-hosted GPU** and upload weights to a private Hugging Face model repo |

**Full instructions:** [Phase 1](docs/PHASE1-SETUP-v1.md) · [Phase 2](docs/PHASE2-SETUP-v1.md) · **[Phase 3](docs/PHASE3-SETUP-v2.md)**.

## Quick start: make a short illustrated anime sequence

1. Save your character identity sheet(s) and scene/background art using the existing library manager. Choose stable IDs, e.g. `aria-001` and `moon-forest`.
2. Open **Generate or continue anime story shot**, select `start`, enter `story-001`, `seq-001`, your saved IDs and a prompt describing the first shot. Begin with `size=512x768`, `guidance_mode=base` and `candidates=1` to establish CPU feasibility.
3. After the run succeeds and commits its image, open the same workflow again using `continue`, the same story and sequence IDs, and a new action/camera prompt. Leave character and scene IDs blank to inherit the prior shot's setting.
4. To use **real image guidance**, choose `ip-adapter` for saved character/scene references, `controlnet` for Canny scene layout, or `both`. Those modes load substantially more neural-network weights and may exceed the 16 GB free CPU-runner limit.
5. After generating several candidates, inspect their saved `continuity_evaluation` similarity details and manually choose your favorite via **Select canonical anime story image**. The next shot conditions on the selected image.

Successful shots are saved to `stories/<story-id>/sequences/<seq-id>/shots/0001/` and numbered sequentially. The repo stores candidate PNGs, the canonical `selected.png`, precise prompts, asset revisions, selected reference paths and hashes, optional guide settings, and the reference-similarity report.

## How Phase 3 continuity really works

- **Character LoRA:** Once trained and registered, character `adapters.character_lora` contains a `hf://namespace/model-name` identifier. Inference loads the LoRA via the configured Hugging Face access token. Training is not performed on the standard free CPU runner.
- **IP-Adapter Plus SDXL:** When selected, the generator conditions on up to four saved character and scene reference images. Single-reference caches can reduce repeated image-encoder work. Multiple simultaneous references still use runtime image encoding.
- **ControlNet Canny:** Uses a saved scene `canny`, `layout`, or `edges` reference, or detects edges in a previous shot, as a structural guide. It is **not** yet pose skeleton estimation or guaranteed control over anatomy.
- **CLIP scoring:** Each candidate can receive a reference-image cosine-similarity ranking. This is **not** true face identification, a subjective art-quality judgment, or proof of narrative correctness. Candidate 1 remains canonical unless you select another one.

The underlying generator is an anime-focused SDXL model, [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1). Guidance modes and LoRAs depend on distinct pretrained components; they are not substituted by prompt text alone.

## Compute and storage limits

- Existing image generation runs on an `ubuntu-24.04` free public GitHub-hosted **CPU** runner, with a 350-minute job limit. SDXL itself is demanding; IP-Adapter and ControlNet can consume substantially more memory.
- **Character LoRA training requires a compatible self-hosted GPU runner labeled `gpu`, and GitHub secret `HF_TOKEN`.** The workflow is installed but cannot run unless you supply that hardware. Finished LoRA weights are stored privately in Hugging Face rather than committed into the public Git repository.
- Reference images and canonical shot illustrations live in this **public repository** and its history. Do not commit private character designs unless you intend them to be public.
- Building reusable embeddings still needs SDXL plus image-encoder inference. It may also exceed CPU memory/time limits.

## Core repository components

```text
assets/characters/<id>/manifest.json      Canonical character identity and reference records
assets/characters/<id>/refs/              Real reference images
assets/characters/<id>/embeds/            Optional safe-tensor IP reference caches
assets/scenes/<id>/manifest.json          Canonical environment identity and references
assets/styles/                            Stable style presets
scripts/library.py                        Character/scene management and LoRA registration (v2)
scripts/story.py                          Shot planning, inheritance and publication (v2)
scripts/generate_shot.py                  SDXL, IP-Adapter, ControlNet and LoRA inference (v5)
scripts/evaluate_continuity.py            Advisory reference-image similarity scorer (v1)
scripts/build_ip_embeddings.py            Safe reference embedding precomputation (v1)
scripts/prepare_lora_dataset.py           Verified training image packaging (v1)
scripts/publish_lora.py                   Private Hugging Face LoRA publication (v2)
stories/<story>/sequences/<sequence>/     Persistent storyboard images and manifests
schemas/                                  Versioned character/scene/shot JSON contracts
docs/PHASE3-SETUP-v2.md                   Phase 3 guide, installation and limitations
```

## Verification and scope

Phase 3 code and workflows have been committed. **No Phase 3 image generation, model training, embedding-building or scoring workflow has been executed or runtime-validated.** Successful source commits alone cannot establish model compatibility on the free CPU runner.

**Next: Phase 4** — comic page assembly, storyboard packages, and handoff into the already installed anime video-generation workflows.
