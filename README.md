# AI Anime Story Generator — v6

A Git-backed production system for **anime characters, reusable environments, sequential illustrations, comic pages, and video-ready storyboards**. Phase 1 saves characters and scenes; Phase 2 generates images; Phase 3 adds visual-reference conditioning and distributed CPU LoRA-training integration; Phase 4 publishes comic pages and video handoffs.

## Open a workflow

| Workflow | What it does |
|---|---|
| [Manage character and scene library (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/manage-asset-library.yml) | Create/update character and environment packs and attach licensed reference images |
| [Generate or continue anime story shot (v3)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml) | Generate numbered illustrations using saved characters/scenes, a last selected frame, and optional Phase 3 guidance |
| [Select canonical anime story image (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/select-story-image.yml) | Select one of the saved candidates without calling the image generator |
| [Cache anime reference embedding (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/cache-reference-embedding.yml) | Save safe, SHA-bound IP-Adapter Plus embeddings for one character/scene reference |
| [Prepare or finalize distributed CPU character LoRA (v2)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/train-character-lora.yml) | Prepare a request for the separate multi-runner CPU training network; verify and register the returned private LoRA only after training finishes |
| [Publish anime comic pages and storyboard (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/publish-storyboard.yml) | Generate permanent comic-page PNGs, a CBZ comic and storyboard ZIP from approved images |
| [Export or send selected anime shot to video generator (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/send-shot-to-video.yml) | Export one selected image using a commit-pinned URL, or optionally dispatch an existing Wan/SVD video workflow |

**Full instructions:** [Phase 1](docs/PHASE1-SETUP-v1.md) · [Phase 2](docs/PHASE2-SETUP-v1.md) · **[Phase 3](docs/PHASE3-SETUP-v3.md)** · **[Phase 4](docs/PHASE4-SETUP-v1.md)**.

## Quick start: make a short illustrated anime sequence

1. Save your character identity sheet(s) and scene/background art using the existing library manager. Choose stable IDs, e.g. `aria-001` and `moon-forest`.
2. Open **Generate or continue anime story shot**, select `start`, enter `story-001`, `seq-001`, your saved IDs and a prompt describing the first shot. Begin with `size=512x768`, `guidance_mode=base` and `candidates=1` to establish CPU feasibility.
3. After the run succeeds and commits its image, open the same workflow again using `continue`, the same story and sequence IDs, and a new action/camera prompt. Leave character and scene IDs blank to inherit the prior shot's setting.
4. To use **real image guidance**, choose `ip-adapter` for saved character/scene references, `controlnet` for Canny scene layout, or `both`. Those modes load substantially more neural-network weights and may exceed the 16 GB free CPU-runner limit.
5. After generating several candidates, inspect their saved `continuity_evaluation` similarity details and manually choose your favorite via **Select canonical anime story image**. The next shot conditions on the selected image.

Successful shots are saved to `stories/<story-id>/sequences/<seq-id>/shots/0001/` and numbered sequentially. The repo stores candidate PNGs, the canonical `selected.png`, precise prompts, asset revisions, selected reference paths and hashes, optional guide settings, and the reference-similarity report.

## Phase 4: publish your comic or animate a shot

Once a sequence has saved selected images, run **Publish anime comic pages and storyboard** with your story/sequence IDs. Optionally select an ordered shot subset and captions. It composes letterboxed panels into permanent PNG pages and archives them as a CBZ and a storyboard ZIP containing the original keyframes, prompts and order. Older exports are never overwritten.

Run **Export or send selected anime shot to video generator** for a chosen `shot_id`. The default `export` mode commits a verified, immutable raw-Git image URL and mapped video inputs without launching any video. Optional `dispatch` requests an existing [Wan/SVD video workflow](https://github.com/anime-productions-love/ai-video-generator/actions) using GitHub secret `VIDEO_REPO_TOKEN` (Actions:write on the video repository). Dispatch acceptance is **not** video completion. Combine successful video clips later using the [existing movie combiner](https://github.com/anime-productions-love/ai-video-generator/actions/workflows/assemble-anime-movie.yml).

**Phase 4 is CPU-only and does not need LoRA training or a GPU.** [Complete Phase 4 instructions](docs/PHASE4-SETUP-v1.md).

## How Phase 3 continuity really works

- **Character LoRA:** Training is requested from the separate distributed GitHub CPU training network (under development). It must genuinely shard model state across runners; CPU memory is not automatically pooled. After real training and output verification, character `adapters.character_lora` records `hf://namespace/model@immutable-commit`, which inference loads using the Hugging Face token.
- **IP-Adapter Plus SDXL:** When selected, the generator conditions on up to four saved character and scene reference images. Single-reference caches can reduce repeated image-encoder work. Multiple simultaneous references still use runtime image encoding.
- **ControlNet Canny:** Uses a saved scene `canny`, `layout`, or `edges` reference, or detects edges in a previous shot, as a structural guide. It is **not** yet pose skeleton estimation or guaranteed control over anatomy.
- **CLIP scoring:** Each candidate can receive a reference-image cosine-similarity ranking. This is **not** true face identification, a subjective art-quality judgment, or proof of narrative correctness. Candidate 1 remains canonical unless you select another one.

The underlying generator is an anime-focused SDXL model, [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1). Guidance modes and LoRAs depend on distinct pretrained components; they are not substituted by prompt text alone.

## Compute and storage limits

- Existing image generation runs on an `ubuntu-24.04` free public GitHub-hosted **CPU** runner, with a 350-minute job limit. SDXL itself is demanding; IP-Adapter and ControlNet can consume substantially more memory.
- **Character LoRA training does not require a self-hosted GPU.** The replaced training workflow runs on an ordinary GitHub CPU runner to prepare a versioned training request or finalize externally trained weights; it does not run SDXL optimization itself. Actual training remains dependent on the separately built multi-runner CPU coordinator. A GitHub secret `HF_TOKEN` is required **only for verification and use of the private Hugging Face LoRA**.
- Reference images and canonical shot illustrations live in this **public repository** and its history. Do not commit private character designs unless you intend them to be public.
- Building reusable embeddings still needs SDXL plus image-encoder inference. It may also exceed CPU memory/time limits.

## Core repository components

```text
assets/characters/<id>/manifest.json      Canonical character identity and reference records
assets/characters/<id>/refs/              Real reference images
assets/characters/<id>/embeds/            Optional safe-tensor IP reference caches
assets/scenes/<id>/manifest.json          Canonical environment identity and references
assets/styles/                            Stable style presets
scripts/library.py                        Character/scene management and LoRA registration (v3)
scripts/story.py                          Shot planning, inheritance and publication (v3)
scripts/generate_shot.py                  SDXL, IP-Adapter, ControlNet and LoRA inference (v6)
scripts/evaluate_continuity.py            Advisory reference-image similarity scorer (v1)
scripts/build_ip_embeddings.py            Safe reference embedding precomputation (v1)
scripts/prepare_lora_dataset.py           Verified distributed training dataset importer (v2)
scripts/distributed_lora.py               Distributed CPU request, provenance verification and immutable model acceptance (v1)
stories/<story>/sequences/<sequence>/     Canonical storyboard, comic exports and video receipts
scripts/assemble_storyboard.py               Comic PNG pages, CBZ and storyboard ZIP exporter (v2)
scripts/video_handoff.py                     Verified shot URL and optional Wan/SVD dispatch (v1)
.github/workflows/publish-storyboard.yml     Comic and storyboard publishing (v1)
.github/workflows/send-shot-to-video.yml     Export-only or remote video generation handoff (v1)
schemas/                                  Versioned character/scene/shot JSON contracts
docs/PHASE3-SETUP-v3.md                   Phase 3 guide with distributed CPU training integration
docs/PHASE4-SETUP-v1.md                   Phase 4 comic/storyboard/video guide
```

## Verification and scope

Phase 4 source code and workflows are committed. **No Phase 4 comic export, cross-repository dispatch, AI generation or tests were executed.** Source read-back verification is not an end-to-end runtime result.

**Distributed CPU training integration:** [Exact request/response contract v1](docs/DISTRIBUTED-CPU-TRAINER-CONTRACT-v1.md) for the other agent building the multi-GitHub-runner network. A prepared training request is not a completed training run.

Phase 4 is installed; the operational next step is generating and approving actual story shots, then using the comic publisher or video handoff. Live performance remains unverified until authorized workflow execution.
