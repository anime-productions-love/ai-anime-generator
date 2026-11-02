# AI Anime Story Generator — v2

A Git-backed anime story-image production system. Phase 1 provides persistent, reusable **character and scene libraries**. Phase 2 adds a real anime-focused **SDXL image generator**, sequential story-shot memory, stored illustration candidates, and user-selectable canonical shots.

## Run the workflows

| Workflow | Purpose |
|---|---|
| [Manage character and scene library (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/manage-asset-library.yml) | Create/update persistent characters and environments; attach reference images |
| [Generate or continue anime story shot (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml) | Generate first shot or continue a sequence using remembered characters, scene and last selected image |
| [Select canonical anime story image (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/select-story-image.yml) | Choose one of 1–4 saved candidates as the official image, without rerunning AI |

**Detailed guides:** [Phase 1 library](docs/PHASE1-SETUP-v1.md) · [Phase 2 story generation](docs/PHASE2-SETUP-v1.md).

## Fast start

1. Create a character and scene in the **Manage character and scene library** workflow, choosing permanent IDs such as `aria-001` and `moon-forest`. Attach direct HTTPS illustration URLs if you have them.
2. Open **Generate or continue anime story shot**. Set `mode: start`, `story_id: story-001`, `sequence_id: seq-001`, `character_ids: aria-001`, `scene_id: moon-forest`, and a prompt describing your first illustrated shot. For CPU feasibility, start at **512 × 768** with **one** image candidate.
3. After a successful run, retrieve the image from `stories/story-001/sequences/seq-001/shots/0001/selected.png`, which the workflow commits to Git. It also uploads a downloadable run artifact.
4. For the next illustration, run the same workflow with `mode: continue`, the **same story and sequence IDs**, and a new shot prompt. Leave character and scene fields blank to inherit them. The previous canonical image becomes the visual reference.
5. If you generated more than one candidate, use **Select canonical anime story image** to choose the best one. The next shot automatically uses the latest selected image.

You can also start with just a story/sequence ID and a text prompt, without existing character or scene records. When saved assets are supplied, the generator recalls their descriptions, tags/attributes, style and verified reference paths.

## What Phase 2 actually provides

- **Real anime-focused model:** [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1) (an SDXL fine-tune) on PyTorch/Diffusers CPU. This is not a dummy renderer. The checkpoint revision is pinned for reproducibility.
- **Persistent story history:** `stories/<story-id>/sequences/<sequence-id>/manifest.json` and numbered shot records.
- **Character/scene recall:** Load saved Phase 1 assets, including canonical prompts, textual identity, scene profiles and reference-image checksums.
- **Shot continuity:** The last official `selected.png` becomes the next shot's image-to-image input, preserving its complete aspect-ratio composition with letterboxing/blur rather than center cropping.
- **Alternatives and approval:** Up to four candidates; candidate one is initially selected and can be changed later without regeneration.
- **Durable publishing:** Successful candidate images, selected PNG and shot metadata are committed into the repository, not dependent on short-lived GitHub Actions artifact retention.
- **Failure discipline:** The workflow saves a shot only after the model has produced every requested PNG; failed inference does not become a new canonical shot.

**Phase 2 reference limitation:** This release image-conditions on **one** picture at a time, giving precedence to the previous selected shot. Other character/scene references are stored and their descriptions included in prompts; the separate Phase 3 system will implement LoRA, IP-Adapter and ControlNet for stronger visual consistency. It is not accurate to claim perfect multi-character identity preservation yet.

## Model, runner, and cost limits

The workflow targets a free public-repository GitHub Actions **CPU** runner. It installs CPU PyTorch and downloads model weights to the ephemeral runner, with a 350-minute job limit. Anime SDXL is resource-intensive; **this Phase 2 generator is installed but its end-to-end inference has not yet been verified on the free runner**. Start with one candidate and a small resolution before investing in a long sequence. There is no guaranteed runtime or memory fit.

The GitHub Actions form deliberately stays within GitHub's ten-input limit. It exposes shot mode, IDs, prompt, resolution, candidate count and seed; the generation script uses a 28-step, 6.5 CFG default. More advanced values are available using the Python command-line scripts.

The repository and its Git history are **public**. Saved character references, generated images and story prompts are public. Respect the upstream model license and the rights to any reference artwork.

## Repository map

```text
assets/characters/<id>/manifest.json         Saved character and reference images
assets/scenes/<id>/manifest.json             Saved environment and reference images
assets/styles/anime-cel-fantasy/manifest.json Reusable rendering style
schemas/*-v1.schema.json                     Character, scene, style, shot, sequence
scripts/library.py                           Phase 1 asset management (v1)
scripts/story.py                             Phase 2 shot planning and canonical selection (v1)
scripts/generate_shot.py                     Phase 2 SDXL image generation (v2)
.github/workflows/manage-asset-library.yml    Library UI (v1)
.github/workflows/generate-story-shot.yml     Image generator UI (v1)
.github/workflows/select-story-image.yml      Candidate selection UI (v1)
stories/<story>/sequences/<sequence>/         Persisted story images and history
docs/PHASE1-SETUP-v1.md                       Phase 1 instructions
docs/PHASE2-SETUP-v1.md                       Phase 2 instructions
```

## Next phases

**Phase 3**: character LoRAs, IP-Adapter reference embeddings, pose/layout controls and identity consistency evaluation. **Phase 4**: comic page assembly, storyboard export and integration into the separate Wan/SVD video-generation system.

**Source verification** is distinct from inference validation. No tests or generation runs were started as part of the Phase 2 installation.
