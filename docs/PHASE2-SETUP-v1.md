# Phase 2 — Anime Storyboard Image Generation (v1)

## What was installed

Phase 2 adds a real image-inference workflow to the persistent character and scene library from Phase 1.

- [Generate or continue anime story shot](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml) creates an initial illustration or automatically appends a new shot to an existing sequence.
- [Select canonical anime story image](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/select-story-image.yml) replaces a shot's selected image with another already-generated candidate, with **no second AI run**.
- `scripts/story.py` is the standard-library story-plan, sequence-memory, persistent-output and selection controller.
- `scripts/generate_shot.py` is the anime image generator using Diffusers and the anime-focused [Animagine XL 3.1 model](https://huggingface.co/cagliostrolab/animagine-xl-3.1).
- `schemas/shot-v1.schema.json` and `schemas/sequence-v1.schema.json` describe the Git-backed output contracts.
- `requirements-image-v1.txt` fixes the Python image-generation dependencies.

**Status:** committed and source-inspected, **not runtime-tested**. A complete CPU image generation on the free GitHub runner has **not** yet been demonstrated. The model can require multiple gigabytes of download and high RAM usage; GitHub's standard public runner has limited RAM and job duration. The workflow stops rather than publishing an empty or failed image.

## 1. Create reusable characters and scenes (once)

Create them through the existing [Phase 1 library manager](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/manage-asset-library.yml). Use **real, permanently committed reference images** whenever available. Example asset IDs:

- Character: `aria-001`
- Scene: `moon-forest`

The generator will look them up automatically. There are no seeded fake characters or scenes in the repo; you create your own.

## 2. Start a story sequence

Open [Generate or continue anime story shot](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml), select **Run workflow**, and enter:

| Field | Example | Meaning |
|---|---|---|
| `mode` | `start` | Creates the first shot; use `continue` for later shots |
| `story_id` | `story-001` | Stable story name |
| `sequence_id` | `seq-001` | Stable sequence within the story |
| `character_ids` | `aria-001` | Comma-separated IDs from your saved character library; may be blank |
| `scene_id` | `moon-forest` | ID from your saved scene library; may be blank |
| `shot_prompt` | `Aria glances up at the distant castle, worried, full-body wide shot` | New pose, action, emotion, and framing |
| `width` | `512` | First-run conservative CPU setting |
| `height` | `768` | First-run portrait resolution |
| `candidates` | `1` | From 1 to 4 separate generated alternatives |
| `seed` | `42` | Reproducibility seed; candidate 2 uses 43, etc. |

On success, the workflow commits a new shot into:

```text
stories/story-001/sequences/seq-001/
  manifest.json
  shots/
    0001/
      manifest.json
      selected.png
      candidates/
        candidate-01.png
        candidate-02.png  (if requested)
```

The shot manifest saves the original prompt, composed prompts, character and scene IDs, referenced asset revisions, exact reference-image paths and hashes, candidate list, selected image hash, generation settings, and previous-shot identity.

A separate GitHub Actions artifact `anime-story-shot-v1-<run-id>` also contains the generated candidates and the detailed preparation plan. The **Git repository, not the seven-day artifact**, owns the durable selected shot and its candidate history.

## 3. Continue a sequence without remembering image URLs

Open the same generation workflow again:

1. Change `mode` to **`continue`**.
2. Enter the **same story and sequence IDs**.
3. Leave `character_ids` and `scene_id` blank to inherit the cast and setting from the sequence.
4. Write the **next** action/panel prompt.
5. Click **Run workflow**.

The planner identifies the next shot number (0002, 0003, ...), loads the latest saved canonical shot PNG, verifies its SHA-256 and uses it as the input image for the next SDXL generation. This is *single-image image-to-image conditioning* rather than training or temporal video continuation.

You can override `character_ids` or `scene_id` to change the cast or setting. The generator then remembers the updated values for subsequent shots.

**If you enter `start` for an existing sequence, it refuses to overwrite prior work.** Use a new sequence ID or `continue`.

## 4. Choose your favorite candidate

With 2–4 generated candidates, the first image is initially the canonical one, without an invented automated quality score. Open [Select canonical anime story image](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/select-story-image.yml) and enter `story_id`, `sequence_id`, the shot number such as `0002`, and the candidate number.

The workflow copies the chosen candidate into `selected.png`, updates its hash and content revision, and commits the change. The next `continue` generation uses the **new selected image** of the most recent shot. Choosing a new image for an older shot does **not** automatically regenerate later shots which were already based on the earlier selection.

## How reference recall works in Phase 2

The system loads all selected character, scene and style manifests and validates every stored reference image. It composes the saved scene and identity descriptions into the current shot prompt, including attribute objects and negative prompts. The actual image-based conditioning currently uses **one** reference picture:

1. If continuing a sequence, **previous canonical shot image** (image-to-image, default denoising strength **0.55**).
2. If starting and a scene exists, the scene's `main` / `background` / `layout` image (strength **0.70**).
3. Otherwise, the first applicable character `full_body` / `three_quarter` / `main` / `face` reference (strength **0.70**).
4. If no images exist, generates from **text prompts only**.

The image is **contained, not cropped**, within the requested canvas, with blurred margins if its aspect ratio differs. Reference images not selected as the conditioning image are still recorded in the shot manifest, but their visual details are **not** independently image-conditioned in Phase 2. Their saved textual identity descriptions are used.

**Important limits:** A saved character reference is not yet a trained LoRA, and Phase 2 does not apply multiple IP-Adapters simultaneously. Character identity, scene geometry, outfit or background can drift between AI shots. The stronger LoRA/IP-Adapter/ControlNet identity system is planned for Phase 3. A seed alone does not lock an identity. SDXL's text encoders can truncate lengthy combined prompts.

## CPU and quality controls

- Default model: `cagliostrolab/animagine-xl-3.1`, pinned to Hugging Face commit `483f0c322568ed13697ed01dd0be07204746d12b`.
- CPU inference: PyTorch `bfloat16`, Diffusers `0.35.2` with VAE slicing/tiling and attention slicing.
- Sampling: DPM++ multistep/Karras, **28 diffusion steps**, **6.5 CFG** (fixed defaults in GitHub v1; CLI supports other values).
- First-run recommendation: **512 × 768, 1 candidate**. Higher resolutions and four candidates may exceed free runner CPU/RAM/time limits.
- Default image output: PNG; output imagery is committed into your **public** repository, including its historical revisions. Do not generate or upload content you intend to keep private.
- Model files are downloaded as needed from Hugging Face rather than committed to Git.
- Character, scene and style LoRA weights are **not yet trained or applied**. Phase 2 is an inference and sequence-memory foundation, not the full final consistency system.

## Running locally

From the repository root, create Phase 1 character/scene packs using `scripts/library.py` if needed. For advanced manual generation:

```bash
python3 scripts/story.py prepare --mode start --story-id story-001 --sequence-id seq-001 \
  --characters aria-001 --scene-id moon-forest \
  --prompt "Aria looks toward the castle in the distance" \
  --width 512 --height 768 --steps 28 --cfg 6.5 --candidates 1 \
  --plan work/shot-plan.json

python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python3 -m pip install -r requirements-image-v1.txt
python3 scripts/generate_shot.py --plan work/shot-plan.json --output-dir work/candidates
python3 scripts/story.py publish --plan work/shot-plan.json --candidate-dir work/candidates
```

These are usage examples, **not commands executed as part of installation**. Replace example character/scene IDs with actual saved packs, or omit those flags to create a text-only first shot.

## Phase 3 boundary

Phase 3 will add genuinely reference-driven character consistency: optional trained character LoRA weights, multiple image-reference conditioning with IP-Adapter, structural pose/depth controls through ControlNet, and explicit continuity evaluation. Phase 4 remains comic page assembly and an export bridge to the video-generator repository.

## Verification and operational limits

All Phase 2 source and workflow files have been installed and fetched from GitHub. **No workflow dispatches, image generations, GPU/CPU benchmarks, or tests were executed**. In particular, the default runner's ability to complete Animagine XL 3.1 inference at the chosen resolution is currently unproven. The workflow requires repository `contents: write` permission and a branch that accepts automation commits.
