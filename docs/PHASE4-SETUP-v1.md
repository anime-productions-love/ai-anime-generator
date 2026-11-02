# Phase 4 — Comic Page Publishing, Storyboard Exports and Video Handoff (v1)

## Installed production stages

Phase 4 adds two lightweight, **CPU-only / no-AI-model** workflows to the established anime-character, scene and storyboard system:

- [Publish anime comic pages and storyboard (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/publish-storyboard.yml)
- [Export or send selected anime shot to video generator (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/send-shot-to-video.yml)

Both consume **canonical images already selected and committed by Phase 2/3**. Neither invents new shots, changes the official selection, nor downloads an SDXL checkpoint. There is no need for the external distributed CPU LoRA trainer to finish before you can publish pages from already-saved images.

**Verification status:** Source files and workflow YAML are committed and checked by reading them back. No Phase 4 GitHub Actions run has been launched; resulting pages, CBZ/ZIP archives, and cross-repository dispatch are not yet operationally verified.

## 1. Create a comic page from a sequence

First generate and select story images using the existing [story-shot generator](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-story-shot.yml) and optional [candidate selector](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/select-story-image.yml).

Open [Publish anime comic pages and storyboard](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/publish-storyboard.yml), select **Run workflow**, and enter:

| Field | Example | Notes |
|---|---|---|
| `story_id` | `story-001` | Must refer to an existing saved story |
| `sequence_id` | `seq-001` | Must have published, selected image shots |
| `shot_ids` | `0001,0002,0003,0004` | Optional; blank exports all shots in original sequence order |
| `captions` | `The forest grew quiet.||A shadow approached.||Aria reached for her sword.||The battle began.` | Optional; one short caption per selected shot separated by `||` |
| `layout` | `comic` | `comic` two columns, `vertical` one-column strip, `grid` three columns |
| `per_page` | `4` | 1–8 panels per page |
| `page_size` | `portrait` | 1400 × 2000 portrait or 2000 × 1400 landscape |

This operation checks the sequence record and verifies the SHA-256 of **every selected canonical image** before assembling anything. It never substitutes noncanonical candidates.

Images are resized proportionally to fit the panels **without cropping or stretching**. White letterboxing may appear; the original art remains unchanged. Captions are placed in a small panel-footer text area rather than editable speech balloons. The current format does not yet support complex editorial comic balloon placement or manga right-to-left reading order.

### Permanent result in your Git repository

Each successful run creates a fresh immutable export identifier based on that GitHub Actions run ID:

```text
stories/<story>/sequences/<sequence>/exports/export-<run-id>/
  manifest.json
  pages/
    page-001.png
    page-002.png
    ...
```

The export record includes source shot IDs/order, chosen-image paths and digests, captions, primary character and scene IDs, original shot prompt, video-handoff-ready image paths and **commit-pinned public image URLs**. Exporting again produces a new folder rather than overwriting prior comic pages or modifying earlier shot manifests.

### Downloadable output from the workflow

The run also uploads an artifact named `anime-storyboard-v1-<run-id>` containing:

```text
comic-pages-v1.cbz       Comic-book archive of numbered page PNGs
storyboard-v1.zip        Shot images + story/shot metadata + comic page PNGs
export-info.json        Export manifest path and page/shot counts
```

CBZ is a ZIP of numbered page images. Many comic readers can open it directly. The artifact is retained for seven days; **the page PNGs and export manifest remain permanently tracked in the Git repository** until someone explicitly changes or removes them.

### Illustration sequence and storyboard behavior

You can export up to **48 selected shots** per run. Ordering is the sequence's saved shot order unless you supply `shot_ids`. The ZIP contains the exact source `selected.png` for each shot (as `images/shot-0001.png`, etc.), not a resized substitute. Use the canonical image to animate a shot later; use the comic-page composite to read the sequence as a comic.

If you alter the chosen candidate of an existing shot, earlier comic exports do **not** change. Run the publisher again to create a new versioned snapshot with updated selections. This preserves durable publication history.

## 2. Export one selected illustration for video

Open [Export or send selected anime shot to video generator](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/send-shot-to-video.yml).

| Field | Default | Notes |
|---|---|---|
| `story_id`, `sequence_id`, `shot_id` | Required | Choose a real previously published shot, e.g. `0001` |
| `engine` | `wan` | `wan` for prompt-directed Wan 2.2 TI2V Turbo; `svd` for SVD-XT |
| `action` | `export` | `export` generates the verified handoff record only; `dispatch` also requests a video workflow |
| `motion_prompt` | Blank | Wan only: describes motion/camera action, falling back to a generic animation prompt plus the shot's original prompt |
| `width`, `height` | 480 × 832 | Used for Wan; SVD has its own native landscape canvas |
| `frames`, `fps` | 17 and 8 | Passed to the target video workflow; SVD's native frame count is normally 25 |

The bridge uses the image from the **selected candidate only**. It validates its recorded SHA-256 and constructs an image URL of the form:

```text
https://raw.githubusercontent.com/anime-productions-love/ai-anime-generator/<source-commit>/stories/<story>/sequences/<sequence>/shots/<shot>/selected.png
```

Using the exact Git commit means the URL does not unexpectedly change when the artist later selects a different candidate for the same shot.

### Default `export` behavior (no permissions or tokens needed)

The workflow only commits a durable handoff receipt such as:

```text
stories/story-001/sequences/seq-001/video-handoffs/handoff-<run-id>.json
```

The receipt records the exact source image URL, hash, prompts, video workflow name, and mapped workflow inputs. You can copy the direct URL into either video generator's existing **image** input field. The handoff does **not** generate video itself and does not modify the video repository.

### Optional `dispatch` behavior (cross-repo GitHub Actions)

To launch a video-generation workflow without copying its URL, create a fine-grained GitHub token that has **Actions: write** permission on **`anime-productions-love/ai-video-generator`**, and save it as the **`VIDEO_REPO_TOKEN` Actions secret in `anime-productions-love/ai-anime-generator`**.

Then select `action=dispatch`. The bridge sends the existing video's GitHub API `workflow_dispatch` event with `source_mode=image`, the exact pinned image URL and matching inputs:

- Wan: `generate-wan22-ti2v-turbo.yml`, source image, motion and negative prompts, full-frame contain-blur, frames, fps, portrait size and seed.
- SVD-XT: `generate-svd-xt.yml`, source image, contain-blur, frame count, FPS, fixed 25 steps / BF16 / standard motion controls and seed. **SVD-XT produces 1024 × 576 landscape video** even if the source shot is portrait, with padding when needed.

If GitHub responds HTTP 204, the bridge records `dispatch_accepted=true`. That means **only that the dispatch request was accepted**, not that video generation started or succeeded. The receipt explicitly leaves `target_run_id=null` and `video_created=false` until later verification. Check [video generator Actions](https://github.com/anime-productions-love/ai-video-generator/actions) for the actual run and download its MP4 artifact after completion.

**Important:** GitHub's normal `GITHUB_TOKEN` for this repository cannot necessarily write to a different repository. The `VIDEO_REPO_TOKEN` secret is optional for exports but required for dispatch. Do not place the token in a workflow input, a prompt, a source file or a public manifest.

## 3. How these outputs feed the film assembly process

After generating several video clips, open the existing [Anime movie scene combiner](https://github.com/anime-productions-love/ai-video-generator/actions/workflows/assemble-anime-movie.yml) in the **video repository**. Enter the completed Wan/SVD **video generation run IDs** in playback order. This existing tool combines those clips into a final MP4. The Phase 4 handoff does not automatically gather or resolve the final video run IDs.

Thus the installed stages are:

```text
Saved character models + environments
            ↓
Sequential anime keyframe generation + chosen image
            ├─→ Comic pages (PNG) and CBZ
            ├─→ Storyboard ZIP (images + prompts + order)
            └─→ Git-pinned chosen image → Wan / SVD → MP4 clips → scene combiner
```

## Scope and limitations

- Phase 4 requires **already-generated, committed story shots**. It cannot compose an empty sequence.
- Page assembly is CPU-only Pillow; it does not train or run a diffusion model, and does **not** require a GPU or the future distributed training network.
- Storyboard, CBZ, comic-page PNG and video handoff manifest generation are installed, **not runtime-validated**.
- A video dispatch relies on the separate video repo's existing Wan/SVD workflows, which themselves may have resource or runtime limits.
- Direct image links work because this is a **public repository**. Repositories with private media would need a different, authenticated image-transfer mechanism rather than publicly pinned raw GitHub links.
- The video's source is the **individual selected story image**, not the composite comic page.
- No additional video model training or automatic clip production was installed in Phase 4. It uses the user's already established workflows and avoids duplicate infrastructure.
- Comics and story ZIPs may be large. GitHub file-size/repository-size limits and Actions artifact retention still apply.

## Files

- `scripts/assemble_storyboard.py` — v2, creates comic PNGs, CBZ, storyboard ZIP, snapshots and pinned links
- `scripts/video_handoff.py` — v1, produces direct image bridge, optional GitHub API dispatch and immutable receipts
- `.github/workflows/publish-storyboard.yml` — v1
- `.github/workflows/send-shot-to-video.yml` — v1
- `docs/PHASE4-SETUP-v1.md` — this guide

Source files were installed and read back from GitHub. No new tests were created or executed, and no generation or export workflows were launched.
