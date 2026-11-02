# Automatic Multi-Clip Wan Video Chains — v1

## Objective

Generate a **multi-minute anime scene unattended** by running one ordinary GitHub Actions Wan video job per clip. You enter all shot prompts and generation settings in a single JSON production plan *before* starting. Each successful job publishes its complete video plus the exact final frame, then automatically dispatches the next job using `workflow_dispatch`.

**No Cloudflare Worker, paid GPU runner, or separate orchestration repository is required.** Each Wan job runs on `ubuntu-24.04` and has an independent **350-minute timeout** (under the GitHub-hosted 6-hour per-job ceiling). The full chain may run over days because it consists of distinct workflow runs, not one continuous six-hour process.

**Installed, not runtime-verified.** No chain job or model-generation tests were executed as part of this implementation. End-to-end viability remains subject to the underlying Wan CPU generator, network availability, output-artifact availability, GitHub Actions quotas, and repository workflow permissions.

## 1. Write the full plan first

A complete editable example is already committed:

[chains/example-anime-sequence-v1.json](../chains/example-anime-sequence-v1.json)

Edit that example *or create a new JSON file in* `chains/`, e.g. `chains/forest-chase-v1.json`. Add one `segments` entry per desired shot, specifying its prompt and any settings that differ from the defaults. You can also add `repeat` to generate several chained shots using the same prompt/settings without retyping them.

Example (all values are examples, not a running campaign):

```json
{
  "version": "v1",
  "title": "An uninterrupted nighttime forest sequence",
  "initial_image": "https://github.com/anime-productions-love/gallery/blob/main/Zero_Two_2.jpg",
  "defaults": {
    "negative_prompt": "flicker, incorrect face, inconsistent clothing, malformed hands",
    "fit_mode": "contain-blur",
    "frames": 42,
    "fps": 4,
    "width": 480,
    "height": 832,
    "seed": 100
  },
  "segments": [
    {
      "prompt": "The character walks forward in a moonlit forest while the camera slowly tracks alongside.",
      "repeat": 3
    },
    {
      "prompt": "The character stops, turns toward a sound, looks uneasy, and then continues.",
      "frames": 42,
      "fps": 6,
      "seed": 101
    },
    {
      "prompt": "The camera moves into a close-up as the character looks determined.",
      "source": "https://github.com/anime-productions-love/gallery/blob/main/Zero_Two_2.jpg",
      "seed": 102
    }
  ],
  "assembly": {
    "enabled": true,
    "width": 480,
    "height": 832,
    "fps": 4,
    "transition": "cut",
    "fade_seconds": "0.3",
    "quality_crf": 20
  }
}
```

This example would produce **five separate Wan runs** (the first entry repeats three times), with an expected 210 delivered frames in total. With mixed 4/6 FPS settings, duration depends on each clip's own encoded playback rate and the movie assembly choice.

### Config details and limits

- `initial_image` — direct image/video URL or private [anime gallery](https://github.com/anime-productions-love/gallery) GitHub file URL. The token `VIDEO_REPO_TOKEN` already used by your consolidated Wan workflow is supported. The image is used only for segment 1 unless a later segment has an explicit `source`.
- `defaults` — optional settings shared by all segments: `frames`, `fps`, `width`, `height`, `fit_mode`, `negative_prompt`, `seed`.
- `segments` — an ordered list of objects with required `prompt`; override any default for a particular clip. Add `repeat: N` to make N consecutive, independently generated continuations using that entry's instructions.
- `source` — optional per-clip image/video URL. If provided for a later shot, it deliberately restarts visual conditioning from that URL instead of the prior frame, while preserving the chain's shot history.
- `assembly` — controls automatic movie assembly; set `enabled: false` to retain all clips without starting the combiner.

Each chain permits up to **120 expanded clips**. The plan file can be at most 128 KiB. Frame counts are 5–121 per clip, FPS is 1–60, width/height are divisible by 16, 256–1536 pixels, and seed is a nonnegative integer. Large frame counts/resolutions may cause the Wan job to exceed its 350-minute limit. Start conservatively until the CPU runtime for your exact settings is known.

The underlying Wan Turbo generator remains pinned to **four model steps**; there is no second, duplicated model-training or generation engine.

## 2. Start the chain ONCE

Open the [Wan 2.2 Turbo Anime Generator workflow v6](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-wan22-ti2v-turbo.yml) on `main`.

Select **Run workflow**, then set:

| Field | Value |
|---|---|
| `chain_plan` | `chains/example-anime-sequence-v1.json` (or your newly saved path) |
| `chain_step` | `1` |
| `chain_previous_run` | Leave empty |
| `chain_id` | Leave empty |

The other manual inputs (`image`, `prompt`, `frames`, `fps`, `seed`, etc.) are ignored when `chain_plan` is set; **each clip uses its own settings from the JSON plan**. The plan path is the only first-run field that must be filled in for chain mode. Normal **single-clip** generation continues to work if `chain_plan` is blank.

The first run ID becomes the stable **chain ID** automatically. This is used when the workflow dispatches each later run, so no manual run IDs are required once the first job is launched.

## 3. What the workflow does automatically

For each clip:

1. Read the saved JSON plan from `chains/` and verify its SHA-256 fingerprint has not changed from the prior run.
2. Resolve the clip configuration for the current step, including a repeated prompt entry if applicable.
3. For the first clip, fetch `initial_image` through the existing authenticated gallery/direct-video reader. For later clips, retrieve **exactly the specified preceding run's** artifact—not merely the newest Wan video that another campaign might have generated.
4. Confirm the earlier artifact's `chain_id`, exact `config_sha256`, previous step number, run ID, shot history, and saved SHA-256 of `last_frame.png`.
5. Generate the new Wan video using the existing pinned CPU stable-diffusion.cpp backend.
6. Verify output `video.mp4` and `last_frame.png`, write `chain-state.json`, and upload the complete `wan22-turbo-v6-<run-id>` artifact.
7. **Only after successful artifact upload**, dispatch the next clip with `chain_step=N+1`, the original chain ID, and the exact current run ID.
8. After the final clip, automatically dispatch the **existing** [Anime Movie Scene Combiner](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/assemble-anime-movie.yml) using the ordered list of all completed clip run IDs.

The combiner then produces a final MP4 movie as a **separate GitHub Actions run and downloadable artifact**. Its success is separate from the final Wan clip's success. The chain controller reports a movie-assembly dispatch, not a completed movie, until the movie job actually succeeds.

### Persistence and resume behavior

Every generated video stays in its own GitHub Actions artifact with its final frame, `chain-state.json`, prompt and metadata. The state stores all earlier completed run IDs and a SHA-256 reference to the immutable production plan. Nothing is silently marked complete merely because a job was submitted.

**If any Wan run fails or times out, the chain stops.** It does not dispatch the next clip or fabricate success. All earlier clip artifacts remain accessible until their retention expires.

To resume after fixing a real failure:
- Re-run the Wan workflow manually using the same `chain_plan`.
- Set `chain_step` to the missing next number (e.g. 7 if scenes 1–6 succeeded).
- Enter `chain_previous_run` as the **exact last successful clip run ID**.
- Enter `chain_id` as the **first run ID** from the original chain.

The script rechecks the predecessor artifact and refuses to continue from an unrelated chain, a different production plan or an altered last frame.

**Avoid changing the JSON plan in place while a chain is running.** The SHA-256 fingerprint is intentionally checked at every transition. To make a different storyline, create a separate versioned plan and start a new chain.

### Artifact retention and long chains

Each individual video artifact currently has a **7-day retention limit**. For a one- or two-day chain, all clips normally remain accessible until final movie assembly. But a very long chain that exceeds retention may be able to continue from its most recent clip yet fail final assembly because its earliest clips have expired. Keep production under seven days per movie (with a margin), or use durable clip archiving before making longer campaigns.

GitHub Actions job concurrency, runtime limits, and fair-use/usage quotas still apply. Each run must finish its own model work within 350 minutes; chaining does **not** extend an individual Wan invocation past the six-hour GitHub-hosted job ceiling.

## 4. Smooth the completed multi-minute movie

After the combiner finishes successfully, copy its **movie assembly run ID** (not the last Wan clip ID).

Open [Smooth or Speed Up Anime Video (RIFE AI v3)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/enhance-anime-video.yml) and enter that movie run ID as `source_run`. Choose e.g. `target_fps=24` and `playback_speed=1.0` for smooth original-duration output, or `1.5` to speed up the motion.

The frame-interpolation script now accepts `anime-movie-v*-<run-id>` artifacts as inputs. **However, its current CPU RIFE implementation supports at most 600 original source frames and 2400 interpolated frames in one run.** A several-minute movie at 4–6 FPS may exceed those limits. Use `method=motion` (streaming FFmpeg optical-flow interpolation, rather than neural RIFE) for longer movies, or smooth the individual short clips first. Large CPU interpolation jobs can still encounter the six-hour time limit.

## 5. Architecture and current verification

- `scripts/wan_chain.py` — v1; configuration parsing and repeat expansion, chain state verification, exact previous frame retrieval, existing Wan generator invocation, artifact recording, dispatch of the next Wan run or movie workflow.
- `.github/workflows/generate-wan22-ti2v-turbo.yml` — v6; four additional optional chain fields, same generation backend, automatic dispatch and no duplicate runner model.
- `chains/example-anime-sequence-v1.json` — editable template using the current private gallery.
- `scripts/interpolate_video.py` — v2; recognizes final `anime-movie` artifacts.
- `.github/workflows/enhance-anime-video.yml` — v3; labels movie source support.
- `docs/WAN-AUTOMATIC-CHAINS-v1.md` — this guide.

GitHub's current Actions limits state a **six-hour GitHub-hosted job limit**, a **35-day maximum for a single workflow run**, and **up to 25 workflow_dispatch input properties**. A chain consisting of separate workflow runs does not share one six-hour clock. The repository's built-in token with `actions: write` can initiate `workflow_dispatch` jobs from the prior workflow (this is an explicit exception to GitHub's normal token-trigger suppression).

**No workflow dispatches, training, tests, image generation, or interpolation runs were executed during implementation.** You need one actual first-run dispatch to validate GitHub's runtime environment and the complete transition.
