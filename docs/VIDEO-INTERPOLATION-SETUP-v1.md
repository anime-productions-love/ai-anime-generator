# Anime Video Frame Interpolation and Playback-Speed Controls — v1

## What this adds

The existing Wan 2.2 and SVD-XT scene generators remain unchanged. A new independent [Smooth or speed up anime video (RIFE AI)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/enhance-anime-video.yml) workflow accepts a completed generation run ID or direct video URL and produces a **new MP4, last_frame.png, and metadata.json**. It does **not** load/re-run Wan or SVD.

You can also give it a direct private [anime-productions-love/gallery](https://github.com/anime-productions-love/gallery) video file URL; it uses the existing narrowly scoped GitHub fine-grained token reader. Other direct downloadable video URLs still work without receiving the gallery token.

**Important:** Code and workflow are committed, but no live interpolation run has been launched. The pinned upstream RIFE release and CPU-only mode are documented by its author, but free GitHub CPU runner performance and runtime remain unverified.

## Why your first Wan clip looked slow

Your [successful Wan run 41448667591](https://github.com/anime-productions-love/ai-video-generator/actions/runs/41448667591) used **42 delivered frames at 6 FPS**, giving about **7 seconds of video**.

- Playing those **same 42 frames** at **12 FPS** gives about **3.5 seconds**, so playback and movement become **2× faster**.
- Playing those **same 42 frames** at **24 FPS** gives about **1.75 seconds**, or **4× faster**.
- Generating an entirely new Wan clip at 12 FPS can give *different motion*, because the model may also consume FPS as a generation condition. The exact two-times timing rule is guaranteed when **retiming the same frames**, not for the motion content of a new generation.

**FPS is not a character-motion-speed instruction.** The network decides which poses and action it generates; FPS tells the video player how quickly to present the resulting sequence. A low FPS can feel choppy and—when holding the total frame count fixed—lengthen the final clip. If Wan created inherently small movement between poses, higher FPS alone cannot create more action.

**Interpolation (RIFE) estimates *new in-between pictures***. At playback speed 1× it makes the video smoother while keeping roughly its original duration. Adjust `playback_speed` separately if you want quicker character/action motion.

## Start with your existing video

Open the new [interpolation workflow](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/enhance-anime-video.yml) and click **Run workflow**.

| Input | Suggested first try | Meaning |
|---|---|---|
| `source_run` | `41448667591` | Your already-completed original Wan video; no new Wan inference |
| `video_url` | *(blank)* | Instead of a run ID, accept a direct downloadable video URL, including private gallery media |
| `source_repo` | `auto` | Check the unified anime repo, then the original video repo |
| `method` | `rife` | Real neural frame interpolation on CPU; choose `motion` for FFmpeg optical flow instead |
| `target_fps` | `24` | Smooth output presentation rate |
| `playback_speed` | `1.5` or `2.0` | 1.5× is moderately faster, 2× is twice the motion speed |
| `quality_crf` | `20` | H.264 output quality (lower number → less loss) |

### What the output timing means

For your 42-frame / 6 FPS / 7-second source:

| `target_fps` | `playback_speed` | Approximate output frames | Approximate duration |
|---:|---:|---:|---:|
| 24 | 1.0× | 168 | 7.00 seconds |
| 24 | 1.5× | 112 | 4.67 seconds |
| 24 | 2.0× | 84 | 3.50 seconds |
| 12 | 2.0× | 42 | 3.50 seconds (little or no interpolation needed) |

These values assume the actual video duration is seven seconds (42 / 6) and constant-frame-rate input. The workflow probes the actual source and records measured output timing rather than claiming exact results from the table.

**Recommended:** `rife`, **24 FPS**, **1.5×** first. If it still looks like slow motion, select **2×**. This makes the action faster *and* smoother without paying for another 2½-hour Wan generation.

### How to use a private-gallery video

Choose `source_run` blank, set `video_url` to a GitHub `blob` or Contents API URL for a real video in `anime-productions-love/gallery`, and use the same processing controls. The GitHub Actions secret must be available in **ai-anime-generator**, not merely the old video repo. Supported secret names are `GITHUB_FINE_GRAINED_TOKEN` (preferred), `GALLERY_REPO_TOKEN`, and `GALLERY_TOKEN`.

## Interpolation methods

**`rife` (neural AI interpolation):** Installs the official portable [RIFE ncnn Vulkan release 20221029](https://github.com/nihui/rife-ncnn-vulkan/releases/tag/20221029), which bundles compatible model data. Its documented `-g -1` option runs on **CPU**; no CUDA or owner-managed GPU is required. The workflow extracts source frames using FFmpeg, calculates how many new frames are needed for the chosen playback rate and speed, calls RIFE `-n <frame-count> -g -1`, then encodes the resulting PNGs as an H.264 MP4. This mode may be **slow on a four-core GitHub CPU runner**, especially with 480×832 frames and large output counts. It may also create visual artifacts around very rapid motion, occlusions, hands, and anime line art.

**`motion` (non-neural):** Uses FFmpeg's `minterpolate` motion estimation directly. This requires no neural checkpoint download and is a useful faster fallback, but it is **not RIFE or an AI neural model**, and may produce different interpolation artifacts.

If the requested output frame count is **not greater** than the source count, RIFE mode uses a simple retime without inventing in-between frames, and the metadata reports the actual method.

## What gets saved

The workflow uploads `anime-smooth-video-v1-<enhancement-run-id>` with:

```text
video.mp4
last_frame.png
metadata.json
```

Metadata includes the source type/run/repository, measured source and output FPS, source and output duration, target FPS, requested speed, actual rendering method, resolution and elapsed processing time.

The artifact is retained for **seven days**, like the original scene artifacts. The source video is **never modified**; you can retry with multiple FPS and speed choices without regenerating Wan video.

## Combine enhanced scenes in a movie

The consolidated [Anime Movie Scene Combiner v4](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/assemble-anime-movie.yml) accepts `anime-smooth-video-v1-<run-id>` artifacts in addition to original `wan22-turbo-v*-<run-id>` and `svd-xt-v*-<run-id>` clips. Paste each completed **interpolation run ID** in the desired playback order, and choose **24 FPS** for the movie output if your enhanced clips use 24 FPS.

The scene combiner may also still retrieve unexpired runs from the original `ai-video-generator` repository. It does not automatically smooth already-generated clips; enhance them first if desired.

**Do not confuse a smoothing run ID with an original Wan/SVD run ID**: for the movie combiner, either is acceptable, but use the smoothing run ID to select the interpolated version.

## Implementation files

- `scripts/interpolate_video.py` — v1: FFprobe timing, source artifact/gallery retrieval, RIFE or FFmpeg flow, MP4 generation and metadata
- `.github/workflows/enhance-anime-video.yml` — v1: manually dispatched CPU job with runtime-selectable playback speed
- `.github/workflows/assemble-anime-movie.yml` — v4: now accepts completed enhanced scenes
- `docs/VIDEO-INTERPOLATION-SETUP-v1.md` — this guide

**Source and GitHub commits are not a runtime proof.** No additional tests, training runs, generation jobs, or interpolation workflows were launched during installation. If the pinned RIFE release fails on your particular runner, the `motion` mode provides an independent non-neural alternative.
