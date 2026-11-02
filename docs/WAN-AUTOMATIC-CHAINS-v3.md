# Continuous Wan Scene Production — v3

## What changed

This replaces the older editable JSON production plan with a **native GitHub Actions launch form**. You fill in **one starting image**, **one shared configuration**, and **2–17 different video motion prompts**. Submit once: each completed clip automatically generates the next clip from the immediately previous clip's final frame and the final clip requests movie assembly.

**[Open: Start continuous Wan anime scene (2–17 videos)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/start-continuous-wan-scene.yml)**

The separate [Wan generator workflow](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/generate-wan22-ti2v-turbo.yml) remains the underlying single-video generator and internal continuation worker. You do **not** need to edit a JSON file or manually fill in its internal `chain_step`, `chain_previous_run`, `chain_id` fields.

## Why the form has 17 prompt boxes

GitHub `workflow_dispatch` permits **at most 25 input fields**. Eight fields are needed for the starting image, video count and shared settings. The remaining 17 fields provide **17 separate video prompts**.

The `video_count` dropdown ranges from **2 to 17**. GitHub cannot dynamically reveal or hide prompt fields based on a dropdown, so all 17 prompt boxes are visible in the launch form. **Fill only prompt boxes 1 through N** for your selected count, leaving all later boxes blank. The workflow verifies the requested number and refuses to silently skip or add instructions.

The 25 inputs are:

- **Video count (2–17)** and **initial image** (private gallery or direct HTTPS image/video URL)
- **Shared negative prompt**, **framing mode**, **frames per clip**, **playback FPS**, **frame size** (`WIDTHxHEIGHT`) and **seed**
- **Video 01 prompt** through **Video 17 prompt**, each a distinct motion/camera instruction (the first two are always required)

The model uses the same Turbo inference settings and all shared settings across every shot. Different videos may have different **motion prompts only**, allowing the story to progress while retaining technical consistency.

## One-time launch example

Open [Start continuous Wan anime scene](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/start-continuous-wan-scene.yml), click **Run workflow** and choose:

| Field | Example |
|---|---|
| Video count | `4` |
| Initial image | `https://github.com/anime-productions-love/gallery/blob/main/Zero_Two_2.jpg` |
| Negative prompt | `flicker, warped face, sudden camera cuts, text` |
| Framing | `contain-blur` |
| Frames per clip | `42` |
| FPS | `4` |
| Frame size | `480x832` |
| Seed | `42` |
| Video 01 prompt | `Character walks forward, hair swaying, camera gently tracking` |
| Video 02 prompt | `Character glances toward a noise; camera continues its smooth track` |
| Video 03 prompt | `Character picks up pace, then slows while looking over shoulder` |
| Video 04 prompt | `Character stops beside a lantern, looking toward the distant horizon` |
| Video 05–17 prompts | **Leave blank** |

At **42 frames per clip / 4 FPS**, four clips have a nominal combined running time around **42 seconds** before transitions and adjustments. For 17 clips, this is approximately **178.5 seconds (2 min 58.5 sec)**. Real duration depends on the video outputs and the final FFmpeg assembly.

## Continuation and integrity

1. The short **start workflow** validates your inputs and commits a frozen plan to `chains/chain-<startup-run-id>.json`. The form builds this file automatically; no file editing is required.
2. It dispatches clip 1 through the existing Wan CPU generator. **Only clip 1 fetches your selected starting image.**
3. After successful generation and artifact upload, the clip's `chain-state.json` lists the exact run ID, plan SHA-256, completed clip history, final PNG SHA-256 and MP4 SHA-256.
4. Clip 2 retrieves **only the designated clip 1 artifact** and uses its verified `last_frame.png` for generation. Clip 3 uses the precise clip 2 final frame, and so on.
5. Every clip uses the **same original frames/FPS/size/seed/framing/negative prompt**; only its written motion prompt changes.
6. After the last clip, the controller dispatches the existing [movie assembler](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/assemble-anime-movie.yml) with all clip IDs in correct order, using clean cuts and the common resolution/FPS.

There is **no per-clip image picker** or custom image-source override after clip 1. A failed or timed-out clip stops the chain. The next clip is never launched on the basis of a merely queued or failed predecessor.

**GitHub limits:** Each Wan clip is a different `ubuntu-24.04` workflow job with a 350-minute timeout. Individual hosted jobs cannot exceed GitHub's 6-hour limit; **separate chained runs can span much longer**. Model performance, runner availability, Actions quotas and the 7-day artifact retention still apply.

## Recovery after a failed clip

All successfully generated clips remain in separate GitHub Actions artifacts for up to **7 days**. If a clip fails, do not restart the whole sequence blindly. The existing internal Wan workflow supports resuming from a known prior successful run by providing the committed `chain_plan`, missing `chain_step`, exact `chain_previous_run`, and original `chain_id`. The controller verifies the preceding clip's artifact and the plan digest before it continues.

The frozen plan is intentionally not editable during a running chain. A change in its contents stops continuation instead of mixing configurations mid-sequence.

## After the complete movie is assembled

Copy the completed **movie-assembly run ID** into [Smooth or speed up anime video](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/enhance-anime-video.yml) to select playback speed and output FPS independently. For longer movies, the streaming FFmpeg `motion` option avoids the RIFE mode's current 600 source-frame / 2400 output-frame limits. Interpolating the movie does not generate new Wan motion; it only creates intermediate frames.

## Implementation versions

- `.github/workflows/start-continuous-wan-scene.yml` — **v1**, 25 native GitHub inputs with 17 prompt boxes
- `scripts/wan_sequence_form.py` — **v1**, validates first N prompts and freezes one shared configuration
- `scripts/wan_chain.py` — **v4**, strict v2 plan, one external image and exact prior final-frame continuation
- `.github/workflows/generate-wan22-ti2v-turbo.yml` — **v6**, existing continuation/dispatch worker (unchanged by the form redesign)
- `docs/WAN-AUTOMATIC-CHAINS-v3.md` — this guide

**Verification scope:** All source files are committed. No new generation, workflow dispatches, training, or tests were executed during the redesign. The automatic chain transitions and final assembly must still be observed in a live run before treating them as runtime-proven.
