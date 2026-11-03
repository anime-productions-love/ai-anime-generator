# Kaggle GPU Motion Rendering Integration — v9

## Implementation status

Uses the existing `anime-productions-love/ai-anime-generator` pipeline:

- [Kaggle GPU Motion Renderer (v11)](../.github/workflows/kaggle-motion-render-v1.yml) — explicit submit/retrieve via GitHub Actions.
- [Bridge v6](../scripts/kaggle_motion_bridge.py) — extracts and validates Google Drive file IDs before submitting an isolated private Kaggle notebook.
- [GPU renderer v8](../kaggle/renderer/render.py) — downloads a link-viewable character image and motion MP4/animated WebP/GIF directly from Google Drive at runtime, then runs Wan2.1 VACE 1.3B with edge guidance.
- [Existing private gallery archiver v3](../scripts/archive_gallery.py) — permanently archives retrieved, checksum-verified results.

**Kaggle dataset storage is NOT used.** No files are temporarily committed to public GitHub. There are no additional workflows, GitHub secrets or GPU systems.

## Source files and sharing

Upload two distinct files into Google Drive:

1. A still character image in PNG, JPEG or WebP format.
2. A motion reference: MP4 video, animated WebP, or animated GIF (must contain at least two frames). Static WebP and GIF files cannot act as motion references.

For **each** file select Share -> General access -> **Anyone with the link** -> **Viewer**, then copy the normal `https://drive.google.com/file/d/FILE_ID/view` link. Anyone possessing such a link can access that media; do not use it for material that must remain strictly private. Shared links are accepted by the public workflow form, so do not include any access tokens, keys, passwords or other confidential query parameters. The bridge extracts the file ID only, and the private Kaggle notebook uses `gdown` to download directly. If Google Drive blocks large file downloads or applies quotas, the render fails with a download error; no file changes are made to Drive.

No Google access credential is required for link-viewable files. Links available only to a specific Google account will not work in the Kaggle notebook.

## Submit one short render

Go to https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/kaggle-motion-render-v1.yml and choose Run workflow on main.

- `operation`: `submit`
- `kaggle_username`: `jamesnexus` (or the correct Kaggle account)
- `image_drive_url`: the copied Google Drive character-image link
- `motion_drive_url`: the copied Google Drive MP4, animated WebP, or animated GIF link
- `prompt`: a non-confidential description of the intended animation
- `negative_prompt`: optional description of artifacts to avoid, appended to built-in safeguards
- `frames`: `17` to begin (request any whole number from 5–81; bridge rounds to nearest supported 4n+1 value, with halfway values rounded upward; requested and applied counts are recorded in submission receipt and job metadata)
- `fps`: `8` for initial review

The bridge does not need the URLs until this submission form is filled in. It validates the URLs, stores only their Drive file IDs in the private Kaggle notebook code, sets `dataset_sources` to an empty list and submits a new per-run private `NvidiaTeslaT4` Kaggle kernel. Internet access is enabled for the Google Drive and model downloads. Its submit receipt is not proof of a render; check the actual Kaggle notebook status.

## Retrieve and archive

When the exact private Kaggle run completes, use the **same GitHub workflow** with `operation=retrieve` and either its original `source_run_id` plus Kaggle username, or the exact `owner/anime-motion-RUN_ID` kernel identifier.

The bridge verifies Kaggle reports completion; downloads the resulting media; requires one `video.mp4`, `last_frame.png` and `result-v1.json`; checks SHA-256 hashes and sizes; and archives a complete ZIP in the **existing private gallery** using `VIDEO_REPO_TOKEN`. It issues a non-media GitHub Actions receipt. Any generated clip requires visual review before reuse as the starting frame of the next clip.

## Limits and dependencies

The experimental renderer uses Canny-edge control derived from one reference video, not true motion retargeting or a guarantee of matching three characters. The renderer now uses validated sequential CPU offload on cuda:0 rather than unsupported mixed VAE placement. A tiled VAE reduces peak encode/decode memory and may introduce tile-edge tone differences. The successful source-bound run used 25 frames and 8 FPS. This confirms output generation and checksum integrity, not animation quality or perfect motion retargeting.

The existing GitHub Actions secrets `KAGGLE_API_TOKEN` (Kaggle CLI) and `VIDEO_REPO_TOKEN` (private gallery write access) remain required. Neither secret is transferred to Kaggle. The Drive URLs identify shared source files, not credentials. Verified success: Kaggle kernel `jamesnexus/anime-motion-render-41585877699` completed successfully (438.1-second GPU execution). The monitor GitHub run `41586403161` retrieved one result manifest, one video and final frame, checked their SHA-256 digests, and verified a permanent private gallery release containing 9 files (1,191,439 bytes). Archive: https://github.com/anime-productions-love/gallery/releases/tag/anime-archive-41586403161 . The user must review visual quality before chaining. File-triggered retry and monitor execution are supported by the existing workflow without new workflow files. The negative prompt form input is wired to the existing bridge `--negative` argument. Custom negatives add to the renderer's built-in safeguards.


## SCAIL-2 Q6_K backend — v9 (installed in source, runtime verification in progress)

The existing Kaggle workflow now supports **\`engine=scail2-q6k\`** in addition to the unchanged VACE backend. It uses the existing \`KAGGLE_API_TOKEN\` GitHub Actions secret to submit a private Kaggle kernel and \`VIDEO_REPO_TOKEN\` to recover/archive generated or failed outputs. A GitHub push with **\`[kaggle-scail]\`** in the commit message triggers a source-bound Q6_K render without workflow_dispatch. The default source image and anime driving video URLs remain the same as in the existing workflow. New workflow fields: \`engine\`, \`kaggle_datasets\` (optional semicolon-separated \`owner/slug\` strings) and \`duration_seconds\` (first-clip length 1–5 seconds).

The Kaggle notebook source is \`kaggle/renderer/scail2.py\`. It uses the existing published [ruashots/scail2-skill](https://github.com/ruashots/scail2-skill) **API-format** SCAIL-2 ComfyUI graph; only the \`DiffusionModelLoaderKJ\` input is replaced with \`UnetLoaderGGUF\` targeting **SCAIL-2-Q6_K.gguf**, and the supporting file names and three-character tracking caps are selected. It does **not** convert the animation reference into Canny edges. The image stays a full-color appearance reference and the driving anime clip is decoded as color video. SCAIL-2's built-in SAM3 + SCAIL2ColoredMask nodes provide character-correspondence masks.

The notebook pins ComfyUI, GGUF loader, VideoHelperSuite, KJNodes, Easy-Use, Custom-Scripts and the published SCAIL-2 CLI source commits. It provisions Python 3.11 when Kaggle starts with Python 3.13, using **\`uv pip\`** for installation into uv's pip-free virtual environment. It refuses to attempt model inference unless two CUDA T4s are visible. It does **not** claim or require that both GPUs perform diffusion; the model is configured with low-VRAM offload and CPU VAE.

Model files are: Q6_K (13.8GB), UMT5 XXL FP8, SAM3.1 multiplex, CLIP Vision, Wan 2.1 VAE, LightX2V rank-64 LoRA. They are reused from Kaggle's read-only \`/kaggle/input\` attachments when present, or fetched from their existing public Hugging Face publishers into ephemeral \`/kaggle/temp\` only after a real free-space check. Digests and source URLs are stored in \`scail-model-inventory-v1.json\`. **For repeatable fast starts, upload/attach those files as private Kaggle datasets and pass their existing \`owner/slug\` strings through the form; otherwise each fresh kernel may download them again.**

Rendered MP4s and per-segment outputs go under \`/kaggle/working/anime-render\`, not transient model scratch. Manifest files and installation logs are retained on failure; the existing bridge salvages and archives them in the private gallery. Successful checksums do not imply acceptable motion or visual appearance. Human review must approve any video before another clip uses its last frame. Current source v6 has not yet demonstrated a complete Q6_K inference result; refer to the latest GitHub workflow status rather than claiming acceptance from submission.
