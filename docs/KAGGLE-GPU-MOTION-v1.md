# Kaggle GPU Motion Rendering Integration — v6

## Implementation status

Uses the existing `anime-productions-love/ai-anime-generator` pipeline:

- [Kaggle GPU Motion Renderer (v7)](../.github/workflows/kaggle-motion-render-v1.yml) — explicit submit/retrieve via GitHub Actions.
- [Bridge v4](../scripts/kaggle_motion_bridge.py) — extracts and validates Google Drive file IDs before submitting an isolated private Kaggle notebook.
- [GPU renderer v6](../kaggle/renderer/render.py) — downloads a link-viewable character image and motion MP4/animated WebP/GIF directly from Google Drive at runtime, then runs Wan2.1 VACE 1.3B with edge guidance.
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

The current experimental renderer uses Canny edge control derived from one reference video, not true multi-view motion fusion or guaranteed pose retargeting. It uses one T4 with CPU offload, even if the selected Kaggle accelerator offers two T4 cards. Initial resolution is 512x288, 17 frames, 20 inference steps. Runtime memory usage, motion fidelity, and throughput are not verified by source installation.

The existing GitHub Actions secrets `KAGGLE_API_TOKEN` (Kaggle CLI) and `VIDEO_REPO_TOKEN` (private gallery write access) remain required. Neither secret is transferred to Kaggle. The Drive URLs identify shared source files, not credentials. This revision has not triggered a Kaggle render or any tests. The negative prompt form input is wired to the existing bridge `--negative` argument. Custom negatives add to the renderer's built-in safeguards.
