# Kaggle GPU Motion Rendering Integration — v1

## Implementation status

Source-installed into the EXISTING \`anime-productions-love/ai-anime-generator\` repository:

- [Kaggle GPU Motion Renderer (v2)](../.github/workflows/kaggle-motion-render-v1.yml) — explicit GitHub Actions submission/retrieval; NOT triggered by pushes.
- [Bridge v2](../scripts/kaggle_motion_bridge.py) — official Kaggle CLI authentication, private per-run kernel metadata, submit, status parsing, retrieval and SHA-256 integrity.
- [Experimental GPU renderer v3](../kaggle/renderer/render.py) — Wan2.1 VACE 1.3B Diffusers + image and motion-control edges.
- [Existing private gallery archiver v3](../scripts/archive_gallery.py) — extended with the kaggle kind; no replacement archive system.

**No Kaggle job has been submitted or rendered by this installation.** Kaggle token secret and gallery secret contents cannot be read back through the GitHub connector. Kernel execution, model download, allocated GPUs, motion fidelity and inference throughput are unverified.

## External one-time prerequisites

1. GitHub Actions repository secret \`KAGGLE_API_TOKEN\` containing your Kaggle API token. Never paste it in a workflow input or public repo file.
2. Kaggle account phone verification and access to GPU T4 x2.
3. A Kaggle **private dataset** containing two files at its root: e.g. \`character.png\` (the target anime character image) and \`motion.mp4\` (the short driving motion example). Its Kaggle slug has the form \`YOUR_USERNAME/YOUR_DATASET\`. Images and footage must not be committed to the public anime-generator repository.
4. For permanent private video retention, the existing GitHub Actions secret \`VIDEO_REPO_TOKEN\` must have Contents:write on the private \`anime-productions-love/gallery\` repository.

## Submit exactly one motion render

Visit https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/kaggle-motion-render-v1.yml and select Run workflow (main branch).

- \`operation\`: \`submit\`
- \`kaggle_username\`: optional when the Kaggle authenticated CLI can identify the account; otherwise supply the profile username (never a secret).
- \`dataset_source\`: private Kaggle dataset, e.g. \`owner/private-anime-assets\`
- \`image_file\`: \`character.png\`
- \`motion_file\`: \`motion.mp4\`
- \`prompt\`: a non-confidential short motion description; public GitHub Actions workflow inputs are not private.
- \`frames\`: default 17, and must be 4n+1, 5–81.
- \`fps\`: default 8.

The workflow creates a unique private kernel \`<kaggle_username>/anime-motion-<GITHUB_RUN_ID>\`, requests \`NvidiaTeslaT4\` (the dual T4 pool), submits the Python renderer via \`kaggle kernels push\`, and uploads **only a submission receipt**, not source media. Submission success **is not evidence that Kaggle rendered a video**. Watch the submitted kernel status at https://www.kaggle.com/code/.

## Retrieve exactly that completed run

On the same GitHub workflow, choose \`operation=retrieve\` and provide either:

- \`source_run_id\`: the original GitHub **submission** run ID (and \`kaggle_username\` if Kaggle does not expose it via CLI), OR
- \`kernel\`: the exact \`owner/anime-motion-<run-id>\` identifier.

The bridge checks Kaggle's remote status. If queued, running, failed or unknown, it stops and does **not** download misleading placeholder output. For a completed run it downloads Kaggle's files, requires exactly one \`video.mp4\`, \`last_frame.png\` and \`result-v1.json\`, verifies their source-issued SHA-256 hashes and minimal sizes, creates \`bridge-result-v1.json\`, and archives the verified set using the existing private gallery release process. The public GitHub repository artifact contains only the **non-media** receipt.

The stored result status is \`RETRIEVED_VERIFIED_NEEDS_HUMAN_REVIEW\`; it is **not** approved to provide the next chained scene's starting frame. Human review remains necessary, especially for malformed anatomy and unintended violence.

## Rendering scope / limitations

The initial renderer uses official \`Wan-AI/Wan2.1-VACE-1.3B-diffusers\` with a reference character image and an *experimental Canny-edge conditioning sequence derived from the driving video*. It is NOT a full body-pose/face motion-transfer implementation, so faithful animation retargeting is not guaranteed. It deliberately requests a small 512x288, 17-frame, 20-step render by default to limit initial GPU consumption. The initial notebook uses **one T4** with model CPU offload; the second T4 is not currently used by the Diffusers pipeline. Do not treat 2 x 16 GB as a single 32 GB VRAM pool. Runtime memory and output quality remain unverified.

The first image of the video is supplied as an explicit visual conditioning frame. Every newly generated video remains **unreviewed** until approved. No existing Wan auto-chain is dispatched; no episode rendering, human approval bypass or automatic distribution of private API tokens to Kaggle occurs.

After one short working sample, the model/backend can be replaced (for example, pose-controlled VACE or a compatible Wan Animate implementation) without rebuilding the Kaggle submission/retrieval workflow. That upgrade is NOT installed now.

## Common failure cases

- **Kaggle authentication**: verify \`KAGGLE_API_TOKEN\` was added in Settings → Secrets and variables → Actions on the anime-generator repository.
- **Kaggle username unavailable**: fill in the \`kaggle_username\` input; the token itself must remain secret.
- **No asset dataset**: create a private Kaggle dataset with the exact input filenames at its root and specify its slug.
- **Kernel still running**: do not submit a new render; use \`operation=retrieve\` again after the exact Kaggle kernel completes.
- **CUDA unavailable / GPU quota exhausted**: inspect the Kaggle notebook version status; the bridge does not fall back to slow CPU inference.
- **Out of VRAM or model download failure**: this initial GPU renderer is experimental; inspect Kaggle logs before changing model or quantization.
- **Gallery archive failure**: the GitHub workflow must have \`VIDEO_REPO_TOKEN\` granting contents write to the private gallery; verified Kaggle results remain available at Kaggle until retrieved/archived.

Official references: Kaggle CLI kernels docs at https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md and Diffusers WanVACEPipeline docs at https://huggingface.co/docs/diffusers/main/api/pipelines/wan.
