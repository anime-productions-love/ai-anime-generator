# Four-Runner Anime Story Director — v1

## Purpose and current status

The [Anime Story Director four-runner workflow](../.github/workflows/anime-story-director-quad-v1.yml) deploys **one selected high-capacity text-generation model at a time**, using exactly four concurrent public GitHub-hosted Linux CPU jobs:

- Rank 0: run coordinator, source-bound prompt assembly, model retrieval, llama.cpp server, draft export
- Ranks 1–3: private `ggml-rpc-server` CPU devices that share model tensors and computation over ephemeral Tailscale connections

It supports **three separately selectable model configurations** stored in [models-v1.json](../scripts/story-director/models-v1.json):

| Model selection | Source weights | GGUF | Role |
|---|---|---|---|
| `muse-glimmer-30b` | [Meta GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) | Q4_K_M, ~16.8 GB | Creative writing and emotion |
| `qwen3.8-27b` | [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_M, ~16.5 GB | Storyboard and structured production |
| `qwen3.6-35b-a3b` | [ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) | Q4_K_M, ~20.4 GB | Efficient production planning |

All three models are configured under their exact published filenames and SHA-256 digests. Their GGUF **weights are not committed** to GitHub (too large for an ordinary source repository); selected weights are retrieved from the model provider at run time, pinned to the current immutable Hugging Face commit and checked against the registered SHA-256 for local downloads.

**Installed does not mean proven running.** The entire setup is committed but no workflow runs, tests, or training actions were triggered. Runtime viability and throughput are still unknown. The four-node llama.cpp RPC backend is experimental and requires a trusted private network.

## Why four runners and how we reused Contract-Automation

The eight-rank [Contract-Automation inference/training workflow](https://github.com/curveyield3/Contract-Automation/blob/main/.github/workflows/lite-solo-ai-inference-v1.yml) uses GitHub OIDC identities, ephemeral Tailscale peers, and runner-rank artifacts for rendezvous. The new Story Director reuses those **communications and coordination concepts**, including the existing [OIDC Tailscale helper](https://github.com/curveyield3/Contract-Automation/blob/main/scripts/browser-agent-tailnet-oidc.sh), rather than adopting its training algorithm.

A standard **public** Ubuntu GitHub-hosted job provides 4 CPU cores, 16 GB RAM, and approximately 14 GB SSD. This setup reserves **four such jobs concurrently** (16 CPU cores, 64 GB aggregate RAM) while leaving other repo concurrency capacity for rendering. They do not become a single 64 GB virtual address space.

The workflow uses pinned `llama.cpp` commit `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` (release b10430), with GGML RPC enabled, no CUDA, and a 16,384-token context window. Each remote rank exposes only its private Tailscale `100.x.x.x:50052` address. Rank 0 validates three exact same-run peer artifacts, then starts `llama-server --rpc ... --n-gpu-layers 99`. The local HTTP inference endpoint binds to `127.0.0.1:8080` and is never a public API.

GGML RPC may introduce substantial cross-runner network latency. Four runners could be **slower** than local single-node inference for smaller models. No throughput or successful model-load claim is made without running the pipeline.

## Mandatory Tailscale prerequisite

In `anime-productions-love/ai-anime-generator` **Settings → Secrets and variables → Actions**, add:

`TS_OAUTH_CLIENT_ID` — the Tailscale workload identity Client ID authorized to register `tag:ai-training` devices through GitHub Actions OIDC for this **exact repository** and branch.

The secret from Contract-Automation cannot be read or copied automatically. Your Tailscale policy must allow the new repository's OIDC issuer/claims and permit communication among temporary `tag:ai-training` nodes on TCP 50052. **No personal computer is required to be online**; all four peers live on GitHub-hosted runners.

There is no public RPC port, public tunnel, persistent device credential or training checkpoint. As in the original Contract-Automation setup, each job's Tailscale identity is ephemeral.

## Disk problem and fallback

Each Q4 GGUF (16.5–20.4 GB) exceeds GitHub's advertised 14 GB SSD budget. Four-runner **RAM aggregation does not combine disk storage**.

The coordinator removes disposable preinstalled SDKs and measures actual free scratch space before downloading. If enough space exists, it downloads the complete selected GGUF and verifies its full SHA-256. If space is too small, it requests the optional [remote-gguf-v1.py](../scripts/story-director/remote-gguf-v1.py) fallback:

1. Resolve an immutable Hugging Face commit.
2. Require the model's published SHA-256 via the source's `X-Linked-Etag` identity and exact `X-Repo-Commit`, with a bounded model file size.
3. Mount a **read-only virtual GGUF** over HTTPS byte-range reads, using the existing provider's content-addressed storage and a bounded approximately 128 MiB in-memory cache.
4. Pass that file path to the same llama.cpp server, which may then distribute tensors across the three RPC workers.

The fallback requires a usable `/dev/fuse` in the hosted runner; if GitHub prohibits FUSE or the CDN fails to honor byte ranges, the workflow stops with an explicit error rather than using corrupt model data or silently selecting a lower-quality quantization.

In the fallback, the provider-supplied pinned file digest is verified but a **full local downloaded byte stream SHA-256 is not computed**, since no full local file is present. This is weaker than the direct-download integrity check. This fallback is source-installed but has not been verified under live GitHub runner conditions. It may be slow and sensitive to CDN range performance.

## How to launch a Story Director draft

Open [Anime Story Director four-runner AI](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/anime-story-director-quad-v1.yml), select `main` then **Run workflow**.

Choose the model (Muse for creative writing, Qwen3.8 for structured film planning, Qwen3.6 for possible token-throughput efficiency), and provide:

- Story text (10–16,000 characters) or an existing `stories/...` source file path.
- Proposed number of clips (2–33), frames per clip, FPS.
- Optional animation style and character IDs from the existing anime character library.

Each job produces a **draft, not an approved movie production**. The output artifact is `anime-story-director-draft-v1-<run-id>`. Its JSON includes the model identity and immutable revision, source SHA, creative notes, intended video prompts and a clear unapproved status. Invalid or truncated model output is saved as raw response with a failure-to-parse marker; clips are **never fabricated** to produce artificial success.

There is no automatic direct rendering, approval bypass, or change to your current Wan video chain. The Story Director's draft must be reviewed before conversion into a frozen production plan.

**Privacy warning:** `ai-anime-generator` is a public repository. Do not submit unpublished private manuscripts through GitHub Actions workflow inputs, GitHub repository files or public artifacts. The Cloudflare console's authenticated private-story upload and approved-draft handoff are separate future integration tasks and are **not** implemented by this release.

## Files

- `.github/workflows/anime-story-director-quad-v1.yml` — quad inference workflow v4, source filename remains v1 for backward-compatible future updates.
- `scripts/story-director/models-v1.json` — the three selectable GGUF definitions with registered SHA-256.
- `scripts/story-director/director-v1.py` — v2 bounded prompt construction, model download, draft generation and provenance.
- `scripts/story-director/rendezvous-v1.py` — v1 exact rank rendezvous and completion monitoring.
- `scripts/story-director/join-tailnet-v1.sh` — v1 reuse of the prior ephemeral Tailscale OIDC connection.
- `scripts/story-director/remote-gguf-v1.py` — v1 HTTP-range/FUSE fallback for insufficient coordinator SSD.

All versioned source files are in `ai-anime-generator`. There are **no committed weight binaries**, no new training process, and no generated test cases.

## Limitations and approval boundary

- Actual CPU inference speed, llama.cpp RPC compatibility for each new model, Tailscale OIDC admission, FUSE support, and the 350-minute per-job runtime are **not yet runtime-verified**.
- Vision-language processing is not yet activated; this v1 distributed workflow uses the main language GGUF without its optional multimodal projector. The registry reserves projector filenames for subsequent visual-review integration.
- Each run serves **one of three models**, not all three simultaneously. Switching selected model starts a new four-runner model-load session.
- If your existing ten-concurrent-runner limit includes other jobs or the account's scheduling policy differs, running four Story Director jobs may queue or consume available capacity.
- No completed generated scene is represented as having passed quality review. Human review and explicit authorization remain required before dispatching expensive video rendering.
