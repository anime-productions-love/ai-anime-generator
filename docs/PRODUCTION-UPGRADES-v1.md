# Production Reliability, Permanent Gallery Archive, Finishing, and Cloudflare Control — v1

## Installed architecture

This update retains the existing unified anime generator and adds capability directly to those workflows, without replacing the Wan engine or creating another backend.

1. **Bounded recovery:** [Recover failed continuous Wan scene v2](../.github/workflows/recover-continuous-wan.yml) observes failed Wan jobs. Eligible network, download, build, render and archive failures may be retried up to two times using GitHub's rerun-failed-jobs API. It intentionally refuses an ambiguous successor-dispatch retry to avoid duplicating scenes. A real render failure may still require rerendering the failed clip. Previously successful clips are preserved.
2. **Isolated archiving:** [Wan scene v9](../.github/workflows/generate-wan22-ti2v-turbo.yml) separates the costly 350-minute generation job from a short `archive-and-advance` job. If archiving fails, retrying that job does not rerender the successful clip.
3. **Permanent private-gallery archives:** [scripts/archive_gallery.py](../scripts/archive_gallery.py) v1 uploads a complete ZIP of output files, plus an inventory with SHA-256, into a GitHub **Release in the private `anime-productions-love/gallery` repository**. Release assets are not GitHub Actions artifacts and have no seven-day Actions expiration timer. Archive release tags are `anime-archive-<run-id>`; files are versioned and never overwritten automatically. Note that manual release deletion, GitHub account policy or service unavailability can still affect storage. The archiver verifies the remote asset name and size, not a full remote download/hash.
4. **Archive coverage:** Successful Wan scene output, final movie output, enhanced/interpolated output, storyboard/comic ZIP/CBZ and image-generation candidates are archived through the same existing script.
5. **Cache:** Wan v9 stores the pinned `stable-diffusion.cpp` CPU build and the Wan/UMT5/VAE model files in separate GitHub Actions caches. The workflow restores caches on subsequent runs and fetches missing files. This is a *performance* cache subject to GitHub's storage and eviction policy (default 10 GB repository cap and seven days of inactivity). A new/evicted cache downloads/compiles again; it does not delete generated video.
6. **Movie finishing:** The continuous-scene form offers a final interpolation preset (method, FPS, playback speed) and seam mode. The movie workflow v6 accepts `motion-seam`, removing duplicate inherited boundary frames; then it archives the combined movie and dispatches [Smooth or speed up anime video](../.github/workflows/enhance-anime-video.yml) for the selected FPS and speed. Long films exceed full-memory RIFE limits and automatically fall back to streaming FFmpeg optical-flow interpolation, with the actual method recorded in metadata. The final smoothing job has its own success/failure status and gallery archive.
7. **Cloudflare interface:** A Cloudflare Worker named `anime-production-console` has been **created and deployed**. Its protected browser dashboard supports 2–33 clip prompts, one starting image, consistent generation settings, seam handling, interpolation method, playback speed and recent workflow status. It operates by committing a frozen plan and dispatching the existing Wan GitHub workflow, not by running AI inference in Cloudflare.

## Essential Cloudflare configuration still required

The new Worker has **no secret bindings yet**. The available Cloudflare secret-writing operation was blocked by a platform safety check; do not claim the dashboard can submit runs until it is configured and a live request has been verified.

In **Cloudflare Dashboard → Workers & Pages → anime-production-console → Settings → Variables and Secrets**, add these **encrypted secrets**:

- `CONTROL_PASSWORD` — a long random password of your choice, used to sign in to the browser dashboard.
- `GITHUB_TOKEN` — a GitHub fine-grained PAT with **Contents: read/write** and **Actions: read/write** on `anime-productions-love/ai-anime-generator`. Its secret value is not available from GitHub's existing Actions secrets and must be supplied separately in Cloudflare. Do not paste it into a URL or public repository.

Your current GitHub Actions `VIDEO_REPO_TOKEN` remains in **ai-anime-generator** and is used by the actual generator jobs to read and archive media in **anime-productions-love/gallery**. The Worker does not require access to that gallery token; it only creates source-bound production plans and launches workflows.

Use the Workers & Pages dashboard to open the deployed Worker and confirm its workers.dev endpoint is enabled; no guessed workers.dev URL is provided here. Before granting access, ensure you are comfortable with password-based authentication; never share the console password publicly.

## Production workflow and acceptance

- Source launch: [Start continuous Wan anime scene](../.github/workflows/start-continuous-wan-scene.yml) v2, native GitHub form with up to 15 prompts + finishing controls, or the Cloudflare console with up to 33 prompts.
- Each clip is generated with **only the previous clip's verified last frame** and the same seed/frame/FPS/size/fit/negative settings; only the action prompt changes.
- After completed generator artifact upload, the separate archive job verifies the private gallery release before requesting the next clip.
- After final clip, the existing movie assembler joins every clip and archives the result; then automatically submits the finishing job.
- Each finished archive release remains private, with no Actions expiration. The Actions artifacts remain temporarily available for chain continuation and movie assembly.
- A successful `workflow_dispatch` POST means submission only. The console lists real GitHub statuses; a successful release upload is not an accepted render until the corresponding GitHub workflow has actually completed.

## Limitations / no runtime validation

- This revision has not launched an expensive Wan inference run, a chain, a Cloudflare console request or any new tests.
- A private gallery fine-grained PAT must include **Contents:write** on the gallery for Release uploads, not just read.
- GitHub Releases have per-asset size restrictions; an unusually large complete-output ZIP may require partitioning before archiving.
- Caching is opportunistic and can be evicted. No model or result is deleted by a cache miss.
- The GitHub Actions `workflow_run` recovery trigger and cross-repository Release upload are installed but still need live operational verification.
- Cloudflare's API credential and console password are not configured; dashboard controls are therefore presently read/write blocked behind their missing secrets.
- Existing old successful videos in `ai-video-generator` remain accessible, and the original generation workflows are not replaced.

## Future AI story planner (proposal, not deployed)

A useful next module would accept a story outline or full script, existing character and scene IDs, desired total runtime and shot/action pacing. An LLM can propose a structured list of **one motion prompt per clip** while embedding a fixed identity/setting bible into the prompts. Before dispatch, the human reviews and edits the ordered shot list, then the system saves the frozen sequence in the same plan schema already accepted by the Wan controller.

This should reuse the Cloudflare console, character/scene memory, and `wan_chain.py` pipeline. It does **not** require another scheduler or worker. It should avoid unsupervised rewriting of character identities, technical settings, or already-completed immutable segments. A model provider/token and quality requirements have not been selected, so autonomous story planning has not been activated.
