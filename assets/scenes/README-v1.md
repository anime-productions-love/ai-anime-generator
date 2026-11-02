# Scene Library — v1

Each saved environment lives in `assets/scenes/<scene-id>/manifest.json`, with permanently stored reference images under `refs/<role>/`.

Create and update scenes via [the GitHub Actions library manager](../../.github/workflows/manage-asset-library.yml), or use `python3 scripts/library.py create --kind scene ...`.

Recommended reference roles: `main`, `alternate_view`, `lighting`, `layout`, `depth`. Save environmental geometry and lighting/camera details in `description`, `prompt` and optional structured `attributes`. Scene embeddings and image generation are future phases.

See [Phase 1 setup](../../docs/PHASE1-SETUP-v1.md).
