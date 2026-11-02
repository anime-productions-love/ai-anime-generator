# Character Library — v1

Each saved character lives in `assets/characters/<character-id>/manifest.json`, with any permanent reference images under `refs/<role>/`.

Create and update characters via [the GitHub Actions library manager](../../.github/workflows/manage-asset-library.yml), or use `python3 scripts/library.py create --kind character ...`.

Recommended reference roles: `face`, `three_quarter`, `full_body`, `expression`, `outfit`. Attach multiple perspectives of the same character, using the same stable ID. Character LoRA training is **not** part of Phase 1; the manifest's adapter fields are placeholders.

See [Phase 1 setup](../../docs/PHASE1-SETUP-v1.md).
