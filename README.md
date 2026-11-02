# AI Anime Story Generator — v1

A persistent, versioned character-and-scene library for building anime storyboards and illustrated sequences. **Phase 1** is the library foundation: create reusable character and environment records, preserve their reference images in Git, and recall them by stable ID.

**Implemented in Phase 1:** the repository asset structure; JSON manifests and JSON Schemas; a dependency-free Python CLI for create, update, reference attachment, listing and retrieval; a GitHub Actions management workflow; and a reusable anime-style preset.

**Not implemented yet:** SDXL/LoRA inference, image embedding generation, LoRA training, AI storyboard generation, shot continuation, and video export. Those belong to subsequent phases. Saving a character pack is not the same as training a character LoRA.

## Create a character or scene in GitHub

Open **[Manage character and scene library](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/manage-asset-library.yml)** → **Run workflow**. Choose `create`, an asset type (`character` or `scene`), a stable ID such as `aria-001` or `moon-forest`, and a name/description. Add a reference image URL if desired; the workflow downloads and commits the image inside the appropriate asset pack.

Repeat with `update` to change metadata or `add-reference` to attach another reference to an existing pack. References are stored in the repository, not dependent on a temporary image host. Direct image URLs must be accessible without a login.

## Recall a saved asset

After cloning the repo, run:

```bash
python3 scripts/library.py list --kind character
python3 scripts/library.py show --kind character --id aria-001
python3 scripts/library.py show --kind scene --id moon-forest
python3 scripts/library.py resolve --kind character --id aria-001
```

`resolve` emits a JSON bundle containing the saved prompt, style, attributes, and paths to the stored reference images. Phase 2 shot generation will consume these same records.

## Repo layout

```text
.github/workflows/manage-asset-library.yml   Create/update packs and references in GitHub
scripts/library.py                            Manage the local Git-backed asset library
schemas/character-v1.schema.json              Character manifest contract
schemas/scene-v1.schema.json                  Environment manifest contract
schemas/style-v1.schema.json                  Style preset contract
assets/characters/<id>/manifest.json          Canonical character identity + references
assets/characters/<id>/refs/<role>/...         Character images
assets/scenes/<id>/manifest.json              Canonical environment + references
assets/scenes/<id>/refs/<role>/...             Environment images
assets/styles/anime-cel-fantasy/manifest.json  Default anime illustration style
stories/                                      Future story/sequence assets (Phase 2)
docs/PHASE1-SETUP-v1.md                       Detailed usage instructions
```

Character and scene IDs must contain 2–64 lowercase alphanumeric, hyphen or underscore characters (begin with a letter or digit). Creating a duplicate ID fails; updating preserves its existing references. The workflow is serialized to avoid competing writes to the same repository branch.

**Reference image limits:** PNG, JPEG, GIF or WebP (up to 15 MiB). Use appropriately licensed artwork. GitHub repository history keeps committed image bytes; large libraries may eventually benefit from external object storage.

## Scope and versions

All initial Phase 1 files and manifests are **v1**. Every future revision of a delivered file should increment that file's version by one. Manifests carry a schema version and a separate content revision so you can evolve characters without silently changing their schema.

See [Phase 1 setup and API guide](docs/PHASE1-SETUP-v1.md) for full commands and workflow details.
