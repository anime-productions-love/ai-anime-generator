# Phase 1 — Persistent Anime Character and Scene Libraries (v1)

## What is now implemented

Phase 1 provides a Git-backed, reusable library for the eventual illustrated-story generator. It stores **characters**, **scenes**, their descriptive prompts and attributes, and actual reference-image bytes. Recalling an asset by its stable ID returns everything Phase 2 will need to construct image-generation references. A default fantasy-anime style preset is included.

**This phase does not generate images, train character LoRAs, produce IP-Adapter embeddings, score images or assemble comics.** The LoRA/embedding fields in a manifest are explicitly empty placeholders until those capabilities exist.

## Fastest way: GitHub Actions interface

Go to [Manage character and scene library (v1)](https://github.com/anime-productions-love/ai-anime-generator/actions/workflows/manage-asset-library.yml), select **Run workflow** on the `main` branch, and enter the fields:

| Field | Example | Description |
|---|---|---|
| `operation` | `create` | `create`, `update`, or `add-reference` |
| `kind` | `character` | `character` or `scene` |
| `asset_id` | `aria-001` | Stable ID. Same ID is used to recall this asset later |
| `name` | `Aria` | Display name; required when creating |
| `description` | `Silver-haired fantasy heroine, long black cloak` | Visual identity or scene facts |
| `prompt` | `Silver hair, blue eyes, black cloak...` | Canonical visual prompt reused later |
| `negative_prompt` | `wrong hair color, missing cloak...` | Features the model should not change |
| `tags` | `heroine,fantasy,protagonist` | Comma-separated lowercase tags |
| `image_url` | `https://example.org/aria.png` | Optional *direct public HTTPS image* (15 MiB maximum) |
| `image_role` | `face` | `face`, `full_body`, `expression`, `outfit`, `main`, `alternate_view`, etc. |

1. **Create character**: `create`, `character`, ID `aria-001`, name `Aria`, and a descriptive prompt. Optionally attach your first image.
2. **Attach more character angles**: `add-reference`, `character`, same ID, a second image URL, role `full_body` or `three_quarter`. This *adds* the reference without replacing the earlier one.
3. **Create a scene**: `create`, `scene`, ID `moon-forest`, name `Moonlit Forest Clearing`. Add an environment prompt and a background photo/illustration with role `main`.
4. **Update character/scene details**: `update` with the same kind/ID and the fields to change. Leaving fields blank during updates means **keep previous values**.
5. Reuse each character or environment later by its ID, `aria-001` and `moon-forest`.

Each successful management workflow commits the new or updated manifest and any reference images to the selected repository branch. You do not need to upload a reference PNG manually if you supply a publicly downloadable URL. The workflow serializes writes on the same branch to avoid collisions.

**Important:** This is a **public GitHub repository**. Images and prompts you save through the workflow become public, including their past Git history. Only supply artwork you are comfortable publishing and have rights to use.

## Direct Python CLI

The CLI uses only Python 3.11+ standard library. Run commands from a cloned working copy of this repository.

**Create a character:**

```bash
python3 scripts/library.py create --kind character --id aria-001 \
  --name "Aria" \
  --description "Blue-eyed, silver-haired anime heroine in a black cloak" \
  --prompt "Aria, blue eyes, long silver hair, fitted black traveling cloak" \
  --negative-prompt "wrong hair color, wrong cloak, extra limbs" \
  --tags "heroine,fantasy"
```

**Create a recurring setting:**

```bash
python3 scripts/library.py create --kind scene --id moon-forest \
  --name "Moonlit Forest Clearing" \
  --description "A small clearing surrounded by tall silver-barked trees" \
  --prompt "Deep blue moonlight, pale mist, glowing motes, consistent tree layout" \
  --tags "forest,night,fantasy"
```

**Attach a local image or a public HTTPS image URL** (use exactly one of `--file` or `--image-url`):

```bash
python3 scripts/library.py add-reference --kind character --id aria-001 \
  --file ./my-artwork/aria-front.png --role face

python3 scripts/library.py add-reference --kind scene --id moon-forest \
  --image-url "https://example.org/my-forest-background.png" --role main
```

`add-reference` copies the real PNG/JPEG/GIF/WebP bytes into the pack under `refs/ROLE/CONTENT_HASH.ext`. It records SHA-256, role, original source (without URL query secrets), and timestamp. Adding the *same role + identical image bytes* twice does not duplicate the reference. Adding a second view produces a separate reference.

**Update existing metadata:**

```bash
python3 scripts/library.py update --kind character --id aria-001 \
  --prompt "Aria, silver hair, blue eyes, black cloak with silver embroidery"

python3 scripts/library.py update --kind character --id aria-001 \
  --attributes-json '{"appearance":{"hair":"silver","eyes":"blue"},"outfits":{"travel":"black cloak"}}'
```

`--attributes-json` is for richer structured identity/environment properties. In v1, setting attributes replaces the `attributes` object as a whole. The standard GitHub workflow exposes the plain text fields; advanced attributes can be edited through the Python CLI and committed with Git.

**List, inspect, and resolve saved assets:**

```bash
python3 scripts/library.py list --kind character
python3 scripts/library.py list --kind scene
python3 scripts/library.py show --kind character --id aria-001
python3 scripts/library.py resolve --kind scene --id moon-forest
```

`resolve` reads the saved pack, checks each reference file against its stored SHA-256 digest, includes the resolved repository paths, and loads the referenced style preset. It prints JSON on stdout so a future shot generator can consume it directly. No network calls are needed for existing packs and their saved images.

## Canonical repository structure

```text
assets/
  characters/
    aria-001/
      manifest.json
      refs/
        face/<sha20>.png
        full_body/<sha20>.png
  scenes/
    moon-forest/
      manifest.json
      refs/
        main/<sha20>.png
  styles/
    anime-cel-fantasy/
      manifest.json
schemas/
  character-v1.schema.json
  scene-v1.schema.json
  style-v1.schema.json
scripts/
  library.py
.github/workflows/
  manage-asset-library.yml
stories/
  [reserved for Phase 2 stories and shot sequences]
```

This structure is illustrative: assets are created when you add them. The default style preset is already committed. Character/scene model packages are **identity records + references**, not trained neural weights yet.

## Manifest contract

Each character and scene pack uses its corresponding `schemas/*-v1.schema.json` contract. The canonical `manifest.json` contains:

- `schema_version: "v1"`, `revision: 1+` (revision increments for content updates)
- `kind`, stable `id`, display `name`, and `description`
- `style_id` (initially `anime-cel-fantasy`)
- `prompt`, `negative_prompt`, `tags`
- `attributes`: optional structured details
- `references`: role/path/SHA-256/source/timestamp
- `adapters`: null slots reserved for later LoRA and IP-Adapter additions
- `created_at` and `updated_at` timestamps

Manifests live in Git and are the **source of truth**; generation artifacts, caches, and model checkpoints are *not* authoritative character memory. Phase 2 should load these saved manifests directly, rather than inventing a parallel database.

## Access control and constraints

- Asset IDs are lowercase and path-safe (2–64 characters); no arbitrary paths are accepted for pack identifiers.
- Downloaded reference files must be recognized PNG/JPEG/GIF/WebP, at most 15 MiB, from a direct public HTTPS URL. The downloader rejects local/private IP addresses, nonstandard ports, HTML responses and redirects to private HTTP origins.
- For local CLI `--file` uploads, only copies of the actual bytes are committed, not references to temporary paths.
- GitHub Actions needs repository **Contents: write** permission to commit updates. The workflow requests it explicitly, but organization policies or branch protections may prevent writes.
- File changes are versioned by Git history; a content revision number is recorded in each manifest. The initial schema version stays `v1` until a deliberate schema migration.
- No default workflow automatically executes tests, trains models, or generates images.

## Next phases (not included)

**Phase 2:** SDXL anime image inference, character/scene reference loading, shot/sequence manifests, continuity inheritance and selected output retention.

**Phase 3:** LoRA training, IP-Adapter embedding caching and continuity evaluation.

**Phase 4:** Comic page assembly, storyboard export and video-generator integration.

## Implementation status

Phase 1 source files, the style preset, and management workflow have been committed. No GitHub Actions workflow execution or model training has been run as part of Phase 1. The practical create/update-from-GitHub interface remains unverified until its first actual dispatch; source-level retrieval does not demonstrate a completed write to GitHub Actions.
