# Story and Sequence Production — v2

The `stories/` folder is now the Git-backed production source for anime story illustration sequences, comics, and video handoff records.

```text
stories/<story-id>/sequences/<sequence-id>/
  manifest.json                     Canonical sequence order / current cast and scene
  shots/<shot-id>/manifest.json     Prompt, refs, selected candidate and image SHA-256
  shots/<shot-id>/selected.png      Official picture used for continuity/video export
  shots/<shot-id>/candidates/       Immutable generated candidate PNGs
  exports/export-<run-id>/         Permanent Phase 4 comic-page PNGs + snapshot manifest
  video-handoffs/handoff-<run-id>.json  Video-ready pinned source URLs + dispatch receipts
```

Use [Generate or continue anime story shot](../.github/workflows/generate-story-shot.yml) to create frames; [Select canonical image](../.github/workflows/select-story-image.yml) to choose the official image; [Publish anime comic pages and storyboard](../.github/workflows/publish-storyboard.yml) to build comic PNGs and CBZ/storyboard ZIP; and [Export or send selected anime shot to video generator](../.github/workflows/send-shot-to-video.yml) for a verified direct image URL or optional cross-repo dispatch.

All later steps use saved selected PNGs and verify SHA-256 checksums; they do not secretly change the original shot selection or generate new animation. See [Phase 4 setup](../docs/PHASE4-SETUP-v1.md).
