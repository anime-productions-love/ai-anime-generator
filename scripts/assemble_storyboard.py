#!/usr/bin/env python3
"""Assemble canonical anime story shots into comic pages and storyboard assets (v1).

No AI inference is performed. Source images are accepted only from the existing
Git-backed sequence manifest, with SHA-256 matching the selected shot records.
"""
import argparse
import hashlib
import io
import json
import math
import shutil
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from library import ROOT, validate_id
from story import sequence_dir

ORIGIN = "https://raw.githubusercontent.com/anime-productions-love/ai-anime-generator"
MAX_SHOTS = 48
MAX_CAPTION = 350


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    if not path.is_file():
        raise ValueError(f"File not found: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_repo_path(path):
    if not isinstance(path, str) or not path.startswith("stories/"):
        raise ValueError("Only selected images from the story repository are accepted")
    candidate = (ROOT / path).resolve()
    if not candidate.is_relative_to(ROOT.resolve()) or not candidate.is_file():
        raise ValueError(f"Selected image missing or outside repo: {path}")
    return candidate


def get_shots(story_id, sequence_id, chosen):
    seqdir = sequence_dir(story_id, sequence_id)
    seq = load_json(seqdir / "manifest.json")
    if seq.get("schema_version") != "v1" or seq.get("story_id") != story_id or seq.get("sequence_id") != sequence_id:
        raise ValueError("Unexpected sequence manifest identity or version")
    available = seq.get("shots", [])
    if not available:
        raise ValueError("Sequence contains no completed shots")
    ids = [x.strip() for x in chosen.replace(";", ",").split(",") if x.strip()] if chosen.strip() else list(available)
    if not 1 <= len(ids) <= MAX_SHOTS or len(set(ids)) != len(ids):
        raise ValueError(f"Choose 1-{MAX_SHOTS} distinct shot IDs")
    results = []
    for shot_id in ids:
        if shot_id not in available:
            raise ValueError(f"Shot {shot_id!r} was not published in this sequence")
        shot = load_json(seqdir / "shots" / shot_id / "manifest.json")
        if shot.get("schema_version") != "v1" or shot.get("shot_id") != shot_id:
            raise ValueError(f"Invalid shot manifest for {shot_id}")
        selected = shot.get("selected_image")
        path = validate_repo_path(selected)
        if sha256(path) != shot.get("selected_sha256"):
            raise ValueError(f"Canonical shot image checksum mismatch: {selected}")
        results.append((shot_id, shot, path))
    return results


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_lines(draw, message, font_obj, max_width, max_lines=3):
    words = message.replace("\n", " ").split()
    lines = []
    line = ""
    for word in words:
        nextline = (line + " " + word).strip()
        if draw.textbbox((0, 0), nextline, font=font_obj)[2] <= max_width:
            line = nextline
        else:
            if line:
                lines.append(line)
            line = word
            if len(lines) == max_lines:
                break
    if len(lines) < max_lines and line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and len(words) > 0:
        lines[-1] = lines[-1][:max(0, len(lines[-1])-3)] + "..."
    return lines


def create_pages(shots, captions, layout, per_page, destination, title, height, width):
    if layout not in ("comic", "vertical", "grid"):
        raise ValueError("Unsupported page layout")
    if width < 800 or height < 1000:
        raise ValueError("Comic page must be at least 800x1000")
    destination.mkdir(parents=True, exist_ok=False)
    pages = []
    margin, gutter, title_height = 50, 20, 110
    columns = 1 if layout == "vertical" else 2
    if layout == "grid":
        columns = 3
    for page_number, start in enumerate(range(0, len(shots), per_page), 1):
        page_shots = shots[start:start+per_page]
        rows = math.ceil(len(page_shots) / columns)
        content_width = width - margin*2
        content_height = height - margin*2 - title_height
        cell_w = (content_width - gutter*(columns-1)) // columns
        cell_h = (content_height - gutter*(rows-1)) // rows
        if cell_w < 160 or cell_h < 190:
            raise ValueError("Too many panels for the requested page size/layout")
        image = Image.new("RGB", (width, height), "#f8f7f4")
        d = ImageDraw.Draw(image)
        heading = font(37)
        label = font(20)
        caption_font = font(20)
        d.text((margin, 35), title[:65], fill="#151515", font=heading)
        d.text((width-margin-140, 50), f"PAGE {page_number}", fill="#555555", font=label)
        for index, (shot_id, shot, source) in enumerate(page_shots):
            local = start+index
            col = index % columns
            row = index // columns
            x = margin + col*(cell_w+gutter)
            y = margin+title_height + row*(cell_h+gutter)
            d.rectangle([x, y, x+cell_w, y+cell_h], outline="#171717", width=4)
            reserved_bottom = 86 if captions[local] else 34
            picture_box = (cell_w-16, cell_h-reserved_bottom-10)
            with Image.open(source) as opened:
                picture = ImageOps.contain(opened.convert("RGB"), picture_box,
                                           method=Image.Resampling.LANCZOS)
            # White letterboxing: do not crop canonical source artwork.
            px = x + (cell_w-picture.width)//2
            py = y + 10 + (picture_box[1]-picture.height)//2
            image.paste(picture, (px, py))
            d.text((x+10,y+cell_h-31), f"SHOT {shot_id}", fill="#303030", font=label)
            if captions[local]:
                d.rectangle([x+6, y+cell_h-reserved_bottom,
                             x+cell_w-6, y+cell_h-35], fill="#ffffff")
                lines = text_lines(d, captions[local], caption_font, cell_w-30, 2)
                for idx,line in enumerate(lines):
                    d.text((x+14, y+cell_h-reserved_bottom+5+idx*24),
                           line, fill="#171717", font=caption_font)
        path = destination / f"page-{page_number:03d}.png"
        image.save(path, optimize=True)
        pages.append(path)
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--shot-ids", default="", help="Comma-separated IDs, blank for all")
    parser.add_argument("--captions", default="", help="Optional per-shot caption text separated by ||")
    parser.add_argument("--layout", choices=("comic", "vertical", "grid"), default="comic")
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--page-size", choices=("portrait", "landscape"), default="portrait")
    parser.add_argument("--export-id", required=True)
    args = parser.parse_args()

    story_id = validate_id(args.story_id)
    sequence_id = validate_id(args.sequence_id)
    if not args.export_id.isdigit() or len(args.export_id) > 24:
        parser.error("--export-id must be a numeric GitHub run ID")
    if not 1 <= args.per_page <= 8:
        parser.error("Panels per page must be 1-8")
    shots = get_shots(story_id, sequence_id, args.shot_ids)
    captions = [x.strip() for x in args.captions.split("||")] if args.captions else []
    if captions and len(captions) != len(shots):
        parser.error("Number of ||-separated captions must equal selected shot count")
    if not captions:
        captions = [""] * len(shots)
    if any(len(c) > MAX_CAPTION for c in captions):
        parser.error(f"Each caption must be no longer than {MAX_CAPTION} characters")

    export_dir = sequence_dir(story_id,sequence_id) / "exports" / ("export-"+args.export_id)
    if export_dir.exists():
        raise ValueError("This export ID already exists; refusing to overwrite")
    export_dir.mkdir(parents=True)
    dimensions = (1400, 2000) if args.page_size == "portrait" else (2000, 1400)
    pages = create_pages(
        shots, captions, args.layout, args.per_page, export_dir / "pages",
        f"{story_id} / {sequence_id}", dimensions[1], dimensions[0],
    )
    export_manifest = {
        "schema_version": "v1",
        "export_id": "export-"+args.export_id,
        "story_id": story_id, "sequence_id": sequence_id,
        "layout": args.layout, "per_page": args.per_page,
        "page_size": args.page_size,
        "shots": [],
        "pages": [],
    }
    for idx, (shot_id, shot, selected_path) in enumerate(shots):
        export_manifest["shots"].append({
            "order": idx+1, "shot_id": shot_id,
            "canonical_path": selected_path.relative_to(ROOT).as_posix(),
            "sha256": shot["selected_sha256"],
            "caption": captions[idx],
            "characters": shot.get("characters", []),
            "scene_id": shot.get("scene_id"),
            "prompt": shot.get("prompt", ""),
            "negative_prompt": shot.get("composed_negative_prompt", ""),
            "video_image_path": selected_path.relative_to(ROOT).as_posix(),
        })
    for p in pages:
        export_manifest["pages"].append({
            "path": p.relative_to(ROOT).as_posix(),
            "sha256": sha256(p),
        })
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Export bundle is run artifact only; canonical selected images/page PNGs
    # and the export manifest remain Git-backed.
    artifact = ROOT / "work" / "production"
    artifact.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact / "storyboard-v1.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname="storyboard.json")
        for shot_id, _, selected_path in shots:
            archive.write(selected_path, arcname=f"images/shot-{shot_id}.png")
        for p in pages:
            archive.write(p, arcname="comic/"+p.name)
    with zipfile.ZipFile(artifact / "comic-pages-v1.cbz", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in pages:
            archive.write(p, arcname=p.name)
    # Text/JSON only handoff, no public link minted until Git commit succeeds.
    (artifact/"export-info.json").write_text(
        json.dumps({"version":"v1","manifest_path":manifest_path.relative_to(ROOT).as_posix(),
                    "page_count":len(pages),"shot_count":len(shots)}, indent=2)+"\n",
        encoding="utf-8",
    )
    print(f"Created {len(pages)} comic page(s) from {len(shots)} verified selected shots")
    print(f"Saved permanent output under {export_dir.relative_to(ROOT)}")


if __name__=="__main__":
    main()
