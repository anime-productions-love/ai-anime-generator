#!/usr/bin/env python3
"""Anime shot/sequence manifests: prepare, publish, and select (v2).

Uses Phase 1 character/scene manifests as canonical identity records.
No model downloads and no network I/O; all work uses local Git-backed files.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from library import ROOT, read_pack, pack_path, validate_id, DEFAULT_STYLE

SCHEMA = "v1"
MAX_CHARACTERS = 6


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_dir(story, sequence):
    return ROOT / "stories" / validate_id(story, "story ID") / "sequences" / validate_id(sequence, "sequence ID")


def load_json(file):
    if not file.is_file():
        raise ValueError(f"Required file not found: {relative(file)}")
    return json.loads(file.read_text(encoding="utf-8"))


def required_reference_images(kind, data):
    references = []
    base = pack_path(kind, data["id"])
    for item in data.get("references", []):
        path = (base / item["path"]).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe {kind} reference: {item['path']}")
        actual = digest(path)
        if actual != item["sha256"]:
            raise ValueError(f"Modified {kind} reference: {item['path']}")
        references.append({
            "kind": kind, "asset_id": data["id"], "role": item["role"],
            "path": relative(path), "sha256": actual,
        })
    return references


def character_ids(value):
    ids = [validate_id(x.strip(), "character ID") for x in value.split(",") if x.strip()]
    if len(ids) > MAX_CHARACTERS:
        raise ValueError(f"Maximum {MAX_CHARACTERS} characters per shot")
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated character ID in a single shot")
    return ids


def optional_attr_notes(asset):
    attributes = asset.get("attributes") or {}
    if not attributes:
        return ""
    return json.dumps(attributes, ensure_ascii=False, sort_keys=True)


def prepare(args):
    directory = sequence_dir(args.story_id, args.sequence_id)
    seq_path = directory / "manifest.json"
    exists = seq_path.exists()
    if args.mode == "start" and exists:
        raise ValueError(
            "This sequence already exists. Choose 'continue' to append a shot "
            "or use a new sequence ID; prior selected images are immutable by generation."
        )
    if args.mode == "continue" and not exists:
        raise ValueError("No such sequence. Use mode 'start' for the first shot.")

    previous = load_json(seq_path) if exists else None
    if previous and (previous.get("schema_version") != SCHEMA
                     or previous.get("story_id") != args.story_id
                     or previous.get("sequence_id") != args.sequence_id):
        raise ValueError("Unrecognized sequence manifest version or identity")

    inherited_ids = previous["current_character_ids"] if previous else []
    inherited_scene = previous["current_scene_id"] if previous else None
    ids = character_ids(args.characters) if args.characters.strip() else inherited_ids
    scene_id = validate_id(args.scene_id, "scene ID") if args.scene_id else inherited_scene
    if not args.prompt.strip():
        raise ValueError("Shot action/visual prompt must not be empty")

    characters = [read_pack("character", identifier) for identifier in ids]
    scene = read_pack("scene", scene_id) if scene_id else None
    style_id = scene["style_id"] if scene else (characters[0]["style_id"] if characters else DEFAULT_STYLE)
    style = load_json(ROOT / "assets" / "styles" / style_id / "manifest.json")
    if style.get("schema_version") != "v1" or style.get("kind") != "style":
        raise ValueError("Unsupported style manifest")

    assets = [*characters, *([scene] if scene else [])]
    references = []
    for asset in assets:
        references.extend(required_reference_images(asset["kind"], asset))

    previous_shot = previous["selected_shot_id"] if previous else None
    previous_image = None
    if previous_shot:
        shot = load_json(directory / "shots" / previous_shot / "manifest.json")
        file = ROOT / shot["selected_image"]
        if not file.is_file() or digest(file) != shot["selected_sha256"]:
            raise ValueError("Previous selected shot image missing or has changed")
        previous_image = relative(file)

    # The entire reference library is recalled and recorded. Phase 2 can
    # condition on ONE image using SDXL img2img; multi-reference IP-Adapter
    # identity conditioning and LoRA are separate planned Phase 3 features.
    input_path = previous_image
    reference_source = "previous-selected-shot" if previous_image else "none"
    if not input_path:
        preference = [
            ("scene", ("main", "background", "layout")),
            ("character", ("full_body", "three_quarter", "main", "face")),
        ]
        for kind, roles in preference:
            if input_path:
                break
            for role in roles:
                match = next(
                    (r for r in references if r["kind"] == kind and r["role"] == role),
                    None,
                )
                if match:
                    input_path, reference_source = match["path"], f"{kind}:{role}"
                    break
        if not input_path and references:
            input_path, reference_source = references[0]["path"], "reference:fallback"

    # Phase 3: record bounded, verified references for actual IP-Adapter /
    # ControlNet conditioning and only registered character LoRA repositories.
    ip_refs = []
    for identifier in ids:
        choices = [r for r in references if r["kind"] == "character" and r["asset_id"] == identifier]
        for role in ("face", "three_quarter", "full_body", "main"):
            candidate = next((r for r in choices if r["role"] == role), None)
            if candidate:
                ip_refs.append(candidate)
                break
    if scene:
        choices = [r for r in references if r["kind"] == "scene"]
        scene_ref = next((r for role in ("main", "background")
                          for r in choices if r["role"] == role), None)
        if scene_ref:
            ip_refs.append(scene_ref)
    # Keep the CPU pipeline bounded: one reference per selected cast member,
    # with a maximum of four for the single IP-Adapter input.
    ip_refs = ip_refs[:4]
    controls = [r for r in references if r["kind"] == "scene" and
                r["role"] in ("canny", "layout", "edges")]
    control_ref = controls[0] if controls else None
    registered_loras = []
    for char in characters:
        ref = (char.get("adapters") or {}).get("character_lora")
        if ref:
            if not re.fullmatch(r"hf://[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", ref):
                raise ValueError(f"Invalid registered LoRA for {char['id']}: expected hf://namespace/repository")
            registered_loras.append({"character_id": char["id"], "repository": ref[5:]})

    negative_parts = [style.get("negative_prompt", "")]
    positive_parts = [style.get("prompt_prefix", ""), "masterpiece, best quality, cinematic anime storyboard"]
    if scene:
        positive_parts += [f"Location: {scene['name']}.", scene.get("description", ""), scene.get("prompt", "")]
        if optional_attr_notes(scene):
            positive_parts.append("Scene properties: " + optional_attr_notes(scene))
        negative_parts.append(scene.get("negative_prompt", ""))
    for character in characters:
        positive_parts += [f"Character {character['name']}:", character.get("description", ""), character.get("prompt", "")]
        if optional_attr_notes(character):
            positive_parts.append("Character identity: " + optional_attr_notes(character))
        negative_parts.append(character.get("negative_prompt", ""))
    positive_parts += [args.prompt.strip()]
    positive = ", ".join(x.strip().strip(",") for x in positive_parts if x and x.strip())
    negative = ", ".join(x.strip().strip(",") for x in negative_parts if x and x.strip())

    next_number = len(previous["shots"]) + 1 if previous else 1
    shot_id = f"{next_number:04d}"
    shot_dir = directory / "shots" / shot_id
    if shot_dir.exists():
        raise ValueError(f"Shot {shot_id} already exists; refusing to overwrite")

    if args.width % 8 or args.height % 8 or not 256 <= args.width <= 1536 or not 256 <= args.height <= 1536:
        raise ValueError("Width/height must be multiples of 8, 256–1536")
    if not 1 <= args.candidates <= 4:
        raise ValueError("Candidate count must be from 1 to 4")
    if not 0 <= args.seed <= 2_147_483_647:
        raise ValueError("Seed must be between 0 and 2147483647")

    plan = {
        "schema_version": SCHEMA,
        "story_id": args.story_id, "sequence_id": args.sequence_id,
        "shot_id": shot_id, "mode": args.mode, "prompt": args.prompt.strip(),
        "composed_prompt": positive, "composed_negative_prompt": negative,
        "character_ids": ids, "scene_id": scene_id, "style_id": style_id,
        "asset_revisions": {
            **{f"character:{x['id']}": x["revision"] for x in characters},
            **({f"scene:{scene['id']}": scene["revision"]} if scene else {}),
        },
        "references": references, "continuity_from_shot": previous_shot,
        "conditioning_image": input_path, "conditioning_source": reference_source,
        "strength": 0.55 if previous_image else (0.70 if input_path else None),
        "width": args.width, "height": args.height,
        "steps": args.steps, "cfg": args.cfg, "seed": args.seed,
        "candidates": args.candidates,
        "guidance_mode": args.guidance_mode,
        "ip_adapter_references": ip_refs,
        "control_reference": control_ref,
        "character_loras": registered_loras,
    }
    if not 4 <= args.steps <= 100 or not 1.0 <= args.cfg <= 15.0:
        raise ValueError("Steps must be 4–100 and guidance CFG must be 1–15")
    write_json(args.plan, plan)
    print(f"Prepared {args.story_id}/{args.sequence_id}/shot-{shot_id}; "
          f"{len(ids)} characters, scene={scene_id}, "
          f"visual conditioning={reference_source}", flush=True)


def publish(args):
    plan = load_json(args.plan)
    directory = sequence_dir(plan["story_id"], plan["sequence_id"])
    seq_path = directory / "manifest.json"
    exists = seq_path.exists()
    if exists != (plan["mode"] == "continue"):
        raise ValueError("Sequence changed since preparation; refusing to publish")
    seq = load_json(seq_path) if exists else None
    shot_id = plan["shot_id"]
    if seq:
        expected = f"{len(seq['shots']) + 1:04d}"
        if shot_id != expected:
            raise ValueError("Sequence advanced after preparation; refusing stale publish")
    out = directory / "shots" / shot_id
    if out.exists():
        raise ValueError("Shot already published; refusing overwrite")
    input_dir = args.candidate_dir
    source_files = [input_dir / f"candidate-{x:02d}.png" for x in range(1, plan["candidates"] + 1)]
    if any(not x.is_file() or x.stat().st_size == 0 for x in source_files):
        raise ValueError("Model did not output all expected candidate PNG files")

    out.mkdir(parents=True)
    candidates_dir = out / "candidates"
    candidates_dir.mkdir()
    image_records = []
    for index, src in enumerate(source_files, 1):
        dst = candidates_dir / f"candidate-{index:02d}.png"
        shutil.copyfile(src, dst)
        image_records.append({"index": index, "path": relative(dst), "sha256": digest(dst)})
    selected = out / "selected.png"
    shutil.copyfile(out / "candidates" / "candidate-01.png", selected)
    shot = {
        "schema_version": SCHEMA, "revision": 1,
        "story_id": plan["story_id"], "sequence_id": plan["sequence_id"],
        "shot_id": shot_id, "prompt": plan["prompt"],
        "composed_prompt": plan["composed_prompt"],
        "composed_negative_prompt": plan["composed_negative_prompt"],
        "characters": plan["character_ids"], "scene_id": plan["scene_id"],
        "style_id": plan["style_id"], "asset_revisions": plan["asset_revisions"],
        "reference_images": plan["references"],
        "continuity_from_shot": plan["continuity_from_shot"],
        "conditioning_image": plan["conditioning_image"],
        "conditioning_source": plan["conditioning_source"],
        "conditioning_strength": plan["strength"],
        "model_id": "cagliostrolab/animagine-xl-3.1", "backend": "diffusers-cpu-bf16",
        "width": plan["width"], "height": plan["height"], "steps": plan["steps"],
        "cfg": plan["cfg"], "seed": plan["seed"],
        "candidates": image_records, "selected_candidate": 1,
        "selected_image": relative(selected), "selected_sha256": digest(selected),
        "created_at": now(), "updated_at": now(),
        "guidance": {
            "mode": plan.get("guidance_mode", "base"),
            "ip_adapter_references": plan.get("ip_adapter_references", []),
            "control_reference": plan.get("control_reference"),
            "character_loras": plan.get("character_loras", []),
        },
        "continuity_evaluation": (
            load_json(args.evaluation) if args.evaluation and args.evaluation.is_file()
            else None
        ),
    }
    write_json(out / "manifest.json", shot)
    if not seq:
        seq = {
            "schema_version": SCHEMA, "revision": 1, "story_id": plan["story_id"],
            "sequence_id": plan["sequence_id"], "shots": [], "selected_shot_id": None,
            "current_character_ids": [], "current_scene_id": None,
            "created_at": now(), "updated_at": now(),
        }
    else:
        seq["revision"] += 1
        seq["updated_at"] = now()
    seq["shots"].append(shot_id)
    seq["selected_shot_id"] = shot_id
    seq["current_character_ids"] = plan["character_ids"]
    seq["current_scene_id"] = plan["scene_id"]
    write_json(seq_path, seq)
    print(f"Published shot {shot_id}, {len(image_records)} candidates; "
          "candidate 1 is initially canonical.", flush=True)


def select(args):
    directory = sequence_dir(args.story_id, args.sequence_id)
    seq = load_json(directory / "manifest.json")
    if args.shot_id not in seq["shots"]:
        raise ValueError("Shot not in sequence")
    path = directory / "shots" / args.shot_id / "manifest.json"
    shot = load_json(path)
    choice = args.candidate
    candidate = next((c for c in shot["candidates"] if c["index"] == choice), None)
    if candidate is None:
        raise ValueError(f"Candidate {choice} does not exist for this shot")
    src = ROOT / candidate["path"]
    if not src.is_file() or digest(src) != candidate["sha256"]:
        raise ValueError("Candidate image is missing or has changed")
    dst = ROOT / shot["selected_image"]
    if shot["selected_candidate"] == choice and dst.exists() and digest(dst) == candidate["sha256"]:
        print("Candidate already canonical; no change")
        return
    shutil.copyfile(src, dst)
    shot["selected_candidate"] = choice
    shot["selected_sha256"] = digest(dst)
    shot["revision"] += 1
    shot["updated_at"] = now()
    write_json(path, shot)
    print(f"Selected shot {args.shot_id} candidate {choice}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="operation", required=True)
    prepare_cmd = cmd.add_parser("prepare")
    prepare_cmd.add_argument("--mode", choices=("start", "continue"), required=True)
    prepare_cmd.add_argument("--story-id", required=True)
    prepare_cmd.add_argument("--sequence-id", required=True)
    prepare_cmd.add_argument("--characters", default="")
    prepare_cmd.add_argument("--scene-id", default="")
    prepare_cmd.add_argument("--prompt", required=True)
    prepare_cmd.add_argument("--width", type=int, default=768)
    prepare_cmd.add_argument("--height", type=int, default=1024)
    prepare_cmd.add_argument("--steps", type=int, default=28)
    prepare_cmd.add_argument("--cfg", type=float, default=6.5)
    prepare_cmd.add_argument("--seed", type=int, default=42)
    prepare_cmd.add_argument("--candidates", type=int, default=1)
    prepare_cmd.add_argument("--guidance-mode",
                             choices=("base", "ip-adapter", "controlnet", "both"),
                             default="base")
    prepare_cmd.add_argument("--plan", type=Path, default=Path("work/shot-plan.json"))

    publish_cmd = cmd.add_parser("publish")
    publish_cmd.add_argument("--plan", type=Path, default=Path("work/shot-plan.json"))
    publish_cmd.add_argument("--candidate-dir", type=Path, default=Path("work/candidates"))
    publish_cmd.add_argument("--evaluation", type=Path)

    select_cmd = cmd.add_parser("select")
    select_cmd.add_argument("--story-id", required=True)
    select_cmd.add_argument("--sequence-id", required=True)
    select_cmd.add_argument("--shot-id", required=True)
    select_cmd.add_argument("--candidate", type=int, required=True)

    args = p.parse_args()
    try:
        {"prepare": prepare, "publish": publish, "select": select}[args.operation](args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        p.exit(1, f"Story operation failed: {e}\n")


if __name__ == "__main__":
    main()
