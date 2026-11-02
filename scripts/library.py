#!/usr/bin/env python3
"""Git-backed character and scene library manager (v3).

Python standard library only. Phase 1 stores durable reference images and
reusable manifests; it does not train LoRAs or invoke an image generator.
"""
import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIND_DIR = {"character": "characters", "scene": "scenes"}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MAX_IMAGE_SIZE = 15 * 1024 * 1024
IMAGE_TYPES = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)
DEFAULT_STYLE = "anime-cel-fantasy"


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_id(value, label="ID"):
    if not ID_PATTERN.fullmatch(value or ""):
        raise ValueError(f"{label} must be 2-64 lowercase letters, digits, '_' or '-', beginning with a letter/digit")
    return value


def validate_role(value):
    if not ROLE_PATTERN.fullmatch(value or ""):
        raise ValueError("reference role must be a lowercase identifier of up to 32 characters")
    return value


def pack_path(kind, identifier):
    if kind not in KIND_DIR:
        raise ValueError("kind must be 'character' or 'scene'")
    validate_id(identifier)
    return ROOT / "assets" / KIND_DIR[kind] / identifier


def parse_tags(raw):
    tags = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if len(tags) > 32 or any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", x) for x in tags):
        raise ValueError("tags: max 32 entries, each lowercase alphanumeric/underscore/hyphen")
    return list(dict.fromkeys(tags))


def validate_style(style_id):
    validate_id(style_id, "style ID")
    target = ROOT / "assets" / "styles" / style_id / "manifest.json"
    if not target.is_file():
        raise ValueError(f"style {style_id!r} not found at {target.relative_to(ROOT)}")
    return style_id


def parse_attributes(value):
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--attributes-json must be valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("--attributes-json must contain a JSON object")
    if len(json.dumps(obj, ensure_ascii=False)) > 16_000:
        raise ValueError("attributes JSON exceeds 16KB")
    return obj


def read_pack(kind, identifier):
    path = pack_path(kind, identifier) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{kind} {identifier!r} not found; create it first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "v1" or data.get("kind") != kind or data.get("id") != identifier:
        raise ValueError(f"unsupported or mismatched manifest: {path}")
    return data


def save_manifest(kind, identifier, manifest):
    package = pack_path(kind, identifier)
    package.mkdir(parents=True, exist_ok=True)
    path = package / "manifest.json"
    output = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    fd, temp = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=package)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(output)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def image_type(header):
    for magic, ext in IMAGE_TYPES:
        if header.startswith(magic):
            return ext
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("reference is not a recognized PNG, JPEG, GIF or WebP image")


def validate_public_https_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("image URL must be public HTTPS without embedded credentials")
    if parsed.port not in (None, 443):
        raise ValueError("reference URLs must use standard HTTPS port 443")
    hostname = parsed.hostname
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("local hostnames are not allowed")
    addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("image URL must resolve only to public addresses")
    return parsed


class SafeRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        validate_public_https_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def get_reference_bytes(args):
    if bool(args.image_url) == bool(args.file):
        raise ValueError("attach exactly one reference source: --image-url or --file")
    if args.image_url:
        validate_public_https_url(args.image_url)
        opener = urllib.request.build_opener(SafeRedirects)
        request = urllib.request.Request(
            args.image_url,
            headers={"User-Agent": "AnimeStoryLibrary/1.0", "Accept": "image/*"},
        )
        with opener.open(request, timeout=40) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("image URL returned HTML, not a raw image")
            data = response.read(MAX_IMAGE_SIZE + 1)
        source = urllib.parse.urlsplit(args.image_url)
        # Do not persist signed URL query tokens in a public Git repository.
        source_name = urllib.parse.urlunsplit(
            (source.scheme, source.netloc, source.path, "", "")
        )
    else:
        reference = Path(args.file).expanduser().resolve(strict=True)
        if not reference.is_file() or reference.stat().st_size > MAX_IMAGE_SIZE:
            raise ValueError("local reference must be a file no larger than 15 MiB")
        data = reference.read_bytes()
        source_name = "local:" + reference.name
    if not data or len(data) > MAX_IMAGE_SIZE:
        raise ValueError("reference must be nonempty and no larger than 15 MiB")
    return data, source_name, image_type(data[:16])


def attach_reference(kind, identifier, manifest, args):
    role = validate_role(args.role)
    data, source_name, extension = get_reference_bytes(args)
    sha = hashlib.sha256(data).hexdigest()
    rel_path = f"refs/{role}/{sha[:20]}{extension}"
    existing = manifest.setdefault("references", [])
    if any(r.get("sha256") == sha and r.get("role") == role for r in existing):
        print(f"Already registered: {kind}/{identifier} {role} {sha[:12]}")
        return False
    destination = pack_path(kind, identifier) / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
            raise ValueError(f"reference collision: {destination}")
    else:
        destination.write_bytes(data)
    existing.append({
        "role": role,
        "path": rel_path,
        "sha256": sha,
        "source": source_name,
        "added_at": stamp(),
    })
    return True


def update_fields(manifest, args):
    changed = False
    for field in ("name", "description", "prompt", "negative_prompt"):
        value = getattr(args, field, None)
        if value is not None and manifest.get(field) != value:
            if field == "name" and not value.strip():
                raise ValueError("name cannot be empty")
            manifest[field] = value
            changed = True
    if args.tags is not None:
        tags = parse_tags(args.tags)
        if manifest.get("tags") != tags:
            manifest["tags"] = tags
            changed = True
    if args.style_id is not None:
        style_id = validate_style(args.style_id)
        if manifest.get("style_id") != style_id:
            manifest["style_id"] = style_id
            changed = True
    if args.attributes_json is not None:
        attrs = parse_attributes(args.attributes_json)
        if manifest.get("attributes") != attrs:
            manifest["attributes"] = attrs
            changed = True
    return changed


def create(args):
    validate_id(args.id)
    if not args.name or not args.name.strip():
        raise ValueError("creating a pack requires --name")
    package = pack_path(args.kind, args.id)
    if package.exists():
        raise ValueError(f"{args.kind} {args.id!r} already exists; choose update")
    validate_style(args.style_id or DEFAULT_STYLE)
    now = stamp()
    data = {
        "schema_version": "v1", "revision": 1, "kind": args.kind,
        "id": args.id, "name": args.name, "description": args.description or "",
        "style_id": args.style_id or DEFAULT_STYLE,
        "prompt": args.prompt or "", "negative_prompt": args.negative_prompt or "",
        "tags": parse_tags(args.tags or ""),
        "attributes": parse_attributes(args.attributes_json) if args.attributes_json else {},
        "references": [],
        "adapters": {"character_lora": None, "ip_adapter_embeddings": None},
        "created_at": now, "updated_at": now,
    }
    if args.image_url or args.file:
        attach_reference(args.kind, args.id, data, args)
    save_manifest(args.kind, args.id, data)
    print(f"Created {args.kind} {args.id} (revision 1)")


def update(args):
    data = read_pack(args.kind, args.id)
    changed = update_fields(data, args)
    if args.image_url or args.file:
        changed = attach_reference(args.kind, args.id, data, args) or changed
    if not changed:
        print(f"No changes to {args.kind} {args.id}")
        return
    data["revision"] += 1
    data["updated_at"] = stamp()
    save_manifest(args.kind, args.id, data)
    print(f"Updated {args.kind} {args.id} (revision {data['revision']})")


def add_reference(args):
    if not (args.image_url or args.file):
        raise ValueError("add-reference needs --image-url or --file")
    data = read_pack(args.kind, args.id)
    if not attach_reference(args.kind, args.id, data, args):
        return
    data["revision"] += 1
    data["updated_at"] = stamp()
    save_manifest(args.kind, args.id, data)
    print(f"Attached {args.role} reference to {args.kind} {args.id}")


def register_lora(args):
    """Register an uploaded character LoRA model as a stable HF Hub reference."""
    if args.kind != "character":
        raise ValueError("LoRA registration is supported for characters only")
    repo_id = args.hub_repo
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id or ""):
        raise ValueError("--hub-repo must be namespace/repository on Hugging Face")
    data = read_pack("character", args.id)
    revision = args.revision
    if revision and not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character Hugging Face model commit")
    new_ref = "hf://" + repo_id + ("@" + revision if revision else "")
    if data["adapters"]["character_lora"] == new_ref:
        print("Character LoRA already registered; no change")
        return
    data["adapters"]["character_lora"] = new_ref
    data["revision"] += 1
    data["updated_at"] = stamp()
    save_manifest("character", args.id, data)
    print(f"Registered character LoRA {new_ref} for {args.id}")


def list_packs(args):
    directory = ROOT / "assets" / KIND_DIR[args.kind]
    packs = []
    for file in sorted(directory.glob("*/manifest.json")):
        obj = json.loads(file.read_text(encoding="utf-8"))
        packs.append({
            "id": obj["id"], "name": obj["name"], "kind": obj["kind"],
            "revision": obj["revision"], "reference_count": len(obj["references"]),
            "style_id": obj["style_id"],
        })
    print(json.dumps(packs, ensure_ascii=False, indent=2))


def show(args):
    data = read_pack(args.kind, args.id)
    if args.operation == "resolve":
        for ref in data["references"]:
            file = pack_path(args.kind, args.id) / ref["path"]
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError(f"missing/changed reference image: {file}")
            ref["repo_path"] = str(file.relative_to(ROOT))
        style_file = ROOT / "assets" / "styles" / data["style_id"] / "manifest.json"
        if not style_file.is_file():
            raise ValueError(f"missing style: {data['style_id']}")
        data["style"] = json.loads(style_file.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for action in ("create", "update", "add-reference", "register-lora", "list", "show", "resolve"):
        p = commands.add_parser(action)
        p.add_argument("--kind", choices=tuple(KIND_DIR), required=True)
        if action != "list":
            p.add_argument("--id", required=True)
        if action == "register-lora":
            p.add_argument("--hub-repo", required=True)
            p.add_argument("--revision")
        if action in ("create", "update", "add-reference"):
            p.add_argument("--image-url")
            p.add_argument("--file")
            p.add_argument("--role", default="main")
        if action in ("create", "update"):
            p.add_argument("--name")
            p.add_argument("--description")
            p.add_argument("--prompt")
            p.add_argument("--negative-prompt")
            p.add_argument("--tags")
            p.add_argument("--style-id")
            p.add_argument("--attributes-json")
    args = parser.parse_args()
    try:
        {
            "create": create, "update": update,
            "add-reference": add_reference, "register-lora": register_lora,
            "list": list_packs,
            "show": show, "resolve": show,
        }[args.operation](args)
    except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        parser.exit(1, f"Library operation failed: {error}\n")


if __name__ == "__main__":
    main()
