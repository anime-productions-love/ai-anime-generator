#!/usr/bin/env python3
"""Download and validate a user-supplied image before expensive model setup (v2)."""
import argparse
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from gallery_source import fetch_gallery_bytes

MAX_BYTES = 24 * 1024 * 1024


def detected_format(payload):
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if payload.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "GIF"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "WEBP"
    if payload.startswith(b"BM"):
        return "BMP"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="inputs/downloaded_image")
    args = parser.parse_args()
    source = args.image.strip()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlsplit(source)

    if parsed.scheme:
        gallery_data = fetch_gallery_bytes(source, MAX_BYTES)
        if gallery_data is not None:
            payload = gallery_data
        else:
            if parsed.scheme not in ("https", "http") or not parsed.netloc:
                parser.error("Image must be a direct HTTP(S) image URL or a repository file path.")
            if parsed.username or parsed.password:
                parser.error("Do not put credentials in an image URL.")
            try:
                request = urllib.request.Request(
                    source,
                    headers={"User-Agent": "Mozilla/5.0 AnimeVideoGenerator/1.0",
                             "Accept": "image/webp,image/png,image/jpeg,image/*"},
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    final_url = urllib.parse.urlsplit(response.geturl())
                    if final_url.scheme not in ("http", "https"):
                        raise ValueError("Unexpected image redirect scheme")
                    payload = response.read(MAX_BYTES + 1)
            except (urllib.error.URLError, ValueError) as exc:
                parser.error(f"Image URL inaccessible: {exc}. Use a direct public image or a committed repository image.")
    else:
        path = Path(source)
        if not path.is_file():
            parser.error(f"Image file not found: {path}. Commit it first or supply a direct public image URL.")
        if path.stat().st_size > MAX_BYTES:
            parser.error("Image exceeds 24 MiB.")
        payload = path.read_bytes()

    if len(payload) > MAX_BYTES:
        parser.error("Image exceeds 24 MiB.")
    kind = detected_format(payload)
    if kind is None:
        parser.error(
            "Image URL did not return a recognizable PNG, JPEG, WebP, GIF, or BMP image. "
            "The host may serve an HTML anti-hotlink page or require a browser login. "
            "Use an image committed to this repository or a raw.githubusercontent.com URL."
        )
    destination.write_bytes(payload)
    print(f"Validated {kind} image ({len(payload)} bytes) -> {destination}", flush=True)


if __name__ == "__main__":
    main()
