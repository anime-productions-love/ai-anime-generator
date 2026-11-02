#!/usr/bin/env python3
"""Read image/video links from the specific private anime gallery via GitHub API (v1).

Only URLs referring to anime-productions-love/gallery can receive the gallery
fine-grained PAT. Other image/video hosts use their existing public download
path without any token. Supports /blob/<ref>/path, /raw/<ref>/path,
raw.githubusercontent.com/.../<ref>/path and GitHub Contents API links.
"""
import base64
import binascii
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

GALLERY = "anime-productions-love/gallery"
HOSTS = {"github.com", "www.github.com", "raw.githubusercontent.com", "api.github.com"}
MAX_REPO_FILE = 100 * 1024 * 1024


def parse_gallery_url(url):
    """Return (ref, repository file path) or None for non-gallery URL."""
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme != "https" or parsed.hostname not in HOSTS:
        return None
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Gallery URLs cannot include credentials or nonstandard ports.")
    parts = [urllib.parse.unquote(p) for p in parsed.path.strip("/").split("/")]
    host = parsed.hostname
    if host == "api.github.com":
        if (len(parts) < 6 or parts[:3] != ["repos", "anime-productions-love", "gallery"]
                or parts[3] != "contents"):
            return None
        repo_path = "/".join(parts[4:])
        params = urllib.parse.parse_qs(parsed.query)
        ref = params.get("ref", ["main"])[0]
    elif host == "raw.githubusercontent.com":
        if len(parts) < 4 or parts[:2] != ["anime-productions-love", "gallery"]:
            return None
        ref, repo_path = parts[2], "/".join(parts[3:])
    else:
        if len(parts) < 5 or parts[:2] != ["anime-productions-love", "gallery"]:
            return None
        if parts[2] not in ("blob", "raw"):
            return None
        ref, repo_path = parts[3], "/".join(parts[4:])
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", ref):
        raise ValueError("Gallery ref must be a single branch/tag or 40-character commit identifier.")
    segments = repo_path.split("/")
    if (not segments or any(x in ("", ".", "..") for x in segments)
            or any("\\" in x or "\x00" in x for x in segments)
            or len(repo_path) > 2048):
        raise ValueError("Gallery URL has an unsafe or empty repository file path.")
    return ref, repo_path


class RejectAuthenticatedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Authenticated gallery API redirect refused to protect the token.")


def fetch_gallery_bytes(url, max_bytes=MAX_REPO_FILE):
    """Return private-gallery file bytes, or None if the URL isn't the gallery.

    Never return HTML, Git LFS pointer text or an API JSON error as media.
    """
    source = parse_gallery_url(url)
    if source is None:
        return None
    token = (os.environ.get("GITHUB_FINE_GRAINED_TOKEN")
             or os.environ.get("GALLERY_REPO_TOKEN")
             or os.environ.get("GALLERY_TOKEN"))
    if not token:
        raise ValueError(
            "Private gallery URL requires a GitHub Actions secret "
            "GITHUB_FINE_GRAINED_TOKEN (or GALLERY_REPO_TOKEN). "
            "The fine-grained token must include anime-productions-love/gallery "
            "with Contents: read permission."
        )
    ref, repo_path = source
    if max_bytes < 1:
        raise ValueError("Invalid gallery file size limit")
    opener = urllib.request.build_opener(RejectAuthenticatedRedirects)
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AnimePrivateGalleryReader/1.0",
    }

    def api_json(path):
        target = "https://api.github.com/" + path
        request = urllib.request.Request(target, headers=headers)
        try:
            with opener.open(request, timeout=90) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > max(max_bytes * 2, MAX_REPO_FILE * 2):
                    raise ValueError("Gallery API response exceeds allowed size")
                payload = response.read(max(max_bytes * 2, MAX_REPO_FILE * 2) + 1)
            return json.loads(payload)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                raise ValueError(
                    f"Cannot access private gallery file (HTTP {error.code}). "
                    "Check the token's selected repository access, Contents:read, "
                    "file path and branch."
                ) from error
            raise ValueError(f"GitHub gallery API failed (HTTP {error.code}).") from error

    escaped_path = urllib.parse.quote(repo_path, safe="/")
    escaped_ref = urllib.parse.quote(ref, safe="")
    record = api_json(
        f"repos/{GALLERY}/contents/{escaped_path}?ref={escaped_ref}"
    )
    if not isinstance(record, dict) or record.get("type") != "file":
        raise ValueError("Gallery URL must identify a file, not a folder.")
    size = record.get("size", 0)
    if not isinstance(size, int) or size > max_bytes:
        raise ValueError(f"Gallery media file exceeds {max_bytes} bytes.")
    if record.get("encoding") == "base64" and record.get("content"):
        try:
            payload = base64.b64decode(record["content"], validate=False)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Invalid base64 gallery media content.") from error
    else:
        sha = record.get("sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Gallery file has no valid Git blob SHA.")
        blob = api_json(f"repos/{GALLERY}/git/blobs/{sha}")
        if blob.get("encoding") != "base64":
            raise ValueError("Git blob content is not base64.")
        payload = base64.b64decode(blob["content"], validate=False)
    if not payload or len(payload) > max_bytes:
        raise ValueError("Gallery file is empty or exceeds the download limit.")
    if (payload.startswith(b"version https://git-lfs.github.com/spec/v1")
            and b"\noid sha256:" in payload[:300]):
        raise ValueError("Gallery file is a Git LFS pointer, not downloadable media bytes.")
    print(f"Read authenticated gallery file ({len(payload)} bytes), ref={ref}, path={repo_path}", flush=True)
    return payload
