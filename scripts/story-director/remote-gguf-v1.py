#!/usr/bin/env python3
"""Read-only HTTPS Range filesystem for pinned GGUFs too large for runner SSD (v1).

Mounts exactly one verified Hugging Face LFS/Xet object, never arbitrary URLs.
Model weight bytes are delivered on demand; only ~128 MiB resides in an LRU
cache. The GitHub runner must expose /dev/fuse and the runtime needs mfusepy.
This is an unverified deployment fallback, not guaranteed native CI support.
"""
import argparse
from collections import OrderedDict
import errno
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import urllib.error
import urllib.request

CHUNK = 8 * 1024 * 1024
CACHE_CHUNKS = 16


class NoCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, url):
        return None


def head_identity(url, expected_sha, expected_commit):
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "anime-story-director-v1"}
    )
    opener = urllib.request.build_opener(NoCrossHostRedirect())
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        response = exc
    etag = (response.headers.get("X-Linked-Etag") or "").strip('"')
    sha = response.headers.get("X-Repo-Commit", "")
    size = int(response.headers.get("X-Linked-Size", "0"))
    if not re.fullmatch(r"[0-9a-f]{64}", etag) or etag != expected_sha:
        raise RuntimeError("Remote LFS/Xet object SHA-256 identity mismatch")
    if sha != expected_commit:
        raise RuntimeError("Remote GGUF immutable revision mismatch")
    if size < 10_000_000_000 or size > 35_000_000_000:
        raise RuntimeError("Remote GGUF size is not in supported range")
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--mountpoint", required=True, type=Path)
    p.add_argument("--provenance", required=True, type=Path)
    args = p.parse_args()
    data = json.loads(args.manifest.read_text())
    if data.get("version") != "v1" or not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", "")):
        p.error("Unknown unpinned GGUF remote manifest")
    url, name, sha = data["remote_url"], data["filename"], data["expected_sha256"]
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+\.gguf", name)
            or not url.startswith("https://huggingface.co/" + data["repository"] + "/resolve/" + data["revision"] + "/")
            or not url.endswith("/" + name)):
        p.error("GGUF manifest URL is not identical to the authorized model source")
    if not Path("/dev/fuse").exists():
        p.error("This ephemeral GitHub runner has no /dev/fuse; remote GGUF mounting unavailable")
    size = head_identity(url, sha, data["revision"])
    args.mountpoint.mkdir(parents=True, exist_ok=True)
    from fuse import FUSE, Operations, FuseOSError

    class RemoteGGUF(Operations):
        def __init__(self):
            self.cache = OrderedDict()
            self.lock = threading.Lock()
            self.filename = "/" + name

        def getattr(self, path, fh=None):
            now = time.time()
            if path == "/":
                return {"st_mode": stat.S_IFDIR | 0o555, "st_nlink": 2,
                        "st_ctime": now, "st_mtime": now, "st_atime": now}
            if path != self.filename:
                raise FuseOSError(errno.ENOENT)
            return {"st_mode": stat.S_IFREG | 0o444, "st_nlink": 1,
                    "st_size": size, "st_ctime": now, "st_mtime": now,
                    "st_atime": now}

        def readdir(self, path, fh):
            return [".", "..", name]

        def open(self, path, flags):
            if path != self.filename:
                raise FuseOSError(errno.ENOENT)
            if flags & os.O_ACCMODE != os.O_RDONLY:
                raise FuseOSError(errno.EROFS)
            return 0

        def _get_chunk(self, chunk_id):
            with self.lock:
                if chunk_id in self.cache:
                    self.cache.move_to_end(chunk_id)
                    return self.cache[chunk_id]
            start = chunk_id * CHUNK
            end = min(size, start + CHUNK) - 1
            req = urllib.request.Request(
                url, headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "anime-story-director-v1",
                }
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                if response.status != 206:
                    raise OSError("Source did not honor HTTP Range requests")
                header = response.headers.get("Content-Range", "")
                if not header.startswith(f"bytes {start}-{end}/"):
                    raise OSError("Wrong response range from model CDN")
                blob = response.read(CHUNK + 1)
            if len(blob) != end - start + 1:
                raise OSError("Incomplete model chunk")
            with self.lock:
                self.cache[chunk_id] = blob
                self.cache.move_to_end(chunk_id)
                while len(self.cache) > CACHE_CHUNKS:
                    self.cache.popitem(last=False)
            return blob

        def read(self, path, length, offset, fh):
            if path != self.filename:
                raise FuseOSError(errno.ENOENT)
            if offset >= size:
                return b""
            end = min(size, offset + length)
            chunks = []
            while offset < end:
                index = offset // CHUNK
                block = self._get_chunk(index)
                inner = offset % CHUNK
                piece = min(end - offset, CHUNK - inner)
                chunks.append(block[inner:inner + piece])
                offset += piece
            return b"".join(chunks)

    receipt = {
        "version": "v1",
        "model": data["model"], "repository": data["repository"],
        "revision": data["revision"], "filename": name,
        "file_sha256": sha, "file_bytes": size,
        "storage_mode": "read_only_https_range_fuse",
        "source_url": data["source_url"],
        "integrity": "pinned Hugging Face LFS/Xet HEAD SHA256 identity",
        "full_byte_stream_sha256_verified": False,
        "vision_projector_included": False,
        "status": "PINNED_REMOTE_MODEL_MOUNT_PENDING",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Mounting pinned remote GGUF with bounded 128 MiB chunk cache", flush=True)
    FUSE(RemoteGGUF(), str(args.mountpoint), foreground=True,
         nothreads=True, ro=True, allow_other=False)


if __name__ == "__main__":
    main()
