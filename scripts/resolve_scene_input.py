#!/usr/bin/env python3
"""Resolve a generation source from an image, previous scene, or video URL (v4).

Use the most recent *successful same-workflow* run with an unexpired artifact
for automatic continuation; never select the current run or another model.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from gallery_source import fetch_gallery_bytes
from prepare_input import detected_format, MAX_BYTES as MAX_IMAGE_BYTES

WORKFLOWS = {
    "generate-svd-xt.yml": "svd-xt",
    "generate-wan22-ti2v-turbo.yml": "wan22-turbo",
}
MAX_VIDEO_BYTES = 512 * 1024 * 1024


def command(argv, *, capture=False):
    result = subprocess.run(
        list(map(str, argv)), check=True, text=True,
        capture_output=capture,
    )
    return result.stdout if capture else None


def gh_json(api_path):
    output = command(["gh", "api", api_path], capture=True)
    return json.loads(output)


LEGACY_VIDEO_REPO = "anime-productions-love/ai-video-generator"


def latest_frame(repository, workflow_file, current_run, destination):
    """Prefer the unified repository, with a legacy fallback for old clips.

    Original ai-video-generator runs are retained for historical continuity,
    particularly Wan run 41448667591. No cross-repository write is attempted.
    """
    prefix = WORKFLOWS[workflow_file]
    repositories = [repository]
    if repository != LEGACY_VIDEO_REPO:
        repositories.append(LEGACY_VIDEO_REPO)
    for source_repo in repositories:
        for page in range(1, 6):
            query = (
                f"repos/{source_repo}/actions/workflows/{workflow_file}/runs"
                f"?status=success&per_page=100&page={page}"
            )
            try:
                runs = gh_json(query).get("workflow_runs", [])
            except subprocess.CalledProcessError:
                if source_repo == repository:
                    raise RuntimeError(
                        "Unable to list successful runs in the unified repository. "
                        "Check Actions:read permissions."
                    )
                print(
                    f"Legacy repo {source_repo} is not readable with the current "
                    "Actions token; legacy continuation unavailable.", flush=True,
                )
                break
            if not runs:
                break
            for item in runs:
                run_id = int(item["id"])
                if (source_repo == repository and run_id >= current_run) or item.get("conclusion") != "success":
                    continue
                try:
                    artifacts = gh_json(
                        f"repos/{source_repo}/actions/runs/{run_id}/artifacts?per_page=100"
                    ).get("artifacts", [])
                except subprocess.CalledProcessError:
                    continue
                matches = [
                    item for item in artifacts
                    if not item.get("expired", False)
                    and re.fullmatch(
                        rf"{re.escape(prefix)}-v[0-9]+-{run_id}",
                        item.get("name", ""),
                    )
                ]
                if len(matches) != 1:
                    continue
                with tempfile.TemporaryDirectory(prefix="previous_scene_") as directory:
                    command([
                        "gh", "run", "download", str(run_id),
                        "--repo", source_repo,
                        "--name", matches[0]["name"],
                        "--dir", directory,
                    ])
                    frames = list(Path(directory).rglob("last_frame.png"))
                    if len(frames) != 1 or frames[0].stat().st_size == 0:
                        raise RuntimeError(
                            f"Successful run {source_repo}/{run_id} has an artifact "
                            "without exactly one final frame."
                        )
                    shutil.copyfile(frames[0], destination)
                print(
                    f"Continuing from {source_repo} {workflow_file} "
                    f"run {run_id} ({matches[0]['name']}).", flush=True,
                )
                return
    raise RuntimeError(
        f"No successful {workflow_file} run with an unexpired last_frame.png "
        "is accessible in the unified or legacy repository. Use an image or "
        "video URL to begin; generation artifacts currently expire in 7 days."
    )


def download_video(url, destination):
    gallery_data = fetch_gallery_bytes(url, MAX_VIDEO_BYTES)
    if gallery_data is not None:
        destination.write_bytes(gallery_data)
        return
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("Video URL must be a direct, public HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise ValueError("Do not supply embedded URL credentials.")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 AnimeVideoGenerator/1.0",
            "Accept": "video/*,application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final = urllib.parse.urlsplit(response.geturl())
            if final.scheme not in ("https", "http"):
                raise ValueError("Video redirect must remain HTTP(S).")
            kind = response.headers.get("Content-Type", "").lower()
            if "text/html" in kind:
                raise ValueError(
                    "Video host returned an HTML page instead of a video. "
                    "Use a direct downloadable video file URL."
                )
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_VIDEO_BYTES:
                raise ValueError("Video is over the 512 MiB download limit.")
            amount = 0
            with destination.open("wb") as out:
                while True:
                    part = response.read(1024 * 1024)
                    if not part:
                        break
                    amount += len(part)
                    if amount > MAX_VIDEO_BYTES:
                        raise ValueError("Video is over the 512 MiB download limit.")
                    out.write(part)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Video host could not be accessed: {error}. "
            "Use a direct public video file URL, not a video player page."
        ) from error
    if destination.stat().st_size == 0:
        raise RuntimeError("Video host returned an empty file.")


def extract_last_video_frame(video_file, destination):
    try:
        # -update 1 overwrites the same image for every decoded video frame;
        # the file left at completion is the final decoded frame.
        command([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-nostdin", "-y", "-i", video_file, "-map", "0:v:0",
            "-an", "-vsync", "0", "-f", "image2", "-update", "1",
            destination,
        ])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Unable to decode the supplied video URL. "
            "Confirm the URL returns a supported video file, not HTML."
        ) from error
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError("The supplied video contained no decodable frames.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mode", choices=("auto", "image", "continue-last", "video-url"),
        required=True,
    )
    parser.add_argument(
        "--input", default="", help="Image URL/path, or direct video URL"
    )
    parser.add_argument("--workflow-file", choices=tuple(WORKFLOWS), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--current-run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("inputs/downloaded_image"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("Invalid owner/repository name")
    try:
        if args.source_mode == "image":
            if not args.input.strip():
                parser.error("Image mode requires a direct image URL or repo file path.")
            command([
                sys.executable, str(Path(__file__).with_name("prepare_input.py")),
                "--image", args.input.strip(), "--output", args.output,
            ])
        elif args.source_mode == "continue-last":
            latest_frame(
                args.repository, args.workflow_file,
                args.current_run_id, args.output,
            )
        else:
            # Accept either an image or a video in auto/video-url modes.
            # This also makes historical video-url submissions containing
            # gallery image links work rather than trying to decode JPG as MP4.
            if not args.input.strip():
                parser.error("Supply a gallery image/video URL or direct media file URL.")
            with tempfile.TemporaryDirectory(prefix="remote_media_") as directory:
                media_file = Path(directory) / "media_input"
                frame = Path(directory) / "last_frame.png"
                download_video(args.input.strip(), media_file)
                with media_file.open("rb") as media:
                    image_format = detected_format(media.read(16))
                if image_format:
                    if media_file.stat().st_size > MAX_IMAGE_BYTES:
                        raise ValueError("Gallery image exceeds the 24 MiB image limit.")
                    shutil.copyfile(media_file, args.output)
                    print(f"Resolved {image_format} image source (no video decode required).", flush=True)
                else:
                    extract_last_video_frame(media_file, frame)
                    shutil.copyfile(frame, args.output)
        if not args.output.is_file() or args.output.stat().st_size == 0:
            raise RuntimeError("No input image was resolved.")
        print(f"Prepared next scene's source image: {args.output}", flush=True)
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"Input resolution failed: {error}\n")


if __name__ == "__main__":
    main()
