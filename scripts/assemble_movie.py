#!/usr/bin/env python3
"""Assemble ordered Wan/SVD GitHub Action scene clips into one film (v2)."""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def run(command):
    print("+ " + shlex.join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True)


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"No readable video stream: {path}")
    stream = info["streams"][0]
    duration = float(info.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError(f"Video has no positive duration: {path}")
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "duration": duration, "frame_rate": stream.get("avg_frame_rate"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Combine generated scene artifacts in order")
    parser.add_argument("--scene-dir", type=Path, default=Path("scene-inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/movie"))
    parser.add_argument("--run-ids", required=True, help="Ordered comma/space/newline-separated GitHub run IDs")
    parser.add_argument("--width", type=int, default=480, help="Final canvas width (default portrait)")
    parser.add_argument("--height", type=int, default=832, help="Final canvas height (default portrait)")
    parser.add_argument("--fps", type=int, default=8, help="Movie playback rate")
    parser.add_argument("--transition", choices=("cut", "fade-black", "motion-seam"), default="cut")
    parser.add_argument("--fade-seconds", type=float, default=0.3)
    parser.add_argument("--crf", type=int, default=20, help="H.264 quality: lower is better")
    args = parser.parse_args()
    if args.width < 64 or args.height < 64 or args.width % 2 or args.height % 2:
        parser.error("Width and height must be even positive dimensions of at least 64")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be from 1 through 60")
    if not 0.0 <= args.fade_seconds <= 5.0:
        parser.error("--fade-seconds must be from 0 through 5")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be from 0 through 51")
    args.run_ids = [x for x in re.split(r"[\s,;]+", args.run_ids.strip()) if x]
    if not args.run_ids or any(not re.fullmatch(r"[0-9]+", x) for x in args.run_ids):
        parser.error("--run-ids must contain numeric GitHub Actions run IDs")
    return args


def main():
    args = parse_args()
    numbered_dirs = sorted(
        p for p in args.scene_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"[0-9]{4,}", p.name)
    )
    if len(numbered_dirs) != len(args.run_ids):
        raise ValueError(
            f"Expected {len(args.run_ids)} downloaded scenes, found {len(numbered_dirs)} "
            f"under {args.scene_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / "_intermediate"
    scratch.mkdir(exist_ok=True)
    normalized = []
    timeline = []
    elapsed = 0.0

    for index, (run_id, directory) in enumerate(zip(args.run_ids, numbered_dirs)):
        candidates = sorted(directory.rglob("video.mp4"))
        if len(candidates) != 1:
            raise ValueError(
                f"Run {run_id}: expected one video.mp4 in downloaded artifact; "
                f"found {len(candidates)}"
            )
        source = candidates[0]
        info = probe(source)
        output = scratch / f"{index + 1:04d}.mp4"

        # Preserve full original composition, even when source and movie
        # use different orientations. Black letterboxing is intentional.
        filter_chain = []
        if args.transition == "motion-seam" and index > 0:
            # The first frame of every successor was derived from the exact
            # prior last frame. Remove its redundant boundary frame to avoid
            # a perceptible freeze before optical-flow interpolation.
            filter_chain.extend(["select=not(eq(n\\,0))",
                                 f"setpts=N/({args.fps}*TB)"])
        filter_chain.extend([
            f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease",
            f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:black",
            f"fps={args.fps}",
            "setsar=1", "format=yuv420p",
        ])
        fade = min(args.fade_seconds, info["duration"] / 3)
        if args.transition == "fade-black" and fade > 0:
            if index > 0:
                filter_chain.append(f"fade=t=in:st=0:d={fade:.6f}")
            if index < len(numbered_dirs) - 1:
                filter_chain.append(
                    f"fade=t=out:st={max(0.0, info['duration'] - fade):.6f}:d={fade:.6f}"
                )

        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", source, "-map", "0:v:0", "-an",
            "-vf", ",".join(filter_chain),
            "-c:v", "libx264", "-preset", "medium", "-crf", args.crf,
            "-r", args.fps, "-pix_fmt", "yuv420p", output,
        ])
        standardized = probe(output)
        normalized.append(output)
        timeline.append({
            "order": index + 1,
            "github_run_id": run_id,
            "source_artifact_video": str(source),
            "source_width": info["width"],
            "source_height": info["height"],
            "source_duration_seconds": round(info["duration"], 3),
            "movie_start_seconds": round(elapsed, 3),
            "movie_duration_seconds": round(standardized["duration"], 3),
        })
        elapsed += standardized["duration"]

    # All segments now share exact codec, pixel format, resolution and FPS.
    # The concat demuxer avoids needlessly re-encoding the completed movie.
    manifest = scratch / "concat.txt"
    manifest.write_text(
        "".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in normalized),
        encoding="utf-8",
    )
    final = args.output_dir / "movie.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", manifest,
        "-c:v", "copy", "-an", "-movflags", "+faststart", final,
    ])
    final_info = probe(final)
    report = {
        "version": "v1",
        "scene_count": len(timeline),
        "ordered_run_ids": args.run_ids,
        "width": args.width, "height": args.height,
        "fps": args.fps, "transition": args.transition,
        "seam_handling": ("duplicate boundary frames removed; optical-flow "
                          "smoothing runs as separate postprocess"
                          if args.transition == "motion-seam" else
                          "original frame joins"),
        "fade_seconds": args.fade_seconds if args.transition == "fade-black" else 0,
        "audio": "No audio. Scene audio is not included in this v1 combiner.",
        "duration_seconds": round(final_info["duration"], 3),
        "timeline": timeline,
    }
    (args.output_dir / "timeline.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    # Avoid uploading duplicate normalized clips and the concat manifest.
    for path in normalized:
        path.unlink()
    manifest.unlink()
    scratch.rmdir()
    print(f"Saved {final} ({final_info['duration']:.2f} seconds)", flush=True)


if __name__ == "__main__":
    main()
