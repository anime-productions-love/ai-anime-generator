#!/usr/bin/env python3
"""Smooth existing Wan/SVD videos via RIFE CPU AI or FFmpeg optical flow (v2).

Independent post-production: does not rerun AI diffusion. Source can be a
successful run in the unified/legacy repository or a private-gallery URL.
Playback speed and output frame rate are deliberately separate controls.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from resolve_scene_input import download_video

CURRENT = "anime-productions-love/ai-anime-generator"
LEGACY = "anime-productions-love/ai-video-generator"
ARTIFACT = re.compile(r"(?:wan22-turbo|svd-xt|anime-movie)-v[0-9]+-([0-9]+)")
MAX_SOURCE_FRAMES = 600
MAX_OUTPUT_FRAMES = 2400


def run(argv, capture=False):
    print("+ " + " ".join(str(x) for x in argv), flush=True)
    proc = subprocess.run(
        [str(x) for x in argv], check=True,
        capture_output=capture, text=capture,
    )
    return proc.stdout if capture else None


def probe(path):
    data = json.loads(run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=index,codec_type,width,height,avg_frame_rate,nb_frames:"
        "format=duration", "-of", "json", path,
    ], capture=True))
    video = next((x for x in data.get("streams", [])
                  if x.get("codec_type") == "video"), None)
    if not video:
        raise ValueError(f"No readable video stream: {path}")
    fps = Fraction(video.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        raise ValueError("Source video reports no valid average frame rate")
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("Source video has invalid duration")
    return {
        "width": int(video["width"]), "height": int(video["height"]),
        "fps": float(fps), "duration": duration,
        "audio": any(x.get("codec_type") == "audio" for x in data["streams"]),
        "nb_frames": video.get("nb_frames"),
    }


def get_run_video(run_id, source_repo, directory):
    if not re.fullmatch(r"[0-9]{4,24}", run_id or ""):
        raise ValueError("Use a numeric GitHub Actions run ID")
    repositories = ([CURRENT, LEGACY] if source_repo == "auto"
                    else [CURRENT if source_repo == "unified" else LEGACY])
    for repo in repositories:
        try:
            result = json.loads(run([
                "gh", "api",
                f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
            ], capture=True))
        except subprocess.CalledProcessError:
            continue
        matches = [
            a for a in result.get("artifacts", [])
            if not a.get("expired", False)
            and ARTIFACT.fullmatch(a.get("name", ""))
            and ARTIFACT.fullmatch(a["name"]).group(1) == run_id
        ]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(
                f"Run {repo}/{run_id} has multiple matching video artifacts; "
                "select an unambiguous completed run."
            )
        unpack = directory / "download"
        unpack.mkdir()
        run([
            "gh", "run", "download", run_id,
            "--repo", repo, "--name", matches[0]["name"], "--dir", unpack,
        ])
        videos = list(unpack.rglob("video.mp4"))
        if len(videos) != 1:
            raise ValueError(f"Expected one video.mp4 in {repo}/{run_id}; found {len(videos)}")
        target = directory / "source.mp4"
        shutil.copyfile(videos[0], target)
        return target, {
            "kind": "github-run",
            "repository": repo, "run_id": run_id,
            "artifact_name": matches[0]["name"],
        }
    raise RuntimeError(
        f"Could not find an unexpired Wan/SVD/movie artifact for run {run_id}. "
        "If it expired, use a direct downloadable video URL instead."
    )


def source_video(args, workspace):
    if bool(args.source_run) == bool(args.video_url):
        raise ValueError(
            "Supply exactly one source: --source-run or --video-url"
        )
    if args.source_run:
        return get_run_video(args.source_run, args.source_repo, workspace)
    output = workspace / "source.mp4"
    download_video(args.video_url, output)
    return output, {"kind": "video-url", "url": args.video_url.split("?", 1)[0]}


def extract_frames(source, directory):
    directory.mkdir(parents=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", source, "-map", "0:v:0", "-an",
        "-vsync", "0", "-start_number", "0",
        directory / "%08d.png",
    ])
    frames = sorted(directory.glob("*.png"))
    if len(frames) < 2 or len(frames) > MAX_SOURCE_FRAMES:
        raise ValueError(
            f"RIFE input must have 2-{MAX_SOURCE_FRAMES} decoded frames; "
            f"this input contains {len(frames)}"
        )
    return frames


def render_rife(source, args, workspace, input_info):
    source_frames = extract_frames(source, workspace / "frames")
    target_count = round(
        (len(source_frames) / input_info["fps"]) *
        args.target_fps / args.speed
    )
    if target_count < 2:
        raise ValueError("Output would contain fewer than 2 frames")
    if target_count > MAX_OUTPUT_FRAMES:
        raise ValueError(
            f"Output would contain {target_count} frames; "
            f"maximum is {MAX_OUTPUT_FRAMES}. Increase playback speed or reduce FPS."
        )

    output = args.output_dir / "video.mp4"
    if target_count <= len(source_frames):
        print("Requested timing requires no intermediate AI frames; "
              "using FFmpeg retiming instead.", flush=True)
        render_motion(source, args, output, interpolate=False)
        return "retime-only"

    binary = os.environ.get("RIFE_BINARY", "")
    model = os.environ.get("RIFE_MODEL", "")
    if not binary or not Path(binary).is_file() or not model or not Path(model).is_dir():
        raise ValueError(
            "RIFE CPU release/model not installed. The GitHub workflow should "
            "install the pinned release for method=rife."
        )
    output_dir = workspace / "rife-output"
    output_dir.mkdir()
    run([
        binary, "-i", workspace / "frames", "-o", output_dir,
        "-n", str(target_count), "-m", model, "-g", "-1",
        "-j", "1:2:2", "-f", "%08d.png",
    ])
    frames = sorted(output_dir.glob("*.png"))
    if len(frames) != target_count:
        raise RuntimeError(
            f"RIFE produced {len(frames)} frames rather than {target_count}; "
            "refusing to silently alter playback timing."
        )
    # Normalize arbitrary output numbering for deterministic FFmpeg import.
    normalized = workspace / "render"
    normalized.mkdir()
    for index, frame in enumerate(frames):
        (normalized / f"{index:08d}.png").symlink_to(frame.resolve())

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", str(args.target_fps),
        "-start_number", "0", "-i", normalized / "%08d.png",
    ]
    if input_info["audio"]:
        command.extend(["-i", source])
    command.extend([
        "-map", "0:v:0", "-c:v", "libx264", "-preset", "medium",
        "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-r", str(args.target_fps),
    ])
    if input_info["audio"]:
        command.extend([
            "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
            "-af", f"atempo={args.speed}",
        ])
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", "-shortest", output])
    run(command)
    return "rife-ncnn-cpu"


def render_motion(source, args, output, interpolate=True):
    if interpolate:
        method = (
            f"setpts=(PTS-STARTPTS)/{args.speed},"
            f"minterpolate=fps={args.target_fps}:mi_mode=mci:"
            "mc_mode=aobmc:me_mode=bidir:vsbmc=1,format=yuv420p"
        )
    else:
        method = (
            f"setpts=(PTS-STARTPTS)/{args.speed},"
            f"fps={args.target_fps},format=yuv420p"
        )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", source, "-map", "0:v:0", "-vf", method,
        "-c:v", "libx264", "-preset", "medium",
        "-crf", str(args.crf), "-r", str(args.target_fps),
        "-pix_fmt", "yuv420p",
    ]
    if probe(source)["audio"]:
        cmd.extend([
            "-map", "0:a:0", "-c:a", "aac", "-b:a", "192k",
            "-af", f"atempo={args.speed}",
        ])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", output])
    run(cmd)


def final_frame(video, target):
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", video, "-map", "0:v:0", "-an",
        "-vsync", "0", "-f", "image2", "-update", "1", target,
    ])
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("Failed to extract enhanced video's final frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="")
    parser.add_argument("--video-url", default="")
    parser.add_argument("--source-repo", choices=("auto", "unified", "legacy"),
                        default="auto")
    parser.add_argument("--method", choices=("rife", "motion"), default="rife")
    parser.add_argument("--target-fps", type=int, default=24)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/enhanced"))
    args = parser.parse_args()
    if not 8 <= args.target_fps <= 60:
        parser.error("Target FPS must be 8–60")
    if args.speed not in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        parser.error("Playback speed must be 0.5, 0.75, 1, 1.25, 1.5, 2, 3, or 4")
    if not 15 <= args.crf <= 35:
        parser.error("CRF must be 15–35")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="anime_video_enhance_") as directory:
        workspace = Path(directory)
        source, origin = source_video(args, workspace)
        info = probe(source)
        if info["width"] > 2048 or info["height"] > 2048:
            raise ValueError("Input exceeds the supported maximum 2048×2048")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / "video.mp4"
        if args.method == "rife":
            actual_method = render_rife(source, args, workspace, info)
        else:
            render_motion(source, args, target)
            actual_method = "ffmpeg-motion-compensated"
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("Enhancement did not produce an MP4")
        result = probe(target)
        final_frame(target, args.output_dir / "last_frame.png")
        metadata = {
            "version": "v1", "source": origin,
            "requested_method": args.method,
            "actual_method": actual_method,
            "source_fps": round(info["fps"], 5),
            "output_fps": round(result["fps"], 5),
            "source_duration_seconds": round(info["duration"], 3),
            "output_duration_seconds": round(result["duration"], 3),
            "target_fps": args.target_fps, "playback_speed": args.speed,
            "expected_output_duration": round(info["duration"] / args.speed, 3),
            "width": result["width"], "height": result["height"],
            "audio_preserved": info["audio"],
            "elapsed_seconds": round(time.monotonic()-started, 1),
            "video_name": "video.mp4",
            "last_frame_name": "last_frame.png",
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Enhanced video saved: {target} | {info['fps']:.2f} FPS to "
            f"{result['fps']:.2f} FPS | {info['duration']:.2f}s to "
            f"{result['duration']:.2f}s | speed {args.speed:g}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
