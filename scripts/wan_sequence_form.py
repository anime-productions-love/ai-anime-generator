#!/usr/bin/env python3
"""Turn the 25-field GitHub form into one immutable Wan sequence plan (v2).

This script accepts ONLY the starting image, ONE shared configuration, and
the first N filled motion-prompt fields. The workflow commits the plan before
dispatching the first generator job. No per-shot source or settings exist.
"""
import argparse
import json
import os
from pathlib import Path
import re
from wan_chain import ROOT, parse_config

MAX_PROMPTS = 15


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing form field {name}")
    return value


def integer(name, minimum, maximum):
    value = required(name)
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{name} must be a whole number")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in range {minimum}–{maximum}")
    return result


def build(run_id):
    if not re.fullmatch(r"[0-9]{4,24}", str(run_id)):
        raise ValueError("Invalid GitHub run ID")
    if os.environ.get("GITHUB_REPOSITORY") != "anime-productions-love/ai-anime-generator":
        raise ValueError("Continuous Wan sequence must start in the unified anime repo")
    number = integer("VIDEO_COUNT", 2, MAX_PROMPTS)
    image = required("START_IMAGE")
    if not image.startswith("https://"):
        raise ValueError("START_IMAGE must be an HTTPS URL to a gallery image or video")
    frame_size = required("FRAME_SIZE")
    matched = re.fullmatch(r"([0-9]{3,4})x([0-9]{3,4})", frame_size)
    if not matched:
        raise ValueError("FRAME_SIZE must be WIDTHxHEIGHT, e.g. 480x832")
    width, height = map(int, matched.groups())
    prompts = []
    for index in range(1, MAX_PROMPTS + 1):
        prompt = os.environ.get(f"PROMPT_{index:02d}", "").strip()
        if index <= number:
            if not prompt:
                raise ValueError(
                    f"Video {index} of {number} needs its own filled prompt box"
                )
            prompts.append(prompt)
        elif prompt:
            raise ValueError(
                f"Video {index} contains text but video_count is only {number}. "
                "Either clear that field or raise the video count."
            )
    settings = {
        "negative_prompt": os.environ.get("NEGATIVE_PROMPT", ""),
        "fit_mode": required("FIT_MODE"),
        "frames": integer("FRAMES", 5, 121),
        "fps": integer("FPS", 1, 60),
        "width": width,
        "height": height,
        "seed": integer("SEED", 0, 2147483647),
    }
    finish = required("FINISH_PRESET")
    found = re.fullmatch(r"(motion|rife)-([0-9]{2})fps-([0-9.]+)x", finish)
    if not found:
        raise ValueError("FINISH_PRESET must be method-FPS-speed, e.g. motion-24fps-1.5x")
    method, fps, speed = found.groups()
    seam = required("SEAM_MODE")
    if seam not in ("cut", "motion-seam"):
        raise ValueError("SEAM_MODE must be cut or motion-seam")
    plan = {
        "version": "v2",
        "title": f"Continuous Wan sequence {run_id}",
        "initial_image": image,
        "settings": settings,
        "prompts": prompts,
        "assembly": {
            "enabled": True, "width": width,
            "height": height, "fps": settings["fps"],
            "transition": seam, "fade_seconds": "0", "quality_crf": 20,
        },
        "postprocess": {
            "method": method, "target_fps": int(fps), "speed": float(speed),
        },
    }
    destination = ROOT / "chains" / f"chain-{run_id}.json"
    if destination.exists():
        raise ValueError(f"Chain plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Persist only user-entered, nonsecret media URLs and generation settings.
    # The gallery PAT stays in GitHub Actions secrets, never this public plan.
    destination.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    try:
        _, resolved, _ = parse_config(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    print(f"Prepared {destination.relative_to(ROOT)}: {len(resolved)} clips, "
          f"{settings['frames']} frames at {settings['fps']} FPS, "
          f"{width}x{height}, seed {settings['seed']}. "
          "Only clip 1 uses the starting image.", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        build(args.run_id)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot start continuous Wan sequence: {exc}\n")


if __name__ == "__main__":
    main()
