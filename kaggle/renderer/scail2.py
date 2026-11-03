#!/usr/bin/env python3
"""SCAIL-2 Q6_K Kaggle ComfyUI notebook driver (v2).

Loads an existing, pinned SCAIL-2 API-format ComfyUI graph. No custom diffusion,
motion mask or pose inference is implemented here. Large weights preferentially
come from read-only /kaggle/input datasets; transient downloads require real
free-space checks. Outputs remain unapproved until human visual review.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

EMBEDDED_JOB_JSON = None  # Filled by the GitHub submit bridge
OUT = Path("/kaggle/working/anime-render")
TMP = Path("/kaggle/temp/scail2")
COMFY = TMP / "ComfyUI"
SOURCE = TMP / "scail2-skill"
REPOS = {
    "ComfyUI": ("Comfy-Org/ComfyUI", "ab08f7b2caa6d4007ecde3762116c6193613f43b"),
    "ComfyUI-GGUF": ("city96/ComfyUI-GGUF", "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"),
    "ComfyUI-VideoHelperSuite": ("Kosinkadink/ComfyUI-VideoHelperSuite", "4d907bee61e92c2e65af3bd6383a4e4d356126d1"),
    "ComfyUI-KJNodes": ("kijai/ComfyUI-KJNodes", "ab72eda7ffad6280422d71ab7d54d3fb87c0ba18"),
    "ComfyUI-Easy-Use": ("yolain/ComfyUI-Easy-Use", "b0b47ca1372117fce575d9e744655344ebff6bef"),
    "ComfyUI-Custom-Scripts": ("pythongosssss/ComfyUI-Custom-Scripts", "609f3afaa74b2f88ef9ce8d939626065e3247469"),
    "scail2-skill": ("ruashots/scail2-skill", "91a79f518fc48ae4546fa815eacffa4bf194a3c6"),
}
WEIGHTS = [
    ("SCAIL-2-Q6_K.gguf", "unet", "realrebelai/SCAIL-2_GGUF",
     "SCAIL-2-Q6_K.gguf", 12_000_000_000),
    ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders",
     "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6_000_000_000),
    ("sam3.1_multiplex_fp16.safetensors", "checkpoints",
     "Comfy-Org/sam3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors", 1_400_000_000),
    ("clip_vision_h.safetensors", "clip_vision",
     "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/clip_vision/clip_vision_h.safetensors", 1_000_000_000),
    ("wan_2.1_vae.safetensors", "vae",
     "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/vae/wan_2.1_vae.safetensors", 180_000_000),
    ("Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors", "loras",
     "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
     "loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
     500_000_000),
]
def run(*command, cwd=None, timeout=900):
    print("RUN " + " ".join(map(str, command[:4])), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True, timeout=timeout)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def clone(name, parent):
    owner_repo, commit = REPOS[name]
    target = parent / name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    run("git", "init", "-q", str(target))
    run("git", "remote", "add", "origin", "https://github.com/" + owner_repo + ".git", cwd=target)
    run("git", "fetch", "--depth", "1", "origin", commit, cwd=target, timeout=400)
    run("git", "checkout", "-q", "FETCH_HEAD", cwd=target)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if actual != commit:
        raise RuntimeError("Upstream commit mismatch: " + name)
    return target

def env_python():
    if sys.version_info[:2] == (3, 11):
        return sys.executable
    # Kaggle may start with Python 3.13, outside SCAIL-2's supported 3.10-3.12.
    run(sys.executable, "-m", "pip", "install", "-q", "uv", timeout=300)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installed without a runnable CLI")
    TMP.mkdir(parents=True, exist_ok=True)
    run(uv, "python", "install", "3.11", timeout=400)
    venv = TMP / "py311"
    run(uv, "venv", "--python", "3.11", str(venv), timeout=300)
    py = str(venv / "bin/python")
    # Python 3.13 Kaggle base packages cannot be imported by Python 3.11.
    # Install a T4-compatible CUDA build in the isolated environment.
    run(uv, "pip", "install", "--python", py,
        "--index-url", "https://download.pytorch.org/whl/cu128",
        "torch==2.8.0", "torchvision==0.23.0", timeout=2400)
    run(uv, "pip", "install", "--python", py,
        "gdown>=5.2,<6", "huggingface_hub>=0.36,<1", "Pillow>=10,<13",
        timeout=700)
    return py

def get_inputs(job, root):
    import gdown
    from PIL import Image
    import re
    def fetch(key, filename):
        file_id = job[key]
        if not re.fullmatch(r"[A-Za-z0-9_-]{15,128}", file_id):
            raise ValueError("Invalid source Google Drive file ID")
        target = root / filename
        if not target.is_file():
            downloaded = gdown.download(id=file_id, output=str(target), quiet=True)
            if not downloaded or not target.is_file():
                raise RuntimeError("Google Drive source unavailable: " + key)
        return target
    inputs = COMFY / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    image_file = fetch("image_drive_file_id", "scail2-reference-source.bin")
    motion_file = fetch("motion_drive_file_id", "scail2-motion-source.bin")
    img = Image.open(image_file).convert("RGB")
    img.save(inputs / "scail2-reference.png")
    # Process the actual full-colour anime driving video. Never convert to edge maps.
    run("ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", motion_file,
        "-t", str(job.get("duration_seconds", 2)),
        "-vf", "fps=16", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        inputs / "scail2-motion.mp4", timeout=120)
    if (inputs / "scail2-motion.mp4").stat().st_size < 1024:
        raise RuntimeError("No decoded driving-video frames")
    shutil.copyfile(inputs / "scail2-reference.png", OUT / "source-image.png")
    return inputs / "scail2-reference.png", inputs / "scail2-motion.mp4", image_file, motion_file

def ensure_models():
    import huggingface_hub
    TMP.mkdir(parents=True, exist_ok=True)
    input_root = Path("/kaggle/input")
    available = {}
    # Ignore extra large datasets; find only the six explicitly required file names.
    mounted = {}
    if input_root.exists():
        for directory, _, files in os.walk(input_root):
            for name in files:
                if name in {w[0] for w in WEIGHTS} and name not in mounted:
                    mounted[name] = Path(directory) / name
    missing = []
    for name, directory, repo, hf_path, min_bytes in WEIGHTS:
        located = mounted.get(name)
        if located and located.stat().st_size >= min_bytes:
            available[name] = located
        else:
            missing.append((name, directory, repo, hf_path, min_bytes))
    free = shutil.disk_usage(TMP).free
    budget = sum(w[4] for w in missing) + 8_000_000_000
    write("scail-preflight-v1.json", {
        "model": "SCAIL-2 Q6_K", "mounted_weights": list(available),
        "missing_weights": [w[0] for w in missing], "free_scratch_bytes": free,
        "minimum_required_free_bytes": budget,
        "cuda_cards_expected": 2, "status": "STORAGE_CHECKED",
    })
    if free < budget:
        raise RuntimeError(
            "Kaggle scratch space insufficient for missing SCAIL-2 model files. "
            "Attach private Kaggle datasets containing: " + ", ".join(w[0] for w in missing)
            + f" (free {free:,} bytes; minimum {budget:,})")
    for name, directory, repo, hf_path, min_bytes in missing:
        path = Path(huggingface_hub.hf_hub_download(
            repo_id=repo, filename=hf_path, cache_dir=str(TMP / "hf-cache")))
        if path.stat().st_size < min_bytes:
            raise RuntimeError("Incomplete Hugging Face checkpoint: " + name)
        available[name] = path
    inventory = []
    for name, directory, repo, hf_path, min_bytes in WEIGHTS:
        target_dir = COMFY / "models" / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise RuntimeError("Refusing to overwrite existing model: " + str(target))
        target.symlink_to(available[name])
        inventory.append({
            "name": name, "source": f"https://huggingface.co/{repo}/resolve/main/{hf_path}",
            "bytes": available[name].stat().st_size,
            "sha256": sha(available[name]),
            "mounted_dataset": str(available[name]).startswith("/kaggle/input/"),
        })
    write("scail-model-inventory-v1.json", {"version": "v1", "models": inventory})
    return inventory

def install_components():
    root = clone("ComfyUI", TMP)
    for name in REPOS:
        if name in ("ComfyUI", "scail2-skill"):
            continue
        clone(name, root / "custom_nodes")
    clone("scail2-skill", TMP)
    py = sys.executable
    run(py, "-m", "pip", "install", "--disable-pip-version-check", "-q",
        "-r", root / "requirements.txt", timeout=1500)
    for name in ("ComfyUI-GGUF", "ComfyUI-VideoHelperSuite", "ComfyUI-KJNodes",
                 "ComfyUI-Easy-Use", "ComfyUI-Custom-Scripts"):
        req = root / "custom_nodes" / name / "requirements.txt"
        if req.is_file():
            run(py, "-m", "pip", "install", "--disable-pip-version-check", "-q",
                "-r", req, timeout=1200)
    run(py, "-m", "pip", "install", "-q", "gdown>=5.2,<6",
        "huggingface_hub>=0.36,<1", "Pillow>=10,<13", timeout=800)
    return root

def configure_api_workflow():
    # Use a published *API-format* SCAIL-2 graph and native color-mask nodes.
    # Only replace the model loader and filenames, not mask/pipeline implementation.
    source = SOURCE / "scripts" / "scail2_workflow.json"
    graph = json.loads(source.read_text(encoding="utf-8"))
    assert graph["74"]["class_type"] == "DiffusionModelLoaderKJ"
    graph["74"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "SCAIL-2-Q6_K.gguf"},
        "_meta": {"title": "SCAIL-2 GGUF Q6_K"}
    }
    graph["9"]["inputs"].update(
        clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", device="cpu")
    graph["11"]["inputs"].update(
        lora_name="Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")
    if graph["86"]["inputs"].get("ckpt_name") != "sam3.1_multiplex_fp16.safetensors":
        graph["86"]["inputs"]["ckpt_name"] = "sam3.1_multiplex_fp16.safetensors"
    # 3 subjects, same ordering in driving video and anime image.
    for ident in ("85", "91"):
        graph[ident]["inputs"]["max_objects"] = 3
    for node in graph.values():
        if node["class_type"] == "VHS_VideoCombine":
            node["inputs"]["save_output"] = True
    source.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    write("scail-graph-config-v1.json", {
        "version": "v1", "model": "SCAIL-2-Q6_K.gguf", "animation_mode": True,
        "characters": 3, "native_colored_masks": True,
        "graph_source": "ruashots/scail2-skill",
        "graph_source_commit": REPOS["scail2-skill"][1]
    })

def animate(job):
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Kaggle did not allocate the requested T4 x2 GPUs")
    cards = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not all("T4" in card for card in cards[:2]):
        raise RuntimeError("Expected T4 x2, allocated: " + repr(cards))
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    write("scail-hardware-v1.json", {
        "version": "v1", "gpus": cards, "cuda_available": torch.cuda.is_available(),
        "compute_capability": [torch.cuda.get_device_capability(i) for i in range(2)],
        "scratch_bytes_free": shutil.disk_usage(TMP).free,
    })
    # Ensure the ComfyUI runtime and model download libraries are available.
    install_components()
    inventory = ensure_models()
    configure_api_workflow()
    image, motion, raw_image, raw_video = get_inputs(job, TMP)
    # ComfyUI stays localhost-only. No notebook token or API secret is exposed.
    logfile = (OUT / "comfyui-v1.log").open("w", encoding="utf-8")
    server = subprocess.Popen([
        sys.executable, "-u", str(COMFY / "main.py"),
        "--listen", "127.0.0.1", "--port", "8188", "--lowvram", "--cpu-vae",
        "--disable-auto-launch"
    ], cwd=COMFY, stdout=logfile, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 240
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/object_info/WanSCAILToVideo", timeout=4) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            if server.poll() is not None:
                raise RuntimeError("ComfyUI exited before SCAIL-2 nodes initialized")
            if time.monotonic() > deadline:
                raise RuntimeError("ComfyUI SCAIL-2 node initialization timeout")
            time.sleep(4)
        # Run the unmodified published API client with the Q6_K graph above.
        command = [
            sys.executable, "-u", str(SOURCE / "scripts" / "scail2.py"),
            "animate", "--character", str(image), "--motion", str(motion),
            "--objects", "3", "--detect", "anime characters",
            "--describe", job["prompt"],
            "--steps", str(job.get("steps", 6)), "--fast",
            "--name", "scail2-q6k"
        ]
        environment = dict(os.environ)
        environment.update(COMFY_DIR=str(COMFY), COMFY_URL="http://127.0.0.1:8188")
        with (OUT / "scail-inference-v1.log").open("w", encoding="utf-8") as transcript:
            proc = subprocess.run(command, env=environment, stdout=transcript,
                                  stderr=subprocess.STDOUT, timeout=4800)
        if proc.returncode:
            raise RuntimeError("Published SCAIL-2 API client returned nonzero status")
        # Fail closed unless an actual rendered video exists.
        outputs = list((COMFY / "output").rglob("*scail2-q6k*.mp4"))
        outputs = [p for p in outputs if "-compare" not in p.name]
        if len(outputs) != 1:
            raise RuntimeError(f"Expected 1 generated SCAIL-2 MP4; found {len(outputs)}")
        shutil.copyfile(outputs[0], OUT / "video.mp4")
        if (OUT / "video.mp4").stat().st_size < 1000:
            raise RuntimeError("SCAIL-2 generated video is empty")
        # Preserve frames *before* optional validation.
        frame_folder = OUT / "generated-frames"
        frame_folder.mkdir(exist_ok=True)
        run("ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", OUT / "video.mp4", frame_folder / "frame-%04d.png", timeout=150)
        frames = sorted(frame_folder.glob("frame-*.png"))
        if not frames:
            raise RuntimeError("Video exists but no frames decoded")
        shutil.copyfile(frames[-1], OUT / "last_frame.png")
        write("raw-output-v1.json", {
            "version": "v1", "status": "GENERATED_VIDEO_SAVED_UNREVIEWED",
            "video_sha256": sha(OUT / "video.mp4"),
            "last_frame_sha256": sha(OUT / "last_frame.png"),
            "frames": len(frames), "approved_to_chain": False,
        })
        write("result-v1.json", {
            "version": "v1", "status": "RENDERED_UNREVIEWED",
            "model": "SCAIL-2-Q6_K.gguf", "renderer_version": "v1",
            "image_sha256": sha(raw_image), "reference_sha256": sha(raw_video),
            "video_sha256": sha(OUT / "video.mp4"),
            "last_frame_sha256": sha(OUT / "last_frame.png"),
            "frames": len(frames), "fps": 16,
            "characters_requested": 3, "model_files": inventory,
            "requires_human_approval_before_chaining": True
        })
        print("SCAIL2_Q6K_GENERATED_UNREVIEWED", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        logfile.close()

def main():
    job = json.loads(EMBEDDED_JOB_JSON) if EMBEDDED_JOB_JSON is not None else json.loads(
        Path(__file__).with_name("job-v1.json").read_text(encoding="utf-8"))
    if job.get("version") != "v1" or job.get("model") != "scail2-q6k":
        raise ValueError("Unsupported SCAIL-2 Kaggle job")
    if "--inner" not in sys.argv and sys.version_info[:2] != (3, 11):
        python311 = env_python()
        run(python311, __file__, "--inner", timeout=7200)
        return
    animate(job)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        files = [{
            "path": x.relative_to(OUT).as_posix(), "size_bytes": x.stat().st_size,
            "sha256": sha(x)
        } for x in sorted(OUT.rglob("*")) if x.is_file()]
        write("failure-v1.json", {
            "version": "v1", "status": "SCAIL2_FAILED_OUTPUTS_PRESERVED",
            "reason": str(error), "traceback": traceback.format_exc()[-10000:],
            "files": files, "approved_to_chain": False,
        })
        raise
