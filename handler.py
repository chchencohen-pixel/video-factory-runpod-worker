"""Runpod Serverless handler for Wan 2.2 TI2V-5B 720p-class video (v2, persistent model).

The application dispatches one clip at a time. Source images and MP4 results
move through short-lived presigned URLs; this worker never holds application
S3 credentials or a Runpod API key.

v2 change: the model is loaded ONCE per worker process and kept on the GPU,
instead of spawning `generate.py` (which re-read ~34GB of weights and used CPU
offload on every job). Measured v1 cost was 332 s per 49-frame clip; the model
itself needs well under 100 s for that on this hardware.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import runpod

WAN_ROOT = Path(os.getenv("WAN_ROOT", "/opt/wan"))
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

MODEL_ID = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")
HF_CACHE_ROOT = Path(os.getenv("RUNPOD_HF_CACHE", "/runpod-volume/huggingface-cache/hub"))
TASK = "ti2v-5B"
# Supported output sizes (Wan 2.2 TI2V-5B official presets). The app sends width/height;
# anything else is rejected instead of silently producing a different resolution.
SIZE_KEYS = {(1280, 704): "1280*704", (704, 1280): "704*1280", (832, 480): "832*480", (480, 832): "480*832"}
DEFAULT_SIZE = (1280, 704)
# Knobs for smaller GPUs. Defaults target a 48GB card: everything on the GPU, no offload.
T5_CPU = os.getenv("WAN_T5_CPU", "false").lower() == "true"
OFFLOAD_MODEL = os.getenv("WAN_OFFLOAD_MODEL", "false").lower() == "true"
PRELOAD = os.getenv("WAN_PRELOAD", "true").lower() == "true"

_model = None
_model_lock = threading.Lock()
_model_error: str | None = None
_load_seconds: float | None = None


def fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def resolve_cached_model(model_id: str = MODEL_ID) -> Path:
    """Return the Model Caching snapshot, handling cache-folder case normalization."""
    if "/" not in model_id:
        raise RuntimeError("WAN_MODEL_ID must use org/model format")
    org, name = model_id.split("/", 1)
    expected_folder = f"models--{org}--{name}"
    model_roots = [HF_CACHE_ROOT / expected_folder]
    model_roots.extend(
        path
        for path in HF_CACHE_ROOT.glob("models--*")
        if path.name.casefold() == expected_folder.casefold() and path.name != expected_folder
    )
    for model_root in model_roots:
        refs_main = model_root / "refs" / "main"
        snapshots = model_root / "snapshots"
        if refs_main.is_file():
            candidate = snapshots / refs_main.read_text().strip()
            if candidate.is_dir():
                return candidate
        if snapshots.is_dir():
            candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if candidates:
                return candidates[-1]
    raise RuntimeError(f"Runpod Model Cache does not contain {model_id}; configure this exact Model value on the endpoint")


def get_model():
    """Load Wan TI2V-5B once per process and keep it resident."""
    global _model, _model_error, _load_seconds
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        started = time.time()
        try:
            import wan  # noqa: WPS433 (runtime import: only available inside the image)
            from wan.configs import WAN_CONFIGS

            model_root = resolve_cached_model()
            _model = wan.WanTI2V(
                config=WAN_CONFIGS[TASK],
                checkpoint_dir=str(model_root),
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=T5_CPU,
                convert_model_dtype=True,
            )
            _model_error = None
        except Exception as error:  # noqa: BLE001
            _model_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            _load_seconds = round(time.time() - started, 1)
        return _model


def get_image(url: str, directory: Path) -> Path:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "image/png").split(";", 1)[0]
    suffix = ".jpg" if content_type == "image/jpeg" else ".png"
    path = directory / f"source{suffix}"
    path.write_bytes(response.content)
    return path


def validate_frames(frames: int) -> None:
    if not 49 <= frames <= 121:
        raise ValueError("frames must be between 49 and 121")
    if (frames - 1) % 4 != 0:
        raise ValueError("frames must satisfy Wan's 4n+1 requirement")


def health_report() -> dict[str, Any]:
    try:
        model_root = resolve_cached_model()
        cache_ready = True
    except RuntimeError:
        model_root = None
        cache_ready = False
    return {
        "ok": True,
        "worker": "wan2.2-ti2v-5b-720p",
        "worker_version": 2,
        "model_id": MODEL_ID,
        "model_cache_ready": cache_ready,
        "model_root": str(model_root) if model_root else None,
        "model_loaded": _model is not None,
        "model_load_seconds": _load_seconds,
        "model_error": _model_error,
        "t5_cpu": T5_CPU,
        "offload_model": OFFLOAD_MODEL,
        "generation_not_run": True,
    }


def generate(job_input: dict[str, Any]) -> dict[str, Any]:
    if job_input.get("action") == "health":
        return health_report()

    mode = job_input.get("mode")
    if mode not in {"t2v", "i2v"}:
        return fail("mode must be t2v or i2v")
    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        return fail("prompt is required")
    output_upload_url = str(job_input.get("output_upload_url", "")).strip()
    if not output_upload_url:
        return fail("output_upload_url is required")
    output_key = str(job_input.get("output_key", "")).strip()
    if not output_key:
        return fail("output_key is required")

    seed = int(job_input.get("seed", 123))
    steps = int(job_input.get("steps", 20))
    frames = int(job_input.get("frames", 49))
    guide_scale = float(job_input.get("guidance_scale", 5.0))
    negative_prompt = str(job_input.get("negative_prompt", "") or "")
    try:
        validate_frames(frames)
    except ValueError as error:
        return fail(str(error))
    if not 3 <= steps <= 50:
        return fail("steps must be between 3 and 50")
    width = int(job_input.get("width", DEFAULT_SIZE[0]))
    height = int(job_input.get("height", DEFAULT_SIZE[1]))
    size_key = SIZE_KEYS.get((width, height))
    if not size_key:
        return fail(f"unsupported size {width}x{height}; use one of {sorted(f'{w}x{h}' for w, h in SIZE_KEYS)}")

    try:
        model = get_model()
    except Exception as error:  # noqa: BLE001
        return fail(f"model load failed: {error}")

    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS[TASK]
    with tempfile.TemporaryDirectory(prefix="video-factory-") as temp_dir:
        temp = Path(temp_dir)
        img = None
        if mode == "i2v":
            image_url = str(job_input.get("image_url", "")).strip()
            if not image_url:
                return fail("image_url is required for i2v")
            img = Image.open(get_image(image_url, temp)).convert("RGB")

        started = time.time()
        video = model.generate(
            prompt,
            img=img,
            size=SIZE_CONFIGS[size_key],
            max_area=MAX_AREA_CONFIGS[size_key],
            frame_num=frames,
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=OFFLOAD_MODEL,
        )
        generation_seconds = round(time.time() - started, 1)
        if video is None:
            return fail("Wan did not produce a video tensor")

        output_path = temp / f"{uuid.uuid4().hex}.mp4"
        save_video(tensor=video[None], save_file=str(output_path), fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
        del video
        if not output_path.exists() or output_path.stat().st_size == 0:
            return fail("Wan did not produce an MP4 output")

        with output_path.open("rb") as stream:
            upload = requests.put(
                output_upload_url,
                data=stream,
                headers={"content-type": "video/mp4"},
                timeout=300,
            )
        upload.raise_for_status()
        return {
            "ok": True,
            "mode": mode,
            "resolution": f"{width}x{height}",
            "frames": frames,
            "steps": steps,
            "seed": seed,
            "generation_seconds": generation_seconds,
            "model_load_seconds": _load_seconds,
            "output_key": output_key,
        }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return generate(job.get("input", {}))
    except requests.RequestException as error:
        return fail(f"asset transfer failed: {error}")
    except Exception as error:  # noqa: BLE001
        return fail(f"unexpected worker error: {error}")


if PRELOAD:
    # Warm the model while the worker boots so the first job pays no load time.
    try:
        get_model()
    except Exception as error:  # noqa: BLE001
        print(f"[worker] model preload failed: {error}", flush=True)

runpod.serverless.start({"handler": handler})
