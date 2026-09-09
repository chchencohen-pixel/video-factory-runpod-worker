"""Static checks; these tests never download Wan weights or invoke a GPU."""

from pathlib import Path


ROOT = Path(__file__).parent


def test_worker_uses_presigned_urls_and_no_application_secret() -> None:
    source = (ROOT / "handler.py").read_text()
    assert "output_upload_url" in source
    assert "image_url" in source
    assert "RUNPOD_API_KEY" not in source
    assert "AWS_SECRET_ACCESS_KEY" not in source


def test_worker_declares_wan_720p_contract() -> None:
    source = (ROOT / "handler.py").read_text()
    assert '"Wan-AI/Wan2.2-TI2V-5B"' in source
    assert 'TASK = "ti2v-5B"' in source
    assert 'SIZE_KEY = "1280*704"' in source
    assert "huggingface-cache" in source
    assert "resolve_cached_model" in source
    assert "casefold()" in source
    assert "validate_frames" in source
    assert 'runpod.serverless.start({"handler": handler})' in source


def test_worker_keeps_the_model_resident_between_jobs() -> None:
    """v2: one wan.WanTI2V instance per process; no generate.py subprocess per job."""
    source = (ROOT / "handler.py").read_text()
    assert "wan.WanTI2V(" in source
    assert "_model_lock" in source
    assert "def get_model" in source
    assert "subprocess" not in source
    assert "generate.py" not in source.replace("spawning `generate.py`", "")
    assert "offload_model=OFFLOAD_MODEL" in source
    assert "t5_cpu=T5_CPU" in source
    assert "save_video(" in source


def test_dockerfile_uses_runtime_and_sdpa_safe_dependency_install() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")
    assert "cuda12.8-cudnn9-runtime" in dockerfile
    assert "cuda12.8-cudnn9-devel" not in dockerfile
    assert "wan-requirements-no-flash.txt" in dockerfile
    assert "sed '/^flash_attn$/d' requirements.txt" in dockerfile
    assert "einops==0.8.1" in dockerfile
    assert "decord==0.6.0" in dockerfile
    assert "librosa==0.10.2.post1" in dockerfile
    assert "peft==0.17.0" in dockerfile
    assert "from .attention import attention" in dockerfile
    assert "flash_attention = attention" in dockerfile
    assert "s/flash_attention(/attention(/g" in dockerfile
