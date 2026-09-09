FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128     WAN_T5_CPU=false     WAN_OFFLOAD_MODEL=false     WAN_PRELOAD=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git wan

WORKDIR /opt/wan
RUN sed '/^flash_attn$/d' requirements.txt > /tmp/wan-requirements-no-flash.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/wan-requirements-no-flash.txt \
    && pip install decord==0.6.0 einops==0.8.1 librosa==0.10.2.post1 peft==0.17.0 runpod==1.7.10 huggingface_hub==0.32.4 requests==2.32.3

# Wan's model module calls flash_attention directly even though its attention
# wrapper implements a PyTorch SDPA fallback. Route those calls through that
# wrapper because Flash Attention 2 is deliberately not installed in this
# portable runtime image.
RUN sed -i \
    -e 's/from \.attention import flash_attention/from .attention import attention/' \
    -e '/from \.attention import attention/a flash_attention = attention' \
    -e 's/flash_attention(/attention(/g' \
    wan/modules/model.py \
    && grep -q 'from .attention import attention' wan/modules/model.py \
    && grep -q '^flash_attention = attention$' wan/modules/model.py \
    && ! grep -q 'flash_attention(' wan/modules/model.py

COPY handler.py /opt/worker/handler.py
WORKDIR /opt/worker

# The model is intentionally not embedded in the image. Configure the endpoint
# Model field with Wan-AI/Wan2.2-TI2V-5B. Runpod Model Caching mounts its
# downloaded snapshot below /runpod-volume/huggingface-cache/hub. FlashAttention
# is intentionally excluded from build because Wan falls back to PyTorch SDPA.
CMD ["python", "handler.py"]
