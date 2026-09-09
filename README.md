# Video Factory — Wan 2.2 TI2V-5B worker

**v2 (2026-09-09): persistent model.** The worker loads `wan.WanTI2V` once per process (preloaded at boot) and keeps it on the GPU; v1 spawned `generate.py` per job, re-reading ~34GB of weights and using CPU offload, which cost 332 s per 49-frame clip. Knobs: `WAN_T5_CPU`, `WAN_OFFLOAD_MODEL` (both `false` by default, tuned for a 48GB card; set `WAN_T5_CPU=true` on 24GB cards), `WAN_PRELOAD`.

This worker is the bounded native-720p-class replacement for the unworkable Hunyuan full-repository cache route. It uses the official public `Wan-AI/Wan2.2-TI2V-5B` model, which supports both text-to-video and image-to-video through the `ti2v-5B` task at `1280*704` and 24 fps.

## Cost-safe endpoint profile

Configure the Runpod endpoint Model field as `Wan-AI/Wan2.2-TI2V-5B`, select one compatible 24GB GPU class, set maximum workers to `1`, keep active workers at `0`, and do **not** attach a Network Volume. Use a larger ephemeral disk than the prior 30GB setting because the model repository is approximately 34.2GB before worker overhead and output files. The endpoint should remain scale-to-zero.

The `health` action only reports whether Model Caching has mounted a snapshot. It does not generate media. Generation accepts a single `t2v` or `i2v` job, downloads any I2V source from a presigned URL, and uploads the resulting MP4 through a presigned PUT URL. No application storage key or Runpod API key belongs in this repository.

## Validation

Run `python run_contract_tests.py` for static contract checks. These tests do not pull weights, build the Docker image, start a GPU, or generate media.
