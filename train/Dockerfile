# syntax=docker/dockerfile:1
#
# The loomtrain image: loom's training plane as a runnable environment.
# Every GPU rental has re-discovered the same dependency traps by pain
# (transformers 5.x drops fused-expert quant_state -> pin 4.55; the CUDA
# build dictates the torch wheel; gguf-py for dequantizing the deployed
# quant). This image codifies those pins, so a rented box becomes
#   docker run --gpus all ghcr.io/ch4r10t33r/loom-train recover-lora ...
# and Vast/RunPod-style providers can take it directly as the box template.
#
# Deliberately separate from the serving image (../Dockerfile): that one is
# a static Zig binary measured in megabytes; this one is a CUDA/torch
# environment measured in gigabytes. The two planes meet only through
# artifact files (LPG1, LRA1, GGUF), never in one image.
#
# Base pinned by digest, not tag (security issue #33): a tag is mutable, so
# an unpinned base silently changes the compiler of every trained artifact.
# pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime as of 2026-09-18; CUDA 12.1
# covers the 3090/4090-class boxes the batteries rent. Re-pin deliberately.
FROM pytorch/pytorch@sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067

# The house pins, codified (decision log 2026-08-11: transformers 5.x
# native-aborts on fused-expert quant state; 4.55.4 is the known-good).
# torch itself comes from the base image and is never reinstalled here.
RUN pip install --no-cache-dir \
        "transformers==4.55.4" \
        "accelerate>=1.0,<2" \
        "datasets>=3.0,<4" \
        "gguf>=0.10" \
        "bitsandbytes>=0.44" \
        "numpy<3"

COPY . /opt/loomtrain
RUN pip install --no-cache-dir /opt/loomtrain

# unbuffered by default: buffered training logs have hidden progress before
# (the house recipe's `python3 -u` baked in)
ENV PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["loomtrain"]
CMD ["--help"]
