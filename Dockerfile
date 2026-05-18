# SuperInfer development image.
#
# This image provides only the system-level dependencies that you can't
# install without sudo: apt packages (libzmq, cppzmq, msgpack-cxx,
# gcc-12, ...) and the ``uv`` binary. The repo itself is meant to be
# bind-mounted at run time so edits flow through immediately and you
# keep host file ownership.
#
# Build:
#   docker build -t superinfer .
#
# Run (bind-mount the repo, map host UID/GID, redirect HOME so uv /
# torch / pip caches stay in the project):
#   docker run --gpus all --shm-size=8g -it --rm \
#       --user "$(id -u):$(id -g)" \
#       -v "$PWD":/workspace -w /workspace \
#       -e HOME=/workspace \
#       superinfer
#
# First time inside the container, set up the venv (it lives at
# ./.venv on the host because /workspace is bind-mounted):
#   uv venv --python 3.12 .venv
#   uv pip install --torch-backend=cu124 \
#       -r requirements-build.txt -r requirements-cuda.txt
#   uv pip install -e . --no-build-isolation

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- System packages ---------------------------------------------------- #
# * gcc-12 / g++-12: nvcc shipped with the cu124 torch wheel rejects
#   gcc >= 13 (the Ubuntu 24.04 default).
# * libzmq3-dev / cppzmq-dev / libmsgpack-cxx-dev: headers consumed by
#   the JIT-built SuperInfer swap thread (vllm/v1/swapper/native/).
# * numactl: AE_AIO_0309_v1.py pins the server to NUMA node 0.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc-12 g++-12 \
        ninja-build \
        libzmq3-dev cppzmq-dev libmsgpack-cxx-dev \
        numactl \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv into /usr/local/bin so it is available to any UID.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv  /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Force gcc-12 / g++-12 globally so vLLM's csrc CMake build (and the
# JIT swap-thread compile at runtime) use a host compiler nvcc accepts.
ENV CC=gcc-12 \
    CXX=g++-12 \
    CUDAHOSTCXX=g++-12

WORKDIR /workspace

CMD ["/bin/bash"]
