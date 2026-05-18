# SuperInfer

[![Paper](https://img.shields.io/badge/Paper-MLSys%202026-b31b1b?logo=readthedocs&logoColor=white)](https://mlsys.org/virtual/2026/poster/3586)
[![Blog](https://img.shields.io/badge/Blog-SuperInfer-2088FF?logo=githubpages&logoColor=white)](https://supercomputing-system-ai-lab.github.io/projects/superinfer/)

Code repository for the MLSys 2026 paper **"SuperInfer: SLO-Aware
Rotary Scheduling and Memory Management for LLM Inference on
Superchips"**. Reference implementation and artifact-evaluation
harness, built as a fork of [vLLM v0.6.6.post1](https://github.com/vllm-project/vllm/tree/v0.6.6.post1).
The original vLLM README is preserved at
[`README.vLLM.md`](README.vLLM.md).

The SuperInfer additions live mostly under:

- [`vllm/v1/swapper/`](vllm/v1/swapper/) — the multi-threaded GPU↔CPU
  KV-cache swapper (Python wrapper + JIT-built C++ thread).
- [`vllm/v1/core/scheduler.py`](vllm/v1/core/scheduler.py) and
  [`vllm/v1/core/kv_cache_manager.py`](vllm/v1/core/kv_cache_manager.py) —
  the LVF scheduler with proactive swap-out budgeting.
- [`vllm/v1/engine/core.py`](vllm/v1/engine/core.py) +
  [`vllm/v1/executor/{uniproc,multiproc}_executor.py`](vllm/v1/executor/) —
  step pipeline that overlaps schedule / swap / model-execute across the
  engine-core and executor processes.
- [`benchmarks/superinfer/`](benchmarks/superinfer/) — the AE-wired
  AsyncEngine HTTP server + load generator.
- [`AE_AIO_0309_v1.py`](AE_AIO_0309_v1.py) — the artifact-evaluation
  driver.

All SuperInfer knobs are exposed as CLI flags on `EngineArgs`
(`--proactive-swap-budget`, `--swapper-block-first`, `--pin-memory-fix`,
`--prefix-cache-fix`).

## Hardware

The configuration validated by the AE runs uses:

- a single NVIDIA GH200 with 144 GB of HBM and 480 GB of DRAM;
  other Ampere / Hopper cards should work but are untested.
- a CUDA 12.4 driver.

## Setup

The fastest path is the included [Dockerfile](Dockerfile); see
[Setup with Docker](#setup-with-docker) below. To build directly on the
host instead, follow the bare-metal steps.

## Setup (bare metal)

### 1. System packages

The native swap thread links against libzmq / cppzmq / msgpack-cxx, and
the cu124 torch wheel ships an nvcc that rejects gcc >= 13 (the Ubuntu
24.04 default), so we also install gcc-12:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    gcc-12 g++-12 \
    libzmq3-dev cppzmq-dev libmsgpack-cxx-dev \
    numactl

# Make nvcc / CMake pick up gcc-12 instead of the system default.
export CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12
```

### 2. Python environment (uv)

Install [uv](https://github.com/astral-sh/uv) if you don't have it
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### 3. Install SuperInfer

```bash
# Install build-time deps (cmake / ninja / setuptools-scm) and runtime
# deps (torch, torchvision, ray, flashinfer, ...). The
# ``--torch-backend=cu124`` flag tells uv to pull torch / torchvision
# from https://download.pytorch.org/whl/cu124 (other deps still come
# from PyPI).
uv pip install --torch-backend=cu124 \
    -r requirements-build.txt -r requirements-cuda.txt

# Build vLLM's native kernels (csrc/) in editable mode against the deps
# we just installed (takes a few minutes the first time). The CC / CXX /
# CUDAHOSTCXX vars from step 1 must be set for nvcc to pick up gcc-12.
CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12 \
    uv pip install -e . --no-build-isolation
```

The SuperInfer C++ swap thread under
[`vllm/v1/swapper/native/`](vllm/v1/swapper/native/) is **not** built
here; it is JIT-compiled by `torch.utils.cpp_extension.load` on the
first request to the swapper (~30 s, cached under
`~/.cache/torch_extensions/`).

### 4. Sanity check

```bash
python -c "from vllm import LLM; print('vLLM import OK')"
python AE_AIO_0309_v1.py --help
```

## Setup with Docker

The image only bakes in the system-level pieces that need sudo to fix
(apt packages, gcc-12, the `uv` binary). Source lives on the host and
is bind-mounted in, so edits flow through immediately and any files
the build produces (the `.venv`, JIT caches, `ae_results/`) keep your
host UID.

Base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`. Host needs an
NVIDIA driver compatible with CUDA 12.8 and
[`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# 1. Build the env image once (~3 min, mostly apt). No source baked in.
docker build -t superinfer .

# 2. Drop into a dev shell. The repo is mounted at /workspace, the
#    container runs as your host UID/GID, and HOME is redirected to
#    /workspace so uv / torch / pip caches stay in the project tree.
docker run --gpus all --shm-size=8g -it --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD":/workspace -w /workspace \
    -e HOME=/workspace \
    superinfer
```

The first time inside the container, run the same install steps as the
bare-metal flow above (steps 2–4). The `.venv` will land at `./.venv`
on the host. Subsequent shells skip the install and go straight to:

```bash
source .venv/bin/activate
python AE_AIO_0309_v1.py --output-dir ./ae_results
```

## Running the artifact evaluation

```bash
python AE_AIO_0309_v1.py --output-dir ./ae_results
```

This sweeps `(model, dataset, rps)` combinations end-to-end (server +
load generator). Each run writes its records / stats to a per-config
subdirectory of `./ae_results/`.

`AE_AIO_0309_v1.py` invokes `<repo>/.venv/bin/python` directly, so
`source .venv/bin/activate` is **not** required before running it.

## Repository layout (post-cleanup)

```
.
├── AE_AIO_0309_v1.py           # artifact-evaluation driver
├── benchmarks/superinfer/      # AE server / client / dataset sampler
├── vllm/                       # forked vLLM 0.6.6.post1 sources
│   └── v1/
│       ├── swapper/            # SuperInfer swap module
│       │   └── native/         # C++ sources, JIT-built on first use
│       ├── core/               # LVF scheduler + KV-cache manager
│       ├── engine/             # engine-core (single-process pipeline)
│       └── executor/           # *ExecutorProcess co-locates the swap thread
├── csrc/                       # vLLM's native CUDA kernels
└── requirements-*.txt          # pinned at the AE-validated versions
```

The pre-cleanup state (with all alternative scheduler / swapper
variants and the full experiment dump) is preserved on the
`backup-um-pre-reset` branch and tagged `pre-cleanup-with-um`.
