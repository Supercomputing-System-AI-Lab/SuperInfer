"""SuperInfer multi-threaded KV-cache swapper.

The actual swap loop lives in C++ (see ``./native/``, JIT-built on first
import — see :mod:`._native`). It is started from the executor process so
that it shares a CUDA context with the model worker.
The Python-side ZMQ sockets that drive the C++ thread live with the
``Executor`` (in the engine-core process); see
``vllm/v1/executor/uniproc_executor.py``.

Block layouts in the CPU mirror are controlled by ``swap_block_first``:
when True the leading dimension is the block index (better for the
multi-threaded copy kernel), otherwise the original vLLM layout is kept.
"""
from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import LayerBlockType
from vllm.v1.swapper import _native

logger = init_logger(__name__)

# IPC paths shared between the Python driver (in the engine-core process)
# and the C++ swap thread (in the executor process).
SWAPPER_DATA_PATH = "ipc:///tmp/swapper-data"
SWAPPER_NOTIFY_PATH = "ipc:///tmp/swapper-notify"


def _gibibytes(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024**3)


def _allocate_pinned_cpu_cache(shape, dtype: torch.dtype,
                               pin_memory_fix: bool) -> torch.Tensor:
    """Allocate the CPU mirror of the KV cache.

    With ``pin_memory_fix`` we explicitly call ``cudaHostRegister`` after
    allocating regular host memory; this works around a torch bug that
    refuses to pin allocations larger than 256 GiB.
    """
    if pin_memory_fix:
        cache = torch.zeros(shape, dtype=dtype, device="cpu")
        cudart = torch.cuda.cudart()
        cudart.cudaHostRegister(cache.data_ptr(),
                                cache.numel() * cache.element_size(), 0)
        assert cache.is_pinned()
        return cache
    return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)


class Swapper:
    """Owns the GPU/CPU KV-cache pair and the native swap thread.

    Instances live in the executor process (so the GPU pointer and the
    pinned host memory share its CUDA context). The native thread reads
    swap requests off ``SWAPPER_DATA_PATH`` and reports completion
    timings on ``SWAPPER_NOTIFY_PATH``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_caches_gpu: torch.Tensor,
        num_cpu_blocks: int,
        *,
        swap_block_first: bool = True,
        pin_memory_fix: bool = True,
    ) -> None:
        self.num_attn_layers = (
            vllm_config.model_config.get_num_layers_by_block_type(
                vllm_config.parallel_config, LayerBlockType.attention))

        self.kv_caches_gpu = kv_caches_gpu
        kv_cache_shape_cpu = list(kv_caches_gpu.shape)
        if swap_block_first:
            kv_cache_shape_cpu[0] = num_cpu_blocks
        else:
            kv_cache_shape_cpu[2] = num_cpu_blocks

        logger.info("Initializing CPU KV caches (block_first=%s)...",
                    swap_block_first)
        self.kv_caches_cpu = _allocate_pinned_cpu_cache(
            kv_cache_shape_cpu, kv_caches_gpu.dtype, pin_memory_fix)
        logger.info("CPU KV caches initialized (%.2f GiB).",
                    _gibibytes(self.kv_caches_cpu))

        _native.start_swapper_thread(
            SWAPPER_DATA_PATH,
            SWAPPER_NOTIFY_PATH,
            self.num_attn_layers,
            self.kv_caches_gpu.data_ptr(),
            self.kv_caches_gpu.element_size(),
            self.kv_caches_gpu.size(),
            self.kv_caches_gpu.stride(),
            self.kv_caches_cpu.data_ptr(),
            self.kv_caches_cpu.element_size(),
            self.kv_caches_cpu.size(),
            self.kv_caches_cpu.stride(),
            swap_block_first,
        )
