import os
import pickle
import struct
from typing import Dict, List, Optional, Tuple

import msgspec
import nvtx
import torch.multiprocessing as mp
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port)
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.swapper import (SWAPPER_DATA_PATH, SWAPPER_NOTIFY_PATH, Swapper)
from vllm.v1.utils import TimeMeasurement
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class UniprocExecutor(Executor):

    def __init__(self,
                 vllm_config: VllmConfig,
                 collect_model_execute_time: bool = False) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.collect_model_execute_time = collect_model_execute_time

        self.worker: Worker = self._create_worker()
        self.worker.initialize()
        self.worker.load_model()

    def _create_worker(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Worker:
        # https://github.com/NVIDIA/nccl/issues/1234
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return Worker(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.worker.determine_num_available_blocks()

    def initialize(self,
                   num_gpu_blocks: int,
                   num_cpu_blocks: int = 0) -> None:
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)
        self.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        self.worker.compile_or_warm_up_model()

    @nvtx.annotate("UniprocExecutor.execute_model", color="blue")
    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        timer = TimeMeasurement("model_execute",
                                self.collect_model_execute_time)
        with timer:
            output = self.worker.execute_model(scheduler_output)
        output.model_execute_time = timer.elapsed_time
        return output

    def profile(self, is_start: bool = True):
        self.worker.profile(is_start)

    def shutdown(self):
        pass

    def check_health(self) -> None:
        # Always healthy as long as the process is alive.
        return


# IPC path for the engine-core -> executor model-execute fast path.
_MODEL_EXEC_IPC = "ipc:///tmp/model-exec-input"


class UniprocExecutorProcess(Executor):
    """Runs :class:`UniprocExecutor` in a dedicated process.

    The model executor lives in a child process so that the SuperInfer
    swap thread can co-locate with it (sharing one CUDA context).
    The parent (engine-core) talks to the child via two channels:

    * ``self._queue_in`` / ``self._queues_out`` — control RPCs (init, swap
      bootstrap, profile, shutdown);
    * a ZMQ ``PUSH``/``PULL`` socket pair on ``_MODEL_EXEC_IPC`` for the
      hot model-execute path (avoids the queue's pickle overhead).

    Once the child has booted the swap thread, the parent also drives it
    directly via ``SWAPPER_DATA_PATH`` / ``SWAPPER_NOTIFY_PATH``.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        ctx = mp.get_context("spawn")
        self._queue_in = ctx.Queue()
        self._queues_out: Dict[str, mp.Queue] = {
            "__init__": ctx.Queue(),
            "determine_num_available_blocks": ctx.Queue(),
            "initialize": ctx.Queue(),
            "execute_model": ctx.Queue(),
            "profile": ctx.Queue(),
            "shutdown": ctx.Queue(),
            "check_health": ctx.Queue(),
            "init_swapper": ctx.Queue(),
        }
        self._inner_process = ctx.Process(
            target=self._inner_process_fn,
            args=(self._queue_in, self._queues_out, vllm_config,
                  vllm_config.observability_config.collect_model_execute_time),
        )
        self._inner_process.start()
        self._queues_out["__init__"].get()  # wait for child to come up

        # Fast path for model execution: ZMQ PUSH/PULL across the IPC.
        zmq_context = zmq.Context()
        self.model_exec_socket = zmq_context.socket(zmq.PUSH)
        self.model_exec_socket.connect(_MODEL_EXEC_IPC)

        # Swapper sockets are bound lazily in ``init_swapper``.
        self._swap_data_socket: Optional[zmq.Socket] = None
        self._swap_notify_socket: Optional[zmq.Socket] = None
        self._swap_encoder = msgspec.msgpack.Encoder()

    @staticmethod
    def _inner_process_fn(
        queue_in: mp.Queue,
        queues_out: Dict[str, mp.Queue],
        vllm_config: VllmConfig,
        collect_model_execute_time: bool,
    ):
        executor = UniprocExecutor(vllm_config, collect_model_execute_time)

        zmq_context = zmq.Context()
        model_exec_socket = zmq_context.socket(zmq.PULL)
        model_exec_socket.bind(_MODEL_EXEC_IPC)

        queues_out["__init__"].put_nowait(None)

        while True:
            # Hot path: model-execute requests have priority and bypass
            # the control queue's pickle overhead.
            if model_exec_socket.poll(timeout=0) == zmq.POLLIN:
                scheduler_output = model_exec_socket.recv_pyobj()
                queues_out["execute_model"].put_nowait(
                    executor.execute_model(scheduler_output))
                continue

            try:
                fn_name, args = queue_in.get_nowait()
            except Exception:
                continue

            if fn_name == "init_swapper":
                num_cpu_blocks, = args
                kv_cache_gpu = (
                    executor.worker.model_runner.kv_caches_gpu_tensor)
                cache_cfg = vllm_config.cache_config
                # Constructed for its side-effects (allocates pinned host
                # memory and starts the C++ swap thread).
                Swapper(
                    vllm_config,
                    kv_cache_gpu,
                    num_cpu_blocks,
                    swap_block_first=cache_cfg.swap_block_first,
                    pin_memory_fix=cache_cfg.pin_memory_fix,
                )
                queues_out["init_swapper"].put_nowait(None)
                continue

            fn = getattr(executor, fn_name)
            result = fn(*args) if args is not None else fn()
            queues_out[fn_name].put_nowait(result)

    # --- Control RPCs ------------------------------------------------- #

    def get_queue_in(self) -> mp.Queue:
        return self._queue_in

    def get_queue_out_execute_model(self) -> mp.Queue:
        return self._queues_out["execute_model"]

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        self._queue_in.put_nowait(("determine_num_available_blocks", None))
        return self._queues_out["determine_num_available_blocks"].get()

    def initialize(self,
                   num_gpu_blocks: int,
                   num_cpu_blocks: int = 0) -> None:
        self._queue_in.put_nowait(
            ("initialize", (num_gpu_blocks, num_cpu_blocks)))
        return self._queues_out["initialize"].get()

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        self._queue_in.put_nowait(("execute_model", (scheduler_output, )))
        return self._queues_out["execute_model"].get()

    def profile(self, is_start: bool = True):
        self._queue_in.put_nowait(("profile", (is_start, )))
        return self._queues_out["profile"].get()

    def shutdown(self):
        self._queue_in.put_nowait(("shutdown", None))
        return self._queues_out["shutdown"].get()

    def check_health(self) -> None:
        self._queue_in.put_nowait(("check_health", None))
        return self._queues_out["check_health"].get()

    # --- Swap fast path ----------------------------------------------- #

    def init_swapper(self, num_cpu_blocks: int):
        # Bind the notify socket on the parent before signalling the child
        # so that the C++ thread's first ``connect`` cannot race the bind.
        ctx = zmq.Context.instance()
        self._swap_data_socket = ctx.socket(zmq.PUSH)
        self._swap_notify_socket = ctx.socket(zmq.PULL)
        self._swap_notify_socket.bind(SWAPPER_NOTIFY_PATH)

        self._queue_in.put_nowait(("init_swapper", (num_cpu_blocks, )))

        self._swap_data_socket.connect(SWAPPER_DATA_PATH)
        return self._queues_out["init_swapper"].get()

    @nvtx.annotate("UniprocExecutorProcess.swap", color="blue")
    def swap(
        self,
        in_mapping: List[Tuple[int, int]],
        out_mapping: List[Tuple[int, int]],
    ) -> float:
        self.start_swap_async(in_mapping, out_mapping)
        return self.wait_swap()

    @nvtx.annotate("UniprocExecutorProcess.start_swap_async", color="blue")
    def start_swap_async(
        self,
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        payload = self._swap_encoder.encode(
            [blocks_to_swap_in, blocks_to_swap_out])
        self._swap_data_socket.send(payload, copy=False, flags=zmq.NOBLOCK)

    @nvtx.annotate("UniprocExecutorProcess.wait_swap", color="blue")
    def wait_swap(self) -> float:
        data = self._swap_notify_socket.recv()
        swap_time_us, = struct.unpack("Q", data)
        return swap_time_us / 1_000_000

    @nvtx.annotate("UniprocExecutorProcess.zmq_start_model_execute_async",
                   color="orange")
    def zmq_start_model_execute_async(self, scheduler_output):
        self.model_exec_socket.send_pyobj(
            scheduler_output,
            flags=zmq.NOBLOCK,
            copy=False,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
