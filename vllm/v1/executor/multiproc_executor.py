import os
import pickle
import signal
import sys
import time
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.process import BaseProcess
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import torch
import zmq
import nvtx
import torch.multiprocessing as mp

from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.executor.multiproc_worker_utils import (
    _add_prefix, get_mp_context, set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_open_port,
                        get_open_zmq_ipc_path)
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import make_zmq_socket
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000


class MultiprocExecutor(Executor):

    def __init__(self, vllm_config: VllmConfig) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)

        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        self.workers: List[WorkerProcHandle] = []
        for rank in range(self.world_size):
            worker = WorkerProc.make_worker_process(vllm_config, rank, rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle)
            self.workers.append(worker)

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()

    def initialize(self, num_gpu_blocks: int, num_cpu_blocks: int=0) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, ))
        self.collective_rpc("compile_or_warm_up_model")

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        num_blocks = self.collective_rpc("determine_num_available_blocks")

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def collective_rpc(self,
                       method: str,
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        """
        Execute an RPC call on workers.
        
        Args:
            method: Name of the worker method to execute
            timeout: Maximum time in seconds to wait for execution. Rases a
                     TimeoutError on timeout. None means wait indefinitely.
            args: Positional arguments to pass to the worker method
            kwargs: Keyword arguments to pass to the worker method

        Returns:
            List of results from each worker
        """
        start_time = time.monotonic()
        kwargs = kwargs or {}

        try:
            self.rpc_broadcast_mq.enqueue((method, args, kwargs))

            responses = [None] * self.world_size
            for w in self.workers:
                dequeue_timeout = timeout - (time.monotonic() - start_time
                                             ) if timeout is not None else None
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        raise RuntimeError("Worker failed")

                responses[w.rank] = result

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except Exception as e:
            # Re-raise any other exceptions
            raise e

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        model_output = self.collective_rpc("execute_model",
                                           args=(scheduler_output, ))[0]
        return model_output

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))
        return

    def _ensure_worker_termination(self):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

        self._cleanup_sockets()

    def _cleanup_sockets(self):
        for w in self.workers:
            # Remove the zmq ipc socket file
            socket_path = w.ready_path.replace("ipc://", "")
            if os and os.path.exists(socket_path):
                os.remove(socket_path)

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if getattr(self, 'shutting_down', False):
            self.shutting_down = True
            for w in self.workers:
                w.worker_response_mq = None
            self._ensure_worker_termination()

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    ready_path: str
    worker_response_mq: MessageQueue  # The worker process writes to this MQ


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        ready_path: str,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config)
        wrapper.init_worker(vllm_config, local_rank, rank,
                            distributed_init_method)
        self.worker = wrapper.worker

        pid = os.getpid()
        _add_prefix(sys.stdout, f"VllmWorker rank={rank}", pid)
        _add_prefix(sys.stderr, f"VllmWorker rank={rank}", pid)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
        worker_response_mq_handle = self.worker_response_mq.export_handle()

        # Send Readiness signal to EngineCore process.
        with make_zmq_socket(ready_path, zmq.constants.PUSH) as ready_socket:
            payload = pickle.dumps(worker_response_mq_handle,
                                   protocol=pickle.HIGHEST_PROTOCOL)
            ready_socket.send_string(WorkerProc.READY_STR)
            ready_socket.send(payload)

        self.worker.initialize()
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> WorkerProcHandle:
        context = get_mp_context()

        # ZMQ path for worker to send ready message and shm_broadcast handle
        # back to core process.
        ready_path = get_open_zmq_ipc_path()

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_path": ready_path,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               daemon=True)
        proc.start()

        # Wait for startup
        worker_response_mq_handle = WorkerProc.wait_for_startup(
            proc, ready_path)

        worker_response_mq = MessageQueue.create_from_handle(
            worker_response_mq_handle, 0)

        return WorkerProcHandle(proc, rank, ready_path, worker_response_mq)

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        try:
            worker = WorkerProc(*args, **kwargs)

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()

            worker.worker_busy_loop()

        except SystemExit:
            logger.debug("Worker interrupted.")

        except BaseException as e:
            logger.exception(e)
            raise

        finally:
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
                worker = None

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> Optional[Handle]:
        """Wait until the Worker is ready."""
        with make_zmq_socket(ready_path, zmq.constants.PULL) as socket:

            # Wait for Worker to send READY.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for WorkerProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("WorkerProc failed to start.")

            message = socket.recv_string()
            assert message == WorkerProc.READY_STR
            handle_frame = socket.recv(copy=False)
            handle = pickle.loads(handle_frame.buffer)
            return handle

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs = self.rpc_broadcast_mq.dequeue()

            try:
                output = getattr(self.worker, method)(*args, **kwargs)
            except BaseException as e:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.FAILURE, e))
                continue

            self.worker_response_mq.enqueue(
                (WorkerProc.ResponseStatus.SUCCESS, output))


from vllm import _custom_ops as ops  # naive swapper

_MODEL_EXEC_IPC = "ipc:///tmp/model-exec-input"


class MultiprocExecutorProcess(Executor):
    """Multi-GPU variant of :class:`UniprocExecutorProcess`.

    Mirrors the single-GPU control plane: the model executor lives in a
    child process and SuperInfer's swap thread is bootstrapped there.
    The Python-side ZMQ sockets driving the C++ swap thread live here in
    the engine-core process.
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

        zmq_context = zmq.Context()
        self.model_exec_socket = zmq_context.socket(zmq.PUSH)
        self.model_exec_socket.connect(_MODEL_EXEC_IPC)

        self._swap_data_socket = None
        self._swap_notify_socket = None
        self._swap_encoder = msgspec.msgpack.Encoder()

    @staticmethod
    def _inner_process_fn(
        queue_in: mp.Queue,
        queues_out: Dict[str, mp.Queue],
        vllm_config: VllmConfig,
        collect_model_execute_time: bool,
    ):
        from vllm.v1.swapper import Swapper

        executor = MultiprocExecutor(vllm_config)

        zmq_context = zmq.Context()
        model_exec_socket = zmq_context.socket(zmq.PULL)
        model_exec_socket.bind(_MODEL_EXEC_IPC)

        queues_out["__init__"].put_nowait(None)

        while True:
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

    # --- Control RPCs -------------------------------------------------- #

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
        from vllm.v1.swapper import (SWAPPER_DATA_PATH, SWAPPER_NOTIFY_PATH)

        ctx = zmq.Context.instance()
        self._swap_data_socket = ctx.socket(zmq.PUSH)
        self._swap_notify_socket = ctx.socket(zmq.PULL)
        self._swap_notify_socket.bind(SWAPPER_NOTIFY_PATH)

        self._queue_in.put_nowait(("init_swapper", (num_cpu_blocks, )))

        self._swap_data_socket.connect(SWAPPER_DATA_PATH)
        return self._queues_out["init_swapper"].get()

    @nvtx.annotate("MultiprocExecutorProcess.swap", color="blue")
    def swap(
        self,
        in_mapping: List[Tuple[int, int]],
        out_mapping: List[Tuple[int, int]],
    ) -> float:
        self.start_swap_async(in_mapping, out_mapping)
        return self.wait_swap()

    @nvtx.annotate("MultiprocExecutorProcess.start_swap_async", color="blue")
    def start_swap_async(
        self,
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        payload = self._swap_encoder.encode(
            [blocks_to_swap_in, blocks_to_swap_out])
        self._swap_data_socket.send(payload, copy=False, flags=zmq.NOBLOCK)

    @nvtx.annotate("MultiprocExecutorProcess.wait_swap", color="blue")
    def wait_swap(self) -> float:
        import struct
        data = self._swap_notify_socket.recv()
        swap_time_us, = struct.unpack("Q", data)
        return swap_time_us / 1_000_000

    @nvtx.annotate("MultiprocExecutorProcess.zmq_start_model_execute_async",
                   color="orange")
    def zmq_start_model_execute_async(self, scheduler_output):
        import pickle
        self.model_exec_socket.send_pyobj(
            scheduler_output,
            flags=zmq.NOBLOCK,
            copy=False,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @nvtx.annotate("UniprocExecutorProcess.zmq_start_model_execute_async", color="orange")
    def zmq_start_model_execute_async(self, scheduler_output):
        # print("[zmq] get model_exec time", time.perf_counter())
        # data_serialized = self.model_exec_encoder.encode(scheduler_output)
        # self.model_exec_socket.send(data_serialized, copy=False, flags=zmq.NOBLOCK)
        self.model_exec_socket.send_pyobj(scheduler_output, flags=zmq.NOBLOCK, copy=False, protocol=pickle.HIGHEST_PROTOCOL)

