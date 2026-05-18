import pickle
import queue
import signal
import threading
import time
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import List, Optional, Tuple, Type

import zmq
import zmq.asyncio
from msgspec import msgpack
from nvtx import nvtx

from vllm.config import CacheConfig, VllmConfig
from vllm.executor.multiproc_worker_utils import get_mp_context
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.usage.usage_lib import UsageContext
from vllm.v1.core.scheduler import Scheduler, SchedulerOutput
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreProfile, EngineCoreRequest,
                            EngineCoreRequestType, EngineCoreRequestUnion,
                            EngineCoreStats, EngineCoreTimeRecord)
from vllm.v1.engine.mm_input_mapper import MMInputMapperServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutorProcess
from vllm.v1.executor.uniproc_executor import UniprocExecutorProcess
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import PickleEncoder
from vllm.v1.utils import TimeMeasurement, make_zmq_socket
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000
LOGGING_TIME_S = POLLING_TIMEOUT_S


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        usage_context: UsageContext,
    ):
        assert vllm_config.model_config.runner_type != "pooling"
        logger.info("Initializing an LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        # SuperInfer always runs the model in a dedicated executor process so
        # that the swap thread can co-locate with the model worker and share
        # its CUDA context. Both single- and multi-GPU paths use the
        # ``*ExecutorProcess`` variants.
        self.model_executor = executor_class(vllm_config)
        assert isinstance(
            self.model_executor,
            (UniprocExecutorProcess, MultiprocExecutorProcess)), (
                "SuperInfer requires UniprocExecutorProcess or "
                "MultiprocExecutorProcess; got "
                f"{type(self.model_executor).__name__}")
        self.queue_in_execute_model = self.model_executor.get_queue_in()
        self.queue_out_execute_model = (
            self.model_executor.get_queue_out_execute_model())

        self.scheduler_output_prev: Optional[SchedulerOutput] = None

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config.cache_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        # The swapper itself lives in the executor process; this just primes
        # the C++ swap thread with the GPU/CPU cache layout.
        self.model_executor.init_swapper(num_cpu_blocks)

        # Setup scheduler.
        self.scheduler = Scheduler(vllm_config.scheduler_config,
                                   vllm_config.cache_config,
                                   vllm_config.lora_config)

        self._last_logging_time = time.time()

        self.mm_input_mapper_server = MMInputMapperServer(
            vllm_config.model_config)

        self.observability_config = vllm_config.observability_config
        self.need_collection = vllm_config.observability_config.need_collection()
        self.prev_lookahead = False

    def _initialize_kv_caches(self,
                              cache_config: CacheConfig) -> Tuple[int, int]:
        start = time.time()
        num_gpu_blocks, num_cpu_blocks = (
            self.model_executor.determine_num_available_blocks())

        if cache_config.num_gpu_blocks_override is not None:
            override = cache_config.num_gpu_blocks_override
            assert num_gpu_blocks >= override, (
                f"num_gpu_blocks_override={override} exceeds the "
                f"profiled budget ({num_gpu_blocks})")
            logger.info("Overriding num_gpu_blocks=%d -> %d",
                        num_gpu_blocks, override)
            num_gpu_blocks = override

        self.model_executor.initialize(num_gpu_blocks, num_cpu_blocks)
        elapsed = time.time() - start
        logger.info(
            "init engine (profile, create kv cache, warmup model) "
            "took %.2f seconds", elapsed)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for an image, then it will be fetched
            # from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client side of the
            # MM mapper, so anything that has a hash must have a HIT cache
            # entry here as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_mapper_server.process_inputs(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)

        self.scheduler.add_request(req)


    @nvtx.annotate("step", color="blue")
    def step(self) -> Tuple[List[EngineCoreOutput], dict]:

        

        has_unfinished_requests = self.scheduler.has_unfinished_requests()
        has_unfinished_executions = (self.scheduler_output_prev is not None) \
            and self.scheduler_output_prev.total_num_scheduled_tokens > 0

        schedule_time = 0
        swap_time = 0
        model_execute_time = 0
        wait_model_execute_time = 0
        update_time = 0
        reschedule_time = 0
        deepcopy_time = 0
        bandwidth = 0
        blocks_swap_in = 0
        blocks_swap_out = 0

        # start model exec in a standalone process
        # if has_unfinished_executions:
        #     self.queue_in_execute_model.put_nowait(("execute_model", (self.scheduler_output_prev, )))

        # run schedule and swapping in main process
        if has_unfinished_requests:
            timer_schedule = TimeMeasurement("schedule", self.observability_config.collect_schedule_time)
            with timer_schedule:
                scheduler_output = self.scheduler.schedule(lookahead=has_unfinished_executions)
            schedule_time = timer_schedule.elapsed_time
        else:
            scheduler_output = None

        need_swapping = (scheduler_output is not None) and \
                        ((len(scheduler_output.blocks_to_swap_in) > 0) or
                         (len(scheduler_output.blocks_to_swap_out) > 0))
        if need_swapping:
            timer_swap = TimeMeasurement("swap", self.observability_config.collect_swap_time)
            with timer_swap:
                self.model_executor.swap(scheduler_output.blocks_to_swap_in, scheduler_output.blocks_to_swap_out)
            swap_time = timer_swap.elapsed_time
            gbs = (len(scheduler_output.blocks_to_swap_in) +
                   len(scheduler_output.blocks_to_swap_out)) * 2 / 1024
            bandwidth = gbs / swap_time
            blocks_swap_in = len(scheduler_output.blocks_to_swap_in)
            blocks_swap_out = len(scheduler_output.blocks_to_swap_out)
            scheduler_output.blocks_to_swap_in = None
            scheduler_output.blocks_to_swap_out = None

        if has_unfinished_executions:
            # wait background model exec
            timer_wait_model_exec = TimeMeasurement("wait_model_exec", self.observability_config.collect_model_execute_time)
            with timer_wait_model_exec:
                output = self.queue_out_execute_model.get()
            # print("get model_exec return time", time.perf_counter())
            model_execute_time = output.model_execute_time
            wait_model_execute_time = timer_wait_model_exec.elapsed_time

            # update
            timer_update = TimeMeasurement("update", self.observability_config.collect_update_time)
            with timer_update:
                engine_core_outputs, stopped_ids = self.scheduler.update_from_output(self.scheduler_output_prev, output, self.prev_lookahead)
            update_time = timer_update.elapsed_time

            # reschedule
            if scheduler_output is None:
                self.scheduler_output_prev = None
            else:
                timer_reschedule = TimeMeasurement("reschedule", self.observability_config.collect_reschedule_time)
                with timer_reschedule:
                    self.scheduler_output_prev = self.scheduler.reschedule(scheduler_output, stopped_ids)
                reschedule_time = timer_reschedule.elapsed_time
            self.prev_lookahead = has_unfinished_executions

        else:
            self.scheduler_output_prev = scheduler_output
            engine_core_outputs = []

        # prevent data race by deepcopy request states
        if self.scheduler_output_prev is not None:
            # TODO(jiahuan): maybe remove it
            # timer_deepcopy = TimeMeasurement("deepcopy", self.observability_config is not None)
            # with timer_deepcopy:
            #     self.scheduler_output_prev.scheduled_running_reqs = \
            #         copy.deepcopy(self.scheduler_output_prev.scheduled_running_reqs)
            # timer_deepcopy.print_if_enabled()
            # deepcopy_time = timer_deepcopy.elapsed_time
            #
            # # start model exec of next iteration, to overlap some queue management time
            # has_unfinished_executions = self.scheduler_output_prev.total_num_scheduled_tokens > 0
            # if has_unfinished_executions:
            #     print("put model_exec time", time.perf_counter())
            #     self.queue_in_execute_model.put_nowait(("execute_model", (self.scheduler_output_prev, )))
            has_unfinished_executions = self.scheduler_output_prev.total_num_scheduled_tokens > 0
            if has_unfinished_executions:
                # print("[zmq] put model_exec time", time.perf_counter())
                self.model_executor.zmq_start_model_execute_async(self.scheduler_output_prev)

        if self.need_collection:
            observability = {
                "scheduler_counter": self.scheduler.schedule_counter,
                "schedule_time": schedule_time,
                "swap_time": swap_time,
                "model_execute_time": model_execute_time,
                "wait_model_execute_time": wait_model_execute_time,
                "update_time": update_time,
                "reschedule_time": reschedule_time,
                "deepcopy_time": deepcopy_time,
                "bandwidth": bandwidth,
                "blocks_swap_in": blocks_swap_in,
                "blocks_swap_out": blocks_swap_out,
            }
        else:
            observability = None

        return engine_core_outputs, observability



    def shutdown(self):
        self.model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def _get_num_requests(self):
        return (
            self.scheduler.get_running_len(),
            self.scheduler.get_waiting_len(),
            self.scheduler.get_swapped_len(),
            self.scheduler.get_total_waiting(),
        )

    def _get_kv_caches_usage(self):
        return self.scheduler.kv_cache_manager.get_kv_caches_usage()

    def get_stats(self):
        n_running, n_waiting, n_swapped, total_waiting = self._get_num_requests()
        gpu_usage, cpu_usage = self._get_kv_caches_usage()
        return {
            "num_requests": {
                "running": n_running,
                "waiting": n_waiting,
                "swapped": n_swapped,
                "total_waiting": total_waiting,
            },
            "kv_caches_usage": {
                "gpu": gpu_usage,
                "cpu": cpu_usage,
            }
        }
    def core_get_time_record(self):
        """Get time record of the scheduler."""
        return self.get_stats()
@dataclass
class EngineCoreProcHandle:
    proc: BaseProcess
    ready_path: str
    input_path: str
    output_path: str
    output_stats_path: str


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
        output_stats_path: str,
    ):
        super().__init__(vllm_config, executor_class, usage_context)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue: queue.Queue[EngineCoreRequestUnion] = queue.Queue()
        self.output_queue: queue.Queue[List[EngineCoreOutput]] = queue.Queue()
        self.input_queue_extra: queue.Queue = queue.Queue()
        self.output_queue_extra: queue.Queue = queue.Queue()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, output_stats_path),
                         daemon=True).start()

        # Send Readiness signal to EngineClient.
        with make_zmq_socket(ready_path, zmq.constants.PUSH) as ready_socket:
            ready_socket.send_string(EngineCoreProc.READY_STR)
        self.need_collection = vllm_config.observability_config.need_collection()

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> None:
        """Wait until the EngineCore is ready."""

        try:
            sync_ctx = zmq.Context()  # type: ignore[attr-defined]
            socket = sync_ctx.socket(zmq.constants.PULL)
            socket.connect(ready_path)

            # Wait for EngineCore to send EngineCoreProc.READY_STR.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for EngineCoreProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("EngineCoreProc failed to start.")

            message = socket.recv_string()
            assert message == EngineCoreProc.READY_STR

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            sync_ctx.destroy(linger=0)

    @staticmethod
    def make_engine_core_process(
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
        output_stats_path: str,
    ) -> EngineCoreProcHandle:
        context = get_mp_context()

        process_kwargs = {
            "input_path": input_path,
            "output_path": output_path,
            "ready_path": ready_path,
            "output_stats_path": output_stats_path,
            "vllm_config": vllm_config,
            "executor_class": executor_class,
            "usage_context": usage_context,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=EngineCoreProc.run_engine_core,
                               kwargs=process_kwargs)
        proc.start()

        # Wait for startup
        EngineCoreProc.wait_for_startup(proc, ready_path)
        return EngineCoreProcHandle(proc=proc,
                                    ready_path=ready_path,
                                    input_path=input_path,
                                    output_path=output_path,
                                    output_stats_path=output_stats_path)

    @staticmethod
    def run_engine_core(*args, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core = None
        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            if engine_core is not None:
                engine_core.shutdown()
                engine_core = None

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(req)
                        break
                    except queue.Empty:
                        self._log_stats()
                        logger.debug("EngineCore busy loop waiting.")
                    except BaseException:
                        raise

            # 2) Handle any new client requests (Abort or Add).
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(req)

            if self.scheduler.has_unfinished_requests():

                if self.observability_config.collect_step_time:
                    start = time.perf_counter()

                # 3) Step the engine core.
                outputs, observability = self.step()

                if self.observability_config.collect_step_time:
                    end = time.perf_counter()
                    observability["step_time"] = end - start

                # 4) Put EngineCoreOutputs into the output queue.
                if self.need_collection:
                    self.output_queue.put_nowait((outputs, observability))
                else:
                    self.output_queue.put_nowait(outputs)

            self._log_stats()

    def _log_stats(self):
        """Log basic stats every LOGGING_TIME_S"""

        now = time.time()

        if now - self._last_logging_time > LOGGING_TIME_S:
            n_running, n_waiting, n_swapped, total_waiting = self._get_num_requests()
            gpu_usage, cpu_usage = self._get_kv_caches_usage()
            logger.info(
                "RUNNING: %s | WAITING: %s | SWAPPED: %s | TOTAL_WAITING: %s, GPU KV Cache: %f%% | CPU KV Cache: %f%%",
                n_running, n_waiting, n_swapped, total_waiting, gpu_usage * 100, cpu_usage * 100,
            )

            self._last_logging_time = now

    def _handle_client_request(self, request: EngineCoreRequestUnion) -> None:
        """Handle EngineCoreRequest or EngineCoreABORT from Client."""

        if isinstance(request, EngineCoreRequest):
            self.add_request(request)
        elif isinstance(request, EngineCoreProfile):
            self.model_executor.profile(request.is_start)
        elif isinstance(request, EngineCoreStats):
            self.output_queue.put_nowait(self.get_stats())
        elif isinstance(request, EngineCoreTimeRecord):
            self.output_queue.put_nowait(self.core_get_time_record())
        else:
            # TODO: make an EngineCoreAbort wrapper
            assert isinstance(request, list)
            self.abort_requests(request)

    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        decoder_add_req = PickleEncoder()
        decoder_abort_req = PickleEncoder()

        with make_zmq_socket(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = type_frame.buffer
                request_data = data_frame.buffer

                # Deserialize the request data.
                if request_type == EngineCoreRequestType.ADD.value:
                    request = decoder_add_req.decode(request_data)
                elif request_type == EngineCoreRequestType.ABORT.value:
                    request = decoder_abort_req.decode(request_data)
                elif request_type == EngineCoreRequestType.PROFILE.value:
                    request = pickle.loads(request_data)
                elif request_type == EngineCoreRequestType.STATS.value:
                    request = pickle.loads(request_data)
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait(request)

    def process_output_socket(self, output_path: str, output_stats_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()
        buffer_stats = bytearray()
        buffer_tr = bytearray()

        with make_zmq_socket(output_path, zmq.constants.PUSH) as socket:
            with make_zmq_socket(output_stats_path, zmq.constants.PUSH) as socket_stats:
                while True:
                    obj = self.output_queue.get()

                    if type(obj) is dict:  # stats
                        outputs = obj
                        encoder.encode_into(outputs, buffer_stats)
                        socket_stats.send_multipart((buffer_stats, ), copy=False)
                    else:
                        if self.need_collection:
                            engine_core_outputs, observability = obj
                        else:
                            engine_core_outputs = obj
                            observability = None
                        outputs = EngineCoreOutputs(outputs=engine_core_outputs, observability=observability)
                        encoder.encode_into(outputs, buffer)
                        socket.send_multipart((buffer, ), copy=False)
