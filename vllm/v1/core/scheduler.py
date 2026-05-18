import time
from collections import deque
from dataclasses import dataclass
from itertools import chain
from typing import (TYPE_CHECKING, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

import msgspec
import nvtx

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestLookaheadContextManager, RequestStatus

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.base import PlaceholderRange

logger = init_logger(__name__)


class Scheduler:
    """SuperInfer LVF scheduler.

    Picks victims for proactive KV-cache swap-out using the LVF policy
    (see the SuperInfer paper for the full description). The
    ``proactive_swap_budget`` knob (``--proactive-swap-budget``) bounds
    how many GPU blocks are kept proactively free so that incoming
    long-prompt requests do not stall on swap latency.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        # TODO: Support LoRA.
        assert lora_config is None, "V1 does not support LoRA yet."

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        num_cpu_blocks = cache_config.num_cpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0

        # Create the KV cache manager (GPU + CPU swap-aware).
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            max_model_len=self.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            prefix_cache_fix=self.scheduler_config.prefix_cache_fix,
        )

        self.block_size = self.cache_config.block_size
        # req_id -> Request
        self.requests: Dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: Set[str] = set()

        # OPTIMIZATION: Cache the RunningRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> RunningRequestData
        self.running_reqs_data: Dict[str, RunningRequestData] = {}

        # Encoder-related.
        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = self.scheduler_config.max_num_encoder_input_tokens  #noqa: E501
        # NOTE(woosuk): For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized and used, regardless of
        # the cache size. This is because the memory space for the encoder cache
        # is preallocated in the profiling run.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=self.scheduler_config.encoder_cache_size)

        self.schedule_counter = 0
        self.swapping_budget = self.scheduler_config.proactive_swap_budget
        self.proactive_swapping = self.swapping_budget > 0
        self.time_record = []

    def schedule_single_running_request(
        self,
        request: Request,
        token_budget: int,
        encoder_budget: int,
        preempted_reqs: List[Request],
        scheduled_running_reqs: List[Request],
        req_to_new_block_ids: Dict[str, List[int]],
        num_scheduled_tokens: Dict[str, int],
        scheduled_encoder_inputs: Dict[str, List[int]],
        allow_preemption: bool = True,
    ) -> Tuple[int, int, int]:  # num_new_tokens, token_budget, encoder_budget
        # NOTE(julian): num_new_tokens is not 1 means chunked prefill
        num_new_tokens = request.num_tokens - request.num_computed_tokens
        assert token_budget > 0
        num_new_tokens = min(num_new_tokens, token_budget)
        assert num_new_tokens > 0

        # Schedule encoder inputs.
        encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = (
            self._try_schedule_encoder_inputs(request,
                                                request.num_computed_tokens,
                                                num_new_tokens,
                                                encoder_budget))
        assert num_new_tokens > 0

        while True:
            
            new_blocks = self.kv_cache_manager.append_slots(
                request, num_new_tokens)
            if allow_preemption and new_blocks is None:
                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                preempted_req = self.running.pop()
                if self.scheduler_config.eviction_policy == "recomputing":
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    preempted_req.num_computed_tokens_next = 0
                    preempted_req.num_tokens_next = preempted_req.num_tokens
                elif self.scheduler_config.eviction_policy == "swapping":
                    self.kv_cache_manager.swap_out(preempted_req)
                    preempted_req.status = RequestStatus.SWAPPED
                else:
                    raise ValueError(f"Invalid eviction policy: {self.scheduler_config.eviction_policy}")

                self.waiting.append(preempted_req)
                self.time_record.append('WAITING | ' + str(round(time.time(), 2)) + ' | ' + preempted_req.request_id)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt.
                    can_schedule = False
                    break
            else:
                # The request can be scheduled.
                can_schedule = True
                break
        if not can_schedule:
            return None, token_budget, encoder_budget # type: ignore
        assert new_blocks is not None

        # Schedule the request.
        scheduled_running_reqs.append(request)
        req_to_new_block_ids[request.request_id] = [
            b.block_id for b in new_blocks
        ]
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = (
                encoder_inputs_to_schedule)
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_budget = new_encoder_budget

        return num_new_tokens, token_budget, encoder_budget

    def schedule_single_waiting_request(
        self,
        request: Request,
        token_budget: int,
        encoder_budget: int,
        scheduled_new_reqs: List[Request],
        scheduled_resumed_reqs: List[Request],
        req_to_new_block_ids: Dict[str, List[int]],
        num_scheduled_tokens: Dict[str, int],
        scheduled_encoder_inputs: Dict[str, List[int]],
    ) -> Tuple[int, int, int]:

        assert not request.lookahead

        # Get already-cached tokens.
        computed_blocks = self.kv_cache_manager.get_computed_blocks(
            request)
        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * self.block_size

        if request.status in [RequestStatus.WAITING, RequestStatus.PREEMPTED]:
            # Number of tokens to be scheduled.
            # We use `request.num_tokens` instead of
            # `request.num_prompt_tokens` to consider the resumed requests,
            # which have output tokens.
            num_new_tokens = request.num_tokens - num_computed_tokens
            if num_new_tokens == 0:
                # The happens when prompt length is divisible by the block
                # size and all blocks are cached. Now we force to recompute
                # the last block. Note that we have to re-compute an entire
                # block because allocate_slots() assumes num_computed_tokens
                # is always a multiple of the block size. This limitation
                # can potentially be removed in the future to slightly
                # improve the performance.
                num_computed_tokens -= self.block_size
                num_new_tokens = self.block_size
                computed_blocks.pop()
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            # Schedule encoder inputs.
            (encoder_inputs_to_schedule, num_new_tokens,
            new_encoder_budget) = self._try_schedule_encoder_inputs(
                request, num_computed_tokens, num_new_tokens,
                encoder_budget)
            if num_new_tokens == 0:
                # The request cannot be scheduled.
                return None, token_budget, encoder_budget # type: ignore
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                return None, token_budget, encoder_budget # type: ignore
        elif request.status == RequestStatus.SWAPPED:
            num_computed_tokens = max(request.num_computed_tokens, num_computed_tokens)
            num_new_tokens = request.num_tokens - num_computed_tokens
            assert num_new_tokens > 0
            if num_new_tokens == 0:
                num_computed_tokens -= self.block_size
                num_new_tokens = self.block_size
                computed_blocks.pop()
            num_new_tokens = min(num_new_tokens, token_budget)
            # FIXME(julian): no encoder support
            encoder_inputs_to_schedule = None
            # swap in
            new_blocks = self.kv_cache_manager.swap_in(request, computed_blocks)
            if new_blocks is None:
                # The request cannot be scheduled.
                return None, token_budget, encoder_budget # type: ignore
        else:
            raise RuntimeError(f"Invalid request status: {request.status}")
        # self.running.append(request)

        if request.status == RequestStatus.WAITING:
            scheduled_new_reqs.append(request)
        elif request.status in [RequestStatus.PREEMPTED, RequestStatus.SWAPPED]:
            scheduled_resumed_reqs.append(request)
        else:
            raise RuntimeError(f"Invalid request status: {request.status}")

        req_to_new_block_ids[request.request_id] = [
            b.block_id for b in computed_blocks + new_blocks
        ]
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request.request_id] = (
                encoder_inputs_to_schedule)
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_budget = new_encoder_budget

        return num_new_tokens, token_budget, encoder_budget


    def schedule_early(
        self,
        token_budget: int,
        encoder_budget: int,
        preempted_reqs: List[Request],
        scheduled_new_reqs: List[Request],
        scheduled_resumed_reqs: List[Request],
        req_to_new_block_ids: Dict[str, List[int]],
        num_scheduled_tokens: Dict[str, int],
        scheduled_encoder_inputs: Dict[str, List[int]],
    ) -> Tuple[int, int]:  # token_budget, encoder_budget

        

        can_hold_all = self.kv_cache_manager.can_hold_all_requests(self.running, self.waiting)

        if can_hold_all:
            return token_budget, encoder_budget

        self.waiting = deque(sorted(self.waiting, key=lambda req: req.in_waiting_since)) # type: ignore

        running_sorted_with_idx = sorted(zip(self.running, range(len(self.running))),
                                            key=lambda req_idx: req_idx[0].in_running_since) # type: ignore
        

        reqs_idx_to_free = []
        
        kv_caches_cpu_usage = self.kv_cache_manager.get_kv_caches_usage()[1]
        swap_cond = (kv_caches_cpu_usage < 0.95
                     and self.kv_cache_manager.free_block_queue.num_free_blocks
                     < self.swapping_budget)
        if self.swapping_budget > 0 and swap_cond:
            block_count = 0
            for running_req, running_idx in running_sorted_with_idx:
                num_new_tokens = running_req.num_tokens - running_req.num_computed_tokens
                if num_new_tokens == 1:
                    num_blocks_to_free = self.kv_cache_manager.get_num_blocks_to_free(running_req)
                    if block_count + num_blocks_to_free >= self.swapping_budget:
                        break
                    block_count += num_blocks_to_free
                    reqs_idx_to_free.append(running_idx)

            if len(reqs_idx_to_free) == 0:
                return token_budget, encoder_budget



            




        # NOTE(mingtao): Same here.
        
        if reqs_idx_to_free is not None:
            reqs_to_free = [self.running[fidx] for fidx in reqs_idx_to_free]
        else:
            reqs_to_free = None

        # swap out
        if reqs_to_free:
            for req_to_free in reqs_to_free:
                self.kv_cache_manager.swap_out(req_to_free)
                req_to_free.status = RequestStatus.SWAPPED
                preempted_reqs.append(req_to_free)
                self.running.remove(req_to_free)
                # self.running.pop(fidx) # type: ignore
                self.waiting.append(req_to_free)
                self.time_record.append('WAITING | ' + str(round(time.time(), 2)) + ' | ' + req_to_free.request_id)


        return token_budget, encoder_budget


    def schedule(self, lookahead: bool = False) -> "SchedulerOutput":
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and num_tokens,
        # which is equal to len(prompt_token_ids) + len(output_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens. This is general enough to cover chunked prefills,
        # prefix caching, and the "jump decoding" optimization in the future.

        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: Dict[str, List[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens

        if self.proactive_swapping:
            token_budget, encoder_budget = self.schedule_early(
                token_budget,
                encoder_budget,
                preempted_reqs,
                scheduled_new_reqs,
                scheduled_resumed_reqs,
                req_to_new_block_ids,
                num_scheduled_tokens,
                scheduled_encoder_inputs,
            )
            # if scheduled_new_reqs
            if preempted_reqs:
                reqs_id_preempted_early = [req.request_id for req in preempted_reqs]
                # req_id_preempted_early = preempted_reqs[0].request_id
            else:
                reqs_id_preempted_early = None
            # NOTE(mingtao): Select the mem2hbm request. New requests
            # are preferred.

            # if scheduled_new_reqs:
            #     req_id_scheduled_early = scheduled_new_reqs[0].request_id
            # elif scheduled_resumed_reqs:
            #     req_id_scheduled_early = scheduled_resumed_reqs[0].request_id
            # else:
            #     req_id_scheduled_early = None
            
            reqs_id_scheduled_early = []
            if scheduled_new_reqs:
                for req in scheduled_new_reqs:
                    assert req.status == RequestStatus.RUNNING, req.status
                    reqs_id_scheduled_early.append(req.request_id)

            if scheduled_resumed_reqs:
                for req in scheduled_resumed_reqs:
                    assert req.status == RequestStatus.RUNNING, req.status
                    reqs_id_scheduled_early.append(req.request_id)
            if len(reqs_id_scheduled_early) == 0:
                reqs_id_scheduled_early = None
        else:
            reqs_id_preempted_early = None
            reqs_id_scheduled_early = None

        # First, schedule the RUNNING requests.
        # NOTE(julian): chunked prefill works as partial prefill
        # NOTE(woosuk): At most 1 request in the RUNNING queue is allowed to be
        # in the "partial" state, where the request has some tokens computed
        # but not all. The constraint is due to the persistent batch in the
        # V1 model runner.
        # TODO(woosuk): Remove this constraint after refactoring model runner.
        has_partial_request = False
        req_index = 0

        while req_index < len(self.running):
            # Only the last request in the RUNNING queue can be "partial".
            assert not has_partial_request
            assert token_budget > 0

            # NOTE(mingtao): Find request from tail.

            request = self.running[req_index]
            
            # if (req_id_scheduled_early is not None) and request.request_id == req_id_scheduled_early:
            if (reqs_id_scheduled_early is not None) and len(reqs_id_scheduled_early) > 0 and request.request_id in reqs_id_scheduled_early:
                req_index += 1
                continue
            # if request.status != RequestStatus.RUNNING:
            #     req_index += 1
            #     continue
            # assert request.status == RequestStatus.RUNNING, request.status
            with RequestLookaheadContextManager(request, lookahead):
                num_new_tokens, token_budget, encoder_budget = self.schedule_single_running_request(
                    request,
                    token_budget,
                    encoder_budget,
                    preempted_reqs,
                    scheduled_running_reqs,
                    req_to_new_block_ids,
                    num_scheduled_tokens,
                    scheduled_encoder_inputs,
                )
            if not num_new_tokens:
                break
            req_index += 1
            # NOTE(julian): has partial means restricted by budget
            if lookahead:
                has_partial_request = (request.num_computed_tokens_next + num_new_tokens
                                    < request.num_tokens_next)
            else:
                has_partial_request = (request.num_computed_tokens + num_new_tokens
                                    < request.num_tokens)

        kv_caches_cpu_usage = self.kv_cache_manager.get_kv_caches_usage()[1]

        # Next, schedule the WAITING requests.
        if (not preempted_reqs) or reqs_id_preempted_early is not None:
            excluded_waiting = []
            while self.waiting:
                # NOTE(julian): it means all tokens budget are used by running requests
                if has_partial_request:
                    break
                if len(self.running) == self.max_num_running_reqs:
                    break
                if token_budget == 0:
                    break
                request = self.waiting[0]

                if kv_caches_cpu_usage > 0.9 and request.status == RequestStatus.WAITING:
                    excluded_waiting.append(request)
                    self.waiting.popleft()
                    continue  # prevent cpu kv caches overflow

                num_new_tokens, token_budget, encoder_budget = self.schedule_single_waiting_request(
                    request,
                    token_budget,
                    encoder_budget,
                    scheduled_new_reqs,
                    scheduled_resumed_reqs,
                    req_to_new_block_ids,
                    num_scheduled_tokens,
                    scheduled_encoder_inputs,
                )
                if not num_new_tokens:
                    break
                self.waiting.popleft()
                self.running.append(request)
                self.time_record.append('RUNNING | ' + str(round(time.time(), 2)) + ' | ' + request.request_id)
                has_partial_request = (request.num_computed_tokens + num_new_tokens
                                < request.num_tokens)
            self.waiting.extend(excluded_waiting)
            curr_t = str(round(time.time(), 2))
            for req in excluded_waiting:
                self.time_record.append('WAITING | ' + curr_t + ' | ' + req.request_id)


        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens, (total_num_scheduled_tokens, self.max_num_scheduled_tokens)
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) == len(self.running)), (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs), len(self.running))

        # pending swapping blocks
        blocks_to_swap_in = self.kv_cache_manager.get_and_reset_pending_blocks_to_swap_in()
        blocks_to_swap_out = self.kv_cache_manager.get_and_reset_pending_blocks_to_swap_out()

        # update time
        now = time.time()
        for req in self.running:
            if req.in_running_since is None:
                req.in_running_since = now
                req.in_waiting_since = None
        for req in self.waiting:
            if req.in_waiting_since is None:
                req.in_running_since = None
                req.in_waiting_since = now

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_block_ids[req.request_id],
                req.num_computed_tokens)
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            ResumedRequestData.from_request(
                req, req_to_new_block_ids[req.request_id],
                req.num_computed_tokens)
            for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_running_request_data(
                req, req_to_new_block_ids[req.request_id],
                req.num_computed_tokens_next if lookahead else req.num_computed_tokens)
            for req in scheduled_running_reqs
        ]
        preempted_req_ids = {req.request_id for req in preempted_reqs}

        # for lookahead
        for req in scheduled_running_reqs:
            st = num_scheduled_tokens[req.request_id]
            if lookahead:
                assert req.num_computed_tokens_next > 0
                assert req.num_tokens_next > 0
                assert req.num_computed_tokens_next + st <= req.num_tokens_next
                is_partial_request = req.num_computed_tokens_next + st < req.num_tokens_next
                if not is_partial_request:
                    req.num_tokens_next += 1
                req.num_computed_tokens_next += st
            else:
                assert req.num_computed_tokens > 0
                assert req.num_tokens > 0
                assert req.num_computed_tokens + st <= req.num_tokens
                is_partial_request = req.num_computed_tokens + st < req.num_tokens
                if is_partial_request:
                    req.num_tokens_next = req.num_tokens
                else:
                    req.num_tokens_next = req.num_tokens + 1
                req.num_computed_tokens_next = req.num_computed_tokens + st

        for req in chain(scheduled_new_reqs, scheduled_resumed_reqs):
            st = num_scheduled_tokens[req.request_id]
            assert req.num_computed_tokens + st <= req.num_tokens
            is_partial_request = req.num_computed_tokens + st < req.num_tokens
            if is_partial_request:
                req.num_tokens_next = req.num_tokens
            else:
                req.num_tokens_next = req.num_tokens + 1
            req.num_computed_tokens_next = req.num_computed_tokens + st

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_resumed_reqs=resumed_reqs_data,
            scheduled_running_reqs=running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            preempted_req_ids=preempted_req_ids,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            counter=self.schedule_counter,
        )

        self.schedule_counter += 1

        self.finished_req_ids = set()
        return scheduler_output
    
    def _make_running_request_data(
        self,
        request: Request,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "RunningRequestData":
        # OPTIMIZATION: Cache the RunningRequestData objects to avoid creating
        # them at each scheduling step.
        if request.request_id in self.running_reqs_data:
            req_data = self.running_reqs_data[request.request_id]
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = RunningRequestData.from_request(request, new_block_ids,
                                                       num_computed_tokens)
            self.running_reqs_data[request.request_id] = req_data
        # TODO(jiahuan): duplicate it to prevent data race
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> Tuple[List[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.
        """
        if not request.has_encoder_inputs():
            return [], num_new_tokens, encoder_budget

        encoder_inputs_to_schedule: List[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info["offset"]
            num_encoder_tokens = pos_info["length"]

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue
            if not self.encoder_cache_manager.can_allocate(request, i):
                # The encoder cache is full. We can only schedule the decoder
                # tokens just before the encoder input.
                num_new_tokens = start_pos - num_computed_tokens
                break
            if num_encoder_tokens > encoder_budget:
                # The encoder budget is exhausted. We can only schedule the
                # decoder tokens up until the encoder input.
                # NOTE(woosuk): We assume that the encoder tokens should be
                # processed altogether, as the encoder usually uses
                # bidirectional attention.
                num_new_tokens = start_pos - num_computed_tokens
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    @nvtx.annotate("reschedule", color="blue")
    def reschedule(
        self,
        scheduler_output: "SchedulerOutput",
        stopped_ids: Set[str],
    ) -> "SchedulerOutput":
        scheduled_new_reqs = scheduler_output.scheduled_new_reqs
        scheduled_resumed_reqs = scheduler_output.scheduled_resumed_reqs
        scheduled_running_reqs = scheduler_output.scheduled_running_reqs
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        preempted_req_ids = scheduler_output.preempted_req_ids
        blocks_to_swap_in = scheduler_output.blocks_to_swap_in
        blocks_to_swap_out = scheduler_output.blocks_to_swap_out
        finished_req_ids = scheduler_output.finished_req_ids
        free_encoder_input_ids = scheduler_output.free_encoder_input_ids

        # remove stopped requests
        scheduled_new_reqs = [req for req in scheduled_new_reqs if req.req_id not in stopped_ids]
        scheduled_resumed_reqs = [req for req in scheduled_resumed_reqs if req.req_id not in stopped_ids]
        scheduled_running_reqs = [req for req in scheduled_running_reqs if req.req_id not in stopped_ids]
        for req_id in stopped_ids:
            if req_id in num_scheduled_tokens:
                num_scheduled_tokens.pop(req_id)
        preempted_req_ids -= stopped_ids
        finished_req_ids.update(stopped_ids)
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        scheduled_encoder_inputs = {req_id: scheduled_encoder_inputs[req_id] for req_id in scheduled_encoder_inputs if req_id not in stopped_ids}
        free_encoder_input_ids = [x for x in free_encoder_input_ids if x[0] not in stopped_ids]

        return SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_resumed_reqs=scheduled_resumed_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            preempted_req_ids=preempted_req_ids,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            finished_req_ids=finished_req_ids,
            free_encoder_input_ids=free_encoder_input_ids,
            counter=scheduler_output.counter,
        )

    @nvtx.annotate("update_from_output", color="blue")
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
        lookahead: bool = False,
    ) -> Tuple[List[EngineCoreOutput], Set[str]]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        engine_core_outputs: List[EngineCoreOutput] = []

        # running_req = [req.req_id for req in scheduler_output.scheduled_running_reqs]
        # waiting_req = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        # resumed_req = [req.req_id for req in scheduler_output.scheduled_resumed_reqs]
        stopped_ids = set()
        bad_ids = model_runner_output.bad_ids
        for req_id in scheduler_output.num_scheduled_tokens.keys():
            request = self.requests[req_id]
            if req_id in bad_ids:
                # request.num_computed_tokens += num_scheduled_tokens[req_id]
                continue
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            assert request.num_computed_tokens <= request.num_tokens
            # FIXME(julian): no encoder support
            if request.num_computed_tokens == request.num_tokens:
                req_index = model_runner_output.req_id_to_index[req_id]
                token_id = sampled_token_ids[req_index]
                request.append_output_token_ids(token_id)
                num_new_tokens = 1
                stopped = self._check_stop(request)
                output = EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=request.output_token_ids[-num_new_tokens:],
                    finished=request.is_finished(),
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason)
                engine_core_outputs.append(output)
                if stopped:
                    stopped_ids.add(req_id)
   

        self.running = [req for req in self.running if req.request_id not in stopped_ids]
        self.waiting = deque([req for req in self.waiting if req.request_id not in stopped_ids])
        curr_t = str(round(time.time(), 2))
        for req_id in stopped_ids:
            self.time_record.append('STOPPED | ' + curr_t + ' | ' + req_id)
        return engine_core_outputs, stopped_ids
        
    def _check_stop(self, request: Request) -> bool:
        if (request.num_tokens >= self.max_model_len
                or request.num_output_tokens >= request.max_tokens):
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            self._free_request(request)
            return True

        sampling_params = request.sampling_params
        last_token_id = request.output_token_ids[-1]
        if (not sampling_params.ignore_eos
                and last_token_id == request.eos_token_id):
            request.status = RequestStatus.FINISHED_STOPPED
            self._free_request(request)
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            self._free_request(request)
            return True
        return False

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.time_record.append('ADDED | ' + str(round(time.time(), 2)) + ' | ' + request.request_id)
        self.requests[request.request_id] = request

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            self.time_record.append('FINISHED | ' + str(round(time.time(), 2)) + ' | ' + request.request_id)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.running_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def get_running_len(self) -> int:
        return len(self.running)

    def get_total_waiting(self) -> int:
        return len(self.waiting)
    def get_waiting_len(self) -> int:
        return sum(r.status != RequestStatus.SWAPPED for r in self.waiting)
    def get_swapped_len(self) -> int:
        return sum(r.status == RequestStatus.SWAPPED for r in self.waiting)
    def get_time_record(self):
        time_record = self.time_record
        self.time_record = []
        return time_record


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List["MultiModalKwargs"]
    mm_hashes: List[str]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt=request.prompt,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class ResumedRequestData:

    req_id: str
    block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "ResumedRequestData":
        return cls(
            req_id=request.request_id,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class RunningRequestData:

    req_id: str
    new_block_ids: List[int]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "RunningRequestData":
        return cls(
            req_id=request.request_id,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    scheduled_new_reqs: List[NewRequestData]
    scheduled_resumed_reqs: List[ResumedRequestData]
    scheduled_running_reqs: List[RunningRequestData]

    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_encoder_inputs: Dict[str, List[int]]

    preempted_req_ids: Set[str]
    finished_req_ids: Set[str]
    free_encoder_input_ids: List[Tuple[str, int]]

    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]

    counter: int