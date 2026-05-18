from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock, KVCacheBlockStatus,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens,
                                         hash_request_tokens)
from vllm.v1.request import Request

logger = init_logger(__name__)


class KVCacheManager:
    """SuperInfer KV-cache manager with paired GPU + CPU block pools."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        max_model_len: int,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
        prefix_cache_fix: bool = True,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = cdiv(max_model_len, block_size)
        self.sliding_window = sliding_window
        self.enable_caching = enable_caching
        # Workaround for a prefix-caching invalidation bug — see SuperInfer
        # paper §A; toggle off via ``--no-prefix-cache-fix`` for ablation.
        self.fix_wrong_prefix_caching = prefix_cache_fix
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        self.num_preallocate_tokens = num_preallocate_tokens
        self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)

        # A Block pool of all kv-cache blocks.
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        self.block_pool_cpu: List[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_cpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)
        self.free_block_queue_cpu = FreeKVCacheBlockQueue(self.block_pool_cpu)

        for block in self.block_pool:
            block.mark_free()
            # NOTE(julian): permanently mapped CPU block for each GPU block
            # this will simplify the swapping management
            block.mapped_cpu_block = self.free_block_queue_cpu.popleft()

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: Dict[BlockHashType, Dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: Dict[str, List[KVCacheBlock]] = {}
        self.req_to_blocks_cpu: Dict[str, List[KVCacheBlock]] = {}

        # NOTE(julian): pending blocks to be swapped
        self.pending_blocks_to_swap_out: List[Tuple[int, int]] = []
        self.pending_blocks_to_swap_in: List[Tuple[int, int]] = []

    def get_and_reset_pending_blocks_to_swap_in(self) -> List[Tuple[int, int]]:
        ret = self.pending_blocks_to_swap_in
        self.pending_blocks_to_swap_in = []
        return ret

    def get_and_reset_pending_blocks_to_swap_out(self) -> List[Tuple[int, int]]:
        ret = self.pending_blocks_to_swap_out
        self.pending_blocks_to_swap_out = []
        return ret

    # NOTE(julian): find prefix caching only for new requests
    def get_computed_blocks(self, request: Request) -> List[KVCacheBlock]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A list of blocks that are computed for the request.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return []

        computed_blocks = []

        # The block hashes for the request may already be computed
        # if the request was preempted and resumed.
        if not request.kv_block_hashes:
            request.set_kv_block_hashes(
                hash_request_tokens(self.block_size, request))
        block_hashes = request.kv_block_hashes

        for block_hash in block_hashes:
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self._get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break

        return computed_blocks

    def get_append_slots_num_new_blocks(self, request: Request, num_tokens: int) -> int:
        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens, self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]
        num_new_blocks = num_required_blocks - len(req_blocks)
        return max(num_new_blocks, 0)

    def append_slots(
        self,
        request: Request,
        num_tokens: int,
    ) -> Optional[List[KVCacheBlock]]:
        """Append slots to the block table of the request.
        We first append slots to already allocated blocks. If the allocated
        blocks are not enough, we allocate new blocks.

        Args:
            request: The request to append slots.
            num_tokens: The number of tokens to append.

        Returns:
            A list of new blocks if new blocks are allocated, or None
            if new blocks are required but cannot be allocated.
        """
        num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                   self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]

        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks > self.free_block_queue.num_free_blocks:
            # Need to allocate new blocks due to insufficient pre-allocated
            # slots, but we cannot allocate new blocks due to the limit.
            return None

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.free_block_queue.num_free_blocks,
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                # TODO(woosuk): Check and reject requests if
                # num_prompt_tokens + max_tokens > max_model_len.
                self.max_num_blocks_per_req - len(req_blocks),
            )
            assert num_new_blocks > 0

            new_blocks = self._get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        if self.fix_wrong_prefix_caching:

            num_cached_full_blocks = request.num_cached_tokens // self.block_size
            num_computed_full_blocks = request.num_computed_tokens // self.block_size

            new_full_blocks = req_blocks[num_cached_full_blocks:num_computed_full_blocks]

            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=num_cached_full_blocks,
                    full_blocks=new_full_blocks,
                    prev_block=req_blocks[num_cached_full_blocks - 1]
                    if num_cached_full_blocks >= 1 else None,
                )
                request.num_cached_tokens = num_computed_full_blocks * self.block_size

            for block in new_full_blocks:
                block.mark_full(clean=True)
                self.pending_blocks_to_swap_out.append((block.block_id, block.mapped_cpu_block.block_id))
            for block in req_blocks[num_computed_full_blocks:num_required_blocks]:
                block.mark_half(dirty=True)

        else:

            num_computed_full_blocks = (request.num_computed_tokens //
                                        self.block_size)

            # NOTE(rickyx): We are assuming the `num_tokens` are actual
            # tokens rather than lookahead slots (e.g. for speculative decoding).
            # TODO(rickyx): When supporting speculative decoding, we will need to
            # differentiate between them so that we can know how many blocks are
            # full after appending the actual tokens.
            num_full_blocks_after_append = (request.num_computed_tokens +
                                            num_tokens) // self.block_size
            assert num_full_blocks_after_append <= len(req_blocks)

            new_full_blocks = req_blocks[
                num_computed_full_blocks:num_full_blocks_after_append]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=num_computed_full_blocks,
                    full_blocks=new_full_blocks,
                    prev_block=req_blocks[num_computed_full_blocks - 1]
                    if num_computed_full_blocks >= 1 else None,
                )

        return new_blocks

    def get_allocated_slots_num_new_blocks(self, request: Request, num_tokens: int, computed_blocks: List[KVCacheBlock]) -> int:
        if num_tokens == 0:
            return 0
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = min(
            num_required_blocks,
            self.max_num_blocks_per_req - len(computed_blocks),
        )
        return num_new_blocks

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        computed_blocks: List[KVCacheBlock],
        no_preallocate: bool = False,
    ) -> Optional[List[KVCacheBlock]]:
        """Allocate slots for a new request.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            computed_blocks: The blocks that have already been computed.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError(
                f"num_tokens must be greater than 0, got {num_tokens}")

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self._touch(computed_blocks)
        else:
            assert not computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        num_required_blocks = cdiv(num_tokens, self.block_size)
        nb = self.free_block_queue.num_free_blocks
        if (num_required_blocks > nb):
            # Cannot allocate new blocks.
            return None

        # Determine the number of new blocks to allocate considering
        # preallocated blocks.
        num_new_blocks = min(
            num_required_blocks + (0 if no_preallocate else self.num_preallocate_blocks),
            self.free_block_queue.num_free_blocks,
            # Should not exceed the maximum number of blocks per request.
            # This is especially because the block table has the shape
            # [..., max_num_blocks_per_req].
            # TODO(woosuk): Check and reject requests if
            # num_prompt_tokens + max_tokens > max_model_len.
            self.max_num_blocks_per_req - len(computed_blocks),
        )
        assert num_new_blocks > 0

        # Concatenate the computed block IDs and the new block IDs.
        new_blocks = self._get_new_blocks(num_new_blocks)
        self.req_to_blocks[request.request_id] = computed_blocks + new_blocks

        if not self.enable_caching:
            return new_blocks

        if self.fix_wrong_prefix_caching:

            request.num_cached_tokens = len(computed_blocks) * self.block_size

            for i in range(num_required_blocks):
                new_blocks[i].mark_half(dirty=True)

        else:

            num_computed_tokens = len(computed_blocks) * self.block_size
            num_full_blocks = (num_computed_tokens + num_tokens) // self.block_size

            new_full_blocks = self.req_to_blocks[
                request.request_id][len(computed_blocks):num_full_blocks]
            if new_full_blocks:
                self._cache_full_blocks(
                    request=request,
                    blk_start_idx=len(computed_blocks),
                    # The new full blocks are the full blocks that are not computed.
                    full_blocks=new_full_blocks,
                    prev_block=computed_blocks[-1] if computed_blocks else None,
                )

        return new_blocks

    def free(self, request: Request, keep_cpu: bool = False, swapped: bool = False) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_blocks.pop(request.request_id, [])
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            block.decr_ref(keep_cpu=keep_cpu)
            if block.ref_cnt == 0:
                block.mark_free()
                self.free_block_queue.append(block)
            elif swapped:
                block.mark_clean()

    def get_swap_in_num_new_blocks(self, request: Request, computed_blocks: List[KVCacheBlock]) -> int:
        cpu_blocks = self.req_to_blocks_cpu[request.request_id]
        num_new_tokens = (len(cpu_blocks) - len(computed_blocks)) * self.block_size
        num_new_blocks = self.get_allocated_slots_num_new_blocks(request, num_new_tokens, computed_blocks)
        return num_new_blocks

    def swap_in(self, request: Request, computed_blocks: List[KVCacheBlock]) -> Optional[List[KVCacheBlock]]:
        cpu_blocks = self.req_to_blocks_cpu[request.request_id]
        num_new_tokens = (len(cpu_blocks) - len(computed_blocks)) * self.block_size
        new_blocks = self.allocate_slots(request, num_new_tokens, computed_blocks, no_preallocate=True)

        if new_blocks is None:
            return None

        # NOTE(julian): free useless swapping blocks
        for cpu_block in cpu_blocks[:len(computed_blocks)]:
            cpu_block.decr_ref()
            if cpu_block.ref_cnt == 0:
                self.free_block_queue_cpu.append(cpu_block)

        # NOTE(julian): relocate the mapping
        for gpu_block, cpu_block in zip(new_blocks, cpu_blocks[len(computed_blocks):]):
            assert gpu_block.mapped_cpu_block is not None
            gpu_block.mapped_cpu_block.decr_ref()
            if gpu_block.mapped_cpu_block.ref_cnt == 0:
                self.free_block_queue_cpu.append(gpu_block.mapped_cpu_block)
            gpu_block.mapped_cpu_block = cpu_block

        # NOTE(julian): restore block status
        for i, block in enumerate(new_blocks):
            blk_idx = len(computed_blocks) + i
            if blk_idx * self.block_size + 1 <= request.num_computed_tokens:
                if (blk_idx + 1) * self.block_size <= request.num_computed_tokens:
                    block.mark_full(clean=True)
                else:
                    block.mark_half(dirty=True)
                self.pending_blocks_to_swap_in.append((block.mapped_cpu_block.block_id, block.block_id))
            else:
                block.mark_empty()

        self.req_to_blocks_cpu.pop(request.request_id)

        return new_blocks

    def swap_out(self, request: Request) -> None:
        start = self.free_block_queue.num_free_blocks
        gpu_blocks = self.req_to_blocks[request.request_id]
        cpu_blocks = [b.mapped_cpu_block for b in gpu_blocks]

        # NOTE(julian): get blocks to swap out
        self.pending_blocks_to_swap_out.extend([(b.block_id, b.mapped_cpu_block.block_id) \
            for b in gpu_blocks if b.is_dirty()])
        # NOTE(julian): free gpu blocks
        self.free(request, keep_cpu=True, swapped=True)

        self.req_to_blocks_cpu[request.request_id] = cpu_blocks # type: ignore
        return self.free_block_queue.num_free_blocks - start
    
    def _get_new_blocks(self, num_blocks: int) -> List[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: List[KVCacheBlock] = []
        idx = 0
        while idx < num_blocks:
            # First allocate blocks.
            curr_block = self.free_block_queue.popleft()
            curr_block.mark_empty()
            assert curr_block.ref_cnt == 0

            # NOTE(julian): COW: if the mapped cpu block is refed, allocated a new one
            assert curr_block.mapped_cpu_block is not None
            if curr_block.mapped_cpu_block.ref_cnt > 0:
                curr_block.mapped_cpu_block = self.free_block_queue_cpu.popleft()

            # If the block is cached, evict it.
            if self.enable_caching:
                self._evict_cached_block(curr_block)

            curr_block.incr_ref()
            ret.append(curr_block)
            idx += 1

        return ret

    def _evict_cached_block(self, block: KVCacheBlock) -> None:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.
        """
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

    def _get_cached_block(self,
                          block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached block by the block hash, or None if cache miss.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached block if it exists, or None.
        """
        if block_hash in self.cached_block_hash_to_block:
            first_block_id = list(
                self.cached_block_hash_to_block[block_hash].keys())[0]
            return self.cached_block_hash_to_block[block_hash][first_block_id]
        return None

    def _touch(self, blocks: List[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
                block.mark_full(clean=True)
            block.incr_ref()

    def _cache_full_blocks(
        self,
        request: Request,
        blk_start_idx: int,
        full_blocks: List[KVCacheBlock],
        prev_block: Optional[KVCacheBlock],
    ) -> None:
        """Cache a list of full blocks for prefix caching.

        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it computes the
        block hashes for the blocks starting from `blk_start_idx` to the end
        of the request's full blocks, updating the metadata for each block
        and caching them in the `cached_block_hash_to_block`.

        Args:
            request: The request to cache the blocks.
            blk_start_idx: The index of the first block in the request's blocks
                to cache.
            full_blocks: The list of blocks to update hash metadata.
            prev_block: The previous block in the chain.
        """
        num_cached_block_hashes = len(request.kv_block_hashes)

        # Update the new blocks with the block hashes through the chain.
        prev_block_hash_value = None
        if prev_block is not None:
            # Previous block must have a block hash because it must be
            # a full, cached block.
            assert prev_block.block_hash is not None
            prev_block_hash_value = prev_block.block_hash.hash_value

        for i, blk in enumerate(full_blocks):
            blk_idx = blk_start_idx + i

            if blk_idx < num_cached_block_hashes:
                # The block hash may already be computed in
                # "get_computed_blocks" if the tokens are not generated by
                # this request (either the prompt tokens or the previously
                # generated tokens with preemption). In this case we simply
                # reuse the block hash.
                block_hash = request.kv_block_hashes[blk_idx]
            else:
                # Otherwise compute the block hash and cache it in the request
                # in case it will be preempted in the future.
                start_token_idx = blk_idx * self.block_size
                end_token_idx = (blk_idx + 1) * self.block_size
                block_tokens = request.all_token_ids[
                    start_token_idx:end_token_idx]
                assert len(block_tokens) == self.block_size, (
                    f"Expected {self.block_size} tokens, got "
                    f"{len(block_tokens)} at {blk_idx}th block for request "
                    f"{request.request_id}({request})")

                # Generate extra keys for multi-modal inputs. Note that since
                # we reach to this branch only when the block is completed with
                # generated tokens, we only need to consider the last mm input.
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, -1)

                # Compute the hash of the current block.
                block_hash = hash_block_tokens(prev_block_hash_value,
                                               block_tokens, extra_keys)
                request.append_kv_block_hashes(block_hash)

            # Update and added the full block to the cache.
            blk.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][blk.block_id] = blk
            prev_block_hash_value = block_hash.hash_value

    def can_hold_all_requests(self, running, waiting) -> bool:
        num_free_blocks = self.free_block_queue.num_free_blocks
        for request in running:
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_blocks = self.get_append_slots_num_new_blocks(request, num_new_tokens)
            num_free_blocks -= num_new_blocks
            if num_free_blocks < 0:
                return False
        for request in waiting:
            computed_blocks = self.get_computed_blocks(request)
            num_new_tokens = request.num_tokens - len(computed_blocks) * self.block_size
            num_new_blocks = self.get_allocated_slots_num_new_blocks(request, num_new_tokens, computed_blocks)
            num_free_blocks -= num_new_blocks
            if num_free_blocks < 0:
                return False
        return True

    def get_num_blocks_to_free(self, request: Request):
        blocks = self.req_to_blocks[request.request_id]
        num = sum(b.ref_cnt == 1 for b in blocks)
        return num

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def get_kv_caches_usage(self) -> Tuple[float, float]:
        gpu_usage = 1 - self.free_block_queue.num_free_blocks / len(self.block_pool)
        cpu_usage = 1 - self.free_block_queue_cpu.num_free_blocks / len(self.block_pool_cpu)
        return gpu_usage, cpu_usage

