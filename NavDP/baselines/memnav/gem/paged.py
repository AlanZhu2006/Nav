# GEM adapter for LingBot-Map (Apache-2.0): BF16 paged working memory.
"""Explicit FA2 backend for the native W64 writer, with demand-grown storage.

The original frame attention, camera/depth heads, parameters and readouts stay
in place. Only global attention and its working-state allocation change. The
eight-view initialization still uses the original joint SDPA computation.
Subsequent FA2 outputs drive the model; numerical equivalence is not assumed.
"""
import hashlib
import inspect
import math
from collections import deque
from pathlib import Path

import torch

from lingbot_map.aggregator.stream import AggregatorStream
from lingbot_map.layers.attention import FlashInferAttention, SDPAAttention
from lingbot_map.layers.block import SDPABlock
from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager


NATIVE_SOURCES = {
    'aggregator/stream.py': '6a19e43cb3ddaa569a56d2b9c6675d1693be0d8ad381803a354de32f75aa0a5f',
    'layers/attention.py': '59d28bdbb80c7d9138ce5b77fffdc636beee7081767c774a5c6ca7a675925419',
    'layers/block.py': '90085ca55a8fdbbcda1529448f8ef4c551f7b3006631923856374d30b098f741',
    'layers/flashinfer_cache.py': '7b2d6985e21910ed9a29f0679d6a0d3cc67b89742f762496e687af0cbb4f41dc',
}


class PagedWorkingMemory(FlashInferKVCacheManager):
    """Reuse native page ordering/eviction, without its 123-page reservation.

    Patch and special allocations share one free list per layer. A layer's
    pool grows in 16-page increments, up to 8+64+1 patch and two special pages
    at the native 320-commit limit. Only one layer is copied during growth.
    No historical KV gather or clone occurs in the attention path.
    """
    def __init__(self, *, num_blocks, tokens_per_frame, num_heads, head_dim, device,
                 scale_frames=8, sliding_window=64, max_total_frames=320):
        import flashinfer

        self.num_blocks = int(num_blocks)
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_special_tokens = 6
        self.patches_per_frame = self.tokens_per_frame - self.num_special_tokens
        if self.patches_per_frame < self.num_special_tokens:
            raise ValueError('A patch page must hold at least the six special tokens')
        self.page_size = self.patches_per_frame  # FA2: exact size, no padded keys.
        self.scale_frames, self.sliding_window = scale_frames, sliding_window
        self.max_total_frames = max_total_frames
        self.max_patch_pages = scale_frames + sliding_window + 1
        self.max_num_pages = self.max_patch_pages + math.ceil(
            max_total_frames * self.num_special_tokens / self.page_size)
        self.num_heads, self.head_dim = num_heads, head_dim
        self.device, self.dtype = torch.device(device), torch.bfloat16
        self.force_fp32 = False
        self.workspace_buffer = torch.empty(128 * 1024 * 1024,
            dtype=torch.uint8, device=self.device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout='NHD', backend='fa2')
        self._qo_indptr = torch.tensor([0, tokens_per_frame],
            dtype=torch.int32, device=self.device)
        self.reset()

    def reset(self):
        # Drop buffers as well as logical occupancy: episode reset must free
        # the long-history pool. The CUDA allocator may retain reserved bytes.
        self.kv_caches = [None] * self.num_blocks
        self.scale_patch_pages = [deque() for _ in range(self.num_blocks)]
        self.live_window_patch_pages = [deque() for _ in range(self.num_blocks)]
        self.all_special_pages = [[] for _ in range(self.num_blocks)]
        self.free_patch_pages = [[] for _ in range(self.num_blocks)]
        self.free_special_pages = self.free_patch_pages  # one physical pool
        self.special_token_count = [0] * self.num_blocks
        self.frame_count = [0] * self.num_blocks
        self._skip_append = self._defer_eviction = False
        self.failure = None
        self.growth_count = self.growth_copied_bytes = 0

    def _reserve_page(self, block_idx):
        if self.free_patch_pages[block_idx]:
            return
        old = self.kv_caches[block_idx]
        previous = 0 if old is None else len(old)
        capacity = min(previous + 16, self.max_num_pages)
        if capacity <= previous:
            raise RuntimeError('Paged working memory exhausted its native capacity')
        grown = torch.empty((capacity, 2, self.page_size, self.num_heads, self.head_dim),
            dtype=self.dtype, device=self.device)
        if old is not None:
            grown[:previous].copy_(old)
            self.growth_copied_bytes += old.numel() * old.element_size()
        self.kv_caches[block_idx] = grown
        self.free_patch_pages[block_idx].extend(range(previous, capacity))
        self.growth_count += 1

    def _write_patch_page(self, block_idx, patch_k, patch_v):
        self._reserve_page(block_idx)
        return super()._write_patch_page(block_idx, patch_k, patch_v)

    def _write_special_tokens(self, block_idx, sp_k, sp_v):
        offset = self.special_token_count[block_idx] % self.page_size
        if offset == 0 or offset + self.num_special_tokens > self.page_size:
            self._reserve_page(block_idx)
        return super()._write_special_tokens(block_idx, sp_k, sp_v)

    def append_frame(self, block_idx, k, v):
        if self.failure is not None:
            raise RuntimeError('Paged working memory failed; reset is required: ' + self.failure)
        if self.frame_count[block_idx] >= self.max_total_frames:
            raise RuntimeError('Native committed-view limit reached')
        try:
            super().append_frame(block_idx, k, v)
        except BaseException as error:
            self.failure = f'{type(error).__name__}: {error}'
            raise

    def attend(self, block_idx, q, k, v):
        """Current KV is temporary on noncommit reads, even if a kernel fails."""
        temporary = self._skip_append
        appended = False
        try:
            self.append_frame(block_idx, k, v)
            appended = True
            if not temporary:
                self.evict_frames(block_idx, self.scale_frames, self.sliding_window)
            return self.compute_attention(block_idx, q)
        except BaseException as error:
            self.failure = f'{type(error).__name__}: {error}'
            raise
        finally:
            if temporary and appended:
                self.rollback_last_frame(block_idx)

    def statistics(self):
        scalar_bytes = torch.empty((), dtype=self.dtype).element_size()
        token_bytes = 2 * self.num_heads * self.head_dim * scalar_bytes
        effective = sum((len(self.scale_patch_pages[i]) +
            len(self.live_window_patch_pages[i])) * self.page_size +
            self.special_token_count[i] for i in range(self.num_blocks)) * token_bytes
        return dict(backend='flashinfer_fa2', dtype='bfloat16',
            committed_views=self.num_frames, effective_kv_bytes=effective,
            allocated_pool_bytes=sum(t.numel() * t.element_size()
                for t in self.kv_caches if t is not None),
            allocated_pages_per_layer=[0 if t is None else len(t) for t in self.kv_caches],
            workspace_bytes=self.workspace_buffer.numel(),
            growth_count=self.growth_count, growth_copied_bytes=self.growth_copied_bytes,
            max_pages_per_layer=self.max_num_pages, failure=self.failure)


class PagedAttention(FlashInferAttention):
    """Keep SDPABlock's residual computation and use explicit paged reads."""
    def forward(self, x, pos=None, num_patches=None, num_special=None, num_frames=None,
                enable_3d_rope=False, kv_cache=None, global_idx=0,
                num_frame_per_block=1, num_frame_for_scale=-1, num_register_tokens=4):
        if not isinstance(kv_cache, PagedWorkingMemory) or x.shape[0] != 1 or self.training:
            raise ValueError('GEM paged attention requires one evaluation stream')
        manager = kv_cache
        count = int(num_frames or 1)
        if count > 1:
            if count != 8 or manager.frame_count[global_idx] != 0 or manager._skip_append:
                raise ValueError('Joint attention is restricted to the original initial eight views')
            # Reuse the exact native joint-initialization implementation.
            return super().forward(x, pos=pos, num_patches=num_patches,
                num_special=num_special, num_frames=num_frames, enable_3d_rope=enable_3d_rope,
                kv_cache=manager, global_idx=global_idx, num_frame_per_block=num_frame_per_block,
                num_frame_for_scale=num_frame_for_scale, num_register_tokens=num_register_tokens)
        q, k, v = self.prepare_qkv(x, pos=pos, enable_3d_rope=enable_3d_rope)
        out = manager.attend(global_idx, q, k, v)
        out = out.reshape(1, x.shape[1], self.num_heads * self.head_dim)
        return self.proj_drop(self.proj(out))


class PagedAggregator(AggregatorStream):
    gem_paged_memory = True

    def _get_flashinfer_manager(self, device, dtype, tokens_per_frame=None):
        # Residual tokens can remain FP32. Q/K/V are projected and consumed
        # under BF16 autocast; pool precision does not follow residual dtype.
        if (torch.device(device).type != 'cuda' or dtype not in (torch.float32, torch.bfloat16)
                or not torch.is_autocast_enabled('cuda')
                or torch.get_autocast_dtype('cuda') != torch.bfloat16):
            raise ValueError('GEM paged working memory requires CUDA BF16 attention reads')
        if tokens_per_frame != 1375:
            raise ValueError('GEM paged working memory requires the fixed 518 square raster')
        if self.kv_cache_manager is None:
            self.kv_cache_manager = PagedWorkingMemory(num_blocks=self.depth,
                tokens_per_frame=tokens_per_frame, num_heads=16, head_dim=64, device=device)
            self.kv_cache_manager._skip_append = self.kv_cache.get('_skip_append', False)
        return self.kv_cache_manager

    def clean_kv_cache(self):
        super().clean_kv_cache()
        # In an isolated scale stream this releases only the temporary pool;
        # the live pool is detached and restored by isolated_stream().
        self.kv_cache_manager = None


def is_paged_aggregator(aggregate):
    return type(aggregate) is PagedAggregator


def enable_paged_memory(model):
    """Select before the first write; keep all checkpoint parameter objects."""
    import flashinfer
    from .read_precision import ReadPrecisionAttention

    aggregate = model.aggregator
    root = Path(inspect.getsourcefile(AggregatorStream)).parents[1]
    if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != digest
           for p, digest in NATIVE_SOURCES.items()):
        raise ValueError('LingBot differs from the verified GEM paged integration contract')
    if model.training or int(aggregate.total_frames_processed) != 0:
        raise ValueError('Select paged memory in evaluation before the first observation')
    if aggregate.kv_cache_manager is not None or any(v is not None and not isinstance(v, bool)
            for v in aggregate.kv_cache.values()):
        raise ValueError('Select paged memory with an empty working state')
    if (aggregate.depth != 24 or aggregate.embed_dim != 1024 or
            aggregate.kv_cache_scale_frames != 8 or aggregate.kv_cache_sliding_window != 64
            or not aggregate.kv_cache_cross_frame_special or not aggregate.kv_cache_include_scale_frames
            or aggregate.kv_cache_camera_only):
        raise ValueError('Paged memory requires native W64, initial8 and all special tokens')
    blocks = list(aggregate.global_blocks)
    installed = is_paged_aggregator(aggregate)
    allowed = (PagedAttention,) if installed else (SDPAAttention, ReadPrecisionAttention)
    if (len(blocks) != 24 or any(type(b) is not SDPABlock or type(b.attn) not in allowed
            or b.attn.num_heads != 16 or b.attn.head_dim != 64
            or 'forward' in b.attn.__dict__ for b in blocks)
            or (not installed and (type(aggregate) is not AggregatorStream or not aggregate.use_sdpa))):
        raise ValueError('Unverified global attention implementation')
    aggregate.__class__ = PagedAggregator
    for block in blocks:
        block.attn.__class__ = PagedAttention
    aggregate.use_sdpa, aggregate.use_flashinfer = False, True
    model.use_sdpa = False
    # Camera head continues to use its original independent SDPA caches.
    return dict(storage='paged_bf16', backend='flashinfer_fa2', flashinfer=flashinfer.__version__,
        global_layers=24, key_dtype='bfloat16', value_dtype='bfloat16',
        maximum_committed_views=320, native_sources=NATIVE_SOURCES.copy())
