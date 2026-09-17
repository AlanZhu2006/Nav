"""Exact BF16 bit-plane storage for the fixed GEM W64 working state.

Every 256 original values stores its minimum exponent and full exponent
range, as 0..8 complete bit planes. Sign/fraction bits remain raw. There is
no quantization, omitted outlier, cross-view predictor, or alternate backend.
The variable payload is allocated at its actual encoded size on the GPU.
"""
from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl

BLOCK_VALUES = 256
WORDS_PER_PLANE = BLOCK_VALUES // 32


@triton.jit
def _analyze(Input, SignMantissa, Metadata, WordCounts, N: tl.constexpr):
    block = tl.program_id(0)
    i = block * 256 + tl.arange(0, 256)
    u = tl.load(Input + i, mask=i < N, other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
    exp = (u >> 7) & 255
    low = tl.min(tl.where(i < N, exp, 255), 0)
    high = tl.max(tl.where(i < N, exp, 0), 0)
    span = high - low
    width = ((span >= 1).to(tl.int32) + (span >= 2).to(tl.int32)
             + (span >= 4).to(tl.int32) + (span >= 8).to(tl.int32)
             + (span >= 16).to(tl.int32) + (span >= 32).to(tl.int32)
             + (span >= 64).to(tl.int32) + (span >= 128).to(tl.int32))
    tl.store(SignMantissa + i, ((u >> 8) & 128) | (u & 127), mask=i < N)
    tl.store(Metadata + block, low | (width << 8))
    tl.store(WordCounts + block, width * 8)


@triton.jit
def _encode_planes(Input, Metadata, Offsets, Planes, N: tl.constexpr):
    block = tl.program_id(0)
    i = block * 256 + tl.arange(0, 256)
    meta = tl.load(Metadata + block)
    low, width = meta & 255, meta >> 8
    u = tl.load(Input + i, mask=i < N, other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
    delta = tl.where(i < N, ((u >> 7) & 255) - low, 0).reshape(8, 32)
    plane = tl.arange(0, 8)
    lane = tl.arange(0, 32)
    bits = ((delta[None, :, :] >> plane[:, None, None]) & 1) << lane[None, None, :]
    words = tl.sum(bits, 2)
    start = tl.load(Offsets + block)
    out = start + plane[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Planes + out, words, mask=plane[:, None] < width)


@triton.jit
def _read_bits(SignMantissa, Metadata, Offsets, Planes, index, valid):
    block, local = index // 256, index % 256
    meta = tl.load(Metadata + block, mask=valid, other=0)
    low, width = meta & 255, meta >> 8
    start = tl.load(Offsets + block, mask=valid, other=0)
    delta = tl.full(index.shape, 0, tl.uint32)
    for bit in tl.static_range(8):
        word = tl.load(Planes + start + bit * 8 + local // 32,
                       mask=valid & (width > bit), other=0).to(tl.uint32)
        delta = delta | (((word >> (local % 32)) & 1) << bit)
    sm = tl.load(SignMantissa + index, mask=valid, other=0).to(tl.uint32)
    return (((sm & 128) << 8) | ((low + delta) << 7) | (sm & 127)).to(tl.uint16)


@triton.jit
def _decode_tensor(SignMantissa, Metadata, Offsets, Planes, Output, N: tl.constexpr):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    bits = _read_bits(SignMantissa, Metadata, Offsets, Planes, index, index < N)
    tl.store(Output + index, bits.to(tl.bfloat16, bitcast=True), mask=index < N)


@dataclass
class LosslessBlock:
    sign_mantissa: torch.Tensor
    metadata: torch.Tensor
    offsets: torch.Tensor
    planes: torch.Tensor
    shape: tuple

    def tensors(self):
        return self.sign_mantissa, self.metadata, self.offsets, self.planes

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def decode(self):
        out = torch.empty(self.shape, device=self.sign_mantissa.device, dtype=torch.bfloat16)
        _decode_tensor[(triton.cdiv(out.numel(), 256),)](*self.tensors(), out, N=out.numel())
        return out


def encode_bf16(tensor):
    """Encode original bits, including signed zero and every NaN bit pattern.

    One size synchronization per committed view/layer permits a real compact
    allocation. This overhead must remain in measured online write costs.
    """
    if tensor.device.type != 'cuda' or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        raise ValueError('Lossless codec requires contiguous CUDA BF16 input')
    n = tensor.numel()
    if not n or n >= 2**31:
        raise ValueError('Lossless block requires 0 < elements < 2**31')
    blocks = triton.cdiv(n, BLOCK_VALUES)
    sm = torch.empty(n, dtype=torch.uint8, device=tensor.device)
    meta = torch.empty(blocks, dtype=torch.int32, device=tensor.device)
    counts = torch.empty_like(meta)
    offsets = torch.empty(blocks + 1, dtype=torch.int32, device=tensor.device)
    _analyze[(blocks,)](tensor, sm, meta, counts, N=n)
    offsets[0].zero_()
    torch.cumsum(counts, 0, dtype=torch.int32, out=offsets[1:])
    words = int(offsets[-1].item())
    planes = torch.empty(words, dtype=torch.int32, device=tensor.device)
    _encode_planes[(blocks,)](tensor, meta, offsets, planes, N=n)
    return LosslessBlock(sm, meta, offsets, planes, tuple(tensor.shape))


def visible_layout(committed, commit, *, tokens_per_frame=1375):
    if not 8 <= committed < 320:
        raise ValueError('Native initialization and an available commit index are required')
    visible = min(committed - 8, 64 - int(bool(commit)))
    evicted = committed - 8 - visible
    return visible, evicted, evicted * 6 + (8 + visible + 1) * tokens_per_frame


class LosslessLayerState:
    def __init__(self, initial_kv):
        if (initial_kv.ndim != 5 or initial_kv.shape[0] != 2 or initial_kv.shape[2] != 8
                or initial_kv.shape[3] <= 6 or initial_kv.dtype != torch.bfloat16
                or initial_kv.device.type != 'cuda'):
            raise ValueError('Lossless state requires the original eight CUDA BF16 views')
        self.heads, self.tokens, self.dim = initial_kv.shape[1], initial_kv.shape[3], initial_kv.shape[4]
        self.patches, self.device = self.tokens - 6, initial_kv.device
        self.initial = initial_kv[:, :, :, 6:].clone(memory_format=torch.contiguous_format)
        self.specials = initial_kv[:, :, :, :6].clone(memory_format=torch.contiguous_format)
        self.count, self.special_capacity = 8, 8
        self.blocks = [None] * 64
        self.pointers = torch.zeros((64, 4), dtype=torch.int64, device=self.device)
        self._initial_storage_bytes = self.initial.numel() * self.initial.element_size()
        self._pointer_storage_bytes = self.pointers.numel() * self.pointers.element_size()
        self._special_bytes_per_view = 2 * self.heads * 6 * self.dim * 2
        self._compressed_window_bytes = 0
        self.failure = None

    def healthy(self):
        if self.failure is not None:
            raise RuntimeError('Lossless layer failed; reset is required: ' + self.failure)

    def commit(self, current_kv):
        self.healthy()
        if self.count >= 320:
            raise RuntimeError('Native committed-view limit reached')
        if (current_kv.shape != (2, self.heads, self.tokens, self.dim)
                or current_kv.dtype != torch.bfloat16 or current_kv.device != self.device):
            raise ValueError('Current BF16 K/V does not match the lossless stream')
        try:
            slot = (self.count - 8) % 64
            block = encode_bf16(current_kv[:, :, 6:].contiguous())
            if self.count >= self.special_capacity:
                capacity = min(320, 8 * math.ceil((self.count + 1) / 8))
                special = torch.empty((2, self.heads, capacity, 6, self.dim),
                                      dtype=torch.bfloat16, device=self.device)
                special[:, :, :self.count].copy_(self.specials[:, :, :self.count])
                self.specials, self.special_capacity = special, capacity
            self.specials[:, :, self.count].copy_(current_kv[:, :, :6])
            addresses = torch.tensor([t.data_ptr() for t in block.tensors()],
                                     dtype=torch.int64, device=self.device)
            self.pointers[slot].copy_(addresses)
            previous_bytes = 0 if self.blocks[slot] is None else self.blocks[slot].storage_bytes
            compressed_bytes = self._compressed_window_bytes + block.storage_bytes - previous_bytes
            self.blocks[slot] = block
            self._compressed_window_bytes = compressed_bytes
            self.count += 1
        except BaseException as error:
            self.failure = f'{type(error).__name__}: {error}'
            raise

    def tensors(self):
        return (self.initial, self.specials, self.pointers) + tuple(
            t for block in self.blocks if block is not None for t in block.tensors())

    def statistics(self):
        # Query accounting reads Python integers only. Update compressed bytes
        # when a view is committed/evicted; keep failures and allocated special
        # capacity live without traversing blocks or retaining tensor handles.
        active = min(self.count - 8, 64)
        compressed = self._compressed_window_bytes
        raw = active * 2 * self.heads * self.patches * self.dim * 2
        protected = self._initial_storage_bytes + self.count * self._special_bytes_per_view
        allocated = (self._initial_storage_bytes + self.special_capacity * self._special_bytes_per_view
                     + self._pointer_storage_bytes + compressed)
        return dict(committed_views=self.count, window_views=active, special_capacity=self.special_capacity,
                    raw_window_bytes=raw, compressed_window_bytes=compressed,
                    effective_storage_bytes=protected + compressed + self._pointer_storage_bytes,
                    allocated_storage_bytes=allocated,
                    failure=self.failure)


@triton.jit(do_not_specialize=['count', 'visible', 'evicted', 'total', 'special_capacity'])
def _decode_layer(Initial, Specials, Pointers, Current, Output,
                    count, visible, evicted, total, special_capacity,
                    HEADS: tl.constexpr, DIM: tl.constexpr, TOKENS: tl.constexpr,
                    PATCHES: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    window_tiles: tl.constexpr = triton.cdiv(2 * HEADS * PATCHES * DIM, BLOCK)
    initial_tiles: tl.constexpr = triton.cdiv(2 * HEADS * 8 * PATCHES * DIM, BLOCK)
    window_end = visible * window_tiles
    initial_end = window_end + initial_tiles
    special_end = initial_end + tl.cdiv(2 * HEADS * count * 6 * DIM, BLOCK)
    if program < window_end:
        view = program // window_tiles
        index = (program % window_tiles) * BLOCK + lane
        valid = index < 2 * HEADS * PATCHES * DIM
        frame = count - visible + view
        slot = (frame - 8) % 64
        # One uniform pointer set and one aligned 256-value codec block per
        # program. No output-token-to-history search in the compressed path.
        sm = tl.load(Pointers + slot * 4).to(tl.pointer_type(tl.uint8))
        meta = tl.load(Pointers + slot * 4 + 1).to(tl.pointer_type(tl.int32))
        offsets = tl.load(Pointers + slot * 4 + 2).to(tl.pointer_type(tl.int32))
        planes = tl.load(Pointers + slot * 4 + 3).to(tl.pointer_type(tl.int32))
        bits = _read_bits(sm, meta, offsets, planes, index, valid)
        head_kind = index // (PATCHES * DIM)
        patch, channel = (index // DIM) % PATCHES, index % DIM
        token = evicted * 6 + (8 + view) * TOKENS + 6 + patch
        destination = (head_kind * total + token) * DIM + channel
    elif program < initial_end:
        index = (program - window_end) * BLOCK + lane
        valid = index < 2 * HEADS * 8 * PATCHES * DIM
        bits = tl.load(Initial + index, mask=valid, other=0).to(tl.uint16, bitcast=True)
        head_kind = index // (8 * PATCHES * DIM)
        view = (index // (PATCHES * DIM)) % 8
        patch, channel = (index // DIM) % PATCHES, index % DIM
        token = evicted * 6 + view * TOKENS + 6 + patch
        destination = (head_kind * total + token) * DIM + channel
    elif program < special_end:
        index = (program - initial_end) * BLOCK + lane
        valid = index < 2 * HEADS * count * 6 * DIM
        head_kind = index // (count * 6 * DIM)
        frame = (index // (6 * DIM)) % count
        special, channel = (index // DIM) % 6, index % DIM
        source = ((head_kind * special_capacity + frame) * 6 + special) * DIM + channel
        bits = tl.load(Specials + source, mask=valid, other=0).to(tl.uint16, bitcast=True)
        first = count - visible
        token = tl.where(frame < 8, evicted * 6 + frame * TOKENS + special,
                         tl.where(frame < first, (frame - 8) * 6 + special,
                                  evicted * 6 + (8 + frame - first) * TOKENS + special))
        destination = (head_kind * total + token) * DIM + channel
    else:
        index = (program - special_end) * BLOCK + lane
        valid = index < 2 * HEADS * TOKENS * DIM
        bits = tl.load(Current + index, mask=valid, other=0).to(tl.uint16, bitcast=True)
        head_kind = index // (TOKENS * DIM)
        token, channel = (index // DIM) % TOKENS, index % DIM
        token = evicted * 6 + (8 + visible) * TOKENS + token
        destination = (head_kind * total + token) * DIM + channel
    tl.store(Output + destination, bits.to(tl.bfloat16, bitcast=True), mask=valid)


class LosslessDecodeWorkspace:
    """One chronological BF16 buffer, reused by the original SDPA layers."""
    def __init__(self):
        self.buffer = None

    def decode(self, state, current_kv, *, commit):
        state.healthy()
        if (current_kv.device != state.device or current_kv.dtype != torch.bfloat16
                or current_kv.shape != (2, state.heads, state.tokens, state.dim)
                or not current_kv.is_contiguous()):
            raise ValueError('Lossless decoder requires the declared contiguous CUDA BF16 current view')
        visible, evicted, total = visible_layout(state.count, commit, tokens_per_frame=state.tokens)
        maximum = (8 + 64 + 1) * state.tokens + (320 - 8 - 64) * 6
        capacity = 2 * state.heads * maximum * state.dim
        if self.buffer is None:
            self.buffer = torch.empty(capacity, dtype=torch.bfloat16, device=state.device)
        if self.buffer.device != state.device or self.buffer.numel() != capacity:
            raise ValueError('Lossless workspace belongs to a different model shape/device')
        programs = (visible * triton.cdiv(2 * state.heads * state.patches * state.dim, 256)
                    + triton.cdiv(2 * state.heads * 8 * state.patches * state.dim, 256)
                    + triton.cdiv(2 * state.heads * state.count * 6 * state.dim, 256)
                    + triton.cdiv(2 * state.heads * state.tokens * state.dim, 256))
        _decode_layer[(programs,)](
            state.initial, state.specials, state.pointers, current_kv, self.buffer,
            state.count, visible, evicted, total, state.special_capacity,
            HEADS=state.heads, DIM=state.dim, TOKENS=state.tokens, PATCHES=state.patches,
            BLOCK=256, num_warps=4)
        output = self.buffer[:2 * state.heads * total * state.dim].view(2, state.heads, total, state.dim)
        return output[0].unsqueeze(0), output[1].unsqueeze(0)

    def statistics(self):
        return dict(allocated_decode_bytes=0 if self.buffer is None else self.buffer.numel() * 2)
