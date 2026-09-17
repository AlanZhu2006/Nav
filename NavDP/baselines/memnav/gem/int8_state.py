"""Fixed INT8 window storage and a shared, per-layer BF16 SDPA decode buffer.

Initial-eight patches and all special tokens remain BF16. The codec uses
per-head/channel key scales and per-head/token value scales, stored as BF16.
No query, goal, attention gate, or geometry estimate enters quantization.
"""
import math

import torch
import triton
import triton.language as tl


def visible_layout(committed, commit, *, initial=8, window=64, tokens_per_frame=1375):
    """Original SDPA token membership before the current observation is stored."""
    if not initial <= committed < 320:
        raise ValueError('A complete initialization and an available native commit index are required')
    previous_window = min(committed-initial, window-int(bool(commit)))
    evicted = committed-initial-previous_window
    total = evicted*6+(initial+previous_window+1)*tokens_per_frame
    return previous_window, evicted, total


def fixed_int8_encode(k, v):
    """Encode [heads,patches,channels] at the declared stored-scale precision."""
    if k.shape != v.shape or k.ndim != 3 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError('The fixed codec consumes the original BF16 patch K/V')
    kf, vf = k.float(), v.float()
    if not bool(torch.isfinite(kf).all() & torch.isfinite(vf).all()):
        raise ValueError('Nonfinite K/V cannot enter the quantized memory')
    km, vm = kf.abs().amax(dim=1), vf.abs().amax(dim=2, keepdim=True)
    ks = (km/127).to(torch.bfloat16)
    vs = (vm/127).to(torch.bfloat16)
    if bool(((ks == 0)&(km != 0)).any() | ((vs == 0)&(vm != 0)).any()):
        raise ValueError('Nonzero K/V group underflows the declared BF16 scale format')
    ks = torch.where(km == 0, torch.ones_like(ks), ks)
    vs = torch.where(vm == 0, torch.ones_like(vs), vs)
    ki = torch.round(kf/ks[:,None,:].float()).clamp(-127,127).to(torch.int8)
    vi = torch.round(vf/vs.float()).clamp(-127,127).to(torch.int8)
    return ki, vi, ks, vs


class Int8LayerState:
    """One layer's actual tensors and logical view identities, without goals."""
    def __init__(self, initial_kv, *, window=64):
        # [K/V,heads,initial_views,all_tokens,channels]
        if (initial_kv.ndim != 5 or initial_kv.shape[0] != 2 or initial_kv.shape[2] != 8
                or initial_kv.dtype != torch.bfloat16 or initial_kv.shape[3] <= 6 or window != 64):
            raise ValueError('INT8 state requires the native initial eight BF16 views and W64')
        self.heads, self.tokens, self.dim = initial_kv.shape[1], initial_kv.shape[3], initial_kv.shape[4]
        if not bool(torch.isfinite(initial_kv).all()):
            raise ValueError('Nonfinite initial geometry features')
        self.patches = self.tokens-6
        self.device = initial_kv.device
        self.window, self.count = window, 8
        self.initial = initial_kv[:,:,:,6:,:].clone(memory_format=torch.contiguous_format)
        self.specials = initial_kv[:,:,:,:6,:].clone(memory_format=torch.contiguous_format)
        self.special_capacity = 8
        # One actual slot provides valid addresses before a window view exists.
        # Logical occupancy is zero. Later growth uses chunks of eight slots.
        self.capacity = 1
        self.codes = torch.empty((2,1,self.heads,self.patches,self.dim), dtype=torch.int8, device=self.device)
        self.key_scales = torch.empty((1,self.heads,self.dim), dtype=torch.bfloat16, device=self.device)
        self.value_scales = torch.empty((1,self.heads,self.patches,1), dtype=torch.bfloat16, device=self.device)
        self.failure = None

    def healthy(self):
        if self.failure is not None:
            raise RuntimeError('INT8 layer state failed; reset is required: '+self.failure)

    def _reserve(self, slot):
        if slot >= self.capacity:
            capacity = min(self.window, 8*math.ceil((slot+1)/8))
            codes = torch.empty((2,capacity,self.heads,self.patches,self.dim),dtype=torch.int8,device=self.device)
            keys = torch.empty((capacity,self.heads,self.dim),dtype=torch.bfloat16,device=self.device)
            values = torch.empty((capacity,self.heads,self.patches,1),dtype=torch.bfloat16,device=self.device)
            occupied = min(self.window,self.count-8)
            codes[:,:occupied].copy_(self.codes[:,:occupied])
            keys[:occupied].copy_(self.key_scales[:occupied])
            values[:occupied].copy_(self.value_scales[:occupied])
            self.codes,self.key_scales,self.value_scales = codes,keys,values
            self.capacity = capacity
        if self.count >= self.special_capacity:
            capacity = min(320,8*math.ceil((self.count+1)/8))
            special = torch.empty((2,self.heads,capacity,6,self.dim),dtype=torch.bfloat16,device=self.device)
            special[:,:,:self.count].copy_(self.specials[:,:,:self.count])
            self.specials,self.special_capacity = special,capacity

    def commit(self, current_kv):
        """Store only after current attention has consumed complete current K/V."""
        self.healthy()
        if self.count >= 320:
            raise RuntimeError('Native committed-view limit reached')
        if (current_kv.shape != (2,self.heads,self.tokens,self.dim)
                or current_kv.dtype != torch.bfloat16 or current_kv.device != self.device):
            raise ValueError('Current BF16 K/V does not match this memory stream')
        try:
            if not bool(torch.isfinite(current_kv[:,:,:6]).all()):
                raise ValueError('Nonfinite special tokens cannot enter the memory')
            slot = (self.count-8)%self.window
            self._reserve(slot)
            ki,vi,ks,vs = fixed_int8_encode(current_kv[0,:,6:],current_kv[1,:,6:])
            self.codes[0,slot].copy_(ki)
            self.codes[1,slot].copy_(vi)
            self.key_scales[slot].copy_(ks)
            self.value_scales[slot].copy_(vs)
            self.specials[:,:,self.count].copy_(current_kv[:,:,:6])
            self.count += 1
        except BaseException as error:
            self.failure = f'{type(error).__name__}: {error}'
            raise

    def statistics(self):
        active = min(self.count-8,self.window)
        per_window = 2*self.heads*self.patches*self.dim + 2*self.heads*(self.dim+self.patches)
        effective = self.initial.numel()*2 + 2*self.heads*self.count*6*self.dim*2 + active*per_window
        tensors = (self.initial,self.specials,self.codes,self.key_scales,self.value_scales)
        return dict(committed_views=self.count,window_views=active,window_capacity=self.capacity,
            special_capacity=self.special_capacity,effective_storage_bytes=effective,
            allocated_storage_bytes=sum(t.numel()*t.element_size() for t in tensors),failure=self.failure)


@triton.jit(do_not_specialize=['count','visible','evicted','total','window_capacity','special_capacity'])
def _decode_layer(Initial, Specials, Codes, KeyScales, ValueScales, Current, Output,
                  count, visible, evicted, total, window_capacity, special_capacity,
                  HEADS: tl.constexpr, DIM: tl.constexpr, TOKENS: tl.constexpr,
                  PATCHES: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    kind = tl.program_id(1)
    valid = index < HEADS*total*DIM
    channel = index%DIM
    token = (index//DIM)%total
    head = index//(DIM*total)
    old = token < evicted*6
    dense_token = token-evicted*6
    dense_view = dense_token//TOKENS
    local_token = dense_token%TOKENS
    current = (~old)&(dense_view == 8+visible)
    frame = tl.where(old,8+token//6,
        tl.where(dense_view<8,dense_view,count-visible+dense_view-8))
    special_token = tl.where(old,token%6,local_token)
    is_special = (old | (local_token<6)) & (~current)
    is_initial = (~old)&(~current)&(dense_view<8)&(local_token>=6)
    is_window = (~old)&(~current)&(dense_view>=8)&(local_token>=6)
    slot = (frame-8)%64
    patch = local_token-6

    special_offset = (((kind*HEADS+head)*special_capacity+frame)*6+special_token)*DIM+channel
    special_value = tl.load(Specials+special_offset,mask=valid&is_special,other=0).to(tl.float32)
    initial_offset = (((kind*HEADS+head)*8+dense_view)*PATCHES+patch)*DIM+channel
    initial_value = tl.load(Initial+initial_offset,mask=valid&is_initial,other=0).to(tl.float32)
    code_offset = ((((kind*window_capacity+slot)*HEADS+head)*PATCHES+patch)*DIM)+channel
    code = tl.load(Codes+code_offset,mask=valid&is_window,other=0).to(tl.float32)
    key_scale = tl.load(KeyScales+(slot*HEADS+head)*DIM+channel,
        mask=valid&is_window&(kind==0),other=0).to(tl.float32)
    value_scale = tl.load(ValueScales+(slot*HEADS+head)*PATCHES+patch,
        mask=valid&is_window&(kind==1),other=0).to(tl.float32)
    decoded = code*(key_scale+value_scale)
    current_offset = ((kind*HEADS+head)*TOKENS+local_token)*DIM+channel
    current_value = tl.load(Current+current_offset,mask=valid&current,other=0).to(tl.float32)
    value = special_value+initial_value+decoded+current_value
    tl.store(Output+kind*HEADS*total*DIM+index,value,mask=valid)


class Int8DecodeWorkspace:
    """One compute buffer reused by layers and serialized scale-state leases."""
    def __init__(self):
        self.buffer = None

    def decode(self,state,current_kv,*,commit):
        state.healthy()
        if state.device.type != 'cuda' or current_kv.device != state.device:
            raise ValueError('Fused INT8 decoding requires the explicit CUDA backend')
        if (current_kv.shape != (2,state.heads,state.tokens,state.dim)
                or current_kv.dtype != torch.bfloat16 or not current_kv.is_contiguous()):
            raise ValueError('Fused decoder requires contiguous current BF16 K/V')
        visible,evicted,total = visible_layout(state.count,commit,tokens_per_frame=state.tokens)
        maximum = (8+64+1)*state.tokens+(320-8-64)*6
        capacity = 2*state.heads*maximum*state.dim
        if self.buffer is None:
            self.buffer = torch.empty(capacity,dtype=torch.bfloat16,device=state.device)
        if (self.buffer.device != state.device or self.buffer.numel() != capacity):
            raise ValueError('Decoder workspace belongs to a different model shape/device')
        _decode_layer[(triton.cdiv(state.heads*total*state.dim,1024),2)](
            state.initial,state.specials,state.codes,state.key_scales,state.value_scales,
            current_kv,self.buffer,state.count,visible,evicted,total,state.capacity,state.special_capacity,
            HEADS=state.heads,DIM=state.dim,TOKENS=state.tokens,PATCHES=state.patches,
            BLOCK=1024,num_warps=4)
        # Dense native SDPA order and strides; unused capacity is outside view.
        output = self.buffer[:2*state.heads*total*state.dim].view(2,state.heads,total,state.dim)
        return output[0].unsqueeze(0),output[1].unsqueeze(0)

    def statistics(self):
        return dict(allocated_decode_bytes=0 if self.buffer is None else self.buffer.numel()*self.buffer.element_size())
