"""Explicit INT8 working-state adapter; original SDPA and readouts are retained."""
import hashlib
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F

from lingbot_map.aggregator.base import slice_expand_and_flatten
from lingbot_map.aggregator.stream import AggregatorStream
from lingbot_map.layers.attention import SDPAAttention
from lingbot_map.layers.block import SDPABlock
from lingbot_map.layers.rope import apply_rotary_emb

from .int8_state import Int8LayerState, Int8DecodeWorkspace
from .read_precision import ReadPrecisionAttention, _gem_key_at_read_precision, NATIVE_ATTENTION_SHA256


class Int8Attention(SDPAAttention):
    """Full current Q/K/V; history-only compression after current attention."""
    def forward(self,x,pos=None,num_patches=None,num_special=None,num_frames=None,
                enable_3d_rope=False,kv_cache=None,global_idx=0,num_frame_per_block=1,
                num_frame_for_scale=-1,num_register_tokens=4):
        if self.training or x.shape[0]!=1 or not isinstance(kv_cache,dict):
            raise ValueError('INT8 GEM attention requires one explicit evaluation stream')
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        q,k=self.q_norm(q),self.k_norm(k)
        if self.rope is not None:
            if enable_3d_rope:
                q,k=apply_rotary_emb(q,pos),apply_rotary_emb(k,pos)
            else:
                q,k=self.rope(q,pos),self.rope(k,pos)
        k=_gem_key_at_read_precision(k,v)
        identity=f'_gem_int8_{global_idx}'
        state=kv_cache.get(identity)
        commit=not kv_cache.get('_skip_append',False)
        try:
            if state is None:
                if num_frames!=8 or N!=8*1375 or not commit:
                    raise ValueError('INT8 memory must start with the original joint eight views')
                # Native joint token order/precision; cloning the retained
                # patches avoids retaining unrelated qkv projection storage.
                out=F.scaled_dot_product_attention(q,k.contiguous(),v.contiguous(),dropout_p=0.)
                initial=torch.stack((k,v)).reshape(2,self.num_heads,8,1375,self.head_dim)
                state=Int8LayerState(initial)
                kv_cache[identity]=state
            else:
                if num_frames!=1 or N!=1375:
                    raise ValueError('Initialized INT8 memory reads one current observation')
                state.healthy()
                current=torch.stack((k[0],v[0])).contiguous()
                keys,values=self._gem_decode_workspace.decode(state,current,commit=commit)
                out=F.scaled_dot_product_attention(q,keys,values,dropout_p=0.)
                if commit:
                    state.commit(current)
            out=out.transpose(1,2).reshape(B,N,self.num_heads*self.head_dim)
            return self.proj_drop(self.proj(out))
        except BaseException as error:
            if state is not None:
                state.failure=f'{type(error).__name__}: {error}'
            raise


class Int8Aggregator(AggregatorStream):
    """Native temporal positions with explicitly typed per-layer memory states."""
    gem_int8_memory=True

    def _prepare_special_tokens(self,B,S_local,S_global,C,num_frame_for_scale=None):
        if B!=1:
            raise ValueError('INT8 GEM requires one serialized observation stream')
        scale_frames=self.num_frame_for_scale if num_frame_for_scale is None else num_frame_for_scale
        state=self.kv_cache.get('_gem_int8_0')
        # Native SDPA's cached frame dimension is capped at initial8+window64;
        # full RoPE commit time remains aggregator.total_frames_processed.
        cached=0 if state is None else min(state.count,72)
        total=cached+S_global
        effective=min(scale_frames,total)
        camera=slice_expand_and_flatten(self.camera_token,B,total)[-S_global:]
        register=slice_expand_and_flatten(self.register_token,B,total)[-S_global:]
        scale=slice_expand_and_flatten(self.scale_token,B,total,first_num_frame=effective)[-S_global:]
        result=torch.cat((camera,register,scale),dim=1)
        if result.shape!=(B*S_global,self.num_special_tokens,C):
            raise ValueError('INT8 temporal special-token layout differs from the model')
        return result

    def int8_statistics(self):
        # Return values only: an observer must not retain prior cache tensors.
        rows=[state.statistics() for i in range(self.depth)
              if (state:=self.kv_cache.get(f'_gem_int8_{i}')) is not None]
        return dict(storage='int8_storage',attention='native_sdpa_bf16',
            key_scale='per_view_head_channel_bf16',value_scale='per_token_head_bf16',
            initialized_layers=len(rows),committed_views=[r['committed_views'] for r in rows],
            effective_storage_bytes=sum(r['effective_storage_bytes'] for r in rows),
            allocated_storage_bytes=sum(r['allocated_storage_bytes'] for r in rows),
            decode_workspace=self.gem_decode_workspace.statistics(),layers=rows)


def enable_int8_storage(model):
    """Select the fixed codec before writing; never reinterpret live caches."""
    import triton
    aggregate=model.aggregator
    expected={Path(inspect.getsourcefile(AggregatorStream)):
        '6a19e43cb3ddaa569a56d2b9c6675d1693be0d8ad381803a354de32f75aa0a5f',
        Path(inspect.getsourcefile(SDPAAttention)):NATIVE_ATTENTION_SHA256,
        Path(inspect.getsourcefile(SDPABlock)):
        '90085ca55a8fdbbcda1529448f8ef4c551f7b3006631923856374d30b098f741'}
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in expected.items()):
        raise ValueError('LingBot differs from the fixed INT8 integration contract')
    if (model.training or not aggregate.use_sdpa or aggregate.kv_cache_manager is not None
            or int(aggregate.total_frames_processed)!=0
            or any(v is not None and not isinstance(v,bool) for v in aggregate.kv_cache.values())):
        raise ValueError('Select INT8 storage in native SDPA evaluation with an empty stream')
    if (aggregate.depth!=24 or aggregate.embed_dim!=1024 or aggregate.kv_cache_scale_frames!=8
            or aggregate.kv_cache_sliding_window!=64 or not aggregate.kv_cache_cross_frame_special
            or not aggregate.kv_cache_include_scale_frames or aggregate.kv_cache_camera_only):
        raise ValueError('INT8 storage requires initial8/W64/all historical special tokens')
    installed=type(aggregate) is Int8Aggregator
    permitted=(Int8Attention,) if installed else (SDPAAttention,ReadPrecisionAttention)
    blocks=list(aggregate.global_blocks)
    if (type(aggregate) not in (AggregatorStream,Int8Aggregator) or len(blocks)!=24
            or any(type(b) is not SDPABlock or type(b.attn) not in permitted
                   or b.attn.num_heads!=16 or b.attn.head_dim!=64
                   or 'forward' in b.attn.__dict__ for b in blocks)):
        raise ValueError('Unverified global attention in INT8 storage selection')
    if not installed:
        aggregate.__class__=Int8Aggregator
        aggregate.gem_decode_workspace=Int8DecodeWorkspace()
        for block in blocks:
            block.attn.__class__=Int8Attention
            block.attn._gem_decode_workspace=aggregate.gem_decode_workspace
    return dict(storage='int8_storage',attention='native_sdpa_bf16',triton=triton.__version__,
        initial_patch_dtype='bfloat16',historical_special_dtype='bfloat16',
        window_kv_dtype='int8',scale_dtype='bfloat16',decode_dtype='bfloat16',
        key_scale_axes='view/head/channel',value_scale_axes='view/head/token',
        maximum_committed_views=320,source_sha256={str(p):h for p,h in expected.items()})
