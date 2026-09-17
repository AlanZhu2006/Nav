# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from LingBot-Map's SDPAAttention; Apache License, Version 2.0.
# GEM change: store each post-RoPE key at the precision consumed by SDPA.
"""Reader-precision storage for the native LingBot streaming working memory.

The forward computation retains native token order, eviction, cloning and
SDPA. It adds one conversion for each new key before caching. Camera state,
checkpoint parameters, depth predictions and downstream matching are owned
by their existing modules. This representation requires CUDA BF16 reads.
"""
import hashlib
import inspect
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from lingbot_map.layers.attention import SDPAAttention
from lingbot_map.layers.rope import apply_rotary_emb

NATIVE_ATTENTION_SHA256 = "59d28bdbb80c7d9138ce5b77fffdc636beee7081767c774a5c6ca7a675925419"

def _gem_key_at_read_precision(key, value):
    if (key.device.type != 'cuda' or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.bfloat16
            or key.dtype != torch.float32 or value.dtype != torch.bfloat16):
        raise RuntimeError('Reader-precision storage requires the verified CUDA BF16 SDPA contract')
    return key.to(dtype=torch.bfloat16)


class ReadPrecisionAttention(SDPAAttention):
    """Native attention with BF16 storage for post-RoPE historical keys."""

    def forward(self, x: Tensor, pos=None,
                num_patches=None, num_special=None, num_frames=None, enable_3d_rope=False,
                kv_cache=None, global_idx=0, num_frame_per_block=1,
                num_frame_for_scale=-1, num_register_tokens=4) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # ========== Batch Mode (no KV cache) ==========
        if kv_cache is None:
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)

        # ========== Streaming Mode (with KV cache dict) ==========
        else:
            if self.rope is not None and not enable_3d_rope:
                q = self.rope(q, pos)
                k = self.rope(k, pos)
            elif self.rope is not None and enable_3d_rope:
                q = apply_rotary_emb(q, pos)
                k = apply_rotary_emb(k, pos)

            camera_token_idx = 0
            scale_token_idx = camera_token_idx + num_register_tokens + 1

            # Check if we should skip appending to cache (non-keyframe in keyframe mode)
            skip_append = kv_cache.get("_skip_append", False)

            if kv_cache[f"k_{global_idx}"] is not None:
                num_frame_per_block = k.shape[2] // kv_cache[f"k_{global_idx}"].shape[3]
            k = _gem_key_at_read_precision(k, v)
            k_reshaped = k.view(B, self.num_heads, num_frame_per_block,
                                N // num_frame_per_block, self.head_dim)
            v_reshaped = v.view(B, self.num_heads, num_frame_per_block,
                                N // num_frame_per_block, self.head_dim)

            if not skip_append:
                # KEYFRAME: store in cache (original behavior)
                if kv_cache[f"k_{global_idx}"] is None:
                    kv_cache[f"k_{global_idx}"] = k_reshaped
                    kv_cache[f"v_{global_idx}"] = v_reshaped
                else:
                    kv_cache[f"k_{global_idx}"] = torch.cat((kv_cache[f"k_{global_idx}"], k_reshaped), dim=2)
                    kv_cache[f"v_{global_idx}"] = torch.cat((kv_cache[f"v_{global_idx}"], v_reshaped), dim=2)

                self._apply_kv_cache_eviction(
                    kv_cache, global_idx, camera_token_idx, scale_token_idx, num_register_tokens
                )

                k_cached = kv_cache[f"k_{global_idx}"].clone()
                v_cached = kv_cache[f"v_{global_idx}"].clone()
            else:
                # NON-KEYFRAME: attend to [cached + current] without storing in cache
                if kv_cache[f"k_{global_idx}"] is not None:
                    k_cached = torch.cat((kv_cache[f"k_{global_idx}"], k_reshaped), dim=2)
                    v_cached = torch.cat((kv_cache[f"v_{global_idx}"], v_reshaped), dim=2)
                else:
                    k_cached = k_reshaped
                    v_cached = v_reshaped
            a, b, c, d, e = k_cached.shape
            k_full = k_cached.reshape(a, b, c * d, e)
            v_full = v_cached.reshape(a, b, c * d, e)

            if f"k_{global_idx}_special" in kv_cache and kv_cache[f"k_{global_idx}_special"] is not None:
                special_k = kv_cache[f"k_{global_idx}_special"]
                special_v = kv_cache[f"v_{global_idx}_special"]
                sa, sb, sc, sd, se = special_k.shape
                k_full = torch.cat([special_k.reshape(sa, sb, sc * sd, se), k_full], dim=2)
                v_full = torch.cat([special_v.reshape(sa, sb, sc * sd, se), v_full], dim=2)

            q_seq_len = q.shape[2]
            x = F.scaled_dot_product_attention(
                q, k_full, v_full,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(B, q_seq_len, self.num_heads * self.head_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def enable_reader_precision(model):
    """Select the tested storage representation without replacing weights.

    Selection occurs with an empty stream. Repeating it after ordinary reset
    is idempotent; changing a populated native cache is explicitly rejected.
    No model parameters or frame-attention modules are replaced.
    """
    aggregate = model.aggregator
    if model.training or not aggregate.use_sdpa or aggregate.kv_cache_manager is not None:
        raise ValueError("Reader-precision storage requires evaluation with native SDPA")
    source = Path(inspect.getsourcefile(SDPAAttention))
    if hashlib.sha256(source.read_bytes()).hexdigest() != NATIVE_ATTENTION_SHA256:
        raise ValueError("LingBot attention differs from the verified reader-precision contract")
    attentions = [block.attn for block in aggregate.global_blocks]
    if len(attentions) != 24:
        raise ValueError("Reader-precision storage requires the verified 24 global layers")
    installed = all(type(attention) is ReadPrecisionAttention for attention in attentions)
    if not installed:
        if not all(type(attention) is SDPAAttention and "forward" not in attention.__dict__
                   for attention in attentions):
            raise ValueError("Global attention contains an unverified implementation")
        if int(aggregate.total_frames_processed) != 0 or any(
                value is not None and not isinstance(value, bool)
                for value in aggregate.kv_cache.values()):
            raise ValueError("Select reader-precision storage before writing the first observation")
        for attention in attentions:
            attention.__class__ = ReadPrecisionAttention
    return dict(storage="reader_precision", global_layers=24,
                key_dtype="bfloat16", value_dtype="bfloat16",
                native_attention_sha256=NATIVE_ATTENTION_SHA256)
