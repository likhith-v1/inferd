"""Patch Qwen3.5 GatedDeltaNet for multi-token cache continuation.

Stock Qwen3.5 only continues linear-attention recurrent state for single-token
decode. Speculative verification needs `seq_len > 1` continuation from cache, so
this adds that branch while leaving prefill and single-step decode untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from transformers.models.qwen3_5 import modeling_qwen3_5 as _q5

_PATCHED = False


def _patched_gdn_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    hidden_states = _q5.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    has_prev = (
        cache_params is not None
        and cache_params.has_previous_state
        and cache_position is not None
    )
    single_step = has_prev and seq_len == 1
    multi_continue = has_prev and seq_len > 1

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if single_step:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state,
            self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
        )
    elif multi_continue:
        mixed_qkv = _q5.torch_causal_conv1d_update(
            mixed_qkv, conv_state,
            self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
        )
    else:  # prefill / no previous state
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv, weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias, activation=self.activation, seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if single_step:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    elif multi_continue:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:  # prefill
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def install_parallel_verify_patch() -> None:
    """Idempotently patch Qwen3_5GatedDeltaNet.forward for multi-token continuation."""
    global _PATCHED
    if _PATCHED:
        return
    _q5.Qwen3_5GatedDeltaNet.forward = _patched_gdn_forward
    _PATCHED = True
