"""
core.qwen35_patch — enable correct MULTI-TOKEN cache continuation for Qwen3.5's
GatedDeltaNet linear-attention layers (required for speculative decoding).

Why this exists
---------------
Qwen3.5 is a hybrid model: some layers are full attention (croppable KV cache),
others are GatedDeltaNet linear attention with a fixed-size recurrent state. The
stock `Qwen3_5GatedDeltaNet.forward` only continues that recurrent state when
`seq_len == 1` (single-step decode). For any `seq_len > 1` from a populated cache
it takes the prefill path — `initial_state=None` and conv state rebuilt from
scratch — which SILENTLY RESTARTS the recurrence and yields wrong logits.

Empirically (0.8B draft, 3-token block): multi-token-from-cache vs full-prefill
ground truth gave max|Δlogit| = 18.0, argmax match 0.67. Token-by-token (the
single-step path) gave 0.22 (bf16 noise), argmax 1.00.

Speculative decoding needs exactly this broken capability: verify γ proposed
tokens in ONE parallel pass continuing from the committed-prefix cache, and
replay accepted tokens after a snapshot/restore. So we patch the forward to add a
THIRD branch — "continuation, seq_len > 1" — that continues the conv state via the
torch reference update and runs the recurrence through the CUDA
`chunk_gated_delta_rule` with `initial_state=recurrent_state` (the kernel already
supports an initial state; HF just hardcodes None). The fast single-step decode
and prefill paths are left untouched.

The patch is validated against the token-by-token ground truth in
core.spec_decode-adjacent tests; install it once before loading any model.
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
    multi_continue = has_prev and seq_len > 1   # the NEW path

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # --- causal conv ------------------------------------------------------
    if single_step:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state,
            self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
        )
    elif multi_continue:
        # torch reference update correctly continues conv state for any seq_len
        # (cats conv_state, convolves, writes back the trailing window in place).
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

    # --- gated delta rule -------------------------------------------------
    if single_step:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    elif multi_continue:
        # Parallel verify: the (CUDA) chunked kernel accepts an initial_state, so
        # it continues correctly from the cached recurrent state for seq_len > 1.
        # Validated vs full-prefill ground truth: max|Δ| 0.22 (bf16), argmax 1.00.
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
