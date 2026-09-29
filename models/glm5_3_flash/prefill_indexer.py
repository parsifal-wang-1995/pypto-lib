# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The kpool DSA indexer, prefill path: projections, scoring, selection.

The pooled-table half of the pipeline lives in
:mod:`models.glm5_3_flash.indexer_cache` under the scheme-B layout decided for
this module: the raw support table is scattered per token, closed pools are
compressed once into a BF16 pooled key table by pool-close events, and the
scorer reads that table directly. What remains here is the per-query half:

1. **Projections.** ``q = wq_b(q_resid)`` [T, 32, 128]; ``k = k_norm(wk(x))``
   [T, 128], where ``k_norm`` is a full **LayerNorm with eps 1e-6** carrying a
   bias — the reference ``Glm5NextTextIndexer`` builds ``nn.LayerNorm(head_dim,
   eps=1e-6)``, so unlike every other norm in this model it subtracts a mean;
   ``head_weights = weights_proj(x) * 32 ** -0.5`` [T, 32]; and
   ``gate_scores = compress_gate @ x`` [T, 128]. Every weight is BF16 in the
   checkpoint. All projection weights are declared K-major ``[in, out]`` — the
   loader transposes the checkpoint's ``nn.Linear`` layout once at load time,
   the way the a2a3 sibling port does, so no kernel slice needs a transposed
   matmul.
2. **Scoring.** ``relu(q . pool_key * 128 ** -0.5)`` per head, then a weighted
   sum over the 32 heads. The query head is Hadamard-128 rotated and quantized
   to INT8 first, and the pooled key rows are quantized per row on read, so the
   dot product runs as an exact INT32 ``mmad`` and both scales re-enter in FP32
   — the a2a3 retarget of the upstream FP8 numerics (see below). Each query row
   scores only its own request's pools: ``seg_start`` locates the request's
   segment in the compacted pool id space and ``pool_count`` is the causally
   visible pool count, ``(position + 1) // index_kpool`` clamped to the
   request's pool count. Everything outside ``[seg_start, seg_start +
   pool_count)`` stays at the finite ``FP32_NEG_INF``.
3. **Selection.** Exact top ``index_topk / index_kpool`` = 512 pools per query
   over the same 262144-candidate cap as the donor, with a ``selected_valid``
   output so the expansion can tell padding from a real pick. Selections are
   emitted as **segment-local** pool ids — request-relative, so pool ``j`` maps
   to raw positions ``4j .. 4j+3`` without knowing the batch layout.
4. **Expansion.** Each selected pool becomes 4 raw logical positions; the
   incomplete tail pool is always appended (``index_kpool_always_select_tail``);
   the row is padded with ``-1`` to the fixed ``TOPK_INDEX_WIDTH`` = 2051.
   **The live positions are front packed**: every valid position precedes every
   ``-1``, so a row's padding is one suffix. This is an ABI guarantee both
   sparse attention kernels rely on — they test one lane to decide a whole
   128-wide block, or a whole row, carries no selection. The reference
   implementation leaves holes (it places the tail at a fixed column even for
   short requests); this kernel compacts, matching what the consumers require.

All 32 indexer heads live on **every** rank: the score sums over heads before
the top-k, so head-sharding would force a cross-rank reduction of partial
scores on every sparse layer, and this kernel is small enough that replication
is cheaper.

**There is no rope here.** ``indexer_rope_interleave`` is set in ``config.json``
but is a vestigial field inherited from the GLM-MoE-DSA base: the model forward
passes ``position_embeddings=None`` and ``Glm5NextTextIndexer.forward`` never
touches a cos/sin table. Combined with ``qk_rope_head_dim = 0``, this model has
no rope anywhere.

**Donors, all a2a3 and all already run by the daily sweep.**
``models/deepseek_v4_flash_mtp/prefill_indexer.py`` is the shape to port. Its
``_cp_topk512_query`` is an exact top-512 over the same 262144-candidate cap,
because DeepSeek-V4-Flash's indexer is itself a ratio-4 compressed selector, and
it is reused here almost verbatim. Two device facts recorded there cost real
debugging time: a narrow (256) sort **faults with 507018**, so the leaf stays
wide (2048); and the merge-stage list must match the leaf, because a 4096 stage
on a 2048-score row "lowers to an illegal AIV config".

What was deleted from the donor: the compressor with its ratio-4 slot rewrite
(superseded by the pooled-table writers) and rope. What was **kept**: the
Hadamard-128 rotation and the INT8 quantization of the indexer query — that
half is GLM's own numerics, not a donor quirk. The upstream kernel rotates each
128-wide query head by a Hadamard-128 and quantizes (FP8 e4m3 upstream, INT8
here because the a2a3 cube has no fp8 ``mmad``). The checkpoint ships no
Hadamard matrix; it is generated at load time, and the fixtures build the
Sylvester form scaled by ``128 ** -0.5``.

vLLM Ascend is no help beyond semantics: ``sparse_attn_indexer_kpool.py``
orchestrates a Triton scorer whose top-k is a fused ``aclnn`` call, so only its
contract was ported, not its kernels.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, FP32_NEG_INF
from models.glm5_3_flash.config import INDEX_BLOCKS_DYN, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import KPOOL_SELECT_K, POOLS_DYN, Q_LORA, T_DYN
from models.glm5_3_flash.config import TOPK_INDEX_WIDTH
from models.glm5_3_flash.indexer_cache import POOL_SCALE_WIDTH
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8

# The sort tree: 2048-wide leaves are the donor's confirmed fault-free width, and
# the 64/256/1024 merge stages fully order one leaf. A 4096 stage on this row
# lowers to an illegal AIV config, so the running merge across leaves uses the
# two-tensor mrgsort instead of a wider stage.
LEAF = 2048
PAIR_WIDTH = 2 * KPOOL_SELECT_K

# k_norm is nn.LayerNorm(head_dim, eps=1e-6) in the reference — mean-subtracted,
# biased, and at a different eps than the backbone's RMSNorm.
K_NORM_EPS = 1e-6
SOFTMAX_SCALE = INDEX_DIM**-0.5
WEIGHTS_SCALE = INDEX_H**-0.5

PROJ_T_TILE = 16
PROJ_K_TILE = 128
Q_OUT_TILE = 256
QH_MM_TILE = 64
SCORE_C_TILE = 64
SCORE_INIT_COLS = 1024
EXPAND_W_TILE = 256
EXPAND_TAIL_CHUNK = 8  # covers the last lanes in one aligned tile
# pl.const accepts literals only; assert keeps the hard-coded start honest.
assert TOPK_INDEX_WIDTH - EXPAND_TAIL_CHUNK == 2043
# pl.full takes float constants even for integer tensors; precompute because
# the tracer does not allow calling float() inside kernel bodies.


def golden_indexer_proj(
    x: torch.Tensor,
    q_resid: torch.Tensor,
    w_q_b: torch.Tensor,
    w_k: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    w_weights: torch.Tensor,
    w_compress_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four indexer projections in FP32, rounded once per output dtype.

    Weights arrive K-major ``[in, out]``; the reference's ``nn.Linear`` layout
    is the transpose, applied once at load time.
    """
    tokens = x.shape[0]
    heads = w_weights.shape[1]
    dim = w_k.shape[1]
    index_q = torch.nn.functional.linear(q_resid.float(), w_q_b.t().float())
    index_q = index_q.reshape(tokens, heads, dim).to(torch.bfloat16)

    projected = torch.nn.functional.linear(x.float(), w_k.t().float())
    mean = projected.mean(dim=-1, keepdim=True)
    centered = projected - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + K_NORM_EPS)
    index_k = (normalized * k_norm_weight.float() + k_norm_bias.float()).to(torch.bfloat16)

    head_weights = torch.nn.functional.linear(x.float(), w_weights.t().float()) * WEIGHTS_SCALE
    gate_scores = torch.nn.functional.linear(x.float(), w_compress_gate.t().float())
    return index_q, index_k, head_weights, gate_scores


@pl.jit.inline
def indexer_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
):
    """Project one dispatch's rows: queries, biased LayerNorm keys, both gates.

    ``T`` must be a multiple of ``PROJ_T_TILE`` = 16; the scheduler pads a
    packed batch up to it and feeds the padding rows ordinary metadata. Four
    independent scopes, one cube accumulator each, so the compiler can lay them
    out concurrently.
    """
    t_dim = pl.tensor.dim(x, 0)
    q_flat = pl.reshape(index_q, [t_dim, INDEX_H * INDEX_DIM])
    q_blocks = (INDEX_H * INDEX_DIM) // Q_OUT_TILE

    for idx in pl.spmd((t_dim // PROJ_T_TILE) * q_blocks, name_hint="indexer_q_proj"):
        t_block = idx // q_blocks
        t0 = t_block * PROJ_T_TILE
        o0 = (idx - t_block * q_blocks) * Q_OUT_TILE
        q_acc = pl.create_tensor([PROJ_T_TILE, Q_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, Q_LORA // PROJ_K_TILE, stage=2):
            k0 = kb * PROJ_K_TILE
            q_acc = pl.matmul_acc(
                q_acc,
                q_resid[t0 : t0 + PROJ_T_TILE, k0 : k0 + PROJ_K_TILE],
                w_q_b[k0 : k0 + PROJ_K_TILE, o0 : o0 + Q_OUT_TILE],
                init_cond=(k0 == 0),
            )
        q_flat[t0 : t0 + PROJ_T_TILE, o0 : o0 + Q_OUT_TILE] = pl.cast(q_acc, pl.BF16, mode="rint")

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_k_proj"):
        t0 = idx * PROJ_T_TILE
        k_acc = pl.create_tensor([PROJ_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            k_acc = pl.matmul_acc(
                k_acc,
                x[t0 : t0 + PROJ_T_TILE, d0 : d0 + PROJ_K_TILE],
                w_k[d0 : d0 + PROJ_K_TILE, 0:INDEX_DIM],
                init_cond=(d0 == 0),
            )
        mean = pl.mul(pl.row_sum(k_acc), 1.0 / INDEX_DIM)
        centered = pl.row_expand_sub(k_acc, mean)
        variance = pl.mul(pl.row_sum(pl.mul(centered, centered)), 1.0 / INDEX_DIM)
        rstd = pl.recip(pl.sqrt(pl.add(variance, K_NORM_EPS)))
        normed = pl.row_expand_mul(centered, rstd)
        weight_row = pl.cast(
            pl.reshape(k_norm_weight, [1, INDEX_DIM])[0:1, 0:INDEX_DIM], pl.FP32, mode="none"
        )
        bias_row = pl.cast(pl.reshape(k_norm_bias, [1, INDEX_DIM])[0:1, 0:INDEX_DIM], pl.FP32, mode="none")
        index_k[t0 : t0 + PROJ_T_TILE, 0:INDEX_DIM] = pl.cast(
            pl.col_expand_add(pl.col_expand_mul(normed, weight_row), bias_row),
            pl.BF16,
            mode="rint",
        )

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_weights_proj"):
        t0 = idx * PROJ_T_TILE
        w_acc = pl.create_tensor([PROJ_T_TILE, INDEX_H], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            w_acc = pl.matmul_acc(
                w_acc,
                x[t0 : t0 + PROJ_T_TILE, d0 : d0 + PROJ_K_TILE],
                w_weights[d0 : d0 + PROJ_K_TILE, 0:INDEX_H],
                init_cond=(d0 == 0),
            )
        head_weights[t0 : t0 + PROJ_T_TILE, 0:INDEX_H] = pl.mul(w_acc, WEIGHTS_SCALE)

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_gate_proj"):
        t0 = idx * PROJ_T_TILE
        g_acc = pl.create_tensor([PROJ_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            g_acc = pl.matmul_acc(
                g_acc,
                x[t0 : t0 + PROJ_T_TILE, d0 : d0 + PROJ_K_TILE],
                w_compress_gate[d0 : d0 + PROJ_K_TILE, 0:INDEX_DIM],
                init_cond=(d0 == 0),
            )
        gate_scores[t0 : t0 + PROJ_T_TILE, 0:INDEX_DIM] = g_acc


def golden_indexer_score(
    index_q: torch.Tensor,
    hadamard: torch.Tensor,
    pool_cache: torch.Tensor,
    pool_scale: torch.Tensor,
    pool_rows: torch.Tensor,
    head_weights: torch.Tensor,
    seg_start: torch.Tensor,
    pool_count: torch.Tensor,
) -> torch.Tensor:
    """Head-weighted ReLU scores of every query against the pooled table.

    The pooled table arrives already quantized — INT8 keys with their per-row
    dequant scales, written at pool close — and the Hadamard rotation and the
    query INT8 quantization mirror the kernel op for op. The reference
    ``Glm5NextTextIndexer`` scores in plain FP32, and the quantized path is the
    a2a3 deployment numerics, so the golden carries the quantization rather than
    approximating around it.
    """
    tokens = index_q.shape[0]
    pools = pool_rows.shape[0]
    scores = torch.full((tokens, pools), FP32_NEG_INF, dtype=torch.float32)
    if pools == 0 or tokens == 0:
        return scores
    rotated = index_q.float().reshape(tokens * INDEX_H, INDEX_DIM) @ hadamard.float()
    q_i8, q_scale = quantize_per_token_int8(rotated)
    gathered = pool_rows.to(torch.long)
    k_i8 = pool_cache[gathered]
    k_scale = pool_scale[gathered, 0:1]
    for token in range(tokens):
        visible = int(pool_count[token])
        if visible <= 0:
            continue
        seg0 = int(seg_start[token])
        q_rows = slice(token * INDEX_H, (token + 1) * INDEX_H)
        dots = torch.matmul(
            k_i8[seg0 : seg0 + visible].to(torch.int32),
            q_i8[q_rows].to(torch.int32).t(),
        ).float()
        dots = dots * k_scale[seg0 : seg0 + visible] * q_scale[q_rows].t()
        dots = torch.relu(dots * SOFTMAX_SCALE)
        scores[token, seg0 : seg0 + visible] = (dots * head_weights[token]).sum(dim=1)
    return scores


@pl.jit.inline
def indexer_score(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_rows: pl.Tensor[[POOLS_DYN], pl.INT32],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
):
    """Score every query row against its request's pooled keys.

    ``pool_rows`` maps each compacted pool id to its physical row in the paged
    pooled table — the host lowers paging, the kernel gathers. The table rows
    are INT8 with per-row dequant scales, quantized once at pool close, so the
    scorer only gathers them into contiguous scratch and never re-quantizes
    (donor C8 shape). The score matrix width must be a multiple of
    :data:`LEAF` with at least one leaf of slack beyond the last request's
    window, so the scorer's ``SCORE_C_TILE`` stores and the top-k's leaf loads
    never address past the tensor. Every lane outside a query's visible window
    stays at ``FP32_NEG_INF``.
    """
    t_dim = pl.tensor.dim(index_q, 0)
    pools = pl.tensor.dim(pool_rows, 0)
    q_flat = pl.reshape(index_q, [t_dim * INDEX_H, INDEX_DIM])

    qh_acc = pl.create_tensor([t_dim * INDEX_H, INDEX_DIM], dtype=pl.FP32)
    for idx in pl.spmd(
        t_dim * INDEX_H // QH_MM_TILE, name_hint="indexer_q_hadamard", allow_early_resolve=True
    ):
        r0 = idx * QH_MM_TILE
        qh_acc[r0 : r0 + QH_MM_TILE, :] = pl.matmul(
            q_flat[r0 : r0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32
        )

    q_i8 = pl.create_tensor([t_dim * INDEX_H, INDEX_DIM], dtype=pl.INT8)
    q_scale_dq = pl.create_tensor([t_dim * INDEX_H, 1], dtype=pl.FP32)
    for idx in pl.spmd(t_dim * INDEX_H // QH_MM_TILE, name_hint="indexer_q_quant", allow_early_resolve=True):
        r0 = idx * QH_MM_TILE
        qh_tile = qh_acc[r0 : r0 + QH_MM_TILE, :]
        qh_amax = pl.full([1, QH_MM_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
        qh_amax = pl.maximum(qh_amax, pl.reshape(pl.row_max(qh_abs), [1, QH_MM_TILE]))
        scale_quant_row = pl.div(pl.full([1, QH_MM_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        q_scale_dq[r0 : r0 + QH_MM_TILE, :] = pl.reshape(pl.recip(scale_quant_row), [QH_MM_TILE, 1])
        qh_scaled = pl.row_expand_mul(qh_tile, pl.reshape(scale_quant_row, [QH_MM_TILE, 1]))
        qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
        qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
        q_i8[r0 : r0 + QH_MM_TILE, :] = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")

    # One pass over the pooled table gathers the scattered INT8 rows and their
    # scales into contiguous GM scratch so the per-token score loops read plain
    # GM slices. The table is already quantized — nothing here re-quantizes.
    # Row moves are Tensor slice-assignments: a tile-level TLOAD of an INT8 row
    # hits an unsupported layout, and the donor reads its C8 cache the same way.
    pool_i8 = pl.create_tensor([pools + SCORE_C_TILE, INDEX_DIM], dtype=pl.INT8)
    pool_scale_g = pl.create_tensor([pools + SCORE_C_TILE, POOL_SCALE_WIDTH], dtype=pl.FP32)
    for c_block in pl.spmd(pools // SCORE_C_TILE, name_hint="indexer_pool_gather"):
        c0 = pl.cast(c_block, pl.INT32) * SCORE_C_TILE
        for r in pl.range(SCORE_C_TILE):
            ri = pl.cast(r, pl.INT32)
            row = pl.cast(c0 + ri, pl.INDEX)
            member = pl.cast(pl.read(pool_rows, [c0 + ri]), pl.INDEX)
            pool_i8[row : row + 1, :] = pool_cache[member : member + 1, :]
            pool_scale_g[row : row + 1, :] = pool_scale[member : member + 1, :]
    # Collapse the replicated scale lanes once, width-proportional work outside
    # the per-token loop: a strided lane-0 column read is not a legal TLOAD, and
    # reducing inside the score loop costs ~40-70 ns per (token, tile) step.
    pool_scale_col = pl.create_tensor([pools + SCORE_C_TILE, 1], dtype=pl.FP32)
    for c_block in pl.spmd(pools // SCORE_C_TILE, name_hint="indexer_pool_scale_pack"):
        c0 = pl.cast(c_block, pl.INT32) * SCORE_C_TILE
        c0_idx = pl.cast(c0, pl.INDEX)
        sc8 = pool_scale_g[c0_idx : c0_idx + SCORE_C_TILE, :]
        summed = pl.reshape(pl.row_sum(sc8), [SCORE_C_TILE, 1])
        pool_scale_col[c0_idx : c0_idx + SCORE_C_TILE, :] = pl.mul(summed, 0.125)

    for token in pl.spmd(t_dim, name_hint="indexer_score_init"):
        for c0 in pl.range(0, pools, SCORE_INIT_COLS):
            index_scores[token : token + 1, c0 : c0 + SCORE_INIT_COLS] = pl.full(
                [1, SCORE_INIT_COLS], dtype=pl.FP32, value=FP32_NEG_INF
            )

    for token in pl.spmd(t_dim, name_hint="indexer_score"):
        visible = pl.max(pl.read(pool_count, [token]), 0)
        if visible > 0:
            seg0 = pl.read(seg_start, [token])
            q_tile_i8 = q_i8[token * INDEX_H : token * INDEX_H + INDEX_H, :]
            q_scale_row = pl.reshape(q_scale_dq[token * INDEX_H : token * INDEX_H + INDEX_H, :], [1, INDEX_H])
            weight_row = head_weights[token : token + 1, 0:INDEX_H]
            for c0 in pl.range(0, visible, SCORE_C_TILE):
                kv_q_i8 = pool_i8[seg0 + c0 : seg0 + c0 + SCORE_C_TILE, :]
                kv_cache_scale_dq = pool_scale_col[seg0 + c0 : seg0 + c0 + SCORE_C_TILE, :]
                dots_i32 = pl.matmul(kv_q_i8, q_tile_i8, out_dtype=pl.INT32, b_trans=True)
                dots = pl.cast(dots_i32, target_type=pl.FP32, mode="none")
                dots = pl.row_expand_mul(dots, kv_cache_scale_dq)
                dots = pl.col_expand_mul(dots, q_scale_row)
                dots = pl.mul(dots, SOFTMAX_SCALE)
                relu = pl.maximum(dots, pl.mul(dots, 0.0))
                weighted = pl.col_expand_mul(relu, weight_row)
                row_score = pl.reshape(pl.row_sum(weighted), [1, SCORE_C_TILE])
                valid_len = pl.min(SCORE_C_TILE, visible - c0)
                masked = pl.fillpad(pl.set_validshape(row_score, 1, valid_len), pad_value=pl.PadValue.min)
                masked = pl.maximum(masked, pl.full([1, SCORE_C_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                index_scores[token : token + 1, seg0 + c0 : seg0 + c0 + SCORE_C_TILE] = masked


def golden_indexer_topk(
    index_scores: torch.Tensor,
    seg_start: torch.Tensor,
    pool_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected segment-local pool ids and which of them are real.

    A query whose visible pool count is below ``KPOOL_SELECT_K`` still fills the
    full width, so the selection carries padding. ``indexer_expand`` needs to
    know which entries are padding to blank whole four-wide groups, and the
    donor kernel emits indices only — the validity output is the addition here.
    """
    tokens = index_scores.shape[0]
    selected = torch.full((tokens, KPOOL_SELECT_K), -1, dtype=torch.int32)
    valid = torch.zeros(tokens, KPOOL_SELECT_K, dtype=torch.int32)
    for token in range(tokens):
        visible = max(int(pool_count[token]), 0)
        if visible > 0:
            seg0 = int(seg_start[token])
            keep = min(KPOOL_SELECT_K, visible)
            picked = torch.topk(index_scores[token, seg0 : seg0 + visible], keep).indices
            selected[token, :keep] = picked.to(torch.int32)
            valid[token, :keep] = 1
    return selected, valid


@pl.jit.incore
def _indexer_topk_query(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    running_pairs: pl.Tensor[[T_DYN, PAIR_WIDTH], pl.FP32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
) -> None:
    """Select one query's top visible pools, one sort leaf at a time."""
    query = pl.tile.get_block_idx()
    pl.store(pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1), [query, 0], selected_pools)
    pl.store(pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0), [query, 0], selected_valid)
    visible_count = pl.max(pl.read(pool_count, [query]), 0)
    if visible_count > 0:
        seg0 = pl.read(seg_start, [query])
        # The running top pairs round-trip through the GM scratch, the donor's
        # way of keeping one SSA type across the leaf loop.
        leaf_count = (visible_count + LEAF - 1) // LEAF
        for leaf in pl.range(leaf_count):
            leaf0 = leaf * LEAF
            leaf_valid = pl.min(LEAF, visible_count - leaf0)
            row_raw = pl.load(index_scores, [query, seg0 + leaf0], [1, LEAF], valid_shape=[1, leaf_valid])
            row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
            row = pl.maximum(row, pl.tile.full([1, LEAF], dtype=pl.FP32, value=FP32_NEG_INF))
            index_ramp = pl.tile.arange(0, [1, LEAF], dtype=pl.INT32)
            leaf_indices = pl.tile.add(index_ramp, pl.cast(leaf0, pl.INT32))
            leaf_pairs = pl.tile.sort32(row, pl.reinterpret_view(leaf_indices, pl.UINT32))
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=1024)
            leaf_pairs = pl.tile.slice(leaf_pairs, [1, PAIR_WIDTH], [0, 0])
            if leaf == 0:
                pl.store(leaf_pairs, [query, 0], running_pairs)
            else:
                pairs = pl.load(running_pairs, [query, 0], [1, PAIR_WIDTH])
                merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(pairs, leaf_pairs, tmp=merge_tmp)
                pl.store(pl.tile.slice(merged, [1, PAIR_WIDTH], [0, 0]), [query, 0], running_pairs)
        pairs = pl.load(running_pairs, [query, 0], [1, PAIR_WIDTH])
        picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        valid_topk = pl.min(visible_count, KPOOL_SELECT_K)
        indices_out = pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1)
        validity_out = pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0)
        for lane in pl.range(valid_topk):
            pl.tile.write(indices_out, [0, lane], pl.tile.read(picked, [0, lane]))
            pl.tile.write(validity_out, [0, lane], pl.const(1, pl.INT32))
        pl.store(indices_out, [query, 0], selected_pools)
        pl.store(validity_out, [query, 0], selected_valid)


@pl.jit.inline
def indexer_topk(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
):
    """Exact top-512 over each query's visible window, one block per query.

    Selections are segment-local pool ids: the sort payload carries the offset
    from ``seg_start``, so ``indexer_expand`` can map pool ``j`` to raw
    positions ``4j .. 4j+3`` without any batch knowledge.

    Returns the region's TaskId so the expansion can order against it.
    """
    t_dim = pl.tensor.dim(index_scores, 0)
    running_pairs = pl.create_tensor([t_dim, PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(t_dim, name_hint="indexer_topk") as topk_tid:
        _indexer_topk_query(
            index_scores, seg_start, pool_count, running_pairs, selected_pools, selected_valid
        )
    return topk_tid


def golden_indexer_expand(
    selected_pools: torch.Tensor,
    selected_valid: torch.Tensor,
    tail_start: torch.Tensor,
    tail_count: torch.Tensor,
    kv_len: torch.Tensor,
) -> torch.Tensor:
    """Expand the selections into front-packed raw positions, one row per query."""
    tokens = selected_pools.shape[0]
    expanded = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
    lanes = torch.arange(TOPK_INDEX_WIDTH)
    pool_raw = lanes // INDEX_KPOOL
    pool_id = pool_raw.clamp(max=selected_pools.shape[1] - 1)
    rem = lanes - pool_raw * INDEX_KPOOL
    for token in range(tokens):
        groups = min(int(kv_len[token]) // INDEX_KPOOL, KPOOL_SELECT_K)
        hist_len = groups * INDEX_KPOOL
        base = int(tail_start[token])
        tail = int(tail_count[token])
        hist_pos = selected_pools[token].to(torch.int64)[pool_id] * INDEX_KPOOL + rem
        tail_offset = lanes - hist_len
        hist_active = ((lanes < hist_len) & (selected_valid[token].to(torch.int64)[pool_id] > 0)).to(
            torch.int64
        )
        tail_active = ((tail_offset >= 0) & (tail_offset < tail)).to(torch.int64)
        positions = hist_active * (hist_pos + 1) + tail_active * (base + tail_offset + 1) - 1
        expanded[token] = positions.to(torch.int32)
    return expanded


@pl.jit.inline
def indexer_expand(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
):
    """Expand the selected pools into raw cache rows, front packed per row.

    **ABI**: ``topk_indices`` is front packed. Each row holds its valid logical
    per-request positions in its leading lanes and pads the remaining suffix
    with ``-1``; a ``-1`` never sits between two valid entries. Both sparse
    attention kernels read this as a contract rather than a convention: they
    test one lane to decide that a 128-wide block, or a whole row, carries no
    selection, so an interleaved ``-1`` would silently drop the live entries
    behind it. See :mod:`models.glm5_3_flash.decode_sparse_attn` and
    :mod:`models.glm5_3_flash.prefill_sparse_attn`.

    The prefix length is ``min(KPOOL_SELECT_K, kv_len // 4) * 4`` — selection
    order within it is free because the attention is a permutation-invariant
    softmax, so this constrains only where the padding goes. The tail pool is
    always appended right behind the selected groups, which is what keeps the
    packing tight for short contexts; the reference leaves holes there instead.
    """
    t_dim = pl.tensor.dim(selected_pools, 0)
    with pl.spmd(t_dim, name_hint="indexer_expand") as expand_tid:
        token = pl.tile.get_block_idx()
        # Scalar bookkeeping stays INT32: pl.read preserves the tensor dtype but
        # pl.min promotes to index, which scalar instructions reject.
        groups = pl.cast(pl.min(pl.read(kv_len, [token]) // INDEX_KPOOL, KPOOL_SELECT_K), pl.INT32)
        hist_len = pl.cast(groups * INDEX_KPOOL, pl.INT32)
        base = pl.cast(pl.read(tail_start, [token]), pl.INT32)
        tail = pl.cast(pl.read(tail_count, [token]), pl.INT32)
        for w0 in pl.unroll(0, TOPK_INDEX_WIDTH - (INDEX_KPOOL - 1), EXPAND_W_TILE):
            lane = pl.arange(pl.cast(w0, pl.INT32), [1, EXPAND_W_TILE], dtype=pl.INT32)
            pool_raw = pl.cast(
                pl.mul(pl.cast(lane, pl.FP32, mode="none"), 1.0 / INDEX_KPOOL),
                pl.INT32,
                mode="trunc",
            )
            pool_id = pl.minimum(pool_raw, KPOOL_SELECT_K - 1)
            sel_row = pl.gather(selected_pools[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_id)
            sel_validity = pl.gather(
                selected_valid[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_id
            )
            hist_slack = pl.neg(pl.sub(lane, hist_len))
            hist_mask = pl.minimum(pl.maximum(hist_slack, 0), 1)
            hist_active = pl.mul(hist_mask, pl.minimum(sel_validity, 1))
            tail_offset = pl.sub(lane, hist_len)
            tail_in_range = pl.minimum(pl.maximum(pl.neg(pl.sub(tail_offset, tail)), 0), 1)
            tail_ge_zero = pl.minimum(pl.maximum(pl.add(tail_offset, 1), 0), 1)
            tail_active = pl.mul(tail_in_range, tail_ge_zero)
            # Position math in FP32 for the float-constant multiplies, rounded
            # once at the store.
            lane_f = pl.cast(lane, pl.FP32, mode="none")
            pool_f = pl.cast(pool_raw, pl.FP32, mode="none")
            rem_f = pl.sub(lane_f, pl.mul(pool_f, 4.0))
            hist_pos_f = pl.add(pl.mul(pl.cast(sel_row, pl.FP32, mode="none"), 4.0), rem_f)
            tail_pos_f = pl.cast(pl.add(pl.add(tail_offset, base), 1), pl.FP32, mode="none")
            positions_f = pl.sub(
                pl.add(
                    pl.mul(pl.cast(hist_active, pl.FP32, mode="none"), pl.add(hist_pos_f, 1.0)),
                    pl.mul(pl.cast(tail_active, pl.FP32, mode="none"), tail_pos_f),
                ),
                1.0,
            )
            topk_indices[token : token + 1, w0 : w0 + EXPAND_W_TILE] = pl.cast(
                positions_f, pl.INT32, mode="none"
            )
        # The odd 2051-wide row cannot end on an aligned tile, so the last chunk
        # is an eight-lane tile starting five lanes early; the overlap is written
        # twice with identical values, which keeps every store on the MTE3 path.
        lane_t = pl.arange(pl.const(2043, pl.INT32), [1, EXPAND_TAIL_CHUNK], dtype=pl.INT32)
        pool_t = pl.cast(
            pl.mul(pl.cast(lane_t, pl.FP32, mode="none"), 1.0 / INDEX_KPOOL),
            pl.INT32,
            mode="trunc",
        )
        pool_t = pl.minimum(pool_t, KPOOL_SELECT_K - 1)
        sel_t = pl.gather(selected_pools[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_t)
        valid_t = pl.gather(selected_valid[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_t)
        hslack_t = pl.neg(pl.sub(lane_t, hist_len))
        hmask_t = pl.minimum(pl.maximum(hslack_t, 0), 1)
        hactive_t = pl.mul(hmask_t, pl.minimum(valid_t, 1))
        offset_t = pl.sub(lane_t, hist_len)
        trange_t = pl.minimum(pl.maximum(pl.neg(pl.sub(offset_t, tail)), 0), 1)
        tge_t = pl.minimum(pl.maximum(pl.add(offset_t, 1), 0), 1)
        tactive_t = pl.mul(trange_t, tge_t)
        lane_tf = pl.cast(lane_t, pl.FP32, mode="none")
        pool_tf = pl.cast(pool_t, pl.FP32, mode="none")
        rem_tf = pl.sub(lane_tf, pl.mul(pool_tf, 4.0))
        hpos_tf = pl.add(pl.mul(pl.cast(sel_t, pl.FP32, mode="none"), 4.0), rem_tf)
        tpos_tf = pl.cast(pl.add(pl.add(offset_t, base), 1), pl.FP32, mode="none")
        pos_tf = pl.sub(
            pl.add(
                pl.mul(pl.cast(hactive_t, pl.FP32, mode="none"), pl.add(hpos_tf, 1.0)),
                pl.mul(pl.cast(tactive_t, pl.FP32, mode="none"), tpos_tf),
            ),
            1.0,
        )
        topk_indices[
            token : token + 1,
            TOPK_INDEX_WIDTH - EXPAND_TAIL_CHUNK : TOPK_INDEX_WIDTH,
        ] = pl.cast(pos_tf, pl.INT32, mode="none")
    return expand_tid


@pl.jit
def indexer_proj_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    index_q: pl.Out[pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16]],
    head_weights: pl.Out[pl.Tensor[[T_DYN, INDEX_H], pl.FP32]],
    gate_scores: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32]],
):
    """Run the four projections for golden.run validation."""
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    index_q.bind_dynamic(0, T_DYN)
    index_k.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_norm_weight,
        k_norm_bias,
        w_weights,
        w_compress_gate,
        index_q,
        index_k,
        head_weights,
        gate_scores,
    )
    return index_q, index_k, head_weights, gate_scores


@pl.jit
def indexer_score_test(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_rows: pl.Tensor[[POOLS_DYN], pl.INT32],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    index_scores: pl.Out[pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32]],
):
    """Score one packed batch for golden.run validation."""
    index_q.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    pool_rows.bind_dynamic(0, POOLS_DYN)
    index_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)
    pool_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    indexer_score(
        index_q,
        hadamard,
        pool_cache,
        pool_scale,
        pool_rows,
        head_weights,
        seg_start,
        pool_count,
        index_scores,
    )
    return index_scores


@pl.jit
def indexer_topk_test(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
):
    """Select the top pools for one batch for golden.run validation."""
    index_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    indexer_topk(index_scores, seg_start, pool_count, selected_pools, selected_valid)
    return selected_pools, selected_valid


@pl.jit
def indexer_expand_test(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    """Expand one batch of selections for golden.run validation."""
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    tail_start.bind_dynamic(0, T_DYN)
    tail_count.bind_dynamic(0, T_DYN)
    kv_len.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    indexer_expand(selected_pools, selected_valid, tail_start, tail_count, kv_len, topk_indices)
    return topk_indices


def sylvester_hadamard(head_dim: int = INDEX_DIM) -> torch.Tensor:
    """The scaled Sylvester Hadamard the load-time converter builds."""
    matrix = torch.ones((1, 1))
    while matrix.shape[0] < head_dim:
        matrix = torch.cat([torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0)
    return (matrix * (head_dim**-0.5)).to(torch.bfloat16)


def build_indexer_proj_specs(tokens: int = 48):
    """Build one deterministic projection batch with all four weight groups."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(67)

    def init_x():
        return torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16()

    def init_q_resid():
        return torch.randn(tokens, Q_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_w_q_b():
        weight = torch.randn(Q_LORA, INDEX_H * INDEX_DIM, generator=generator) * 0.02
        return weight.bfloat16()

    def init_w_k():
        return (torch.randn(D, INDEX_DIM, generator=generator) * 0.02).bfloat16()

    def init_k_norm_weight():
        return (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()

    def init_k_norm_bias():
        return (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16()

    def init_w_weights():
        return (torch.randn(D, INDEX_H, generator=generator) * 0.05).bfloat16()

    def init_w_compress_gate():
        return (torch.randn(D, INDEX_DIM, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=init_q_resid),
        TensorSpec("w_q_b", [Q_LORA, INDEX_H * INDEX_DIM], torch.bfloat16, init_value=init_w_q_b),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_w_k),
        TensorSpec("k_norm_weight", [INDEX_DIM], torch.bfloat16, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [INDEX_DIM], torch.bfloat16, init_value=init_k_norm_bias),
        TensorSpec("w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_w_weights),
        TensorSpec("w_compress_gate", [D, INDEX_DIM], torch.bfloat16, init_value=init_w_compress_gate),
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16),
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32),
    ]


def _pool_batch(tokens: int, counts: tuple[int, ...], positions: tuple[int, ...]):
    """Derive the packed score/topk batch geometry from per-request pool counts.

    Returns the compacted width (leaf-aligned with slack), the pooled-table row
    count, the per-query ``seg_start`` and causally clamped ``pool_count``, and
    the pool row indirection. ``positions`` is each query's request-local
    position.
    """
    pools_true = sum(counts)
    width = ((pools_true + 2 * LEAF - 2) // LEAF) * LEAF
    per_request = tokens // len(counts)
    if len(positions) != tokens or per_request * len(counts) != tokens:
        raise ValueError("positions must cover every token; counts must tile the batch")
    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    for index, count in enumerate(counts):
        for row in range(per_request):
            token = index * per_request + row
            seg_start[token] = sum(counts[:index])
            pool_count[token] = min((positions[token] + 1) // INDEX_KPOOL, count)
    pool_table_rows = ((pools_true + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE) * 4
    generator = torch.Generator().manual_seed(71)
    pool_rows = torch.zeros(width, dtype=torch.int32)
    pool_rows[:pools_true] = torch.randperm(pool_table_rows, generator=generator)[:pools_true]
    return width, pool_table_rows, seg_start, pool_count, pool_rows


def build_indexer_score_specs(
    tokens: int = 8,
    counts: tuple[int, ...] | None = None,
    positions: tuple[int, ...] | None = None,
):
    """Build one deterministic two-request score batch.

    Request 0 owns 3000 pools so its longest query spans two sort leaves, and
    the query positions ramp from a first token (visible count 0) to a full
    window, so the causal clamp and the leaf-crossing window both execute.
    ``counts``/``positions`` override the fixture geometry for the
    business-shape benchmark points: one request whose positions are the
    ``history`` ramp in front of the measured chunk.
    """
    from golden import TensorSpec

    counts = (3000, 140) if counts is None else counts
    positions = (2, 6, 5000, 11999, 3, 7, 500, 559) if positions is None else positions
    width, pool_table_rows, seg_start, pool_count, pool_rows = _pool_batch(tokens, counts, positions)

    generator = torch.Generator().manual_seed(73)

    def init_index_q():
        return torch.randn(tokens, INDEX_H, INDEX_DIM, generator=generator).bfloat16()

    # One draw feeds both tables so every INT8 row and its scale stay paired.
    prior_i8, prior_scale_col = quantize_per_token_int8(
        torch.randn(pool_table_rows, INDEX_DIM, generator=generator, dtype=torch.float32)
    )

    def init_pool_cache():
        return prior_i8

    def init_pool_scale():
        # The writer replicates the scale across all POOL_SCALE_WIDTH lanes and
        # the scorer reads the row sum scaled by 1/8, so the prior table must
        # carry the replication too.
        return prior_scale_col.expand(-1, POOL_SCALE_WIDTH).contiguous().clone()

    def init_head_weights():
        return torch.randn(tokens, INDEX_H, generator=generator)

    return [
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16, init_value=init_index_q),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec("pool_cache", [pool_table_rows, INDEX_DIM], torch.int8, init_value=init_pool_cache),
        TensorSpec(
            "pool_scale", [pool_table_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale
        ),
        TensorSpec("pool_rows", [width], torch.int32, init_value=lambda: pool_rows),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32, init_value=init_head_weights),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("index_scores", [tokens, width], torch.float32),
    ]


def build_indexer_topk_specs(
    tokens: int = 8,
    counts: tuple[int, ...] | None = None,
    positions: tuple[int, ...] | None = None,
):
    """Score-and-select on one batch: synthetic scores, real windows.

    The score rows are random values inside each query's window and
    ``FP32_NEG_INF`` outside it, which is exactly the shape :func:`indexer_score`
    leaves behind, so the sort sees realistic value structure without depending
    on the scorer's numerics. ``counts``/``positions`` override the fixture
    geometry for the business-shape benchmark points.
    """
    from golden import TensorSpec

    from models.glm5_3_flash.config import FP32_NEG_INF as NEG_INF

    counts = (3000, 140) if counts is None else counts
    positions = (2, 6, 5000, 11999, 3, 7, 500, 559) if positions is None else positions
    width, _, seg_start, pool_count, _ = _pool_batch(tokens, counts, positions)

    generator = torch.Generator().manual_seed(79)

    def init_index_scores():
        rows = torch.rand(tokens, width, generator=generator) * 10.0
        mask = torch.full((tokens, width), NEG_INF)
        for token in range(tokens):
            seg0 = int(seg_start[token])
            mask[token, seg0 : seg0 + int(pool_count[token])] = 1.0
        return rows * mask

    return [
        TensorSpec("index_scores", [tokens, width], torch.float32, init_value=init_index_scores),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("selected_pools", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("selected_valid", [tokens, KPOOL_SELECT_K], torch.int32),
    ]


def build_indexer_expand_specs(tokens: int = 8, lengths: tuple[int, ...] | None = None):
    """Expand selections across short rows, partial groups and a full row.

    The lengths ramp from a context too short to close one pool to one that
    fills all 512 selections, so the packing suffix, the tail window and the
    full-prefix case all execute in one batch. ``lengths`` overrides the ramp
    for the business-shape benchmark points (one full-width row per token).
    """
    from golden import TensorSpec

    lengths = (2, 3, 6, 15, 63, 1039, 2048, 2051) if lengths is None else lengths
    if len(lengths) != tokens:
        raise ValueError("lengths must cover every token")
    generator = torch.Generator().manual_seed(83)

    def init_selected_pools():
        picked = torch.randint(0, 64, (tokens, KPOOL_SELECT_K), generator=generator)
        return picked.to(torch.int32)

    def init_selected_valid():
        valid = torch.zeros(tokens, KPOOL_SELECT_K, dtype=torch.int32)
        for token, length in enumerate(lengths):
            keep = min(KPOOL_SELECT_K, length // INDEX_KPOOL)
            valid[token, :keep] = 1
        return valid

    def init_tail_start():
        return torch.tensor([length - length % INDEX_KPOOL for length in lengths], dtype=torch.int32)

    def init_tail_count():
        return torch.tensor([length % INDEX_KPOOL for length in lengths], dtype=torch.int32)

    def init_kv_len():
        return torch.tensor(lengths, dtype=torch.int32)

    return [
        TensorSpec(
            "selected_pools",
            [tokens, KPOOL_SELECT_K],
            torch.int32,
            init_value=init_selected_pools,
        ),
        TensorSpec(
            "selected_valid",
            [tokens, KPOOL_SELECT_K],
            torch.int32,
            init_value=init_selected_valid,
        ),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=init_tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=init_tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=init_kv_len),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32),
    ]


def golden_indexer_proj_case(tensors):
    """Fill the expected projections for :func:`build_indexer_proj_specs`."""
    index_q, index_k, head_weights, gate_scores = golden_indexer_proj(
        tensors["x"],
        tensors["q_resid"],
        tensors["w_q_b"],
        tensors["w_k"],
        tensors["k_norm_weight"],
        tensors["k_norm_bias"],
        tensors["w_weights"],
        tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores


def golden_indexer_score_case(tensors):
    """Fill the expected scores for :func:`build_indexer_score_specs`."""
    tensors["index_scores"][:] = golden_indexer_score(
        tensors["index_q"],
        tensors["hadamard"],
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["pool_rows"],
        tensors["head_weights"],
        tensors["seg_start"],
        tensors["pool_count"],
    )


def golden_indexer_topk_case(tensors):
    """Fill the expected selections for :func:`build_indexer_topk_specs`."""
    selected, valid = golden_indexer_topk(
        tensors["index_scores"], tensors["seg_start"], tensors["pool_count"]
    )
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid


def golden_indexer_expand_case(tensors):
    """Fill the expected expansion for :func:`build_indexer_expand_specs`."""
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"],
        tensors["selected_valid"],
        tensors["tail_start"],
        tensors["tail_count"],
        tensors["kv_len"],
    )


def _self_check() -> None:
    """Exercise the four goldens on CPU with inlined fixtures.

    Kept local rather than in ``_golden_smoke.py`` because that module is shared
    across the whole model directory.
    """
    torch.manual_seed(85)
    heads, dim, hidden = 4, 16, 64
    x = torch.randn(8, hidden).bfloat16()
    q_resid = torch.randn(8, 32).bfloat16()
    w_q_b = torch.randn(32, heads * dim).bfloat16() * 0.1
    w_k = torch.randn(hidden, dim).bfloat16() * 0.1
    k_w = torch.randn(dim).bfloat16()
    k_b = torch.randn(dim).bfloat16()
    index_q, index_k, _, _ = golden_indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_w,
        k_b,
        torch.randn(hidden, heads).bfloat16() * 0.1,
        torch.randn(hidden, dim).bfloat16() * 0.1,
    )
    assert index_q.shape == (8, heads, dim), index_q.shape
    assert index_k.dtype is torch.bfloat16
    manual_q = torch.nn.functional.linear(q_resid.float(), w_q_b.t().float())
    assert torch.allclose(index_q.float().flatten(1), manual_q, atol=2e-2), "q projection drifted"
    raw = torch.nn.functional.linear(x.float(), w_k.t().float())
    manual_k = torch.nn.functional.layer_norm(raw, (dim,), k_w.float(), k_b.float(), K_NORM_EPS)
    assert torch.allclose(index_k.float(), manual_k.to(torch.bfloat16).float(), atol=2e-2)

    scores = torch.tensor([[9.0, 9.0, 5.0, 1.0, 3.0, 0.0], [0.0, 0.0, 9.0, 1.0, 5.0, 1.0]])
    seg = torch.tensor([0, 2], dtype=torch.int32)
    count = torch.tensor([3, 4], dtype=torch.int32)
    selected, valid = golden_indexer_topk(scores, seg, count)
    assert valid[:, 0].tolist() == [1, 1] and valid[0, 3].item() == 0
    assert selected[0, 0] in (0, 1) and selected[0, 2] == 2, selected[0]

    picked = torch.tensor([[2, 1, 0, 0]], dtype=torch.int32)
    picked_valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.int32)
    tail_start = torch.tensor([12], dtype=torch.int32)
    tail_count = torch.tensor([2], dtype=torch.int32)
    kv = torch.tensor([14], dtype=torch.int32)
    expanded = golden_indexer_expand(picked, picked_valid, tail_start, tail_count, kv)
    expected = [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13]
    assert expanded[0, :14].tolist() == expected, expanded[0, :14]
    assert (expanded[0, 14:] == -1).all(), "padding is not a pure suffix"
    print("[GOLDEN] PASS prefill_indexer self-check")


def main():
    """Prove the goldens on CPU, then validate each stage on device.

    The projections round once to BF16 or hold FP32, so their budget is one
    BF16 ulp with a small outlier allowance. The scorer carries the INT8
    quantization of both operands, so a tie-swapped top-k is judged through the
    paired-score comparator rather than index equality. The expansion is pure
    integer lane math and compares exactly.
    """
    import argparse

    from golden import ratio_allclose, run, topk_pair_compare

    _self_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", default="all", choices=["proj", "score", "topk", "expand", "all"])
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run the case at a business shape without golden validation; the shape "
        "comes from --tokens/--history, timing from PYPTO_BENCH=1",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=0,
        help="history tokens in front of the --tokens chunk; defines the bench geometry",
    )
    args = parser.parse_args()

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} "
            f"min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} "
            f"max={stats.device_us_max:.1f} rounds={stats.rounds}"
        )

    def bench_label(name: str, pools: int = 0) -> str:
        label = f"{name} T={args.tokens} history={args.history}"
        if pools:
            width = ((pools + 2 * LEAF - 2) // LEAF) * LEAF
            label += f" pools={pools} width={width}"
        return label

    bench_counts = None
    bench_positions = None
    if args.bench:
        pools = (args.tokens + args.history) // INDEX_KPOOL
        bench_counts = (pools,)
        bench_positions = tuple(range(args.history, args.history + args.tokens))
        if args.tokens % 16:
            parser.error("--bench needs a token count that is a multiple of 16")

    def selected_pools_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        scores = inputs["index_scores"].float()
        seg0 = inputs["seg_start"].long().unsqueeze(1)
        invalid = actual < 0
        cols = (actual.long() + seg0).clamp(0, scores.shape[-1] - 1)
        paired = torch.gather(scores, 1, cols)
        paired = torch.where(invalid, torch.full_like(paired, -torch.inf), paired)
        synth_outputs = {**actual_outputs, "_selected_paired_scores": paired}
        return topk_pair_compare("_selected_paired_scores")(
            actual,
            expected,
            actual_outputs=synth_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    def exact_compare(actual, expected, **_kwargs):
        exact = torch.equal(actual.cpu(), expected.cpu())
        return exact, "" if exact else "    integer output differs from golden"

    results = []
    if args.case in ("proj", "all"):
        if args.bench:
            extra = dict(golden_fn=None)
            label = bench_label("proj")
        else:
            extra = dict(
                golden_fn=golden_indexer_proj_case,
                rtol=1.0 / 64,
                atol=1e-3,
                compare_fn={
                    "index_q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
                    "index_k": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
                    "head_weights": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
                    "gate_scores": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
                },
            )
        results.append(
            run(
                fn=indexer_proj_test,
                specs=build_indexer_proj_specs(args.tokens),
                config={"platform": args.platform, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("score", "all"):
        if args.bench:
            specs = build_indexer_score_specs(args.tokens, bench_counts, bench_positions)
            extra = dict(golden_fn=None)
            label = bench_label("score", bench_counts[0])
        else:
            specs = build_indexer_score_specs()
            extra = dict(
                golden_fn=golden_indexer_score_case,
                rtol=1.0 / 128,
                atol=1e-4,
                compare_fn={
                    "index_scores": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
                },
            )
        results.append(
            run(
                fn=indexer_score_test,
                specs=specs,
                config={"platform": args.platform, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("topk", "all"):
        if args.bench:
            specs = build_indexer_topk_specs(args.tokens, bench_counts, bench_positions)
            extra = dict(golden_fn=None)
            label = bench_label("topk", bench_counts[0])
        else:
            specs = build_indexer_topk_specs()
            extra = dict(
                golden_fn=golden_indexer_topk_case,
                rtol=1e-3,
                atol=1e-3,
                compare_fn={
                    "selected_pools": selected_pools_compare,
                    "selected_valid": exact_compare,
                },
            )
        results.append(
            run(
                fn=indexer_topk_test,
                specs=specs,
                config={"platform": args.platform, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    if args.case in ("expand", "all"):
        if args.bench:
            specs = build_indexer_expand_specs(args.tokens, (args.tokens + args.history,) * args.tokens)
            extra = dict(golden_fn=None)
            label = bench_label("expand")
        else:
            specs = build_indexer_expand_specs()
            extra = dict(
                golden_fn=golden_indexer_expand_case,
                rtol=0.0,
                atol=0.0,
            )
        results.append(
            run(
                fn=indexer_expand_test,
                specs=specs,
                config={"platform": args.platform, "device_id": args.device},
                compile_only=args.compile_only,
                **extra,
            )
        )
        if args.bench:
            print_bench(label, results[-1])
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "LEAF",
    "PAIR_WIDTH",
    "build_indexer_expand_specs",
    "build_indexer_proj_specs",
    "build_indexer_score_specs",
    "build_indexer_topk_specs",
    "golden_indexer_expand",
    "golden_indexer_proj",
    "golden_indexer_score",
    "golden_indexer_topk",
    "indexer_expand",
    "indexer_expand_test",
    "indexer_proj",
    "indexer_proj_test",
    "indexer_score",
    "indexer_score_test",
    "indexer_topk",
    "indexer_topk_test",
    "sylvester_hadamard",
]


if __name__ == "__main__":
    main()
