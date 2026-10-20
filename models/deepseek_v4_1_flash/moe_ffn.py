# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 routed-expert MoE FFN (gate-after, single card).

Scope: only the routed-expert FFN compute core lives here —
activation gather + dynamic MXFP8 requant, GMM1 (one W13 N=4608 GEMM
over the merged W1/W3 weights), SwiGLU with
route-weight application + requant, GMM2 (W2), and the top-k weighted
aggregation. The MoE orchestration (``moe.py``), the router (``gate.py``),
the EP dispatch/combine transport, and the shared expert
(``expert_shared.py``) are delivered elsewhere and stay outside this file;
this kernel starts directly from the gate's outputs and is validatable on
one device.

Primary deployment shape: single card, b=12 with s1=6 decode rows per
request (T=72, high-throughput decode). The low-latency shape is b=1
(T=6); pass ``--batch 1``. In this regime most of the 384 routed experts
see 0-2 rows, so the kernel is weight-bandwidth bound: MXFP4-packed
weights stay in GM and are dequantized to MXFP8 in-kernel through the
two-lane byte LUT + gather path (see ``expert_routed.py``).

Inputs (the gate-after ABI):
  - ``x_normed``     [T_PAD, D] BF16       router-normed activations
  - ``topk_indices`` [T, TOPK] INT32       per-token expert ids in [0, E)
  - ``topk_weights`` [T, TOPK] FP32        normalized route weights (x1.5)
  - ``routed_w13_packed`` [E, 92160, 128] UINT8  MXFP4 Cube-lane tiles,
    per-expert row concatenation of the W1 and W3 tile blocks (W1 first)
  - ``routed_w13_scale`` [E, K_SCALE, 4608] E8M0, MX_B_NN physical order,
    gate scale columns [0, 2304) and up scale columns [2304, 4608)
  - ``routed_w2_packed`` / ``routed_w2_scale``  W2, same tile form
  - ``mxfp4_pair_lut`` [2, 256] INT16      AIV FP4->FP8 pair decoder table
Output:
  - ``out`` [T, D] BF16 = sum_topk routed FFN (FP32 accumulate, one rint)

Device-side grouping (histogram + prefix + scatter over T*TOPK routes)
turns the gate's per-token routing into per-expert compact row lists, so
no host-side dispatch artifacts enter the ABI. In an EP deployment those
lists are exactly what the dispatch stage hands over; here they are built
in-kernel from ``topk_indices``.

Run this file directly to validate on one device (default ``-p a5``).
"""
import sys

import pypto.language as pl

from config import FLASH as M


def _parse_cli_int(name: str, default: int) -> int:
    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


# Deployment shape: b=12 requests x s1=6 decode rows (primary), b=1 (low latency).
BATCH = _parse_cli_int("batch", 12)
S1 = 6
T = BATCH * S1

# model config
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
# Routed experts resident on THIS card. The full table is 384 (single-card
# validation); the 32-card high-throughput deployment (GBS=384 -> 12
# requests/card x 6 DSpark verify rows = T=72) shards it EP32 -> 12
# experts/card. Dispatch then delivers only this card's rows; balanced
# routing gives the same 432 rows at T=72, so E=12 T=72 reproduces the
# EP32 per-card compute core. Import-time knob: --experts.
E = _parse_cli_int("experts", M.n_routed_experts)
TOPK = M.num_experts_per_tok

T_PAD = ((T + 15) // 16) * 16
TOTAL = T * TOPK
# Activation/hidden staging is chunked per expert: one expert's rows are
# processed in CHUNK_ROWS-row windows (whole 16-row tiles), each window in
# its own pl.scope so the ring heap recycles it. Staging offsets stay
# window-relative (0 .. CHUNK_ROWS), which keeps every MX_A_ZZ scale view
# provably 16-aligned without a dynamic per-expert base.
#
# RECV_TILE=64: one xgather/GMM1/act/GMM2 instance serves the whole
# 64-row chunk. The GMM instances are weight-load bound, so a 16-row tile
# re-read the expert's full weights once per tile (3x at 36 rows/expert)
# and tripled the AICPU submit count; one instance per chunk reads them
# once. The AIV quant kernels still process the window in AIV_ROW_TILE-row
# sub-tiles to bound UB; every MX_A_ZZ scale atom stays 16-row aligned.
RECV_TILE = 64
AIV_ROW_TILE = 16
CHUNK_ROWS = 64
CHUNK_TILES = (CHUNK_ROWS + RECV_TILE - 1) // RECV_TILE
MAX_TILES_PER_EXPERT = (TOTAL + RECV_TILE - 1) // RECV_TILE
# Expert-grouped GMM2 output staging (gather-reduce form): every expert's
# rows land in one RECV_TILE-padded region of ``y_grouped`` so each whole-
# tile fix-pipe store stays inside its own expert's rows; the reduce then
# gathers each token's TOPK rows back through the route -> padded-row
# inverse map. Worst case: TOTAL rows plus RECV_TILE-1 of padding per
# non-empty expert (at most min(TOTAL, E) of them).
Y_ROWS = TOTAL + (RECV_TILE - 1) * min(TOTAL, E)

# tiling (mirrors expert_routed.py so the packed weight ABI is identical)
MX_GROUP = 32
K_SCALE = D // MX_GROUP
H_SCALE = MOE_INTER // MX_GROUP
MX_K_TILE = 256
MX_MM_INTER_TILE = 256
MX_K_SCALE_GROUPS = MX_K_TILE // MX_GROUP
MX_MM_TASK_TILE = 768
MX_W2_D_OUT_TILE = 256
MX_W2_TASK_TILE = 1024
MX_PACKED_LANE_ROWS = MX_K_TILE // 2
MX_PACKED_LANE_COLS = MX_MM_INTER_TILE // 2
MX_W1_PACKED_TILES = 2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)
MX_W3_PACKED_TILES = 2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)
MX_W2_PACKED_TILES = 2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE)
MX_W1_PACKED_ROWS = MX_W1_PACKED_TILES * MX_PACKED_LANE_ROWS
MX_W3_PACKED_ROWS = MX_W3_PACKED_TILES * MX_PACKED_LANE_ROWS
MX_W2_PACKED_ROWS = MX_W2_PACKED_TILES * MX_PACKED_LANE_ROWS
# W13 ABI: per expert the W1 and W3 packed tile rows are concatenated
# (W1 first) into one carrier so GMM1 is a single N=2*MOE_INTER matmul.
MX_W13_PACKED_TILES = MX_W1_PACKED_TILES + MX_W3_PACKED_TILES
MX_W13_PACKED_ROWS = MX_W13_PACKED_TILES * MX_PACKED_LANE_ROWS
assert MOE_INTER % MX_MM_TASK_TILE == 0
assert D % MX_K_TILE == 0
assert MX_MM_TASK_TILE % MX_MM_INTER_TILE == 0
assert D % MX_W2_TASK_TILE == 0
assert MX_W2_TASK_TILE % MX_W2_D_OUT_TILE == 0

# input activation requant tiling (mirrors gate.py)
XQ_TILE = 256
XQ_TASK_TILE = XQ_TILE * 4
XQ_GROUPS = XQ_TILE // MX_GROUP
XQ_SCALE_TMP = ((64 + (AIV_ROW_TILE // 16) * XQ_GROUPS + 31) // 32) * 32
assert D % XQ_TASK_TILE == 0

# SwiGLU requant tiling (mirrors expert_routed.py)
ACT_INTER_TILE = 256
QUANT_TILE = 768
ACT_GATE_INNER = QUANT_TILE // ACT_INTER_TILE
assert MOE_INTER % QUANT_TILE == 0
SWIGLU_GROUPS = QUANT_TILE // MX_GROUP
SWIGLU_SCALE_TMP = ((64 + (AIV_ROW_TILE // 16) * SWIGLU_GROUPS + 31) // 32) * 32

# top-k aggregation tiling: one AIV block per token. RED_TILE=512 spawned
# 720 tiny blocks whose ~0.53us/block serialized AICPU completion handling
# made moe_ffn_reduce a 380us tail (21% of the E=12 window); 72 blocks
# keep the whole aggregation off the critical path.
RED_TILE = 5120
assert D % RED_TILE == 0

IDX_PAD = 8
AUX_PAD = 8

# Across-layer-mean dequant std (typical layer) of the real routed experts.
ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}


@pl.jit.inline(auto_scope=False)
def moe_ffn_compute(
    x_normed: pl.Tensor[[T_PAD, D], pl.BF16],
    topk_indices: pl.Tensor[[T, TOPK], pl.INT32],
    topk_weights: pl.Tensor[[T, TOPK], pl.FP32],
    routed_w13_packed: pl.Tensor[
        [E, MX_W13_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8,
    ],
    routed_w13_scale: pl.Tensor[
        [E * K_SCALE, 2 * MOE_INTER], pl.FP8E8M0, pl.MX_B_NN,
    ],
    routed_w2_packed: pl.Tensor[
        [E, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8,
    ],
    routed_w2_scale: pl.Tensor[[E * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    out: pl.Tensor[[T, D], pl.BF16],
):
    input_ready = pl.system.task_dummy(deps=[])

    # ------------------------------------------------------------------
    # Stage 0 (moe_ffn_group, one CORE_GROUP task) groups T*TOPK routes
    # into per-expert compact row lists; stages 1-4 then run the per-expert
    # gather + requant, GMM1, SwiGLU + requant. One shared scope keeps the
    # metadata tensors alive for every consumer.
    # ------------------------------------------------------------------
    with pl.scope():
        expert_count = pl.create_tensor([E, 1], dtype=pl.INT32)
        list_base = pl.create_tensor([E, 1], dtype=pl.INT32)
        padded_base = pl.create_tensor([E, 1], dtype=pl.INT32)
        tok_ids = pl.create_tensor([TOTAL + RECV_TILE, 1], dtype=pl.INT32)
        w_compact = pl.create_tensor([TOTAL + RECV_TILE, 1], dtype=pl.FP32)
        route_pos = pl.create_tensor([TOTAL, 1], dtype=pl.INT32)
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="moe_ffn_group", deps=[input_ready],
        ) as group_tid:
            counts = pl.array.create(E, pl.INT32)
            for e in pl.range(E):
                counts[e] = 0
            indices_tile = pl.tile.load(
                topk_indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK],
            )
            weights_tile = pl.tile.load(
                topk_weights, [0, 0], [T, AUX_PAD], valid_shape=[T, TOPK],
            )
            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.tile.read(indices_tile, [t, k])
                    counts[eid] = counts[eid] + 1
            compact = pl.const(0, pl.INT32)
            padded = pl.const(0, pl.INT32)
            for e in pl.range(E):
                c = counts[e]
                pl.write(list_base, [e, 0], compact)
                pl.write(padded_base, [e, 0], padded)
                compact = pl.cast(compact + c, pl.INT32)
                padded = pl.cast(
                    padded + ((c + RECV_TILE - 1) // RECV_TILE) * RECV_TILE,
                    pl.INT32,
                )
                counts[e] = 0
            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.tile.read(indices_tile, [t, k])
                    pos = pl.read(list_base, [eid, 0]) + counts[eid]
                    pl.write(tok_ids, [pos, 0], pl.cast(t, pl.INT32))
                    pl.write(w_compact, [pos, 0], pl.tile.read(weights_tile, [t, k]))
                    pl.write(
                        route_pos,
                        [t * TOPK + k, 0],
                        pl.read(padded_base, [eid, 0]) + counts[eid],
                    )
                    counts[eid] = counts[eid] + 1
            for r in pl.range(RECV_TILE):
                pl.write(w_compact, [TOTAL + r, 0], pl.const(0.0, pl.FP32))
                pl.write(tok_ids, [TOTAL + r, 0], pl.const(0, pl.INT32))
            for e in pl.range(E):
                pl.write(expert_count, [e, 0], counts[e])

        # manual_dep: every expert's GMM2 writes only its own padded region,
        # so the tasks need no ordering between each other. Auto OverlapMap
        # tracking would chain the 12 per-expert GMM2 waves WAW-serial on this
        # one shared tensor (measured: window 1374us, AIC occupancy 24%);
        # correctness comes from the reduce's explicit expert-completion deps.
        y_grouped = pl.create_tensor(
            [Y_ROWS, D], dtype=pl.BF16, manual_dep=True,
        )
        expert_completion_tids = pl.array.create(E, pl.TASK_ID)

        # ------------------------------------------------------------------
        # Stages 1-6, fused per expert: W1/W3/W2 in-kernel FP4->FP8 dequant
        # once per expert, then per 64-row chunk: gather + requant, GMM1
        # (fix-pipe drains BF16), SwiGLU + requant, GMM2 (fix-pipe BF16)
        # whose epilogue stores whole tiles contiguously straight into the
        # expert's padded region of y_grouped — no row-transpose kernel.
        # ------------------------------------------------------------------
        for local_e in pl.parallel(E):
            tile_completion_tids = pl.array.create(MAX_TILES_PER_EXPERT, pl.TASK_ID)
            for tile in pl.range(MAX_TILES_PER_EXPERT):
                tile_completion_tids[tile] = pl.system.task_dummy(deps=[])
            n_rows = pl.read(expert_count, [local_e, 0])
            routed_w13_packed_e = routed_w13_packed[local_e, :, :]
            routed_w2_packed_e = routed_w2_packed[local_e, :, :]
            if n_rows > 0:
                lbase = pl.cast(pl.read(list_base, [local_e, 0]), pl.INDEX)
                pbase = pl.cast(pl.read(padded_base, [local_e, 0]), pl.INDEX)
                with pl.scope():
                    # W13 staging: one continuous [D, 2*MOE_INTER] FP8
                    # buffer. The gate dequant fills columns [0, MOE_INTER)
                    # from the carrier's W1 tile rows, the up dequant fills
                    # [MOE_INTER, 2*MOE_INTER) from the W3 tile rows - both
                    # through static per-task bases. manual_dep keeps the two
                    # dequant waves concurrent (disjoint column ranges; the
                    # merged GMM1 below depends on both completions).
                    w13_mxfp8 = pl.create_tensor(
                        [D, 2 * MOE_INTER], dtype=pl.FP8E4M3FN, manual_dep=True,
                    )
                    with pl.spmd(
                        2 * (MOE_INTER // MX_MM_TASK_TILE),
                        name_hint="ffn_gate_mxfp4_aiv",
                        deps=[input_ready],
                    ) as w1_deq_tid:
                        bidx_d1 = pl.tile.get_block_idx()
                        nb_d1 = bidx_d1 // 2
                        w1_lane = bidx_d1 % 2
                        nbase_d1 = nb_d1 * MX_MM_TASK_TILE
                        w1_lut_tile = pl.load(
                            mxfp4_pair_lut, [w1_lane, 0], [1, 256],
                            target_memory=pl.Mem.Vec,
                        )
                        for ng in pl.range(MX_MM_TASK_TILE // MX_MM_INTER_TILE):
                            n0 = nbase_d1 + ng * MX_MM_INTER_TILE
                            for k0 in pl.range(0, D, MX_K_TILE):
                                w1_tile_id = (
                                    (n0 // MX_MM_INTER_TILE) * (D // MX_K_TILE)
                                    + k0 // MX_K_TILE
                                )
                                w1_packed_row = (
                                    w1_tile_id * 2 * MX_PACKED_LANE_ROWS
                                    + w1_lane * MX_PACKED_LANE_ROWS
                                )
                                w1_packed_tile = pl.load(
                                    routed_w13_packed_e,
                                    [w1_packed_row, 0],
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    target_memory=pl.Mem.Vec,
                                )
                                w1_packed_u16 = pl.cast(w1_packed_tile, target_type=pl.UINT16)
                                w1_packed_indices = pl.reinterpret_view(w1_packed_u16, pl.INT16)
                                w1_gather_tmp = pl.create_tile(
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    dtype=pl.INT16, target_memory=pl.Mem.Vec,
                                )
                                w1_pair_codes = pl.tile.gather(
                                    w1_lut_tile, w1_packed_indices, w1_gather_tmp,
                                )
                                w1_bytes = pl.reinterpret_view(
                                    w1_pair_codes, pl.INT8,
                                    shape=[MX_PACKED_LANE_ROWS, MX_MM_INTER_TILE],
                                )
                                w13_mxfp8 = pl.store(
                                    pl.reinterpret_view(w1_bytes, pl.FP8E4M3FN),
                                    [k0 + w1_lane * MX_PACKED_LANE_ROWS, n0],
                                    w13_mxfp8,
                                )

                    with pl.spmd(
                        2 * (MOE_INTER // MX_MM_TASK_TILE),
                        name_hint="ffn_up_mxfp4_aiv",
                        deps=[input_ready],
                    ) as w3_deq_tid:
                        bidx_d3 = pl.tile.get_block_idx()
                        nb_d3 = bidx_d3 // 2
                        w3_lane = bidx_d3 % 2
                        nbase_d3 = nb_d3 * MX_MM_TASK_TILE
                        w3_lut_tile = pl.load(
                            mxfp4_pair_lut, [w3_lane, 0], [1, 256],
                            target_memory=pl.Mem.Vec,
                        )
                        for ng in pl.range(MX_MM_TASK_TILE // MX_MM_INTER_TILE):
                            n0 = nbase_d3 + ng * MX_MM_INTER_TILE
                            for k0 in pl.range(0, D, MX_K_TILE):
                                w3_tile_id = (
                                    (n0 // MX_MM_INTER_TILE) * (D // MX_K_TILE)
                                    + k0 // MX_K_TILE
                                )
                                w3_packed_row = (
                                    MX_W1_PACKED_ROWS
                                    + w3_tile_id * 2 * MX_PACKED_LANE_ROWS
                                    + w3_lane * MX_PACKED_LANE_ROWS
                                )
                                w3_packed_tile = pl.load(
                                    routed_w13_packed_e,
                                    [w3_packed_row, 0],
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    target_memory=pl.Mem.Vec,
                                )
                                w3_packed_u16 = pl.cast(w3_packed_tile, target_type=pl.UINT16)
                                w3_packed_indices = pl.reinterpret_view(w3_packed_u16, pl.INT16)
                                w3_gather_tmp = pl.create_tile(
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    dtype=pl.INT16, target_memory=pl.Mem.Vec,
                                )
                                w3_pair_codes = pl.tile.gather(
                                    w3_lut_tile, w3_packed_indices, w3_gather_tmp,
                                )
                                w3_bytes = pl.reinterpret_view(
                                    w3_pair_codes, pl.INT8,
                                    shape=[MX_PACKED_LANE_ROWS, MX_MM_INTER_TILE],
                                )
                                w13_mxfp8 = pl.store(
                                    pl.reinterpret_view(w3_bytes, pl.FP8E4M3FN),
                                    [
                                        k0 + w3_lane * MX_PACKED_LANE_ROWS,
                                        MOE_INTER + n0,
                                    ],
                                    w13_mxfp8,
                                )

                    w2_mxfp8 = pl.create_tensor([MOE_INTER, D], dtype=pl.FP8E4M3FN)
                    with pl.spmd(
                        2 * (D // MX_W2_TASK_TILE),
                        name_hint="ffn_w2_mxfp4_aiv",
                        deps=[input_ready],
                    ) as w2_deq_tid:
                        bidx_d2 = pl.tile.get_block_idx()
                        wb_d2 = bidx_d2 // 2
                        w2_lane = bidx_d2 % 2
                        dbase_d2 = wb_d2 * MX_W2_TASK_TILE
                        w2_lut_tile = pl.load(
                            mxfp4_pair_lut, [w2_lane, 0], [1, 256],
                            target_memory=pl.Mem.Vec,
                        )
                        for dg in pl.range(MX_W2_TASK_TILE // MX_W2_D_OUT_TILE):
                            d0 = dbase_d2 + dg * MX_W2_D_OUT_TILE
                            for k0 in pl.range(0, MOE_INTER, MX_K_TILE):
                                w2_tile_id = (
                                    (d0 // MX_W2_D_OUT_TILE) * (MOE_INTER // MX_K_TILE)
                                    + k0 // MX_K_TILE
                                )
                                w2_packed_row = (
                                    w2_tile_id * 2 * MX_PACKED_LANE_ROWS
                                    + w2_lane * MX_PACKED_LANE_ROWS
                                )
                                w2_packed_tile = pl.load(
                                    routed_w2_packed_e,
                                    [w2_packed_row, 0],
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    target_memory=pl.Mem.Vec,
                                )
                                w2_packed_u16 = pl.cast(w2_packed_tile, target_type=pl.UINT16)
                                w2_packed_indices = pl.reinterpret_view(w2_packed_u16, pl.INT16)
                                w2_gather_tmp = pl.create_tile(
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    dtype=pl.INT16, target_memory=pl.Mem.Vec,
                                )
                                w2_pair_codes = pl.tile.gather(
                                    w2_lut_tile, w2_packed_indices, w2_gather_tmp,
                                )
                                w2_bytes = pl.reinterpret_view(
                                    w2_pair_codes, pl.INT8,
                                    shape=[MX_PACKED_LANE_ROWS, MX_W2_D_OUT_TILE],
                                )
                                w2_mxfp8 = pl.store(
                                    pl.reinterpret_view(w2_bytes, pl.FP8E4M3FN),
                                    [k0 + w2_lane * MX_PACKED_LANE_ROWS, d0],
                                    w2_mxfp8,
                                )

                    n_chunks = (n_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
                    for chunk in pl.range(n_chunks):
                        chunk_row0 = chunk * CHUNK_ROWS
                        chunk_rows = pl.min(CHUNK_ROWS, n_rows - chunk_row0)
                        chunk_tiles = (chunk_rows + RECV_TILE - 1) // RECV_TILE
                        with pl.scope():
                            x_stage_mx = pl.create_tensor(
                                [CHUNK_ROWS, D], dtype=pl.FP8E4M3FN,
                            )
                            x_stage_scale_backing = pl.create_tensor(
                                [1, CHUNK_ROWS * K_SCALE], dtype=pl.FP8E8M0,
                            )
                            x_stage_scale_mx = pl.tensor.view(
                                x_stage_scale_backing, [CHUNK_ROWS, K_SCALE],
                                layout=pl.MX_A_ZZ,
                            )
                            h_mx = pl.create_tensor(
                                [CHUNK_ROWS, MOE_INTER], dtype=pl.FP8E4M3FN,
                            )
                            h_scale_backing = pl.create_tensor(
                                [1, CHUNK_ROWS * H_SCALE], dtype=pl.FP8E8M0,
                            )

                            chunk_hidden_tids = pl.array.create(CHUNK_TILES, pl.TASK_ID)
                            for t in pl.range(CHUNK_TILES):
                                chunk_hidden_tids[t] = pl.system.task_dummy(deps=[])

                            for t in pl.parallel(chunk_tiles):
                                t0 = t * RECV_TILE
                                valid_rows = pl.min(RECV_TILE, chunk_rows - t0)
                                src0 = chunk_row0 + t0
                                with pl.scope():
                                    # Gather this tile's activations by route
                                    # id and requant to MXFP8 (ceil E8M0
                                    # scales, like gate.py). Transport is
                                    # batched: the 16 rows' source tokens
                                    # arrive as one 64B tile window (no
                                    # per-row scalar index reads from GM) and
                                    # each row is DMA'd straight into its
                                    # tile slot by pl.gather_row (GM -> UB
                                    # direct, no load+assemble round trip).
                                    # The quantizer itself is unchanged, so
                                    # the staged bytes are bit-identical to
                                    # the original per-row path.
                                    with pl.spmd(
                                        D // XQ_TASK_TILE,
                                        name_hint="ffn_xgather_quant",
                                        deps=[group_tid],
                                    ) as xquant_tid:
                                        blk = pl.tile.get_block_idx()
                                        n_sub = (valid_rows + AIV_ROW_TILE - 1) // AIV_ROW_TILE
                                        for rq in pl.range(
                                            n_sub * (XQ_TASK_TILE // XQ_TILE)
                                        ):
                                            rb = (
                                                rq // (XQ_TASK_TILE // XQ_TILE)
                                            ) * AIV_ROW_TILE
                                            qc = rq % (XQ_TASK_TILE // XQ_TILE)
                                            k0 = blk * XQ_TASK_TILE + qc * XQ_TILE
                                            # Reads may spill past this
                                            # expert's rows into the next
                                            # expert's region or the
                                            # zero-filled slack (token 0
                                            # rows; only valid_rows are
                                            # consumed later).
                                            tok_tile = pl.load(
                                                tok_ids,
                                                [lbase + src0 + rb, 0],
                                                [AIV_ROW_TILE, 1],
                                            )
                                            xq_bf16 = pl.create_tile(
                                                [AIV_ROW_TILE, XQ_TILE],
                                                dtype=pl.BF16,
                                                target_memory=pl.Mem.Vec,
                                            )
                                            for r in pl.range(AIV_ROW_TILE):
                                                tok_r = pl.cast(
                                                    pl.tile.read(
                                                        tok_tile, [r, 0],
                                                    ),
                                                    pl.INDEX,
                                                )
                                                xq_bf16 = pl.gather_row(
                                                    xq_bf16,
                                                    x_normed,
                                                    [r, 0],
                                                    [tok_r, k0],
                                                    [1, XQ_TILE],
                                                )
                                            # The ceil-scale bit trick needs
                                            # FP32 values.
                                            xq = pl.cast(xq_bf16, pl.FP32)
                                            quant_groups = pl.reshape(
                                                xq, [AIV_ROW_TILE * XQ_GROUPS, 32],
                                            )
                                            quant_tmp = pl.create_tile(
                                                [AIV_ROW_TILE * XQ_GROUPS, 32], dtype=pl.FP32,
                                            )
                                            quant_maximum = pl.maximum(
                                                pl.row_max(
                                                    pl.abs(quant_groups), tmp_tile=quant_tmp,
                                                ),
                                                1e-4,
                                            )
                                            quant_bits = pl.reinterpret_view(
                                                pl.mul(quant_maximum, 1.0 / 448.0), pl.INT32,
                                            )
                                            quant_exponent = pl.shrs(
                                                pl.add(quant_bits, 8388607), 23,
                                            )
                                            quant_scale = pl.reinterpret_view(
                                                pl.shls(quant_exponent, 23), pl.FP32,
                                            )
                                            quant_normalized = pl.row_expand_div(
                                                quant_groups, quant_scale,
                                            )
                                            quant_clipped = pl.minimum(
                                                pl.maximum(quant_normalized, -448.0), 448.0,
                                            )
                                            quant_payload = pl.cast(
                                                quant_clipped, pl.FP8E4M3FN, mode="rint",
                                            )
                                            x_quant = pl.reshape(
                                                quant_payload, [AIV_ROW_TILE, XQ_TILE],
                                            )
                                            quant_signed = pl.sub(
                                                quant_exponent,
                                                pl.mul(pl.shrs(quant_exponent, 7), 256),
                                            )
                                            quant_codes = pl.reinterpret_view(
                                                pl.cast(quant_signed, pl.INT8), pl.UINT8,
                                            )
                                            quant_flat = pl.reshape(
                                                quant_codes, [1, AIV_ROW_TILE * XQ_GROUPS],
                                            )
                                            quant_pack_tmp = pl.create_tile(
                                                [1, XQ_SCALE_TMP], dtype=pl.UINT8,
                                            )
                                            quant_packed = pl.tmov_x2zz(
                                                quant_flat, quant_pack_tmp, group_axis=1,
                                                dst_rows=AIV_ROW_TILE, dst_cols=XQ_GROUPS,
                                            )
                                            scale_quant = pl.reinterpret_view(
                                                quant_packed, pl.FP8E8M0,
                                            )
                                            x_stage_mx = pl.store(
                                                x_quant, [t0 + rb, k0], x_stage_mx,
                                            )
                                            scale_offset = (t0 + rb) * K_SCALE + (
                                                blk * (XQ_TASK_TILE // XQ_TILE) + qc
                                            ) * AIV_ROW_TILE * XQ_GROUPS
                                            x_stage_scale_backing = pl.store(
                                                pl.reshape(
                                                    scale_quant,
                                                    [1, AIV_ROW_TILE * XQ_GROUPS],
                                                ),
                                                [0, scale_offset],
                                                x_stage_scale_backing,
                                            )

                                    # GMM1 is ONE N=4608 matmul over the
                                    # merged W13 staging: gate occupies output
                                    # columns [0, MOE_INTER), up occupies
                                    # [MOE_INTER, 2*MOE_INTER). The scale row
                                    # base is the single static local_e*K_SCALE
                                    # (no block-index-derived slot bases), and
                                    # K/N tile widths stay 256x256. It drains
                                    # through the fix-pipe straight to BF16:
                                    # exactly the pre-SwiGLU rint CANN applies,
                                    # with no FP32 GM round-trip.
                                    w13_tile_bf16 = pl.create_tensor(
                                        [RECV_TILE, 2 * MOE_INTER], dtype=pl.BF16,
                                    )
                                    with pl.spmd(
                                        2 * (MOE_INTER // MX_MM_TASK_TILE),
                                        name_hint="ffn_w13_mxfp4_aic",
                                        deps=[w1_deq_tid, w3_deq_tid, xquant_tid],
                                    ) as w13_mxfp4_aic_tid:
                                        nb_idx = pl.tile.get_block_idx()
                                        n_base = nb_idx * MX_MM_TASK_TILE
                                        for ng in pl.range(
                                            MX_MM_TASK_TILE // MX_MM_INTER_TILE
                                        ):
                                            n0 = n_base + ng * MX_MM_INTER_TILE
                                            w13_scale_row_base = local_e * K_SCALE
                                            w13_acc = pl.create_tile(
                                                [RECV_TILE, MX_MM_INTER_TILE],
                                                dtype=pl.FP32, target_memory=pl.Mem.Acc,
                                            )
                                            for k0 in pl.range(0, D, MX_K_TILE):
                                                ks = k0 // MX_GROUP
                                                w13_tile = pl.load(
                                                    w13_mxfp8, [k0, n0],
                                                    [MX_K_TILE, MX_MM_INTER_TILE],
                                                    target_memory=pl.Mem.Mat,
                                                )
                                                xs_k = pl.load(
                                                    x_stage_mx, [t0, k0],
                                                    [RECV_TILE, MX_K_TILE],
                                                )
                                                xs_scale_k = pl.load(
                                                    x_stage_scale_mx, [t0, ks],
                                                    [RECV_TILE, MX_K_SCALE_GROUPS],
                                                    target_memory=pl.Mem.Mat,
                                                )
                                                w13_scale_k = pl.load(
                                                    routed_w13_scale,
                                                    [w13_scale_row_base + ks, n0],
                                                    [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE],
                                                    target_memory=pl.Mem.Mat,
                                                )
                                                if k0 == 0:
                                                    w13_acc = pl.matmul_mx(
                                                        xs_k, xs_scale_k,
                                                        w13_tile, w13_scale_k,
                                                    )
                                                else:
                                                    w13_acc = pl.matmul_mx_acc(
                                                        w13_acc, xs_k, xs_scale_k,
                                                        w13_tile, w13_scale_k,
                                                    )
                                            w13_tile_bf16 = pl.store(
                                                w13_acc, [0, n0], w13_tile_bf16,
                                            )

                                    h_tile_mx = h_mx[t0 : t0 + RECV_TILE]
                                    h_tile_scale_backing = h_scale_backing[
                                        :, t0 * H_SCALE : (t0 + RECV_TILE) * H_SCALE
                                    ]
                                    with pl.spmd(
                                        MOE_INTER // QUANT_TILE,
                                        name_hint="ffn_gate_up_act_quant",
                                        deps=[w13_mxfp4_aic_tid],
                                    ) as gate_up_act_quant_tid:
                                        ab_idx = pl.tile.get_block_idx()
                                        a_base = ab_idx * QUANT_TILE
                                        n_sub = (valid_rows + AIV_ROW_TILE - 1) // AIV_ROW_TILE
                                        for rs in pl.range(n_sub):
                                            rb = rs * AIV_ROW_TILE
                                            h_fp32 = pl.tile.full(
                                                [AIV_ROW_TILE, QUANT_TILE], dtype=pl.FP32,
                                                value=0.0,
                                            )
                                            route_weight = pl.load(
                                                w_compact, [lbase + src0 + rb, 0],
                                                [AIV_ROW_TILE, 1],
                                            )
                                            for ag in pl.pipeline(ACT_GATE_INNER, stage=2):
                                                a0 = a_base + ag * ACT_INTER_TILE
                                                h_a0 = ag * ACT_INTER_TILE
                                                # The fix-pipe already rounded
                                                # gate/up to BF16 on drain (CANN's
                                                # pre-SwiGLU rint); widen to FP32
                                                # for the activation math.
                                                gate_2d = pl.cast(
                                                    pl.load(
                                                        w13_tile_bf16, [rb, a0],
                                                        [AIV_ROW_TILE, ACT_INTER_TILE],
                                                    ),
                                                    pl.FP32,
                                                )
                                                up_2d = pl.cast(
                                                    pl.load(
                                                        w13_tile_bf16,
                                                        [rb, a0 + MOE_INTER],
                                                        [AIV_ROW_TILE, ACT_INTER_TILE],
                                                    ),
                                                    pl.FP32,
                                                )
                                                if SWIGLU_LIMIT > 0.0:
                                                    gate_2d = pl.minimum(
                                                        gate_2d, SWIGLU_LIMIT,
                                                    )
                                                    up_2d = pl.maximum(
                                                        pl.minimum(up_2d, SWIGLU_LIMIT),
                                                        -SWIGLU_LIMIT,
                                                    )
                                                sigmoid = pl.recip(
                                                    pl.add(pl.exp(pl.neg(gate_2d)), 1.0),
                                                )
                                                silu = pl.mul(gate_2d, sigmoid)
                                                gated = pl.row_expand_mul(
                                                    pl.mul(silu, up_2d), route_weight,
                                                )
                                                gated = pl.cast(
                                                    pl.cast(gated, pl.BF16, mode="rint"),
                                                    pl.FP32,
                                                )
                                                h_fp32 = pl.tile.assemble(
                                                    h_fp32, gated, [0, h_a0],
                                                )
                                            # Row-local quant: rows past
                                            # valid_rows carry finite slack
                                            # values and only ever land in the
                                            # expert's y_grouped padding, so no
                                            # validshape/fillpad dance.
                                            # The released activation quantizer
                                            # rounds the group scale upward.
                                            sq_input = h_fp32
                                            sq_values = pl.reshape(
                                                sq_input, [AIV_ROW_TILE * SWIGLU_GROUPS, 32],
                                            )
                                            sq_reduce_tmp = pl.create_tile(
                                                [AIV_ROW_TILE * SWIGLU_GROUPS, 32],
                                                dtype=pl.FP32,
                                            )
                                            sq_maximum = pl.maximum(
                                                pl.row_max(
                                                    pl.abs(sq_values), tmp_tile=sq_reduce_tmp,
                                                ),
                                                1e-4,
                                            )
                                            sq_bits = pl.reinterpret_view(
                                                pl.mul(sq_maximum, 1.0 / 448.0), pl.INT32,
                                            )
                                            sq_exponent = pl.shrs(
                                                pl.add(sq_bits, 8388607), 23,
                                            )
                                            sq_scale = pl.reinterpret_view(
                                                pl.shls(sq_exponent, 23), pl.FP32,
                                            )
                                            sq_normalized = pl.row_expand_div(
                                                sq_values, sq_scale,
                                            )
                                            sq_clipped = pl.minimum(
                                                pl.maximum(sq_normalized, -448.0), 448.0,
                                            )
                                            sq_quantized = pl.cast(
                                                sq_clipped, pl.FP8E4M3FN, mode="rint",
                                            )
                                            h_quant = pl.reshape(
                                                sq_quantized, [AIV_ROW_TILE, QUANT_TILE],
                                            )
                                            sq_signed_exponent = pl.sub(
                                                sq_exponent,
                                                pl.mul(pl.shrs(sq_exponent, 7), 256),
                                            )
                                            sq_codes = pl.reinterpret_view(
                                                pl.cast(sq_signed_exponent, pl.INT8),
                                                pl.UINT8,
                                            )
                                            sq_flat = pl.reshape(
                                                sq_codes, [1, AIV_ROW_TILE * SWIGLU_GROUPS],
                                            )
                                            sq_temporary = pl.create_tile(
                                                [1, SWIGLU_SCALE_TMP], dtype=pl.UINT8,
                                            )
                                            sq_packed = pl.tmov_x2zz(
                                                sq_flat, sq_temporary, group_axis=1,
                                                dst_rows=AIV_ROW_TILE,
                                                dst_cols=SWIGLU_GROUPS,
                                            )
                                            h_scale = pl.reinterpret_view(
                                                sq_packed, pl.FP8E8M0,
                                            )
                                            h_tile_mx = pl.store(
                                                h_quant, [rb, a_base], h_tile_mx,
                                            )
                                            scale_offset = rb * H_SCALE + (
                                                ab_idx * AIV_ROW_TILE * (QUANT_TILE // MX_GROUP)
                                            )
                                            h_tile_scale_backing = pl.store(
                                                pl.reshape(
                                                    h_scale,
                                                    [1, AIV_ROW_TILE * (QUANT_TILE // MX_GROUP)],
                                                ),
                                                [0, scale_offset],
                                                h_tile_scale_backing,
                                            )
                                    chunk_hidden_tids[t] = gate_up_act_quant_tid

                            for tt in pl.parallel(chunk_tiles):
                                tt0 = tt * RECV_TILE
                                src1 = chunk_row0 + tt0
                                h_tile_mx = h_mx[tt0 : tt0 + RECV_TILE]
                                h_tile_scale_backing = h_scale_backing[
                                    :, tt0 * H_SCALE : (tt0 + RECV_TILE) * H_SCALE
                                ]
                                h_tile_scale_mx = pl.tensor.view(
                                    h_tile_scale_backing, [RECV_TILE, H_SCALE],
                                    layout=pl.MX_A_ZZ,
                                )
                                with pl.scope():
                                    # GMM2 drains through the fix-pipe straight
                                    # to BF16 (the golden rint): no FP32 GM
                                    # round-trip, no separate round task. The
                                    # drain targets the expert's padded region
                                    # of y_grouped directly, so every
                                    # [RECV_TILE, MX_W2_D_OUT_TILE] block lands
                                    # as one vectorized contiguous store — the
                                    # old per-row transpose kernel is gone, and
                                    # rows past the expert's valid rows fall
                                    # into its own padding, never read.
                                    with pl.spmd(
                                        D // MX_W2_TASK_TILE,
                                        name_hint="ffn_w2_mxfp4_aic",
                                        deps=[w2_deq_tid, chunk_hidden_tids[tt]],
                                    ) as w2_mxfp4_aic_tid:
                                        wb_idx = pl.tile.get_block_idx()
                                        d_base = wb_idx * MX_W2_TASK_TILE
                                        for dg in pl.range(
                                            MX_W2_TASK_TILE // MX_W2_D_OUT_TILE
                                        ):
                                            d0 = d_base + dg * MX_W2_D_OUT_TILE
                                            w2_scale_row_base = local_e * H_SCALE
                                            y_acc = pl.create_tile(
                                                [RECV_TILE, MX_W2_D_OUT_TILE],
                                                dtype=pl.FP32, target_memory=pl.Mem.Acc,
                                            )
                                            for k0 in pl.range(0, MOE_INTER, MX_K_TILE):
                                                ks = k0 // MX_GROUP
                                                w2_tile = pl.load(
                                                    w2_mxfp8, [k0, d0],
                                                    [MX_K_TILE, MX_W2_D_OUT_TILE],
                                                    target_memory=pl.Mem.Mat,
                                                )
                                                h_k = pl.load(
                                                    h_tile_mx, [0, k0],
                                                    [RECV_TILE, MX_K_TILE],
                                                )
                                                h_scale_k = pl.load(
                                                    h_tile_scale_mx, [0, ks],
                                                    [RECV_TILE, MX_K_SCALE_GROUPS],
                                                    target_memory=pl.Mem.Mat,
                                                )
                                                w2_scale_k = pl.load(
                                                    routed_w2_scale,
                                                    [w2_scale_row_base + ks, d0],
                                                    [MX_K_SCALE_GROUPS, MX_W2_D_OUT_TILE],
                                                    target_memory=pl.Mem.Mat,
                                                )
                                                if k0 == 0:
                                                    y_acc = pl.matmul_mx(
                                                        h_k, h_scale_k, w2_tile, w2_scale_k,
                                                    )
                                                else:
                                                    y_acc = pl.matmul_mx_acc(
                                                        y_acc, h_k, h_scale_k,
                                                        w2_tile, w2_scale_k,
                                                    )
                                            y_grouped = pl.store(
                                                y_acc, [pbase + src1, d0],
                                                y_grouped,
                                            )
                                    tile_completion_tids[
                                        chunk * CHUNK_TILES + tt
                                    ] = w2_mxfp4_aic_tid
            expert_completion_tids[local_e] = pl.system.task_dummy(
                deps=[tile_completion_tids],
            )

        # ------------------------------------------------------------------
        # Stage 8: top-k weighted aggregation, FP32 accumulate, one rint.
        # Gather-read form: each block reads its token's TOPK rows straight
        # out of the expert-grouped buffer through the route -> padded-row
        # inverse map, with expert-granularity dependencies on the per-expert
        # GMM2 completion events (no intermediate global-barrier dummy task).
        # ------------------------------------------------------------------
        with pl.spmd(
            T * (D // RED_TILE),
            name_hint="moe_ffn_reduce",
            deps=[expert_completion_tids[e] for e in range(E)],
        ):
            blk = pl.tile.get_block_idx()
            tok = blk // (D // RED_TILE)
            d0 = (blk % (D // RED_TILE)) * RED_TILE
            acc = pl.tile.full([1, RED_TILE], dtype=pl.FP32, value=0.0)
            for k in pl.range(TOPK):
                row = pl.cast(pl.read(route_pos, [tok * TOPK + k, 0]), pl.INDEX)
                acc = pl.add(
                    acc,
                    pl.cast(pl.load(y_grouped, [row, d0], [1, RED_TILE]), pl.FP32),
                )
            out = pl.store(
                pl.cast(acc, target_type=pl.BF16, mode="rint"), [tok, d0], out,
            )
    return out


@pl.jit(auto_scope=False)
def moe_ffn_test(
    x_normed: pl.Tensor[[T_PAD, D], pl.BF16],
    topk_indices: pl.Tensor[[T, TOPK], pl.INT32],
    topk_weights: pl.Tensor[[T, TOPK], pl.FP32],
    routed_w13_packed: pl.Tensor[
        [E, MX_W13_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8,
    ],
    routed_w13_scale: pl.Tensor[
        [E * K_SCALE, 2 * MOE_INTER], pl.FP8E8M0, pl.MX_B_NN,
    ],
    routed_w2_packed: pl.Tensor[
        [E, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8,
    ],
    routed_w2_scale: pl.Tensor[[E * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    out = moe_ffn_compute(
        x_normed, topk_indices, topk_weights,
        routed_w13_packed, routed_w13_scale,
        routed_w2_packed, routed_w2_scale,
        mxfp4_pair_lut, out,
    )
    return out


def golden_moe_ffn(tensors):
    """Torch reference for the gate-after routed MoE FFN with top-k aggregation."""
    import torch
    import torch.nn.functional as F

    from quant import (
        decode_e8m0_codes_v41 as decode_e8m0_codes,
        host_quant_swiglu_mxfp8_v41 as host_quant_swiglu_mxfp8,
        matmul_mx_golden_v41 as matmul_mx_golden,
        unpack_mxfp4_weight_tiles,
    )

    x = tensors["x_normed"][:T].float()
    indices = tensors["topk_indices"].to(torch.int64)
    weights = tensors["topk_weights"].float()
    w13_scale = decode_e8m0_codes(tensors["routed_w13_scale"], side="b").reshape(
        E, K_SCALE, 2 * MOE_INTER,
    )
    w1_scale = w13_scale[:, :, :MOE_INTER]
    w3_scale = w13_scale[:, :, MOE_INTER:]
    w2_scale = decode_e8m0_codes(tensors["routed_w2_scale"], side="b").reshape(
        E, H_SCALE, D,
    )

    y_buf = torch.zeros(TOTAL, D)
    for e in range(E):
        hits = (indices == e).nonzero(as_tuple=False)
        if hits.numel() == 0:
            continue
        tokens = hits[:, 0]
        route_w = weights[hits[:, 0], hits[:, 1]].reshape(-1, 1)
        x_sub = x[tokens]
        x_fp8, x_codes = host_quant_swiglu_mxfp8(x_sub, return_e8m0=True)
        w13_packed_e = tensors["routed_w13_packed"][e]
        w1_fp8 = unpack_mxfp4_weight_tiles(
            w13_packed_e[:MX_W1_PACKED_ROWS].reshape(
                MX_W1_PACKED_TILES, MX_K_TILE * MX_MM_INTER_TILE // 4,
            ),
            D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE,
        )
        w3_fp8 = unpack_mxfp4_weight_tiles(
            w13_packed_e[MX_W1_PACKED_ROWS:].reshape(
                MX_W3_PACKED_TILES, MX_K_TILE * MX_MM_INTER_TILE // 4,
            ),
            D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE,
        )
        w2_fp8 = unpack_mxfp4_weight_tiles(
            tensors["routed_w2_packed"][e].reshape(
                MX_W2_PACKED_TILES, MX_K_TILE * MX_W2_D_OUT_TILE // 4,
            ),
            MOE_INTER, D, MX_K_TILE, MX_W2_D_OUT_TILE,
        )
        gate = matmul_mx_golden(x_fp8, x_codes, w1_fp8, w1_scale[e]).to(torch.bfloat16).float()
        up = matmul_mx_golden(x_fp8, x_codes, w3_fp8, w3_scale[e]).to(torch.bfloat16).float()
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = (F.silu(gate) * up * route_w).bfloat16().float()
        h_fp8, h_codes = host_quant_swiglu_mxfp8(h, return_e8m0=True)
        y = matmul_mx_golden(h_fp8, h_codes, w2_fp8, w2_scale[e]).to(torch.bfloat16).float()
        assign = (tokens * TOPK + hits[:, 1]).to(torch.int64)
        y_buf[assign] = y

    out = y_buf.reshape(T, TOPK, D).sum(dim=1).to(torch.bfloat16)
    tensors["out"][:] = out


def build_tensor_specs():
    import torch

    from golden.spec import TensorSpec
    from quant import (
        build_mxfp4_pair_lut,
        pack_mx_b_scale,
        unpack_mx_b_scale,
    )
    from expert_routed import gen_routed_mxfp4_weights

    generator = torch.Generator().manual_seed(20261016)
    x_normed = torch.randn(T_PAD, D, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    if T < T_PAD:
        x_normed[T:, :] = 0
    topk_indices = torch.stack(
        [torch.randperm(E, generator=generator)[:TOPK] for _ in range(T)],
    ).to(torch.int32)
    topk_weights = torch.rand(T, TOPK, generator=generator, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * M.routed_scaling_factor

    # The MXFP4 weight fixtures cost minutes of host time; keep them behind
    # the spec lambdas so --golden-data replays never regenerate them, and
    # memoize so the six weight specs share one generation.
    weight_cache: dict = {}

    def gen_weights():
        if "packed" not in weight_cache:
            weight_cache["packed"] = gen_routed_mxfp4_weights(
                E, ROUTED_DEQUANT_STD, seed_base=10,
            )
        return weight_cache["packed"]

    mxfp4_pair_lut = build_mxfp4_pair_lut()

    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    def gen_w13_scale():
        """Merged W13 scale backing: pack the LOGICAL column concatenation.

        MX_B_NN physically interleaves (n-block, k-pair), so a byte-level cat
        of the W1/W3 backings is not the merged physical layout; decode both
        to logical [E*K_SCALE, MOE_INTER] rows, concatenate along N, and
        re-pack into [E*K_SCALE, 2*MOE_INTER].
        """
        logical = torch.cat(
            [
                unpack_mx_b_scale(gen_weights()[1].view(torch.uint8)),
                unpack_mx_b_scale(gen_weights()[3].view(torch.uint8)),
            ],
            dim=1,
        )
        packed = pack_mx_b_scale(logical)
        return packed.view(fp8_e8m0) if fp8_e8m0 is not torch.uint8 else packed

    return [
        TensorSpec("x_normed", [T_PAD, D], torch.bfloat16, init_value=lambda: x_normed),
        TensorSpec("topk_indices", [T, TOPK], torch.int32, init_value=lambda: topk_indices),
        TensorSpec("topk_weights", [T, TOPK], torch.float32, init_value=lambda: topk_weights),
        TensorSpec(
            "routed_w13_packed", [E, MX_W13_PACKED_ROWS, MX_PACKED_LANE_COLS],
            torch.uint8,
            init_value=lambda: torch.cat(
                [gen_weights()[0], gen_weights()[2]], dim=1,
            ),
        ),
        TensorSpec(
            "routed_w13_scale", [E * K_SCALE, 2 * MOE_INTER], fp8_e8m0,
            init_value=gen_w13_scale,
        ),
        TensorSpec(
            "routed_w2_packed", [E, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            torch.uint8, init_value=lambda: gen_weights()[4],
        ),
        TensorSpec(
            "routed_w2_scale", [E * H_SCALE, D], fp8_e8m0,
            init_value=lambda: gen_weights()[5],
        ),
        TensorSpec("mxfp4_pair_lut", [2, 256], torch.int16, init_value=lambda: mxfp4_pair_lut),
        TensorSpec("out", [T, D], torch.bfloat16),
    ]


def validate(argv=None):
    """Validate the gate-after routed MoE FFN on one device."""
    import argparse

    from golden.runner import run
    from golden.validation import ratio_reldiff

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a5",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="requests per card (s1=6 rows each); import-time knob")
    parser.add_argument("--experts", type=int, default=None,
                        help="routed experts resident on this card "
                        "(384=single-card, 12=EP32 per-card form); import-time knob")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args(argv)

    return run(
        fn=moe_ffn_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_ffn,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            # All 384 experts keep MXFP8 weight staging and activation tiles
            # live in one invocation; give the ring heap the same headroom
            # expert_routed.py asks for.
            ring_heap=1_073_741_824,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 routed output rows, ~1 ULP after one FP32 top-k accumulate.
            "out": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
