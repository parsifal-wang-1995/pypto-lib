# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The kpool DSA indexer, decode path: one continuous-batch step, end to end.

Decode runs the same per-query math as prefill, so the kernels live in
:mod:`models.glm5_3_flash.prefill_indexer` and this file contributes the
composition: one dispatch of ``DECODE_ROWS_PER_REQUEST`` = 4 packed rows per
request (one live token plus three MTP spec tokens) drives the whole pipeline,

1. :func:`~models.glm5_3_flash.prefill_indexer.indexer_proj` projects the 4
   rows' queries, keys, head weights and gate scores;
2. :func:`~models.glm5_3_flash.indexer_cache.indexer_cache_write` scatters the
   raw ``[key, gate]`` rows into the support table;
3. :func:`~models.glm5_3_flash.indexer_cache.indexer_pool_write` closes the one
   pool this step completes per request — four new rows always finish exactly
   one pool window, whatever the request's length alignment, so the pooled
   table grows by exactly one row per request per step and the incomplete tail
   pool is never stored (``index_kpool_always_select_tail`` force-selects its
   raw positions in the expansion);
4. :func:`~models.glm5_3_flash.prefill_indexer.indexer_score` scores the
   queries against the pooled table, this time including the just-closed row;
5. :func:`~models.glm5_3_flash.prefill_indexer.indexer_topk` selects the top
   512 pools per query row;
6. :func:`~models.glm5_3_flash.prefill_indexer.indexer_expand` emits the
   front-packed 2051-wide index list both sparse attention kernels consume.

Everything the composition needs beyond the tensors is host-lowered metadata —
physical cache rows and segment offsets per :mod:`models.glm5_3_flash.metadata`
— the same way the standalone kernel tests consume it. The projection stage
needs a token count that is a multiple of 16 (the cube's row tile), which a
decode batch reaches by padding to whole requests; padded rows carry ordinary
metadata and simply compute selections nobody gathers.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, D, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_STATE_WIDTH, KPOOL_SELECT_K, POOLS_DYN
from models.glm5_3_flash.config import Q_LORA, TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN
from models.glm5_3_flash.indexer_cache import POOL_VALID_WIDTH
from models.glm5_3_flash.indexer_cache import indexer_cache_write, indexer_pool_write
from models.glm5_3_flash.prefill_indexer import LEAF
from models.glm5_3_flash.prefill_indexer import golden_indexer_expand, golden_indexer_proj
from models.glm5_3_flash.prefill_indexer import golden_indexer_score, golden_indexer_topk
from models.glm5_3_flash.prefill_indexer import indexer_expand, indexer_proj
from models.glm5_3_flash.prefill_indexer import indexer_score, indexer_topk, sylvester_hadamard
from models.glm5_3_flash.indexer_cache import golden_indexer_cache_write, golden_indexer_pool_write


@pl.jit
def decode_indexer_step_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    index_slots: pl.Tensor[[T_DYN], pl.INT32],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_rows: pl.Tensor[[POOLS_DYN], pl.INT32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    raw_cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
    pool_cache: pl.InOut[
        pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16]
    ],
    pool_valid: pl.Out[pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32]],
    index_q: pl.Out[pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16]],
    head_weights: pl.Out[pl.Tensor[[T_DYN, INDEX_H], pl.FP32]],
    gate_scores: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32]],
    index_scores: pl.Out[pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32]],
    selected_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    """Run one whole decode step for golden.run validation.

    The raw scatter, the pool close, the score, the selection and the expansion
    order themselves through their shared tensors; every stage's output is
    exposed so the harness can validate the composition, not just the tail.
    """
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    index_slots.bind_dynamic(0, T_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    tail_start.bind_dynamic(0, T_DYN)
    tail_count.bind_dynamic(0, T_DYN)
    kv_len.bind_dynamic(0, T_DYN)
    pool_token_slots.bind_dynamic(0, POOLS_DYN)
    pool_slots.bind_dynamic(0, POOLS_DYN)
    pool_rows.bind_dynamic(0, POOLS_DYN)
    pool_valid.bind_dynamic(0, POOLS_DYN)
    index_q.bind_dynamic(0, T_DYN)
    index_k.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(0, T_DYN)
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)

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
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, pool_token_slots, pool_slots, pool_cache, pool_valid
    )
    indexer_score(
        index_q,
        hadamard,
        pool_cache,
        pool_rows,
        head_weights,
        seg_start,
        pool_count,
        index_scores,
    )
    indexer_topk(index_scores, seg_start, pool_count, selected_pools, selected_valid)
    indexer_expand(
        selected_pools, selected_valid, tail_start, tail_count, kv_len, topk_indices
    )
    return (
        index_q,
        index_k,
        head_weights,
        gate_scores,
        raw_cache,
        pool_cache,
        pool_valid,
        index_scores,
        selected_pools,
        selected_valid,
        topk_indices,
    )


def build_decode_indexer_step_specs(requests: int = 4):
    """Build one deterministic decode step: four requests, one closing pool each.

    The requests carry prior contexts of different lengths, so one closing pool
    is made of four fresh rows, another of fresh rows mixed with rows cached by
    earlier steps, and the pooled table starts with rows only earlier steps
    could have written. Every request contributes exactly
    ``DECODE_ROWS_PER_REQUEST`` query rows.
    """
    from golden import TensorSpec

    from models.glm5_3_flash.config import DECODE_ROWS_PER_REQUEST

    rows_per_request = DECODE_ROWS_PER_REQUEST
    tokens = requests * rows_per_request
    if tokens % 16:
        raise ValueError("the projection stage needs a token count that is a multiple of 16")
    prior_counts = (13, 5, 40, 2)[:requests]
    new_counts = tuple(count + 1 for count in prior_counts)
    pools_true = sum(new_counts)
    width = ((pools_true + 2 * LEAF - 2) // LEAF) * LEAF

    generator = torch.Generator().manual_seed(89)
    prior_tokens = [4 * count for count in prior_counts]

    def row_map():
        """Physical raw row of every request's logical token, past and new."""
        rows = {}
        used = 0
        for request in range(requests):
            total = prior_tokens[request] + rows_per_request
            pages = torch.randperm(8, generator=generator)[: (total + 127) // 128 + 1]
            for position in range(total):
                rows[(request, position)] = int(pages[position // 128]) * 128 + position % 128
            used += 1
        return rows

    mapping = row_map()

    def raw_rows_for(request, first, count):
        return [mapping[(request, position)] for position in range(first, first + count)]

    pool_row_of = {}
    taken = set()
    pool_table_rows = ((pools_true + 3) // 4) * 4 + 4

    def distinct_pool_row():
        candidate = int(torch.randint(0, pool_table_rows, (1,), generator=generator))
        while candidate in taken:
            candidate = (candidate + 1) % pool_table_rows
        taken.add(candidate)
        return candidate

    for request in range(requests):
        for pool in range(new_counts[request]):
            pool_row_of[(request, pool)] = distinct_pool_row()

    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    tail_start = torch.zeros(tokens, dtype=torch.int32)
    tail_count = torch.zeros(tokens, dtype=torch.int32)
    kv_len = torch.zeros(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            token = request * rows_per_request + row
            position = prior_tokens[request] + row
            length = position + 1
            seg_start[token] = sum(new_counts[:request])
            pool_count[token] = min(length // INDEX_KPOOL, new_counts[request])
            tail_count[token] = length % INDEX_KPOOL
            tail_start[token] = length - int(tail_count[token])
            kv_len[token] = length

    pool_rows = torch.zeros(width, dtype=torch.int32)
    flat = 0
    for request in range(requests):
        for pool in range(new_counts[request]):
            pool_rows[flat] = pool_row_of[(request, pool)]
            flat += 1

    closing_slots = torch.full((requests, INDEX_KPOOL), -1, dtype=torch.int32)
    closing_dest = torch.zeros(requests, dtype=torch.int32)
    for request in range(requests):
        closing = new_counts[request] - 1
        first = closing * INDEX_KPOOL
        closing_slots[request] = torch.tensor(
            raw_rows_for(request, first, INDEX_KPOOL), dtype=torch.int32
        )
        closing_dest[request] = pool_row_of[(request, closing)]

    index_slots = torch.zeros(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            index_slots[request * rows_per_request + row] = mapping[
                (request, prior_tokens[request] + row)
            ]

    def init_x():
        return torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16()

    def init_q_resid():
        return torch.randn(tokens, Q_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_raw_cache():
        rows = max(mapping.values()) + 1
        cache = torch.randn(rows, INDEX_STATE_WIDTH, generator=generator)
        cache[:, :INDEX_DIM] = cache[:, :INDEX_DIM].bfloat16().float()
        return cache

    def init_pool_cache():
        cache = torch.randn(pool_table_rows, INDEX_DIM, generator=generator).bfloat16()
        return cache

    shapes = {
        "w_q_b": (Q_LORA, INDEX_H * INDEX_DIM),
        "w_k": (D, INDEX_DIM),
        "w_weights": (D, INDEX_H),
        "w_compress_gate": (D, INDEX_DIM),
    }

    def init_weights(name, scale):
        def build():
            return (torch.randn(*shapes[name], generator=generator) * scale).bfloat16()

        return build

    specs = [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=init_q_resid),
        TensorSpec(
            "w_q_b",
            [Q_LORA, INDEX_H * INDEX_DIM],
            torch.bfloat16,
            init_value=init_weights("w_q_b", 0.02),
        ),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_k", 0.02)),
        TensorSpec(
            "k_norm_weight",
            [INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16(),
        ),
        TensorSpec(
            "k_norm_bias",
            [INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16(),
        ),
        TensorSpec(
            "w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_weights("w_weights", 0.05)
        ),
        TensorSpec(
            "w_compress_gate",
            [D, INDEX_DIM],
            torch.bfloat16,
            init_value=init_weights("w_compress_gate", 0.02),
        ),
        TensorSpec(
            "hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard
        ),
        TensorSpec(
            "compress_ape",
            [INDEX_KPOOL, INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16(),
        ),
        TensorSpec("index_slots", [tokens], torch.int32, init_value=lambda: index_slots),
        TensorSpec(
            "pool_token_slots", [requests, INDEX_KPOOL], torch.int32, init_value=lambda: closing_slots
        ),
        TensorSpec("pool_slots", [requests], torch.int32, init_value=lambda: closing_dest),
        TensorSpec("pool_rows", [width], torch.int32, init_value=lambda: pool_rows),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=lambda: tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=lambda: tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=lambda: kv_len),
        TensorSpec(
            "raw_cache",
            [max(mapping.values()) + 1, INDEX_STATE_WIDTH],
            torch.float32,
            init_value=init_raw_cache,
        ),
        TensorSpec(
            "pool_cache", [pool_table_rows, INDEX_DIM], torch.bfloat16, init_value=init_pool_cache
        ),
        TensorSpec("pool_valid", [requests, POOL_VALID_WIDTH], torch.int32),
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16),
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32),
        TensorSpec("index_scores", [tokens, width], torch.float32),
        TensorSpec("selected_pools", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("selected_valid", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32),
    ]
    return specs


def golden_decode_indexer_step_case(tensors):
    """Fill every expected output of one decode step by composing the goldens."""
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
    tensors["raw_cache"][:] = golden_indexer_cache_write(
        tensors["raw_cache"], index_k, gate_scores, tensors["index_slots"]
    )
    pool_cache, pool_valid = golden_indexer_pool_write(
        tensors["pool_cache"],
        tensors["raw_cache"],
        tensors["compress_ape"],
        tensors["pool_token_slots"],
        tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_valid"][:] = pool_valid
    tensors["index_scores"][:] = golden_indexer_score(
        tensors["index_q"],
        tensors["hadamard"],
        tensors["pool_cache"],
        tensors["pool_rows"],
        tensors["head_weights"],
        tensors["seg_start"],
        tensors["pool_count"],
    )
    selected, valid = golden_indexer_topk(
        tensors["index_scores"], tensors["seg_start"], tensors["pool_count"]
    )
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"],
        tensors["selected_valid"],
        tensors["tail_start"],
        tensors["tail_count"],
        tensors["kv_len"],
    )
    for token in range(tensors["topk_indices"].shape[0]):
        row = tensors["topk_indices"][token]
        live = int((row >= 0).sum())
        assert (row[:live] >= 0).all() and (row[live:] == -1).all(), (
            f"row {token} violates the front-packed index ABI"
        )


def main():
    """Prove the composed golden on CPU, then validate one decode step on device."""
    import argparse

    from golden import ratio_allclose, run, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--requests", type=int, default=4)
    args = parser.parse_args()

    def selected_pools_compare(
        actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol
    ):
        scores = actual_outputs["index_scores"].float()
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

    result = run(
        fn=decode_indexer_step_test,
        specs=build_decode_indexer_step_specs(args.requests),
        golden_fn=golden_decode_indexer_step_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1.0 / 64,
        atol=1e-3,
        compare_fn={
            "index_q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            "index_k": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            "head_weights": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
            "gate_scores": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
            # The raw rows carry the projection rounding of the composed step,
            # unlike the standalone scatter, which moves bytes at zero tolerance.
            "raw_cache": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            "pool_cache": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
            "pool_valid": exact_compare,
            "index_scores": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
            "selected_pools": selected_pools_compare,
            "selected_valid": exact_compare,
            "topk_indices": exact_compare,
        },
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "build_decode_indexer_step_specs",
    "decode_indexer_step_test",
    "golden_decode_indexer_step_case",
    "indexer_cache_write",
    "indexer_expand",
    "indexer_proj",
    "indexer_score",
    "indexer_topk",
    "indexer_pool_write",
]


if __name__ == "__main__":
    main()
