# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The indexer's own paged state cache: raw support rows plus a pooled key table.

Two tables, written by two kernels, make up the DSA indexer's persistent state:

* the **raw support table** — one FP32 row per token, ``[key(128), gate_scores(128)]``
  (:data:`INDEX_STATE_WIDTH` wide), paged at :data:`BLOCK_SIZE` like the MLA latent
  pool. :func:`indexer_cache_write` scatters one dispatch's rows into it. It is the
  authoritative record the pooled table is derived from: scoring never reads it, but
  every pool that closes is compressed from these rows, so re-pooling is always
  possible and the gate projections are never recomputed.
* the **pooled key table** — one BF16 :data:`INDEX_DIM`-wide row per closed pool,
  paged at :data:`INDEX_STATE_BLOCK_SIZE` pools per block. :func:`indexer_pool_write`
  closes pools: for one pool event it reads the four member raw rows, takes the
  learned per-dimension softmax ``softmax(gate_scores + ape)`` over the four lanes —
  the reference ``Glm5NextTextIndexer.get_pooled_states`` does this in FP32 — and
  stores the weighted key. The score then reads this table directly, which is the
  whole point of the layout: a query scans one pooled row per four tokens instead of
  pooling on the fly.

A pool closes exactly when its fourth token is cached, so the write side is driven by
**pool-close events**: prefill emits one event per pool the chunk completes, a decode
dispatch of ``DECODE_ROWS_PER_REQUEST`` = 4 rows per request completes exactly one
pool per request. The incomplete tail pool is never stored —
``index_kpool_always_select_tail`` force-selects its raw positions in the expansion,
so it is not a scoring candidate and needs no pooled row. This is the same shape the
a2a3 sibling port arrives at (``deepseek_v4_flash_mtp`` keeps a compressed
per-four-token table and a rolling tail) and the shape vLLM Ascend's page classes
describe; what has no donor is the pooling math itself, because the gate scores here
come from the cached projection rather than an in-kernel recompute.

``Glm5NextTextIndexer.forward`` packs a ``valid`` channel next to key and gate. That
channel only exists to let pooling start at the first real token of a left-padded
dense batch; a paged cache has no left padding, so this ABI drops it the way
``AscendIndexerKPoolStateSpec`` does, and a pool event whose member slot is ``-1``
(padded row) is rejected through :func:`indexer_pool_write`'s ``pool_valid`` output
instead.

FP32 keys and gates cost 1 KB per token per rank per DSA layer on the raw side and
128 B per token on the pooled side. The a2a3 sibling quantizes its own indexer cache
to INT8 with a per-row FP32 scale, which would take the pooled half from 256 B to
about 64 B per four tokens. Treat that as a follow-up, not a default: the pooled key
feeds a ``relu``-gated score whose sensitivity to INT8 has not been measured for this
checkpoint.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, INDEX_BLOCKS_DYN, INDEX_DIM
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_STATE_WIDTH, POOLS_DYN, TABLE_DYN, T_DYN


# The per-event validity flag is one INT32 lane padded to the 32-byte minimum of
# an a2a3 vector tile; the flag sits in column 0 and columns 1-7 stay zero.
POOL_VALID_WIDTH = 8


def golden_indexer_cache_write(
    cache: torch.Tensor,
    index_k: torch.Tensor,
    gate_scores: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's raw rows, skipping ``-1`` slots.

    The guard is the ABI, not politeness: an unmasked ``-1`` indexes the last row of
    the pool through Python's negative indexing and silently corrupts it.
    """
    packed = torch.cat([index_k, gate_scores], dim=-1)
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = packed.to(cache.dtype)[written]
    return updated


@pl.jit.inline
def indexer_cache_write(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
):
    """Scatter one step's raw indexer rows into the paged pool, one block per row.

    A slot of ``-1`` marks a row that owns no cache position — a padded row in a
    packed batch — and is skipped, so a negative slot cannot wrap into the tail of
    the pool. The key half is widened to FP32 on the way in: the pooled-table writer
    re-reads these rows in FP32, and the storage ABI is FP32 end to end.

    Returns the region's TaskId so a later reader can order against the write.
    """
    tokens = pl.tensor.dim(index_k, 0)
    with pl.spmd(tokens, name_hint="indexer_cache_write") as write_tid:
        token = pl.tile.get_block_idx()
        slot_i32 = pl.read(slots, [token])
        if slot_i32 >= 0:
            slot = pl.cast(slot_i32, pl.INDEX)
            key_f32 = pl.cast(pl.load(index_k, [token, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
            pl.store(key_f32, [slot, 0], cache)
            gate_f32 = pl.load(gate_scores, [token, 0], [1, INDEX_DIM])
            pl.store(gate_f32, [slot, INDEX_DIM], cache)
    return write_tid


def golden_indexer_pool_write(
    pool_cache: torch.Tensor,
    raw_cache: torch.Tensor,
    compress_ape: torch.Tensor,
    pool_token_slots: torch.Tensor,
    pool_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Close each pool event: per-dimension softmax over its four gate lanes.

    Args:
        pool_cache: ``[POOL_TABLE * INDEX_STATE_BLOCK_SIZE, INDEX_DIM]`` pooled table
            holding the prior state.
        raw_cache: ``[TABLE * BLOCK_SIZE, INDEX_STATE_WIDTH]`` raw support table.
        compress_ape: ``[INDEX_KPOOL, INDEX_DIM]`` learned pool-lane bias.
        pool_token_slots: ``[events, INDEX_KPOOL]`` physical raw row of each member
            token, ``-1`` for a padded member.
        pool_slots: ``[events]`` destination pooled row of each event.

    Returns:
        The updated pool table and the per-event validity (all four members present).
    """
    updated = pool_cache.clone()
    valid = torch.zeros(pool_slots.numel(), POOL_VALID_WIDTH, dtype=torch.int32)
    for event in range(pool_slots.numel()):
        members = pool_token_slots[event].to(torch.long)
        destination = int(pool_slots[event])
        if bool((members >= 0).all()):
            keys = raw_cache[members, 0:INDEX_DIM].float()
            gates = raw_cache[members, INDEX_DIM:].float()
            probabilities = torch.softmax(gates + compress_ape.float(), dim=0)
            updated[destination] = (probabilities * keys).sum(dim=0).to(pool_cache.dtype)
            valid[event, 0] = 1
        else:
            updated[destination] = 0
            valid[event, 0] = 0
    return updated, valid


@pl.jit.incore
def _indexer_pool_write_event(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.BF16],
    pool_valid: pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32],
) -> None:
    """Close one pool event; the caller gives this scope one block per event."""
    event = pl.tile.get_block_idx()
    slot0 = pl.read(pool_token_slots, [event, 0])
    slot1 = pl.read(pool_token_slots, [event, 1])
    slot2 = pl.read(pool_token_slots, [event, 2])
    slot3 = pl.read(pool_token_slots, [event, 3])
    members_present = pl.min(pl.min(slot0, slot1), pl.min(slot2, slot3))
    # 0/1 flag: every member slot present. It rides in column 0 of a [1, 8] INT32
    # tile, the narrowest store the 32-byte vector alignment accepts.
    valid_flag = pl.cast(pl.min(pl.max(members_present, 0), 1), pl.INT32)
    flag_tile = pl.tile.full([1, POOL_VALID_WIDTH], dtype=pl.INT32, value=0)
    destination = pl.cast(pl.read(pool_slots, [event]), pl.INDEX)
    if members_present >= 0:
        # Per-lane logits and keys, kept as [1, INDEX_DIM] rows so every step is
        # an elementwise row operation; the softmax reduces over the four lanes.
        key0 = pl.load(raw_cache, [pl.cast(slot0, pl.INDEX), 0], [1, INDEX_DIM])
        key1 = pl.load(raw_cache, [pl.cast(slot1, pl.INDEX), 0], [1, INDEX_DIM])
        key2 = pl.load(raw_cache, [pl.cast(slot2, pl.INDEX), 0], [1, INDEX_DIM])
        key3 = pl.load(raw_cache, [pl.cast(slot3, pl.INDEX), 0], [1, INDEX_DIM])
        gate0 = pl.load(raw_cache, [pl.cast(slot0, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate1 = pl.load(raw_cache, [pl.cast(slot1, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate2 = pl.load(raw_cache, [pl.cast(slot2, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate3 = pl.load(raw_cache, [pl.cast(slot3, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        ape0 = pl.cast(pl.load(compress_ape, [0, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape1 = pl.cast(pl.load(compress_ape, [1, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape2 = pl.cast(pl.load(compress_ape, [2, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape3 = pl.cast(pl.load(compress_ape, [3, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        logit0 = pl.add(gate0, ape0)
        logit1 = pl.add(gate1, ape1)
        logit2 = pl.add(gate2, ape2)
        logit3 = pl.add(gate3, ape3)
        lane_max = pl.maximum(pl.maximum(logit0, logit1), pl.maximum(logit2, logit3))
        weight0 = pl.exp(pl.sub(logit0, lane_max))
        weight1 = pl.exp(pl.sub(logit1, lane_max))
        weight2 = pl.exp(pl.sub(logit2, lane_max))
        weight3 = pl.exp(pl.sub(logit3, lane_max))
        denominator = pl.add(pl.add(weight0, weight1), pl.add(weight2, weight3))
        weighted_sum = pl.mul(weight0, key0)
        weighted_sum = pl.add(weighted_sum, pl.mul(weight1, key1))
        weighted_sum = pl.add(weighted_sum, pl.mul(weight2, key2))
        weighted_sum = pl.add(weighted_sum, pl.mul(weight3, key3))
        pool_key = pl.div(weighted_sum, denominator)
        pl.store(pl.cast(pool_key, pl.BF16, mode="rint"), [destination, 0], pool_cache)
        pl.tile.write(flag_tile, [0, 0], valid_flag)
    else:
        zero_row = pl.tile.full([1, INDEX_DIM], dtype=pl.BF16, value=0.0)
        pl.store(zero_row, [destination, 0], pool_cache)
    pl.store(flag_tile, [event, 0], pool_valid)


@pl.jit.inline
def indexer_pool_write(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.BF16],
    pool_valid: pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32],
):
    """Write one pooled key row per pool-close event, one block per event.

    The softmax runs per dimension over the four pool lanes in FP32, matching the
    reference numerics: ``weights = softmax(gate + ape)`` along the lane axis, then
    ``pool_key = sum(weights * key)``. An event with a ``-1`` member slot is a padded
    pool: it writes a zero row and reports ``pool_valid = 0`` so the scorer can drop
    it. The destination row itself is always written, which keeps the output fully
    defined for the golden harness.

    Returns the region's TaskId so the scorer can order against the pool writes.
    """
    events = pl.tensor.dim(pool_token_slots, 0)
    with pl.spmd(events, name_hint="indexer_pool_write") as pool_tid:
        _indexer_pool_write_event(
            raw_cache, compress_ape, pool_token_slots, pool_slots, pool_cache, pool_valid
        )
    return pool_tid


@pl.jit
def indexer_cache_write_test(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
):
    """Run one raw scatter for golden.run validation."""
    index_k.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    indexer_cache_write(index_k, gate_scores, slots, cache)
    return cache


@pl.jit
def indexer_pool_write_test(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.BF16]],
    pool_valid: pl.Out[pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32]],
):
    """Close every pool event of one dispatch for golden.run validation."""
    pool_token_slots.bind_dynamic(0, POOLS_DYN)
    pool_slots.bind_dynamic(0, POOLS_DYN)
    pool_valid.bind_dynamic(0, POOLS_DYN)
    indexer_pool_write(raw_cache, compress_ape, pool_token_slots, pool_slots, pool_cache, pool_valid)
    return pool_cache, pool_valid


def build_indexer_cache_write_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic raw scatter: unordered distinct slots and a padded row.

    Row 3 carries ``-1``, and the destination rows are deliberately out of order
    across pages so the write cannot pass by walking the pool linearly. ``cache`` is
    ``InOut`` so its untouched rows are uploaded and can be asserted, rather than
    read back as allocator residue.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(59)
    cache_rows = pages * BLOCK_SIZE

    def init_index_k():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32).bfloat16()

    def init_gate_scores():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32)

    def init_cache():
        return torch.randn(cache_rows, INDEX_STATE_WIDTH, generator=generator, dtype=torch.float32)

    def init_slots():
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16, init_value=init_index_k),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32, init_value=init_gate_scores),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec("cache", [cache_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_cache),
    ]


def build_indexer_pool_write_specs(events: int = 6, raw_pages: int = 2, pool_pages: int = 8):
    """Build one deterministic pool-close batch: fresh events and one padded pool.

    Member slots are grouped four-per-event out of consecutive raw rows so the
    golden's pooling window is unambiguous, the destinations are a permutation of
    pooled rows, and event 4 carries a ``-1`` member to exercise the padded-pool
    rejection path. The raw table is built as the cache write would leave it: BF16
    keys widened to FP32 beside FP32 gates.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(61)
    raw_rows = raw_pages * BLOCK_SIZE
    pool_rows = pool_pages * INDEX_STATE_BLOCK_SIZE
    member_rows = events * INDEX_KPOOL
    if member_rows > raw_rows or events > pool_rows:
        raise ValueError("fixture sizes exceed the requested pool geometry")

    def init_raw_cache():
        keys = torch.randn(raw_rows, INDEX_DIM, generator=generator)
        keys = keys.bfloat16().float()
        gates = torch.randn(raw_rows, INDEX_DIM, generator=generator)
        return torch.cat([keys, gates], dim=-1)

    def init_compress_ape():
        return torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16()

    def init_pool_token_slots():
        base = torch.randperm(raw_rows - INDEX_KPOOL + 1, generator=generator)[:events]
        offsets = torch.arange(INDEX_KPOOL)
        slots = (base.unsqueeze(1) + offsets).to(torch.int32)
        slots[4, 3] = -1
        return slots

    def init_pool_slots():
        chosen = torch.randperm(pool_rows, generator=generator)[:events]
        return chosen.to(torch.int32)

    def init_pool_cache():
        return torch.randn(pool_rows, INDEX_DIM, generator=generator, dtype=torch.float32).bfloat16()

    return [
        TensorSpec("raw_cache", [raw_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_raw_cache),
        TensorSpec("compress_ape", [INDEX_KPOOL, INDEX_DIM], torch.bfloat16, init_value=init_compress_ape),
        TensorSpec(
            "pool_token_slots",
            [events, INDEX_KPOOL],
            torch.int32,
            init_value=init_pool_token_slots,
        ),
        TensorSpec("pool_slots", [events], torch.int32, init_value=init_pool_slots),
        TensorSpec("pool_cache", [pool_rows, INDEX_DIM], torch.bfloat16, init_value=init_pool_cache),
        TensorSpec("pool_valid", [events, POOL_VALID_WIDTH], torch.int32),
    ]


def golden_indexer_cache_write_case(tensors):
    """Fill the expected raw pool state for :func:`build_indexer_cache_write_specs`."""
    tensors["cache"][:] = golden_indexer_cache_write(
        tensors["cache"], tensors["index_k"], tensors["gate_scores"], tensors["slots"]
    )


def golden_indexer_pool_write_case(tensors):
    """Fill the expected pooled table and validity for :func:`build_indexer_pool_write_specs`."""
    pool_cache, pool_valid = golden_indexer_pool_write(
        tensors["pool_cache"],
        tensors["raw_cache"],
        tensors["compress_ape"],
        tensors["pool_token_slots"],
        tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_valid"][:] = pool_valid


def _self_check() -> None:
    """Prove both goldens on CPU with inlined fixtures, before any device run.

    Kept local rather than in ``_golden_smoke.py`` because that module is shared
    across the whole model directory.
    """
    cache = torch.randn(4 * BLOCK_SIZE, INDEX_STATE_WIDTH)
    index_k = torch.randn(8, INDEX_DIM).bfloat16()
    gates = torch.randn(8, INDEX_DIM)
    slots = torch.tensor([5, 0, -1, 11, 300, 128, 257, 3], dtype=torch.int32)
    updated = golden_indexer_cache_write(cache, index_k, gates, slots)
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected = torch.cat([index_k[row].float(), gates[row]], dim=-1)
            assert torch.equal(updated[slot], expected), f"slot {slot} did not receive its row"
    assert torch.equal(updated[-1], cache[-1]), "a -1 slot wrapped into the last row"

    pool_cache = torch.randn(8, INDEX_DIM).bfloat16()
    raw_cache = torch.cat(
        [
            torch.randn(32, INDEX_DIM).bfloat16().float(),
            torch.randn(32, INDEX_DIM),
        ],
        dim=-1,
    )
    ape = torch.randn(INDEX_KPOOL, INDEX_DIM).bfloat16()
    pool_token_slots = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, -1, 15], [16, 17, 18, 19]],
        dtype=torch.int32,
    )
    pool_slots = torch.tensor([7, 3, 0, 5, 2], dtype=torch.int32)
    new_cache, valid = golden_indexer_pool_write(pool_cache, raw_cache, ape, pool_token_slots, pool_slots)
    assert valid[:, 0].tolist() == [1, 1, 1, 0, 1], valid
    members = pool_token_slots[0].long()
    weights = torch.softmax(raw_cache[members, INDEX_DIM:].float() + ape.float(), dim=0)
    expected = (weights * raw_cache[members, 0:INDEX_DIM].float()).sum(dim=0)
    assert torch.allclose(new_cache[7].float(), expected, atol=1e-2), "pool 0 mismatch"
    assert torch.equal(new_cache[5], torch.zeros(INDEX_DIM, dtype=torch.bfloat16))
    assert torch.equal(new_cache[1], pool_cache[1]), "an untouched pooled row changed"
    print("[GOLDEN] PASS indexer_cache self-check")


def main():
    """Prove the goldens on CPU, then validate both writers on device.

    The raw scatter is pure data movement, so it runs at zero tolerance. The pooled
    row is an FP32 softmax rounded once to BF16; device and reference differ only in
    transcendentals, which the BF16 rounding almost always absorbs, so the budget is
    one BF16 ulp with a small outlier allowance.
    """
    import argparse

    from golden import ratio_allclose, run

    _self_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", default="both", choices=["write", "pool", "both"])
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--events", type=int, default=6)
    parser.add_argument("--raw-pages", type=int, default=2)
    parser.add_argument("--pool-pages", type=int, default=8)
    args = parser.parse_args()

    results = []
    if args.case in ("write", "both"):
        results.append(
            run(
                fn=indexer_cache_write_test,
                specs=build_indexer_cache_write_specs(args.tokens, args.pages),
                golden_fn=golden_indexer_cache_write_case,
                config={"platform": args.platform, "device_id": args.device},
                rtol=0.0,
                atol=0.0,
                compile_only=args.compile_only,
            )
        )
    if args.case in ("pool", "both"):
        results.append(
            run(
                fn=indexer_pool_write_test,
                specs=build_indexer_pool_write_specs(args.events, args.raw_pages, args.pool_pages),
                golden_fn=golden_indexer_pool_write_case,
                config={"platform": args.platform, "device_id": args.device},
                rtol=1.0 / 64,
                atol=1e-3,
                compare_fn={
                    "pool_cache": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
                },
                compile_only=args.compile_only,
            )
        )
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "build_indexer_cache_write_specs",
    "build_indexer_pool_write_specs",
    "golden_indexer_cache_write",
    "golden_indexer_cache_write_case",
    "golden_indexer_pool_write",
    "golden_indexer_pool_write_case",
    "indexer_cache_write",
    "indexer_cache_write_test",
    "indexer_pool_write",
    "indexer_pool_write_test",
]


if __name__ == "__main__":
    main()
