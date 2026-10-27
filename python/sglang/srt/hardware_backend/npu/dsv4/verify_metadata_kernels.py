"""Fused NPU verify metadata kernels; imported lazily by device dispatch."""

import triton
import triton.language as tl


@triton.jit
def _compact_c4_locations(
    LOC, OUT, N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    loc = tl.load(LOC + i, i < N, other=-1)
    keep = (i < N) & (loc >= 0) & (loc % 4 == 3)
    rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    count = tl.sum(keep.to(tl.int32), 0)
    tl.store(OUT + i, 0, (i < M) & (i >= count))
    tl.store(OUT + rank, loc // 4, keep & (rank < M))


def compact_c4_locations_npu(loc, out):
    _compact_c4_locations[(1,)](
        loc,
        out,
        loc.numel(),
        out.numel(),
        triton.next_power_of_2(max(loc.numel(), out.numel())),
    )


@triton.jit
def _fill_verify_compression_metadata(
    POS,
    LIVE,
    START,
    SEQUSED,
    C4,
    C128,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    C4_SIZE: tl.constexpr,
    C128_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    row = i // WIDTH
    live = tl.load(LIVE + row, row < BS, other=0)
    pos = tl.load(POS + i, i < BS * WIDTH, other=0)
    absolute = live + i % WIDTH + 1
    valid = (i < BS * WIDTH) & (live > 0)

    # The graph's request buffers are refreshed in the same launch.
    row_live = tl.load(LIVE + i, i < BS, other=0)
    tl.store(START + i, row_live, i < BS)
    tl.store(SEQUSED + i, tl.where(row_live > 0, WIDTH, 0), i < BS)

    keep4 = valid & (absolute % 4 == 0)
    rank4 = tl.cumsum(keep4.to(tl.int32), 0) - 1
    count4 = tl.sum(keep4.to(tl.int32), 0)
    tl.store(C4 + i, 0, (i < C4_SIZE) & (i >= count4))
    tl.store(C4 + rank4, pos - 3, keep4 & (rank4 < C4_SIZE))

    keep128 = valid & (absolute % 128 == 0)
    rank128 = tl.cumsum(keep128.to(tl.int32), 0) - 1
    count128 = tl.sum(keep128.to(tl.int32), 0)
    tl.store(C128 + i, 0, (i < C128_SIZE) & (i >= count128))
    tl.store(C128 + rank128, pos - 127, keep128 & (rank128 < C128_SIZE))


def fill_verify_compression_metadata_npu(
    positions,
    live_seq_lens,
    start_pos,
    seqused,
    c4_positions,
    c128_positions,
    width,
):
    _fill_verify_compression_metadata[(1,)](
        positions,
        live_seq_lens,
        start_pos,
        seqused,
        c4_positions,
        c128_positions,
        start_pos.numel(),
        width,
        c4_positions.numel(),
        c128_positions.numel(),
        triton.next_power_of_2(
            max(
                start_pos.numel() * width,
                c4_positions.numel(),
                c128_positions.numel(),
                1,
            )
        ),
    )
