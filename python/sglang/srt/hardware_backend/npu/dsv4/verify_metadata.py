"""Fixed-shape device helpers for the DSV4 speculative verify path."""

from collections.abc import Sequence

import torch


def compact_verify_c4_locations(
    out_full_loc: torch.Tensor,
    live_seq_lens: Sequence[int],
    verify_width: int,
) -> torch.Tensor:
    """Select C4 slots without a device-to-host output-size query.

    The caller supplies real, nonempty requests backed by full-cache pages
    whose size is divisible by four. Page offsets therefore preserve logical
    token positions modulo four. The exact output size follows from the live
    prefix lengths, independently of the device-resident physical page IDs.
    Reserved allocation lengths must not be used for ``live_seq_lens``.
    """
    output_size = sum(
        (length + verify_width) // 4 - length // 4 for length in live_seq_lens
    )
    result = out_full_loc.new_empty((output_size,))
    if output_size == 0:
        return result

    if out_full_loc.device.type == "npu":
        from .verify_metadata_kernels import compact_c4_locations_npu

        compact_c4_locations_npu(out_full_loc, result)
        return result

    result.zero_()
    keep = (out_full_loc >= 0) & ((out_full_loc % 4) == 3)
    ranks = torch.cumsum(keep.to(torch.int64), dim=0) - 1
    selected = torch.where(keep, out_full_loc // 4, torch.zeros_like(out_full_loc))
    result.scatter_add_(0, ranks.clamp(min=0, max=output_size - 1), selected)
    return result


def fill_verify_compression_metadata(
    positions: torch.Tensor,
    live_seq_lens: torch.Tensor,
    start_pos: torch.Tensor,
    seqused: torch.Tensor,
    c4_positions: torch.Tensor,
    c128_positions: torch.Tensor,
    verify_width: int,
) -> None:
    """Refresh fixed graph buffers from device prefixes, including padded rows."""
    if positions.device != live_seq_lens.device:
        raise ValueError("verify positions and live prefixes must share a device")
    if positions.device.type == "npu":
        from .verify_metadata_kernels import fill_verify_compression_metadata_npu

        fill_verify_compression_metadata_npu(
            positions,
            live_seq_lens,
            start_pos,
            seqused,
            c4_positions,
            c128_positions,
            verify_width,
        )
        return

    bs = start_pos.numel()
    live = live_seq_lens[:bs]
    start_pos.copy_(live)
    seqused.copy_((live > 0).to(seqused.dtype) * verify_width)
    absolute = live[:, None] + torch.arange(1, verify_width + 1, device=live.device)
    for ratio, dst in ((4, c4_positions), (128, c128_positions)):
        dst.zero_()
        if dst.numel() == 0 or positions.numel() == 0:
            continue
        keep = ((absolute % ratio == 0) & (live[:, None] > 0)).reshape(-1)
        ranks = torch.cumsum(keep.to(torch.int64), dim=0) - 1
        values = positions[: bs * verify_width].reshape(-1) + 1 - ratio
        selected = torch.where(keep & (ranks < dst.numel()), values, 0)
        dst.scatter_add_(
            0, ranks.clamp(min=0, max=dst.numel() - 1), selected.to(dst.dtype)
        )
