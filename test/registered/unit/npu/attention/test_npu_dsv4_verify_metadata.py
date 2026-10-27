"""CPU regressions for DSV4's fixed-shape verify metadata preparation."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_build_dsv4_verify_bundle,
)
from sglang.srt.hardware_backend.npu.dsv4.verify_metadata import (
    compact_verify_c4_locations,
    fill_verify_compression_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def physical_slots(lengths, width):
    # Remap every logical page, including backwards jumps in physical IDs.
    return [
        ((pos // 128 * 17 + row * 101 + 31) % 251 + 1) * 128 + pos % 128
        for row, length in enumerate(lengths)
        for pos in range(length, length + width)
    ]


class TestVerifyC4Locations(unittest.TestCase):
    def test_matches_original_selection_with_remapped_pages(self):
        prefixes = (1, 2, 3, 4, 7, 120, 126, 127, 128, 129, 130, 255, 256, 8190)
        for dtype in (torch.int32, torch.int64):
            for width in (1, 2, 5, 6, 9, 16):
                for length in prefixes:
                    lengths = [length, length + 3]
                    slots = torch.tensor(physical_slots(lengths, width), dtype=dtype)
                    expected = slots[(slots >= 0) & ((slots % 4) == 3)] // 4
                    with self.subTest(dtype=dtype, width=width, lengths=lengths):
                        actual = compact_verify_c4_locations(slots, lengths, width)
                        torch.testing.assert_close(actual, expected)

    def test_empty_and_no_completed_c4_group(self):
        for lengths, width in (([], 6), ([1], 1), ([4, 8], 2)):
            slots = torch.tensor(physical_slots(lengths, width), dtype=torch.int64)
            result = compact_verify_c4_locations(slots, lengths, width)
            self.assertEqual(result.numel(), 0)

    @staticmethod
    def batch(lengths, width, page_size=128, req_ids=None, slots=None):
        req_ids = req_ids if req_ids is not None else list(range(1, len(lengths) + 1))
        slots = torch.tensor(
            physical_slots(lengths, width) if slots is None else slots,
            dtype=torch.int64,
        )
        return SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_c128_sidecar=torch.arange(8 * 128).reshape(8, 128),
                c128_page_size=16,
            ),
            out_cache_loc_dsv4=SimpleNamespace(),
            req_pool_indices_cpu=torch.tensor(req_ids),
            seq_lens_cpu=torch.tensor(lengths),
            seq_lens=torch.tensor(lengths),
            out_cache_loc=slots,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=page_size,
                translate_loc_from_full_to_swa=lambda value: value + 10000,
            ),
        )

    def test_bundle_uses_live_prefix_and_keeps_c128_boundary(self):
        batch = self.batch([127, 254], 6)
        with patch(
            "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks.compact_verify_c4_locations",
            wraps=compact_verify_c4_locations,
        ) as compact:
            result = maybe_build_dsv4_verify_bundle(batch, 6)
        compact.assert_called_once()
        self.assertEqual(compact.call_args.args[1], [127, 254])
        expected = batch.out_cache_loc[batch.out_cache_loc % 4 == 3] // 4
        torch.testing.assert_close(result.out_c4_loc, expected)
        torch.testing.assert_close(result.out_full_loc, batch.out_cache_loc)
        torch.testing.assert_close(result.out_swa_loc, batch.out_cache_loc + 10000)
        self.assertEqual(result.out_c128_loc.tolist(), [128 * 16, 256 * 16 + 1])

    def test_general_fallback_keeps_invalid_slots_and_padding_semantics(self):
        for page_size, lengths, req_ids, slots in (
            (1, [127], [1], [8, -1, 3, 23, 40, -1]),
            (128, [0], [0], [-1, -1, -1, -1, -1, -1]),
        ):
            batch = self.batch(lengths, 6, page_size, req_ids, slots)
            with patch(
                "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks.compact_verify_c4_locations"
            ) as compact:
                result = maybe_build_dsv4_verify_bundle(batch, 6)
            compact.assert_not_called()
            values = batch.out_cache_loc
            torch.testing.assert_close(
                result.out_c4_loc, values[(values >= 0) & (values % 4 == 3)] // 4
            )


def load_compression_methods():
    # Load the real backend methods without importing hardware/model modules.
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py"
    )
    tree = ast.parse(source.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4AscendAttnBackend"
    )
    names = {
        "_stable_compact_1d",
        "_fill_verify_positions_cmp_padding_one_device",
        "_refresh_graph_target_verify_compress_1d_direct",
    }
    selected = [
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "torch": torch,
        "fill_verify_compression_metadata": fill_verify_compression_metadata,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return type("VerifyBackend", (), {name: namespace[name] for name in names})


class TestDsparkDeviceCompression(unittest.TestCase):
    def test_dspark_refresh_uses_device_positions_across_c4_c128_and_padding(self):
        backend_type = load_compression_methods()
        for lengths in ([126, 130], [127, 255], [0, 126], [8190, 1]):
            width = 6
            backend = backend_type()
            backend._is_eagle_algorithm = False
            backend._is_dspark_algorithm = True
            backend._dsv4_compress_ratios = (4, 128)
            backend._dsv4_unique_compress_ratios = (4, 128)
            fm = SimpleNamespace(
                start_pos=torch.zeros(2, dtype=torch.int32),
                seqused=torch.zeros(2, dtype=torch.int32),
                positions_cmp_padding_c4=torch.full((5,), -99, dtype=torch.int64),
                positions_cmp_padding_c128=torch.full((2,), -99, dtype=torch.int64),
            )
            positions = torch.tensor(
                [length + col for length in lengths for col in range(width)]
            )
            ctx = SimpleNamespace(
                fm=fm,
                bs=2,
                tokens_per_bs=width,
                device=torch.device("cpu"),
                live_seq_lens=torch.tensor(lengths, dtype=torch.int32),
                forward_batch=SimpleNamespace(
                    positions=positions, out_cache_loc_dsv4=None
                ),
            )
            backend._refresh_graph_target_verify_compress_1d_direct(ctx)
            self.assertEqual(fm.start_pos.tolist(), lengths)
            self.assertEqual(
                fm.seqused.tolist(), [width if length > 0 else 0 for length in lengths]
            )
            for ratio, size in ((4, 5), (128, 2)):
                expected = [
                    length + col + 1 - ratio
                    for length in lengths
                    if length > 0
                    for col in range(width)
                    if (length + col + 1) % ratio == 0
                ][:size]
                expected += [0] * (size - len(expected))
                actual = getattr(fm, f"positions_cmp_padding_c{ratio}")
                self.assertEqual(actual.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
