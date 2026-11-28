"""Run on one Ascend device: verify outputs and graph lifetime."""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs
from sglang.srt.speculative.eagle_post_verify_graphs import _PostVerifyGraphs


@unittest.skipUnless(torch.npu.is_available(), "Ascend NPU required")
class TestNPULatency(unittest.TestCase):
    def test_commit_outputs_match_reference(self):
        for bs in [1, 2, 3, 4, 16]:
            width = 4
            predict = torch.arange(bs * width, device="npu", dtype=torch.int32) + 100
            indices = torch.arange(bs * width, device="npu", dtype=torch.int64).view(
                bs, width
            )
            accepted = torch.arange(bs, device="npu", dtype=torch.int32) % width + 1
            lengths = torch.arange(bs, device="npu", dtype=torch.int64) + 62
            new_lengths, bonus, steps, draft = prepare_verify_commit_outputs(
                predict,
                indices,
                accepted,
                lengths,
                num_draft_tokens=width,
                mamba_track_interval=64,
            )
            torch.testing.assert_close(new_lengths.cpu(), (lengths + accepted).cpu())
            expected_indices = (
                torch.arange(bs, device="npu") * width + accepted - 1
            ).long()
            torch.testing.assert_close(bonus.cpu(), predict[expected_indices].cpu())
            torch.testing.assert_close(steps[0].cpu(), (accepted - 1).long().cpu())
            torch.testing.assert_close(draft[0].cpu(), (accepted - 1).cpu())
            torch.testing.assert_close(draft[1].cpu(), expected_indices.cpu())
            torch.testing.assert_close(draft[2].cpu(), predict.long().cpu())

    def test_graph_outputs_and_side_stream_commit(self):
        state = torch.zeros(8, device="npu", dtype=torch.int64)

        def body(slot):
            d = slot.staged
            lengths, bonus, steps, draft = prepare_verify_commit_outputs(
                d["predict"],
                d["indices"],
                d["accepted"],
                d["lengths"],
                num_draft_tokens=4,
            )
            return (
                d["predict"],
                d["accepted"],
                d["indices"],
                lengths,
                bonus,
                draft,
                steps,
            )

        def commit(slot, outputs):
            state[slot.staged["rows"]] = outputs[3]

        graphs = _PostVerifyGraphs(
            body=body,
            commit=commit,
            sim_rng=torch.Generator().manual_seed(1),
            device_type="npu",
        )
        previous = None
        for turn in range(12):
            accepted = torch.tensor(
                [1 + turn % 4, 1 + (turn + 2) % 4], device="npu", dtype=torch.int32
            )
            lengths = torch.tensor(
                [turn + 20, turn + 40], device="npu", dtype=torch.int64
            )
            inputs = {
                "predict": torch.arange(8, device="npu", dtype=torch.int32)
                + 100
                + turn,
                "indices": torch.arange(8, device="npu", dtype=torch.int64).view(2, 4),
                "accepted": accepted,
                "lengths": lengths,
                "rows": torch.tensor([1, 6], device="npu"),
            }
            outputs = graphs.run(
                key=(2,),
                inputs=inputs,
                make_slot=lambda staged: SimpleNamespace(staged=staged, num_requests=2),
                sim_max_len=4,
            )
            graphs.join()
            torch.testing.assert_close(state[[1, 6]].cpu(), (lengths + accepted).cpu())
            torch.testing.assert_close(outputs[3].cpu(), (lengths + accepted).cpu())
            if previous is not None:
                value, expected = previous
                torch.testing.assert_close(value.cpu(), expected)
            previous = (outputs[3], outputs[3].cpu().clone())


if __name__ == "__main__":
    unittest.main()
