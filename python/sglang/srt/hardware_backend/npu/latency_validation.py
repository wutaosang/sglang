"""Expensive differential checks for the opt-in Ascend latency paths."""

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.decl import STATE_PART

logger = logging.getLogger(__name__)


@contextmanager
def capture_mtp_stages(model, rows):
    values, handles = {}, []

    def hook(name):
        def save(module, args, output):
            values[name + ".input"] = args[0][rows].clone()
            output = output[0] if isinstance(output, tuple) else output
            values[name + ".output"] = output[rows].clone()
            if len(args) > 1 and isinstance(args[1], torch.Tensor):
                values[name + ".residual"] = args[1][rows].clone()

        return save

    layer = model.layers[0]
    for name, module in [
        ("o_proj", layer.o_proj),
        ("gate_up", layer.mlp.gate_up_proj),
        ("down", layer.mlp.down_proj),
        ("norm", model.norm),
    ]:
        handles.append(module.register_forward_hook(hook(name)))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def compare_mtp_stages(actual, reference):
    for name in reference.keys() & actual.keys():
        error = (actual[name].float() - reference[name].float()).abs()
        logger.info(
            "NPU MTP stage %s max_abs=%s mean_abs=%s reference_max=%s",
            name,
            float(error.max()),
            float(error.mean()),
            float(reference[name].abs().max()),
        )


def snapshot_verify_state(target_worker, state_slots, batch_size):
    pool = target_worker.model_runner.token_to_kv_pool.layer_router.part_view(
        STATE_PART
    )
    if pool is None or state_slots is None:
        return SimpleNamespace(cache=None, slots=None, conv=None)
    cache = pool.get_speculative_mamba2_params_all_layers()
    slots = state_slots.working[:batch_size]
    return SimpleNamespace(
        cache=cache,
        slots=state_slots,
        working=slots,
        conv=[value[:, slots].clone() for value in cache.conv],
    )


def validate_verify_state(
    snapshot, target_worker, batch, verify_input, logits_output, outputs
):
    from sglang.srt.speculative.eagle_utils import eagle_sample
    from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

    expected = eagle_sample(verify_input, batch, logits_output, None)
    for actual, reference in zip(outputs[:3], expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if snapshot.cache is not None:
        assert batch.mamba_track_indices is None, (
            "validation currently requires radix cache off"
        )
        cache, slots = snapshot.cache, snapshot.working
        temporal_after = cache.temporal[:, slots].clone()
        conv_after = [value[:, slots].clone() for value in cache.conv]
        # Conv rollback is not idempotent. Restore its verify-end inputs before
        # executing the original commit, rather than comparing two rollbacks.
        for value, before in zip(cache.conv, snapshot.conv):
            value[:, slots] = before
        commit_mamba_states_after_verify(
            target_worker,
            batch,
            expected[1],
            expected[2],
            verify_input.draft_token_num,
            state_slots=snapshot.slots,
        )
        torch.testing.assert_close(
            cache.temporal[:, slots], temporal_after, rtol=0, atol=0, equal_nan=True
        )
        for value, after in zip(cache.conv, conv_after):
            torch.testing.assert_close(
                value[:, slots], after, rtol=0, atol=0, equal_nan=True
            )
    logger.info(
        "NPU post-verify differential check exact: batch=%d", len(batch.seq_lens)
    )
