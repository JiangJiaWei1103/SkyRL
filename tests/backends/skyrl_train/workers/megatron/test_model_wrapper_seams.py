"""``MegatronModelWrapper`` routes grad finalization and packed scoring through overridable methods.

Run with:
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_model_wrapper_seams.py
"""

import pytest
import torch

try:
    from skyrl.backends.skyrl_train.distributed.megatron import model_utils
    from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
        MegatronModelWrapper,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


def test_pending_grad_sync_replays_through_the_finalize_seam():
    class Wrapper(MegatronModelWrapper):
        def __init__(self):
            self.actor_module = []
            self._pending_grad_sync = {"num_tokens": None}
            self.finalized = []

        def _finalize_model_grads(self, num_tokens):
            self.finalized.append(num_tokens)

    wrapper = Wrapper()
    wrapper.run_pending_grad_sync()
    wrapper.run_pending_grad_sync()
    assert wrapper.finalized == [None, None]
    assert wrapper._pending_grad_sync is None


def test_default_scale_logits_matches_in_place_division():
    wrapper = MegatronModelWrapper.__new__(MegatronModelWrapper)
    logits = torch.tensor([[2.0, 4.0]])
    assert wrapper._scale_logits(logits, 1.0) is logits
    assert torch.equal(wrapper._scale_logits(logits, 2.0), torch.tensor([[1.0, 2.0]]))


@pytest.mark.parametrize("chunk_size", [None, 2, 8])
def test_token_logprob_fn_replaces_the_distributed_op_and_keeps_the_scatter(chunk_size):
    vocab = 5
    lengths = [3, 2]
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    target = torch.tensor([[1, 2, 3, 4, 0]])
    logits = torch.log_softmax(torch.randn(1, 5, vocab), dim=-1)
    calls = []

    def token_logprob_fn(chunk_logits, chunk_targets):
        calls.append(chunk_logits.shape[0])
        return chunk_logits.gather(1, chunk_targets.unsqueeze(1)).squeeze(1)

    attention_mask = torch.zeros((2, 4), dtype=torch.long)
    attention_mask[0, 1:] = 1
    attention_mask[1, 2:] = 1
    out = model_utils.from_parallel_logits_to_logprobs_packed_sequences(
        logits,
        target,
        cu_seqlens,
        4,
        vocab_start_index=0,
        vocab_end_index=vocab,
        group=None,
        chunk_size=chunk_size,
        attention_mask=attention_mask,
        token_logprob_fn=token_logprob_fn,
    )
    expected = torch.zeros((2, 3))
    expected[0, 1] = logits[0, 0, 2]
    expected[0, 2] = logits[0, 1, 3]
    expected[1, 2] = logits[0, 3, 0]
    assert torch.allclose(out, expected)
    assert sum(calls) == sum(lengths)
    assert len(calls) == (1 if chunk_size in (None, 8) else 3)
