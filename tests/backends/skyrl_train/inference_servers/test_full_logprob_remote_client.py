"""The remote client requests full-vocabulary rows only when asked and decodes them bit-exactly.

uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/inference_servers/test_full_logprob_remote_client.py
"""

import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    PackedField,
    pack_full_logprobs,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteGenerateClient,
    RemoteInferenceClient,
)

_ROWS = np.array([[-3.0, -0.25, -2.0], [-1.5, -4.0, -0.5]], dtype=np.float32)
_TOKENS = [1, 2]


def _choice(with_rows: bool) -> dict:
    sampled = _ROWS[np.arange(len(_TOKENS)), _TOKENS]
    choice = {
        "token_ids": _TOKENS,
        "finish_reason": "stop",
        "logprobs": {"content": [{"logprob": float(value)} for value in sampled]},
    }
    if with_rows:
        choice[PackedField.FULL_LOGPROBS.value] = pack_full_logprobs(_ROWS)
    return choice


@pytest.mark.asyncio
async def test_generate_requests_and_decodes_full_rows(monkeypatch):
    client = RemoteGenerateClient(proxy_url="http://unused")
    captured = {}

    async def fake_post(url, json, headers, *, packed_side_channels=False):
        captured.update(url=url, json=json, packed_side_channels=packed_side_channels)
        return {"choices": [_choice(with_rows=True)]}

    monkeypatch.setattr(client, "_post", fake_post)
    sampling_params = {"temperature": 1.0, "logprobs": 1}
    result = await client.generate(
        prompt_token_ids=[7, 8],
        sampling_params=sampling_params,
        session_id=None,
        model="default",
        return_full_logprobs=True,
    )

    assert captured["url"].endswith("/skyrl/v1/generate")
    assert captured["json"]["return_full_logprobs"] is True
    assert captured["json"]["sampling_params"] == sampling_params
    assert captured["packed_side_channels"] is True
    assert result.full_logprobs.tobytes() == _ROWS.tobytes()
    assert result.response_logprobs == _ROWS[np.arange(len(_TOKENS)), _TOKENS].tolist()


@pytest.mark.asyncio
async def test_generate_without_the_flag_uses_the_native_route_and_carries_no_rows(monkeypatch):
    client = RemoteGenerateClient(proxy_url="http://unused")
    captured = {}

    async def fake_post(url, json, headers, *, packed_side_channels=False):
        captured.update(url=url, json=json)
        return {"choices": [_choice(with_rows=False)]}

    monkeypatch.setattr(client, "_post", fake_post)
    result = await client.generate(
        prompt_token_ids=[7, 8], sampling_params={"temperature": 1.0, "logprobs": 0}, session_id=None, model="default"
    )

    assert captured["url"].endswith("/inference/v1/generate")
    assert "return_full_logprobs" not in captured["json"]
    assert result.full_logprobs is None


@pytest.mark.asyncio
async def test_generate_refuses_a_missing_envelope_when_rows_were_requested(monkeypatch):
    client = RemoteGenerateClient(proxy_url="http://unused")

    async def fake_post(url, json, headers, *, packed_side_channels=False):
        return {"choices": [_choice(with_rows=False)]}

    monkeypatch.setattr(client, "_post", fake_post)
    with pytest.raises(ValueError, match="packed full_logprobs"):
        await client.generate(
            prompt_token_ids=[7],
            sampling_params={"temperature": 1.0, "logprobs": 1},
            session_id=None,
            model="default",
            return_full_logprobs=True,
        )


@pytest.mark.asyncio
async def test_batch_generate_opts_in_per_batch(monkeypatch):
    client = RemoteInferenceClient(
        proxy_url="http://unused", server_urls=["http://unused"], data_parallel_size=1, model_name="m"
    )
    client.enable_return_full_logprobs = True
    seen = []

    async def fake_single(**kwargs):
        seen.append(kwargs["return_full_logprobs"])
        return {
            "stop_reason": "stop",
            "response_ids": _TOKENS,
            "response_logprobs": [-0.25, -0.5],
            "routed_experts": None,
            PackedField.ROLLOUT_SAMPLE_SUPPORT.value: None,
            PackedField.FULL_LOGPROBS.value: _ROWS if kwargs["return_full_logprobs"] else None,
        }

    async def fake_detokenize(batches):
        return ["text"] * len(batches)

    monkeypatch.setattr(client, "_generate_single", fake_single)
    monkeypatch.setattr(client, "detokenize", fake_detokenize)
    sampling_params = {"temperature": 1.0, "logprobs": 1}

    output = await client.generate(
        {"prompt_token_ids": [[1, 2]], "sampling_params": sampling_params, "return_full_logprobs": True}, model="m"
    )
    assert output["response_full_logprobs"][0] is _ROWS

    output = await client.generate({"prompt_token_ids": [[1, 2]], "sampling_params": sampling_params}, model="m")
    assert output["response_full_logprobs"] is None
    assert seen == [True, False]
