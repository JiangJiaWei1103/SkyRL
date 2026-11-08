"""``init_model`` stays in SkyRL; subclasses supply the model, optimizer and wrapper through these seams.

Run with:
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_init_model_seams.py
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.train.config import SkyRLTrainConfig

try:
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker
    from skyrl.backends.skyrl_train.workers.megatron.model_flops import (
        unsupported_reason,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


class _Policy(megatron_worker.MegatronPolicyWorkerBase):
    def __init__(self, cfg, calls):
        self.cfg = cfg
        self.calls = calls
        self._rank = 0
        self._local_rank = 0
        self._is_lora = False
        self.policy_loss_fn = None
        self.strategy = SimpleNamespace(hf_config=None)
        self.profiler = None

    def init_configs(self, model_path, megatron_config, model_config_kwargs, transformer_config_kwargs, **kwargs):
        self.calls.append("init_configs")
        self.provider = None
        self.bridge = None
        self.is_vlm = False
        self.enable_router_replay = False
        self.megatron_config = megatron_config

    def make_megatron_module(self, **kwargs):
        self.calls.append(("make_megatron_module", kwargs))
        return [torch.nn.Linear(2, 2)]

    def _build_optimizer(self):
        self.calls.append("_build_optimizer")
        return SimpleNamespace(param_groups=[{"lr": 0.0}])

    def _build_model_wrapper(self):
        self.calls.append("_build_model_wrapper")
        return SimpleNamespace(actor_module=self.actor_module)

    def _set_expandable_segments(self, enabled):
        self.calls.append(("expandable_segments", enabled))


class _Ref(megatron_worker.MegatronRefWorkerBase):
    def __init__(self, cfg, calls):
        self.cfg = cfg
        self.calls = calls
        self._rank = 0
        self._local_rank = 0
        self.strategy = SimpleNamespace(hf_config=None)

    def init_configs(self, model_path, megatron_config, model_config_kwargs, transformer_config_kwargs, **kwargs):
        self.calls.append("init_configs")
        self.provider = None
        self.bridge = None
        self.is_vlm = False

    def make_megatron_module(self, **kwargs):
        self.calls.append(("make_megatron_module", kwargs))
        return [torch.nn.Linear(2, 2)]

    def _build_model_wrapper(self):
        self.calls.append("_build_model_wrapper")
        return SimpleNamespace(actor_module=self.actor_module)

    def _set_expandable_segments(self, enabled):
        self.calls.append(("expandable_segments", enabled))


@pytest.fixture
def lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(megatron_worker, "snapshot_download", lambda path: calls.append("snapshot_download"))
    monkeypatch.setattr(megatron_worker.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(megatron_worker, "print_model_size", lambda module: calls.append("print_model_size"))
    monkeypatch.setattr(
        megatron_worker,
        "get_megatron_optimizer_param_scheduler",
        lambda **kwargs: calls.append(("scheduler", kwargs["optimizer"])) or "scheduler",
    )
    return calls


def _trainer_cfg(*overrides):
    return SkyRLTrainConfig.from_cli_overrides(["trainer.strategy=megatron", *overrides]).trainer


def test_policy_init_model_orders_seams_and_tolerates_a_missing_provider(lifecycle):
    cfg = _trainer_cfg("trainer.policy.megatron_config.freeze_moe_router=true")
    worker = _Policy(cfg, lifecycle)

    worker.init_model("not-a-local-path", num_training_steps=7)

    names = [call if isinstance(call, str) else call[0] for call in lifecycle]
    assert names == [
        "init_configs",
        "make_megatron_module",
        "snapshot_download",
        "barrier",
        "print_model_size",
        "_build_optimizer",
        "scheduler",
        "_build_model_wrapper",
        "expandable_segments",
    ]
    module_kwargs = lifecycle[1][1]
    assert module_kwargs["wrap_with_ddp"] is True
    assert module_kwargs["freeze_moe_router_params"] is True
    assert module_kwargs["freeze_dsa_indexer_params"] is False
    assert module_kwargs["enable_router_replay"] is False
    assert lifecycle[6][1] is worker.optimizer
    assert worker.scheduler == "scheduler"
    assert worker.model.actor_module is worker.actor_module
    assert worker.provider is None and worker.bridge is None
    assert isinstance(worker.empty_cuda_cache, bool) and isinstance(worker.empty_cuda_cache_after_backward, bool)


def test_policy_inference_only_init_builds_no_optimizer(lifecycle):
    worker = _Policy(_trainer_cfg("trainer.policy.inference_only_init=true"), lifecycle)

    worker.init_model("not-a-local-path")

    names = [call if isinstance(call, str) else call[0] for call in lifecycle]
    assert "_build_optimizer" not in names and "scheduler" not in names
    assert lifecycle[1][1]["wrap_with_ddp"] is False
    assert worker.optimizer is None and worker.scheduler is None


def test_ref_init_model_orders_seams(lifecycle):
    worker = _Ref(_trainer_cfg(), lifecycle)

    worker.init_model("not-a-local-path")

    names = [call if isinstance(call, str) else call[0] for call in lifecycle]
    assert names == [
        "init_configs",
        "make_megatron_module",
        "snapshot_download",
        "barrier",
        "print_model_size",
        "_build_model_wrapper",
        "expandable_segments",
    ]
    assert lifecycle[1][1]["wrap_with_ddp"] is False
    assert worker.model.actor_module is worker.actor_module


def test_model_flops_names_the_missing_provider():
    assert unsupported_reason(None) is not None


def test_hf_export_and_draft_source_refuse_without_a_bridge():
    worker = _Policy(_trainer_cfg(), [])
    worker.bridge = None
    worker.actor_module = [torch.nn.Linear(2, 2)]
    with pytest.raises(ValueError, match="Megatron-Bridge"):
        worker.save_hf_model("/tmp/unused", tokenizer=None)
    with pytest.raises(ValueError, match="Megatron-Bridge"):
        worker._build_draft_weight_source(torch.bfloat16)
