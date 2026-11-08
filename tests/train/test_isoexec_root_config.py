"""The ``isoexec`` opt-in lives at the config root, selects the worker classes, and reaches validate_cfg.

uv run --isolated --extra dev --extra skyrl-train pytest tests/train/test_isoexec_root_config.py
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import prepare_runtime_environment, validate_cfg
from tests.train.util import example_dummy_config


@pytest.mark.parametrize("key", ["enable_isoexec", "trainer.isoexec", "generator.isoexec"])
def test_nested_or_legacy_isoexec_fields_are_rejected(key):
    with pytest.raises(ValueError, match="Invalid fields"):
        SkyRLTrainConfig.from_cli_overrides([key + "=true"])


@pytest.mark.parametrize("overrides", [[], ["isoexec=null"]])
def test_absent_or_null_isoexec_is_disabled(overrides):
    assert SkyRLTrainConfig.from_cli_overrides(overrides).isoexec is None


def test_cli_preserves_the_isoexec_mapping():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        ["isoexec.model_package=isoexec.models.qwen3_5", "isoexec.full_logprob_comparision=true"]
    )
    assert cfg.isoexec == {"model_package": "isoexec.models.qwen3_5", "full_logprob_comparision": True}


@pytest.fixture
def validate_cfg_without_ray(monkeypatch):
    from skyrl.backends.skyrl_train.utils import ppo_utils
    from skyrl.train.utils import utils

    monkeypatch.setattr(utils, "validate_batch_sizes", lambda cfg: None)
    monkeypatch.setattr(utils, "validate_generator_cfg", lambda cfg: None)
    monkeypatch.setattr(ppo_utils, "repopulate_all_registries", lambda: None)
    monkeypatch.setattr(
        ppo_utils.PolicyLossRegistry,
        "list_available",
        classmethod(lambda cls: [example_dummy_config().trainer.algorithm.policy_loss_type]),
    )
    monkeypatch.setattr(
        ppo_utils.AdvantageEstimatorRegistry,
        "list_available",
        classmethod(lambda cls: [example_dummy_config().trainer.algorithm.advantage_estimator]),
    )


def _fake_isoexec(monkeypatch, **attrs):
    for name in ("isoexec", "isoexec.integrations", "isoexec.integrations.skyrl"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for leaf, members in attrs.items():
        module = ModuleType(f"isoexec.integrations.skyrl.{leaf}")
        for key, value in members.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)


def test_validate_cfg_hands_the_config_to_isoexec_once(monkeypatch, validate_cfg_without_ray):
    seen = []
    _fake_isoexec(monkeypatch, bootstrap={"validate_config": lambda cfg: seen.append(cfg)})
    cfg = example_dummy_config()
    cfg.isoexec = {"model_package": "isoexec.models.qwen3_5"}

    validate_cfg(cfg)

    assert seen == [cfg]


class _AbsentIsoExec:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "isoexec" or name.startswith("isoexec."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


def test_validate_cfg_names_the_missing_isoexec_package(monkeypatch, validate_cfg_without_ray):
    monkeypatch.delitem(sys.modules, "isoexec", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_AbsentIsoExec, *sys.meta_path])
    cfg = example_dummy_config()
    cfg.isoexec = {"model_package": "isoexec.models.qwen3_5"}
    with pytest.raises(RuntimeError, match="IsoExec package"):
        validate_cfg(cfg)


def test_validate_cfg_ignores_isoexec_when_unset(monkeypatch, validate_cfg_without_ray):
    monkeypatch.setitem(sys.modules, "isoexec", None)
    validate_cfg(example_dummy_config())


def test_extra_runtime_env_vars_reach_the_ray_runtime_env():
    cfg = SkyRLTrainConfig()
    cfg.trainer.extra_runtime_env_vars = {"ISOEXEC_EXAMPLE": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "2"}
    env = prepare_runtime_environment(cfg)
    assert env["ISOEXEC_EXAMPLE"] == "1"
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "2"


def test_setup_trainer_selects_isoexec_workers_only_when_opted_in(monkeypatch, tmp_path):
    from skyrl.train.entrypoints.main_base import BasePPOExp

    policy, ref = object(), object()
    _fake_isoexec(monkeypatch, workers={"PolicyWorker": policy, "RefWorker": ref})
    selected = []
    trainer = SimpleNamespace(_vllm_metrics_scraper=None, build_models=lambda *classes: selected.append(classes))
    monkeypatch.setattr(BasePPOExp, "get_cfg_as_str", lambda self, cfg: "")
    monkeypatch.setattr(BasePPOExp, "get_tracker", lambda self: object())
    monkeypatch.setattr(BasePPOExp, "get_inference_client", lambda self: object())
    monkeypatch.setattr(BasePPOExp, "get_generator", lambda *args: object())
    monkeypatch.setattr(BasePPOExp, "get_trainer", lambda *args, **kwargs: trainer)
    monkeypatch.setattr(BasePPOExp, "get_trajectory_logger", lambda self: None, raising=False)
    native = pytest.importorskip("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")

    for enabled in (True, False):
        exp = object.__new__(BasePPOExp)
        exp.cfg = SkyRLTrainConfig.from_cli_overrides(
            ["trainer.strategy=megatron", f"trainer.export_path={tmp_path}", f"trainer.ckpt_path={tmp_path}"]
        )
        exp.cfg.isoexec = {"model_package": "isoexec.models.qwen3_5"} if enabled else None
        exp.tokenizer = exp.train_dataset = exp.eval_dataset = exp.colocate_pg = None
        assert exp._setup_trainer() is trainer

    assert selected == [
        (policy, native.CriticWorker, ref),
        (native.PolicyWorker, native.CriticWorker, native.RefWorker),
    ]


def test_tinker_backend_refuses_isoexec():
    backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")
    overrides = backend.MegatronBackendOverrides(**{"isoexec": {"model_package": "isoexec.models.qwen3_5"}})
    with pytest.raises(ValueError, match="Tinker backend"):
        backend._build_skyrl_train_config("model", overrides)
