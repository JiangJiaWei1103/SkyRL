# IsoExec models on the Megatron backend

`isoexec` at the config root selects IsoExec-built models for Megatron training and vLLM inference. SkyRL keeps the
worker lifecycle, the batch pipeline and the trainer loop; IsoExec supplies the model, optimizer, wrapper, scoring op,
engine registration and weight receivers through the subclass seams listed in `.agents/docs/backends/megatron.md`.

```yaml
isoexec:
  model_package: isoexec.models.qwen3_5
  full_logprob_comparision: true
```

With `full_logprob_comparision: true`, IsoExec compares every loss-active token's complete vocabulary row from the
engine against the trainer's own row during the policy forward and refuses the step on the first differing bit. The
rows travel through SkyRL's generic side channel, so the config must also enable it:

| Key | Value | Owner |
|---|---|---|
| `generator.inference_engine.enable_return_full_logprobs` | `true` when `full_logprob_comparision` is true | SkyRL transport |
| `generator.inference_engine.logprobs_mode` | `raw_logprobs` (IsoExec's engine applies the temperature itself) | SkyRL |
| `trainer.algorithm.rollout_train_logprobs_abs_diff_max` | `0.0` (sampled-token logprobs must match exactly) | SkyRL trainer |
| `trainer.strategy` / `generator.inference_engine.backend` | `megatron` / `vllm` | SkyRL |
| `trainer.bf16`, `trainer.remove_microbatch_padding`, `generator.batched` | `true` | SkyRL |
| `trainer.policy.language_model_only`, `trainer.ref.language_model_only` | `true` | SkyRL |
| `trainer.ckpt_interval`, `trainer.hf_save_interval`, `trainer.resume_mode` | `-1`, `-1`, `null` (no native checkpoint/export yet) | IsoExec refuses otherwise |

`isoexec.integrations.skyrl.config.resolve` runs once from `validate_cfg`. It refuses every unsupported setting
(LoRA, MTP, FP8, torch.compile, router replay, speculative decoding, a VLM checkpoint, ...), resolves the checkpoint
to a local snapshot, writes the vLLM engine kwargs into `generator.inference_engine.engine_init_kwargs`, and the worker
and engine environment into `trainer.extra_runtime_env_vars`. Nothing else in SkyRL imports IsoExec except
`BasePPOExp._setup_trainer`, which picks the IsoExec worker classes.

## Environment

IsoExec runs its own pinned runtime (`make setup` in the IsoExec repository builds it from
`environments/default.toml`). Run SkyRL with that interpreter and `RAY_ADDRESS=local`; the IsoExec package must be
importable on the driver, the trainer workers and the inference engines.

## Run

```bash
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash examples/train/isoexec/run_qwen3_5_0_8b_gsm8k.sh
```

The script trains Qwen3.5-0.8B for one GRPO step on one GPU with the full-vocabulary comparison on. It was written
against the seams in this tree and has not been executed as part of this change.
