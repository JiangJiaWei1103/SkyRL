set -x

# One GRPO step of Qwen3.5-0.8B on GSM8K with IsoExec models, one GPU, full-vocabulary logprob comparison.
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/train/isoexec/run_qwen3_5_0_8b_gsm8k.sh

DATA_DIR="$HOME/data/gsm8k"
RUN_DIR="${RUN_DIR:-$HOME/runs/isoexec-qwen3_5-0_8b}"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
PYTHON="${ISOEXEC_PYTHON:-python}"

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S="${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-1200}"

"$PYTHON" -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  isoexec.model_package=isoexec.models.qwen3_5 \
  isoexec.full_logprob_comparision=true \
  generator.inference_engine.enable_return_full_logprobs=true \
  generator.inference_engine.logprobs_mode=raw_logprobs \
  trainer.algorithm.rollout_train_logprobs_abs_diff_max=0.0 \
  trainer.strategy=megatron \
  trainer.bf16=true \
  trainer.policy.model.path="$MODEL" \
  trainer.ref.model.path="$MODEL" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.policy.use_torch_compile=false \
  trainer.remove_microbatch_padding=true \
  trainer.fused_lm_head_logprob=false \
  trainer.gradient_checkpointing=false \
  trainer.mtp.enabled=false \
  trainer.recompute_old_logprobs_per_minibatch=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=1 \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=false \
  trainer.policy.megatron_config.transformer_config_kwargs.deterministic_mode=true \
  trainer.ref.megatron_config.tensor_model_parallel_size=1 \
  trainer.ref.megatron_config.pipeline_model_parallel_size=1 \
  trainer.ref.megatron_config.expert_model_parallel_size=1 \
  trainer.ref.megatron_config.expert_tensor_parallel_size=1 \
  trainer.ref.megatron_config.transformer_config_kwargs.sequence_parallel=false \
  trainer.epochs=1 \
  trainer.max_training_steps=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_tokens_per_microbatch=-1 \
  trainer.max_prompt_length=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.logger=console \
  trainer.project_name=isoexec-gsm8k \
  trainer.run_name=qwen3_5_0_8b \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_path="$RUN_DIR/checkpoints" \
  trainer.export_path="$RUN_DIR/exports" \
  generator.batched=true \
  generator.max_turns=1 \
  generator.n_samples_per_prompt=5 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enable_prefix_caching=false \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  generator.inference_engine.max_num_batched_tokens=2048 \
  generator.inference_engine.engine_init_kwargs.max_model_len=2048 \
  generator.inference_engine.engine_init_kwargs.max_num_seqs=20 \
  environment.env_class=gsm8k \
  $@
