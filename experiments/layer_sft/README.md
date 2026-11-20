# Direct projection tuning for a small language model

This is a prepared experimental training path, not a trained model or a deployment change. It adapts `openbmb/MiniCPM5-2B` by directly updating selected original projection weights. An optional `Qwen/Qwen3-1.7B` baseline provides a separate model comparison. No private dataset, personal results, machine identity, or model weights are included here.

## Scope

- `layer_train.py`: offline input audit, baseline generation, and direct weight training.
- `prepare_models.py`: explicit download of pinned public safetensors checkpoints with file verification.
- `spec.example.json`: model and experiment configuration; supply a private data directory and an isolated output path.
- `requirements.txt`: compatible Transformers/tokenizer dependencies; PyTorch is supplied by the execution environment.
- `test_preparation.py`: synthetic CPU-only checks for target masking, provenance, input integrity and snapshot enforcement. These do not load a real tokenizer or perform tensor operations.

The training path opens the last two Llama decoder blocks' `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` matrices. It keeps the rest frozen and compares their fingerprints afterwards. Trainable parameters are FP32; the model forward pass uses CUDA BF16 autocast. This changes existing parameters, without LoRA or layer removal.

The last two blocks are an experiment budget choice, not a claim that personality or a knowledge category lives in these layers. The recipe has not been shown to improve answer quality. Original checkpoints and existing deployments remain separate.

## Model revision and environment

MiniCPM5-2B uses standard `LlamaForCausalLM`; model-specific Python is not executed. The example pins the complete Hugging Face revision. The download command checks the resolved commit, provider file sizes, available LFS hashes and all local hashes, then writes `snapshot-lock.json`. Training uses `local_files_only=True` and `trust_remote_code=False`.

The [official deployment guide](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/transformers.md) lists Transformers 4.57.3 and PyTorch 2.7.1 as a CUDA 12 compatibility combination. Use an isolated environment appropriate to the actual machine; do not modify shared CUDA drivers to run this experiment.

MiniCPM5-2B officially supports native Think mode. A template branch that can insert empty thinking is not evidence that non-thinking inference is supported. All generated comparisons use `enable_thinking=True`.

## Private data contract

Provide `train.jsonl`, `dev.jsonl` and the evaluation JSONL files named in the spec. `manifest.json` must have a `splits` mapping keyed by each filename without `.jsonl`, with `sha256`, `rows`, and the complete unique `group_ids` list. Split groups must be disjoint. Prepare semantic event/source separation before creating this manifest; identifiers alone cannot prove independence.

Each training row has `id`, `group_id`, `kind`, `messages`, `target_message_index`, `loss_on`, `user_approved`, and provenance fields. Messages end in a user turn followed by the intended final assistant response. `loss_on` is `last_assistant_only`, and `target_message_index` names that last turn.

Research-authored examples use an admitted research kind, `user_approved=false`, and `target_status=researcher_written_not_user_approved`; grounded examples also require sources. Explicitly approved feedback uses `kind=user_approved_feedback` and an intake fingerprint matching the freshly verified snapshot. Synthetic examples must never be relabelled as real user approval.

The official complete chat template identifies the final answer. A sentinel validates the exact rendered span; token prefix equality must hold. Earlier assistant messages, including rejected responses, receive no loss. Input length violations fail instead of truncating evidence or the target.

This recipe has no supervised reasoning traces. Template-inserted empty thinking is masked context, which does not establish that native reasoning behavior is preserved. Inspect reasoning completion and actual final answers after training.

## Execution

```bash
python prepare_models.py spec.json
python layer_train.py audit --spec spec.json --output /private/path/tokenizer-audit
```

Actual `run` or `baseline` requires a GPU availability launcher that has verified the intended device and bound exactly one physical GPU UUID through `CUDA_VISIBLE_DEVICES`. A UUID string by itself is not a reservation or resource scheduler. The local training entry refuses to run without that binding.

```bash
python layer_train.py run --spec spec.json
# With a separate spec and output directory for the other original base:
python layer_train.py baseline --spec qwen-baseline-spec.json
```

The output directory must be new for every command. Do not send audit and training to the same destination. No command changes a service or existing daily model.

## Outputs and interpretation

`candidate.safetensors` and `last_candidate.safetensors` contain **absolute FP32 values for selected original parameters**. They are not full standalone models or PEFT adapters. Reload against the identical base revision and parameter names; the trainer checks tensor hashes and development loss reproduction.

The run record reports optimizer steps, changed tensor elements, effective BF16 changes, frozen parameter fingerprints, checkpoint hashes, and actual generation outputs. A change confined to FP32 rounding with no effective BF16 difference is identified explicitly. Development NLL can retain the original base; it does not decide deployment.

Evaluation text is read only after checkpoint selection closes. New-base and trained-model generation share prompt, native reasoning mode, sampling and fixed seeds. Cross-model comparisons use independently recorded model settings. Token counts are not directly comparable across different tokenizers.

Generated records preserve raw output and extracted final answers. An unclosed thinking block produces a null final answer. They distinguish EOS from token-budget truncation. Only complete text-generation duration is measured; first-token latency and speech latency are explicitly unmeasured.

There is no automatic factual-accuracy score. Inspect source support, subject attribution, temporal status, completeness, unsupported additions and instruction following. Repeated seeds do not create independent user events. A prepared or successful training run does not establish improved personal assistance.

## Validation status

The code has had static review and synthetic CPU checks. Actual tokenizer encoding, CUDA forward/backward, numerical checkpoint reload and real answer comparisons remain runtime gates. Do not interpret the included tests as model quality evidence.
