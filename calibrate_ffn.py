#!/usr/bin/env python3
"""Calibrate dense Qwen3 FFN channels from local text without changing weights.

For each layer and channel j, the score is
    mean_over_tokens(z_j ** 2) * sum(down_proj.weight[:, j] ** 2),
where z is the input to down_proj (activation(gate_proj(x)) * up_proj(x)).
This is a task-distribution-dependent contribution proxy, not a detector of
"useless knowledge". It ignores cross-channel interactions and is not evidence
that pruning will preserve accuracy or improve latency.

Only an unquantized, single-file model.safetensors dense Qwen3 checkpoint is
supported. All Hugging Face loading is local-only, with remote code disabled.
Input is UTF-8 JSONL: every line must be an object with a nonempty string `text`.
No chat template is applied; provide exactly the text to calibrate on. Samples
are never truncated or padded. Output contains scores and metadata, never text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator


class CalibrationError(Exception):
    """An actionable error whose message contains no calibration text."""


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_records(path: Path, digest: Any = None) -> Iterator[tuple[int, str]]:
    """Yield validated records, optionally hashing the exact source bytes."""
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if digest is not None:
                digest.update(raw)
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CalibrationError(
                    f"JSONL line {line_number} is not a valid UTF-8 JSON object."
                ) from exc
            if not isinstance(record, dict):
                raise CalibrationError(f"JSONL line {line_number} is not an object.")
            value = record.get("text")
            if not isinstance(value, str) or not value.strip():
                raise CalibrationError(
                    f"JSONL line {line_number} must contain nonempty string 'text'."
                )
            yield line_number, value


def validate_config(config: dict[str, Any]) -> dict[str, int]:
    if config.get("model_type") != "qwen3":
        raise CalibrationError("Only standard dense model_type='qwen3' is supported.")
    if any(config.get(key) for key in (
        "num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size",
    )):
        raise CalibrationError("MoE checkpoints are not supported.")
    if any(config.get(key) for key in (
        "quantization_config", "compression_config", "quant_method",
        "load_in_4bit", "load_in_8bit",
    )):
        raise CalibrationError("Quantized or compressed checkpoints are not supported.")
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise CalibrationError("Biased attention or FFN projections are not supported.")
    dimensions: dict[str, int] = {}
    for key in ("num_hidden_layers", "hidden_size", "intermediate_size"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CalibrationError(f"Config field '{key}' must be a positive integer.")
        dimensions[key] = value
    return dimensions


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Never overwrite an existing result, and remove a failed partial write."""
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    data_path = args.data_jsonl.expanduser().resolve()
    if args.output.expanduser().is_symlink():
        raise CalibrationError("--output cannot be an existing symbolic link.")
    output = args.output.expanduser().resolve()
    if not model_dir.is_dir():
        raise CalibrationError("--model-dir must be an existing local directory.")
    if not data_path.is_file():
        raise CalibrationError("--data-jsonl must be an existing local file.")
    if output == model_dir or model_dir in output.parents:
        raise CalibrationError("--output must be outside the source model directory.")
    if output == data_path or output.exists():
        raise CalibrationError("--output must be a new file and cannot replace the dataset.")
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise CalibrationError("Local config.json and single-file model.safetensors are required.")
    if (model_dir / "model.safetensors.index.json").exists():
        raise CalibrationError("Sharded checkpoints are not supported by this prototype.")
    if (model_dir / "adapter_config.json").exists():
        raise CalibrationError("Adapter redirects are unsupported; provide a standalone base checkpoint.")
    try:
        raw_config = config_path.read_bytes()
        config = json.loads(raw_config.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError("config.json must contain valid UTF-8 JSON.") from exc
    if not isinstance(config, dict):
        raise CalibrationError("config.json must contain an object.")
    dimensions = validate_config(config)
    config_hash = hashlib.sha256(raw_config).hexdigest()
    weights_hash = sha256_file(weights_path)
    tokenizer_hashes = {
        name: sha256_file(model_dir / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "added_tokens.json", "vocab.json", "merges.txt")
        if (model_dir / name).is_file()
    }

    # Validate every line before loading model weights. Read again for inference;
    # compare hashes afterwards so a changing input cannot silently mix datasets.
    initial_digest = hashlib.sha256()
    expected_samples = sum(1 for _ in text_records(data_path, initial_digest))
    if expected_samples == 0:
        raise CalibrationError("The calibration dataset is empty.")
    data_hash = initial_digest.hexdigest()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise CalibrationError(
            "torch, transformers with dense Qwen3 support, and safetensors are required; "
            "this script does not install dependencies or download models."
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise CalibrationError("CUDA was requested but is unavailable.")
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise CalibrationError("MPS was requested but is unavailable.")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=dtype, attn_implementation="eager",
    )
    if any(getattr(model, key, False) for key in (
        "is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit",
    )):
        raise CalibrationError("The loaded model is quantized; calibration is unsupported.")
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) != dimensions["num_hidden_layers"]:
        raise CalibrationError("Loaded model does not have the expected dense Qwen3 layers.")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise CalibrationError("Loaded model configuration is not dense Qwen3.")
    if any(getattr(model.config, key, None) != value for key, value in dimensions.items()):
        raise CalibrationError("Loaded model dimensions do not match source config.json.")

    hidden = dimensions["hidden_size"]
    intermediate = dimensions["intermediate_size"]
    layer_count = dimensions["num_hidden_layers"]
    down_projections = []
    column_energy = []
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise CalibrationError(f"Layer {index} is missing a standard FFN.")
        for name, expected_shape in (
            ("gate_proj", (intermediate, hidden)),
            ("up_proj", (intermediate, hidden)),
            ("down_proj", (hidden, intermediate)),
        ):
            projection = getattr(mlp, name, None)
            if type(projection) is not torch.nn.Linear:
                raise CalibrationError(f"Layer {index} {name} is not a standard dense Linear.")
            if projection.bias is not None:
                raise CalibrationError(f"Layer {index} {name} has an unsupported bias.")
            if tuple(projection.weight.shape) != expected_shape:
                raise CalibrationError(f"Layer {index} {name} has an unexpected shape.")
            if not projection.weight.is_floating_point():
                raise CalibrationError(f"Layer {index} {name} has a non-floating weight.")
        down = mlp.down_proj
        energy = down.weight.detach().to(device="cpu", dtype=torch.float64).square().sum(dim=0)
        if not bool(torch.isfinite(energy).all()):
            raise CalibrationError(f"Layer {index} has non-finite down-projection weights.")
        down_projections.append(down)
        column_energy.append(energy)

    model.eval()
    model.to(device)
    activation_squares = [torch.zeros(intermediate, dtype=torch.float64) for _ in layers]
    observed_tokens = [0] * layer_count
    handles = []

    def make_hook(index: int):
        def capture(_module, inputs):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise CalibrationError(f"Layer {index} down_proj hook received unexpected inputs.")
            values = inputs[0].detach()
            if values.ndim < 2 or values.shape[-1] != intermediate:
                raise CalibrationError(f"Layer {index} FFN activation shape is unsupported.")
            flattened = values.reshape(-1, intermediate)
            # Small chunks bound temporary allocation. Reduce in float32 on the
            # execution device; accumulate statistics in float64 on the CPU.
            for chunk in flattened.split(256, dim=0):
                sums = chunk.float().square().sum(dim=0).to(device="cpu", dtype=torch.float64)
                if not bool(torch.isfinite(sums).all()):
                    raise CalibrationError(f"Layer {index} produced non-finite activation statistics.")
                activation_squares[index].add_(sums)
            observed_tokens[index] += flattened.shape[0]
        return capture

    sample_count = token_count = 0
    min_tokens: int | None = None
    max_tokens = 0
    inference_digest = hashlib.sha256()
    try:
        handles = [projection.register_forward_pre_hook(make_hook(index))
                   for index, projection in enumerate(down_projections)]
        with torch.no_grad():
            for line_number, text in text_records(data_path, inference_digest):
                encoded = tokenizer(
                    text, add_special_tokens=True, truncation=False,
                    padding=False, return_tensors="pt", return_attention_mask=True,
                )
                length = int(encoded["input_ids"].shape[-1])
                if length <= 0:
                    raise CalibrationError(f"JSONL line {line_number} produced no tokens.")
                if length > args.max_tokens:
                    raise CalibrationError(
                        f"JSONL line {line_number} has {length} tokens, exceeding "
                        f"--max-tokens={args.max_tokens}; no truncation was performed."
                    )
                # The backbone has the same FFN computation as the causal LM,
                # and avoids allocating unnecessary vocabulary-sized logits.
                result = backbone(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    use_cache=False, output_attentions=False,
                    output_hidden_states=False, return_dict=True,
                )
                del result
                sample_count += 1
                token_count += length
                min_tokens = length if min_tokens is None else min(min_tokens, length)
                max_tokens = max(max_tokens, length)
    finally:
        for handle in handles:
            handle.remove()

    if inference_digest.hexdigest() != data_hash or sample_count != expected_samples:
        raise CalibrationError("Calibration input changed during the run; scores were not saved.")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_hash:
        raise CalibrationError("Source config changed during the run; scores were not saved.")
    if sha256_file(weights_path) != weights_hash:
        raise CalibrationError("Source weights changed during the run; scores were not saved.")
    for name, original_hash in tokenizer_hashes.items():
        if not (model_dir / name).is_file() or sha256_file(model_dir / name) != original_hash:
            raise CalibrationError("Source tokenizer changed during the run; scores were not saved.")
    layer_scores: dict[str, list[float]] = {}
    for index in range(layer_count):
        if observed_tokens[index] != token_count or token_count <= 0:
            raise CalibrationError(f"Layer {index} hook token count does not match the input.")
        scores = (activation_squares[index] / token_count * column_energy[index]).tolist()
        if len(scores) != intermediate or not all(math.isfinite(x) and x >= 0 for x in scores):
            raise CalibrationError(f"Layer {index} contains invalid scores.")
        layer_scores[str(index)] = scores

    payload = {
        "schema_version": 1,
        "score_method": "activation_weight_energy",
        "score_formula": "mean_tokens(down_proj_input[j]^2) * sum_i(down_proj.weight[i,j]^2)",
        "interpretation": "Calibration-distribution contribution proxy; not useless-knowledge identification.",
        "model_config_sha256": config_hash,
        "model_weights_sha256": weights_hash,
        "tokenizer_files_sha256": tokenizer_hashes,
        "calibration_data_sha256": data_hash,
        "model_dimensions": dimensions,
        "calibration": {
            "samples": sample_count, "tokens": token_count,
            "max_tokens_limit": args.max_tokens,
            "min_sample_tokens": min_tokens, "max_sample_tokens": max_tokens,
            "add_special_tokens": True, "chat_template_applied": False,
            "truncation": False, "padding": False, "use_cache": False,
            "device": args.device, "dtype": args.dtype,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "layer_token_counts": observed_tokens,
        },
        "layer_scores": layer_scores,
    }
    write_new_json(output, payload)
    return {
        "status": "calibration_completed_source_checkpoint_unchanged",
        "score_method": payload["score_method"],
        "layers": layer_count, "channels_per_layer": intermediate,
        "samples": sample_count, "tokens": token_count,
        "model_config_sha256": config_hash, "model_weights_sha256": weights_hash,
        "calibration_data_sha256": data_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--max-tokens", type=positive_int, default=2048,
                        help="Hard per-sample token limit including special tokens; never truncates.")
    args = parser.parse_args()
    try:
        summary = calibrate(args)
    except CalibrationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        # Third-party exceptions can include input details; do not print text,
        # tracebacks, or exception messages from tokenization/model execution.
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "message": "Calibration failed; no successful result is claimed."}),
              file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
